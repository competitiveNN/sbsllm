"""OpenAI-compatible server for sbsllm.

Exposes a local HTTP server that accepts OpenAI-format chat completion requests
and routes them to the appropriate chat website via browser automation.
"""

from __future__ import annotations

import json
import logging
import re
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar

from playwright.sync_api import Error as PlaywrightError

from .browser import (
    BrowserError,
    BrowserOperationTimeout,
    capture_response,
    check_page_health,
    get_page_snapshot,
    inject_and_submit,
    is_new_response,
    wait_for_response,
)
from .inject import extract_js, inject_prompt, submit_js
from .sites import get_site

# Default server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_REQUEST_BYTES = 1_000_000
# Default timeout for browser operations (seconds)
DEFAULT_BROWSER_TIMEOUT = 60
# How long the extracted answer/thinking may stay unchanged before we treat
# the web chat as finished and terminate the local response (seconds). This
# is the primary completion signal: web chats rarely clear their "generating"
# marker reliably, so waiting for `done` alone left the local chat spinning.
DEFAULT_RESPONSE_IDLE_TIMEOUT = 3.0
# Grace period after a site stops reporting generation before its `done`
# flag is trusted. Covers the brief window where a stop control flickers
# out mid-render and would otherwise truncate the answer.
DEFAULT_RESPONSE_DONE_CONFIRM = 0.75
# How long an unchanged payload is tolerated while the site still claims to
# be generating. Web chats pause mid-answer (thinking, re-render, rate limit)
# and cutting there truncates the reply -- but a site whose spinner never
# clears must not hold the tab until browser_timeout.
DEFAULT_BUSY_PATIENCE = 20.0
# After submitting, how long to wait for the web chat to produce any output at
# all. Without this a selector mismatch meant silence until browser_timeout,
# and the local chat simply spun forever.
DEFAULT_FIRST_TOKEN_TIMEOUT = 60.0
# SSE comment cadence while waiting, so the connection shows liveness.
DEFAULT_KEEPALIVE_INTERVAL = 10.0
# Poll interval between extraction passes (seconds)
DEFAULT_POLL_INTERVAL = 0.25
# How long a request waits for its model tab to become free (seconds). This
# must comfortably exceed a full answer: a chat with thinking enabled can hold
# its tab for a minute, and open-webui fires follow-up requests (e.g. title
# generation) while the main one is still streaming. 10s produced spurious
# 503s on a tab that was merely busy answering.
DEFAULT_BROWSER_LOCK_TIMEOUT = 180
# Default interval for periodic health checks (seconds)
DEFAULT_HEALTH_INTERVAL = 30
# Request ID header name
REQUEST_ID_HEADER = "X-Request-ID"

# Metrics storage
_request_count = 0
_request_latencies = []
_error_count = 0
_error_types: dict[str, int] = {}
_metrics_lock = threading.Lock()


def _update_metrics(elapsed: float, error: bool, error_type: str | None = None) -> None:
    """Update request metrics."""
    global _request_count, _error_count
    with _metrics_lock:
        _request_count += 1
        _request_latencies.append(elapsed)
        if error:
            _error_count += 1
            if error_type:
                _error_types[error_type] = _error_types.get(error_type, 0) + 1
        # Keep only last 1000 latencies
        if len(_request_latencies) > 1000:
            _request_latencies.pop(0)


class _SBSHTTPServer(ThreadingHTTPServer):
    """Threading server with bounded request tracking for shutdown."""

    daemon_threads = True
    block_on_close = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._active_requests = 0
        self._active_lock = threading.Lock()
        self._active_done = threading.Event()
        self._active_done.set()
        # Request handlers read their runtime settings off `self.server`.
        # These must live here: keeping them only on the Server wrapper made
        # every handler fall back to the defaults, so the configured
        # browser_timeout was ignored and no tab lock was ever taken.
        self.model_locks: _ModelLockRegistry = _ModelLockRegistry()
        self.browser_lock_timeout: float = DEFAULT_BROWSER_LOCK_TIMEOUT
        self.browser_timeout: float = DEFAULT_BROWSER_TIMEOUT
        self.response_idle_timeout: float = DEFAULT_RESPONSE_IDLE_TIMEOUT
        self.response_done_confirm: float = DEFAULT_RESPONSE_DONE_CONFIRM
        self.busy_patience: float = DEFAULT_BUSY_PATIENCE
        self.first_token_timeout: float = DEFAULT_FIRST_TOKEN_TIMEOUT
        self.keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL
        self.poll_interval: float = DEFAULT_POLL_INTERVAL
        self.model_map: dict[str, str] = {}
        self.tab_map: dict[str, object] = {}

    def process_request(self, request: Any, client_address: Any) -> None:
        with self._active_lock:
            self._active_requests += 1
            self._active_done.clear()
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_finished()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_finished()

    def _request_finished(self) -> None:
        with self._active_lock:
            self._active_requests = max(0, self._active_requests - 1)
            if self._active_requests == 0:
                self._active_done.set()

    def wait_for_idle(self, timeout: float) -> bool:
        return self._active_done.wait(timeout)


class _ModelLockRegistry:
    """One lock per browser tab, created on demand.

    A single global lock serialised every model against every other model, so a
    request to one chat blocked an unrelated request to a different chat for
    the entire answer. Playwright dispatches all page operations to a single
    worker thread, so distinct pages are already safe to drive concurrently;
    only requests targeting the *same* tab need to serialise, because one tab
    shows one conversation.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get(self, model: str) -> threading.Lock:
        """Return the lock for `model`, creating it if needed."""
        with self._guard:
            lock = self._locks.get(model)
            if lock is None:
                lock = threading.Lock()
                self._locks[model] = lock
            return lock

    def busy_models(self) -> list[str]:
        """Return the models whose tab is currently in use."""
        with self._guard:
            return [m for m, lock in self._locks.items() if lock.locked()]


class OpenAIHandler(BaseHTTPRequestHandler):
    """Handler for OpenAI-compatible chat completion requests."""

    # Class-level config (set by Server)
    model_map: ClassVar[dict[str, str]] = {}
    tab_map: ClassVar[dict[str, object]] = {}

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging to keep output clean."""

    def _send_json(
        self, status: int, data: dict, request_id: str | None = None
    ) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if request_id:
            self.send_header(REQUEST_ID_HEADER, request_id)
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error(
        self,
        status: int,
        message: str,
        error_type: str = "invalid_request_error",
        request_id: str | None = None,
    ) -> None:
        """Send an error response in OpenAI format."""
        self._send_json(
            status,
            {
                "error": {
                    "message": message,
                    "type": error_type,
                    "param": None,
                    "code": None,
                }
            },
            request_id,
        )

    def _send_sse_comment(self) -> None:
        """Emit an SSE comment. Ignored by clients, but proves liveness.

        A disconnect surfaces as BrokenPipeError/ConnectionError to the caller,
        which already handles a client going away mid-stream.
        """
        self.wfile.write(b": keepalive\n\n")
        self.wfile.flush()

    def _send_sse(self, data: str) -> None:
        """Write one server-sent event and flush it immediately."""
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def _send_sse_headers(self, request_id: str) -> None:
        """Send headers for an OpenAI-compatible streaming response."""
        # A streaming body has no Content-Length, so the client can only
        # detect the end of the response by the [DONE] sentinel or by the
        # connection closing. This handler speaks HTTP/1.0 and never really
        # keeps the socket alive, so advertising `Connection: keep-alive`
        # left clients waiting on a socket that stayed open indefinitely.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        if request_id:
            self.send_header(REQUEST_ID_HEADER, request_id)
        self.end_headers()
        self.wfile.flush()

    def do_GET(self) -> None:
        """Handle GET requests."""
        request_id = (
            self.headers.get(REQUEST_ID_HEADER.lower()) or uuid.uuid4().hex[:16]
        )
        if self.path == "/v1/models":
            self._handle_models(request_id)
        elif self.path == "/health":
            self._handle_health(request_id)
        elif self.path == "/metrics":
            self._handle_metrics(request_id)
        else:
            self._send_error(404, f"Not found: {self.path}", "not_found", request_id)

    def _handle_health(self, request_id: str) -> None:
        """Return health status including browser connectivity."""
        server = getattr(self, "server", None)
        browser_connected = False
        active_tabs = 0
        tab_details = {}
        logger = logging.getLogger(__name__)
        if server is not None:
            try:
                from sbsllm.browser import check_page_health, is_running

                browser_connected = is_running()
                if browser_connected:
                    for model, page in server.tab_map.items():
                        is_healthy = (
                            check_page_health(page) if page is not None else False
                        )
                        if is_healthy:
                            active_tabs += 1
                        tab_details[model] = {
                            "connected": is_healthy,
                            "page_open": page is not None,
                        }
            except Exception:
                logger.debug("Health check failed to get browser status", exc_info=True)
        self._send_json(
            200,
            {
                "status": "ok" if browser_connected else "degraded",
                "browser_connected": browser_connected,
                "active_tabs": active_tabs,
                "total_tabs": len(server.tab_map) if server is not None else 0,
                "models": list(self.model_map.keys()),
                "tab_details": tab_details,
                "request_id": request_id,
            },
            request_id,
        )

    def _handle_metrics(self, request_id: str) -> None:
        """Return Prometheus-compatible metrics."""
        with _metrics_lock:
            request_count = _request_count
            error_count = _error_count
            error_types = dict(_error_types)
            if _request_latencies:
                avg_latency = sum(_request_latencies) / len(_request_latencies)
                min_latency = min(_request_latencies)
                max_latency = max(_request_latencies)
            else:
                avg_latency = min_latency = max_latency = 0
        metrics = (
            f"# HELP sbsllm_requests_total Total number of requests\n"
            f"# TYPE sbsllm_requests_total counter\n"
            f"sbsllm_requests_total {request_count}\n"
            f"# HELP sbsllm_request_duration_seconds Request duration in seconds\n"
            f"# TYPE sbsllm_request_duration_seconds summary\n"
            f"sbsllm_request_duration_seconds_avg {avg_latency:.6f}\n"
            f"sbsllm_request_duration_seconds_min {min_latency:.6f}\n"
            f"sbsllm_request_duration_seconds_max {max_latency:.6f}\n"
            f"# HELP sbsllm_errors_total Total number of errors\n"
            f"# TYPE sbsllm_errors_total counter\n"
            f"sbsllm_errors_total {error_count}\n"
        )
        for err_type, count in sorted(error_types.items()):
            metrics += (
                f"# HELP sbsllm_errors_by_type_total Errors by type\n"
                f"# TYPE sbsllm_errors_by_type_total counter\n"
                f'sbsllm_errors_by_type_total{{type="{err_type}"}} {count}\n'
            )
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        if request_id:
            self.send_header(REQUEST_ID_HEADER, request_id)
        self.end_headers()
        self.wfile.write(metrics.encode("utf-8"))

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            request_id = (
                self.headers.get(REQUEST_ID_HEADER.lower()) or uuid.uuid4().hex[:16]
            )
            self._send_error(404, f"Not found: {self.path}", "not_found", request_id)

    def _handle_models(self, request_id: str) -> None:
        """Return available models."""
        models = []
        for model_id, site_id in self.model_map.items():
            models.append(
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "sbsllm",
                    "permission": [
                        {
                            "id": f"modelperm-{model_id}",
                            "object": "model_permission",
                            "created": int(time.time()),
                            "allow_create_engine": False,
                            "allow_sampling": True,
                            "allow_logprobs": False,
                            "allow_search_indices": False,
                            "allow_view": True,
                            "allow_fine_tuning": False,
                            "organization": "*",
                            "group": None,
                            "is_blocking": False,
                        }
                    ],
                    "root": site_id,
                    "parent": None,
                }
            )
        self._send_json(200, {"object": "list", "data": models}, request_id)

    def _recover_page(self, model: str, site_id: str) -> object | None:
        """Attempt to recover a crashed/stale browser tab.

        Opens a fresh page for the site URL and updates the server's tab_map
        so subsequent requests use the new page. Returns the new page or None.
        """
        from sbsllm.browser import recover_page

        site = get_site(site_id)
        url = site["url"] if site else None
        if not url:
            return None
        logger = logging.getLogger(__name__)
        logger.info(
            "recovering page for model=%s site=%s url=%s",
            model, site_id, url,
        )
        new_page = recover_page(url)
        if new_page is not None:
            # `self.tab_map` is the dict shared with the Server that owns the
            # pages. Writing to `self.server.tab_map` would update a different
            # mapping and leave later requests pointing at the dead page.
            self.tab_map[model] = new_page
        return new_page

    def _inject_and_submit(
        self, page, site_id: str, prompt: str, tab_index: int
    ) -> dict:
        inject_js = inject_prompt(site_id, prompt)
        submit_js_val = submit_js(site_id)

        # Check page health before attempting operation
        if not check_page_health(page):
            logger = logging.getLogger(__name__)
            logger.warning(f"Page {tab_index} is unhealthy")
            return {
                "tab": tab_index,
                "inject": "BROWSER_ERROR: page unhealthy",
                "submit": None,
            }

        return inject_and_submit(page, inject_js, submit_js_val, tab_index)

    def _inject_and_submit_with_recovery(
        self,
        page,
        model: str,
        site_id: str,
        prompt: str,
        tab_index: int,
        max_retries: int = 1,
    ) -> tuple[dict, object | None]:
        """Inject and submit, recovering the page if it is unhealthy.

        Returns (status_dict, final_page). When recovery succeeds the returned
        page is the new page and the caller should update its local reference.
        """
        status = self._inject_and_submit(page, site_id, prompt, tab_index)

        # Only attempt recovery if the page was unhealthy to begin with.
        if (
            status.get("inject") == "BROWSER_ERROR: page unhealthy"
            and max_retries > 0
        ):
            logger = logging.getLogger(__name__)
            logger.warning(
                "Page unhealthy for model=%s; attempting recovery", model
            )
            new_page = self._recover_page(model, site_id)
            if new_page is not None:
                page = new_page
                status = self._inject_and_submit(
                    page, site_id, prompt, tab_index
                )
        return status, page

    def _message_text(self, message: dict) -> str:
        """Extract text from an OpenAI message, including multipart content."""
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = " ".join(text_parts)
        return content if isinstance(content, str) else str(content or "")

    def _strip_meta_tags(self, text: str) -> str:
        """Strip the local chat's meta-instruction wrapper, keeping only the
        actual user message.

        The local chat may bundle a system instruction (task, guidelines,
        output format, chat history) together with the actual user message in
        a single user-role message. Web chats receive only the trailing user
        message — i.e. the last "USER:" turn inside the <chat_history> block
        that the local chat appends to the instruction. If there is no
        <chat_history> block, the message is assumed to already be the user
        message and is returned as-is (stripped).
        """
        if not text:
            return ""
        history_match = re.search(
            r"<chat_history>(.*?)</chat_history>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if history_match:
            user_turns = re.findall(
                r"USER:\s*(.*?)(?=\n\s*ASSISTANT:|\Z)",
                history_match.group(1),
                re.DOTALL | re.IGNORECASE,
            )
            if user_turns:
                return user_turns[-1].strip()
        return text.strip()

    def _build_web_prompt(self, messages: list[dict]) -> str:
        """Return only the latest user message for a web chat."""
        for message in reversed(messages):
            if message.get("role", "user") == "user":
                content = self._strip_meta_tags(self._message_text(message))
                if content:
                    return content
        return ""

    def _acquire_browser_lock(self, model: str):
        """Acquire the lock for `model`'s tab; returns (lock, acquired, timeout).

        The lock is per tab, not global, so a request to one chat never blocks
        an unrelated chat. `lock` is None when acquisition failed, so the
        caller cannot accidentally release a lock it does not hold.
        """
        server_obj = getattr(self, "server", None)
        registry = getattr(server_obj, "model_locks", None)
        browser_lock_timeout = self._server_setting(
            "browser_lock_timeout", DEFAULT_BROWSER_LOCK_TIMEOUT
        )
        if registry is None:
            return None, True, 0
        lock = registry.get(model)
        acquired = lock.acquire(timeout=browser_lock_timeout)
        return (lock if acquired else None), acquired, browser_lock_timeout

    def _release_browser_lock(self, browser_lock) -> None:
        if browser_lock is not None:
            browser_lock.release()

    def _busy_models(self) -> list[str]:
        """Models whose tab is currently serving another request."""
        registry = getattr(getattr(self, "server", None), "model_locks", None)
        if registry is None:
            return []
        return registry.busy_models()

    def _send_busy_error(self, request_id: str, model: str, timeout: float) -> None:
        """Report a tab-busy conflict with actionable detail."""
        busy = ", ".join(self._busy_models()) or model
        self._send_error(
            503,
            f"Model '{model}' is already answering another request (busy: {busy}). "
            f"Waited {timeout:g}s. Send one request per model at a time, or raise "
            "browser_lock_timeout.",
            "server_error",
            request_id,
        )

    def _server_setting(self, name: str, default: Any) -> Any:
        """Read a runtime setting published by Server onto the HTTP server."""
        server_obj = getattr(self, "server", None)
        value = getattr(server_obj, name, None)
        return default if value is None else value

    def _sse_chunk(
        self,
        completion_id: str,
        created: int,
        model: str,
        delta: dict,
        finish_reason: str | None = None,
    ) -> None:
        """Write one OpenAI-compatible `chat.completion.chunk` SSE event."""
        self._send_sse(
            json.dumps(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": delta, "finish_reason": finish_reason}
                    ],
                }
            )
        )

    def _stream_web_chat(
        self,
        page,
        extraction: str,
        baseline: dict | None,
        completion_id: str,
        created: int,
        model: str,
        browser_timeout: float,
    ) -> dict:
        """Poll a web chat and stream its thinking + answer as they appear.

        Returns the final extraction result with a `stop_reason` saying why
        polling ended, so the handler can report a real failure instead of
        leaving the client waiting.

        Termination is guaranteed. The loop exits on the first of:
          * the site finishing after having been seen generating (`site_done`),
          * the payload going idle while the site is not generating (`idle`),
          * an unchanged payload outlasting the busy patience (`busy_timeout`),
          * no output at all after the prompt (`no_output`),
          * the overall budget elapsing (`budget_exhausted`).

        Truncation guard: web chats pause routinely mid-answer, so an unchanged
        payload is not treated as finished while the site still claims to be
        generating. The busy patience is what stops that tolerance from turning
        into an indefinite hang.
        """
        setting = self._server_setting
        idle_timeout = float(setting("response_idle_timeout", DEFAULT_RESPONSE_IDLE_TIMEOUT))
        done_confirm = float(setting("response_done_confirm", DEFAULT_RESPONSE_DONE_CONFIRM))
        busy_patience = float(setting("busy_patience", DEFAULT_BUSY_PATIENCE))
        first_token_timeout = float(
            setting("first_token_timeout", DEFAULT_FIRST_TOKEN_TIMEOUT)
        )
        poll_interval = float(setting("poll_interval", DEFAULT_POLL_INTERVAL))
        keepalive_interval = float(
            setting("keepalive_interval", DEFAULT_KEEPALIVE_INTERVAL)
        )
        started_at = time.monotonic()
        deadline = started_at + max(float(browser_timeout), 1.0)
        last_content = ""
        last_thinking = ""
        role_sent = False
        saw_busy = False
        got_output = False
        last_change_at: float | None = None
        last_keepalive_at: float = started_at
        response = {
            "found": False,
            "content": "",
            "thinking": None,
            "busy": False,
            "done": False,
            "count": 0,
        }

        def finish(reason: str, *, done: bool = True) -> dict:
            response["stop_reason"] = reason
            response["done"] = done
            return response

        while True:
            now = time.monotonic()
            if now > deadline:
                return finish("budget_exhausted", done=bool(response.get("done")))
            try:
                response = capture_response(page, extraction)
            except (BrowserError, BrowserOperationTimeout) as e:
                # A wedged browser thread surfaces here; end the stream
                # instead of holding the browser lock until the deadline.
                raise BrowserError(f"Response capture failed: {e}") from e

            content = response.get("content") or ""
            thinking = response.get("thinking") or ""
            busy = bool(response.get("busy"))
            if busy:
                saw_busy = True

            is_new = self._is_new_response(response, baseline)
            if is_new and (content or thinking):
                got_output = True

            if not is_new:
                # Nothing new on screen yet. Never spin silently: if the web
                # chat has produced nothing at all, say so and stop, instead
                # of holding the tab until the budget runs out.
                if not got_output and now - started_at >= first_token_timeout:
                    return finish("no_output", done=False)
                if now - last_keepalive_at >= keepalive_interval:
                    # A long silence reads as a dead connection; an SSE
                    # comment keeps the client (and any proxy) satisfied.
                    self._send_sse_comment()
                    last_keepalive_at = now
                time.sleep(poll_interval)
                continue

            delta: dict[str, Any] = {}
            if not role_sent:
                # Announce the role on the first chunk even when it only
                # carries reasoning, so thinking-only turns stay valid.
                delta["role"] = "assistant"
                role_sent = True
            changed = False
            if content != last_content:
                changed = True
                delta["content"] = content.removeprefix(last_content)
                last_content = content
            if thinking and thinking != last_thinking:
                changed = True
                delta["thinking"] = thinking.removeprefix(last_thinking)
                last_thinking = thinking

            if delta:
                self._sse_chunk(completion_id, created, model, delta)

            now = time.monotonic()
            if changed:
                # Track only real payload movement. Resetting on `busy` as
                # well made the elapsed time always ~0, so the busy-patience
                # guard could never fire and a site with a stuck spinner held
                # the tab until the budget ran out.
                last_change_at = now
            stable_for = now - last_change_at if last_change_at is not None else 0.0

            if saw_busy and not busy and response.get("done") and stable_for >= done_confirm:
                return finish("site_done")
            if not busy and (content or thinking) and stable_for >= idle_timeout:
                return finish("idle")
            if busy and (content or thinking) and stable_for >= busy_patience:
                # The site never stopped claiming to generate, yet the text
                # has not moved for a long time. Do not wait forever.
                return finish("busy_timeout", done=False)
            if not got_output and now - started_at >= first_token_timeout:
                return finish("no_output", done=False)
            time.sleep(poll_interval)

    def _handle_streaming_chat_completions(
        self,
        request_id: str,
        model: str,
        site_id: str,
        page,
        tab_index: int,
        prompt: str,
        request_start: float,
        browser_timeout: float,
    ) -> None:
        """Handle an SSE-streamed chat completion."""
        logger = logging.getLogger(__name__)
        browser_lock, lock_acquired, lock_timeout = self._acquire_browser_lock(model)
        if not lock_acquired:
            logger.warning(
                "chat_completion tab_busy",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "lock_timeout": lock_timeout,
                    "busy_models": self._busy_models(),
                    "request_id": request_id,
                    "stream": True,
                },
            )
            self._send_busy_error(request_id, model, lock_timeout)
            _update_metrics(time.monotonic() - request_start, error=True, error_type="lock_timeout")
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        extraction = extract_js(site_id)
        headers_sent = False
        try:
            if extraction is None:
                logger.warning(
                    "chat_completion extraction_unsupported",
                    extra={
                        "model": model,
                        "site_id": site_id,
                        "tab_index": tab_index,
                        "request_id": request_id,
                        "stream": True,
                    },
                )
                self._send_error(
                    502,
                    f"Response extraction is not supported for site: {site_id}. "
                    f"This site may require a login or page refresh.",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="extraction_unsupported")
                return
            baseline = capture_response(page, extraction)
            status, page = self._inject_and_submit_with_recovery(
                page, model, site_id, prompt, tab_index
            )
            if not isinstance(status, dict):
                logger.error(
                    "chat_completion invalid_status",
                    extra={
                        "model": model,
                        "tab_index": tab_index,
                        "request_id": request_id,
                        "stream": True,
                        "status": str(status),
                    },
                )
                self._send_error(
                    500,
                    "Browser automation returned an invalid status",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="invalid_status")
                return
            browser_error = next(
                (
                    status[field]
                    for field in ("inject", "submit")
                    if isinstance(status.get(field), str)
                    and status[field].startswith("BROWSER_ERROR:")
                ),
                None,
            )
            if browser_error:
                snapshot = get_page_snapshot(page)
                logger.warning(
                    "chat_completion browser_error",
                    extra={
                        "model": model,
                        "tab_index": tab_index,
                        "request_id": request_id,
                        "stream": True,
                        "error": browser_error,
                        "page_url": snapshot.get("url"),
                        "page_title": snapshot.get("title"),
                    },
                )
                self._send_error(
                    502,
                    f"Browser error: {browser_error}. "
                    f"Check the browser tab for CAPTCHA or login prompts.",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="browser_error")
                return
            if status.get("inject") != "OK" or status.get("submit") not in {
                "OK",
                "ENTER_SENT",
                "ENTER_SENT_UNVERIFIED",
            }:
                logger.warning(
                    "chat_completion inject_submit_failed",
                    extra={
                        "model": model,
                        "tab_index": tab_index,
                        "request_id": request_id,
                        "stream": True,
                        "inject_status": status.get("inject"),
                        "submit_status": status.get("submit"),
                    },
                )
                self._send_error(
                    502,
                    f"Browser automation failed to inject or submit the prompt. "
                    f"Inject: {status.get('inject', 'unknown')}, Submit: {status.get('submit', 'unknown')}. "
                    f"Try refreshing the browser tab or restarting sbsllm.",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="inject_submit_failed")
                return
            if not extraction:
                self._send_error(
                    502,
                    f"Response extraction is not supported for site: {site_id}",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="extraction_unsupported")
                return

            self._send_sse_headers(request_id)
            headers_sent = True
            response = self._stream_web_chat(
                page,
                extraction,
                baseline,
                completion_id,
                created,
                model,
                browser_timeout,
            )
            stop_reason = response.get("stop_reason", "unknown")
            stopped_cleanly = bool(response.get("done"))
            elapsed = time.monotonic() - request_start
            logger.info(
                "chat_completion stream_end",
                extra={
                    "model": model,
                    "site_id": site_id,
                    "tab_index": tab_index,
                    "request_id": request_id,
                    "stream": True,
                    "stop_reason": stop_reason,
                    "duration_ms": int(elapsed * 1000),
                    "chars": len(response.get("content") or ""),
                    "has_thinking": bool(response.get("thinking")),
                },
            )
            if not response.get("content") and not response.get("thinking"):
                # Never end an empty stream without saying why: a silent
                # empty response is indistinguishable from a hang in the
                # local chat, and gives nothing to debug with.
                if response.get("login_wall"):
                    detail = (
                        f"The {site_id} chat requires a signed-in session. "
                        f"Please log in to {site_id} in the browser tab, then retry."
                    )
                else:
                    detail = {
                        "no_output": (
                            f"The {site_id} page produced no answer within "
                            f"{self._server_setting('first_token_timeout', DEFAULT_FIRST_TOKEN_TIMEOUT):g}s. "
                            "Check the browser tab is logged in, that the message "
                            "was actually sent, and that the response selectors "
                            "in sites.py still match the page."
                        ),
                        "busy_timeout": (
                            f"The {site_id} page kept reporting 'generating' without "
                            "changing its text. Returning what was captured."
                        ),
                        "budget_exhausted": (
                            f"The {site_id} answer did not finish within "
                            f"{browser_timeout:g}s. Returning what was captured."
                        ),
                    }.get(
                        stop_reason,
                        f"No answer text could be extracted from {site_id}.",
                    )
                self._sse_chunk(
                    completion_id, created, model, {"content": f"[sbsllm] {detail}"}
                )

            # Always finish the stream: a terminal chunk followed by the
            # [DONE] sentinel. Without this the local chat kept waiting.
            self._sse_chunk(
                completion_id,
                created,
                model,
                {},
                finish_reason="stop" if stopped_cleanly else "length",
            )
            self._send_sse("[DONE]")
            headers_sent = True
        except (BrokenPipeError, ConnectionError) as e:
            # Client disconnected mid-stream; log and exit silently
            elapsed = time.monotonic() - request_start
            logger.info(
                "chat_completion client_disconnected",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "duration_ms": int(elapsed * 1000),
                    "request_id": request_id,
                    "stream": True,
                    "error": str(e),
                },
            )
            _update_metrics(elapsed, error=True, error_type="client_disconnected")
        except BrowserError as e:
            elapsed = time.monotonic() - request_start
            snapshot = get_page_snapshot(page)
            logger.warning(
                "chat_completion request_end",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "status_code": 502,
                    "duration_ms": int(elapsed * 1000),
                    "error": str(e),
                    "error_type": "browser_error",
                    "request_id": request_id,
                    "stream": True,
                    "page_url": snapshot.get("url"),
                    "page_title": snapshot.get("title"),
                },
            )
            _update_metrics(elapsed, error=True, error_type="browser_error")
            if not headers_sent:
                self._send_error(502, f"Browser error: {e}. Try restarting the browser.", "server_error", request_id)
            else:
                self._send_sse(
                    json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "length",
                                }
                            ],
                            "error": {"message": str(e), "type": "server_error"},
                        }
                    )
                )
                # Always close the stream, even on failure, or the client
                # waits for a response that will never come.
                self._send_sse("[DONE]")
        except Exception as e:
            elapsed = time.monotonic() - request_start
            snapshot = get_page_snapshot(page)
            logger.exception(
                "chat_completion request_end",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "status_code": 500,
                    "duration_ms": int(elapsed * 1000),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                    "stream": True,
                    "page_url": snapshot.get("url"),
                    "page_title": snapshot.get("title"),
                },
            )
            _update_metrics(elapsed, error=True, error_type=type(e).__name__)
            if not headers_sent:
                self._send_error(500, f"Internal error: {e}", "server_error", request_id)
            else:
                self._send_sse(
                    json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "length",
                                }
                            ],
                            "error": {"message": str(e), "type": "server_error"},
                        }
                    )
                )
                # Always close the stream, even on failure, or the client
                # waits for a response that will never come.
                self._send_sse("[DONE]")
        finally:
            self._release_browser_lock(browser_lock)

    def _is_new_response(self, response: dict, baseline: dict | None) -> bool:
        """Delegate to the shared implementation.

        This used to be a second, divergent copy: it was missing the thinking
        check, so the non-streaming path could never recognise a
        thinking-only turn as a new response.
        """
        return is_new_response(response, baseline)

    def _handle_chat_completions(self) -> None:
        """Handle a chat completion request."""
        # Generate or extract request ID for correlation (early, for error responses)
        request_id = (
            self.headers.get(REQUEST_ID_HEADER.lower()) or uuid.uuid4().hex[:16]
        )

        # Read and parse request body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send_error(
                400, "Invalid 'Content-Length' header", request_id=request_id
            )
            return
        if content_length < 0:
            self._send_error(
                400, "Invalid 'Content-Length' header", request_id=request_id
            )
            return
        if content_length > MAX_REQUEST_BYTES:
            self._send_error(
                413, "Request body is too large", "server_error", request_id=request_id
            )
            return
        if content_length == 0:
            self._send_error(400, "Request body is empty", request_id=request_id)
            return

        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_error(400, f"Invalid JSON: {e}", request_id=request_id)
            return
        if not isinstance(data, dict):
            self._send_error(
                400, "Request body must be a JSON object", request_id=request_id
            )
            return

        # Extract model
        model = data.get("model")
        if not isinstance(model, str) or not model:
            self._send_error(400, "Missing 'model' field", request_id=request_id)
            return

        # Map model to site
        site_id = self.model_map.get(model)
        if not site_id:
            available = ", ".join(self.model_map.keys())
            self._send_error(
                400,
                f"Unknown model: {model!r}. Available models: {available}. "
                f"Use GET /v1/models to list available models.",
                "invalid_request_error",
                request_id=request_id,
            )
            return

        # Extract user messages
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_error(400, "Missing 'messages' field", request_id=request_id)
            return
        if not all(isinstance(message, dict) for message in messages):
            self._send_error(
                400, "Each message must be a JSON object", request_id=request_id
            )
            return

        # Build prompt from messages. Web chats receive only the latest user
        # message; the browser tab already contains prior conversation context.
        prompt = self._build_web_prompt(messages)
        if not prompt.strip():
            self._send_error(
                400, "No user message content found", request_id=request_id
            )
            return

        # Get page for this model
        page = self.tab_map.get(model)
        if page is None:
            self._send_error(
                502,
                f"No browser tab for model: {model}. The browser tab may have crashed. "
                f"Check the browser window and restart the server.",
                "server_error",
                request_id=request_id,
            )
            return
        tab_index = list(self.model_map.keys()).index(model) + 1
        stream = data.get("stream") is True

        # Request logging
        request_start = time.monotonic()
        logger = logging.getLogger(__name__)
        logger.info(
            "chat_completion request_start",
            extra={
                "model": model,
                "tab_index": tab_index,
                "messages": len(messages),
                "request_id": request_id,
                "prompt_preview": prompt[:200],
                "stream": stream,
            },
        )

        browser_timeout = self._server_setting(
            "browser_timeout", DEFAULT_BROWSER_TIMEOUT
        )
        if stream:
            self._handle_streaming_chat_completions(
                request_id,
                model,
                site_id,
                page,
                tab_index,
                prompt,
                request_start,
                browser_timeout,
            )
            return

        # Serialize browser automation: Playwright pages are not thread-safe.
        browser_lock, lock_acquired, lock_timeout = self._acquire_browser_lock(model)
        if not lock_acquired:
            logger.warning(
                "chat_completion tab_busy",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "lock_timeout": lock_timeout,
                    "busy_models": self._busy_models(),
                    "request_id": request_id,
                    "stream": False,
                },
            )
            self._send_busy_error(request_id, model, lock_timeout)
            _update_metrics(time.monotonic() - request_start, error=True, error_type="lock_timeout")
            return

        status = None
        response_result = None
        try:
            extraction = extract_js(site_id)
            baseline = capture_response(page, extraction) if extraction else None
            status, page = self._inject_and_submit_with_recovery(
                page, model, site_id, prompt, tab_index
            )
            if (
                isinstance(status, dict)
                and status.get("inject") == "OK"
                and status.get("submit") in {"OK", "ENTER_SENT", "ENTER_SENT_UNVERIFIED"}
                and extraction
            ):
                response_result = wait_for_response(
                    page,
                    extraction,
                    browser_timeout,
                    idle_timeout=self._server_setting(
                        "response_idle_timeout", DEFAULT_RESPONSE_IDLE_TIMEOUT
                    ),
                    poll_interval=self._server_setting(
                        "poll_interval", DEFAULT_POLL_INTERVAL
                    ),
                    baseline=baseline,
                )
        except BrowserError as e:
            elapsed = time.monotonic() - request_start
            snapshot = get_page_snapshot(page)
            logger.warning(
                "chat_completion request_end",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "status_code": 502,
                    "duration_ms": int(elapsed * 1000),
                    "error": str(e),
                    "error_type": "browser_error",
                    "request_id": request_id,
                    "page_url": snapshot.get("url"),
                    "page_title": snapshot.get("title"),
                },
            )
            _update_metrics(elapsed, error=True, error_type="browser_error")
            self._send_error(502, f"Browser error: {e}. Try restarting the browser.", "server_error", request_id)
            return
        except BrowserOperationTimeout as e:
            # A browser operation that never completed means the Playwright
            # worker thread is wedged (e.g. a page.evaluate() on a stalled
            # page main thread). Every later browser call will fail too, so
            # say so plainly instead of reporting a generic 500.
            elapsed = time.monotonic() - request_start
            logger.error(
                "chat_completion browser_unresponsive",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "duration_ms": int(elapsed * 1000),
                    "error": str(e),
                    "error_type": "browser_timeout",
                    "request_id": request_id,
                },
            )
            _update_metrics(elapsed, error=True, error_type="browser_timeout")
            self._send_error(
                502,
                f"Browser is unresponsive: {e}. The browser worker thread is "
                "blocked and will not recover on its own — restart sbsllm.",
                "server_error",
                request_id,
            )
            return
        except Exception as e:
            elapsed = time.monotonic() - request_start
            snapshot = get_page_snapshot(page)
            logger.exception(
                "chat_completion request_end",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "status_code": 500,
                    "duration_ms": int(elapsed * 1000),
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "request_id": request_id,
                    "page_url": snapshot.get("url"),
                    "page_title": snapshot.get("title"),
                },
            )
            _update_metrics(elapsed, error=True, error_type=type(e).__name__)
            self._send_error(500, f"Internal error: {e}", "server_error", request_id)
            return
        finally:
            self._release_browser_lock(browser_lock)

        if not isinstance(status, dict):
            logger.error(
                "chat_completion invalid_status",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "request_id": request_id,
                    "status": str(status),
                },
            )
            self._send_error(
                500,
                "Browser automation returned an invalid status",
                "server_error",
                request_id,
            )
            _update_metrics(time.monotonic() - request_start, error=True, error_type="invalid_status")
            return
        browser_error = next(
            (
                status[field]
                for field in ("inject", "submit")
                if isinstance(status.get(field), str)
                and status[field].startswith("BROWSER_ERROR:")
            ),
            None,
        )
        if browser_error:
            snapshot = get_page_snapshot(page)
            logger.warning(
                "chat_completion browser_error",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "request_id": request_id,
                    "error": browser_error,
                    "page_url": snapshot.get("url"),
                    "page_title": snapshot.get("title"),
                },
            )
            self._send_error(
                502,
                f"Browser error: {browser_error}. Check the browser tab for CAPTCHA or login prompts.",
                "server_error",
                request_id,
            )
            _update_metrics(time.monotonic() - request_start, error=True, error_type="browser_error")
            return
        if status.get("inject") != "OK" or status.get("submit") not in {
            "OK",
            "ENTER_SENT",
            "ENTER_SENT_UNVERIFIED",
        }:
            logger.warning(
                "chat_completion inject_submit_failed",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "request_id": request_id,
                    "inject_status": status.get("inject"),
                    "submit_status": status.get("submit"),
                },
            )
            if status.get("inject") != "OK":
                content = f"Failed to inject prompt: {status['inject']}"
            else:
                content = f"Failed to submit: {status['submit']}"
            response = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "request_id": request_id,
            }
            _update_metrics(time.monotonic() - request_start, error=False)
            self._send_json(200, response, request_id)
            return

        if extraction:
            if response_result is None:
                self._send_error(
                    502,
                    f"Response extraction is not supported for site: {site_id}",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="extraction_unsupported")
                return
            if response_result.get("timed_out"):
                self._send_error(
                    504,
                    "Request timeout: assistant response was not completed. "
                    "The chat may be slow or stuck. Try again or refresh the page.",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="timeout")
                return
            if not response_result.get("found"):
                snapshot = get_page_snapshot(page)
                page_text = snapshot.get("text_preview", "")[:200]
                logger.warning(
                    "chat_completion no_response",
                    extra={
                        "model": model,
                        "tab_index": tab_index,
                        "request_id": request_id,
                        "page_url": snapshot.get("url"),
                        "page_title": snapshot.get("title"),
                        "page_text_preview": page_text,
                        "login_wall": response_result.get("login_wall"),
                    },
                )
                if response_result.get("login_wall"):
                    self._send_error(
                        502,
                        f"The {site_id} chat requires a signed-in session. "
                        f"Please log in to {site_id} in the browser tab, then retry.",
                        "server_error",
                        request_id,
                    )
                    _update_metrics(time.monotonic() - request_start, error=True, error_type="login_required")
                    return
                self._send_error(
                    504,
                    "No assistant response was detected after submission. "
                    "The chat may have encountered an error or is still loading. "
                    "Check the browser tab for error messages.",
                    "server_error",
                    request_id,
                )
                _update_metrics(time.monotonic() - request_start, error=True, error_type="no_response")
                return
            content = response_result.get("content", "")
            thinking = response_result.get("thinking") or ""
        else:
            # No extraction for this site — surface a visible error instead of
            # returning a fake "Prompt sent to X successfully" message, which
            # the local chat would render as the assistant's answer.
            self._send_error(
                502,
                f"Response extraction is not supported for site: {site_id}. "
                f"This site may require a login or page refresh.",
                "server_error",
                request_id,
            )
            _update_metrics(time.monotonic() - request_start, error=True, error_type="extraction_unsupported")
            return

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "request_id": request_id,
        }
        if thinking:
            response["thinking"] = thinking

        elapsed = time.monotonic() - request_start
        logger.info(
            "chat_completion request_end",
            extra={
                "model": model,
                "tab_index": tab_index,
                "status_code": 200,
                "duration_ms": int(elapsed * 1000),
                "request_id": request_id,
            },
        )

        _update_metrics(elapsed, error=False)
        self._send_json(200, response, request_id)

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build a prompt string from OpenAI messages format."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = " ".join(text_parts)
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        return "\n\n".join(parts)


class Server:
    """OpenAI-compatible HTTP server."""

    def _recover_page(self, model: str, site_id: str) -> object | None:
        """Attempt to recover a crashed/stale browser tab.

        Opens a fresh page for the site URL and updates the server's tab_map
        so subsequent requests use the new page. Returns the new page or None.
        """
        from sbsllm.browser import recover_page

        site = get_site(site_id)
        url = site["url"] if site else None
        if not url:
            return None
        logger = logging.getLogger(__name__)
        logger.info(
            "recovering page for model=%s site=%s url=%s",
            model, site_id, url,
        )
        new_page = recover_page(url)
        if new_page is not None:
            self.tab_map[model] = new_page
        return new_page

    def __init__(
        self,
        model_map: dict[str, str] | None = None,
        tab_map: dict[str, object] | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        browser_timeout: int = DEFAULT_BROWSER_TIMEOUT,
        browser_lock_timeout: int = DEFAULT_BROWSER_LOCK_TIMEOUT,
        health_interval: float = DEFAULT_HEALTH_INTERVAL,
        response_idle_timeout: float = DEFAULT_RESPONSE_IDLE_TIMEOUT,
        response_done_confirm: float = DEFAULT_RESPONSE_DONE_CONFIRM,
        busy_patience: float = DEFAULT_BUSY_PATIENCE,
        first_token_timeout: float = DEFAULT_FIRST_TOKEN_TIMEOUT,
        keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self.model_map = model_map or {}
        self.tab_map = tab_map or {}
        self.host = host
        self.port = port
        self.browser_timeout = browser_timeout
        self.browser_lock_timeout = browser_lock_timeout
        self.health_interval = max(float(health_interval), 1.0)
        self.response_idle_timeout = max(float(response_idle_timeout), 0.1)
        self.response_done_confirm = max(float(response_done_confirm), 0.0)
        self.busy_patience = max(float(busy_patience), 0.1)
        self.first_token_timeout = max(float(first_token_timeout), 0.1)
        self.keepalive_interval = max(float(keepalive_interval), 0.1)
        self.poll_interval = max(float(poll_interval), 0.01)
        self._server: _SBSHTTPServer | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._serve_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._state_lock = threading.Lock()
        self.model_locks = _ModelLockRegistry()
        self._health_thread: threading.Thread | None = None
        self._shutdown_initiated = False
        self._signal_handler_installed = False
        self._previous_signal_handlers: dict[int, Any] = {}

    def install_signal_handlers(self) -> bool:
        """Install graceful-shutdown signal handlers from the main thread."""
        return self._install_signal_handlers()

    def _install_signal_handlers(self) -> bool:
        """Install signal handlers when called from the main thread."""
        if self._signal_handler_installed:
            return True
        if threading.current_thread() is not threading.main_thread():
            return False
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                self._previous_signal_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._signal_handler)
            self._signal_handler_installed = True
            return True
        except (ValueError, OSError):
            return False

    def _restore_signal_handlers(self) -> None:
        if not self._signal_handler_installed:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        for signum, handler in self._previous_signal_handlers.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError):
                pass
        self._previous_signal_handlers.clear()
        self._signal_handler_installed = False

    def _signal_handler(self, signum: int, frame: object) -> None:
        """Handle shutdown signals."""
        if not self._shutdown_initiated:
            self._shutdown_initiated = True
            print(f"\nReceived signal {signum}, shutting down...", flush=True)
            self.stop()

    @property
    def url(self) -> str:
        """Return the URL clients can use, including an ephemeral bound port."""
        port = self.port
        if self._server is not None:
            port = self._server.server_address[1]
        return f"http://{self.host}:{port}/"

    def start(self, announce: bool = True) -> None:
        """Start the server and signal once its socket is bound."""
        self._stop_requested.clear()
        self._ready.clear()
        self._startup_error = None
        self._shutdown_initiated = False
        self._serve_thread = threading.current_thread()
        self._start_health_monitor()
        try:
            # Handler instances receive this server's maps without mutating shared
            # class state, so concurrent or restarted servers cannot cross routes.
            handler_cls = type(
                "ConfiguredOpenAIHandler",
                (OpenAIHandler,),
                {"model_map": self.model_map, "tab_map": self.tab_map},
            )

            with self._state_lock:
                self._server = _SBSHTTPServer((self.host, self.port), handler_cls)
                server = self._server
            server.daemon_threads = True
            server.block_on_close = False
            server.timeout = 0.1
            # Handlers resolve these via `self.server`; without them they
            # silently fell back to defaults.
            server.model_locks = self.model_locks
            server.browser_lock_timeout = self.browser_lock_timeout
            server.browser_timeout = self.browser_timeout
            server.response_idle_timeout = self.response_idle_timeout
            server.response_done_confirm = self.response_done_confirm
            server.busy_patience = self.busy_patience
            server.first_token_timeout = self.first_token_timeout
            server.keepalive_interval = self.keepalive_interval
            server.poll_interval = self.poll_interval
            server.model_map = self.model_map
            server.tab_map = self.tab_map
            if announce:
                print(
                    f"sbsllm server listening on {self.url.rstrip('/')}",
                    flush=True,
                )
                print(
                    f"OpenAI-compatible server URL: {self.url}",
                    flush=True,
                )
                print(f"Models: {list(self.model_map.keys())}", flush=True)
            self._ready.set()
            try:
                while not self._stop_requested.is_set():
                    try:
                        server.handle_request()
                    except TimeoutError:
                        continue
                    except OSError:
                        if self._stop_requested.is_set():
                            break
                        raise
                    except KeyboardInterrupt:
                        print("\nShutting down server...", flush=True)
                        break
            finally:
                self._close_server()
        except BaseException as e:
            self._startup_error = e
            self._ready.set()
            self._close_server()
            raise

    def wait_until_ready(self, timeout: float) -> None:
        """Wait until the server is bound or report its startup failure."""
        if not self._ready.wait(timeout):
            raise TimeoutError(f"Server did not start within {timeout} seconds")
        if self._startup_error is not None:
            raise RuntimeError(
                f"Server failed to start: {self._startup_error}"
            ) from self._startup_error

    def _close_server(self) -> _SBSHTTPServer | None:
        with self._state_lock:
            server = self._server
            self._server = None
        if server is not None:
            try:
                server.server_close()
            except OSError:
                pass
        if self._serve_thread is threading.current_thread():
            self._serve_thread = None
        return server

    def stop(self) -> None:
        """Stop accepting requests and wait briefly for active handlers."""
        self._stop_requested.set()
        serve_thread = self._serve_thread
        server = self._close_server()
        if (
            server is not None
            and serve_thread is not None
            and serve_thread is not threading.current_thread()
        ):
            serve_thread.join(timeout=5)
            if not server.wait_for_idle(max(5.0, float(self.browser_timeout))):
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Server shutdown timed out with %s active request(s)",
                    server._active_requests,
                )
        self._restore_signal_handlers()

    def _start_health_monitor(self) -> None:
        """Start a background thread that periodically checks tab health."""
        if self._health_thread is not None and self._health_thread.is_alive():
            return
        self._health_thread = threading.Thread(
            target=self._health_monitor_loop, daemon=True, name="health-monitor"
        )
        self._health_thread.start()

    def _stop_health_monitor(self) -> None:
        """Signal the health monitor thread to stop."""
        self._stop_requested.set()
        ht = self._health_thread
        if ht is not None and ht is not threading.current_thread() and ht.is_alive():
            ht.join(timeout=2)

    def _health_monitor_loop(self) -> None:
        """Periodically check every tab's health and recover unhealthy ones."""
        logger = logging.getLogger(__name__)
        while not self._stop_requested.is_set():
            try:
                self._check_tab_health_once(logger)
            except Exception:
                logger.debug("Health monitor iteration failed", exc_info=True)
            self._stop_requested.wait(timeout=self.health_interval)

    def _check_tab_health_once(self, logger) -> None:
        """Check health of all known tabs; recover unhealthy ones."""
        from sbsllm.browser import check_page_health, is_running

        if not is_running():
            logger.warning("Health monitor: browser not running")
            return
        with self._state_lock:
            tab_items = list(self.tab_map.items())
        for model, page in tab_items:
            if page is None:
                continue
            try:
                healthy = check_page_health(page)
            except (PlaywrightError, RuntimeError, BrowserError, BrowserOperationTimeout):
                healthy = False
            if not healthy:
                logger.warning(
                    "Health monitor: tab for model=%s unhealthy; recovering",
                    model,
                )
                site_id = self.model_map.get(model)
                if site_id:
                    new_page = self._recover_page(model, site_id)
                    if new_page is not None:
                        logger.info(
                            "Health monitor: recovered tab for model=%s",
                            model,
                        )


def create_server(
    model_map: dict[str, str] | None = None,
    tab_map: dict[str, object] | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    browser_timeout: int = DEFAULT_BROWSER_TIMEOUT,
    browser_lock_timeout: int = DEFAULT_BROWSER_LOCK_TIMEOUT,
    health_interval: float = DEFAULT_HEALTH_INTERVAL,
    response_idle_timeout: float = DEFAULT_RESPONSE_IDLE_TIMEOUT,
    response_done_confirm: float = DEFAULT_RESPONSE_DONE_CONFIRM,
    busy_patience: float = DEFAULT_BUSY_PATIENCE,
    first_token_timeout: float = DEFAULT_FIRST_TOKEN_TIMEOUT,
    keepalive_interval: float = DEFAULT_KEEPALIVE_INTERVAL,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
) -> Server:
    """Create a new server instance."""
    return Server(
        model_map=model_map or {},
        tab_map=tab_map or {},
        host=host,
        port=port,
        browser_timeout=browser_timeout,
        browser_lock_timeout=browser_lock_timeout,
        health_interval=health_interval,
        response_idle_timeout=response_idle_timeout,
        response_done_confirm=response_done_confirm,
        busy_patience=busy_patience,
        first_token_timeout=first_token_timeout,
        keepalive_interval=keepalive_interval,
        poll_interval=poll_interval,
    )
