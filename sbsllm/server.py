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

from .browser import (
    BrowserError,
    capture_response,
    check_page_health,
    get_page_snapshot,
    inject_and_submit,
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
# Default timeout for acquiring browser lock (seconds)
DEFAULT_BROWSER_LOCK_TIMEOUT = 10
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
    # ruff: noqa: PLW0602 - in-place modification of globals (ruff false positive)
    global _request_count, _request_latencies, _error_count, _error_types
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

    def _send_sse(self, data: str) -> None:
        """Write one server-sent event and flush it immediately."""
        try:
            self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionError):
            raise

    def _send_sse_headers(self, request_id: str) -> None:
        """Send headers for an OpenAI-compatible streaming response."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
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
        if new_page is not None and self.server is not None:
            self.server.tab_map[model] = new_page
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

    def _acquire_browser_lock(self):
        """Acquire the browser lock and return (lock, acquired, timeout)."""
        server_obj = getattr(self, "server", None)
        browser_lock = getattr(server_obj, "browser_lock", None)
        browser_lock_timeout = getattr(
            server_obj, "browser_lock_timeout", DEFAULT_BROWSER_LOCK_TIMEOUT
        )
        if browser_lock is None:
            return None, True, 0
        acquired = browser_lock.acquire(timeout=browser_lock_timeout)
        return browser_lock, acquired, browser_lock_timeout

    def _release_browser_lock(self, browser_lock) -> None:
        if browser_lock is not None:
            browser_lock.release()

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
        browser_lock, lock_acquired, lock_timeout = self._acquire_browser_lock()
        if not lock_acquired:
            logger.warning(
                "chat_completion lock_acquired",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "lock_timeout": lock_timeout,
                    "request_id": request_id,
                    "stream": True,
                },
            )
            self._send_error(
                503,
                "Server busy: could not acquire browser lock within timeout. "
                "Reduce concurrent requests or increase browser_lock_timeout.",
                "server_error",
                request_id,
            )
            _update_metrics(time.monotonic() - request_start, error=True, error_type="lock_timeout")
            return

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        extraction = extract_js(site_id)
        headers_sent = False
        last_emitted = ""
        last_thinking = ""
        role_sent = False
        response = {
            "found": False,
            "content": "",
            "thinking": None,
            "done": False,
            "count": 0,
        }
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

            deadline = time.monotonic() + max(float(browser_timeout), 1.0)
            idle_deadline = deadline
            idle_timeout = min(float(browser_timeout), 10.0)
            self._send_sse_headers(request_id)
            headers_sent = True
            while time.monotonic() <= deadline:
                response = capture_response(page, extraction)

                # Check done condition even when content hasn't changed,
                # so we don't miss a transition from loading -> idle.
                if response.get("done") and self._is_new_response(response, baseline):
                    content = response.get("content", "")
                    thinking = response.get("thinking") or ""
                    delta = {}
                    if content and not role_sent:
                        delta = {"role": "assistant", "content": content}
                        role_sent = True
                        last_emitted = content
                    elif content != last_emitted:
                        if content.startswith(last_emitted):
                            delta_content = content[len(last_emitted) :]
                        else:
                            delta_content = content
                        if delta_content:
                            delta = {"content": delta_content}
                        last_emitted = content
                    if thinking and thinking != last_thinking:
                        delta["thinking"] = thinking.removeprefix(last_thinking)
                        last_thinking = thinking
                    if delta:
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
                                            "delta": delta,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                        )
                    break

                if not self._is_new_response(response, baseline):
                    time.sleep(0.25)
                    continue

                content = response.get("content", "")
                thinking = response.get("thinking") or ""
                if content and not role_sent:
                    delta = {"role": "assistant", "content": content}
                    role_sent = True
                    last_emitted = content
                elif content != last_emitted:
                    if content.startswith(last_emitted):
                        delta_content = content[len(last_emitted) :]
                    else:
                        delta_content = content
                    if delta_content:
                        delta = {"content": delta_content}
                    else:
                        delta = {}
                    last_emitted = content
                else:
                    delta = {}

                if thinking and thinking != last_thinking:
                    delta["thinking"] = thinking.removeprefix(last_thinking)
                    last_thinking = thinking

                if delta:
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
                                        "delta": delta,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    )
                    headers_sent = True

                if response.get("done"):
                    break
                # Track idle time: reset timer whenever content changes
                if content != last_emitted or thinking != last_thinking:
                    idle_deadline = time.monotonic() + idle_timeout
                elif time.monotonic() >= idle_deadline and response.get("content"):
                    # Content is stable for the idle window — treat as done
                    response["done"] = True
                    break
                time.sleep(0.25)

            if not role_sent and response.get("content"):
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
                                    "delta": {"role": "assistant", "content": response["content"]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                )
                headers_sent = True

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
                                "finish_reason": "stop" if response.get("done") else "length",
                            }
                        ],
                    }
                )
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
                self._send_sse("[DONE]")
        except Exception as e:  # noqa: BLE001
            elapsed = time.monotonic() - request_start
            snapshot = get_page_snapshot(page)
            logger.error(
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
                exc_info=True,
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
                self._send_sse("[DONE]")
        finally:
            self._release_browser_lock(browser_lock)

    def _is_new_response(self, response: dict, baseline: dict | None) -> bool:
        if baseline is None:
            return response.get("found", False)
        if response.get("count", 0) > baseline.get("count", 0):
            return True
        return response.get("found", False) and response.get("content", "") != baseline.get(
            "content", ""
        )

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

        server_obj = getattr(self, "server", None)
        browser_timeout = getattr(
            server_obj, "browser_timeout", DEFAULT_BROWSER_TIMEOUT
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
        browser_lock, lock_acquired, lock_timeout = self._acquire_browser_lock()
        if not lock_acquired:
            logger.warning(
                "chat_completion lock_acquired",
                extra={
                    "model": model,
                    "tab_index": tab_index,
                    "lock_timeout": lock_timeout,
                    "request_id": request_id,
                    "stream": False,
                },
            )
            self._send_error(
                503,
                "Server busy: could not acquire browser lock within timeout. "
                "Reduce concurrent requests or increase browser_lock_timeout.",
                "server_error",
                request_id,
            )
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
                    idle_timeout=min(float(browser_timeout), 10.0),
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
        except Exception as e:  # noqa: BLE001
            elapsed = time.monotonic() - request_start
            snapshot = get_page_snapshot(page)
            logger.error(
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
                exc_info=True,
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
                logger.warning(
                    "chat_completion no_response",
                    extra={
                        "model": model,
                        "tab_index": tab_index,
                        "request_id": request_id,
                        "page_url": snapshot.get("url"),
                        "page_title": snapshot.get("title"),
                        "page_text_preview": snapshot.get("text_preview", "")[:200],
                    },
                )
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
        else:
            content = f"Prompt sent to {site_id} successfully."

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
        if response_result is not None and response_result.get("thinking"):
            response["thinking"] = response_result["thinking"]

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

    def __init__(
        self,
        model_map: dict[str, str] | None = None,
        tab_map: dict[str, object] | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        browser_timeout: int = DEFAULT_BROWSER_TIMEOUT,
        browser_lock_timeout: int = DEFAULT_BROWSER_LOCK_TIMEOUT,
    ):
        self.model_map = model_map or {}
        self.tab_map = tab_map or {}
        self.host = host
        self.port = port
        self.browser_timeout = browser_timeout
        self.browser_lock_timeout = browser_lock_timeout
        self._server: _SBSHTTPServer | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._serve_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._state_lock = threading.Lock()
        self.browser_lock = threading.Lock()
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


def create_server(
    model_map: dict[str, str] | None = None,
    tab_map: dict[str, object] | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    browser_timeout: int = DEFAULT_BROWSER_TIMEOUT,
    browser_lock_timeout: int = DEFAULT_BROWSER_LOCK_TIMEOUT,
) -> Server:
    """Create a new server instance."""
    return Server(
        model_map=model_map or {},
        tab_map=tab_map or {},
        host=host,
        port=port,
        browser_timeout=browser_timeout,
        browser_lock_timeout=browser_lock_timeout,
    )
