"""Browser automation via Playwright + Chromium."""

from __future__ import annotations

import logging
import os
import queue
import random
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from playwright.sync_api import BrowserContext, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError

logger = logging.getLogger(__name__)

# Logger for structured JSON state transition events in wait_for_response.
# Separate from the module logger so callers can filter these without
# disabling general debug noise.
_state_logger = logging.getLogger(__name__ + ".response_state")

# Logger for worker lifecycle events (start/stop/drain/exit). Separate so
# operators can trace queue state without sifting through per-poll noise.
_worker_logger = logging.getLogger(__name__ + ".worker_lifecycle")

# Isolated user data directory for Chromium profile
# The persistent user-data directory holds login sessions between runs, so
# it must be stable across invocations. But in CI and parallel test suites
# every process needs its own copy: two Chromium instances locking the same
# profile is the classic "could not connect to browser" hang. SBSLLM_USER_DATA_DIR
# lets callers (and tests) point at a private directory; the default is the
# well-known path used by the CLI.
USER_DATA_DIR = "/tmp/sbsllm-chrome"


def user_data_dir() -> str:
    """Return the active user-data directory, honoring SBSLLM_USER_DATA_DIR.

    Read at call time rather than import time so tests can switch profiles
    mid-process without reloading the module.
    """
    return os.environ.get("SBSLLM_USER_DATA_DIR") or USER_DATA_DIR


# Timeout for page operations (seconds)
PAGE_TIMEOUT = 30

# Retry configuration for transient failures
MAX_RETRIES = 3
BASE_RETRY_DELAY = 0.5  # seconds
MAX_RETRY_DELAY = 5.0  # seconds
RETRY_JITTER = 0.1  # seconds

# Timeout for browser queue operations (seconds)
BROWSER_OPERATION_TIMEOUT = 120
# Upper bound for a single response-extraction poll (seconds). page.evaluate()
# takes no timeout of its own, so a page with a stalled main thread blocks
# forever; this bound makes such a poll fail fast and lets the request end
# with a terminal chunk instead of holding the browser lock.
CAPTURE_TIMEOUT = 15.0

# How long an unchanged payload is tolerated while the site still claims to
# be generating. Web chats pause mid-answer (thinking, re-render, rate limit)
# and cutting there truncates the reply -- but a site whose spinner never
# clears must not hold the tab until browser_timeout.
DEFAULT_BUSY_PATIENCE = 20.0
# Tolerance for a *thinking-only* stall: content is still empty, only the
# reasoning trace is present, and the site keeps reporting "generating".
# Thinking traces update in bursts with natural pauses between chunks, and
# models like o1 / deepseek-reasoner / Claude-thinking can think for minutes,
# so this must be far larger than DEFAULT_BUSY_PATIENCE. A site whose
# thinking never moves again after this is genuinely stuck.
DEFAULT_THINKING_PATIENCE = 120.0
# After submitting, how long to wait for the web chat to produce any output
# at all. Without this a selector mismatch meant silence until browser_timeout,
# and the local chat simply spun forever.
DEFAULT_FIRST_TOKEN_TIMEOUT = 60.0

# Global browser state
_context: BrowserContext | None = None
_playwright_instance: Any = None

# Type variable for retry decorator
F = TypeVar("F", bound=Callable[..., Any])


# Queue-based dispatch for Playwright operations.
# Playwright binds to a specific OS thread (via greenlet); all Playwright API
# calls must run on that thread or a "Cannot switch to a different thread"
# RuntimeError is raised by gevent/Playwright.
_playwright_thread_id: int | None = None
_worker_start_lock = threading.Lock()
_browser_queue: queue.Queue[Any] = queue.Queue()
_browser_thread: threading.Thread | None = None


def _browser_worker_loop() -> None:
    """Process browser operations from the queue on the Playwright thread."""
    global _playwright_thread_id
    worker_thread = threading.current_thread()
    _playwright_thread_id = worker_thread.ident
    try:
        while True:
            item = _browser_queue.get()
            try:
                if item is None:
                    _worker_logger.info(
                        "browser_worker_received_sentinel",
                        extra={"action": "exiting"},
                    )
                    break
                func, args, kwargs, event, result_holder = item
                try:
                    result = func(*args, **kwargs)
                    result_holder.append(("ok", result))
                except Exception as e:  # noqa: BLE001
                    result_holder.append(("error", e))
                finally:
                    event.set()
            finally:
                _browser_queue.task_done()
    finally:
        if _playwright_thread_id == worker_thread.ident:
            _playwright_thread_id = None
            _worker_logger.info(
                "browser_worker_exited",
                extra={"action": "cleared thread id"},
            )


def _is_browser_thread() -> bool:
    return (
        _playwright_thread_id is not None
        and threading.current_thread().ident == _playwright_thread_id
    )


def _drain_browser_queue() -> int:
    """Discard all pending items from the browser queue.

    A worker that exited mid-operation (or was faked to be dead by a caller)
    can leave orphaned request items and ``None`` shutdown sentinels in the
    shared queue. Those linger across the process lifetime because the queue is
    module-level, so a freshly started worker would immediately dequeue a stale
    ``None`` and exit before ever seeing its real work item -- surfacing as
    "Browser worker stopped before completing the operation".

    Draining is safe only when no live worker is consuming the queue; callers
    must hold that invariant (no worker running, or worker already dead).
    If a worker is currently alive we refuse to drain: the queue is FIFO and a
    live worker may be blocked in ``get()`` on an item we would otherwise
    discard, which would silently drop a real in-flight operation. The ``0``
    return value signals "nothing drained" to callers that count on it.
    """
    live = _browser_thread is not None and _browser_thread.is_alive()
    if live:
        _worker_logger.info(
            "browser_queue_drain_refused",
            extra={"reason": "live worker is consuming the queue"},
        )
        return 0
    drained = 0
    while True:
        try:
            _browser_queue.get_nowait()
        except queue.Empty:
            break
        drained += 1
    if drained:
        logger.debug("Drained %d stale item(s) from browser queue", drained)
        _worker_logger.info(
            "browser_queue_drained",
            extra={"drained": drained},
        )
    return drained


def _start_browser_worker() -> threading.Thread:
    """Start and return the singleton browser worker thread."""
    global _browser_thread, _playwright_thread_id
    with _worker_start_lock:
        if _browser_thread is None or not _browser_thread.is_alive():
            if _browser_thread is not None:
                _playwright_thread_id = None
                _worker_logger.info(
                    "browser_worker_dead_restarting",
                    extra={"reason": "previous worker not alive"},
                )
            # Discard any stale sentinels/orphaned items left behind by a
            # previous worker so the new one doesn't exit before doing real
            # work. We're the only thread that touches the queue right now
            # (the old worker is dead), so this is race-free.
            drained = _drain_browser_queue()
            _browser_thread = threading.Thread(
                target=_browser_worker_loop,
                daemon=True,
                name="browser-worker",
            )
            _browser_thread.start()
            _worker_logger.info(
                "browser_worker_started",
                extra={"drained": drained},
            )
        assert _browser_thread is not None
        return _browser_thread


def _stop_browser_worker(timeout: float = 5.0) -> None:
    """Ask the browser worker to exit after its current operation completes."""
    global _browser_thread
    with _worker_start_lock:
        worker = _browser_thread
        if worker is None:
            return
        if worker is threading.current_thread():
            return
    # If the worker already exited on its own (crash, or faked dead by a
    # caller), sending a ``None`` sentinel would leave it stuck in the shared
    # queue: the next worker started by _start_browser_worker would dequeue it
    # immediately and exit before doing real work. Don't enqueue; just clear
    # the reference and let the stale items drain on next restart.
    if not worker.is_alive():
        with _worker_start_lock:
            if _browser_thread is worker:
                _browser_thread = None
        _worker_logger.info(
            "browser_worker_already_dead",
            extra={"action": "cleared reference, no sentinel enqueued"},
        )
        return
    _browser_queue.put(None)
    worker.join(timeout)
    if worker.is_alive():
        logger.warning("Browser worker did not stop within %ss", timeout)
        _worker_logger.warning(
            "browser_worker_did_not_stop",
            extra={"timeout": timeout},
        )
        return
    with _worker_start_lock:
        if _browser_thread is worker:
            _browser_thread = None
            _worker_logger.info(
                "browser_worker_stopped",
                extra={"action": "joined and cleared"},
            )


def run_in_browser_thread(
    func: Callable[..., Any],
    *args: Any,
    operation_timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a function on the Playwright browser worker thread.

    Playwright pages are not thread-safe; all Playwright API calls must run
    on the thread where sync_playwright was started. This function dispatches
    work to that thread via a queue, blocking the caller until completion.

    Args:
        func: The function to execute on the browser thread.
        *args: Positional arguments to pass to func.
        operation_timeout: Seconds to wait for completion. Defaults to
            BROWSER_OPERATION_TIMEOUT. Polling calls that can block on a
            stalled page (e.g. page.evaluate, which takes no timeout of its
            own) should pass a tighter bound. Named distinctly from a plain
            `timeout` so it cannot shadow a wrapped function's own kwarg.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The return value of func.

    Raises:
        BrowserWorkerStopped: If the worker thread dies before completing the
            operation (e.g. it was stopped during shutdown or crashed). The
            worker is dead -- the caller should treat this as a browser-level
            failure (e.g. a 502 with a "restart sbsllm" hint), not a 500. A
            fresh worker is started transparently on the next call. We do
            *not* silently re-run the operation here, because queued operations
            have side effects (e.g. submitting a prompt) and re-running a
            partially-executed one risks a duplicate send.
        BrowserOperationTimeout: If the operation exceeds its deadline. The
            remaining worker thread is still blocked in `func`, so the browser
            will not recover on its own and sbsllm must be restarted.
        RuntimeError: If the operation completed without producing a result.
        Any exception raised by func.
    """
    if _is_browser_thread():
        return func(*args, **kwargs)

    wait_for = (
        BROWSER_OPERATION_TIMEOUT
        if operation_timeout is None
        else float(operation_timeout)
    )
    event = threading.Event()
    result_holder: list[tuple[str, Any]] = []
    worker = _start_browser_worker()
    _browser_queue.put((func, args, kwargs, event, result_holder))
    deadline = time.monotonic() + wait_for
    while not event.wait(timeout=0.1):
        # Liveness is checked against the worker we enqueued with. If it died,
        # the item was either drained on its restart or the worker crashed
        # before dequeuing it -- either way our item will not be processed, so
        # surface it as a browser-level failure (502) rather than a generic
        # 500. We deliberately do NOT retry: see the BrowserWorkerStopped
        # docstring -- queued ops are not re-run due to side-effect risk.
        if not worker.is_alive():
            _worker_logger.info(
                "browser_worker_died_during_dispatch",
                extra={"reason": "captured worker not alive while waiting"},
            )
            raise BrowserWorkerStopped(
                "Browser worker stopped before completing the operation"
            )
        if time.monotonic() >= deadline:
            raise BrowserOperationTimeout(
                f"Browser operation timed out after {wait_for:g}s"
            )
    if not result_holder:
        raise RuntimeError("Browser operation completed without result")
    status, result = result_holder[0]
    if status == "error":
        raise result
    return result


def is_transient_error(error: PlaywrightError) -> bool:
    """Check if a Playwright error is likely transient."""
    error_str = str(error).lower()
    transient_keywords = [
        "timeout",
        "connection",
        "closed",
        "detached",
        "context",
        "target",
        "session",
        "navigation",
        "net::",
        "protocol error",
    ]
    return any(keyword in error_str for keyword in transient_keywords)


def retry_with_backoff(
    max_retries: int = MAX_RETRIES,
    base_delay: float = BASE_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
    jitter: float = RETRY_JITTER,
    transient_only: bool = True,
) -> Callable[[F], F]:
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        jitter: Random jitter added to delay (seconds)
        transient_only: Only retry on transient errors (PlaywrightError)
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except PlaywrightError as e:
                    last_error = e
                    if attempt == max_retries:
                        break
                    if transient_only and not is_transient_error(e):
                        break
                    delay = min(
                        base_delay * (2**attempt) + random.uniform(0, jitter), max_delay
                    )
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                except Exception:
                    # Non-Playwright errors are not retried
                    raise
            raise last_error

        return wrapper  # type: ignore

    return decorator


class BrowserError(Exception):
    """Raised when a browser operation fails."""


class BrowserOperationTimeout(RuntimeError):
    """A queued browser operation did not finish within its deadline.

    Subclasses RuntimeError so existing callers that guard against a wedged
    browser keep working, but stays distinct from a genuine internal error.
    Note this means the worker thread is still blocked in the call: the
    browser will not recover on its own and sbsllm must be restarted.
    """


class BrowserWorkerStopped(BrowserError, RuntimeError):
    """The browser worker thread died before completing an operation.

    Unlike ``BrowserOperationTimeout`` (the worker is still wedged in the call),
    here the worker is *dead* -- typically because it was explicitly stopped
    (e.g. during shutdown) or crashed. This is a browser-level failure, not an
    internal error, so callers should treat it like other browser errors
    (e.g. surface a 502 and suggest restarting sbsllm) rather than a 500.

    It subclasses both ``BrowserError`` (so existing ``except BrowserError``
    handlers cover it) and ``RuntimeError`` (so the legacy guard in the server
    still catches it).
    """


def setup_logging(
    level: str = "INFO", log_file: str | None = None, json_format: bool = False
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        json_format: If True, output logs in JSON format for structured logging
    """
    import logging as _logging

    handlers: list[_logging.Handler] = [_logging.StreamHandler()]
    if log_file:
        handlers.append(_logging.FileHandler(log_file))

    if json_format:
        # Custom JSON formatter for structured logging
        import json as _json
        from datetime import datetime, timezone

        class JsonFormatter(_logging.Formatter):
            def format(self, record: _logging.LogRecord) -> str:
                log_obj = {
                    "timestamp": datetime.fromtimestamp(
                        record.created, tz=timezone.utc
                    ).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                # Add extra fields if present
                if hasattr(record, "extra"):
                    log_obj.update(record.extra)
                # Add exception info if present
                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)
                return _json.dumps(log_obj)

        formatter = JsonFormatter()
    else:
        formatter = _logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )

    for handler in handlers:
        handler.setFormatter(formatter)

    _logging.basicConfig(
        level=getattr(_logging, level.upper(), _logging.INFO),
        handlers=handlers,
    )


def is_headless() -> bool:
    """Return True when sbsllm should launch Chromium headless.

    Reads the ``SBSLLM_HEADLESS`` environment variable. Defaults to a visible
    browser so the user can log in to chat sites interactively; CI / container
    environments can set ``SBSLLM_HEADLESS=true`` (or any of ``1``, ``yes``,
    ``on``) to run without a window.

    Returns:
        True if Chromium should launch headless, False otherwise.
    """
    headless_env = os.environ.get("SBSLLM_HEADLESS", "").strip().lower()
    return headless_env in ("1", "true", "yes", "on")


def _find_system_chromium() -> str | None:
    """Return path to a system Chromium binary if available, else None."""
    import shutil

    for candidate in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
    ):
        path = shutil.which(candidate)
        if path:
            return path
    return None


def ensure_browser(chrome_bin: str | None = None) -> BrowserContext:
    """Ensure Chromium is running. Launch if not.

    Returns the persistent browser context, which is created with an isolated
    user-data directory so login sessions persist between runs.
    """

    def _do_ensure() -> BrowserContext:
        global _context, _playwright_instance

        if _context is not None and is_running():
            logger.info("Chromium is already running.")
            return _context

        # If we held a stale context/driver that is no longer connected, tear it
        # down before launching a fresh instance so we don't leak the old Playwright
        # driver or hold a lock on the persistent user-data directory.
        if _context is not None or _playwright_instance is not None:
            logger.info("Previous Chromium disconnected — cleaning up before relaunch.")
            close_browser()

        logger.info("Starting Chromium...")
        try:
            playwright_instance = sync_playwright().start()
            _playwright_instance = playwright_instance

            # Allow CI / container environments to override headless mode via
            # the SBSLLM_HEADLESS env var. Defaults to a visible browser so
            # the user can log in to chat sites interactively.
            headless = is_headless()

            launch_args: dict[str, Any] = {
                "headless": headless,
                "java_script_enabled": True,
                "args": [
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",
                ],
            }
            if chrome_bin:
                launch_args["executable_path"] = chrome_bin
            else:
                system_chromium = _find_system_chromium()
                if system_chromium:
                    launch_args["executable_path"] = system_chromium
                else:
                    launch_args["channel"] = "chromium"

            context = playwright_instance.chromium.launch_persistent_context(
                user_data_dir(), **launch_args
            )
            _context = context
            return context
        except PlaywrightError as e:
            if _playwright_instance is not None:
                try:
                    _playwright_instance.stop()
                except PlaywrightError:
                    pass
                _playwright_instance = None
            _context = None
            logger.error(f"Failed to launch Chromium: {e}")
            raise BrowserError(f"Failed to launch Chromium: {e}") from e

    # Calls originating on the browser worker must run directly; queueing them
    # would deadlock because the worker is already processing a browser job.
    if _is_browser_thread():
        return _do_ensure()
    return run_in_browser_thread(_do_ensure)


def _find_blank_page(browser: BrowserContext) -> Page | None:
    try:
        pages = browser.pages
    except PlaywrightError:
        return None

    for page in pages:
        try:
            if page.url in ("about:blank", ""):
                return page
        except PlaywrightError:
            pass
    return None


def _close_blank_pages(browser: BrowserContext, keep_page: Page | None = None) -> None:
    try:
        pages = browser.pages
    except PlaywrightError:
        return

    for page in pages:
        if page is keep_page:
            continue
        try:
            if page.url in ("about:blank", ""):
                page.close()
        except PlaywrightError:
            pass


def _do_open_page(
    url: str, chrome_bin: str | None = None, setup_js: str | None = None
) -> Page:
    """Open a URL on the Playwright browser thread."""
    browser = ensure_browser(chrome_bin)
    page = _find_blank_page(browser)

    if page is None:
        try:
            page = browser.new_page()
        except PlaywrightError as e:
            logger.warning(f"Opening a new tab failed; recreating browser context: {e}")
            close_browser()
            browser = ensure_browser(chrome_bin)
            try:
                page = browser.new_page()
            except PlaywrightError as retry_error:
                close_browser()
                raise BrowserError(
                    f"Failed to open page: {url} - {retry_error}"
                ) from retry_error

    page.set_default_timeout(PAGE_TIMEOUT * 1000)
    try:
        page.goto(url)
    except PlaywrightError as e:
        try:
            page.close()
        except PlaywrightError:
            pass
        _close_blank_pages(browser)
        logger.error(f"Failed to open page: {url} - {e}")
        raise BrowserError(f"Failed to open page: {url} - {e}") from e

    # Run site-specific setup (e.g. set Deep Think level on z.ai) after load.
    if setup_js:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=5000)
            page.evaluate(setup_js)
        except Exception:
            logger.debug("setup_js failed for %s", url, exc_info=True)

    _close_blank_pages(browser, keep_page=page)
    return page


def open_page(
    url: str, chrome_bin: str | None = None, setup_js: str | None = None
) -> Page:
    """Open a URL in a new page/tab. Returns the page."""
    if _is_browser_thread():
        return _do_open_page(url, chrome_bin, setup_js)
    return run_in_browser_thread(_do_open_page, url, chrome_bin, setup_js)


def close_browser() -> None:
    """Close the browser and clean up."""

    def _do_close() -> None:
        global _context, _playwright_instance

        if _context is not None:
            try:
                _context.close()
            except PlaywrightError as e:
                logger.debug(f"Error closing browser: {e}")
            except Exception:  # noqa: BLE001
                logger.debug("Unexpected error closing browser context")
            _context = None

        if _playwright_instance is not None:
            try:
                _playwright_instance.stop()
            except PlaywrightError as stop_error:
                logger.debug(
                    f"Failed to stop Playwright after launch error: {stop_error}"
                )
            except Exception:  # noqa: BLE001
                logger.debug("Unexpected error stopping Playwright")
            _playwright_instance = None

    if _is_browser_thread():
        _do_close()
        return

    if _context is None:
        return
    try:
        return run_in_browser_thread(_do_close)
    except (RuntimeError, BrowserOperationTimeout):
        pass
    _stop_browser_worker()


def _do_run_js(page: Page, js: str) -> str:
    try:
        result = page.evaluate(js)
        return str(result) if result is not None else ""
    except PlaywrightError as e:
        logger.error(f"JS execution failed: {e}")
        return f"BROWSER_ERROR: {e}"


def _do_run_js_value(page: Page, js: str) -> Any:
    # NOTE: page.evaluate() accepts no timeout and page.set_default_timeout()
    # does not bound it (verified: a promise that never settles hangs the
    # call indefinitely). A stalled page main thread is bounded by
    # run_in_browser_thread's deadline in run_js_value instead.
    try:
        return page.evaluate(js)
    except PlaywrightError as e:
        logger.error(f"JS execution failed: {e}")
        raise BrowserError(f"JS execution failed: {e}") from e


def run_js(page: Page, js: str) -> str:
    """Execute JS in a page, preserving Playwright thread affinity."""
    if _is_browser_thread():
        return _do_run_js(page, js)
    try:
        return run_in_browser_thread(_do_run_js, page, js)
    except Exception as e:  # noqa: BLE001
        logger.error(f"JS execution failed: {e}")
        return f"BROWSER_ERROR: {e}"


def run_js_value(page: Page, js: str, timeout: float | None = None) -> Any:
    """Execute JS and return its structured result on the browser thread.

    Unlike `run_js`, failures propagate: this is used by response capture,
    where a failed extraction must end the request rather than be reported
    as a page string.
    """
    if _is_browser_thread():
        return _do_run_js_value(page, js)
    return run_in_browser_thread(
        _do_run_js_value,
        page,
        js,
        operation_timeout=CAPTURE_TIMEOUT if timeout is None else timeout,
    )


def _normalize_response(result: Any) -> dict:
    """Normalize a response extraction result returned by page.evaluate."""
    if not isinstance(result, dict):
        return {
            "found": False,
            "content": "",
            "thinking": None,
            "busy": False,
            "done": False,
            "count": 0,
        }
    content = result.get("content")
    if not isinstance(content, str):
        content = str(content or "").strip()
    thinking = result.get("thinking")
    if not isinstance(thinking, str) or not thinking.strip():
        thinking = None
    else:
        thinking = thinking.strip()
    return {
        "found": bool(result.get("found")),
        "content": content,
        "thinking": thinking,
        "busy": bool(result.get("busy")),
        "done": bool(result.get("done")),
        "count": int(result.get("count") or 0),
        "content_html": result.get("content_html"),
        "thinking_html": result.get("thinking_html"),
    }


def capture_response(
    page: Page,
    extract_js: str,
    retries: int = 2,
    base_delay: float = 0.15,
    convert: Callable[[dict], dict] | None = None,
) -> dict:
    """Capture the current assistant response from a page.

    Retries transient Playwright errors (page mid-navigation, connection
    retry, detached frame) with a short exponential backoff. A single blip
    used to abort the whole request; retrying inside the poll keeps the stream
    alive across momentary hiccups instead of surfacing a 502 to the client.

    Additionally retries ``BrowserWorkerStopped``: when the worker thread dies
    mid-dispatch, a fresh worker is started transparently on the next call
    (the old queue item is drained), and since this operation is read-only
    (no prompt submission) re-running it is safe and idempotent. This turns
    a transient worker death into a recoverable blip. ``BrowserOperationTimeout``
    (a wedged, still-alive worker) is never retried.

    Args:
        page: The Playwright page to query.
        extract_js: The extraction script returned by ``extract_js``.
        retries: Number of additional attempts after the first failure.
        base_delay: Seconds between attempts (doubles each retry).
        convert: Optional callable that transforms the normalized result
            dict (e.g. HTML-to-markdown conversion). Applied once, after
            normalization, to every capture including retries.

    Returns:
        Normalized extraction result dict.
    """
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            result = _normalize_response(run_js_value(page, extract_js))
            if convert is not None:
                result = convert(result)
            return result
        except BrowserOperationTimeout:
            # A wedged worker thread will not recover on retry; re-raise
            # immediately so the caller can end the request.
            raise
        except BrowserWorkerStopped as e:
            # The worker died mid-dispatch. A new worker is started
            # transparently on the next call, and capture_response is
            # idempotent (read-only), so retrying is safe -- the old item
            # was drained on restart and re-running the read produces the
            # same result. This turns a transient worker death into a
            # recoverable blip instead of a hard 502.
            last_error = e
            if attempt >= retries:
                raise
            delay = min(base_delay * (2**attempt), 1.0)
            logger.debug(
                "capture_response worker stopped on attempt %d/%d: %s; "
                "retrying in %.2fs",
                attempt + 1,
                retries + 1,
                e,
                delay,
            )
            time.sleep(delay)
        except PlaywrightError as e:
            last_error = e
            if not is_transient_error(e) or attempt >= retries:
                raise
            delay = min(base_delay * (2**attempt), 1.0)
            logger.debug(
                "capture_response transient error on attempt %d/%d: %s; retrying in %.2fs",
                attempt + 1,
                retries + 1,
                e,
                delay,
            )
            time.sleep(delay)
    # Unreachable: the loop either returns or raises. Kept for clarity.
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error
    raise BrowserError("capture_response failed after retries")  # pragma: no cover


def is_new_response(response: dict, baseline: dict | None) -> bool:
    """True once this poll reflects a new assistant turn, not the old one.

    Single source of truth for both the streaming and non-streaming paths.

    Reasoning counts: while a web chat is still thinking the answer text is
    empty, so keying on content alone discarded the whole trace and never
    recognised a thinking-only turn as a new response.
    """
    if baseline is None:
        return bool(response.get("found")) or bool(response.get("thinking"))
    if response.get("count", 0) > baseline.get("count", 0):
        return True
    if response.get("content", "") != baseline.get("content", ""):
        return True
    thinking = response.get("thinking") or ""
    return bool(thinking) and thinking != (baseline.get("thinking") or "")


class ResponsePoller:
    """Completion state machine shared by both polling paths.

    ``wait_for_response`` (non-streaming) and ``_stream_web_chat``
    (streaming) used to keep private copies of the termination
    rules, and the copies drifted: the non-streaming one had no
    busy-patience guard, missed the "not busy" half of the
    done-confirmation rule, and reset its stability timer on any
    dict change (a flickering ``busy`` flag or a ticking ``count``
    restarted the clock), while the streaming one treated an
    ``idle_timeout`` of 0 as "finished immediately". Every
    termination rule lives here now, so both paths agree by
    construction -- change a rule in this class only.

    ``observe`` records one captured payload and returns a stop
    reason (``site_done``, ``idle``, ``busy_timeout``,
    ``thinking_timeout``, ``no_output``) or ``None`` to keep
    polling. ``is_new`` and ``changed`` are exposed for callers
    that react to payload movement (the streaming loop emits SSE
    deltas from them).
    """

    def __init__(
        self,
        *,
        baseline: dict | None,
        idle_timeout: float,
        done_confirm: float,
        busy_patience: float,
        thinking_patience: float,
        first_token_timeout: float,
        started_at: float,
        logger: logging.Logger | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.baseline = baseline
        self.idle_timeout = float(idle_timeout)
        self.done_confirm = float(done_confirm)
        self.busy_patience = float(busy_patience)
        self.thinking_patience = float(thinking_patience)
        self.first_token_timeout = float(first_token_timeout)
        self.started_at = float(started_at)
        self._log = logger or _state_logger
        self._context = dict(context or {})
        self.saw_busy = False
        self.got_output = False
        self.is_new = False
        self.changed = False
        self.last_change_at: float | None = None
        self.last_content = ""
        self.last_thinking = ""

    def observe(self, current: dict, now: float) -> str | None:
        """Record one captured payload; return a stop reason or None."""
        busy = bool(current.get("busy"))
        content = current.get("content") or ""
        thinking = current.get("thinking") or ""
        if busy:
            self.saw_busy = True
        self.is_new = is_new_response(current, self.baseline)
        self.changed = False
        if not self.is_new:
            # Nothing new on screen yet (same as the pre-submit
            # baseline). Never read the old answer as this turn's.
            return self._no_output(now)

        if content or thinking:
            self.got_output = True
        # Track only real payload movement: a flickering `busy` flag
        # or a ticking `count` must not restart the stability clock,
        # or the patience guards could never fire.
        if content != self.last_content or thinking != self.last_thinking:
            self.changed = True
            self.last_change_at = now
            self.last_content = content
            self.last_thinking = thinking
            self._log.debug(
                "response_state: payload_changed",
                extra={
                    **self._context,
                    "content_len": len(content),
                    "thinking_len": len(thinking),
                    "busy": busy,
                    "done": current.get("done"),
                    "count": current.get("count"),
                },
            )
        stable_for = (
            now - self.last_change_at if self.last_change_at is not None else 0.0
        )
        # A site seen generating is trusted once it stops *and* the
        # text has been stable for `done_confirm`. The `not busy`
        # half matters: a site that still claims to be generating
        # has not finished, whatever its `done` flag says.
        if (
            self.saw_busy
            and not busy
            and current.get("done")
            and stable_for >= self.done_confirm
        ):
            self._log.debug(
                "response_state: done_after_stable",
                extra={
                    **self._context,
                    "stable_for": stable_for,
                    "count": current.get("count"),
                },
            )
            return "site_done"
        # Sites rarely clear their `done` flag, so a stable payload
        # ends the wait even without a generation signal. An
        # idle_timeout of 0 disables the rule rather than finishing
        # on the first poll.
        if (
            self.idle_timeout > 0
            and not busy
            and (content or thinking)
            and stable_for >= self.idle_timeout
        ):
            self._log.debug(
                "response_state: idle_complete",
                extra={
                    **self._context,
                    "stable_for": stable_for,
                    "count": current.get("count"),
                },
            )
            return "idle"
        # A stall while the site still reports generating. A
        # thinking-only stall (content empty, reasoning present) gets
        # far more patience than an answer stall: reasoning traces
        # pause naturally between chunks and thinking models think
        # for minutes.
        if busy and (content or thinking):
            thinking_only = not content and bool(thinking)
            patience = self.thinking_patience if thinking_only else self.busy_patience
            if stable_for >= patience:
                reason = "thinking_timeout" if thinking_only else "busy_timeout"
                self._log.debug(
                    f"response_state: {reason}",
                    extra={
                        **self._context,
                        "stable_for": stable_for,
                        "patience": patience,
                        "thinking_only": thinking_only,
                    },
                )
                return reason
        return self._no_output(now)

    def _no_output(self, now: float) -> str | None:
        if not self.got_output and now - self.started_at >= self.first_token_timeout:
            self._log.debug(
                "response_state: no_output",
                extra={
                    **self._context,
                    "elapsed": now - self.started_at,
                },
            )
            return "no_output"
        return None


def _with_last_thinking(current: dict, seen_thinking: str | None) -> dict:
    """Re-attach the last reasoning trace a site dropped on completion.

    Some chats (Tencent AI Studio) collapse their reasoning disclosure
    when the answer completes, so the final capture reports no thinking
    even though a full trace streamed a moment earlier. Keep the last
    non-empty trace seen while the turn was live so the non-streaming
    caller still receives it -- the streaming loop already delivered the
    deltas as they arrived, so this only affects the one-shot path.
    """
    if (
        seen_thinking
        and current.get("thinking") is None
        and not current.get("thinking_timeout")
        and not current.get("no_output")
        and not current.get("busy_timeout")
        and not current.get("timed_out")
    ):
        current = dict(current)
        current["thinking"] = seen_thinking
    return current


def wait_for_response(
    page: Page,
    extract_js: str,
    timeout: float,
    poll_interval: float = 0.25,
    idle_timeout: float | None = None,
    baseline: dict | None = None,
    done_confirm: float = 0.75,
    thinking_patience: float = 120.0,
    busy_patience: float = DEFAULT_BUSY_PATIENCE,
    first_token_timeout: float = DEFAULT_FIRST_TOKEN_TIMEOUT,
    convert: Callable[[dict], dict] | None = None,
) -> dict:
    """Poll a page until a new assistant response is complete or timeout.

    Shares its completion rules with the streaming loop via
    ResponsePoller, so both paths terminate identically: a site seen
    generating is trusted once it stops *and* the text has been
    stable for `done_confirm`; a site that never signals is trusted
    after the text has been unchanged for `idle_timeout`; a stall
    while the page still reports generation ends after
    `busy_patience` (answer) or `thinking_patience` (reasoning-only
    trace); a prompt that produces no output at all ends after
    `first_token_timeout`.

    An unchanged payload is never read as finished while the page
    still reports generation, because web chats pause mid-answer
    (thinking, re-render, rate limiting) and stopping there
    truncates the reply.

    Args:
        convert: Optional callable that transforms the normalized
            capture result (e.g. HTML-to-markdown conversion). Passed
            through to ``capture_response`` on every poll so the
            streamed deltas and the final answer use the same
            transformed text.
    """
    started_at = time.monotonic()
    deadline = started_at + max(float(timeout), 0.0)
    poller = ResponsePoller(
        baseline=baseline,
        idle_timeout=float(idle_timeout or 0.0),
        done_confirm=float(done_confirm),
        busy_patience=float(busy_patience),
        thinking_patience=float(thinking_patience),
        first_token_timeout=float(first_token_timeout),
        started_at=started_at,
    )
    last = {
        "found": False,
        "content": "",
        "thinking": None,
        "busy": False,
        "done": False,
        "count": 0,
    }
    seen_thinking = None
    while time.monotonic() <= deadline:
        current = capture_response(page, extract_js, convert=convert)
        stop = poller.observe(current, time.monotonic())
        if poller.is_new and current.get("thinking"):
            seen_thinking = current["thinking"]
        if stop == "site_done":
            return _with_last_thinking(current, seen_thinking)
        if stop == "idle":
            current["done"] = True
            return _with_last_thinking(current, seen_thinking)
        if stop == "thinking_timeout":
            current["thinking_timeout"] = True
            return _with_last_thinking(current, seen_thinking)
        if stop == "busy_timeout":
            current["busy_timeout"] = True
            return _with_last_thinking(current, seen_thinking)
        if stop == "no_output":
            current["no_output"] = True
            return _with_last_thinking(current, seen_thinking)
        last = current
        time.sleep(max(float(poll_interval), 0.01))
    _state_logger.debug(
        "response_state: budget_exhausted",
        extra={"timed_out": True, "count": last.get("count")},
    )
    return _with_last_thinking({**last, "timed_out": True}, seen_thinking)


def _do_inject_and_submit(
    page: Page, inject_js: str, submit_js: str, tab_index: int
) -> dict:
    """Execute injection and submit on a page. Returns status dict.

    Runs on the Playwright browser thread.
    """
    status = {"tab": tab_index, "inject": None, "submit": None}

    try:
        # Bring page to front
        page.bring_to_front()
        time.sleep(0.3)

        # Check if page is still valid (not closed)
        if page.is_closed():
            logger.warning(f"Page {tab_index} is closed")
            status["inject"] = "BROWSER_ERROR: page closed"
            return status

        # Inject prompt
        inject_result = run_js(page, inject_js)
        status["inject"] = inject_result

        if inject_result != "OK":
            logger.warning(f"Inject failed on tab {tab_index}: {inject_result}")
            return status

        time.sleep(0.3)

        # Check if page is still valid before submit
        if page.is_closed():
            logger.warning(f"Page {tab_index} closed during operation")
            status["submit"] = "BROWSER_ERROR: page closed"
            return status

        # Submit
        submit_result = run_js(page, submit_js)
        status["submit"] = submit_result

        if submit_result not in ("OK", "ENTER_SENT", "ENTER_SENT_UNVERIFIED"):
            logger.warning(f"Submit failed on tab {tab_index}: {submit_result}")

    except PlaywrightError as e:
        logger.error(f"Browser error on tab {tab_index}: {e}")
        status["inject"] = f"BROWSER_ERROR: {e}"

    return status


@retry_with_backoff(max_retries=MAX_RETRIES)
def inject_and_submit(
    page: Page, inject_js: str, submit_js: str, tab_index: int
) -> dict:
    """Execute injection and submit on a page. Returns status dict.

    Retries on transient Playwright errors with exponential backoff.
    Dispatches to the Playwright browser thread via queue.
    """
    return run_in_browser_thread(
        _do_inject_and_submit, page, inject_js, submit_js, tab_index
    )


def check_page_health(page: Page) -> bool:
    """Check if a page is still responsive and not closed."""

    def _do_check() -> bool:
        if page.is_closed():
            return False
        page.evaluate("1 + 1")
        return True

    try:
        return run_in_browser_thread(_do_check)
    except (PlaywrightError, RuntimeError, BrowserError, BrowserOperationTimeout):
        return False


def get_page_snapshot(page: Page) -> dict:
    """Capture a diagnostic snapshot of the page for error logging.

    Returns a dict with url, title, and visible text preview.
    Safe to call even on closed pages.
    """
    snapshot = {"url": None, "title": None, "text_preview": None}
    if page.is_closed():
        snapshot["error"] = "page closed"
        return snapshot

    def _do_snapshot() -> dict:
        try:
            url = page.url
            title = page.title()
            text = page.evaluate(
                "() => document.body ? document.body.innerText.slice(0, 500) : ''"
            )
            return {"url": url, "title": title, "text_preview": text}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    try:
        result = run_in_browser_thread(_do_snapshot)
        snapshot.update(result)
    except Exception as e:  # noqa: BLE001
        snapshot["error"] = f"snapshot failed: {e}"
    return snapshot


def recover_page(url: str, chrome_bin: str | None = None) -> Page | None:
    """Attempt to recover a closed/stale page by opening a new one."""
    try:
        logger.info(f"Attempting to recover page for {url}")
        return open_page(url, chrome_bin)
    except BrowserError as e:
        logger.error(f"Failed to recover page for {url}: {e}")
        return None


def is_running() -> bool:
    """Check if the browser is still connected."""
    if _context is None:
        return False

    def _check() -> bool:
        try:
            browser = _context.browser
            return browser is not None and browser.is_connected()
        except PlaywrightError:
            return False

    if _is_browser_thread():
        return _check()

    try:
        return run_in_browser_thread(_check)
    except Exception:  # noqa: BLE001
        return False
