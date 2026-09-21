"""Browser automation via Playwright + Chromium."""

from __future__ import annotations

import logging
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

# Isolated user data directory for Chromium profile
USER_DATA_DIR = "/tmp/sbsllm-chrome"

# Timeout for page operations (seconds)
PAGE_TIMEOUT = 30

# Retry configuration for transient failures
MAX_RETRIES = 3
BASE_RETRY_DELAY = 0.5  # seconds
MAX_RETRY_DELAY = 5.0  # seconds
RETRY_JITTER = 0.1  # seconds

# Timeout for browser queue operations (seconds)
BROWSER_OPERATION_TIMEOUT = 120

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


def _is_browser_thread() -> bool:
    return (
        _playwright_thread_id is not None
        and threading.current_thread().ident == _playwright_thread_id
    )


def _start_browser_worker() -> threading.Thread:
    """Start and return the singleton browser worker thread."""
    global _browser_thread, _playwright_thread_id
    with _worker_start_lock:
        if _browser_thread is None or not _browser_thread.is_alive():
            if _browser_thread is not None:
                _playwright_thread_id = None
            _browser_thread = threading.Thread(
                target=_browser_worker_loop,
                daemon=True,
                name="browser-worker",
            )
            _browser_thread.start()
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
    _browser_queue.put(None)
    worker.join(timeout)
    if worker.is_alive():
        logger.warning("Browser worker did not stop within %ss", timeout)
        return
    with _worker_start_lock:
        if _browser_thread is worker:
            _browser_thread = None


def run_in_browser_thread(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute a function on the Playwright browser worker thread.

    Playwright pages are not thread-safe; all Playwright API calls must run
    on the thread where sync_playwright was started. This function dispatches
    work to that thread via a queue, blocking the caller until completion.

    Args:
        func: The function to execute on the browser thread.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        The return value of func.

    Raises:
        RuntimeError: If the operation times out.
        Any exception raised by func.
    """
    if _is_browser_thread():
        return func(*args, **kwargs)

    worker = _start_browser_worker()
    event = threading.Event()
    result_holder: list[tuple[str, Any]] = []
    _browser_queue.put((func, args, kwargs, event, result_holder))
    deadline = time.monotonic() + BROWSER_OPERATION_TIMEOUT
    while not event.wait(timeout=0.1):
        if not worker.is_alive():
            raise RuntimeError("Browser worker stopped before completing the operation")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Browser operation timed out after {BROWSER_OPERATION_TIMEOUT}s"
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

            launch_args: dict[str, Any] = {
                "headless": False,
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
                USER_DATA_DIR, **launch_args
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


def _do_open_page(url: str, chrome_bin: str | None = None) -> Page:
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

    _close_blank_pages(browser, keep_page=page)
    return page


def open_page(url: str, chrome_bin: str | None = None) -> Page:
    """Open a URL in a new page/tab. Returns the page."""
    if _is_browser_thread():
        return _do_open_page(url, chrome_bin)
    return run_in_browser_thread(_do_open_page, url, chrome_bin)


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

    run_in_browser_thread(_do_close)
    _stop_browser_worker()


def _do_run_js(page: Page, js: str) -> str:
    try:
        result = page.evaluate(js)
        return str(result) if result is not None else ""
    except PlaywrightError as e:
        logger.error(f"JS execution failed: {e}")
        return f"BROWSER_ERROR: {e}"


def _do_run_js_value(page: Page, js: str) -> Any:
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


def run_js_value(page: Page, js: str) -> Any:
    """Execute JS and return its structured result on the browser thread."""
    if _is_browser_thread():
        return _do_run_js_value(page, js)
    return run_in_browser_thread(_do_run_js_value, page, js)


def _normalize_response(result: Any) -> dict:
    """Normalize a response extraction result returned by page.evaluate."""
    if not isinstance(result, dict):
        return {
            "found": False,
            "content": "",
            "thinking": None,
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
        "done": bool(result.get("done")),
        "count": int(result.get("count") or 0),
    }


def capture_response(page: Page, extract_js: str) -> dict:
    """Capture the current assistant response from a page."""
    return _normalize_response(run_js_value(page, extract_js))


def _is_new_response(response: dict, baseline: dict | None) -> bool:
    if baseline is None:
        return response["found"]
    if response["count"] > baseline.get("count", 0):
        return True
    return response["found"] and response["content"] != baseline.get("content", "")


def wait_for_response(
    page: Page,
    extract_js: str,
    timeout: float,
    poll_interval: float = 0.25,
    idle_timeout: float | None = None,
    baseline: dict | None = None,
) -> dict:
    """Poll a page until a new assistant response is complete or timeout."""
    deadline = time.monotonic() + max(float(timeout), 0.0)
    stable_since: float | None = None
    previous: dict | None = None
    last = {
        "found": False,
        "content": "",
        "thinking": None,
        "done": False,
        "count": 0,
    }
    while time.monotonic() <= deadline:
        last = capture_response(page, extract_js)
        if _is_new_response(last, baseline) and last["done"]:
            return last
        if last["found"] and last.get("content"):
            if previous is not None and last == previous:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif idle_timeout and time.monotonic() - stable_since >= idle_timeout:
                    return last
            else:
                stable_since = None
        previous = last
        time.sleep(max(float(poll_interval), 0.01))
    return {**last, "timed_out": True}


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
    except Exception:  # noqa: BLE001
        return False


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
