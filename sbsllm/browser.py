"""Browser automation via Playwright + Chromium."""

from __future__ import annotations

import logging
import time
from typing import Any

from playwright.sync_api import BrowserContext, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError

logger = logging.getLogger(__name__)

# Isolated user data directory for Chromium profile
USER_DATA_DIR = "/tmp/sbsllm-chrome"

# Timeout for page operations (seconds)
PAGE_TIMEOUT = 30

# Global browser state
_context: BrowserContext | None = None
_playwright_instance: Any = None


class BrowserError(Exception):
    """Raised when a browser operation fails."""


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure logging for the application."""
    import logging as _logging

    handlers: list[_logging.Handler] = [_logging.StreamHandler()]
    if log_file:
        handlers.append(_logging.FileHandler(log_file))

    _logging.basicConfig(
        level=getattr(_logging, level.upper(), _logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def ensure_browser(chrome_bin: str | None = None) -> BrowserContext:
    """Ensure Chromium is running. Launch if not.

    Returns the persistent browser context, which is created with an isolated
    user-data directory so login sessions persist between runs.
    """
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
            "args": [
                "--no-first-run",
                "--no-default-browser-check",
            ],
        }
        if chrome_bin:
            launch_args["executable_path"] = chrome_bin

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


def open_page(url: str, chrome_bin: str | None = None) -> Page:
    """Open a URL in a new page/tab. Returns the page."""
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


def close_browser() -> None:
    """Close the browser and clean up."""
    global _context, _playwright_instance

    if _context is not None:
        try:
            _context.close()
        except PlaywrightError as e:
            logger.debug(f"Error closing browser: {e}")
        _context = None

    if _playwright_instance is not None:
        try:
            _playwright_instance.stop()
        except PlaywrightError as stop_error:
            logger.debug(f"Failed to stop Playwright after launch error: {stop_error}")
        _playwright_instance = None


def run_js(page: Page, js: str) -> str:
    """Execute JS in a page. Returns the result as string."""
    try:
        result = page.evaluate(js)
        return str(result) if result is not None else ""
    except PlaywrightError as e:
        logger.error(f"JS execution failed: {e}")
        return f"BROWSER_ERROR: {e}"


def inject_and_submit(
    page: Page, inject_js: str, submit_js: str, tab_index: int
) -> dict:
    """Execute injection and submit on a page. Returns status dict."""
    status = {"tab": tab_index, "inject": None, "submit": None}

    try:
        # Bring page to front
        page.bring_to_front()
        time.sleep(0.3)

        # Inject prompt
        inject_result = run_js(page, inject_js)
        status["inject"] = inject_result

        if inject_result != "OK":
            logger.warning(f"Inject failed on tab {tab_index}: {inject_result}")
            return status

        time.sleep(0.3)

        # Submit
        submit_result = run_js(page, submit_js)
        status["submit"] = submit_result

        if submit_result != "OK":
            logger.warning(f"Submit failed on tab {tab_index}: {submit_result}")

    except PlaywrightError as e:
        logger.error(f"Browser error on tab {tab_index}: {e}")
        status["inject"] = f"BROWSER_ERROR: {e}"

    return status


def is_running() -> bool:
    """Check if the browser is still connected."""
    if _context is None:
        return False
    try:
        browser = _context.browser
        return browser is not None and browser.is_connected()
    except PlaywrightError:
        return False
