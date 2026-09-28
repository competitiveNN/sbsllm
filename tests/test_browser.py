"""Tests for browser.py."""

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import sbsllm.browser as browser_module
from sbsllm.browser import (
    USER_DATA_DIR,
    BrowserError,
    check_page_health,
    close_browser,
    ensure_browser,
    inject_and_submit,
    is_running,
    open_page,
    run_in_browser_thread,
    run_js,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_browser_globals():
    browser_module._stop_browser_worker()
    browser_module._context = None
    browser_module._playwright_instance = None
    yield
    browser_module._stop_browser_worker()
    browser_module._context = None
    browser_module._playwright_instance = None


class TestSetupLogging:
    def test_default_logging(self):
        with patch("sbsllm.browser.logging.basicConfig") as mock_config:
            setup_logging()
            mock_config.assert_called_once()
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["level"] == 20  # INFO

    def test_debug_logging(self):
        with patch("sbsllm.browser.logging.basicConfig") as mock_config:
            setup_logging(level="DEBUG")
            call_kwargs = mock_config.call_args[1]
            assert call_kwargs["level"] == 10  # DEBUG

    def test_with_log_file(self, tmp_path):
        log_file = tmp_path / "test.log"
        with patch("sbsllm.browser.logging.basicConfig") as mock_config:
            setup_logging(log_file=str(log_file))
            call_kwargs = mock_config.call_args[1]
            handlers = call_kwargs["handlers"]
            assert len(handlers) == 2  # StreamHandler + FileHandler


class TestEnsureBrowser:
    def _mock_context(self):
        """Build a persistent-context mock with a connected browser."""
        mock_browser = MagicMock()
        mock_browser.is_connected.return_value = True
        mock_context = MagicMock()
        mock_context.browser = mock_browser
        return mock_context, mock_browser

    def test_launches_browser(self):
        mock_context, _ = self._mock_context()
        with patch("sbsllm.browser.sync_playwright") as mock_pw:
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            context = ensure_browser()

            assert context == mock_context
            mock_instance.chromium.launch_persistent_context.assert_called_once()
            # user_data_dir is the first positional arg
            assert (
                mock_instance.chromium.launch_persistent_context.call_args[0][0]
                == USER_DATA_DIR
            )
            # profile is passed via user_data_dir, not a --user-data-dir arg
            assert "--user-data-dir=" not in str(
                mock_instance.chromium.launch_persistent_context.call_args
            )
            # JavaScript is enabled
            call_kwargs = mock_instance.chromium.launch_persistent_context.call_args[1]
            assert call_kwargs.get("java_script_enabled") is True

    def test_uses_channel_when_no_system_chromium(self):
        mock_context, _ = self._mock_context()
        with (
            patch("sbsllm.browser.sync_playwright") as mock_pw,
            patch("sbsllm.browser._find_system_chromium", return_value=False),
        ):
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            ensure_browser()

            call_kwargs = mock_instance.chromium.launch_persistent_context.call_args[1]
            assert "channel" in call_kwargs
            assert call_kwargs["channel"] == "chromium"
            assert "executable_path" not in call_kwargs

    def test_auto_detects_system_chromium(self):
        mock_context, _ = self._mock_context()
        with (
            patch("sbsllm.browser.sync_playwright") as mock_pw,
            patch(
                "sbsllm.browser._find_system_chromium", return_value="/usr/bin/chromium"
            ),
        ):
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            ensure_browser()

            call_kwargs = mock_instance.chromium.launch_persistent_context.call_args[1]
            assert call_kwargs["executable_path"] == "/usr/bin/chromium"
            assert "channel" not in call_kwargs

    def test_headless_env_var_controls_mode(self):
        """SBSLLM_HEADLESS env var should toggle headless mode."""
        mock_context, _ = self._mock_context()
        with (
            patch("sbsllm.browser.sync_playwright") as mock_pw,
            patch.dict(os.environ, {"SBSLLM_HEADLESS": "true"}),
            patch("sbsllm.browser._find_system_chromium", return_value=False),
        ):
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            ensure_browser()

            call_kwargs = mock_instance.chromium.launch_persistent_context.call_args[1]
            assert call_kwargs["headless"] is True

    def test_headless_defaults_to_false(self):
        """Without SBSLLM_HEADLESS set, headless must default to False."""
        mock_context, _ = self._mock_context()
        with (
            patch("sbsllm.browser.sync_playwright") as mock_pw,
            patch.dict(os.environ, {}, clear=False),
            patch("sbsllm.browser._find_system_chromium", return_value=False),
        ):
            # Ensure the env var is not set during this test.
            os.environ.pop("SBSLLM_HEADLESS", None)
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            ensure_browser()

            call_kwargs = mock_instance.chromium.launch_persistent_context.call_args[1]
            assert call_kwargs["headless"] is False

    def test_reuses_existing_browser(self):
        mock_context, _ = self._mock_context()
        with patch("sbsllm.browser.sync_playwright") as mock_pw:
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            context1 = ensure_browser()
            context2 = ensure_browser()

            assert context1 == context2
            mock_instance.chromium.launch_persistent_context.assert_called_once()

    def test_disconnected_browser_is_cleaned_before_relaunch(self):
        stale_context, stale_browser = self._mock_context()
        stale_browser.is_connected.return_value = False
        new_context, _ = self._mock_context()
        stale_instance = MagicMock()

        with (
            patch("sbsllm.browser._context", stale_context),
            patch("sbsllm.browser._playwright_instance", stale_instance),
            patch("sbsllm.browser.close_browser") as mock_close,
            patch("sbsllm.browser.sync_playwright") as mock_pw,
        ):
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = new_context

            context = ensure_browser()

        assert context == new_context
        mock_close.assert_called_once()

    def test_custom_chrome_bin(self):
        mock_context, _ = self._mock_context()
        with patch("sbsllm.browser.sync_playwright") as mock_pw:
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            ensure_browser(chrome_bin="/usr/bin/google-chrome")

            call_kwargs = mock_instance.chromium.launch_persistent_context.call_args[1]
            assert call_kwargs["executable_path"] == "/usr/bin/google-chrome"


class TestIsHeadless:
    """Unit tests for the is_headless() helper."""

    def test_truthy_values(self):
        """All common truthy values should return True."""
        for val in ("1", "true", "TRUE", "True", "yes", "YES", "on", "ON"):
            with patch.dict(os.environ, {"SBSLLM_HEADLESS": val}):
                assert browser_module.is_headless() is True, f"{val!r} should be truthy"

    def test_falsy_values(self):
        """All common falsy values should return False."""
        for val in ("0", "false", "FALSE", "no", "off", "", "anything"):
            with patch.dict(os.environ, {"SBSLLM_HEADLESS": val}):
                assert browser_module.is_headless() is False, f"{val!r} should be falsy"

    def test_missing_env_var(self):
        """When the env var is not set, is_headless() must return False."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SBSLLM_HEADLESS", None)
            assert browser_module.is_headless() is False

    def test_whitespace_is_stripped(self):
        """Leading/trailing whitespace should be stripped."""
        with patch.dict(os.environ, {"SBSLLM_HEADLESS": "  true  "}):
            assert browser_module.is_headless() is True


class TestOpenPage:
    def test_reuses_initial_page(self):
        with patch("sbsllm.browser.ensure_browser") as mock_ensure:
            mock_browser = MagicMock()
            mock_ensure.return_value = mock_browser
            blank_page = MagicMock()
            blank_page.url = "about:blank"
            mock_browser.pages = [blank_page]
            mock_page = MagicMock()
            mock_browser.new_page.return_value = mock_page

            page = open_page("https://example.com")

            assert page == blank_page
            blank_page.close.assert_not_called()
            mock_browser.new_page.assert_not_called()
            blank_page.goto.assert_called_once_with("https://example.com")

    def test_creates_page_when_no_initial_page_exists(self):
        with patch("sbsllm.browser.ensure_browser") as mock_ensure:
            mock_browser = MagicMock()
            mock_ensure.return_value = mock_browser
            mock_browser.pages = []
            mock_page = MagicMock()
            mock_browser.new_page.return_value = mock_page

            page = open_page("https://example.com")

            assert page == mock_page
            mock_browser.new_page.assert_called_once()
            mock_page.goto.assert_called_once_with("https://example.com")

    def test_closes_page_after_goto_failure(self):
        with patch("sbsllm.browser.ensure_browser") as mock_ensure:
            mock_browser = MagicMock()
            mock_ensure.return_value = mock_browser
            mock_browser.pages = []
            mock_page = MagicMock()
            mock_browser.new_page.return_value = mock_page
            from playwright.sync_api import Error as PlaywrightError

            mock_page.goto.side_effect = PlaywrightError("navigation failed")

            with pytest.raises(BrowserError, match="navigation failed"):
                open_page("https://example.com")

            mock_page.close.assert_called_once()

    def test_recreates_context_after_new_page_failure(self):
        from playwright.sync_api import Error as PlaywrightError

        first_browser = MagicMock()
        first_browser.new_page.side_effect = PlaywrightError(
            "Target.createTarget failed"
        )
        second_browser = MagicMock()
        second_browser.pages = []
        page = MagicMock()
        second_browser.new_page.return_value = page

        with (
            patch(
                "sbsllm.browser.ensure_browser",
                side_effect=[first_browser, second_browser],
            ) as mock_ensure,
            patch("sbsllm.browser.close_browser") as mock_close,
        ):
            result = open_page("https://example.com")

        assert result == page
        assert mock_ensure.call_count == 2
        mock_close.assert_called_once()
        page.goto.assert_called_once_with("https://example.com")

    def test_closes_context_after_new_page_retry_failure(self):
        from playwright.sync_api import Error as PlaywrightError

        first_browser = MagicMock()
        first_browser.new_page.side_effect = PlaywrightError("first failure")
        second_browser = MagicMock()
        second_browser.new_page.side_effect = PlaywrightError("second failure")

        with (
            patch(
                "sbsllm.browser.ensure_browser",
                side_effect=[first_browser, second_browser],
            ),
            patch("sbsllm.browser.close_browser") as mock_close,
            pytest.raises(BrowserError, match="second failure"),
        ):
            open_page("https://example.com")

        assert mock_close.call_count == 2


class TestIsRunning:
    def test_true_when_browser_connected(self):
        mock_browser = MagicMock()
        mock_browser.is_connected.return_value = True
        mock_context = MagicMock()
        mock_context.browser = mock_browser

        with patch("sbsllm.browser._context", mock_context):
            assert is_running() is True

    def test_false_when_no_browser(self):
        with patch("sbsllm.browser._context", None):
            assert is_running() is False

    def test_false_when_disconnected(self):
        mock_browser = MagicMock()
        mock_browser.is_connected.return_value = False
        mock_context = MagicMock()
        mock_context.browser = mock_browser

        with patch("sbsllm.browser._context", mock_context):
            assert is_running() is False


class TestRunJs:
    def test_executes_js(self):
        mock_page = MagicMock()
        mock_page.evaluate.return_value = "result"

        result = run_js(mock_page, "1+1")

        assert result == "result"
        mock_page.evaluate.assert_called_once_with("1+1")

    def test_returns_empty_string_for_none(self):
        mock_page = MagicMock()
        mock_page.evaluate.return_value = None

        result = run_js(mock_page, "null")

        assert result == ""

    def test_handles_error(self):
        from playwright.sync_api import Error as PlaywrightError

        mock_page = MagicMock()
        mock_page.evaluate.side_effect = PlaywrightError("eval failed")

        result = run_js(mock_page, "invalid()")

        assert "ERROR" in result


class TestInjectAndSubmit:
    def test_successful_flow(self):
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate.side_effect = ["OK", "OK"]

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["tab"] == 1
        assert result["inject"] == "OK"
        assert result["submit"] == "OK"
        mock_page.bring_to_front.assert_called_once()

    def test_inject_fails(self):
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate.return_value = "NO_INPUT"

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["inject"] == "NO_INPUT"
        assert result["submit"] is None

    def test_submit_fails(self):
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate.side_effect = ["OK", "NO_BUTTON"]

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["inject"] == "OK"
        assert result["submit"] == "NO_BUTTON"

    def test_browser_error(self):
        from playwright.sync_api import Error as PlaywrightError

        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate.side_effect = PlaywrightError("connection lost")

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert "BROWSER_ERROR" in result["inject"]

    def test_bring_to_front_error(self):
        from playwright.sync_api import Error as PlaywrightError

        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.bring_to_front.side_effect = PlaywrightError("page lost")

        result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["inject"] == "BROWSER_ERROR: page lost"
        assert result["submit"] is None


class TestCloseBrowser:
    def test_closes_browser(self):
        mock_context = MagicMock()
        mock_instance = MagicMock()

        with (
            patch("sbsllm.browser._context", mock_context),
            patch("sbsllm.browser._playwright_instance", mock_instance),
        ):
            close_browser()

        mock_context.close.assert_called_once()
        mock_instance.stop.assert_called_once()

    def test_handles_close_errors(self):
        from playwright.sync_api import Error as PlaywrightError

        mock_context = MagicMock()
        mock_context.close.side_effect = PlaywrightError("close failed")
        mock_instance = MagicMock()
        mock_instance.stop.side_effect = PlaywrightError("stop failed")

        with (
            patch("sbsllm.browser._context", mock_context),
            patch("sbsllm.browser._playwright_instance", mock_instance),
        ):
            close_browser()

        mock_context.close.assert_called_once()
        mock_instance.stop.assert_called_once()

    def test_handles_no_browser(self):
        with (
            patch("sbsllm.browser._context", None),
            patch("sbsllm.browser._playwright_instance", None),
        ):
            close_browser()  # Should not raise


class TestBrowserQueue:
    """Tests for the run_in_browser_thread queue mechanism."""

    def test_run_in_browser_thread_runs_function(self):
        result_holder: list[int] = []

        def append_val(x: int) -> int:
            result_holder.append(x)
            return x * 2

        result = run_in_browser_thread(append_val, 21)
        assert result == 42
        assert result_holder == [21]

    def test_run_in_browser_thread_passes_kwargs(self):
        result = run_in_browser_thread(lambda a, b=0: a + b, 10, b=5)
        assert result == 15

    def test_run_in_browser_thread_propagates_exception(self):
        def raise_error() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_in_browser_thread(raise_error)

    def test_run_in_browser_thread_returns_none(self):
        result = run_in_browser_thread(lambda: None)
        assert result is None

    def test_run_in_browser_thread_with_page_mock(self):
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate.return_value = True

        def _check(page: MagicMock) -> bool:
            if page.is_closed():
                return False
            page.evaluate("1 + 1")
            return True

        result = run_in_browser_thread(_check, mock_page)
        assert result is True
        mock_page.is_closed.assert_called_once()
        mock_page.evaluate.assert_called_once_with("1 + 1")

    def test_inject_and_submit_uses_queue(self):
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate.side_effect = ["OK", "OK"]

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["tab"] == 1
        assert result["inject"] == "OK"
        assert result["submit"] == "OK"

    def test_check_page_health_returns_true(self):
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        mock_page.evaluate.return_value = True
        assert check_page_health(mock_page) is True

    def test_check_page_health_returns_false_when_closed(self):
        mock_page = MagicMock()
        mock_page.is_closed.return_value = True
        assert check_page_health(mock_page) is False

    def test_is_running_true(self):
        mock_browser = MagicMock()
        mock_browser.is_connected.return_value = True
        mock_context = MagicMock()
        mock_context.browser = mock_browser
        with patch("sbsllm.browser._context", mock_context):
            assert is_running() is True

    def test_is_running_false_when_none(self):
        with patch("sbsllm.browser._context", None):
            assert is_running() is False

    def test_is_running_false_on_exception(self):
        mock_context = MagicMock()
        mock_context.browser = MagicMock()
        mock_context.browser.is_connected.side_effect = RuntimeError("greenlet")
        with patch("sbsllm.browser._context", mock_context):
            assert is_running() is False

    def test_run_in_browser_thread_runs_on_separate_thread(self):
        thread_ids: list[int | None] = []

        def capture_thread_id() -> int:
            thread_ids.append(threading.current_thread().ident)
            return thread_ids[-1]

        # Start a worker thread via run_in_browser_thread
        result = run_in_browser_thread(capture_thread_id)

        # At least one call should have run on a different thread
        # (the browser worker thread)
        assert len(thread_ids) == 1
        # The main thread ID should differ from the worker thread ID
        main_thread_id = threading.current_thread().ident
        assert result == thread_ids[0]
        assert result != main_thread_id

    def test_close_browser_from_browser_thread(self):
        """close_browser called from the browser thread should run directly."""
        mock_context = MagicMock()
        mock_instance = MagicMock()
        with (
            patch("sbsllm.browser._context", mock_context),
            patch("sbsllm.browser._playwright_instance", mock_instance),
        ):
            run_in_browser_thread(close_browser)

        mock_context.close.assert_called_once()
        mock_instance.stop.assert_called_once()

    def test_nested_browser_dispatch_runs_inline(self):
        def nested() -> int:
            return run_in_browser_thread(lambda: 7)

        assert run_in_browser_thread(nested) == 7

    def test_browser_operation_timeout(self):
        with (
            patch("sbsllm.browser.BROWSER_OPERATION_TIMEOUT", 0.01),
            pytest.raises(RuntimeError, match="timed out"),
        ):
            run_in_browser_thread(time.sleep, 0.2)

    def test_run_js_dispatches_to_browser_thread(self):
        mock_page = MagicMock()
        mock_page.evaluate.return_value = "result"

        assert run_js(mock_page, "return 'result'") == "result"
        mock_page.evaluate.assert_called_once_with("return 'result'")


class TestRunJsValue:
    def test_evaluate_takes_no_timeout_argument(self):
        """Guard against the earlier bug: page.evaluate() accepts no
        `timeout`, and a MagicMock hides that. Assert on the real API."""
        import inspect

        from playwright.sync_api import Page

        params = inspect.signature(Page.evaluate).parameters
        assert "timeout" not in params, "page.evaluate() has no timeout parameter"

        page = MagicMock()
        page.evaluate.return_value = {"found": True}
        with patch.object(browser_module, "_is_browser_thread", return_value=True):
            browser_module.run_js_value(page, "JS")
        # Positional only: expression, then optional arg.
        assert page.evaluate.call_args.args == ("JS",)
        assert not page.evaluate.call_args.kwargs

    def test_capture_timeout_is_below_operation_timeout(self):
        assert browser_module.CAPTURE_TIMEOUT < browser_module.BROWSER_OPERATION_TIMEOUT

    def test_capture_poll_is_bounded(self):
        """A hung page must fail the poll instead of blocking forever."""
        page = MagicMock()
        with (
            patch.object(browser_module, "_is_browser_thread", return_value=False),
            patch.object(
                browser_module, "run_in_browser_thread", side_effect=RuntimeError("t/o")
            ) as dispatch,
            pytest.raises(RuntimeError),
        ):
            browser_module.run_js_value(page, "JS")
        assert dispatch.call_args.kwargs["operation_timeout"] == (
            browser_module.CAPTURE_TIMEOUT
        )

    def test_capture_timeout_is_overridable(self):
        page = MagicMock()
        with (
            patch.object(browser_module, "_is_browser_thread", return_value=False),
            patch.object(
                browser_module, "run_in_browser_thread", side_effect=RuntimeError("t/o")
            ) as dispatch,
            pytest.raises(RuntimeError),
        ):
            browser_module.run_js_value(page, "JS", timeout=2.0)
        assert dispatch.call_args.kwargs["operation_timeout"] == 2.0

    def test_evaluate_error_becomes_browser_error(self):
        page = MagicMock()
        page.evaluate.side_effect = browser_module.PlaywrightError("boom")
        with (
            patch.object(browser_module, "_is_browser_thread", return_value=True),
            pytest.raises(browser_module.BrowserError),
        ):
            browser_module.run_js_value(page, "JS")


class _FakeClock:
    """Virtual clock: only advances when the code under test sleeps."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(seconds, 0.01)


class TestWaitForResponse:
    def _run(self, polls, **kwargs):
        """Run wait_for_response on a fake clock; returns (result, elapsed)."""
        clock = _FakeClock()
        with (
            patch.object(
                browser_module, "capture_response", side_effect=self._polls(polls)
            ),
            patch.object(browser_module.time, "sleep", side_effect=clock.sleep),
            patch.object(browser_module.time, "monotonic", side_effect=clock.monotonic),
        ):
            start = clock.t
            result = browser_module.wait_for_response(MagicMock(), "JS", **kwargs)
            return result, clock.t - start

    def _polls(self, sequence):
        it = iter(sequence)

        def capture(page, js):
            try:
                return dict(next(it))
            except StopIteration:
                return dict(sequence[-1])

        return capture

    def _result(
        self, content="", *, thinking=None, done=False, busy=False, found=True, count=1
    ):
        return {
            "found": found,
            "content": content,
            "thinking": thinking,
            "busy": busy,
            "done": done,
            "count": count,
        }

    def test_returns_immediately_when_done(self):
        """A site seen generating, then finished, must not wait the idle window."""
        polls = [
            self._result("hi", busy=True, done=False),
            self._result("hi", busy=False, done=True),
        ]
        result, elapsed = self._run(polls, timeout=5, idle_timeout=30, done_confirm=0.5)
        assert result["content"] == "hi"
        assert result["done"] is True
        assert "timed_out" not in result
        assert elapsed < 2.0

    def test_returns_on_idle_window(self):
        """Sites rarely clear their `done` flag; a stable payload must end it."""
        polls = [self._result("stable answer")] * 3
        result, elapsed = self._run(polls, timeout=5, idle_timeout=0.3)
        assert result["content"] == "stable answer"
        assert result["done"] is True
        assert "timed_out" not in result
        assert elapsed < 2.0

    def test_thinking_only_turn_completes(self):
        """Regression: the non-streaming path had a divergent copy of
        _is_new_response without the thinking check, so a thinking-only turn
        was never recognised and the request ran to timeout."""
        polls = [self._result("", thinking="reasoning")] * 3
        result, elapsed = self._run(
            polls, timeout=5, idle_timeout=0.3, baseline=self._result("")
        )
        assert result["thinking"] == "reasoning"
        assert result["done"] is True
        assert "timed_out" not in result
        assert elapsed < 2.0

    def test_growing_content_resets_idle(self):
        calls = {"n": 0}

        def capture(page, js):
            calls["n"] += 1
            return self._result("x" * calls["n"])

        with (
            patch.object(browser_module, "capture_response", side_effect=capture),
            patch.object(browser_module.time, "sleep"),
            patch.object(
                browser_module.time, "monotonic", side_effect=iter(range(400)).__next__
            ),
        ):
            result = browser_module.wait_for_response(
                MagicMock(), "JS", timeout=3, idle_timeout=100
            )
        assert result["timed_out"] is True
        assert result["done"] is False

    def test_ignores_previous_answer_via_baseline(self):
        """The answer on screen before we sent anything must not be returned
        as this turn's answer."""
        polls = [self._result("old answer", count=1)]
        result, _ = self._run(
            polls, timeout=1, baseline=self._result("old answer", count=1)
        )
        assert result["timed_out"] is True
        assert result["done"] is False


class TestIsNewResponseShared:
    def test_server_and_browser_agree(self):
        """The server used to keep a divergent copy that lacked the thinking
        check, so the non-streaming path never saw a thinking-only turn."""
        from sbsllm.browser import is_new_response
        from sbsllm.server import OpenAIHandler

        handler = OpenAIHandler.__new__(OpenAIHandler)
        base = {
            "found": True,
            "content": "",
            "thinking": None,
            "busy": False,
            "done": False,
            "count": 1,
        }
        cases = [
            ({**base, "thinking": "hmm"}, base),
            ({**base, "content": "x"}, base),
            ({**base, "count": 2}, base),
            (base, base),
            ({**base, "thinking": "hmm"}, None),
            (base, None),
        ]
        for response, baseline in cases:
            assert handler._is_new_response(response, baseline) == is_new_response(
                response, baseline
            ), (response, baseline)

    def test_thinking_only_turn_is_new(self):
        from sbsllm.browser import is_new_response

        base = {
            "found": True,
            "content": "",
            "thinking": None,
            "busy": False,
            "done": False,
            "count": 1,
        }
        assert is_new_response({**base, "thinking": "reasoning"}, base) is True


class TestNormalizeResponse:
    def test_missing_result_is_not_found(self):
        out = browser_module._normalize_response(None)
        assert out["found"] is False
        assert out["busy"] is False

    def test_string_result_is_coerced(self):
        out = browser_module._normalize_response({"content": 42, "count": "3"})
        assert out["content"] == "42"
        assert out["count"] == 3

    def test_blank_thinking_becomes_none(self):
        out = browser_module._normalize_response({"content": "x", "thinking": "   "})
        assert out["thinking"] is None

    def test_thinking_is_stripped(self):
        out = browser_module._normalize_response({"content": "x", "thinking": " hmm "})
        assert out["thinking"] == "hmm"

    def test_busy_is_carried_through(self):
        assert (
            browser_module._normalize_response({"content": "x", "busy": True})["busy"]
            is True
        )
        assert browser_module._normalize_response({"content": "x"})["busy"] is False


class TestBrowserOperationTimeout:
    def test_is_a_runtime_error_subclass(self):
        """Callers that already guarded RuntimeError must keep working."""
        assert issubclass(browser_module.BrowserOperationTimeout, RuntimeError)
        assert not issubclass(
            browser_module.BrowserOperationTimeout, browser_module.BrowserError
        )

    def test_timeout_raises_the_dedicated_type(self):
        # Keep the sleep short: a timed-out call leaves the worker blocked in
        # the function, so later dispatches queue behind it.
        with pytest.raises(browser_module.BrowserOperationTimeout):
            browser_module.run_in_browser_thread(time.sleep, 0.4, operation_timeout=0.1)

    def test_per_call_timeout_overrides_default(self):
        assert browser_module.CAPTURE_TIMEOUT < browser_module.BROWSER_OPERATION_TIMEOUT
        with pytest.raises(browser_module.BrowserOperationTimeout) as exc:
            browser_module.run_in_browser_thread(time.sleep, 0.4, operation_timeout=0.1)
        assert "0.1s" in str(exc.value)

    def test_positional_args_still_reach_the_function(self):
        assert browser_module.run_in_browser_thread(lambda a, b: a + b, 1, 2) == 3

    def test_wrapped_function_can_still_take_a_timeout_kwarg(self):
        """`operation_timeout` must not shadow a wrapped function's own
        `timeout` argument."""
        seen = {}

        def takes_timeout(value, timeout=None):
            seen["timeout"] = timeout
            return value

        result = browser_module.run_in_browser_thread(
            takes_timeout, 7, operation_timeout=5.0, timeout=0.25
        )
        assert result == 7
        assert seen["timeout"] == 0.25


class TestCaptureResponseRetry:
    """Tests for the retry/backoff logic added to capture_response."""

    def test_succeeds_after_transient_errors(self):
        """Transient Playwright errors should be retried, then succeed."""
        from playwright.sync_api import Error as PwError

        fake_page = MagicMock()
        extraction = "return {found:true,content:'final answer',thinking:null,busy:false,done:true,count:1};"
        call_counts = {"n": 0}

        def flaky_evaluate(page, js):
            call_counts["n"] += 1
            if call_counts["n"] <= 2:
                raise PwError("Target closed: transient error")
            return {
                "found": True,
                "content": "final answer",
                "thinking": None,
                "busy": False,
                "done": True,
                "count": 1,
            }

        with patch("sbsllm.browser.run_js_value", side_effect=flaky_evaluate):
            result = browser_module.capture_response(
                fake_page, extraction, retries=3, base_delay=0.01
            )
            assert result["content"] == "final answer"
            assert call_counts["n"] == 3

    def test_retries_on_connection_error(self):
        """Connection errors are transient and should be retried."""
        from playwright.sync_api import Error as PwError

        fake_page = MagicMock()
        extraction = "return {found:true};"
        call_counts = {"n": 0}

        def flaky_evaluate(page, js):
            call_counts["n"] += 1
            if call_counts["n"] == 1:
                raise PwError("Connection closed")
            return {
                "found": True,
                "content": "ok",
                "thinking": None,
                "busy": False,
                "done": True,
                "count": 1,
            }

        with patch("sbsllm.browser.run_js_value", side_effect=flaky_evaluate):
            result = browser_module.capture_response(
                fake_page, extraction, retries=2, base_delay=0.01
            )
            assert result["content"] == "ok"
            assert call_counts["n"] == 2

    def test_browser_operation_timeout_not_retried(self):
        """A wedged browser worker (BrowserOperationTimeout) must not be retried."""
        fake_page = MagicMock()
        extraction = "return {found:true};"

        with (
            patch(
                "sbsllm.browser.run_js_value",
                side_effect=browser_module.BrowserOperationTimeout("wedged"),
            ),
            pytest.raises(browser_module.BrowserOperationTimeout),
        ):
            browser_module.capture_response(fake_page, extraction, retries=5)

    def test_non_transient_error_surfaces_immediately(self):
        """A non-transient Playwright error must not be retried."""
        from playwright.sync_api import Error as PwError

        fake_page = MagicMock()
        extraction = "return {found:true};"
        call_counts = {"n": 0}

        def non_transient(page, js):
            call_counts["n"] += 1
            raise PwError("SyntaxError in page")

        with patch("sbsllm.browser.run_js_value", side_effect=non_transient):
            with pytest.raises(PwError):
                browser_module.capture_response(
                    fake_page, extraction, retries=3, base_delay=0.01
                )
            assert call_counts["n"] == 1, "non-transient error should not be retried"

    def test_exhausts_retries_and_raises(self):
        """After exhausting retries, the last error is re-raised."""
        from playwright.sync_api import Error as PwError

        fake_page = MagicMock()
        extraction = "return {found:true};"

        with (
            patch(
                "sbsllm.browser.run_js_value",
                side_effect=PwError("connection timeout"),
            ),
            pytest.raises(PwError),
        ):
            browser_module.capture_response(
                fake_page, extraction, retries=2, base_delay=0.01
            )
