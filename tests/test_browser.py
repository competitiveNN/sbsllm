"""Tests for browser.py."""

from unittest.mock import MagicMock, patch

import pytest

import sbsllm.browser as browser_module
from sbsllm.browser import (
    USER_DATA_DIR,
    BrowserError,
    close_browser,
    ensure_browser,
    inject_and_submit,
    is_running,
    open_page,
    run_js,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_browser_globals():
    browser_module._context = None
    browser_module._playwright_instance = None
    yield
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
            patch("sbsllm.browser._find_system_chromium", return_value=True),
        ):
            mock_instance = MagicMock()
            mock_pw.return_value.start.return_value = mock_instance
            mock_instance.chromium.launch_persistent_context.return_value = mock_context

            ensure_browser()

            call_kwargs = mock_instance.chromium.launch_persistent_context.call_args[1]
            assert call_kwargs["executable_path"] == "/usr/bin/chromium-browser"
            assert "channel" not in call_kwargs

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
        mock_page.evaluate.side_effect = ["OK", "OK"]

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["tab"] == 1
        assert result["inject"] == "OK"
        assert result["submit"] == "OK"
        mock_page.bring_to_front.assert_called_once()

    def test_inject_fails(self):
        mock_page = MagicMock()
        mock_page.evaluate.return_value = "NO_INPUT"

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["inject"] == "NO_INPUT"
        assert result["submit"] is None

    def test_submit_fails(self):
        mock_page = MagicMock()
        mock_page.evaluate.side_effect = ["OK", "NO_BUTTON"]

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert result["inject"] == "OK"
        assert result["submit"] == "NO_BUTTON"

    def test_browser_error(self):
        from playwright.sync_api import Error as PlaywrightError

        mock_page = MagicMock()
        mock_page.evaluate.side_effect = PlaywrightError("connection lost")

        with patch("sbsllm.browser.time.sleep"):
            result = inject_and_submit(mock_page, "inject_js", "submit_js", 1)

        assert "BROWSER_ERROR" in result["inject"]

    def test_bring_to_front_error(self):
        from playwright.sync_api import Error as PlaywrightError

        mock_page = MagicMock()
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
