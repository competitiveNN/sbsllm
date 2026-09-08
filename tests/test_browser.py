"""Tests for browser.py."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sbsllm.browser import (
    BASEDIR,
    BrowserError,
    ensure_qutebrowser,
    focus_tab,
    inject_and_submit,
    is_running,
    open_tab,
    open_tabs,
    run_command,
    run_js,
    setup_logging,
    with_retry,
)


class TestRunCommand:
    def test_returns_completed_process(self):
        with patch("sbsllm.browser.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = run_command("qutebrowser", ":jseval", "1")
            assert result.returncode == 0

    def test_passes_basedir(self):
        with patch("sbsllm.browser.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_command("qutebrowser", ":jseval", "1")
            call_args = mock_run.call_args[0][0]
            assert "--basedir" in call_args
            assert str(BASEDIR) in call_args

    def test_passes_correct_arguments(self):
        with patch("sbsllm.browser.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            run_command("qutebrowser", ":open", "-t", "https://example.com")
            call_args = mock_run.call_args[0][0]
            assert ":open" in call_args
            assert "-t" in call_args
            assert "https://example.com" in call_args


class TestIsRunning:
    def test_returns_true_when_running(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0)
            assert is_running("qutebrowser") is True

    def test_returns_false_when_not_running(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=1)
            assert is_running("qutebrowser") is False

    def test_returns_false_on_timeout(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=2)
            assert is_running("qutebrowser") is False

    def test_returns_false_on_file_not_found(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.side_effect = FileNotFoundError()
            assert is_running("qutebrowser") is False


class TestEnsureQutebrowser:
    def test_already_running(self, capsys):
        with patch("sbsllm.browser.is_running", return_value=True):
            with patch("sbsllm.browser.logger") as mock_logger:
                ensure_qutebrowser("qutebrowser")
                mock_logger.info.assert_called_with("qutebrowser is already running.")

    def test_launches_when_not_running(self):
        with patch("sbsllm.browser.is_running") as mock_running:
            # First call returns False, then True after launch
            mock_running.side_effect = [False, True, True]
            with patch("sbsllm.browser.subprocess.Popen") as mock_popen:
                ensure_qutebrowser("qutebrowser")
                mock_popen.assert_called_once()

    def test_raises_on_timeout(self):
        with patch("sbsllm.browser.is_running", return_value=False):
            with patch("sbsllm.browser.time.sleep"):  # Don't actually sleep
                with patch("sbsllm.browser.subprocess.Popen"):
                    with pytest.raises(BrowserError, match="did not start"):
                        ensure_qutebrowser("qutebrowser")


class TestOpenTab:
    def test_returns_true_on_success(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0, stderr="")
            assert open_tab("qutebrowser", "https://example.com") is True

    def test_returns_false_on_failure(self, capsys):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=1, stderr="error")
            assert open_tab("qutebrowser", "https://example.com") is False

    def test_sends_correct_command(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0, stderr="")
            open_tab("qutebrowser", "https://chatgpt.com/")
            call_args = mock_cmd.call_args[0]
            assert call_args[1] == ":open"
            assert call_args[2] == "-t"
            assert call_args[3] == "https://chatgpt.com/"


class TestOpenTabs:
    def test_opens_multiple_tabs(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0, stderr="")
            with patch("sbsllm.browser.time.sleep"):
                results = open_tabs("qutebrowser", ["https://a.com", "https://b.com"])
                assert len(results) == 2
                assert all(results)

    def test_returns_mixed_results(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.side_effect = [
                MagicMock(returncode=0, stderr=""),
                MagicMock(returncode=1, stderr="fail"),
            ]
            with patch("sbsllm.browser.time.sleep"):
                results = open_tabs("qutebrowser", ["https://a.com", "https://b.com"])
                assert results == [True, False]


class TestFocusTab:
    def test_returns_true_on_success(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0)
            assert focus_tab("qutebrowser", 1) is True

    def test_returns_false_on_failure(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=1)
            assert focus_tab("qutebrowser", 2) is False

    def test_sends_correct_index(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0)
            focus_tab("qutebrowser", 3)
            call_args = mock_cmd.call_args[0]
            assert call_args[1] == ":tab-focus"
            assert call_args[2] == "3"


class TestRunJs:
    def test_returns_stdout(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0, stdout="  result  \n")
            result = run_js("qutebrowser", "1+1")
            assert result == "result"

    def test_quiet_by_default(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0, stdout="")
            run_js("qutebrowser", "1+1")
            call_args = mock_cmd.call_args[0]
            assert "--quiet" in call_args

    def test_no_quiet_when_false(self):
        with patch("sbsllm.browser.run_command") as mock_cmd:
            mock_cmd.return_value = MagicMock(returncode=0, stdout="")
            run_js("qutebrowser", "1+1", quiet=False)
            call_args = mock_cmd.call_args[0]
            assert "--quiet" not in call_args


class TestInjectAndSubmit:
    def test_successful_flow(self):
        with patch("sbsllm.browser.focus_tab", return_value=True):
            with patch("sbsllm.browser.run_js") as mock_js:
                mock_js.side_effect = ["OK", "OK"]
                with patch("sbsllm.browser.time.sleep"):
                    result = inject_and_submit("qutebrowser", 1, "inject_js", "submit_js")
                    assert result["tab"] == 1
                    assert result["inject"] == "OK"
                    assert result["submit"] == "OK"

    def test_tab_focus_fails(self):
        with patch("sbsllm.browser.focus_tab", return_value=False):
            result = inject_and_submit("qutebrowser", 1, "inject_js", "submit_js")
            assert result["inject"] == "FAILED_TAB_FOCUS"
            assert result["submit"] is None

    def test_inject_fails(self):
        with patch("sbsllm.browser.focus_tab", return_value=True):
            with patch("sbsllm.browser.run_js") as mock_js:
                mock_js.return_value = "NO_INPUT"
                with patch("sbsllm.browser.time.sleep"):
                    result = inject_and_submit("qutebrowser", 1, "inject_js", "submit_js")
                    assert result["inject"] == "NO_INPUT"
                    assert result["submit"] is None

    def test_submit_fails(self):
        with patch("sbsllm.browser.focus_tab", return_value=True):
            with patch("sbsllm.browser.run_js") as mock_js:
                mock_js.side_effect = ["OK", "NO_BUTTON"]
                with patch("sbsllm.browser.time.sleep"):
                    result = inject_and_submit("qutebrowser", 1, "inject_js", "submit_js")
                    assert result["inject"] == "OK"
                    assert result["submit"] == "NO_BUTTON"


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


class TestWithRetry:
    def test_succeeds_first_try(self):
        mock_func = MagicMock(return_value="success")
        decorated = with_retry(mock_func, max_retries=3, delay=0.1)
        result = decorated()
        assert result == "success"
        assert mock_func.call_count == 1

    def test_retries_on_exception(self):
        mock_func = MagicMock(
            side_effect=[subprocess.TimeoutExpired("x", 1), "success"]
        )
        decorated = with_retry(mock_func, max_retries=3, delay=0.01)
        result = decorated()
        assert result == "success"
        assert mock_func.call_count == 2

    def test_raises_after_max_retries(self):
        mock_func = MagicMock(
            side_effect=subprocess.TimeoutExpired("x", 1)
        )
        decorated = with_retry(mock_func, max_retries=2, delay=0.01)
        with pytest.raises(subprocess.TimeoutExpired):
            decorated()
        assert mock_func.call_count == 3  # initial + 2 retries

    def test_no_retry_on_success(self):
        mock_func = MagicMock(return_value="ok")
        decorated = with_retry(mock_func, max_retries=5, delay=0.01)
        decorated()
        assert mock_func.call_count == 1
