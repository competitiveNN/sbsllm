"""Tests for cli.py."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sbsllm.cli as cli_module
from sbsllm.browser import BrowserError
from sbsllm.cli import (
    build_parser,
    get_prompt,
    list_sites,
    main,
    run,
    run_server,
    wait_for_login,
)
from sbsllm.config import Config


class TestBuildParser:
    def test_returns_parser(self):
        parser = build_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_list_sites_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--list-sites"])
        assert args.list_sites is True

    def test_config_option(self):
        parser = build_parser()
        args = parser.parse_args(["-c", "/path/to/config.yaml"])
        assert args.config == Path("/path/to/config.yaml")

    def test_prompt_option(self):
        parser = build_parser()
        args = parser.parse_args(["-p", "hello world"])
        assert args.prompt == "hello world"

    def test_login_wait_option(self):
        parser = build_parser()
        args = parser.parse_args(["--login-wait", "60"])
        assert args.login_wait == 60

    def test_login_wait_rejects_negative(self, capsys):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--login-wait", "-1"])
        assert exc_info.value.code == 2
        assert "must be non-negative" in capsys.readouterr().err

    def test_server_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--server"])
        assert args.server is True

    def test_host_option(self):
        parser = build_parser()
        args = parser.parse_args(["--host", "0.0.0.0"])
        assert args.host == "0.0.0.0"

    def test_port_option(self):
        parser = build_parser()
        args = parser.parse_args(["--port", "9000"])
        assert args.port == 9000

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.config is None
        assert args.prompt is None
        assert args.login_wait is None
        assert args.list_sites is False
        assert args.server is False
        assert args.host == "127.0.0.1"
        assert args.port == 8080


class TestListSites:
    def test_prints_sites(self, capsys):
        list_sites()
        captured = capsys.readouterr()
        assert "chatgpt" in captured.out
        assert "claude" in captured.out
        assert "Available chat sites" in captured.out


class TestGetPrompt:
    def test_single_line(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda: "hello")
        # Need to handle the empty line termination
        inputs = iter(["hello", ""])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))
        result = get_prompt()
        assert result == "hello"

    def test_multiline(self, monkeypatch):
        inputs = iter(["line1", "line2", "line3", ""])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))
        result = get_prompt()
        assert result == "line1\nline2\nline3"

    def test_eof_returns_empty(self, monkeypatch):
        def raise_eof():
            raise EOFError()

        monkeypatch.setattr("builtins.input", lambda: raise_eof())
        result = get_prompt()
        assert result == ""


class TestRun:
    def test_browser_error_returns_1(self, capsys):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        with patch("sbsllm.cli.ensure_browser", side_effect=BrowserError("fail")):
            result = run(config, "hello", None)
            assert result == 1
            captured = capsys.readouterr()
            assert "Error: fail" in captured.err

    def test_some_tabs_failed_continues(self):
        config = Config(chats=["chatgpt", "claude"], login_wait=0, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch(
                "sbsllm.cli.open_page",
                side_effect=[MagicMock(), BrowserError("open failed")],
            ),
            patch("builtins.input", side_effect=["y", ""]),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "OK",
                "submit": "OK",
            }
            result = run(config, "hello", None)
            assert result == 0

    def test_open_page_failure_continues(self, capsys):
        config = Config(chats=["chatgpt", "claude"], login_wait=0, chrome_bin=None)
        page = MagicMock()
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch(
                "sbsllm.cli.open_page", side_effect=[page, RuntimeError("open failed")]
            ),
            patch("builtins.input", side_effect=["y", ""]),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "OK",
                "submit": "OK",
            }
            result = run(config, "hello", None)
            assert result == 0
            captured = capsys.readouterr()
            assert "open failed" in captured.err
            assert "SKIPPED" in captured.out
            mock_inject.assert_called_once()

    def test_some_tabs_failed_aborts(self):
        config = Config(chats=["chatgpt", "claude"], login_wait=0, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch(
                "sbsllm.cli.open_page",
                side_effect=[MagicMock(), BrowserError("open failed")],
            ),
            patch("builtins.input", side_effect=["n"]),
        ):
            result = run(config, "hello", None)
            assert result == 1

    def test_eof_during_login_wait(self):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("builtins.input", side_effect=EOFError()),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "OK",
                "submit": "OK",
            }
            result = run(config, "hello", None)
            assert result == 0

    def test_inject_failure_status(self, capsys):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("builtins.input"),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "NO_INPUT",
                "submit": None,
            }
            result = run(config, "hello", None)
            assert result == 0
            captured = capsys.readouterr()
            assert "INJECT FAILED: NO_INPUT" in captured.out

    def test_submit_failure_status(self, capsys):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("builtins.input"),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "OK",
                "submit": "NO_BUTTON",
            }
            result = run(config, "hello", None)
            assert result == 0
            captured = capsys.readouterr()
            assert "SUBMIT FAILED: NO_BUTTON" in captured.out

    def test_empty_prompt_returns_1(self, capsys):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("builtins.input", lambda: ""),
        ):
            result = run(config, None, None)
            assert result == 1

    def test_successful_run(self):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("builtins.input"),  # Skip login wait
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser") as mock_close,
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "OK",
                "submit": "OK",
            }
            result = run(config, "hello", None)
            assert result == 0
            mock_inject.assert_called_once()
            mock_close.assert_called_once()

    def test_login_wait_override(self):
        config = Config(chats=["chatgpt"], login_wait=30, chrome_bin=None)
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("builtins.input"),
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "OK",
                "submit": "OK",
            }
            result = run(config, "hello", 0)
            assert result == 0

    def test_server_starts_before_login_and_prompt(self):
        config = Config(chats=["chatgpt"], login_wait=30, chrome_bin=None)
        server = MagicMock()
        server.url = "http://127.0.0.1:8080/"
        thread = MagicMock()
        events = []

        def start_server(_server):
            events.append("server")
            return thread

        def wait_for_login(_wait):
            events.append("login")

        def get_prompt():
            events.append("prompt")
            return "hello"

        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("sbsllm.cli._start_server", side_effect=start_server),
            patch("sbsllm.cli.wait_for_login", side_effect=wait_for_login),
            patch("sbsllm.cli.get_prompt", side_effect=get_prompt),
            patch(
                "sbsllm.cli.inject_and_submit",
                return_value={"inject": "OK", "submit": "OK"},
            ),
            patch("sbsllm.cli.close_browser"),
        ):
            assert run(config, None, None) == 0

        assert events == ["server", "login", "prompt"]

    def test_server_startup_failure_cleans_up(self):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        server = MagicMock()
        server.url = "http://127.0.0.1:8080/"

        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("sbsllm.cli.create_server", return_value=server),
            patch("sbsllm.cli._start_server", return_value=None),
            patch("sbsllm.cli.close_browser") as mock_close,
        ):
            result = run(config, "hello", None)

        assert result == 1
        server.stop.assert_called_once()
        mock_close.assert_called_once()

    def test_login_wait_timeout(self):
        with (
            patch("sbsllm.cli.select.select", return_value=([], [], [])),
            patch("builtins.input") as mock_input,
        ):
            wait_for_login(5)

        mock_input.assert_not_called()

    def test_login_wait_enter_before_timeout(self, monkeypatch):
        inputs = iter([""])
        monkeypatch.setattr("builtins.input", lambda: next(inputs))
        wait_for_login(0)


class TestRunServer:
    def test_server_browser_error(self, capsys):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        with patch("sbsllm.cli.ensure_browser", side_effect=BrowserError("fail")):
            result = run_server(config, "127.0.0.1", 8080)
            assert result == 1

    def test_server_open_page_failure(self):
        config = Config(chats=["chatgpt", "claude"], login_wait=0, chrome_bin=None)
        page = MagicMock()
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch(
                "sbsllm.cli.open_page", side_effect=[page, RuntimeError("open failed")]
            ),
            patch("sbsllm.cli.create_server") as mock_create,
        ):
            mock_server = MagicMock()
            mock_create.return_value = mock_server
            result = run_server(config, "127.0.0.1", 8080)
            assert result == 0
            mock_create.assert_called_once_with(
                model_map={"chatgpt": "chatgpt"},
                tab_map={"chatgpt": page},
                host="127.0.0.1",
                port=8080,
                browser_timeout=180,
                browser_lock_timeout=300,
                response_idle_timeout=3.0,
            )
            mock_server.start.assert_called_once()

    def test_server_success(self):
        config = Config(chats=["chatgpt", "claude"], login_wait=0, chrome_bin=None)
        pages = [MagicMock(), MagicMock()]
        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=pages),
            patch("sbsllm.cli.create_server") as mock_create,
            patch("builtins.print") as mock_print,
        ):
            mock_server = MagicMock()
            mock_server.url = "http://0.0.0.0:9000/"
            mock_create.return_value = mock_server
            result = run_server(config, "0.0.0.0", 9000)
            assert result == 0
            mock_create.assert_called_once_with(
                model_map={"chatgpt": "chatgpt", "claude": "claude"},
                tab_map={"chatgpt": pages[0], "claude": pages[1]},
                host="0.0.0.0",
                port=9000,
                browser_timeout=180,
                browser_lock_timeout=300,
                response_idle_timeout=3.0,
            )
            mock_server.start.assert_called_once()
            url_calls = [
                call
                for call in mock_print.call_args_list
                if "http://0.0.0.0:9000/v1/chat/completions" in str(call)
                or "OpenAI-compatible server URL: http://0.0.0.0:9000/" in str(call)
            ]
            assert url_calls
            for call in url_calls:
                assert call.kwargs.get("flush") is True

    def test_server_startup_failure_cleans_up(self):
        config = Config(chats=["chatgpt"], login_wait=0, chrome_bin=None)
        server = MagicMock()
        server.url = "http://127.0.0.1:8080/"

        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=[MagicMock()]),
            patch("sbsllm.cli.create_server", return_value=server),
            patch("sbsllm.cli._start_server", return_value=None),
            patch("sbsllm.cli.close_browser") as mock_close,
        ):
            result = run_server(config, "127.0.0.1", 8080)

        assert result == 1
        server.stop.assert_called_once()
        mock_close.assert_called_once()


class TestMain:
    def test_list_sites_exits(self):
        with patch("sbsllm.cli.build_parser") as mock_parser:
            mock_parser.return_value.parse_args.return_value = MagicMock(
                list_sites=True,
                server=False,
                config=None,
                prompt=None,
                login_wait=None,
                host="127.0.0.1",
                port=8080,
            )
            with patch("sbsllm.cli.list_sites"):
                main()

    def test_normal_flow(self):
        with patch("sbsllm.cli.build_parser") as mock_parser:
            mock_parser.return_value.parse_args.return_value = MagicMock(
                list_sites=False,
                server=False,
                config=None,
                prompt="hello",
                login_wait=None,
                host="0.0.0.0",
                port=9000,
                chrome_bin=None,
                json_log=False,
            )
            with patch("sbsllm.cli.load_config") as mock_load:
                mock_load.return_value = Config(
                    chats=["chatgpt"], login_wait=0, chrome_bin=None
                )
                with patch("sbsllm.cli.run", return_value=0) as mock_run:
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0

            mock_run.assert_called_once_with(
                mock_load.return_value,
                "hello",
                None,
                None,
                "0.0.0.0",
                9000,
                False,  # json_log
            )

    def test_run_returns_error_code(self):
        with patch("sbsllm.cli.build_parser") as mock_parser:
            mock_parser.return_value.parse_args.return_value = MagicMock(
                list_sites=False,
                server=False,
                config=None,
                prompt="hello",
                login_wait=None,
                host="127.0.0.1",
                port=8080,
            )
            with patch("sbsllm.cli.load_config") as mock_load:
                mock_load.return_value = Config(
                    chats=["chatgpt"], login_wait=0, chrome_bin=None
                )
                with patch("sbsllm.cli.run", return_value=1):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1

    def test_server_mode(self):
        with patch("sbsllm.cli.build_parser") as mock_parser:
            mock_parser.return_value.parse_args.return_value = MagicMock(
                list_sites=False,
                server=True,
                config=None,
                prompt=None,
                login_wait=None,
                host="127.0.0.1",
                port=8080,
            )
            with patch("sbsllm.cli.load_config") as mock_load:
                mock_load.return_value = Config(
                    chats=["chatgpt"], login_wait=0, chrome_bin=None
                )
                with patch("sbsllm.cli.run_server", return_value=0):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 0


class TestMainCall:
    """Test the if __name__ == '__main__' entry point."""

    def test_main_call(self):
        with (
            patch("sbsllm.cli.build_parser") as mock_parser,
            patch("sbsllm.cli.load_config") as mock_load,
            patch("sbsllm.cli.run", return_value=0),
            pytest.raises(SystemExit),
        ):
            mock_parser.return_value.parse_args.return_value = MagicMock(
                list_sites=False,
                server=False,
                config=None,
                prompt="hello",
                login_wait=None,
                host="127.0.0.1",
                port=8080,
            )
            mock_load.return_value = Config(
                chats=["chatgpt"], login_wait=0, chrome_bin=None
            )
            # Call main directly
            from sbsllm.cli import main

            main()

    def test_name_main_guard_exists(self):
        """Verify the if __name__ == '__main__' guard exists in source."""
        import inspect

        source = inspect.getsource(cli_module)
        assert 'if __name__ == "__main__":' in source
        assert "main()" in source


class TestDirectExecution:
    """Tests for direct module execution (python -m sbsllm.cli)."""

    def test_direct_execution_help(self):
        """Test running module directly with --help."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "sbsllm.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "sbsllm" in result.stdout

    def test_direct_execution_json_log_flag(self):
        """Test running module directly with --json-log flag."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "sbsllm.cli", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert "--json-log" in result.stdout


class TestMultiChatFanout:
    """Integration tests for multi-chat fanout scenario."""

    def test_fanout_to_multiple_chats(self):
        """Test that a single prompt is fanned out to all configured chats."""
        config = Config(
            chats=["chatgpt", "claude", "deepseek"], login_wait=0, chrome_bin=None
        )
        pages = [MagicMock(), MagicMock(), MagicMock()]

        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=pages),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("builtins.input"),  # Skip login wait
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.return_value = {
                "tab": 1,
                "inject": "OK",
                "submit": "OK",
            }
            result = run(config, "Test prompt", None)
            assert result == 0
            # Should call inject_and_submit for each chat
            assert mock_inject.call_count == 3
            # Verify each call has correct arguments
            for i, call in enumerate(mock_inject.call_args_list):
                args, _ = call
                page, _inject_js, _submit_js, tab_index = args
                assert tab_index == i + 1
                assert page == pages[i]

    def test_fanout_with_one_failed_tab(self):
        """Test fanout continues to other tabs when one fails."""
        config = Config(
            chats=["chatgpt", "claude", "deepseek"], login_wait=0, chrome_bin=None
        )
        pages = [MagicMock(), MagicMock(), MagicMock()]

        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=pages),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("builtins.input"),  # Skip login wait
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            # First and third succeed, second fails
            mock_inject.side_effect = [
                {"tab": 1, "inject": "OK", "submit": "OK"},
                {"tab": 2, "inject": "NO_INPUT", "submit": None},
                {"tab": 3, "inject": "OK", "submit": "OK"},
            ]
            result = run(config, "Test prompt", None)
            assert result == 0
            assert mock_inject.call_count == 3

    def test_fanout_with_all_failed_tabs(self):
        """Test fanout handles all tabs failing gracefully."""
        config = Config(chats=["chatgpt", "claude"], login_wait=0, chrome_bin=None)
        pages = [MagicMock(), MagicMock()]

        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=pages),
            patch("sbsllm.cli.inject_and_submit") as mock_inject,
            patch("builtins.input"),  # Skip login wait
            patch("sbsllm.cli.create_server"),
            patch("sbsllm.cli.close_browser"),
        ):
            mock_inject.side_effect = [
                {"tab": 1, "inject": "NO_INPUT", "submit": None},
                {"tab": 2, "inject": "NO_BUTTON", "submit": "NO_BUTTON"},
            ]
            result = run(config, "Test prompt", None)
            assert result == 0
            assert mock_inject.call_count == 2

    def test_server_mode_model_map_matches_tabs(self):
        """Test that in server mode, model_map only includes successful tabs."""
        config = Config(
            chats=["chatgpt", "claude", "deepseek"], login_wait=0, chrome_bin=None
        )
        pages = [MagicMock(), None, MagicMock()]  # claude fails

        with (
            patch("sbsllm.cli.ensure_browser"),
            patch("sbsllm.cli.open_page", side_effect=pages),
            patch("sbsllm.cli.create_server") as mock_create,
        ):
            mock_server = MagicMock()
            mock_server.url = "http://127.0.0.1:8080/"
            mock_create.return_value = mock_server
            result = run_server(config, "127.0.0.1", 8080)
            assert result == 0
            # model_map should only include successful tabs
            call_args = mock_create.call_args[1]
            assert call_args["model_map"] == {
                "chatgpt": "chatgpt",
                "deepseek": "deepseek",
            }
            assert "claude" not in call_args["model_map"]
            assert call_args["tab_map"] == {"chatgpt": pages[0], "deepseek": pages[2]}

    def test_json_log_flag_parses(self):
        """Test that --json-log flag is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["--json-log"])
        assert args.json_log is True

    def test_json_log_flag_default_false(self):
        """Test that --json-log defaults to False."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.json_log is False
