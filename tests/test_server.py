"""Tests for server.py."""

import json
import socket
import threading
import time
import types
from unittest.mock import MagicMock, patch

from sbsllm.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    OpenAIHandler,
    Server,
    _ModelLockRegistry,
    create_server,
)


class TestCreateServer:
    def test_creates_server(self):
        server = create_server()
        assert isinstance(server, Server)

    def test_default_values(self):
        server = create_server()
        assert server.model_map == {}
        assert server.tab_map == {}
        assert server.host == DEFAULT_HOST
        assert server.port == DEFAULT_PORT

    def test_custom_values(self):
        tab = object()
        server = create_server(
            model_map={"gpt-4": "chatgpt"},
            tab_map={"gpt-4": tab},
            host="0.0.0.0",
            port=9000,
        )
        assert server.model_map == {"gpt-4": "chatgpt"}
        assert server.tab_map == {"gpt-4": tab}
        assert server.host == "0.0.0.0"
        assert server.port == 9000


class TestServerStartStop:
    def test_start_and_stop(self):
        server = create_server(port=0)  # Use port 0 for auto-assign
        mock_http_server = MagicMock()
        mock_http_server.handle_request.side_effect = [
            socket.timeout,
            KeyboardInterrupt(),
        ]

        with patch("sbsllm.server._SBSHTTPServer", return_value=mock_http_server):
            # Start in a thread
            thread = threading.Thread(target=server.start, daemon=True)
            thread.start()
            # Wait for the socket to be bound rather than sleeping a fixed
            # amount: otherwise stop() can land before the serve loop runs.
            server.wait_until_ready(5)
            server.stop()
            thread.join(timeout=2)

        mock_http_server.handle_request.assert_called()
        # No shutdown call; stop uses server_close
        mock_http_server.server_close.assert_called()

    def test_stop_without_start(self):
        server = create_server()
        server.stop()  # Should not raise

    def test_url_uses_bound_ephemeral_port(self):
        server = create_server(port=0)
        server._server = MagicMock()
        server._server.server_address = ("127.0.0.1", 43123)

        assert server.url == "http://127.0.0.1:43123/"


class TestServerStartupOutput:
    """Server startup output must flush so it appears in redirected stdout."""

    def test_start_logs_and_flushes(self):
        server = create_server(host="127.0.0.1", port=9090)
        mock_http_server = MagicMock()
        mock_http_server.handle_request.side_effect = [
            socket.timeout,
            KeyboardInterrupt(),
        ]

        with (
            patch("sbsllm.server._SBSHTTPServer", return_value=mock_http_server),
            patch("builtins.print") as mock_print,
        ):
            thread = threading.Thread(target=server.start, daemon=True)
            thread.start()
            time.sleep(0.1)
            server.stop()
            thread.join(timeout=2)

        calls = [
            call
            for call in mock_print.call_args_list
            if "OpenAI-compatible server URL:" in str(call)
        ]
        assert calls
        assert "http://127.0.0.1:" in str(calls[0])
        assert calls[0].kwargs.get("flush") is True


class TestOpenAIHandlerModels:
    def test_models_empty(self):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.model_map = {}
        handler.tab_map = {}
        handler.server = MagicMock(browser_lock=threading.Lock())
        handler.headers = {"x-request-id": "test-request-id"}
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()

        OpenAIHandler._handle_models(handler, "test-request-id")
        handler._send_json.assert_called_once()
        call_args = handler._send_json.call_args[0]
        assert call_args[0] == 200
        data = call_args[1]
        assert data["object"] == "list"
        assert data["data"] == []
        assert call_args[2] == "test-request-id"

    def test_models_with_entries(self):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.model_map = {"gpt-4": "chatgpt", "claude-3": "claude"}
        handler.tab_map = {}
        handler.server = MagicMock(browser_lock=threading.Lock())
        handler.headers = {"x-request-id": "test-request-id"}
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()

        OpenAIHandler._handle_models(handler, "test-request-id")

        call_args = handler._send_json.call_args[0]
        assert call_args[0] == 200
        data = call_args[1]
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "gpt-4"
        assert data["data"][0]["root"] == "chatgpt"
        assert call_args[2] == "test-request-id"


class TestOpenAIHandlerChatCompletions:
    def _make_handler(
        self,
        request_body: bytes,
        content_length: int | None = None,
        prompt_result: str = "Hello!",
    ):
        """Create a real handler instance with the given request body."""
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.headers = {
            "Content-Length": str(content_length or len(request_body)),
            "x-request-id": "test-request-id",
        }
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = request_body
        handler.model_map = {"gpt-4": "chatgpt", "claude-3": "claude"}
        # Create page mocks with health check methods
        page_mock = MagicMock()
        page_mock.is_closed.return_value = False
        page_mock.evaluate.return_value = 2
        handler.tab_map = {"gpt-4": page_mock, "claude-3": page_mock}
        handler._build_prompt = MagicMock(return_value=prompt_result)
        # Provide a server mock with browser_lock for thread-safe browser access
        handler.server = MagicMock(
            browser_lock=threading.Lock(),
            browser_timeout=60,
            browser_lock_timeout=10,
        )
        # Mock output methods to capture calls
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()
        return handler

    def test_missing_body(self):
        handler = self._make_handler(b"", content_length=0)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Request body is empty", request_id="test-request-id"
        )

    def test_invalid_json(self):
        handler = self._make_handler(b"not json")
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once()
        call_kwargs = handler._send_error.call_args.kwargs
        assert call_kwargs.get("request_id") == "test-request-id"

    def test_missing_model(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Missing 'model' field", request_id="test-request-id"
        )

    def test_unknown_model(self):
        body = json.dumps(
            {"model": "unknown-model", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once()
        call_kwargs = handler._send_error.call_args.kwargs
        assert call_kwargs.get("request_id") == "test-request-id"

    def test_missing_messages(self):
        body = json.dumps({"model": "gpt-4"}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Missing 'messages' field", request_id="test-request-id"
        )

    def test_missing_browser_tab(self):
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)
        handler.tab_map = {}

        OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        assert "No browser tab for model: gpt-4" in call_args[1]
        assert "The browser tab may have crashed" in call_args[1]
        assert call_args[2] == "server_error"
        assert (
            handler._send_error.call_args.kwargs.get("request_id") == "test-request-id"
        )

    def test_successful_completion(self):
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {"tab": 1, "inject": "OK", "submit": "OK"}
            with patch("sbsllm.server.inject_prompt") as mock_inject:
                mock_inject.return_value = "inject_js"
                with patch("sbsllm.server.submit_js") as mock_submit_js:
                    mock_submit_js.return_value = "submit_js"
                    with patch("sbsllm.server.extract_js") as mock_extract:
                        mock_extract.return_value = None
                        with patch(
                            "sbsllm.server.check_page_health", return_value=True
                        ):
                            OpenAIHandler._handle_chat_completions(handler)

        # No extraction JS for this site → visible 502, not a fake
        # "Prompt sent to X successfully" answer.
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        assert "extraction is not supported" in call_args[1].lower()
        assert call_args[2] == "server_error"
        assert call_args[3] == "test-request-id"

    def test_inject_failure(self):
        """Inject failure must surface as a 502, not a fake assistant message."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {"tab": 1, "inject": "NO_INPUT", "submit": None}
            with (
                patch("sbsllm.server.inject_prompt") as mock_inject,
                patch("sbsllm.server.submit_js") as mock_submit_js,
                patch("sbsllm.server.extract_js", return_value="extract_js"),
                patch("sbsllm.server.check_page_health", return_value=True),
                patch(
                    "sbsllm.server.capture_response",
                    return_value={
                        "content": "",
                        "thinking": None,
                        "busy": False,
                        "done": False,
                        "count": 0,
                        "found": False,
                    },
                ),
            ):
                mock_inject.return_value = "inject_js"
                mock_submit_js.return_value = "submit_js"
                OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        assert "Failed to inject" in call_args[1]
        assert call_args[2] == "server_error"
        assert call_args[3] == "test-request-id"

    def test_submit_failure(self):
        """Submit failure must surface as a 502, not a fake assistant message."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {"tab": 1, "inject": "OK", "submit": "NO_BUTTON"}
            with (
                patch("sbsllm.server.inject_prompt") as mock_inject,
                patch("sbsllm.server.submit_js") as mock_submit_js,
                patch("sbsllm.server.extract_js", return_value=None),
                patch("sbsllm.server.check_page_health", return_value=True),
            ):
                mock_inject.return_value = "inject_js"
                mock_submit_js.return_value = "submit_js"
                OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        assert "Failed to submit" in call_args[1]
        assert call_args[2] == "server_error"
        assert call_args[3] == "test-request-id"

    def test_browser_error(self):
        from sbsllm.browser import BrowserError

        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with (
            patch("sbsllm.server.inject_and_submit", side_effect=BrowserError("fail")),
            patch("sbsllm.server.inject_prompt") as mock_inject,
        ):
            mock_inject.return_value = "inject_js"
            with (
                patch("sbsllm.server.submit_js") as mock_submit_js,
                patch("sbsllm.server.check_page_health", return_value=True),
                patch(
                    "sbsllm.server.capture_response",
                    return_value={
                        "content": "",
                        "thinking": None,
                        "busy": False,
                        "done": False,
                        "count": 0,
                        "found": False,
                    },
                ),
            ):
                mock_submit_js.return_value = "submit_js"
                OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args
        assert call_args[0][0] == 502
        assert (
            "Browser error from chatgpt: fail" in call_args[0][1]
            or "Browser error: fail" in call_args[0][1]
        )
        assert call_args[0][2] == "server_error"
        assert call_args[0][3] == "test-request-id"

    def test_browser_page_error_returns_502(self):
        """A stale/closed page fields a PlaywrightError as status, not a 200."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {
                "tab": 1,
                "inject": "BROWSER_ERROR: page lost",
                "submit": None,
            }
            with patch("sbsllm.server.inject_prompt") as mock_inject:
                mock_inject.return_value = "inject_js"
                with (
                    patch("sbsllm.server.submit_js") as mock_submit_js,
                    patch("sbsllm.server.check_page_health", return_value=True),
                    patch(
                        "sbsllm.server.capture_response",
                        return_value={
                            "content": "",
                            "thinking": None,
                            "busy": False,
                            "done": False,
                            "count": 0,
                            "found": False,
                        },
                    ),
                ):
                    mock_submit_js.return_value = "submit_js"
                    OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        assert "Browser error" in call_args[1]
        assert call_args[3] == "test-request-id"

    def test_internal_error(self):
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with (
            patch("sbsllm.server.inject_and_submit", side_effect=RuntimeError("boom")),
            patch("sbsllm.server.inject_prompt") as mock_inject,
            patch("sbsllm.server.submit_js") as mock_submit_js,
            patch("sbsllm.server.check_page_health", return_value=True),
            patch(
                "sbsllm.server.capture_response",
                return_value={
                    "content": "",
                    "thinking": None,
                    "busy": False,
                    "done": False,
                    "count": 0,
                    "found": False,
                },
            ),
        ):
            mock_inject.return_value = "inject_js"
            mock_submit_js.return_value = "submit_js"
            OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 500
        assert "Internal error" in call_args[1]
        assert call_args[3] == "test-request-id"

    def _patch_success(self, response_result):
        """Patch the non-streaming path to return a successful response."""
        return (
            patch(
                "sbsllm.server.inject_and_submit",
                return_value={"tab": 1, "inject": "OK", "submit": "OK"},
            ),
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch("sbsllm.server.check_page_health", return_value=True),
            patch(
                "sbsllm.server.capture_response",
                return_value={
                    "found": False,
                    "content": "",
                    "thinking": None,
                    "busy": False,
                    "done": False,
                    "count": 0,
                },
            ),
            patch("sbsllm.server.wait_for_response", return_value=response_result),
            patch(
                "sbsllm.server.get_page_snapshot",
                return_value={"url": "http://x", "title": "t", "text_preview": ""},
            ),
        )

    def test_successful_completion_returns_200(self):
        """Non-streaming success must return 200 with the assistant content."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        patches = self._patch_success(
            {"found": True, "content": "4", "thinking": None, "done": True}
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            OpenAIHandler._handle_chat_completions(handler)

        handler._send_json.assert_called_once()
        call_args = handler._send_json.call_args[0]
        assert call_args[0] == 200
        data = call_args[1]
        assert data["choices"][0]["message"]["content"] == "4"
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_non_streaming_returns_thinking(self):
        """Non-streaming path must surface thinking in the response body."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        patches = self._patch_success(
            {"found": True, "content": "4", "thinking": "2+2=4", "done": True}
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            OpenAIHandler._handle_chat_completions(handler)

        handler._send_json.assert_called_once()
        call_args = handler._send_json.call_args[0]
        data = call_args[1]
        assert data["thinking"] == "2+2=4"

    def test_non_streaming_timeout_returns_504(self):
        """A timed-out non-streaming request must surface as 504, not 200."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        patches = self._patch_success(
            {
                "found": False,
                "content": "",
                "thinking": None,
                "timed_out": True,
                "done": False,
            }
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 504
        assert "timeout" in call_args[1].lower()
        assert call_args[3] == "test-request-id"

    def test_non_streaming_no_response_returns_504(self):
        """No assistant response detected must surface as 504."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        patches = self._patch_success(
            {
                "found": False,
                "content": "",
                "thinking": None,
                "login_wall": False,
                "done": False,
            }
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 504
        assert "No assistant response" in call_args[1]
        assert call_args[3] == "test-request-id"

    def test_non_streaming_login_wall_returns_502(self):
        """A login wall must surface as 502 with a sign-in hint, not 504."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        patches = self._patch_success(
            {
                "found": False,
                "content": "",
                "thinking": None,
                "login_wall": True,
                "done": False,
            }
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
        ):
            OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        assert "signed-in session" in call_args[1]
        assert call_args[3] == "test-request-id"


class TestRequestParsingEdgeCases:
    """Test request parsing edge cases."""

    def _make_handler(self, request_body: bytes, content_length: int | None = None):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.headers = {
            "Content-Length": str(content_length or len(request_body)),
            "x-request-id": "test-request-id",
        }
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = request_body
        handler.model_map = {"gpt-4": "chatgpt", "claude-3": "claude"}
        page_mock = MagicMock()
        page_mock.is_closed.return_value = False
        page_mock.evaluate.return_value = 2
        handler.tab_map = {"gpt-4": page_mock, "claude-3": page_mock}
        handler._build_prompt = MagicMock(return_value="Hello!")
        handler.server = MagicMock(
            browser_lock=threading.Lock(),
            browser_timeout=60,
            browser_lock_timeout=10,
        )
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()
        return handler

    def test_array_request_body(self):
        """Array as JSON root should be rejected."""
        handler = self._make_handler(b"[]")
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Request body must be a JSON object", request_id="test-request-id"
        )

    def test_string_request_body(self):
        """String as JSON root should be rejected."""
        handler = self._make_handler(b'"just a string"')
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Request body must be a JSON object", request_id="test-request-id"
        )

    def test_number_request_body(self):
        """Number as JSON root should be rejected."""
        handler = self._make_handler(b"42")
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Request body must be a JSON object", request_id="test-request-id"
        )

    def test_null_request_body(self):
        """Null as JSON root should be rejected."""
        handler = self._make_handler(b"null")
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Request body must be a JSON object", request_id="test-request-id"
        )

    def test_messages_as_string(self):
        """Messages as string should be rejected."""
        body = json.dumps({"model": "gpt-4", "messages": "not a list"}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Missing 'messages' field", request_id="test-request-id"
        )

    def test_messages_as_number(self):
        """Messages as number should be rejected."""
        body = json.dumps({"model": "gpt-4", "messages": 123}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Missing 'messages' field", request_id="test-request-id"
        )

    def test_messages_as_null(self):
        """Messages as null should be rejected."""
        body = json.dumps({"model": "gpt-4", "messages": None}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Missing 'messages' field", request_id="test-request-id"
        )

    def test_malformed_content_length_negative(self):
        """Negative Content-Length should be rejected."""
        handler = self._make_handler(
            b'{"model": "gpt-4", "messages": []}', content_length=-1
        )
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Invalid 'Content-Length' header", request_id="test-request-id"
        )

    def test_malformed_content_length_non_numeric(self):
        """Non-numeric Content-Length should be rejected."""
        handler = self._make_handler(b'{"model": "gpt-4", "messages": []}')
        handler.headers["Content-Length"] = "not_a_number"
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Invalid 'Content-Length' header", request_id="test-request-id"
        )

    def test_missing_content_length(self):
        """Missing Content-Length should be treated as 0."""
        handler = self._make_handler(b'{"model": "gpt-4", "messages": []}')
        del handler.headers["Content-Length"]
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Request body is empty", request_id="test-request-id"
        )

    def test_content_length_exceeds_max(self):
        """Content-Length exceeding MAX_REQUEST_BYTES should be rejected."""
        handler = self._make_handler(b"{}", content_length=2_000_000)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 413
        assert "too large" in call_args[1]

    def test_empty_messages_list(self):
        """Empty messages list should be rejected."""
        body = json.dumps({"model": "gpt-4", "messages": []}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Missing 'messages' field", request_id="test-request-id"
        )

    def test_message_not_dict(self):
        """Non-dict message in list should be rejected."""
        body = json.dumps({"model": "gpt-4", "messages": ["not a dict"]}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(
            400, "Each message must be a JSON object", request_id="test-request-id"
        )


class TestBuildPrompt:
    def test_single_user_message(self):
        handler = MagicMock(spec=OpenAIHandler)
        messages = [{"role": "user", "content": "Hello!"}]
        result = OpenAIHandler._build_prompt(handler, messages)
        assert "[User]" in result
        assert "Hello!" in result

    def test_system_and_user(self):
        handler = MagicMock(spec=OpenAIHandler)
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi!"},
        ]
        result = OpenAIHandler._build_prompt(handler, messages)
        assert "[System]" in result
        assert "You are helpful." in result
        assert "[User]" in result
        assert "Hi!" in result

    def test_multi_part_content(self):
        handler = MagicMock(spec=OpenAIHandler)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            }
        ]
        result = OpenAIHandler._build_prompt(handler, messages)
        assert "Part 1" in result
        assert "Part 2" in result

    def test_empty_messages(self):
        handler = MagicMock(spec=OpenAIHandler)
        result = OpenAIHandler._build_prompt(handler, [])
        assert result == ""


class TestHTTPMethods:
    """Test HTTP method routing."""

    def test_do_get_models(self):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.path = "/v1/models"
        handler.model_map = {"gpt-4": "chatgpt"}
        handler.tab_map = {}
        handler.server = MagicMock(browser_lock=threading.Lock())
        handler.headers = {"x-request-id": "test-request-id"}
        handler._handle_models = MagicMock()
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()

        OpenAIHandler.do_GET(handler)
        handler._handle_models.assert_called_once_with("test-request-id")

    def test_do_get_health(self):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.path = "/health"
        handler.model_map = {"gpt-4": "chatgpt"}
        handler.tab_map = {}
        handler.server = MagicMock(browser_lock=threading.Lock())
        handler.headers = {"x-request-id": "test-request-id"}
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()

        OpenAIHandler.do_GET(handler)
        handler._send_json.assert_called_once()
        call_args = handler._send_json.call_args[0]
        assert call_args[0] == 200
        data = call_args[1]
        assert data["status"] in ("ok", "degraded")
        assert "browser_connected" in data
        assert "active_tabs" in data
        assert "models" in data
        assert data.get("request_id") == "test-request-id"
        assert call_args[2] == "test-request-id"

    def test_do_get_not_found(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/unknown"
        handler.headers = {"x-request-id": "test-request-id"}
        OpenAIHandler.do_GET(handler)
        handler._send_error.assert_called_once_with(
            404, "Not found: /unknown", "not_found", "test-request-id"
        )

    def test_do_post_chat_completions(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/v1/chat/completions"
        OpenAIHandler.do_POST(handler)
        handler._handle_chat_completions.assert_called_once()

    def test_do_post_not_found(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/unknown"
        handler.headers = {"x-request-id": "test-request-id"}
        OpenAIHandler.do_POST(handler)
        handler._send_error.assert_called_once_with(
            404, "Not found: /unknown", "not_found", "test-request-id"
        )


class TestHandlerHelpers:
    """Test handler helper methods (log_message, _send_json, _send_error)."""

    def test_log_message_suppresses(self):
        handler = MagicMock(spec=OpenAIHandler)
        # log_message should not raise and should suppress output
        result = OpenAIHandler.log_message(handler, "test %s", "arg")
        assert result is None

    def test_send_json_writes_response(self):
        handler = MagicMock()
        OpenAIHandler._send_json(handler, 200, {"key": "value"})
        handler.send_response.assert_called_once_with(200)
        handler.send_header.assert_called_once_with("Content-Type", "application/json")
        handler.end_headers.assert_called_once()
        handler.wfile.write.assert_called_once()

    def test_send_error_format(self):
        handler = MagicMock(spec=OpenAIHandler)
        OpenAIHandler._send_error(handler, 400, "Bad request", "invalid_request_error")
        handler._send_json.assert_called_once()
        call_args = handler._send_json.call_args[0]
        assert call_args[0] == 400
        assert call_args[1]["error"]["message"] == "Bad request"
        assert call_args[1]["error"]["type"] == "invalid_request_error"
        assert call_args[1]["error"]["param"] is None
        assert call_args[1]["error"]["code"] is None


class TestBuildPromptAssistant:
    """Test _build_prompt with assistant role (uncovered lines 204-205)."""

    def test_assistant_message(self):
        handler = MagicMock(spec=OpenAIHandler)
        messages = [{"role": "assistant", "content": "I can help!"}]
        result = OpenAIHandler._build_prompt(handler, messages)
        assert "[Assistant]" in result
        assert "I can help!" in result

    def test_all_roles_combined(self):
        handler = MagicMock(spec=OpenAIHandler)
        messages = [
            {"role": "system", "content": "Be nice."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = OpenAIHandler._build_prompt(handler, messages)
        assert "[System]" in result
        assert "[User]" in result
        assert "[Assistant]" in result


class TestStripMetaTags:
    """The local chat wraps a meta-instruction (task/guidelines/chat history)
    around the actual user message in a single user-role message. Web chats
    must receive only the trailing user message, so the wrapper is stripped."""

    def _strip(self, text: str) -> str:
        return OpenAIHandler._strip_meta_tags(
            OpenAIHandler.__new__(OpenAIHandler), text
        )

    def test_strips_full_meta_instruction(self):
        text = (
            "### Task:\n"
            "Generate a concise title summarizing the chat history.\n"
            "### Guidelines:\n"
            "- Keep it short\n"
            "### Chat History:\n"
            "<chat_history>\n"
            "USER: hello\n"
            "ASSISTANT: \n"
            "</chat_history>"
        )
        assert self._strip(text) == "hello"

    def test_strips_multi_turn_history_keeping_last_user(self):
        text = (
            "### Chat History:\n"
            "<chat_history>\n"
            "USER: first question\n"
            "ASSISTANT: first answer\n"
            "USER: test\n"
            "ASSISTANT: \n"
            "</chat_history>"
        )
        assert self._strip(text) == "test"

    def test_plain_message_returned_as_is(self):
        assert self._strip("just a message") == "just a message"

    def test_empty_returns_empty(self):
        assert self._strip("") == ""

    def test_no_history_block_keeps_full_text(self):
        text = "### Task:\nDo something"
        assert self._strip(text) == "### Task:\nDo something"


class TestPageRecovery:
    """Test _inject_and_submit_with_recovery and _recover_page."""

    def _make_handler(self):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.model_map = {"gpt-4": "chatgpt", "claude-3": "claude"}
        page_mock = MagicMock()
        page_mock.is_closed.return_value = False
        page_mock.evaluate.return_value = 2
        handler.tab_map = {"gpt-4": page_mock, "claude-3": page_mock}
        handler.server = MagicMock(
            browser_lock=threading.Lock(),
            browser_timeout=60,
            browser_lock_timeout=10,
        )
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()
        return handler

    def test_no_recovery_when_healthy(self):
        """A healthy page should be passed straight through."""
        handler = self._make_handler()
        page = handler.tab_map["gpt-4"]
        with (
            patch("sbsllm.server.check_page_health", return_value=True),
            patch("sbsllm.server.inject_and_submit") as mock_submit,
        ):
            mock_submit.return_value = {"tab": 1, "inject": "OK", "submit": "OK"}
            status, final_page = handler._inject_and_submit_with_recovery(
                page, "gpt-4", "chatgpt", "Hello!", 1
            )
        assert status == {"tab": 1, "inject": "OK", "submit": "OK"}
        assert final_page is page
        mock_submit.assert_called_once()

    def test_recovery_on_unhealthy_page(self):
        """An unhealthy page should trigger recovery and retry."""
        handler = self._make_handler()
        old_page = handler.tab_map["gpt-4"]
        new_page = MagicMock()
        new_page.is_closed.return_value = False
        new_page.evaluate.return_value = 2

        health_calls = []

        def fake_health(page):
            health_calls.append(page)
            # Return False for the old page, True for the new page.
            return page is new_page

        submit_calls = []

        def fake_submit(page, *args, **kwargs):
            submit_calls.append(page)
            return {"tab": 1, "inject": "OK", "submit": "OK"}

        with (
            patch("sbsllm.server.check_page_health", side_effect=fake_health),
            patch("sbsllm.server.inject_and_submit", side_effect=fake_submit),
            patch.object(
                handler, "_recover_page", return_value=new_page
            ) as mock_recover,
        ):
            status, final_page = handler._inject_and_submit_with_recovery(
                old_page, "gpt-4", "chatgpt", "Hello!", 1
            )
        mock_recover.assert_called_once_with("gpt-4", "chatgpt")
        assert final_page is new_page
        assert status == {"tab": 1, "inject": "OK", "submit": "OK"}
        # inject_and_submit should only be called on the recovered page.
        assert submit_calls == [new_page]

    def test_recovery_failure_propagates_unhealthy_status(self):
        """If recovery fails, the unhealthy status is returned."""
        handler = self._make_handler()
        old_page = handler.tab_map["gpt-4"]

        with (
            patch("sbsllm.server.check_page_health", return_value=False),
            patch.object(handler, "_recover_page", return_value=None) as mock_recover,
        ):
            status, final_page = handler._inject_and_submit_with_recovery(
                old_page, "gpt-4", "chatgpt", "Hello!", 1
            )
        mock_recover.assert_called_once()
        assert status.get("inject") == "BROWSER_ERROR: page unhealthy"
        assert final_page is old_page

    def test_recovery_not_attempted_on_non_health_error(self):
        """Recovery only triggers for 'page unhealthy' errors."""
        handler = self._make_handler()
        page = handler.tab_map["gpt-4"]
        with (
            patch("sbsllm.server.check_page_health", return_value=True),
            patch("sbsllm.server.inject_and_submit") as mock_submit,
            patch.object(handler, "_recover_page") as mock_recover,
        ):
            mock_submit.return_value = {"tab": 1, "inject": "NO_INPUT", "submit": None}
            status, _final_page = handler._inject_and_submit_with_recovery(
                page, "gpt-4", "chatgpt", "Hello!", 1
            )
        mock_recover.assert_not_called()
        assert status.get("inject") == "NO_INPUT"


class TestServerStartWithInterrupt:
    """Test Server.start() KeyboardInterrupt handling."""

    def test_start_handles_keyboard_interrupt(self):
        server = create_server(port=0)
        mock_http_server = MagicMock()
        mock_http_server.handle_request.side_effect = [
            socket.timeout,
            KeyboardInterrupt(),
        ]

        with patch("sbsllm.server._SBSHTTPServer", return_value=mock_http_server):
            server.start()

        mock_http_server.handle_request.assert_called()
        mock_http_server.server_close.assert_called()

    def test_stop_with_no_server(self):
        server = create_server()
        server._server = None
        server.stop()  # Should not raise

    def test_signal_handlers_install_only_on_main_thread(self):
        server = create_server()
        with patch("sbsllm.server.signal.signal") as mock_signal:
            assert server.install_signal_handlers() is True
            assert mock_signal.call_count == 2

        server.stop()
        with patch("sbsllm.server.signal.signal") as mock_signal:
            worker = threading.Thread(target=server.install_signal_handlers)
            worker.start()
            worker.join(timeout=2)
            mock_signal.assert_not_called()

    def test_start_uses_instance_scoped_handler_maps(self):
        server = create_server(
            model_map={"first": "site-one"}, tab_map={"first": object()}
        )
        mock_http_server = MagicMock()
        mock_http_server.handle_request.side_effect = KeyboardInterrupt()

        with patch(
            "sbsllm.server._SBSHTTPServer", return_value=mock_http_server
        ) as cls:
            server.start(announce=False)

        handler_cls = cls.call_args.args[1]
        assert handler_cls.model_map is server.model_map
        assert handler_cls.tab_map is server.tab_map


class TestHandlerSSEHelpers:
    """Unit tests for OpenAIHandler SSE/output helpers."""

    def _make_handler(self):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.wfile = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.close_connection = False
        handler.headers = {}
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        return handler

    def test_send_sse_comment(self):
        handler = self._make_handler()
        handler._send_sse_comment()
        handler.wfile.write.assert_called_once_with(b": keepalive\n\n")
        handler.wfile.flush.assert_called_once()

    def test_send_sse(self):
        handler = self._make_handler()
        handler._send_sse("data")
        handler.wfile.write.assert_called_once_with(b"data: data\n\n")

    def test_send_sse_headers(self):
        handler = self._make_handler()
        handler._send_sse_headers("req-1")
        assert handler.close_connection is True
        handler.send_response.assert_called_once_with(200)
        headers = handler.send_header.call_args_list
        header_names = [c.args[0] for c in headers]
        assert "Content-Type" in header_names
        assert "Connection" in header_names
        assert "X-Accel-Buffering" in header_names

    def test_send_sse_headers_without_request_id(self):
        handler = self._make_handler()
        handler._send_sse_headers("")
        # request_id is empty — should still send headers without the X-Request-ID header
        header_names = [c.args[0] for c in handler.send_header.call_args_list]
        assert "X-Request-ID" not in header_names

    def test_sse_chunk(self):
        handler = self._make_handler()
        handler._send_sse = MagicMock()
        handler._sse_chunk("id1", 123, "gpt-4", {"content": "hi"})
        handler._send_sse.assert_called_once()
        data = json.loads(handler._send_sse.call_args[0][0])
        assert data["id"] == "id1"
        assert data["choices"][0]["delta"]["content"] == "hi"
        assert data["choices"][0]["finish_reason"] is None

    def test_sse_chunk_with_finish_reason(self):
        handler = self._make_handler()
        handler._send_sse = MagicMock()
        handler._sse_chunk("id1", 123, "gpt-4", {}, finish_reason="stop")
        handler._send_sse.assert_called_once()
        data = json.loads(handler._send_sse.call_args[0][0])
        assert data["choices"][0]["finish_reason"] == "stop"

    def test_server_setting_fallback(self):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        server = types.SimpleNamespace(browser_timeout=60)
        handler.server = server
        # When server doesn't have the attribute, falls through to default
        result = handler._server_setting("nonexistent_setting", 42)
        assert result == 42

    def test_acquire_browser_lock_timeout(self):
        """When the lock is held, _acquire_browser_lock returns a timeout."""
        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=0.01,
        )
        # Hold the lock so the second attempt times out
        lock, acquired, _ = handler._acquire_browser_lock("gpt-4")
        assert acquired is True
        try:
            result = handler._acquire_browser_lock("gpt-4")
            assert result == (None, False, 0.01)
        finally:
            lock.release()

    def test_send_busy_error(self):
        """_send_busy_error must send a 503 with lock_timeout hint."""
        handler = self._make_handler()
        handler.model_map = {"gpt-4": "chatgpt"}
        handler.tab_map = {"gpt-4": MagicMock()}
        handler._busy_models = MagicMock(return_value=["gpt-4"])
        handler._server_setting = MagicMock(return_value=30.0)
        handler._send_busy_error("req-1", "gpt-4", 30.0)
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 503
        assert "busy" in call_args[1].lower()

    def test_busy_models(self):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        registry = _ModelLockRegistry()
        handler.server = types.SimpleNamespace(model_locks=registry)
        handler.model_locks = registry
        result = handler._busy_models()
        assert isinstance(result, list)

    def _make_streaming_handler(self, **server_attrs):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.model_map = {"gpt-4": "chatgpt"}
        handler.tab_map = {"gpt-4": MagicMock()}
        handler._build_web_prompt = MagicMock(return_value="Hello!")
        handler._inject_and_submit_with_recovery = MagicMock(
            return_value=({"inject": "OK", "submit": "OK"}, MagicMock())
        )
        handler._server_setting = MagicMock(return_value=60)
        handler._acquire_browser_lock = MagicMock(
            return_value=(threading.Lock(), True, 30.0)
        )
        handler._release_browser_lock = MagicMock()
        handler._send_error = MagicMock()
        handler._send_sse = MagicMock(side_effect=lambda x: x)
        handler._send_sse_headers = MagicMock()
        handler._send_sse_comment = MagicMock()
        handler._sse_chunk = MagicMock()
        handler.wfile = MagicMock()
        handler._streaming_sse_done = []
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=30,
            browser_timeout=60,
            **server_attrs,
        )
        return handler

    def test_streaming_lock_timeout(self):
        """When the tab is busy, a 503 is sent."""
        handler = self._make_streaming_handler()
        handler._acquire_browser_lock = MagicMock(
            return_value=(threading.Lock(), False, 30.0)
        )
        handler._handle_streaming_chat_completions(
            "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
        )
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 503

    def test_streaming_extraction_unsupported(self):
        handler = self._make_streaming_handler()
        handler._send_sse_headers = MagicMock()
        with patch("sbsllm.server.extract_js", return_value=None):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 502

    def test_streaming_client_disconnected(self):
        """A BrokenPipeError during streaming is caught and logged."""
        handler = self._make_streaming_handler()
        handler._send_sse_headers = MagicMock()
        handler._sse_chunk = MagicMock()  # simulate normal chunks
        handler._send_sse = MagicMock(side_effect=BrokenPipeError("gone"))
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch("sbsllm.server.capture_response") as mock_cap,
            patch("sbsllm.server.time") as fake_time,
        ):
            clock = FakeClock()
            fake_time.monotonic.side_effect = clock.monotonic
            fake_time.sleep.side_effect = clock.sleep
            fake_time.time.return_value = 1_700_000_000
            mock_cap.return_value = {"found": None}
            # We need _send_sse_headers to set headers_sent=True
            # The mock _send_sse_headers is a MagicMock, doesn't track state
            # Let's use side_effect to simulate: first call sends headers, then
            # _stream_web_chat runs, then _send_sse raises BrokenPipeError
            handler._send_sse_headers = MagicMock()
            handler._send_sse = MagicMock(side_effect=BrokenPipeError("gone"))
            handler._sse_chunk = MagicMock()

            def send_sse(data):
                raise BrokenPipeError("gone")

            handler._send_sse = send_sse
            # Make _stream_web_chat return quickly via mock
            handler._stream_web_chat = MagicMock(
                return_value={
                    "content": "hi",
                    "thinking": None,
                    "done": True,
                    "stop_reason": "site_done",
                }
            )
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        # Should not raise — BrokenPipeError is caught

    def test_streaming_browser_error(self):
        """BROWSER_ERROR in inject/submit status sends a 502 before headers."""
        handler = self._make_streaming_handler()
        handler._inject_and_submit_with_recovery = MagicMock(
            return_value=(
                {"inject": "BROWSER_ERROR: page closed", "submit": None},
                MagicMock(),
            )
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 502
        assert "chatgpt" in handler._send_error.call_args[0][1].lower()
        assert "CAPTCHA" in handler._send_error.call_args[0][1]

    def test_streaming_inject_submit_failed(self):
        """Inject or submit returning a non-OK code sends 502 before headers."""
        handler = self._make_streaming_handler()
        handler._inject_and_submit_with_recovery = MagicMock(
            return_value=(
                {"inject": "NO_INPUT", "submit": "NO_BUTTON"},
                MagicMock(),
            )
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 502
        msg = handler._send_error.call_args[0][1]
        assert "chatgpt" in msg.lower()
        assert "NO_INPUT" in msg

    def test_streaming_invalid_status(self):
        """Non-dict status from inject_and_submit sends 500 before headers."""
        handler = self._make_streaming_handler()
        handler._inject_and_submit_with_recovery = MagicMock(
            return_value=(42, MagicMock())
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 500
        assert "chatgpt" in handler._send_error.call_args[0][1].lower()

    def test_streaming_browser_error_after_headers(self):
        """BrowserError from _stream_web_chat (headers sent) ends the SSE."""
        from sbsllm.browser import BrowserError

        handler = self._make_streaming_handler()
        handler._stream_web_chat = MagicMock(side_effect=BrowserError("eval failed"))
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch(
                "sbsllm.server.get_page_snapshot",
                return_value={"url": "https://grok.com/", "title": "Grok"},
            ),
            patch("sbsllm.server.time") as fake_time,
        ):
            clock = FakeClock()
            fake_time.monotonic.side_effect = clock.monotonic
            fake_time.sleep.side_effect = clock.sleep
            fake_time.time.return_value = 1_700_000_000
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        # Headers were sent, so the error goes out as SSE chunks + [DONE]
        handler._send_sse.assert_any_call("[DONE]")
        # The error is embedded in an SSE data line
        sent = handler._send_sse.call_args_list
        error_text = ""
        for call in sent:
            data = call[0][0]
            if isinstance(data, str) and "server_error" in data:
                error_text = data
                break
        assert "eval failed" in error_text

    def test_streaming_generic_exception_after_headers(self):
        """Generic Exception from _stream_web_chat (headers sent) ends the SSE."""
        handler = self._make_streaming_handler()
        handler._stream_web_chat = MagicMock(side_effect=RuntimeError("boom"))
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch(
                "sbsllm.server.get_page_snapshot",
                return_value={"url": "https://grok.com/", "title": "Grok"},
            ),
            patch("sbsllm.server.time") as fake_time,
        ):
            clock = FakeClock()
            fake_time.monotonic.side_effect = clock.monotonic
            fake_time.sleep.side_effect = clock.sleep
            fake_time.time.return_value = 1_700_000_000
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        handler._send_sse.assert_any_call("[DONE]")
        sent = handler._send_sse.call_args_list
        error_text = ""
        for call in sent:
            data = call[0][0]
            if isinstance(data, str) and "server_error" in data:
                error_text = data
                break
        assert "boom" in error_text

    def test_streaming_login_wall_empty_response(self):
        """Empty answer with login_wall sends a login message via SSE."""
        handler = self._make_streaming_handler()
        handler._stream_web_chat = MagicMock(
            return_value={
                "content": "",
                "thinking": None,
                "done": True,
                "stop_reason": "site_done",
                "login_wall": True,
            }
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        # Should have sent at least the content chunk + finish + [DONE]
        handler._sse_chunk.assert_called()
        handler._send_sse.assert_any_call("[DONE]")
        # The 4th positional arg to _sse_chunk is the delta dict; find the
        # content chunk that carries the login-wall message.
        text = ""
        for call in handler._sse_chunk.call_args_list:
            delta = call[0][3]
            if isinstance(delta, dict) and "content" in delta:
                text += delta["content"]
        assert "signed-in" in text.lower(), text

    def test_streaming_success_sends_terminal_and_done(self):
        """A normal streaming response sends a finish_reason chunk + [DONE]."""
        handler = self._make_streaming_handler()
        handler._stream_web_chat = MagicMock(
            return_value={
                "content": "Here is the answer.",
                "thinking": "Let me reason about this.",
                "done": True,
                "stop_reason": "site_done",
            }
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        # The handler must emit a terminal SSE chunk with finish_reason
        terminal_call = None
        for call in handler._sse_chunk.call_args_list:
            if call[1].get("finish_reason"):
                terminal_call = call
                break
        assert terminal_call is not None
        assert terminal_call[1]["finish_reason"] == "stop"
        handler._send_sse.assert_any_call("[DONE]")

    def test_streaming_partial_content_on_timeout(self):
        """Partial content with a non-clean stop_reason still terminates the SSE."""
        handler = self._make_streaming_handler()
        handler._stream_web_chat = MagicMock(
            return_value={
                "content": "Partial answer",
                "thinking": None,
                "done": False,
                "stop_reason": "no_output",
            }
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        # Content was present, so the handler skips the empty-response path
        # and sends the terminal chunk with finish_reason.
        terminal_call = None
        for call in handler._sse_chunk.call_args_list:
            if call[1].get("finish_reason"):
                terminal_call = call
                break
        assert terminal_call is not None
        assert terminal_call[1]["finish_reason"] == "length"
        handler._send_sse.assert_any_call("[DONE]")

    def test_streaming_browser_operation_timeout_before_headers(self):
        """BrowserOperationTimeout from inject/submit (before headers) → 502."""
        from sbsllm.browser import BrowserOperationTimeout

        handler = self._make_streaming_handler()
        handler._inject_and_submit_with_recovery = MagicMock(
            side_effect=BrowserOperationTimeout("worker thread blocked")
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch("sbsllm.server.time") as fake_time,
        ):
            fake_time.monotonic.return_value = 0.0
            fake_time.time.return_value = 1_700_000_000
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        # Headers were NOT sent, so _send_error is used (not SSE)
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        assert "chatgpt" in call_args[1].lower()
        assert "unresponsive" in call_args[1].lower()
        handler._send_sse_headers.assert_not_called()

    def test_stream_web_chat_thinking_then_content_deltas(self):
        """_stream_web_chat emits role, thinking, and content as ordered deltas."""
        handler = self._make_streaming_handler()

        def mock_setting(name, default):
            values = {
                "response_idle_timeout": 1.0,
                "response_done_confirm": 0.01,
                "busy_patience": 60.0,
                "thinking_patience": 60.0,
                "first_token_timeout": 5.0,
                "poll_interval": 0.01,
                "keepalive_interval": 60.0,
            }
            return values.get(name, default)

        handler._server_setting = MagicMock(side_effect=mock_setting)

        responses = [
            {
                "found": True,
                "content": "",
                "thinking": "Let me think",
                "busy": True,
                "done": False,
                "count": 1,
            },
            {
                "found": True,
                "content": "Answer",
                "thinking": "Let me think",
                "busy": True,
                "done": False,
                "count": 2,
            },
            {
                "found": True,
                "content": "Answer",
                "thinking": "Let me think",
                "busy": False,
                "done": True,
                "count": 3,
            },
            {
                "found": True,
                "content": "Answer",
                "thinking": "Let me think",
                "busy": False,
                "done": True,
                "count": 4,
            },
        ]
        call_state = {"i": 0}

        def mock_capture(*_args, **_kwargs):
            idx = min(call_state["i"], len(responses) - 1)
            call_state["i"] += 1
            return responses[idx]

        with (
            patch("sbsllm.server.capture_response", side_effect=mock_capture),
            patch("sbsllm.server.time") as fake_time,
        ):
            clock = FakeClock()
            fake_time.monotonic.side_effect = clock.monotonic
            fake_time.sleep.side_effect = clock.sleep
            fake_time.time.return_value = 1_700_000_000

            result = handler._stream_web_chat(
                MagicMock(),
                "EXTRACT",
                None,
                "chatcmpl-test",
                1234567890,
                "gpt-4",
                60.0,
            )

        assert result["stop_reason"] == "site_done"
        assert result["done"] is True

        chunk_calls = handler._sse_chunk.call_args_list
        assert len(chunk_calls) >= 2

        # First delta carries the role announcement
        first_delta = chunk_calls[0][0][3]
        assert first_delta.get("role") == "assistant"

        # Thinking delta is sent on the first chunk (with role)
        thinking_deltas = [
            c[0][3].get("thinking", "") for c in chunk_calls if "thinking" in c[0][3]
        ]
        assert "".join(thinking_deltas) == "Let me think"

        # Content delta appears after thinking, with just the content
        content_deltas = [
            c[0][3]
            for c in chunk_calls
            if "content" in c[0][3] and "role" not in c[0][3]
        ]
        assert len(content_deltas) >= 1
        assert content_deltas[0]["content"] == "Answer"


class TestHandlerHealthMetrics:
    """Tests for health and metrics endpoints."""

    def _make_handler(self):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.headers = {"x-request-id": "test-request-id"}
        handler.model_map = {"gpt-4": "chatgpt"}
        handler.tab_map = {"gpt-4": MagicMock()}
        handler._send_json = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()
        return handler

    def test_health_degraded_browser(self):
        handler = self._make_handler()
        handler.server = MagicMock()
        handler.server.tab_map = {"gpt-4": MagicMock()}
        with patch("sbsllm.browser.is_running", return_value=False):
            handler._handle_health("req-1")
        handler._send_json.assert_called_once()
        data = handler._send_json.call_args[0][1]
        assert data["browser_connected"] is False
        assert data["status"] == "degraded"

    def test_health_with_tabs(self):
        handler = self._make_handler()
        handler.server = MagicMock()
        handler.server.tab_map = {"gpt-4": MagicMock()}
        with (
            patch("sbsllm.browser.is_running", return_value=True),
            patch("sbsllm.browser.check_page_health", return_value=True),
        ):
            handler._handle_health("req-1")
        handler._send_json.assert_called_once()
        data = handler._send_json.call_args[0][1]
        assert data["browser_connected"] is True
        assert data["active_tabs"] == 1
        assert data["tab_details"]["gpt-4"]["connected"] is True

    def test_health_exception(self):
        handler = self._make_handler()
        handler.server = MagicMock()
        handler.server.tab_map = {"gpt-4": MagicMock()}
        with patch("sbsllm.browser.is_running", side_effect=RuntimeError("boom")):
            handler._handle_health("req-1")
        handler._send_json.assert_called_once()
        data = handler._send_json.call_args[0][1]
        assert data["browser_connected"] is False
        assert data["status"] == "degraded"

    def test_metrics_with_data(self):
        handler = self._make_handler()
        with patch("sbsllm.server._request_latencies", [0.1, 0.2, 0.3]):
            handler._handle_metrics("req-1")
        handler.send_response.assert_called_once_with(200)
        written = handler.wfile.write.call_args[0][0]
        text = written.decode("utf-8")
        assert "sbsllm_requests_total" in text
        assert "sbsllm_request_duration_seconds_avg" in text

    def test_do_get_not_found(self):
        handler = self._make_handler()
        handler.path = "/nonexistent"
        handler._send_error = MagicMock()
        handler.do_GET()
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 404


class TestNonStreamingPaths:
    """Tests for non-streaming _handle_chat_completions branches."""

    def _make_handler(self, prompt_result="Hello!"):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler.headers = {
            "Content-Length": str(len(body)),
            "x-request-id": "test-request-id",
        }
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body
        handler.model_map = {"gpt-4": "chatgpt"}
        page_mock = MagicMock()
        page_mock.is_closed.return_value = False
        page_mock.evaluate.return_value = 2
        handler.tab_map = {"gpt-4": page_mock}
        handler._build_web_prompt = MagicMock(return_value=prompt_result)
        handler.server = MagicMock(
            browser_lock=threading.Lock(),
            browser_timeout=60,
            browser_lock_timeout=10,
        )
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()
        return handler

    def test_non_streaming_browser_operation_timeout(self):
        """BrowserOperationTimeout must surface as a 502."""
        from sbsllm.browser import BrowserOperationTimeout

        handler = self._make_handler()
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch(
                "sbsllm.server.inject_and_submit",
                side_effect=BrowserOperationTimeout("wedged"),
            ),
        ):
            handler._handle_chat_completions()
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 502
        assert "unresponsive" in handler._send_error.call_args[0][1].lower()

    def test_non_streaming_internal_error(self):
        """An unexpected exception must surface as a 500."""
        handler = self._make_handler()
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch("sbsllm.server.inject_and_submit", side_effect=RuntimeError("boom")),
        ):
            handler._handle_chat_completions()
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 500

    def test_non_streaming_invalid_status(self):
        """Non-dict status from inject_and_submit is treated as invalid."""
        handler = self._make_handler()
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch("sbsllm.server.inject_and_submit", return_value="not a dict"),
        ):
            handler._handle_chat_completions()
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 500

    def test_non_streaming_no_extraction_supported(self):
        """When extract_js is None (site without response_selectors), 502."""
        handler = self._make_handler()
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value=None),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch(
                "sbsllm.server.inject_and_submit",
                return_value={"tab": 1, "inject": "OK", "submit": "OK"},
            ),
        ):
            handler._handle_chat_completions()
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 502


class TestServerEndToEndErrorPaths:
    """Integration tests verifying error messages include site name and context."""

    def _make_handler(self, prompt_result="Hello!"):
        import threading

        handler = OpenAIHandler.__new__(OpenAIHandler)
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler.headers = {
            "Content-Length": str(len(body)),
            "x-request-id": "test-request-id",
        }
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body
        handler.model_map = {"gpt-4": "chatgpt"}
        page_mock = MagicMock()
        page_mock.is_closed.return_value = False
        page_mock.evaluate.return_value = 2
        handler.tab_map = {"gpt-4": page_mock}
        handler._build_web_prompt = MagicMock(return_value=prompt_result)
        handler.server = MagicMock(
            browser_lock=threading.Lock(),
            browser_timeout=60,
            browser_lock_timeout=10,
        )
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        handler.wfile.write = MagicMock()
        return handler

    def test_browser_error_message_includes_site_name(self):
        """502 browser error must reference the site name."""
        from sbsllm.browser import BrowserError

        handler = self._make_handler()
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch(
                "sbsllm.server.inject_and_submit",
                side_effect=BrowserError("fail"),
            ),
        ):
            handler._handle_chat_completions()
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args
        assert call_args[0][0] == 502
        assert "chatgpt" in call_args[0][1].lower()

    def test_browser_operation_timeout_message_includes_site_name(self):
        """502 browser timeout must reference the site name."""
        from sbsllm.browser import BrowserOperationTimeout

        handler = self._make_handler()
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch(
                "sbsllm.server.inject_and_submit",
                side_effect=BrowserOperationTimeout("wedged"),
            ),
        ):
            handler._handle_chat_completions()
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args
        assert call_args[0][0] == 502
        assert "chatgpt" in call_args[0][1].lower()

    def test_internal_error_message_includes_site_name(self):
        """500 internal error must reference the site name."""
        handler = self._make_handler()
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
            patch("sbsllm.server.inject_and_submit", side_effect=RuntimeError("boom")),
        ):
            handler._handle_chat_completions()
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args
        assert call_args[0][0] == 500
        assert "chatgpt" in call_args[0][1].lower()

    def test_no_browser_tab_message_includes_site_name(self):
        """502 missing tab must reference the site name."""
        handler = self._make_handler()
        handler.tab_map = {}
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args
        assert call_args[0][0] == 502
        assert "chatgpt" in call_args[0][1].lower()


class FakeClock:
    """Minimal fake clock matching the one in test_streaming.py."""

    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += max(seconds, 0.01)
