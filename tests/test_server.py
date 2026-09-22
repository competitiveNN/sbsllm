"""Tests for server.py."""

import json
import socket
import threading
import time
from unittest.mock import MagicMock, patch

from sbsllm.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    OpenAIHandler,
    Server,
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
            time.sleep(0.1)
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
                    OpenAIHandler._handle_chat_completions(handler)

        handler._send_json.assert_called_once()
        call_args = handler._send_json.call_args[0]
        assert call_args[0] == 200
        data = call_args[1]
        assert data["object"] == "chat.completion"
        assert data["model"] == "gpt-4"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "sent to chatgpt" in data["choices"][0]["message"]["content"]
        assert data.get("request_id") == "test-request-id"

    def test_inject_failure(self):
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {"tab": 1, "inject": "NO_INPUT", "submit": None}
            with patch("sbsllm.server.inject_prompt") as mock_inject:
                mock_inject.return_value = "inject_js"
                with patch("sbsllm.server.submit_js") as mock_submit_js:
                    mock_submit_js.return_value = "submit_js"
                    OpenAIHandler._handle_chat_completions(handler)

        data = handler._send_json.call_args[0][1]
        assert "Failed to inject" in data["choices"][0]["message"]["content"]

    def test_submit_failure(self):
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {"tab": 1, "inject": "OK", "submit": "NO_BUTTON"}
            with patch("sbsllm.server.inject_prompt") as mock_inject:
                mock_inject.return_value = "inject_js"
                with patch("sbsllm.server.submit_js") as mock_submit_js:
                    mock_submit_js.return_value = "submit_js"
                    OpenAIHandler._handle_chat_completions(handler)

        data = handler._send_json.call_args[0][1]
        assert "Failed to submit" in data["choices"][0]["message"]["content"]

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
            ):
                mock_submit_js.return_value = "submit_js"
                OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once_with(
            502, "Browser error: fail. Try restarting the browser.", "server_error", "test-request-id"
        )

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
                with patch("sbsllm.server.submit_js") as mock_submit_js:
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
        ):
            mock_inject.return_value = "inject_js"
            mock_submit_js.return_value = "submit_js"
            OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 500
        assert "Internal error" in call_args[1]
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
