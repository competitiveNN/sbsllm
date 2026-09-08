"""Tests for server.py."""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

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
        assert server.qb_bin == "qutebrowser"
        assert server.model_map == {}
        assert server.tab_map == {}
        assert server.host == DEFAULT_HOST
        assert server.port == DEFAULT_PORT

    def test_custom_values(self):
        server = create_server(
            qb_bin="/usr/bin/qb",
            model_map={"gpt-4": "chatgpt"},
            tab_map={"gpt-4": 1},
            host="0.0.0.0",
            port=9000,
        )
        assert server.qb_bin == "/usr/bin/qb"
        assert server.model_map == {"gpt-4": "chatgpt"}
        assert server.tab_map == {"gpt-4": 1}
        assert server.host == "0.0.0.0"
        assert server.port == 9000


class TestServerStartStop:
    def test_start_and_stop(self):
        server = create_server(port=0)  # Use port 0 for auto-assign
        mock_http_server = MagicMock()

        with patch("sbsllm.server.HTTPServer", return_value=mock_http_server):
            # Start in a thread
            thread = threading.Thread(target=server.start, daemon=True)
            thread.start()
            time.sleep(0.1)
            server.stop()
            thread.join(timeout=2)

        mock_http_server.serve_forever.assert_called_once()
        mock_http_server.shutdown.assert_called_once()

    def test_stop_without_start(self):
        server = create_server()
        server.stop()  # Should not raise


class TestOpenAIHandlerModels:
    def test_models_empty(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.model_map = {}
        OpenAIHandler.model_map = {}
        OpenAIHandler._handle_models(handler)
        # Verify it was called
        assert handler._send_json.called

    def test_models_with_entries(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.model_map = {"gpt-4": "chatgpt", "claude-3": "claude"}
        OpenAIHandler.model_map = handler.model_map
        OpenAIHandler._handle_models(handler)

        call_args = handler._send_json.call_args
        assert call_args[0][0] == 200
        data = call_args[0][1]
        assert data["object"] == "list"
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "gpt-4"
        assert data["data"][0]["root"] == "chatgpt"


class TestOpenAIHandlerChatCompletions:
    def _make_handler(self, request_body: bytes, content_length: int = None, prompt_result: str = "Hello!"):
        """Create a mock handler with the given request body."""
        handler = MagicMock(spec=OpenAIHandler)
        handler.headers = {"Content-Length": str(content_length or len(request_body))}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = request_body
        handler.model_map = {"gpt-4": "chatgpt", "claude-3": "claude"}
        handler.tab_map = {"gpt-4": 1, "claude-3": 2}
        handler.qb_bin = "qutebrowser"
        handler._build_prompt.return_value = prompt_result
        return handler

    def test_missing_body(self):
        handler = self._make_handler(b"", content_length=0)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(400, "Request body is empty")

    def test_invalid_json(self):
        handler = self._make_handler(b"not json")
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once()
        assert "Invalid JSON" in handler._send_error.call_args[0][1]

    def test_missing_model(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(400, "Missing 'model' field")

    def test_unknown_model(self):
        body = json.dumps({
            "model": "unknown-model",
            "messages": [{"role": "user", "content": "hi"}]
        }).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once()
        assert "Unknown model" in handler._send_error.call_args[0][1]

    def test_missing_messages(self):
        body = json.dumps({"model": "gpt-4"}).encode()
        handler = self._make_handler(body)
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(400, "Missing 'messages' field")

    def test_empty_user_content(self):
        body = json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "  "}]
        }).encode()
        handler = self._make_handler(body, prompt_result="  ")
        OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once_with(400, "No user message content found")

    def test_successful_completion(self):
        body = json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }).encode()
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

    def test_inject_failure(self):
        body = json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }).encode()
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
        body = json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }).encode()
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

        body = json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit", side_effect=BrowserError("fail")):
            with patch("sbsllm.server.inject_prompt") as mock_inject:
                mock_inject.return_value = "inject_js"
                with patch("sbsllm.server.submit_js") as mock_submit_js:
                    mock_submit_js.return_value = "submit_js"
                    OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        assert "Browser error" in handler._send_error.call_args[0][1]

    def test_internal_error(self):
        body = json.dumps({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello!"}]
        }).encode()
        handler = self._make_handler(body)

        with patch("sbsllm.server.inject_and_submit", side_effect=RuntimeError("boom")):
            with patch("sbsllm.server.inject_prompt") as mock_inject:
                mock_inject.return_value = "inject_js"
                with patch("sbsllm.server.submit_js") as mock_submit_js:
                    mock_submit_js.return_value = "submit_js"
                    OpenAIHandler._handle_chat_completions(handler)

        handler._send_error.assert_called_once()
        assert "Internal error" in handler._send_error.call_args[0][1]


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
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ]
        }]
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
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/v1/models"
        OpenAIHandler.do_GET(handler)
        handler._handle_models.assert_called_once()

    def test_do_get_health(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/health"
        OpenAIHandler.do_GET(handler)
        handler._send_json.assert_called_once_with(200, {"status": "ok"})

    def test_do_get_not_found(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/unknown"
        OpenAIHandler.do_GET(handler)
        handler._send_error.assert_called_once_with(404, "Not found: /unknown", "not_found")

    def test_do_post_chat_completions(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/v1/chat/completions"
        OpenAIHandler.do_POST(handler)
        handler._handle_chat_completions.assert_called_once()

    def test_do_post_not_found(self):
        handler = MagicMock(spec=OpenAIHandler)
        handler.path = "/unknown"
        OpenAIHandler.do_POST(handler)
        handler._send_error.assert_called_once_with(404, "Not found: /unknown", "not_found")


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


class TestServerStartWithInterrupt:
    """Test Server.start() KeyboardInterrupt handling (lines 239-240)."""

    def test_start_handles_keyboard_interrupt(self):
        server = create_server(port=0)
        mock_http_server = MagicMock()
        mock_http_server.serve_forever.side_effect = KeyboardInterrupt()

        with patch("sbsllm.server.HTTPServer", return_value=mock_http_server):
            server.start()

        mock_http_server.shutdown.assert_called_once()

    def test_stop_with_no_server(self):
        server = create_server()
        server._server = None
        server.stop()  # Should not raise
