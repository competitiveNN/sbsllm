"""Tests for server.py."""

import inspect
import json
import socket
import threading
import time
import types
from typing import ClassVar
from unittest.mock import MagicMock, patch

from sbsllm.server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    OpenAIHandler,
    Server,
    _ModelLockRegistry,
    _TurnRegistry,
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

    def test_unknown_site_in_model_map_warns(self):
        """A model_map whose site id is not a real SITES entry must log a
        warning at construction time, instead of surfacing only per-request
        as a 500 long after the operator has walked away."""
        with patch("sbsllm.server.logging") as mock_logging:
            create_server(model_map={"gpt-4": "not-a-real-site"})
        mock_logger = mock_logging.getLogger.return_value
        mock_logger.warning.assert_called_once()
        call = mock_logger.warning.call_args
        assert call.args[0] == "server unknown_site_in_model_map"
        assert "not-a-real-site" in call.kwargs["extra"]["unknown_sites"]

    def test_known_site_in_model_map_is_silent(self):
        """A fully valid model_map must not log anything."""
        with patch("sbsllm.server.logging") as mock_logging:
            create_server(model_map={"gpt-4": "chatgpt", "claude-3": "claude"})
        mock_logger = mock_logging.getLogger.return_value
        mock_logger.warning.assert_not_called()


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
        assert call_kwargs.get("code") == "model_not_found"

    def test_inject_failure_sets_error_code(self):
        """The non-streaming inject/submit failure carries a stable code."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)
        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {
                "tab": 1,
                "inject": "NO_INPUT",
                "submit": None,
            }
            with (
                patch("sbsllm.server.inject_prompt"),
                patch("sbsllm.server.submit_js"),
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
                OpenAIHandler._handle_chat_completions(handler)
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 502
        assert (
            handler._send_error.call_args.kwargs.get("code") == "inject_submit_failed"
        )

    def test_timeout_sets_error_code(self):
        """The 504 timeout error carries the response_timeout code."""
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
        assert handler._send_error.call_args[0][0] == 504
        assert handler._send_error.call_args.kwargs.get("code") == "response_timeout"

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

    def test_non_streaming_exposes_reasoning_content_in_message(self):
        """Thinking must also appear in choices[0].message.reasoning_content
        so OpenAI-compatible clients reading the message object see the trace."""
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

        call_args = handler._send_json.call_args[0]
        data = call_args[1]
        message = data["choices"][0]["message"]
        assert message["reasoning_content"] == "2+2=4"
        assert message["content"] == "4"

    def test_non_streaming_returns_web_chat_response_not_placeholder(self):
        """Regression: non-streaming must return the actual captured web-chat
        response, never a placeholder like 'Prompt sent to X successfully.'"""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        real_answer = "The capital of France is Paris."
        patches = self._patch_success(
            {"found": True, "content": real_answer, "thinking": None, "done": True}
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
        data = handler._send_json.call_args[0][1]
        assert data["choices"][0]["message"]["content"] == real_answer
        assert "prompt sent" not in data["choices"][0]["message"]["content"].lower()

    def test_non_streaming_usage_is_estimated_not_zero(self):
        """Close the AGENTS.md Design TODO: token usage must be populated with
        a non-zero heuristic estimate, not the previous stub of all zeros."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        real_answer = "The capital of France is Paris, and it is beautiful in spring."
        patches = self._patch_success(
            {"found": True, "content": real_answer, "thinking": None, "done": True}
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
        data = handler._send_json.call_args[0][1]
        usage = data["usage"]
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )

    def test_estimate_tokens_is_cjk_aware(self):
        """CJK chars count ~1 token each; non-CJK ~4 chars/token. A flat
        chars/4 split under-counts the CJK prompts this tool targets."""
        from sbsllm.server import _estimate_tokens

        # Pure Han: each character maps to ~1 token.
        assert _estimate_tokens("你好世界") == 4
        # Pure ASCII: 8 chars // 4 = 2 tokens.
        assert _estimate_tokens("abcdefgh") == 2
        # Mixed: 2 Han (2) + 4 ASCII (4 // 4 = 1) = 3.
        assert _estimate_tokens("你好abcd") == 3
        # CJK fullwidth punctuation (U+FF0C) is outside the CJK ranges, so it
        # counts at the non-CJK rate: 4 Han (4) + 1 punctuation (1 // 4 = 0).
        assert _estimate_tokens("你好，世界") == 4
        # Emoji are a single code point each and non-CJK; 4 emoji -> 4 // 4 = 1.
        assert _estimate_tokens("😀😀😀😀") == 1
        # Empty and whitespace-only stay safe (0 / min 1 for non-empty).
        assert _estimate_tokens("") == 0
        assert _estimate_tokens(" ") == 1

    def test_estimate_tokens_edge_cases(self):
        """Pin the heuristic on long, mixed, and non-Han CJK inputs."""
        from sbsllm.server import _estimate_tokens

        # Long ASCII: 1000 chars // 4 = 250 tokens (floor division).
        assert _estimate_tokens("a" * 1000) == 250
        # Long Han: each character maps to ~1 token.
        assert _estimate_tokens("你" * 1000) == 1000
        # Katakana (U+30AB..) and Hangul (U+AC00..) count as CJK too.
        assert _estimate_tokens("カタカナ") == 4
        assert _estimate_tokens("한국어") == 3
        # CJK-majority mix: 9 Han (9) + 4 ASCII (4 // 4 = 1) = 10.
        assert _estimate_tokens("你" * 9 + "abcd") == 10
        # Non-CJK remainder floors: 3 ASCII -> 3 // 4 = 0 -> min 1.
        assert _estimate_tokens("abc") == 1
        # Always a non-negative int for any mix.
        assert isinstance(_estimate_tokens("Hello, 世界!"), int)

    def test_estimate_tokens_cjk_extension_boundary(self):
        """CJK Extension A (U+3400..4DBF) sits outside the covered
        blocks and counts at the non-CJK rate -- pin the boundary so a
        range widening is a deliberate, visible change."""
        from sbsllm.server import _estimate_tokens

        # 4 Ext-A chars, all non-CJK per the heuristic: 4 // 4 = 1.
        assert _estimate_tokens("㐀㐁㐂㐃") == 1
        # Mixed with basic Han: 2 Han (2) + 4 Ext-A (4 // 4 = 1) = 3.
        assert _estimate_tokens("你好㐀㐁㐂㐃") == 3

    def test_is_cjk_script_coverage(self):
        """_is_cjk covers Han, Hiragana, Katakana and Hangul
        syllables -- and deliberately nothing else (fullwidth
        punctuation, Ext-A, emoji count at the non-CJK rate)."""
        from sbsllm.server import _is_cjk

        assert _is_cjk("中")  # Han (U+4E2D)
        assert _is_cjk("あ")  # Hiragana (U+3042)
        assert _is_cjk("ア")  # Katakana (U+30A2)
        assert _is_cjk("한")  # Hangul (U+D55C)
        assert not _is_cjk("a")
        assert not _is_cjk("，")  # fullwidth comma (U+FF0C)
        assert not _is_cjk("😀")  # emoji (U+1F600)
        assert not _is_cjk("㐀")  # CJK Ext A (U+3400)

    def test_usage_fields_computes_estimates(self):
        """The shared usage helper estimates prompt/completion/total."""
        from sbsllm.server import _usage_fields

        # "Hello world" (11 ASCII) -> 11 // 4 = 2; "The answer" (10) -> 2.
        usage = _usage_fields("Hello world", "The answer", None)
        assert usage["prompt_tokens"] == 2
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 4

    def test_usage_fields_thinking_none_and_empty_match(self):
        """thinking=None and thinking='' contribute the same (zero)."""
        from sbsllm.server import _usage_fields

        assert _usage_fields("p", "c", None) == _usage_fields("p", "c", "")
        # Thinking folds into completion_tokens: prompt "你好" (2) +
        # content "c" (min 1) + thinking "你好" (2) = 3 completion.
        usage = _usage_fields("你好", "c", "你好")
        assert usage["prompt_tokens"] == 2
        assert usage["completion_tokens"] == 3
        assert usage["total_tokens"] == 5

    def test_usage_fields_all_empty_is_zero(self):
        """No prompt, no content, no thinking -> all-zero usage.

        Guards the non-negative clamping contract explicitly, since the
        heuristic uses floor division which could otherwise drift on
        tiny inputs.
        """
        from sbsllm.server import _usage_fields

        usage = _usage_fields("", "", None)
        assert usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # Whitespace-only is non-empty input and must still be >= 1 token.
        assert _usage_fields(" ", "", None)["prompt_tokens"] == 1
        assert _usage_fields("", " ", None)["completion_tokens"] == 1

    def test_usage_fields_total_is_prompt_plus_completion(self):
        """total_tokens must always equal prompt + completion."""
        from sbsllm.server import _estimate_tokens, _usage_fields

        for prompt, content, thinking in [
            ("Hello world", "The answer", None),
            ("你好世界", "カタカナと한국어", "deepseek-reasoner"),
            ("a" * 50, "b" * 50, "c" * 50),
        ]:
            usage = _usage_fields(prompt, content, thinking)
            assert (
                usage["total_tokens"]
                == usage["prompt_tokens"] + usage["completion_tokens"]
            )
            expected_completion = _estimate_tokens(content) + _estimate_tokens(thinking)
            assert usage["completion_tokens"] == expected_completion

    def test_estimate_tokens_handles_surrogate_and_combining(self):
        """Combining marks and astral (emoji) code points are counted as
        single characters each and the heuristic never raises on odd input."""
        from sbsllm.server import _estimate_tokens

        # 'e' + combining acute = 2 non-CJK chars -> 2 // 4 = 0 -> min 1.
        assert _estimate_tokens("e\u0301") == 1
        # Family-emoji sequence = 5 non-CJK code points -> 5 // 4 = 1.
        assert _estimate_tokens("👨\u200d👩\u200d👧") == 1

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

    def test_non_streaming_timeout_returns_partial_content(self):
        """A timeout with partial content on screen returns the truncated
        answer (200 + finish_reason="length"), not a bare 504 that
        loses it — mirroring OpenAI's max_tokens truncation behavior."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        partial = "The capital of France is Pa"
        patches = self._patch_success(
            {
                "found": True,
                "content": partial,
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

        handler._send_error.assert_not_called()
        handler._send_json.assert_called_once()
        call_args = handler._send_json.call_args[0]
        assert call_args[0] == 200
        data = call_args[1]
        assert data["choices"][0]["message"]["content"] == partial
        assert data["choices"][0]["finish_reason"] == "length"
        assert data["usage"]["completion_tokens"] > 0

    def test_non_streaming_timeout_returns_partial_thinking(self):
        """A thinking-only stall on timeout still returns the reasoning
        trace rather than discarding it."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        thinking = "Let me reason about this step by step."
        patches = self._patch_success(
            {
                "found": True,
                "content": "",
                "thinking": thinking,
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

        handler._send_json.assert_called_once()
        data = handler._send_json.call_args[0][1]
        message = data["choices"][0]["message"]
        assert message["reasoning_content"] == thinking
        assert data["choices"][0]["finish_reason"] == "length"

    def test_non_streaming_timeout_whitespace_only_returns_504(self):
        """A whitespace-only capture on timeout is 'nothing captured' and
        must still surface as 504, not a 200 with a blank answer."""
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello!"}]}
        ).encode()
        handler = self._make_handler(body)

        patches = self._patch_success(
            {
                "found": True,
                "content": "   \n\t  ",
                "thinking": "   ",
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

        handler._send_json.assert_not_called()
        handler._send_error.assert_called_once()
        assert handler._send_error.call_args[0][0] == 504

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


class TestBuildCompletionResponse:
    """_build_completion_response builds the OpenAI completion body."""

    def _build(self, content, thinking, finish_reason="stop"):
        return OpenAIHandler._build_completion_response(
            MagicMock(spec=OpenAIHandler),
            "gpt-4",
            "Hello!",
            content,
            thinking,
            finish_reason,
            "test-request-id",
        )

    def test_no_thinking_omits_reasoning_fields(self):
        """thinking=None/'' must not leak reasoning_content or `thinking`."""
        for thinking in (None, ""):
            resp = self._build("The answer", thinking)
            message = resp["choices"][0]["message"]
            assert message["content"] == "The answer"
            assert "reasoning_content" not in message
            assert "thinking" not in resp
            assert resp["choices"][0]["finish_reason"] == "stop"
            assert resp["usage"]["completion_tokens"] > 0

    def test_with_thinking_exposes_reasoning_content(self):
        """A thinking trace appears in both reasoning_content and `thinking`."""
        from sbsllm.server import _estimate_tokens

        resp = self._build("The answer", "Let me think.", "length")
        message = resp["choices"][0]["message"]
        assert message["reasoning_content"] == "Let me think."
        assert resp["thinking"] == "Let me think."
        assert resp["choices"][0]["finish_reason"] == "length"
        # usage folds the thinking trace into completion_tokens.
        assert resp["usage"]["completion_tokens"] == (
            _estimate_tokens("The answer") + _estimate_tokens("Let me think.")
        )

    def test_usage_and_request_id_present(self):
        """The completion body carries object/model/request_id and coherent usage."""
        resp = self._build("ok", None)
        assert resp["object"] == "chat.completion"
        assert resp["model"] == "gpt-4"
        assert resp["request_id"] == "test-request-id"
        usage = resp["usage"]
        assert usage["total_tokens"] == (
            usage["prompt_tokens"] + usage["completion_tokens"]
        )


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

    def test_send_error_includes_code(self):
        """A passed code is surfaced in the OpenAI error envelope."""
        handler = MagicMock(spec=OpenAIHandler)
        OpenAIHandler._send_error(
            handler,
            400,
            "Bad request",
            "invalid_request_error",
            code="my_code",
        )
        handler._send_json.assert_called_once()
        error = handler._send_json.call_args[0][1]["error"]
        assert error["code"] == "my_code"
        assert error["type"] == "invalid_request_error"


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

    def test_empty_chat_history_returns_empty(self):
        """An empty <chat_history> block means no user message was sent."""
        text = "### Task: Summarize\n<chat_history>\n</chat_history>"
        assert self._strip(text) == ""

    def test_open_webui_inst_format(self):
        """Open-WebUI wraps instructions in [INST] <<SYS>>...<</SYS>> ... [/INST]."""
        text = (
            "[INST] <<SYS>>You are a helpful assistant.<</SYS>>\n"
            "USER: Hello\n"
            "ASSISTANT: Hi there\n"
            "[/INST]\n"
            "What is Python?"
        )
        assert self._strip(text) == "What is Python?"

    def test_plain_user_assistant_blocks_without_history_wrapper(self):
        """Plain USER: / ASSISTANT: blocks (no <chat_history> wrapper)
        should still extract the last user turn."""
        text = "Task: Respond briefly.\nUSER: Hello\nASSISTANT: Hi there\nUSER: Goodbye"
        assert self._strip(text) == "Goodbye"


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

    def test_streaming_browser_worker_stopped_returns_502(self):
        """A BrowserWorkerStopped (worker died mid-dispatch) must surface as a
        browser-level 502, not a generic 500 internal error.

        BrowserWorkerStopped subclasses BrowserError, so the existing
        `except BrowserError` handler covers it -- this guards against future
        refactors that weaken the inheritance or catch a bare RuntimeError
        first.
        """
        from sbsllm.browser import BrowserWorkerStopped

        handler = self._make_streaming_handler()
        handler._inject_and_submit_with_recovery = MagicMock(
            side_effect=BrowserWorkerStopped("worker stopped mid-dispatch")
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
        handler._send_error.assert_called_once()
        call_args = handler._send_error.call_args[0]
        assert call_args[0] == 502
        # The streaming BrowserError message references the failure itself
        # (the non-streaming path is the one that includes the site name).
        assert "worker stopped mid-dispatch" in call_args[1].lower()
        assert "restart" in call_args[1].lower()
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

    def test_stream_web_chat_thinking_only_times_out(self):
        """A thinking-only turn (no content) times out via thinking_patience."""
        handler = self._make_streaming_handler()

        def mock_setting(name, default):
            values = {
                "response_idle_timeout": 1.0,
                "response_done_confirm": 0.01,
                "busy_patience": 60.0,
                "thinking_patience": 0.02,
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
                "thinking": "Deep reasoning...",
                "busy": True,
                "done": False,
                "count": 1,
            },
            {
                "found": True,
                "content": "",
                "thinking": "Deep reasoning...",
                "busy": True,
                "done": False,
                "count": 2,
            },
            {
                "found": True,
                "content": "",
                "thinking": "Deep reasoning...",
                "busy": True,
                "done": False,
                "count": 3,
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

        assert result["stop_reason"] == "thinking_timeout"
        assert result["done"] is False

        # Thinking delta was emitted on the first poll with role
        chunk_calls = handler._sse_chunk.call_args_list
        assert len(chunk_calls) >= 1
        first_delta = chunk_calls[0][0][3]
        assert first_delta.get("role") == "assistant"
        assert first_delta.get("thinking") == "Deep reasoning..."
        # No content deltas for a thinking-only turn
        content_deltas = [
            c for c in chunk_calls if "content" in c[0][3] and "role" not in c[0][3]
        ]
        assert len(content_deltas) == 0

    def test_streaming_thinking_only_response_terminates(self):
        """A thinking-only response from _stream_web_chat still gets [DONE]."""
        handler = self._make_streaming_handler()
        handler._stream_web_chat = MagicMock(
            return_value={
                "content": "",
                "thinking": "Just reasoning, no answer yet.",
                "done": True,
                "stop_reason": "budget_exhausted",
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
        # Thinking is present, so the empty-response error path is skipped.
        # The handler still sends the terminal chunk + [DONE].
        terminal_call = None
        for call in handler._sse_chunk.call_args_list:
            if call[1].get("finish_reason"):
                terminal_call = call
                break
        assert terminal_call is not None
        assert terminal_call[1]["finish_reason"] == "stop"
        handler._send_sse.assert_any_call("[DONE]")

    def test_streaming_final_chunk_carries_usage(self):
        """Regression: the terminal SSE chunk must include non-zero token
        usage. Before the content/thinking-extraction fix, the final-usage
        block referenced unbound locals and raised NameError, so no terminal
        chunk with usage (nor [DONE]) was ever emitted on a happy path."""
        handler = self._make_streaming_handler()
        handler._stream_web_chat = MagicMock(
            return_value={
                "content": "Here is a thoughtful answer about Paris.",
                "thinking": "Let me reason about this carefully.",
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
        terminal_call = None
        for call in handler._sse_chunk.call_args_list:
            if call[1].get("finish_reason"):
                terminal_call = call
                break
        assert terminal_call is not None, "no terminal chunk was emitted"
        usage = terminal_call[1].get("usage")
        assert usage is not None, "terminal chunk missing usage block"
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert (
            usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        )
        handler._send_sse.assert_any_call("[DONE]")


class TestSSEConformance:
    """OpenAI SSE wire-format conformance for the streaming path."""

    @staticmethod
    def _make_handler():
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
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=30,
            browser_timeout=60,
        )
        return handler

    def _run_success(self, handler):
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

    def test_usage_only_on_terminal_chunk(self):
        """`usage` must ride only on the final chunk; the delta
        chunks emitted while polling carry no usage block."""
        handler = self._make_handler()

        def fake_stream(*_args, **_kwargs):
            # The delta chunks _stream_web_chat emits while polling.
            handler._sse_chunk("chatcmpl-1", 1, "gpt-4", {"content": "Hello"})
            handler._sse_chunk("chatcmpl-1", 1, "gpt-4", {"content": " world"})
            return {
                "content": "Hello world",
                "thinking": None,
                "done": True,
                "stop_reason": "site_done",
            }

        handler._stream_web_chat = MagicMock(side_effect=fake_stream)
        self._run_success(handler)

        chunk_calls = handler._sse_chunk.call_args_list
        assert len(chunk_calls) == 3
        with_usage = [c for c in chunk_calls if c[1].get("usage") is not None]
        assert len(with_usage) == 1
        terminal = with_usage[0]
        assert terminal is chunk_calls[-1]
        assert terminal[1]["finish_reason"] == "stop"
        assert terminal[1]["usage"]["prompt_tokens"] > 0
        assert terminal[1]["usage"]["completion_tokens"] > 0
        # Intermediate delta chunks: no usage, no finish_reason.
        for call in chunk_calls[:-1]:
            assert "usage" not in call[1]
            assert call[1].get("finish_reason") is None

    def test_terminal_chunk_precedes_done_sentinel(self):
        """Wire order: delta chunks, then the terminal chunk
        (finish_reason set), then the [DONE] sentinel -- last."""
        handler = self._make_handler()
        events = []
        handler._sse_chunk = MagicMock(
            side_effect=lambda *a, **k: events.append(("chunk", k.get("finish_reason")))
        )
        handler._send_sse = MagicMock(
            side_effect=lambda data: events.append(("sse", data))
        )

        def fake_stream(*_args, **_kwargs):
            handler._sse_chunk("chatcmpl-1", 1, "gpt-4", {"content": "Hello"})
            return {
                "content": "Hello",
                "thinking": None,
                "done": True,
                "stop_reason": "site_done",
            }

        handler._stream_web_chat = MagicMock(side_effect=fake_stream)
        self._run_success(handler)

        assert events, "no SSE events were emitted"
        assert events[-1] == ("sse", "[DONE]")
        terminal_idx = max(i for i, e in enumerate(events) if e[0] == "chunk" and e[1])
        assert terminal_idx < len(events) - 1
        # Every chunk before the terminal one is a plain delta.
        for event in events[:terminal_idx]:
            if event[0] == "chunk":
                assert event[1] is None

    def test_reasoning_content_mirrors_thinking_delta(self):
        """thinking deltas are mirrored byte-for-byte as
        reasoning_content, and concatenated deltas reconstruct each
        stream without duplication."""
        handler = self._make_handler()

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
                "thinking": "Let me",
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
        state = {"i": 0}

        def mock_capture(*_args, **_kwargs):
            idx = min(state["i"], len(responses) - 1)
            state["i"] += 1
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
        thinking_parts = []
        content_parts = []
        for call in handler._sse_chunk.call_args_list:
            delta = call[0][3]
            if "thinking" in delta:
                assert delta["reasoning_content"] == delta["thinking"]
                thinking_parts.append(delta["thinking"])
            if "content" in delta:
                content_parts.append(delta["content"])
        assert "".join(thinking_parts) == "Let me think"
        assert "".join(content_parts) == "Answer"


class TestStreamNonStreamParity:
    """Streaming and non-streaming paths must honor one contract."""

    @staticmethod
    def _make_non_streaming_handler():
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

    @staticmethod
    def _make_streaming_handler():
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
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=30,
            browser_timeout=60,
        )
        return handler

    def test_inject_failure_same_error_contract(self):
        """A failed inject/submit answers with 502, error type
        server_error and code inject_submit_failed on both paths."""
        # Non-streaming path.
        ns_handler = self._make_non_streaming_handler()
        with patch("sbsllm.server.inject_and_submit") as mock_submit:
            mock_submit.return_value = {
                "tab": 1,
                "inject": "NO_INPUT",
                "submit": None,
            }
            with (
                patch("sbsllm.server.inject_prompt"),
                patch("sbsllm.server.submit_js"),
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
                OpenAIHandler._handle_chat_completions(ns_handler)
        ns_handler._send_error.assert_called_once()
        ns_args = ns_handler._send_error.call_args
        assert ns_args[0][0] == 502
        assert ns_args[0][2] == "server_error"
        assert ns_args.kwargs.get("code") == "inject_submit_failed"

        # Streaming path.
        s_handler = self._make_streaming_handler()
        s_handler._inject_and_submit_with_recovery = MagicMock(
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
            s_handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        s_handler._send_error.assert_called_once()
        s_args = s_handler._send_error.call_args
        assert s_args[0][0] == 502
        assert s_args[0][2] == "server_error"
        assert s_args.kwargs.get("code") == "inject_submit_failed"

    def test_timeout_with_content_same_finish_reason(self):
        """A truncated answer reports finish_reason="length" on both
        paths: streaming terminal chunk vs non-streaming 200 response."""
        partial = "The capital of France is Pa"

        # Streaming: terminal chunk carries finish_reason="length".
        s_handler = self._make_streaming_handler()
        s_handler._stream_web_chat = MagicMock(
            return_value={
                "content": partial,
                "thinking": None,
                "done": False,
                "stop_reason": "budget_exhausted",
            }
        )
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value={"found": False, "count": 0},
            ),
        ):
            s_handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        terminal = None
        for call in s_handler._sse_chunk.call_args_list:
            if call[1].get("finish_reason"):
                terminal = call
                break
        assert terminal is not None
        assert terminal[1]["finish_reason"] == "length"

        # Non-streaming: 200 with the same finish_reason.
        ns_handler = self._make_non_streaming_handler()
        patches = (
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
            patch(
                "sbsllm.server.wait_for_response",
                return_value={
                    "found": True,
                    "content": partial,
                    "thinking": None,
                    "timed_out": True,
                    "done": False,
                },
            ),
            patch(
                "sbsllm.server.get_page_snapshot",
                return_value={"url": "http://x", "title": "t", "text_preview": ""},
            ),
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
            OpenAIHandler._handle_chat_completions(ns_handler)

        ns_handler._send_error.assert_not_called()
        ns_handler._send_json.assert_called_once()
        data = ns_handler._send_json.call_args[0][1]
        assert data["choices"][0]["message"]["content"] == partial
        assert data["choices"][0]["finish_reason"] == "length"


class TestSettingsPlumbing:
    """Runtime settings must survive the Server -> _SBSHTTPServer hop.

    Handlers read settings off `self.server`, so a setting added to
    Server.__init__ but not published in Server.start (or not defaulted
    in _SBSHTTPServer.__init__) silently reverts to the module default.
    """

    RUNTIME_SETTINGS: ClassVar[dict[str, float]] = {
        "browser_timeout": 123,
        "browser_lock_timeout": 234,
        "response_idle_timeout": 1.5,
        "response_done_confirm": 0.25,
        "busy_patience": 2.5,
        "thinking_patience": 3.5,
        "first_token_timeout": 4.5,
        "keepalive_interval": 5.5,
        "poll_interval": 0.05,
        "duplicate_prompt_cooldown": 6.5,
    }

    def test_create_server_signature_matches_server_init(self):
        """The factory exposes exactly Server.__init__'s parameters."""
        server_params = set(inspect.signature(Server.__init__).parameters)
        factory_params = set(inspect.signature(create_server).parameters)
        assert server_params - {"self"} == factory_params

    def test_start_publishes_settings_to_http_server(self):
        """Every runtime setting is published onto the HTTP server the
        handlers read from, with the configured value."""
        server = create_server(port=0, **self.RUNTIME_SETTINGS)
        mock_http_server = MagicMock()
        mock_http_server.handle_request.side_effect = [
            socket.timeout,
            KeyboardInterrupt(),
        ]
        with patch("sbsllm.server._SBSHTTPServer", return_value=mock_http_server):
            thread = threading.Thread(target=server.start, daemon=True)
            thread.start()
            server.wait_until_ready(5)
            server.stop()
            thread.join(timeout=2)
        for name, value in self.RUNTIME_SETTINGS.items():
            assert getattr(mock_http_server, name) == value, name

    def test_http_server_defaults_match_module_constants(self):
        """A fresh _SBSHTTPServer defaults every runtime setting to
        its module-level DEFAULT_ constant."""
        from sbsllm.server import (
            DEFAULT_BROWSER_LOCK_TIMEOUT,
            DEFAULT_BROWSER_TIMEOUT,
            DEFAULT_BUSY_PATIENCE,
            DEFAULT_DUPLICATE_PROMPT_COOLDOWN,
            DEFAULT_FIRST_TOKEN_TIMEOUT,
            DEFAULT_KEEPALIVE_INTERVAL,
            DEFAULT_POLL_INTERVAL,
            DEFAULT_RESPONSE_DONE_CONFIRM,
            DEFAULT_RESPONSE_IDLE_TIMEOUT,
            DEFAULT_THINKING_PATIENCE,
            _SBSHTTPServer,
        )

        httpd = _SBSHTTPServer(("127.0.0.1", 0), OpenAIHandler)
        try:
            assert httpd.browser_timeout == DEFAULT_BROWSER_TIMEOUT
            assert httpd.browser_lock_timeout == DEFAULT_BROWSER_LOCK_TIMEOUT
            assert httpd.response_idle_timeout == DEFAULT_RESPONSE_IDLE_TIMEOUT
            assert httpd.response_done_confirm == DEFAULT_RESPONSE_DONE_CONFIRM
            assert httpd.busy_patience == DEFAULT_BUSY_PATIENCE
            assert httpd.thinking_patience == DEFAULT_THINKING_PATIENCE
            assert httpd.first_token_timeout == DEFAULT_FIRST_TOKEN_TIMEOUT
            assert httpd.keepalive_interval == DEFAULT_KEEPALIVE_INTERVAL
            assert httpd.poll_interval == DEFAULT_POLL_INTERVAL
            assert httpd.duplicate_prompt_cooldown == DEFAULT_DUPLICATE_PROMPT_COOLDOWN
            assert isinstance(httpd.turn_registry, _TurnRegistry)
            assert isinstance(httpd.model_locks, _ModelLockRegistry)
        finally:
            httpd.server_close()


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
        # BrowserOperationTimeout means the Playwright worker thread is
        # wedged and every later browser call will fail too; the message
        # names the site and warns of a stuck worker thread.
        msg = handler._send_error.call_args[0][1].lower()
        assert "chatgpt" in msg
        assert "unresponsive" in msg

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


class TestTurnRegistry:
    """Unit tests for the duplicate-submit guard's storage."""

    def _record(self, registry, model="gpt-4", prompt="hello"):
        registry.record(model, prompt)

    def test_fresh_registry_never_duplicate(self):
        registry = _TurnRegistry()
        assert registry.is_duplicate("gpt-4", "hello", 60.0) is False

    def test_same_prompt_is_duplicate(self):
        registry = _TurnRegistry()
        self._record(registry)
        assert registry.is_duplicate("gpt-4", "hello", 60.0) is True

    def test_different_prompt_is_not_duplicate(self):
        registry = _TurnRegistry()
        self._record(registry)
        assert registry.is_duplicate("gpt-4", "goodbye", 60.0) is False

    def test_scope_is_per_model(self):
        registry = _TurnRegistry()
        self._record(registry)
        assert registry.is_duplicate("claude-3", "hello", 60.0) is False

    def test_cooldown_zero_disables_guard(self):
        registry = _TurnRegistry()
        self._record(registry)
        assert registry.is_duplicate("gpt-4", "hello", 0.0) is False

    def test_entry_expires_after_cooldown(self):
        registry = _TurnRegistry()
        clock = FakeClock()
        with patch("sbsllm.server.time") as fake_time:
            fake_time.monotonic.side_effect = clock.monotonic
            self._record(registry)  # recorded at t=1000
            # No retry for 61s: the entry ages out.
            clock.t = 1000.0 + 61.0
            assert registry.is_duplicate("gpt-4", "hello", 60.0) is False

    def test_duplicate_slides_cooldown_window(self):
        """A retry train cannot out-wait a fixed window: each suppressed
        duplicate slides the window forward, so the prompt is never
        re-sent for as long as retries keep arriving."""
        registry = _TurnRegistry()
        clock = FakeClock()
        with patch("sbsllm.server.time") as fake_time:
            fake_time.monotonic.side_effect = clock.monotonic
            self._record(registry)  # recorded at t=1000
            clock.t = 1050.0
            # Retries 50s apart forever: every one lands inside the window
            # that the previous retry slid forward.
            for _ in range(5):
                assert registry.is_duplicate("gpt-4", "hello", 60.0) is True
                clock.t += 50.0
            # Once the train stops, the entry ages out after one cooldown.
            clock.t += 61.0
            assert registry.is_duplicate("gpt-4", "hello", 60.0) is False


class TestDuplicatePromptSuppression:
    """A client retry train must not re-post the same prompt to a tab.

    One POST = one inject+submit, but retrying clients POST the identical
    request again after a failure (SSE reconnect, fetch/proxy retry), which
    re-sends the prompt -- the reported repeating-loop bug. A duplicate
    within `duplicate_prompt_cooldown` attaches to the turn already on
    screen instead: its finished answer is returned directly, and an
    in-flight one is followed without re-injecting.
    """

    COOLDOWN: ClassVar[float] = 60.0
    ATTACHED: ClassVar[dict] = {
        "found": True,
        "content": "the answer",
        "thinking": None,
        "busy": False,
        "done": True,
        "count": 1,
    }
    IN_FLIGHT: ClassVar[dict] = {
        "found": False,
        "content": "",
        "thinking": None,
        "busy": True,
        "done": False,
        "count": 1,
    }

    def _make_streaming_handler(self, registry, **server_attrs):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.model_map = {"gpt-4": "chatgpt"}
        handler.tab_map = {"gpt-4": MagicMock()}
        handler._inject_and_submit_with_recovery = MagicMock(
            return_value=({"inject": "OK", "submit": "OK"}, MagicMock())
        )
        handler._server_setting = MagicMock(return_value=self.COOLDOWN)
        handler._acquire_browser_lock = MagicMock(
            return_value=(threading.Lock(), True, 30.0)
        )
        handler._release_browser_lock = MagicMock()
        handler._send_error = MagicMock()
        handler._send_sse = MagicMock(side_effect=lambda x: x)
        handler._send_sse_headers = MagicMock()
        handler._send_sse_comment = MagicMock()
        handler._sse_chunk = MagicMock()
        handler._stream_web_chat = MagicMock(
            return_value={
                "content": "the answer",
                "thinking": None,
                "done": True,
                "stop_reason": "site_done",
            }
        )
        handler.wfile = MagicMock()
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=30,
            browser_timeout=60,
            turn_registry=registry,
            duplicate_prompt_cooldown=self.COOLDOWN,
            **server_attrs,
        )
        return handler

    def _call_streaming(self, handler, prompt="hi"):
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value=dict(self.ATTACHED),
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-1", "gpt-4", "chatgpt", MagicMock(), 1, prompt, 0.0, 60.0
            )

    def test_streaming_duplicate_attaches_without_resubmitting(self):
        registry = _TurnRegistry()
        handler = self._make_streaming_handler(registry)
        self._call_streaming(handler, "hi")
        assert handler._inject_and_submit_with_recovery.call_count == 1
        assert registry.is_duplicate("gpt-4", "hi", self.COOLDOWN) is True

        self._call_streaming(handler, "hi")
        # The retry must NOT re-inject -- one submission total.
        assert handler._inject_and_submit_with_recovery.call_count == 1
        # And the retry still got the answer that is on screen: it sent its
        # own SSE stream (headers once per request) ending in [DONE] with
        # the attached answer as a delta.
        assert handler._send_sse_headers.call_count == 2
        done_calls = [
            call
            for call in handler._send_sse.call_args_list
            if call.args[0] == "[DONE]"
        ]
        assert done_calls, "stream must end with [DONE]"
        content_chunks = [
            call
            for call in handler._sse_chunk.call_args_list
            if call.args[3].get("content") == "the answer"
        ]
        assert content_chunks, "attached answer must be streamed"

    def test_streaming_new_prompt_submits_again(self):
        registry = _TurnRegistry()
        handler = self._make_streaming_handler(registry)
        self._call_streaming(handler, "hi")
        self._call_streaming(handler, "different question")
        assert handler._inject_and_submit_with_recovery.call_count == 2

    def test_streaming_cooldown_zero_disables_guard(self):
        registry = _TurnRegistry()
        handler = self._make_streaming_handler(registry)
        handler._server_setting = MagicMock(return_value=0)
        self._call_streaming(handler, "hi")
        self._call_streaming(handler, "hi")
        assert handler._inject_and_submit_with_recovery.call_count == 2

    def test_streaming_duplicate_in_flight_follows_turn(self):
        registry = _TurnRegistry()
        handler = self._make_streaming_handler(registry)
        # First request submits and records.
        self._call_streaming(handler, "hi")
        assert handler._inject_and_submit_with_recovery.call_count == 1

        # Retry arrives while the turn is still generating: no re-inject,
        # the response is followed via the poller instead.
        handler._stream_web_chat.reset_mock()
        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT"),
            patch(
                "sbsllm.server.capture_response",
                return_value=dict(self.IN_FLIGHT),
            ),
        ):
            handler._handle_streaming_chat_completions(
                "req-2", "gpt-4", "chatgpt", MagicMock(), 1, "hi", 0.0, 60.0
            )
        assert handler._inject_and_submit_with_recovery.call_count == 1
        assert handler._stream_web_chat.call_count == 1
        baseline = handler._stream_web_chat.call_args.args[2]
        assert baseline["busy"] is True

    def _make_nonstreaming_handler(self, registry, prompt="hello"):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": prompt}]}
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
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=10,
            browser_timeout=60,
            turn_registry=registry,
            duplicate_prompt_cooldown=self.COOLDOWN,
        )
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        return handler

    def _call_nonstreaming(self, handler):
        with (
            patch("sbsllm.server.inject_prompt", return_value="inject_js"),
            patch("sbsllm.server.submit_js", return_value="submit_js"),
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch("sbsllm.server.check_page_health", return_value=True),
            patch("sbsllm.server.capture_response", return_value=dict(self.ATTACHED)),
            patch(
                "sbsllm.server.inject_and_submit",
                return_value={"tab": 1, "inject": "OK", "submit": "OK"},
            ) as mock_submit,
            patch(
                "sbsllm.server.wait_for_response",
                return_value={
                    "found": True,
                    "content": "the answer",
                    "thinking": None,
                    "done": True,
                    "count": 1,
                },
            ),
        ):
            handler._handle_chat_completions()
            return mock_submit.call_count

    def test_nonstreaming_duplicate_attaches_without_resubmitting(self):
        registry = _TurnRegistry()
        first = self._make_nonstreaming_handler(registry, "hello")
        submits = self._call_nonstreaming(first)
        assert submits == 1
        assert first._send_json.call_args[0][0] == 200

        second = self._make_nonstreaming_handler(registry, "hello")
        submits += self._call_nonstreaming(second)
        # The retry must NOT re-submit -- one submission total.
        assert submits == 1
        # And the retry still received the answer on screen.
        assert second._send_json.call_args[0][0] == 200
        assert "the answer" in json.dumps(second._send_json.call_args[0][1])

    def test_nonstreaming_new_prompt_submits_again(self):
        registry = _TurnRegistry()
        first = self._make_nonstreaming_handler(registry, "hello")
        submits = self._call_nonstreaming(first)
        second = self._make_nonstreaming_handler(registry, "a different question")
        submits += self._call_nonstreaming(second)
        assert submits == 2


class TestMultiModel:
    """The virtual `multi` model fans a prompt out to every tab."""

    BASELINE: ClassVar[dict] = {
        "found": False,
        "content": "",
        "thinking": None,
        "busy": False,
        "done": False,
        "count": 0,
    }

    @staticmethod
    def _page(name):
        page = MagicMock()
        page.is_closed.return_value = False
        page.name = name
        return page

    def _make_handler(self, model_map, tab_map, stream=False):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        body = json.dumps(
            {
                "model": "multi",
                "messages": [{"role": "user", "content": "hello"}],
                **({"stream": True} if stream else {}),
            }
        ).encode()
        handler.headers = {
            "Content-Length": str(len(body)),
            "x-request-id": "test-request-id",
        }
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body
        handler.model_map = model_map
        handler.tab_map = tab_map
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=10,
            browser_timeout=60,
        )
        handler._send_json = MagicMock()
        handler._send_error = MagicMock()
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = MagicMock()
        return handler

    def test_server_registers_multi_model(self):
        server = create_server(
            model_map={"chatgpt": "chatgpt", "claude": "claude"},
            tab_map={},
        )
        assert server.model_map["multi"] == "multi"
        single = create_server(model_map={"chatgpt": "chatgpt"}, tab_map={})
        assert "multi" not in single.model_map

    def test_multi_aggregates_every_answer(self):
        handler = self._make_handler(
            {"chatgpt": "chatgpt", "claude": "claude", "multi": "multi"},
            {"chatgpt": self._page("chatgpt"), "claude": self._page("claude")},
        )
        outcomes = {
            id(handler.tab_map["chatgpt"]): {
                "found": True,
                "content": "GPT answer",
                "thinking": None,
                "done": True,
                "count": 1,
            },
            id(handler.tab_map["claude"]): {
                "found": True,
                "content": "Claude answer",
                "thinking": "Claude reasoning",
                "done": True,
                "count": 1,
            },
        }

        def fake_wait_by_page(page, extraction, _timeout, **kwargs):
            return dict(outcomes[id(page)])

        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch("sbsllm.server.check_page_health", return_value=True),
            patch(
                "sbsllm.server.capture_response",
                return_value=dict(self.BASELINE),
            ),
            patch(
                "sbsllm.server.inject_and_submit",
                return_value={"tab": 1, "inject": "OK", "submit": "OK"},
            ),
            patch(
                "sbsllm.server.wait_for_response",
                side_effect=fake_wait_by_page,
            ),
        ):
            handler._handle_chat_completions()

        assert handler._send_json.call_args[0][0] == 200
        response = handler._send_json.call_args[0][1]
        assert response["model"] == "multi"
        assert response["choices"][0]["finish_reason"] == "stop"
        content = response["choices"][0]["message"]["content"]
        assert content == ("### chatgpt\n\nGPT answer\n\n### claude\n\nClaude answer")
        assert response["choices"][0]["message"]["reasoning_content"] == (
            "### claude\n\nClaude reasoning"
        )
        assert response["usage"]["prompt_tokens"] > 0
        assert response["usage"]["completion_tokens"] > 0

    def test_multi_failure_fails_whole_request(self):
        handler = self._make_handler(
            {"chatgpt": "chatgpt", "claude": "claude", "multi": "multi"},
            {"chatgpt": self._page("chatgpt"), "claude": self._page("claude")},
        )
        outcomes = {
            id(handler.tab_map["chatgpt"]): {
                "found": False,
                "content": "",
                "timed_out": True,
            },
            id(handler.tab_map["claude"]): {
                "found": True,
                "content": "Claude answer",
                "done": True,
                "count": 1,
            },
        }

        def fake_wait_by_page(page, extraction, _timeout, **kwargs):
            return dict(outcomes[id(page)])

        with (
            patch("sbsllm.server.extract_js", return_value="EXTRACT_JS"),
            patch("sbsllm.server.check_page_health", return_value=True),
            patch(
                "sbsllm.server.capture_response",
                return_value=dict(self.BASELINE),
            ),
            patch(
                "sbsllm.server.inject_and_submit",
                return_value={"tab": 1, "inject": "OK", "submit": "OK"},
            ),
            patch(
                "sbsllm.server.wait_for_response",
                side_effect=fake_wait_by_page,
            ),
        ):
            handler._handle_chat_completions()

        assert handler._send_json.call_count == 0
        assert handler._send_error.call_args[0][0] == 504
        assert "failed on chatgpt" in handler._send_error.call_args[0][1]

    def test_multi_streaming_rejected(self):
        handler = self._make_handler(
            {"chatgpt": "chatgpt", "claude": "claude", "multi": "multi"},
            {"chatgpt": self._page("chatgpt"), "claude": self._page("claude")},
            stream=True,
        )
        handler._handle_chat_completions()
        assert handler._send_error.call_args[0][0] == 400
        assert "stream" in handler._send_error.call_args[0][1].lower()

    def test_multi_requires_two_tabs(self):
        handler = self._make_handler(
            {"chatgpt": "chatgpt", "multi": "multi"},
            {"chatgpt": self._page("chatgpt")},
        )
        handler._handle_chat_completions()
        assert handler._send_error.call_args[0][0] == 502

    def test_acquire_all_browser_locks(self):
        handler = OpenAIHandler.__new__(OpenAIHandler)
        handler.server = types.SimpleNamespace(
            model_locks=_ModelLockRegistry(),
            browser_lock_timeout=10,
        )
        locks, acquired, _timeout = handler._acquire_all_browser_locks(["a", "b"])
        assert acquired is True
        assert len(locks) == 2
        # A second acquisition of the held locks fails and releases
        # nothing it does not hold.
        handler.server.browser_lock_timeout = 0.01
        locks2, acquired2, _ = handler._acquire_all_browser_locks(["a", "b"])
        assert acquired2 is False
        assert locks2 == []
        for browser_lock in locks:
            browser_lock.release()
