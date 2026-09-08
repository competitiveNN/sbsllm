"""OpenAI-compatible server for sbsllm.

Exposes a local HTTP server that accepts OpenAI-format chat completion requests
and routes them to the appropriate chat website via browser automation.
"""

from __future__ import annotations

import json
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

from .browser import BrowserError, ensure_qutebrowser, inject_and_submit
from .inject import inject_prompt, submit_js
from .sites import get_site, list_sites

# Default server settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


class OpenAIHandler(BaseHTTPRequestHandler):
    """Handler for OpenAI-compatible chat completion requests."""

    # Class-level config (set by Server)
    qb_bin: str = "qutebrowser"
    model_map: dict[str, str] = {}
    tab_map: dict[str, int] = {}

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging to keep output clean."""
        pass

    def _send_json(self, status: int, data: dict) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error(self, status: int, message: str, error_type: str = "invalid_request_error") -> None:
        """Send an error response in OpenAI format."""
        self._send_json(status, {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        })

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/v1/models":
            self._handle_models()
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_error(404, f"Not found: {self.path}", "not_found")

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        else:
            self._send_error(404, f"Not found: {self.path}", "not_found")

    def _handle_models(self) -> None:
        """Return available models."""
        models = []
        for model_id, site_id in self.model_map.items():
            models.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "sbsllm",
                "permission": [{
                    "id": f"modelperm-{model_id}",
                    "object": "model_permission",
                    "created": int(time.time()),
                    "allow_create_engine": False,
                    "allow_sampling": True,
                    "allow_logprobs": False,
                    "allow_search_indices": False,
                    "allow_view": True,
                    "allow_fine_tuning": False,
                    "organization": "*",
                    "group": None,
                    "is_blocking": False,
                }],
                "root": site_id,
                "parent": None,
            })
        self._send_json(200, {"object": "list", "data": models})

    def _handle_chat_completions(self) -> None:
        """Handle a chat completion request."""
        # Read and parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_error(400, "Request body is empty")
            return

        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return

        # Extract model
        model = data.get("model")
        if not model:
            self._send_error(400, "Missing 'model' field")
            return

        # Map model to site
        site_id = self.model_map.get(model)
        if not site_id:
            available = ", ".join(self.model_map.keys())
            self._send_error(
                400,
                f"Unknown model: {model!r}. Available: {available}",
                "invalid_request_error",
            )
            return

        # Extract user messages
        messages = data.get("messages", [])
        if not messages:
            self._send_error(400, "Missing 'messages' field")
            return

        # Build prompt from messages
        prompt = self._build_prompt(messages)
        if not prompt.strip():
            self._send_error(400, "No user message content found")
            return

        # Get tab index for this model
        tab_index = self.tab_map.get(model, 1)

        # Send prompt to chat site
        try:
            inject_js = inject_prompt(site_id, prompt)
            submit_js_val = submit_js(site_id)
            status = inject_and_submit(self.qb_bin, tab_index, inject_js, submit_js_val)
        except BrowserError as e:
            self._send_error(502, f"Browser error: {e}", "server_error")
            return
        except Exception as e:
            self._send_error(500, f"Internal error: {e}", "server_error")
            return

        # Build response
        if status["inject"] == "OK" and status["submit"] == "OK":
            content = f"Prompt sent to {site_id} successfully."
        elif status["inject"] != "OK":
            content = f"Failed to inject prompt: {status['inject']}"
        else:
            content = f"Failed to submit: {status['submit']}"

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        self._send_json(200, response)

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build a prompt string from OpenAI messages format."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = " ".join(text_parts)
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        return "\n\n".join(parts)


class Server:
    """OpenAI-compatible HTTP server."""

    def __init__(
        self,
        qb_bin: str = "qutebrowser",
        model_map: dict[str, str] | None = None,
        tab_map: dict[str, int] | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ):
        self.qb_bin = qb_bin
        self.model_map = model_map or {}
        self.tab_map = tab_map or {}
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None

    def start(self) -> None:
        """Start the server."""
        # Configure handler class
        OpenAIHandler.qb_bin = self.qb_bin
        OpenAIHandler.model_map = self.model_map
        OpenAIHandler.tab_map = self.tab_map

        self._server = HTTPServer((self.host, self.port), OpenAIHandler)
        print(f"sbsllm server listening on http://{self.host}:{self.port}")
        print(f"Models: {list(self.model_map.keys())}")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            self._server = None


def create_server(
    qb_bin: str = "qutebrowser",
    model_map: dict[str, str] | None = None,
    tab_map: dict[str, int] | None = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> Server:
    """Create a new server instance."""
    return Server(
        qb_bin=qb_bin,
        model_map=model_map or {},
        tab_map=tab_map or {},
        host=host,
        port=port,
    )
