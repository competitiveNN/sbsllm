"""End-to-end integration test: real headless Chromium + real extraction JS.

These tests close the gap that live AI sites left: we can't drive z.ai, grok,
or google end-to-end because of CAPTCHA and login walls. Instead we serve a
mock chat page whose DOM mirrors the z.ai structure, then run the *real* sbsllm
injection/submit/extraction JS against it inside a real browser.

This tests the JS templates (inject, submit, extraction) against live browser
DOM APIs (innerText, getComputedStyle, getClientRects) without needing
real AI sites that require login/CAPTCHA.
"""

from __future__ import annotations

import http.server
import socketserver
import threading

import pytest

from sbsllm.inject import extract_js, inject_prompt, submit_js
from sbsllm.sites import SITES, _response_js

pw = pytest.importorskip("playwright.sync_api")

_ZAI = SITES["zai"]

_GROK = SITES["grok"]


@pytest.fixture(autouse=True)
def _isolated_chrome_profile(monkeypatch, tmp_path_factory):
    """Give every test in this module its own Chromium user-data directory.

    The integration tests launch real headless Chromium. Two instances
    locking the same profile is the classic "could not connect to browser"
    hang, so without isolation the suite flakes whenever a previous test's
    browser did not shut down cleanly. The default path is untouched for
    the CLI; only tests see a private directory.
    """
    profile = tmp_path_factory.mktemp("sbsllm-profile-")
    monkeypatch.setenv("SBSLLM_USER_DATA_DIR", str(profile))
    yield

_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock zai</title></head>
<body>
  <form id="chat-form">
    <textarea id="chat-input" placeholder="Ask anything"></textarea>
    <button type="submit" id="send-message-button">Send</button>
  </form>
  <div id="response-content-container">
    <div class="markdown-prose"></div>
  </div>
  <script>
    var sendBtn = document.getElementById('send-message-button');
    var form = document.getElementById('chat-form');
    var input = document.getElementById('chat-input');
    var container = document.querySelector('#response-content-container .markdown-prose');
    // Track submit count so tests can assert exactly-once delivery.
    window.__submitCount = 0;
    window.__submitPrompts = [];
    input.addEventListener('input', function() {
      sendBtn.disabled = !this.value.trim();
    });
    form.addEventListener('submit', function(e) {
      e.preventDefault();
      window.__submitCount++;
      window.__submitPrompts.push(input.value);
      var prompt = input.value;
      container.innerHTML = '<div class="thinking-chain-container"><div class="thinking-body"><p>Thinking: ' + prompt + '</p></div></div>';
      var stop = document.createElement('button');
      stop.setAttribute('aria-label', 'Stop generating');
      stop.textContent = 'Stop';
      container.parentNode.appendChild(stop);
      setTimeout(function() {
        container.innerHTML = '<p dir="auto">You asked: ' + prompt + '</p>';
        stop.remove();
      }, 500);
    });
  </script>
</body></html>
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_MOCK_PAGE_HTML.encode())

    def log_message(self, *args):  # silence
        pass


@pytest.fixture(scope="module")
def mock_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


class TestJSTemplates:
    """Test the actual JS templates against a real browser DOM."""

    def test_page_loads(self, mock_server):
        """The mock page must load with the expected DOM structure."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                has_input = page.evaluate("!!document.getElementById('chat-input')")
                has_form = page.evaluate("!!document.getElementById('chat-form')")
                has_container = page.evaluate(
                    "!!document.querySelector('#response-content-container .markdown-prose')"
                )
                assert has_input, "textarea should exist"
                assert has_form, "form should exist"
                assert has_container, "response container should exist"
            finally:
                browser.close()

    def test_inject_template_finds_textarea(self, mock_server):
        """The inject JS must find and return OK for the mock textarea."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                status = page.evaluate(_ZAI["inject"])
                assert status == "OK", f"inject should find textarea, got {status}"
            finally:
                browser.close()

    def test_inject_template_sets_value(self, mock_server):
        """The inject JS must set the textarea value via property setter."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                # First inject an empty value (template uses PROMPT_PLACEHOLDER)
                status = page.evaluate(_ZAI["inject"])
                assert status == "OK"

                # Now set a value directly.
                page.evaluate(
                    """
                    (function() {
                        const el = document.getElementById('chat-input');
                        const setter = Object.getOwnPropertyDescriptor(
                            HTMLTextAreaElement.prototype, 'value').set;
                        setter.call(el, 'hello world');
                        el.dataset.sbsllmInput = 'true';
                        return 'OK';
                    })()
                    """
                )
                val = page.evaluate("document.getElementById('chat-input').value")
                assert val == "hello world", f"textarea value: {val!r}"
            finally:
                browser.close()

    def test_submit_template_clicks_button(self, mock_server):
        """The submit JS must locate the button and trigger form submission."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                # Set up the input.
                page.evaluate(
                    """
                    (function() {
                        const el = document.getElementById('chat-input');
                        el.value = 'test prompt';
                        el.dataset.sbsllmInput = 'true';
                        return 'OK';
                    })()
                    """
                )
                status = page.evaluate(_ZAI["submit_js"])
                assert status == "OK", f"submit failed: {status}"

                # The form handler should have started generating.
                page.wait_for_timeout(600)
                # The mock page writes the thinking block first, then the answer.
                # Check that the form handler fired by looking for the answer.
                has_answer = page.evaluate(
                    "!!document.querySelector('#response-content-container .markdown-prose p')"
                )
                assert has_answer, "answer should appear after submit"
            finally:
                browser.close()

    def test_extraction_reads_answer(self, mock_server):
        """The extraction JS must correctly read the answer from markdown-prose."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                extraction = _response_js(
                    _ZAI["response_selectors"],
                    _ZAI["thinking_selectors"],
                    _ZAI["loading_selectors"],
                    _ZAI.get("login_wall_selectors", []),
                )

                # Simulate a completed response already in the DOM.
                page.evaluate(
                    """
                    document.querySelector('#response-content-container .markdown-prose').innerHTML =
                        '<p dir="auto">Hello from mock chat!</p>';
                    """
                )
                result = page.evaluate(extraction)
                assert result["content"] == "Hello from mock chat!", (
                    f"extracted: {result}"
                )
                assert result["found"] is True
                assert result["busy"] is False
                assert result["done"] is True
            finally:
                browser.close()

    def test_extraction_separates_thinking_from_answer(self, mock_server):
        """The extraction JS must prune thinking and keep answer separate."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                extraction = _response_js(
                    _ZAI["response_selectors"],
                    _ZAI["thinking_selectors"],
                    _ZAI["loading_selectors"],
                    _ZAI.get("login_wall_selectors", []),
                )

                # Put thinking + answer in the DOM (exactly the case we fixed).
                page.evaluate(
                    """
                    document.querySelector('#response-content-container .markdown-prose').innerHTML =
                        '<div class="thinking-chain-container"><div class="thinking-body"><p>Reasoning: 2+2=4</p></div></div>' +
                        '<p dir="auto">The answer is 4</p>';
                    """
                )
                result = page.evaluate(extraction)
                assert result["thinking"] is not None, "thinking should be extracted"
                assert (
                    "Reasoning" in result["thinking"]
                    or "reasoning" in result["thinking"].lower()
                )
                assert result["content"] == "The answer is 4", (
                    f"content should be answer only, got: {result['content']!r}"
                )
                assert "2+2" not in result["content"], "thinking leaked into content!"
            finally:
                browser.close()

    def test_loading_dots_signal_busy(self, mock_server):
        """Loading indicators must be detected as busy (z.ai dots case)."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                extraction = _response_js(
                    _ZAI["response_selectors"],
                    _ZAI["thinking_selectors"],
                    _ZAI["loading_selectors"],
                    _ZAI.get("login_wall_selectors", []),
                )

                # Set up a response with loading dots (z.ai animating during generation).
                page.evaluate(
                    """
                    document.querySelector('#response-content-container .markdown-prose').innerHTML =
                        '<div class="thinking-chain-container"><div class="thinking-body"><p>Computing...</p></div></div>';
                    """
                )
                # Add animated dots (z.ai style).
                page.evaluate(
                    """
                    const container = document.querySelector('#response-content-container');
                    container.appendChild(document.createElement('button')).setAttribute('aria-label', 'Stop generating');
                    """
                )

                result = page.evaluate(extraction)
                # Loading + thinking means busy.
                assert result["busy"] is True, f"expected busy, got: {result}"
            finally:
                browser.close()

    def test_full_inject_submit_extract_cycle(self, mock_server):
        """End-to-end: inject prompt, submit, let mock page generate, extract answer."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")

            try:
                prompt = "what is 2+2"

                # 1. Inject via the real z.ai template.
                inject_status = page.evaluate(_ZAI["inject"])
                assert inject_status == "OK"

                # 2. Set the prompt value (inject template uses placeholder).
                page.evaluate(
                    f"""
                    (function() {{
                        const el = document.getElementById('chat-input');
                        const setter = Object.getOwnPropertyDescriptor(
                            HTMLTextAreaElement.prototype, 'value').set;
                        setter.call(el, '{prompt}');
                        el.dispatchEvent(new Event('input', {{bubbles:true}}));
                        el.dispatchEvent(new Event('change', {{bubbles:true}}));
                        el.dataset.sbsllmInput = 'true';
                        return 'OK';
                    }})()
                    """
                )

                # 3. Submit via the real z.ai template.
                submit_status = page.evaluate(_ZAI["submit_js"])
                assert submit_status == "OK"

                # 4. Wait for the mock page to generate (it writes the answer after 500ms).
                page.wait_for_timeout(800)

                # 5. Extract response.
                extraction = _response_js(
                    _ZAI["response_selectors"],
                    _ZAI["thinking_selectors"],
                    _ZAI["loading_selectors"],
                    _ZAI.get("login_wall_selectors", []),
                )
                result = page.evaluate(extraction)

                assert result["found"] or result["content"], f"no content: {result}"
                assert prompt in result["content"], (
                    f"prompt not in answer: {result['content']!r}"
                )
                assert result["done"] is True, f"not done: {result}"
            finally:
                browser.close()


# --- Grok integration tests ---

_GROK = SITES["grok"]

_GROK_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock grok</title></head>
<body>
  <form id="chat-form">
    <div id="composer" class="tiptap ProseMirror" contenteditable="true" data-placeholder="Ask anything..."></div>
    <button data-testid="chat-submit" type="submit">Send</button>
  </form>
  <div id="response-container">
    <div class="message-bubble assistant">
      <div class="prose-chat"></div>
    </div>
  </div>
  <script>
    var form = document.getElementById('chat-form');
    var input = document.getElementById('composer');
    var responseDiv = document.querySelector('.prose-chat');
    var stopBtn = document.createElement('button');
    stopBtn.setAttribute('aria-label', 'Stop generating');
    var submitted = false;
    form.addEventListener('submit', function(e) {
      e.preventDefault();
      if (submitted) return;
      submitted = true;
      var prompt = input.textContent || '';
      responseDiv.innerHTML = '<p>' + prompt + '</p>';
      document.body.appendChild(stopBtn);
    });
  </script>
</body></html>
"""


class _GrokHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_GROK_MOCK_PAGE_HTML.encode())

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def mock_grok_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _GrokHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


class TestGrokIntegration:
    """Tests that the real grok inject/submit/extract JS works against a mock
    page mirroring grok's TipTap/ProseMirror contenteditable structure."""

    def test_grok_inject_finds_tiptap(self, mock_grok_server):
        """The grok inject JS must find the TipTap contenteditable div."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_server, wait_until="networkidle")
            try:
                inject_js = inject_prompt("grok", "test prompt")
                status = page.evaluate(inject_js)
                assert status == "OK", f"inject should find tiptap, got {status}"
            finally:
                browser.close()

    def test_grok_inject_sets_content(self, mock_grok_server):
        """TipTap contenteditable should receive the prompt text."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_server, wait_until="networkidle")
            try:
                inject_js = inject_prompt("grok", "hello grok")
                status = page.evaluate(inject_js)
                assert status == "OK"
                text = page.evaluate("document.getElementById('composer').textContent")
                assert "hello grok" in text, f"textContent: {text!r}"
            finally:
                browser.close()

    def test_grok_submit_clicks_button(self, mock_grok_server):
        """The grok submit JS must find and click the send button."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_server, wait_until="networkidle")
            try:
                # Set up input
                page.evaluate(inject_prompt("grok", "grok question"))
                status = page.evaluate(_GROK["submit_js"])
                assert status == "OK", f"submit failed: {status}"
                # The form's submit handler should have populated the response.
                page.wait_for_timeout(100)
                has_answer = page.evaluate("!!document.querySelector('.prose-chat p')")
                assert has_answer, "answer should appear after submit"
            finally:
                browser.close()

    def test_grok_full_cycle(self, mock_grok_server):
        """End-to-end: inject, submit, extract response from grok mock."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_server, wait_until="networkidle")
            try:
                prompt = "what is 2+2"
                # 1. Inject
                inject_status = page.evaluate(inject_prompt("grok", prompt))
                assert inject_status == "OK", f"inject: {inject_status}"
                # 2. Submit
                submit_status = page.evaluate(submit_js("grok"))
                assert submit_status == "OK", f"submit: {submit_status}"
                # 3. Extract
                extraction = _response_js(
                    _GROK["response_selectors"],
                    _GROK["thinking_selectors"],
                    _GROK["loading_selectors"],
                    _GROK.get("login_wall_selectors", []),
                )
                result = page.evaluate(extraction)
                assert result["found"] or result["content"], f"no content: {result}"
                assert prompt in result["content"], (
                    f"prompt not in answer: {result['content']!r}"
                )
            finally:
                browser.close()

    def test_grok_post_inject_clears_existing_content(self, mock_grok_server):
        """post_inject_js must clear existing editor content before inserting."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_server, wait_until="networkidle")
            try:
                # Pre-populate the composer with stale content.
                page.evaluate(
                    """
                    () => {
                        document.getElementById('composer').textContent =
                            'stale content from a previous turn';
                    }
                    """
                )
                # Run full inject (inject + post_inject combined).
                status = page.evaluate(inject_prompt("grok", "fresh prompt"))
                assert status == "OK"
                # post_inject should have cleared stale content and replaced it.
                text = page.evaluate("document.getElementById('composer').textContent")
                assert "fresh prompt" in text, f"textContent: {text!r}"
                assert "stale content" not in text, (
                    f"stale content should have been cleared: {text!r}"
                )
            finally:
                browser.close()

    def test_grok_post_inject_dispatches_input_events(self, mock_grok_server):
        """post_inject_js must dispatch beforeinput and input events so editors
        with internal state models (TipTap/ProseMirror) sync up."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_server, wait_until="networkidle")
            try:
                # Set up event counters before injection.
                page.evaluate(
                    """
                    () => {
                        window.__sbsllmEventCounts = { beforeinput: 0, input: 0 };
                        const el = document.getElementById('composer');
                        el.addEventListener('beforeinput', () => {
                            window.__sbsllmEventCounts.beforeinput++;
                        });
                        el.addEventListener('input', () => {
                            window.__sbsllmEventCounts.input++;
                        });
                    }
                    """
                )
                status = page.evaluate(inject_prompt("grok", "event test"))
                assert status == "OK"
                counts = page.evaluate("window.__sbsllmEventCounts")
                assert counts["beforeinput"] >= 1, (
                    f"beforeinput event not dispatched: {counts}"
                )
                assert counts["input"] >= 1, f"input event not dispatched: {counts}"
            finally:
                browser.close()


# --- Grok textarea fallback tests ---

_GROK_TEXTAREA_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock grok textarea</title></head>
<body>
  <form id="chat-form">
    <textarea id="grok-input" aria-label="Ask Grok anything" placeholder="Ask anything"></textarea>
    <button data-testid="chat-submit" type="submit">Send</button>
  </form>
  <div id="response-container">
    <div class="message-bubble assistant">
      <div class="prose-chat"></div>
    </div>
  </div>
  <script>
    var form = document.getElementById('chat-form');
    var input = document.getElementById('grok-input');
    var responseDiv = document.querySelector('.prose-chat');
    var stopBtn = document.createElement('button');
    stopBtn.setAttribute('aria-label', 'Stop generating');
    var submitted = false;
    // Enable submit button when input has value (mirrors real grok behavior)
    input.addEventListener('input', function() {
      document.querySelector('button[data-testid="chat-submit"]').disabled = !this.value.trim();
    });
    form.addEventListener('submit', function(e) {
      e.preventDefault();
      if (submitted) return;
      submitted = true;
      var prompt = input.value;
      responseDiv.innerHTML = '<p>' + prompt + '</p>';
      document.body.appendChild(stopBtn);
    });
  </script>
</body></html>
"""


class _GrokTextareaHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_GROK_TEXTAREA_MOCK_PAGE_HTML.encode())

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def mock_grok_textarea_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _GrokTextareaHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
        thread.join(timeout=5)


class TestGrokTextareaIntegration:
    """Tests that grok inject/submit/extract JS works when the input falls
    back to a plain textarea (non-contenteditable) instead of TipTap."""

    def test_grok_textarea_inject_sets_value(self, mock_grok_textarea_server):
        """The grok inject JS must find the textarea and set its value."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_textarea_server, wait_until="networkidle")
            try:
                inject_js = inject_prompt("grok", "textarea prompt")
                status = page.evaluate(inject_js)
                assert status == "OK", f"inject should find textarea, got {status}"
                val = page.evaluate("document.getElementById('grok-input').value")
                assert "textarea prompt" in val, f"textarea value: {val!r}"
            finally:
                browser.close()

    def test_grok_textarea_post_inject_preserves_value(self, mock_grok_textarea_server):
        """post_inject_js must NOT wipe the textarea value (regression: the old
        contenteditable-only post_inject did innerText='' + execCommand('insertText')
        which clobbered textarea values when sbsllmValue was unset)."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_textarea_server, wait_until="networkidle")
            try:
                page.evaluate(inject_prompt("grok", "preserved"))
                val = page.evaluate("document.getElementById('grok-input').value")
                assert "preserved" in val, f"value wiped by post_inject: {val!r}"
            finally:
                browser.close()

    def test_grok_textarea_full_cycle(self, mock_grok_textarea_server):
        """End-to-end: inject, submit, extract response from grok textarea mock."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_grok_textarea_server, wait_until="networkidle")
            try:
                prompt = "hello textarea grok"
                inject_status = page.evaluate(inject_prompt("grok", prompt))
                assert inject_status == "OK", f"inject: {inject_status}"
                submit_status = page.evaluate(submit_js("grok"))
                assert submit_status == "OK", f"submit: {submit_status}"
                page.wait_for_timeout(100)
                extraction = _response_js(
                    _GROK["response_selectors"],
                    _GROK["thinking_selectors"],
                    _GROK["loading_selectors"],
                    _GROK.get("login_wall_selectors", []),
                )
                result = page.evaluate(extraction)
                assert result["content"], f"no content: {result}"
                assert prompt in result["content"], (
                    f"prompt not in answer: {result['content']!r}"
                )
            finally:
                browser.close()


# --- Google integration tests ---

_GOOGLE = SITES["google"]

_GOOGLE_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock google</title></head>
<body>
  <ms-prompt-box>
    <ms-autosize-textarea data-value="">
      <textarea aria-label="Enter a prompt"></textarea>
    </ms-autosize-textarea>
    <ms-run-button>
      <button aria-label="Run" class="run-button" type="submit" disabled>Run</button>
    </ms-run-button>
  </ms-prompt-box>
  <ms-chat-turn>
    <div class="chat-turn-container model"></div>
  </ms-chat-turn>
  <script>
    var box = document.querySelector('ms-prompt-box');
    var taWrapper = document.querySelector('ms-autosize-textarea');
    var textarea = taWrapper.querySelector('textarea');
    var runBtn = document.querySelector('button.run-button');
    var chatTurn = document.querySelector('ms-chat-turn .chat-turn-container.model');
    textarea.addEventListener('input', function() {
      // Sync internal state when input fires.
      taWrapper.setAttribute('data-value', textarea.value);
      runBtn.disabled = !textarea.value.trim();
    });
    runBtn.addEventListener('click', function(e) {
      e.preventDefault();
      runBtn.disabled = true;
      chatTurn.innerHTML = '<p>' + textarea.value + '</p>';
    });
  </script>
</body></html>
"""


class _GoogleHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_GOOGLE_MOCK_PAGE_HTML.encode())

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def mock_google_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _GoogleHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


class TestGoogleIntegration:
    """Tests that the real google inject/submit/extract JS works against a mock
    page mirroring google AI Studio's ms-* web-component structure.

    The google bug: content was inserted into the textarea but the Run button
    stayed disabled because the ms-autosize-textarea wrapper's data-value was
    not synced. The post_inject_js fix addresses this."""

    def test_google_inject_finds_textarea(self, mock_google_server):
        """The google inject JS must find the ms-prompt-box textarea."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_server, wait_until="networkidle")
            try:
                inject_js = inject_prompt("google", "test prompt")
                status = page.evaluate(inject_js)
                assert status == "OK", f"inject should find textarea, got {status}"
            finally:
                browser.close()

    def test_google_post_inject_syncs_data_value(self, mock_google_server):
        """post_inject_js must sync ms-autosize-textarea data-value so Run button enables."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_server, wait_until="networkidle")
            try:
                inject_js = inject_prompt("google", "hello google")
                page.evaluate(inject_js)

                # The Run button should now be enabled after post_inject_js syncs data-value.
                is_disabled = page.evaluate(
                    "document.querySelector('button.run-button').disabled"
                )
                assert not is_disabled, (
                    "Run button should be enabled after inject + post_inject"
                )
            finally:
                browser.close()

    def test_google_submit_clicks_run(self, mock_google_server):
        """The google submit JS must find and click the Run button."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_server, wait_until="networkidle")
            try:
                page.evaluate(inject_prompt("google", "test prompt"))
                status = page.evaluate(_GOOGLE["submit_js"])
                assert status == "OK", f"submit failed: {status}"
                page.wait_for_timeout(100)
                has_answer = page.evaluate(
                    "!!document.querySelector('ms-chat-turn .chat-turn-container.model p')"
                )
                assert has_answer, "answer should appear after submit"
            finally:
                browser.close()

    def test_google_full_cycle(self, mock_google_server):
        """End-to-end: inject, submit, extract response from google mock."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_server, wait_until="networkidle")
            try:
                prompt = "explain quantum computing"
                page.evaluate(inject_prompt("google", prompt))
                submit_status = page.evaluate(submit_js("google"))
                assert submit_status == "OK", f"submit: {submit_status}"

                extraction = _response_js(
                    _GOOGLE["response_selectors"],
                    _GOOGLE["thinking_selectors"],
                    _GOOGLE["loading_selectors"],
                    _GOOGLE.get("login_wall_selectors", []),
                )
                result = page.evaluate(extraction)
                assert result["found"] or result["content"], f"no content: {result}"
                assert prompt in result["content"], (
                    f"prompt not in answer: {result['content']!r}"
                )
            finally:
                browser.close()


# --- Google Shadow DOM integration test ---

# Mock page that mirrors Google AI Studio's ms-* web components with Shadow DOM.
# The Run button lives inside ms-run-button's shadow root, which standard
# querySelector cannot reach. This reproduces the real-world DOM structure.
_GOOGLE_SHADOW_MOCK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock google shadow</title></head>
<body>
  <ms-prompt-box></ms-prompt-box>
  <ms-run-button></ms-run-button>
  <ms-chat-turn>
    <div class="chat-turn-container model"></div>
  </ms-chat-turn>
  <script>
    // Attach shadow roots and wire up behaviour to mirror Google AI Studio.
    (function() {
        var promptBox = document.querySelector('ms-prompt-box');
        var shadow1 = promptBox.attachShadow({mode: 'open'});
        var taWrapper = document.createElement('ms-autosize-textarea');
        taWrapper.setAttribute('data-value', '');
        var shadow2 = taWrapper.attachShadow({mode: 'open'});
        var textarea = document.createElement('textarea');
        textarea.setAttribute('aria-label', 'Enter a prompt');
        shadow2.appendChild(textarea);
        shadow1.appendChild(taWrapper);

        var runButtonHost = document.querySelector('ms-run-button');
        var runShadow = runButtonHost.attachShadow({mode: 'open'});
        var runBtn = document.createElement('button');
        runBtn.setAttribute('aria-label', 'Run');
        runBtn.className = 'run-button';
        runBtn.type = 'submit';
        runBtn.disabled = true;
        runShadow.appendChild(runBtn);

        textarea.addEventListener('input', function() {
            taWrapper.setAttribute('data-value', textarea.value);
            runBtn.disabled = !textarea.value.trim();
        });
        runBtn.addEventListener('click', function(e) {
            e.preventDefault();
            var modelDiv = document.querySelector('.chat-turn-container.model');
            modelDiv.innerHTML = '<p>' + textarea.value + '</p>';
        });
    })();
  </script>
</body></html>
"""


class _GoogleShadowHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_GOOGLE_SHADOW_MOCK_HTML.encode())

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def mock_google_shadow_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _GoogleShadowHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


class TestGoogleShadowDom:
    """Google AI Studio uses ms-* web components with Shadow DOM.
    The submit and post_inject JS must traverse shadowRoot to find the Run button."""

    def test_google_shadow_submit_clicks_run_button(self, mock_google_shadow_server):
        """submit_js must find the Run button inside ms-run-button's shadow root."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_shadow_server, wait_until="networkidle")
            try:
                page.evaluate(inject_prompt("google", "shadow test prompt"))
                status = page.evaluate(submit_js("google"))
                assert status == "OK", f"submit failed in shadow DOM: {status}"
                page.wait_for_timeout(100)
                has_answer = page.evaluate(
                    "!!document.querySelector('.chat-turn-container.model p')"
                )
                assert has_answer, "answer should appear after shadow DOM submit"
                answer = page.evaluate(
                    "document.querySelector('.chat-turn-container.model p').textContent"
                )
                assert "shadow test prompt" in answer, f"unexpected answer: {answer!r}"
            finally:
                browser.close()

    def test_google_shadow_post_inject_enables_button(self, mock_google_shadow_server):
        """post_inject_js must find the marker nested two shadow levels
        deep and enable the Run button inside the shadow root."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_shadow_server, wait_until="networkidle")
            try:
                page.evaluate(inject_prompt("google", "enable me"))
                status = page.evaluate(SITES["google"]["post_inject_js"])
                assert status == "OK", (
                    "post_inject must find the marker nested inside "
                    f"ms-autosize-textarea's shadow root, got {status}"
                )
                is_disabled = page.evaluate(
                    "document.querySelector('ms-run-button').shadowRoot"
                    ".querySelector('button.run-button').disabled"
                )
                assert not is_disabled, (
                    "Run button should be enabled after post_inject via shadow DOM"
                )
            finally:
                browser.close()


_GOOGLE_CHROME_MOCK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock google chrome</title></head>
<body>
  <ms-chat-turn>
    <div class="chat-turn-container model">
      <div class="turn-header">
        <span class="author-label">Model</span>
        <time>1:32 PM</time>
      </div>
      <ms-chat-turn-options>
        <button><span class="material-symbols-outlined">edit</span></button>
        <button><span class="material-symbols-outlined">more_vert</span></button>
      </ms-chat-turn-options>
      <div class="turn-content"><p>Hello! How can I help you today?</p></div>
      <div class="turn-footer">
        <button><span class="material-symbols-outlined">thumb_up</span></button>
        <button><span class="material-symbols-outlined">thumb_down</span></button>
      </div>
    </div>
  </ms-chat-turn>
</body></html>
"""

# Same turn, but without .turn-content: the fallback container
# selector matches the whole turn, so the chrome sits INSIDE the
# matched response and only response_exclude_selectors can prune it.
_GOOGLE_CHROME_NO_TURN_CONTENT_HTML = _GOOGLE_CHROME_MOCK_HTML.replace(
    '<div class="turn-content"><p>Hello! How can I help you today?</p></div>',
    "<p>Hello! How can I help you today?</p>",
)


class _GoogleChromeHandler(http.server.BaseHTTPRequestHandler):
    html = _GOOGLE_CHROME_MOCK_HTML

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(self.html.encode())

    def log_message(self, *args):
        pass


class _GoogleChromeNoTurnContentHandler(_GoogleChromeHandler):
    html = _GOOGLE_CHROME_NO_TURN_CONTENT_HTML


# A "Thinking" label rendered OUTSIDE any ms-thought-chunk (a status
# chip on its own line). Only the template's text safety net can remove
# it; the element prune will not.
_GOOGLE_THINKING_LABEL_LINE_MOCK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock google label line</title></head>
<body>
  <ms-chat-turn>
    <div class="chat-turn-container model">
      <div class="turn-content">
        <div class="thinking-status-chip">Thinking</div>
        <p>The shadows soften into light,</p>
        <p>A quiet stillness holds the room.</p>
      </div>
    </div>
  </ms-chat-turn>
</body></html>
"""


_GOOGLE_THINKING_ANSWER_PREFIX_MOCK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock google label prefix</title></head>
<body>
  <ms-chat-turn>
    <div class="chat-turn-container model">
      <div class="turn-content">
        <p>Thinking about the sun, a quiet glow on the wall.</p>
      </div>
    </div>
  </ms-chat-turn>
</body></html>
"""


_GOOGLE_THINKING_MOCK_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock google thinking</title></head>
<body>
  <ms-chat-turn>
    <div class="chat-turn-container model">
      <div class="turn-header">
        <span class="author-label">google</span>
        <time>1:32 PM</time>
      </div>
      <div class="turn-content">
        <ms-prompt-chunk>
          <ms-thought-chunk>
            <div class="thought-header">
              <span class="material-symbols-outlined">psychology</span>
              <span class="thought-label">Thinking</span>
            </div>
          </ms-thought-chunk>
        </ms-prompt-chunk>
        <ms-prompt-chunk>
          <ms-cmark-node class="cmark-node">
            <p>A whisper of wind through the open door,</p>
            <p>Sunlight pooling across the floor.</p>
          </ms-cmark-node>
        </ms-prompt-chunk>
      </div>
      <div class="turn-footer">
        <button><span class="material-symbols-outlined">thumb_up</span></button>
        <button><span class="material-symbols-outlined">thumb_down</span></button>
      </div>
    </div>
  </ms-chat-turn>
</body></html>
"""


class _GoogleThinkingHandler(_GoogleChromeHandler):
    html = _GOOGLE_THINKING_MOCK_HTML


class _GoogleThinkingLabelLineHandler(_GoogleChromeHandler):
    html = _GOOGLE_THINKING_LABEL_LINE_MOCK_HTML


class _GoogleThinkingAnswerPrefixHandler(_GoogleChromeHandler):
    html = _GOOGLE_THINKING_ANSWER_PREFIX_MOCK_HTML


@pytest.fixture(scope="module")
def mock_google_thinking_label_line_server():
    with socketserver.TCPServer(
        ("127.0.0.1", 0), _GoogleThinkingLabelLineHandler
    ) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def mock_google_thinking_answer_prefix_server():
    with socketserver.TCPServer(
        ("127.0.0.1", 0), _GoogleThinkingAnswerPrefixHandler
    ) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def mock_google_thinking_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _GoogleThinkingHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def mock_google_chrome_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _GoogleChromeHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def mock_google_chrome_no_turn_content_server():
    with socketserver.TCPServer(
        ("127.0.0.1", 0), _GoogleChromeNoTurnContentHandler
    ) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


class TestGoogleExtractionChrome:
    """Google AI Studio renders turn chrome (action icons, the
    model/timestamp header, feedback buttons) around the answer.
    The extraction must relay the answer, not the chrome."""

    CHROME = (
        "edit", "more_vert", "thumb_up", "thumb_down",
        "Model", "1:32 PM",
    )

    def test_turn_content_selector_scopes_to_the_answer(
        self, mock_google_chrome_server
    ):
        """The primary selector targets .turn-content, so the header,
        options menu and feedback bar outside it never reach the
        local chat."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_chrome_server, wait_until="networkidle")
            try:
                result = page.evaluate(extract_js("google"))
                assert result["found"] is True
                assert "Hello! How can I help you today?" in result["content"]
                for chrome in self.CHROME:
                    assert chrome not in result["content"], (
                        f"turn chrome {chrome!r} leaked into the answer: "
                        f"{result['content']!r}"
                    )
            finally:
                browser.close()

    def test_exclusions_prune_chrome_inside_the_container(
        self, mock_google_chrome_no_turn_content_server
    ):
        """When the fallback container selector matches (no
        .turn-content), response_exclude_selectors must prune the
        buttons, icon ligatures, options menu, footer and header
        from the response clone."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(
                mock_google_chrome_no_turn_content_server,
                wait_until="networkidle",
            )
            try:
                result = page.evaluate(extract_js("google"))
                assert result["found"] is True
                assert "Hello! How can I help you today?" in result["content"]
                for chrome in self.CHROME:
                    assert chrome not in result["content"], (
                        f"turn chrome {chrome!r} leaked into the answer: "
                        f"{result['content']!r}"
                    )
            finally:
                browser.close()

    def test_collapsed_thought_chunk_label_does_not_leak(
        self, mock_google_thinking_server
    ):
        """A collapsed ms-thought-chunk renders only its 'Thinking'
        label. The label must not reach the relayed answer, and a bare
        label must not be forwarded as thinking content."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_google_thinking_server, wait_until="networkidle")
            try:
                result = page.evaluate(extract_js("google"))
                assert result["found"] is True
                assert "A whisper of wind through the open door," in result[
                    "content"
                ]
                assert "Thinking" not in result["content"], (
                    f"thinking label leaked into the answer: "
                    f"{result['content']!r}"
                )
                assert not result["thinking"] or (
                    result["thinking"].strip() != "Thinking"
                ), f"bare thinking label forwarded: {result['thinking']!r}"
            finally:
                browser.close()

    def test_standalone_thinking_label_line_is_stripped(
        self, mock_google_thinking_label_line_server
    ):
        """A 'Thinking' label on its own line (status chip, not
        inside any thinking selector) is stripped by the template's
        safety net, while the answer lines after it are kept."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(
                mock_google_thinking_label_line_server,
                wait_until="networkidle",
            )
            try:
                result = page.evaluate(extract_js("google"))
                assert result["found"] is True
                assert "Thinking" not in result["content"], (
                    f"standalone thinking label leaked: {result['content']!r}"
                )
                assert "The shadows soften into light" in result["content"]
                assert "A quiet stillness holds the room" in result["content"]
            finally:
                browser.close()

    def test_answer_starting_with_thinking_word_is_preserved(
        self, mock_google_thinking_answer_prefix_server
    ):
        """An answer that merely starts with the word 'Thinking'
        (lowercase continuation) is NOT stripped; only a
        disclosure label is."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(
                mock_google_thinking_answer_prefix_server,
                wait_until="networkidle",
            )
            try:
                result = page.evaluate(extract_js("google"))
                assert result["found"] is True
                assert result["content"].startswith(
                    "Thinking about the sun"
                ), f"real answer truncated: {result['content']!r}"
            finally:
                browser.close()


_ZAI = SITES["zai"]


class TestZaiPostInject:
    """Tests for z.ai's post_inject_js.

    The post_inject_js no longer dispatches Enter key events (that caused a
    double-send: Enter triggered the form's submit handler, then submit_js
    clicked the send button again). It now only verifies the marked input
    exists and returns OK.
    """

    def test_zai_post_inject_returns_ok(self, mock_server):
        """post_inject_js must return OK when the marked input is present."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")
            try:
                # Set up the input and mark it, simulating what the main inject IIFE does.
                page.evaluate("""
                    (function() {
                        const el = document.getElementById('chat-input');
                        const setter = Object.getOwnPropertyDescriptor(
                            HTMLTextAreaElement.prototype, 'value').set;
                        setter.call(el, 'test');
                        el.dataset.sbsllmInput = 'true';
                        return 'OK';
                    })()
                """)

                # Run the post_inject_js block.
                post_inject = _ZAI["post_inject_js"]
                status = page.evaluate(post_inject)
                assert status == "OK", f"post_inject should return OK, got {status}"
            finally:
                browser.close()

    def test_zai_post_inject_handles_missing_input(self, mock_server):
        """post_inject_js must return NO_MARKED_INPUT when no input is marked."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")
            try:
                post_inject = _ZAI["post_inject_js"]
                status = page.evaluate(post_inject)
                assert status == "NO_MARKED_INPUT", (
                    f"expected NO_MARKED_INPUT, got {status}"
                )
            finally:
                browser.close()

    def test_zai_post_inject_does_not_dispatch_enter(self, mock_server):
        """post_inject_js must NOT dispatch Enter key events (double-send fix)."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")
            try:
                # Set up the input and mark it.
                page.evaluate("""
                    (function() {
                        const el = document.getElementById('chat-input');
                        const setter = Object.getOwnPropertyDescriptor(
                            HTMLTextAreaElement.prototype, 'value').set;
                        setter.call(el, 'test');
                        el.dataset.sbsllmInput = 'true';
                        return 'OK';
                    })()
                """)

                # Count submit events on the form before post_inject.
                page.evaluate("""
                    window.__submitCount = 0;
                    document.getElementById('chat-form').addEventListener(
                        'submit', function() { window.__submitCount++; }
                    );
                """)

                # Run the post_inject_js block.
                post_inject = _ZAI["post_inject_js"]
                status = page.evaluate(post_inject)
                assert status == "OK"

                # The form must NOT have been submitted by post_inject_js.
                count = page.evaluate("window.__submitCount")
                assert count == 0, (
                    f"post_inject_js triggered {count} form submit(s); "
                    "expected 0 (double-send bug)"
                )
            finally:
                browser.close()

    def test_zai_full_cycle_with_thinking_streaming(self, mock_server):
        """End-to-end: inject → submit → extract growing thinking → answer.

        The mock page writes thinking content first, then the answer after a
        delay. This verifies the real extraction JS separates thinking from
        answer across multiple polls, mirroring how a live z.ai turn streams
        a reasoning trace before the final reply.
        """
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")
            try:
                prompt = "what is 2+2"

                # 1. Inject via the real z.ai template.
                inject_status = page.evaluate(_ZAI["inject"])
                assert inject_status == "OK"

                # 2. Set the prompt value and mark the input.
                page.evaluate(
                    f"""
                    (function() {{
                        const el = document.getElementById('chat-input');
                        const setter = Object.getOwnPropertyDescriptor(
                            HTMLTextAreaElement.prototype, 'value').set;
                        setter.call(el, '{prompt}');
                        el.dispatchEvent(new Event('input', {{bubbles:true}}));
                        el.dispatchEvent(new Event('change', {{bubbles:true}}));
                        el.dataset.sbsllmInput = 'true';
                        return 'OK';
                    }})()
                    """
                )

                # 3. Submit via the real z.ai template.
                submit_status = page.evaluate(_ZAI["submit_js"])
                assert submit_status == "OK"

                # 4. Extract during the thinking phase (before the 500ms answer swap).
                page.wait_for_timeout(100)
                extraction = _response_js(
                    _ZAI["response_selectors"],
                    _ZAI["thinking_selectors"],
                    _ZAI["loading_selectors"],
                    _ZAI.get("login_wall_selectors", []),
                )
                thinking_phase = page.evaluate(extraction)
                assert thinking_phase["busy"] is True, (
                    f"expected busy during thinking, got {thinking_phase}"
                )
                assert thinking_phase["thinking"] is not None, (
                    "thinking should be present during thinking phase"
                )
                assert (
                    "Thinking" in thinking_phase["thinking"]
                    or "thinking" in thinking_phase["thinking"].lower()
                )

                # 5. Wait for the answer to arrive, then extract again.
                page.wait_for_timeout(600)
                final = page.evaluate(extraction)
                assert final["content"] is not None, (
                    "answer should be present after generation"
                )
                assert prompt in final["content"], (
                    f"prompt not in answer: {final['content']!r}"
                )
                assert final["done"] is True, f"not done: {final}"

                # 6. The thinking must NOT leak into the final answer content.
                assert "thinking" not in final["content"].lower(), (
                    f"thinking leaked into answer: {final['content']!r}"
                )

                # 7. The form must have been submitted exactly once.
                submit_count = page.evaluate("window.__submitCount")
                assert submit_count == 1, (
                    f"form submitted {submit_count} times; expected exactly 1 "
                    "(double-send bug)"
                )
            finally:
                browser.close()

    def test_zai_submit_sends_exactly_once(self, mock_server):
        """The full inject + post_inject + submit flow must submit the form
        exactly once.

        This is the regression test for the z.ai double-send bug: previously
        post_inject_js dispatched Enter key events, which triggered the form's
        native submit handler BEFORE submit_js clicked the send button,
        resulting in two submissions. The fix removed Enter dispatch from
        post_inject_js; this test asserts the form is submitted exactly once.
        """
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_server, wait_until="networkidle")
            try:
                prompt = "double-send regression test"

                # Run the full inject (which includes post_inject_js).
                inject_status = page.evaluate(inject_prompt("zai", prompt))
                assert inject_status == "OK", f"inject failed: {inject_status}"

                # Verify the prompt was set.
                val = page.evaluate(
                    "document.getElementById('chat-input').value"
                )
                assert prompt in val, f"prompt not in input: {val!r}"

                # Submit via the real z.ai submit template.
                submit_status = page.evaluate(_ZAI["submit_js"])
                assert submit_status == "OK", f"submit failed: {submit_status}"

                # The form must have been submitted exactly once.
                submit_count = page.evaluate("window.__submitCount")
                assert submit_count == 1, (
                    f"form submitted {submit_count} times; expected exactly 1 "
                    "(double-send bug)"
                )
                # And the submitted prompt must match.
                submitted_prompts = page.evaluate("window.__submitPrompts")
                assert submitted_prompts == [prompt], (
                    f"unexpected submitted prompts: {submitted_prompts!r}"
                )
            finally:
                browser.close()


_KIMI = SITES["kimi"]
_KIMI_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock kimi</title></head>
<body>
  <form id="kimi-form">
    <div id="input-root">
      <div class="chat-input-editor" data-sbsllm-input="true"
           data-sbsllm-value=""
           contenteditable="true"
           placeholder="What would you like to know">
      </div>
    </div>
    <button type="submit" id="kimi-send">Send</button>
  </form>
  <div id="kimi-conversation"></div>
  <script>
    var form = document.getElementById('kimi-form');
    var input = document.querySelector('div.chat-input-editor[contenteditable="true"]');
    var convo = document.getElementById('kimi-conversation');
    var sendBtn = document.getElementById('kimi-send');
    function syncValue() {
      input.dataset.sbsllmValue = input.innerText || '';
      var marked = document.querySelector('[data-sbsllm-input="true"]');
      if (marked) marked.dataset.sbsllmValue = input.innerText || '';
    }
    input.addEventListener('input', syncValue);
    form.addEventListener('submit', function(e) {
      e.preventDefault();
      var prompt = input.innerText || '';
      var stop = document.createElement('button');
      stop.setAttribute('aria-label', 'Stop generating');
      stop.textContent = 'Stop';
      document.body.appendChild(stop);
      setTimeout(function() {
        var msg = document.createElement('div');
        msg.setAttribute('data-role', 'assistant');
        var markdown = document.createElement('div');
        markdown.className = 'markdown';
        markdown.textContent = 'Kimi received: ' + prompt;
        msg.appendChild(markdown);
        convo.appendChild(msg);
        stop.remove();
      }, 400);
    });
  </script>
</body></html>
"""


class _KimiHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_KIMI_MOCK_PAGE_HTML.encode())

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def mock_kimi_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _KimiHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


# --- Shadow-DOM extraction tests ---

# Several supported sites (Google AI Studio's ms-* web components, custom
# elements on z.ai / grok) render the assistant answer inside a web
# component's Shadow DOM. document.querySelector cannot reach into shadow
# roots, so a light-DOM-only scan returned nothing and the server reported
# "no output" on a perfectly healthy reply. This mock mirrors that structure.
_SHADOW_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock shadow</title></head>
<body>
  <h1>Shadow host page</h1>
  <my-chat-turn data-turn-role="model">
    <!-- The real content lives in the component's shadow root. -->
  </my-chat-turn>
  <my-chat-turn data-turn-role="user">
  </my-chat-turn>
  <script>
    class MyChatTurn extends HTMLElement {
      connectedCallback() {
        const root = this.attachShadow({ mode: 'open' });
        const role = this.getAttribute('data-turn-role');
        if (role === 'model') {
          const prose = document.createElement('div');
          prose.className = 'prose-chat';
          prose.textContent = 'Answer from inside the shadow DOM';
          const stop = document.createElement('button');
          stop.setAttribute('aria-label', 'Stop generating');
          root.appendChild(prose);
          root.appendChild(stop);
        } else {
          const prose = document.createElement('div');
          prose.className = 'prose-chat';
          prose.textContent = 'User turn';
          root.appendChild(prose);
        }
      }
    }
    customElements.define('my-chat-turn', MyChatTurn);
  </script>
</body></html>
"""


class _ShadowHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_SHADOW_MOCK_PAGE_HTML.encode())

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def mock_shadow_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _ShadowHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


class TestShadowDomExtraction:
    """The response extraction JS must reach into web-component shadow roots.

    Regression guard: a light-DOM-only scan returned ``found: false`` for
    every site whose assistant turn renders inside a shadow tree, which made
    the server report "no output" on replies that were fully on screen.
    """

    def test_extraction_finds_answer_inside_shadow_root(self, mock_shadow_server):
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_shadow_server, wait_until="networkidle")
            try:
                extraction = _response_js(
                    ["[data-turn-role='model'] .prose-chat"],
                    ["[class*='thinking']"],
                    [],
                    [],
                )
                result = page.evaluate(extraction)
                assert result["found"] is True, (
                    f"extraction missed shadow-DOM answer: {result}"
                )
                assert "shadow DOM" in result["content"], (
                    f"wrong content: {result['content']!r}"
                )
                assert result["busy"] is False
            finally:
                browser.close()

    def test_extraction_falls_back_to_shadow_when_light_dom_empty(
        self, mock_shadow_server
    ):
        """A selector that matches nothing in the light DOM must still find
        the answer by walking shadow roots."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_shadow_server, wait_until="networkidle")
            try:
                # `.chat-turn-container .prose-chat` matches nothing in the
                # light DOM (the host element has no children) but does match
                # inside the component's shadow root.
                extraction = _response_js(
                    [".chat-turn-container .prose-chat"],
                    [],
                    [],
                    [],
                )
                result = page.evaluate(extraction)
                assert result["found"] is True, (
                    f"shadow fallback failed: {result}"
                )
                assert "shadow DOM" in result["content"]
            finally:
                browser.close()


class TestKimiIntegration:
    """kimi.ai uses a `div.chat-input-editor[contenteditable="true"]`
    composite editor. The inject + post_inject (contenteditable sync) +
    submit JS must cooperate to set the prompt and click send, and the
    response selectors must read the answer back (without leaking any
    thinking-chain element)."""

    def test_kimi_inject_targets_contenteditable_editor(self, mock_kimi_server):
        """inject_prompt must find the contenteditable editor and mark it."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_kimi_server, wait_until="networkidle")
            try:
                status = page.evaluate(inject_prompt("kimi", "kimi test prompt"))
                assert status == "OK", f"inject should find editor, got {status}"
                has_marker = page.evaluate(
                    "!!document.querySelector('[data-sbsllm-input=\"true\"]')"
                )
                assert has_marker, "editor should be marked after inject"
            finally:
                browser.close()

    def test_kimi_full_cycle(self, mock_kimi_server):
        """inject -> submit -> extract must round-trip on contenteditable
        kimi like it does for the textarea-driven sites."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_kimi_server, wait_until="networkidle")
            try:
                assert page.evaluate(inject_prompt("kimi", "kimi full cycle")) == "OK"
                assert page.evaluate(submit_js("kimi")) == "OK"
                # Wait for the async 400ms response to land.
                page.wait_for_function(
                    "!!document.querySelector('[data-role=\"assistant\"]')",
                    timeout=2000,
                )
                extracted = page.evaluate(extract_js("kimi"))
                assert extracted["found"] is True, f"expected answer: {extracted}"
                assert "kimi full cycle" in extracted["content"]
                assert extracted["busy"] is False, "still busy after response"
            finally:
                browser.close()

    def test_kimi_response_selector_excludes_thinking_markdown(self):
        """Kimi nests the thinking trace in a
        .markdown-container.toolcall-content-text inside
        .thinking-container. A bare '.markdown' fallback matched that
        node first, so the thinking text was relayed as the answer.
        The response selector must scope to
        .markdown-container:not(.toolcall-content-text)."""
        from sbsllm.sites import SITES

        selectors = SITES["kimi"]["response_selectors"]
        assert any(
            ".markdown-container:not(.toolcall-content-text) .markdown" in s
            for s in selectors
        )
        # The bare fallback must not be the one that reaches the
        # thinking node.
        assert ".markdown'" not in " ".join(selectors)

    def test_kimi_response_container_scopes_to_newest_turn(self):
        """Without a response container, the last answer .markdown in the
        document belongs to the PREVIOUS turn while the current one is
        still thinking (it has a thinking block but no answer yet), so the
        previous turn's poem was relayed as this turn's answer. Kimi now
        scopes to .chat-content-item, so a pending turn reports
        found=false until its own answer .markdown appears."""
        from sbsllm.sites import SITES

        assert SITES["kimi"].get("response_container") == ".chat-content-item"

    def test_kimi_thinking_selectors_match_container_not_inner_blocks(self):
        """Kimi renders the thinking trace as a chain of sibling
        .toolcall-content-text blocks inside one
        .thinking-container. Matching each block separately relayed the
        trace once per step (the screenshot showed the same paragraph
        repeated). The selectors must lead with the container so its whole
        subtree is read in one pass."""
        from sbsllm.sites import SITES

        selectors = SITES["kimi"]["thinking_selectors"]
        assert selectors[0] == ".toolcall-container.thinking-container"
        assert ".toolcall-content-text" not in selectors

    def test_kimi_thinking_does_not_leak_into_answer(self, mock_kimi_server):
        """A Kimi turn whose thinking trace is already on screen must
        still be captured as thinking, not relayed as the answer."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_kimi_server, wait_until="networkidle")
            try:
                # Seed a thinking block + the answer in the same segment.
                page.evaluate("""() => {
                    const seg = document.createElement('div');
                    seg.className = 'segment segment-assistant';
                    const box = document.createElement('div');
                    box.className = 'segment-content-box';
                    const rollup = document.createElement('div');
                    rollup.className = 'toolcall-rollup';
                    const think = document.createElement('div');
                    think.className = 'toolcall-container thinking-container block-container is-flat-think';
                    const thinkContent = document.createElement('div');
                    thinkContent.className = 'toolcall-content';
                    const thinkMd = document.createElement('div');
                    thinkMd.className = 'markdown-container toolcall-content-text';
                    const thinkM = document.createElement('div');
                    thinkM.className = 'markdown';
                    thinkM.textContent = 'Thinking complete\\nCompute 17*23.';
                    thinkMd.appendChild(thinkM);
                    thinkContent.appendChild(thinkMd);
                    think.appendChild(thinkContent);
                    rollup.appendChild(think);
                    const ansMd = document.createElement('div');
                    ansMd.className = 'markdown-container';
                    const ansM = document.createElement('div');
                    ansM.className = 'markdown';
                    ansM.textContent = '17 × 23 = 391';
                    ansMd.appendChild(ansM);
                    rollup.appendChild(ansMd);
                    box.appendChild(rollup);
                    seg.appendChild(box);
                    document.body.appendChild(seg);
                }""")
                result = page.evaluate(extract_js("kimi"))
                assert result["found"] is True
                assert "Compute 17*23" in (result.get("thinking") or ""), (
                    f"thinking not captured: {result.get('thinking')!r}"
                )
                assert "Compute 17*23" not in result["content"], (
                    f"thinking leaked into answer: {result['content']!r}"
                )
                assert "17 × 23 = 391" in result["content"]
            finally:
                browser.close()

    def test_kimi_post_inject_is_idempotent(self, mock_kimi_server):
        """Running the full inject+post_inject cycle repeatedly must not
        accumulate copies of the prompt in the editor. Kimi's composer is
        a Lexical editor whose model is not cleared by execCommand('delete'),
        so without a real Ctrl+A + Delete clear each cycle appended the
        prompt again and the relayed answer carried it N times."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_kimi_server, wait_until="networkidle")
            try:
                prompt = "kimi idempotency check 42"
                js = inject_prompt("kimi", prompt)
                for _ in range(5):
                    assert page.evaluate(js) == "OK"
                    page.wait_for_timeout(50)
                text = page.evaluate(
                    "() => document.querySelector('div.chat-input-editor')"
                    " ? document.querySelector('div.chat-input-editor').innerText : ''"
                )
                assert text.count(prompt) == 1, (
                    f"prompt accumulated in editor: {text!r}"
                )
            finally:
                browser.close()


# --- meta.ai /prompt/<uuid> page tests ---

# After the first message, meta.ai navigates to /prompt/<uuid>, which
# renders a "Conversation title" input (type=text) next to the
# composer. The composer itself is a visible contenteditable div with
# a hidden textarea mirror (placeholder "Ask Meta AI..."). An inject
# selector list that includes a generic `input[type="text"]` matches
# the title field first, so from the second message on the prompt was
# typed into the conversation title instead of the composer: the site
# re-sent the previous turn's text (or nothing once the composer
# cleared), and the local chat read the stale answer. This mock
# mirrors that page structure.
_META_PROMPT_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock meta prompt</title></head>
<body>
  <form id="meta-form">
    <input type="text" id="conv-title" placeholder="Conversation title">
    <textarea id="meta-composer" placeholder="Ask Meta AI..."
              style="display:none"></textarea>
    <div id="meta-editor" contenteditable="true"></div>
    <button type="submit" id="meta-send" aria-label="Send" disabled>Send</button>
  </form>
  <div id="thread">
    <div class="group/assistant-message" data-testid="assistant-message"
         data-streaming-state="DONE" data-streaming-complete="true">
      <div class="mt-4"><div class="markdown-content">
        <div dir="auto" class="ur-markdown prose">
          <div class="space-y-4"><p dir="auto">Previous answer</p></div>
        </div>
      </div></div>
      <div class="group/assistant-message-actions">
        <button aria-label="Like this response"><svg></svg></button>
        <button aria-label="Dislike this response"><svg></svg></button>
      </div>
    </div>
  </div>
  <script>
    var title = document.getElementById('conv-title');
    var composer = document.getElementById('meta-composer');
    var editor = document.getElementById('meta-editor');
    var sendBtn = document.getElementById('meta-send');
    var form = document.getElementById('meta-form');
    var thread = document.getElementById('thread');
    window.__submitPrompts = [];
    // The site syncs the hidden textarea into the visible editor and
    // the send button's disabled state.
    composer.addEventListener('input', function() {
      editor.textContent = this.value;
      sendBtn.disabled = !this.value.trim();
    });
    form.addEventListener('submit', function(e) {
      e.preventDefault();
      window.__submitPrompts.push(composer.value);
      var prompt = composer.value;
      composer.value = '';
      editor.textContent = '';
      sendBtn.disabled = true;
      var row = document.createElement('div');
      row.className = 'group/assistant-message';
      row.setAttribute('data-testid', 'assistant-message');
      row.setAttribute('data-streaming-state', 'STREAMING');
      row.setAttribute('data-streaming-complete', 'false');
      row.innerHTML =
        '<div class="mt-4"><div class="markdown-content">' +
        '<div dir="auto" class="ur-markdown prose"><div class="space-y-4">' +
        '<p dir="auto">Generating...</p></div></div></div></div>' +
        '<div class="group/assistant-message-actions">' +
        '<button aria-label="Like this response"><svg></svg></button>' +
        '<button aria-label="Dislike this response"><svg></svg></button>' +
        '</div>';
      thread.appendChild(row);
      setTimeout(function() {
        row.querySelector('p[dir="auto"]').textContent = 'Answer: ' + prompt;
        row.setAttribute('data-streaming-state', 'DONE');
        row.setAttribute('data-streaming-complete', 'true');
      }, 300);
    });
  </script>
</body></html>
"""


class _MetaPromptHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(_META_PROMPT_MOCK_PAGE_HTML.encode())

    def log_message(self, *args):  # silence
        pass


@pytest.fixture(scope="module")
def mock_meta_prompt_server():
    with socketserver.TCPServer(("127.0.0.1", 0), _MetaPromptHandler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}"
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield url
    thread.join(timeout=5)


class TestMetaPromptPage:
    """meta.ai's post-first-message page: the prompt must land in the
    composer, never in the conversation-title input."""

    def test_inject_targets_composer_not_title_input(self, mock_meta_prompt_server):
        """Regression guard: the inject selector list used to contain a
        generic `input[type="text"]`, which matched the conversation-title
        field rendered on /prompt/<uuid> before the composer -- so the
        second and later prompts never reached the chat."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_meta_prompt_server, wait_until="networkidle")
            try:
                assert page.evaluate(
                    inject_prompt("meta", "second turn prompt")
                ) == "OK"
                # The trap: the title input must stay empty.
                title_value = page.evaluate(
                    "document.getElementById('conv-title').value"
                )
                assert title_value == "", "prompt leaked into the title input"
                # The prompt reached the hidden composer mirror...
                composer_value = page.evaluate(
                    "document.getElementById('meta-composer').value"
                )
                assert composer_value == "second turn prompt"
                # ...and the site sync carried it into the visible editor.
                editor_text = page.evaluate(
                    "document.getElementById('meta-editor').textContent"
                )
                assert editor_text == "second turn prompt"
                # The marked input is the composer, not the title field.
                marked_placeholder = page.evaluate(
                    "document.querySelector('[data-sbsllm-input=\"true\"]')"
                    "?.getAttribute('placeholder')"
                )
                assert marked_placeholder == "Ask Meta AI..."
            finally:
                browser.close()

    def test_full_cycle_sends_prompt_once(self, mock_meta_prompt_server):
        """inject -> submit -> extract round-trips on the /prompt page:
        the form submit handler receives exactly the injected prompt and
        the new assistant row is extracted (not the previous answer)."""
        with pw.sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(mock_meta_prompt_server, wait_until="networkidle")
            try:
                assert page.evaluate(
                    inject_prompt("meta", "third turn prompt")
                ) == "OK"
                assert page.evaluate(submit_js("meta")) == "OK"
                page.wait_for_function(
                    "(() => { const rows ="
                    " document.querySelectorAll('[data-testid=\"assistant-message\"]');"
                    " return rows.length === 2 && rows[1].getAttribute("
                    "'data-streaming-complete') === 'true'; })()",
                    timeout=3000,
                )
                submitted = page.evaluate("window.__submitPrompts")
                assert submitted == ["third turn prompt"], submitted
                extracted = page.evaluate(extract_js("meta"))
                assert extracted["found"] is True, f"expected answer: {extracted}"
                assert "third turn prompt" in extracted["content"]
                assert "Previous answer" not in extracted["content"]
                assert extracted["busy"] is False
            finally:
                browser.close()
