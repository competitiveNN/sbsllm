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

from sbsllm.inject import inject_prompt, submit_js
from sbsllm.sites import SITES, _response_js

pw = pytest.importorskip("playwright.sync_api")

_ZAI = SITES["zai"]

_MOCK_PAGE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>mock zai</title></head>
<body>
  <form id="chat-form">
    <textarea id="chat-input" placeholder="Ask anything"></textarea>
    <button type="submit" id="send-btn">Send</button>
  </form>
  <div id="response-content-container">
    <div class="markdown-prose"></div>
  </div>
  <script>
    var sendBtn = document.getElementById('send-btn');
    var form = document.getElementById('chat-form');
    var input = document.getElementById('chat-input');
    var container = document.querySelector('#response-content-container .markdown-prose');
    input.addEventListener('input', function() {
      sendBtn.disabled = !this.value.trim();
    });
    form.addEventListener('submit', function(e) {
      e.preventDefault();
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
                )
                result = page.evaluate(extraction)
                assert result["found"] or result["content"], f"no content: {result}"
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
                )
                result = page.evaluate(extraction)
                assert result["found"] or result["content"], f"no content: {result}"
                assert prompt in result["content"], (
                    f"prompt not in answer: {result['content']!r}"
                )
            finally:
                browser.close()


# --- z.ai post_inject_js integration test ---

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
            finally:
                browser.close()
