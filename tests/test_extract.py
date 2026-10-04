"""Tests for response extraction from web chat pages.

These run the generated extraction JS in a real Chromium so the assertions
cover actual DOM behaviour (innerText, computed style, getClientRects) rather
than a hand-written model of it. Skipped when no browser is available.
"""

import pytest

from sbsllm.sites import SITES, _response_js

playwright_api = pytest.importorskip("playwright.sync_api")

CHROMIUM_CANDIDATES = [
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]

# The z.ai markup exactly as reported when the local chat hung: a collapsed
# "Thought Process" disclosure nested inside the answer container, plus a
# decorative element whose class contains "loading".
ZAI_COLLAPSED_THINKING = """
<div id="response-content-container"><!----><div class="markdown-prose">
  <div class="w-full thinking-chain-container my-4 svelte-bugqhi" data-direct="false">
    <div class="flex gap-1 justify-between items-center pl-1 svelte-bugqhi">
      <button class="flex items-center gap-1 svelte-bugqhi">
        <svg width="16" height="16"></svg>
        <span class="text-sm svelte-bugqhi">Thought Process</span>
        <svg width="16" height="16"></svg>
      </button>
    </div>
  </div>
  <p dir="auto" class="svelte-4sys19">Hello! I'm GLM, trained by Z.ai. How can I
  help you today?</p>
</div><!----></div>
<div aria-hidden="true"></div>
<div class="skeleton loading w-full h-4" style="opacity:0"></div>
"""

ZAI_EXPANDED_THINKING = """
<div id="response-content-container"><div class="markdown-prose">
  <div class="thinking-chain-container">
    <div class="flex justify-between">
      <button aria-expanded="true"><span>Thought Process</span></button>
    </div>
    <div class="thinking-body"><p>First I greet them. Then I offer help.</p></div>
  </div>
  <p dir="auto">Hi there</p>
</div></div>
"""

ZAI_STILL_GENERATING = """
<div id="response-content-container"><div class="markdown-prose">
  <p dir="auto">Partial ans</p>
</div></div>
<button aria-label="Stop generating">Stop</button>
"""

# A thinking selector so broad it also wraps the answer. The reply still lives
# in a markdown block, which is how the answer-host guard recognises it.
ZAI_THINKING_WRAPS_ANSWER = """
<div id="response-content-container"><div class="markdown-prose">
  <div class="thinking-wrapper"><div class="markdown-prose">
    <p>reasoning text</p><p>the actual answer</p></div></div>
</div></div>
"""

# The reasoning block while the answer has not started yet. This is every
# z.ai response's thinking phase, and it is why an empty pruned result must
# stay empty: falling back to the unpruned text rendered the reasoning into
# the answer as well, so the local chat showed the thinking twice.
ZAI_THINKING_ONLY = """
<div id="response-content-container"><div class="markdown-prose">
  <div class="thinking-chain-container">
    <div><button>Thought Process</button></div>
    <div class="thinking-body"><p>still reasoning</p></div>
  </div>
  <p id="ans"></p>
</div></div>
"""


def _launch(playwright):
    errors = []
    for executable in CHROMIUM_CANDIDATES:
        import os

        if not os.path.exists(executable):
            continue
        try:
            return playwright.chromium.launch(
                executable_path=executable, args=["--no-sandbox"]
            )
        except Exception as e:  # noqa: BLE001
            errors.append(f"{executable}: {e}")
    try:
        return playwright.chromium.launch()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"no usable Chromium: {errors + [str(e)]}")


@pytest.fixture(scope="module")
def extract_page():
    with playwright_api.sync_playwright() as pw:
        browser = _launch(pw)
        try:
            yield browser.new_page()
        finally:
            browser.close()


def _extract(page, html, site_id="zai"):
    page.set_content(f"<html><body>{html}</body></html>")
    js = _response_js(
        SITES[site_id]["response_selectors"],
        SITES[site_id]["thinking_selectors"],
        SITES[site_id]["loading_selectors"],
        SITES[site_id].get("login_wall_selectors", []),
    )
    return page.evaluate(js)


class TestBusySignal:
    """The `busy` flag is what stops answers being cut short.

    It once went missing from the extraction JS while the server still read
    it, so every poll looked idle and the stream ended at the first pause.
    """

    def test_extraction_reports_busy_while_generating(self, extract_page):
        result = _extract(extract_page, ZAI_STILL_GENERATING)
        assert result["busy"] is True
        assert result["done"] is False

    def test_extraction_reports_not_busy_once_finished(self, extract_page):
        result = _extract(extract_page, ZAI_COLLAPSED_THINKING)
        assert result["busy"] is False
        assert result["done"] is True

    def test_busy_key_is_always_present(self, extract_page):
        for html in (
            ZAI_COLLAPSED_THINKING,
            ZAI_STILL_GENERATING,
            ZAI_EXPANDED_THINKING,
        ):
            assert "busy" in _extract(extract_page, html)


class TestZaiExtraction:
    def test_collapsed_disclosure_yields_answer_only(self, extract_page):
        """Regression: this markup left the local chat streaming forever."""
        result = _extract(extract_page, ZAI_COLLAPSED_THINKING)
        assert (
            result["content"]
            == "Hello! I'm GLM, trained by Z.ai. How can I help you today?"
        )
        # The toggle label is UI chrome, not a reasoning trace.
        assert result["thinking"] is None
        # A decorative `.loading` node must not pin done to false.
        assert result["done"] is True
        assert result["found"] is True

    def test_expanded_disclosure_yields_reasoning_and_answer(self, extract_page):
        result = _extract(extract_page, ZAI_EXPANDED_THINKING)
        assert result["content"] == "Hi there"
        assert result["thinking"] == "First I greet them. Then I offer help."
        assert result["done"] is True

    def test_stop_control_keeps_stream_open(self, extract_page):
        result = _extract(extract_page, ZAI_STILL_GENERATING)
        assert result["content"] == "Partial ans"
        assert result["done"] is False

    def test_broad_thinking_selector_does_not_delete_answer(self, extract_page):
        """A thinking candidate that also hosts a markdown block is an answer
        host, so it is left alone rather than pruned away."""
        result = _extract(extract_page, ZAI_THINKING_WRAPS_ANSWER)
        assert "the actual answer" in result["content"]

    def test_thinking_phase_yields_no_answer_text(self, extract_page):
        """Regression: the reasoning leaked into `content` for the whole
        thinking phase, so the local chat rendered the thinking twice."""
        result = _extract(extract_page, ZAI_THINKING_ONLY)
        assert result["content"] == "", result["content"]
        assert result["thinking"] == "still reasoning"

    def test_full_response_with_thinking_and_answer(self, extract_page):
        """End-to-end reasoning-leak scenario: a complete z.ai response with
        both thinking and answer phases. The thinking must be pruned from
        content and only the answer must remain."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose">
                <div class="thinking-chain-container">
                    <div class="flex justify-between">
                        <button aria-expanded="true"><span>Thought Process</span></button>
                    </div>
                    <div class="thinking-body">
                        <p>The user asked what 2+2 is. I need to compute this.</p>
                        <p>2 + 2 = 4. Simple arithmetic.</p>
                    </div>
                </div>
                <p dir="auto">4</p>
            </div>
        </div>
        """
        result = _extract(extract_page, html)
        # The answer must be present
        assert result["content"] == "4", result["content"]
        # The thinking must be captured separately
        assert result["thinking"] is not None
        assert "The user asked what 2+2 is" in result["thinking"]
        assert "2 + 2 = 4" in result["thinking"]
        # The thinking must NOT leak into content
        assert "2+2" not in result["content"]
        assert "compute" not in result["content"]
        assert result["done"] is True
        assert result["found"] is True

    def test_thinking_pruned_not_replaced(self, extract_page):
        """The prune must remove the thinking subtree, not string-replace its
        text. If the old string-replace approach were used, the thinking
        label would remain in content."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose">
                <div class="thinking-chain-container">
                    <div><button>Thought Process</button></div>
                    <div class="thinking-body"><p>Reasoning here</p></div>
                </div>
                <p dir="auto">The answer is 42</p>
            </div>
        </div>
        """
        result = _extract(extract_page, html)
        assert result["content"] == "The answer is 42"
        assert "Reasoning here" not in result["content"]
        assert "Thought Process" not in result["content"]
        assert result["thinking"] == "Reasoning here"


class TestParagraphStructure:
    """Answers must keep their paragraph breaks.

    The extraction used to squash every whitespace run into one space, so
    a multi-paragraph reply reached the local chat as a single run-on line.
    """

    def test_unpruned_answer_keeps_paragraph_breaks(self, extract_page):
        html = """
        <div id="response-content-container"><div class="markdown-prose">
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
        </div></div>
        """
        result = _extract(extract_page, html)
        assert result["content"] == "First paragraph.\n\nSecond paragraph."

    def test_pruned_answer_keeps_paragraph_breaks(self, extract_page):
        """The prune path reads a detached clone, which has no layout and
        therefore no block-boundary newlines; the walker must re-introduce
        them from the DOM structure."""
        html = """
        <div id="response-content-container"><div class="markdown-prose">
          <div class="thinking-chain-container">
            <div><button>Thought Process</button></div>
            <div class="thinking-body"><p>secret reasoning</p></div>
          </div>
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
        </div></div>
        """
        result = _extract(extract_page, html)
        assert result["content"] == "First paragraph.\n\nSecond paragraph."
        assert "secret reasoning" not in result["content"]

    def test_source_line_break_inside_paragraph_is_not_a_newline(self, extract_page):
        """HTML source wrapping inside one <p> is not a paragraph break."""
        html = """
        <div id="response-content-container"><div class="markdown-prose">
          <div class="thinking-chain-container">
            <div><button>Thought Process</button></div>
          </div>
          <p>Wrapped source text
          stays one line.</p>
        </div></div>
        """
        result = _extract(extract_page, html)
        assert result["content"] == "Wrapped source text stays one line."

    def test_br_tag_breaks_the_line(self, extract_page):
        html = """
        <div id="response-content-container"><div class="markdown-prose">
          <div class="thinking-chain-container">
            <div><button>Thought Process</button></div>
          </div>
          <p>line one<br>line two</p>
        </div></div>
        """
        result = _extract(extract_page, html)
        assert result["content"] == "line one\nline two"


class TestWorkingPrefixStripped:
    def test_working_prefix_is_removed_and_marks_streaming(self, extract_page):
        html = """
        <div id="response-content-container"><div class="markdown-prose">
        <p>Working for 12s</p></div></div>
        <button aria-label="Stop generating">Stop</button>
        """
        result = _extract(extract_page, html)
        assert result["content"] == ""
        assert result["done"] is False


class TestZaiLoadingDots:
    """z.ai renders an animated dot loader inside #response-content-container
    while a reply is being generated. Without a loading selector for the dots,
    `busy` is always False and the server terminates the stream before any
    content arrives."""

    ZAI_LOADING_DOTS = """
    <div id="response-content-container"><div class="flex py-1">
        <div class="container svelte-m0sfji">
            <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
            <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
            <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
            <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
        </div>
    </div></div>
    """

    ZAI_LOADING_DOTS_WITH_ANSWER = """
    <div id="response-content-container">
        <div class="markdown-prose"><p>4</p></div>
        <div class="flex py-1">
            <div class="container svelte-m0sfji">
                <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
                <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
                <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
                <div class="dot bg-[#0d0d0d]/80 svelte-m0sfji"></div>
            </div>
        </div>
    </div>
    <button aria-label="Stop generating">Stop</button>
    """

    def test_loading_dots_signal_busy(self, extract_page):
        """Loading dots inside the response container must report busy=True."""
        result = _extract(extract_page, self.ZAI_LOADING_DOTS)
        assert result["busy"] is True, "loading dots should signal busy"
        assert result["done"] is False
        assert result["content"] == ""

    def test_loading_dots_with_answer_still_busy(self, extract_page):
        """Even with partial content, loading dots mean the reply is still
        streaming."""
        result = _extract(extract_page, self.ZAI_LOADING_DOTS_WITH_ANSWER)
        assert result["busy"] is True
        assert result["done"] is False
        assert "4" in result["content"]

    def test_no_loading_dots_means_not_busy(self, extract_page):
        """Without loading dots, a completed response should not be busy."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>42</p></div>
        </div>
        """
        result = _extract(extract_page, html)
        assert result["busy"] is False
        assert result["done"] is True
        assert result["content"] == "42"

    def test_login_wall_detected_after_submit(self, extract_page):
        """A page that shows 'Sign up to continue' after submission must
        report login_wall=True so the server can surface a login error
        instead of a silent empty response."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>Say hello</p></div>
        </div>
        <div>Continue your conversation</div>
        <div>Sign up to continue seamlessly with Grok</div>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True, (
            "login_wall must be True when the page shows a sign-up wall"
        )

    def test_no_login_wall_on_normal_page(self, extract_page):
        """A normal page without login prompts must report login_wall=False."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>42</p></div>
        </div>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is False

    def test_login_wall_pattern_sign_in_to_continue(self, extract_page):
        """'Sign in to continue' is a login-wall signal."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>hello</p></div>
        </div>
        <div>Sign in to continue using X</div>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True

    def test_login_wall_pattern_please_log_in(self, extract_page):
        """'Please log in to continue' is a login-wall signal."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>hello</p></div>
        </div>
        <div>Please log in to continue</div>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True

    def test_login_wall_pattern_login_to_continue(self, extract_page):
        """'Login to continue' is a login-wall signal."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>hello</p></div>
        </div>
        <div>Login to continue your session</div>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True

    def test_login_wall_pattern_must_be_logged_in(self, extract_page):
        """'You must be logged in' is a login-wall signal."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>hello</p></div>
        </div>
        <div>You must be logged in to use this feature</div>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True

    def test_login_wall_does_not_false_positive_on_footer(self, extract_page):
        """A normal footer with 'Terms of Service' and 'Privacy Policy'
        must NOT trigger a login wall."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>42</p></div>
        </div>
        <footer>
            <a>Terms of Service</a>
            <a>Privacy Policy</a>
            <a>Contact us</a>
        </footer>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is False


class TestLoginWallSelectors:
    """Site-specific login-wall selectors must trigger login_wall=True when
    a visible login element is present, but NOT when the element is hidden."""

    # zai selectors: ['a[href*="login" i]', 'button[aria-label*="Log in" i]']
    # (Note: 'button[aria-label*="Sign in" i]' was removed from zai because
    # the persistent nav sign-in button on the working chat page caused false
    # positives. Real login walls are still caught by text-pattern + the
    # remaining 'Log in' button / login-link selectors.)

    def test_login_wall_detected_via_visible_selector(self, extract_page):
        """A visible 'Sign in' button matching site selectors triggers
        login_wall=True even without any text-pattern match."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p></p></div>
        </div>
        <button aria-label="Log in to your account">Log in</button>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True

    def test_login_wall_not_triggered_by_hidden_element(self, extract_page):
        """A hidden login button (display:none) must NOT trigger login_wall."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>42</p></div>
        </div>
        <button aria-label="Sign in" style="display:none">Sign in</button>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is False

    def test_login_wall_not_triggered_by_invisible_aria_hidden(self, extract_page):
        """A login link with aria-hidden that takes no space must NOT trigger."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>42</p></div>
        </div>
        <a href="/login" aria-hidden="true" style="opacity:0">Login</a>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is False

    def test_login_wall_selector_with_response_content(self, extract_page):
        """When the page has a real response AND a login button is visible,
        login_wall should still be True — the server uses it to warn even
        if content exists."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>Hello!</p></div>
        </div>
        <button aria-label="Log in">Log in</button>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True

    def test_login_wall_combined_text_and_selector(self, extract_page):
        """Both text-pattern and selector paths can fire; result is still True."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>hello</p></div>
        </div>
        <div>Please log in to continue</div>
        <a href="/login">Login</a>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is True

    def test_login_wall_none_when_no_login_elements(self, extract_page):
        """A page with no login elements must report login_wall=False."""
        html = """
        <div id="response-content-container">
            <div class="markdown-prose"><p>42</p></div>
        </div>
        <nav><a href="/home">Home</a><a href="/about">About</a></nav>
        """
        result = _extract(extract_page, html)
        assert result["login_wall"] is False


class TestMetaExtraction:
    """meta.ai message markup, as dumped from the live site.

    The message row is `group/assistant-message` (so `.assistant-message`
    never matches) with `data-testid="assistant-message"`. The action bar
    shares the class substring (`group/assistant-message-actions`), so the
    old `[class*="assistant-message"]` catch-all matched both and selection
    took the LAST match — the icon-only action buttons. Content therefore
    came back empty and replies were never relayed to the local chat.
    """

    @staticmethod
    def _message(answer, *, streaming=False):
        state = (
            'data-streaming-state="STREAMING" data-streaming-complete="false"'
            if streaming
            else 'data-streaming-state="DONE" data-streaming-complete="true"'
        )
        return f"""
        <div data-slot="flexbox" class="min-h-0 min-w-0 flex flex-col shrink-0">
          <div class="relative w-full min-w-0">
            <div class="group/assistant-message relative min-w-0"
                 data-testid="assistant-message" {state}>
              <div class="mx-auto flex w-full max-w-3xl items-start gap-3">
                <div class="-ms-1 flex shrink-0 items-center">
                  <img alt="" width="24" height="24" class="invisible"
                       src="/images/cot_logo_static/orbit.png">
                </div>
              </div>
              <div class="mt-4"><div class="markdown-content min-w-0">
                <div dir="auto" class="ur-markdown prose prose-trimmed citation-aware">
                  <div class="space-y-4 flex flex-col gap-6">
                    <p dir="auto">{answer}</p>
                  </div>
                </div>
              </div></div>
              <div class="group/assistant-message-actions mx-auto flex min-h-11">
                <button aria-label="Like this response"><svg viewBox="0 0 24 24"></svg></button>
                <button aria-label="Dislike this response"><svg viewBox="0 0 24 24"></svg></button>
                <button aria-label="Copy response"><svg viewBox="0 0 24 24"></svg></button>
                <button aria-label="Share"><svg viewBox="0 0 24 24"></svg></button>
              </div>
              <div aria-hidden="true" class="rounded-22 absolute inset-0"></div>
            </div>
          </div>
        </div>
        """

    def test_answer_is_extracted_not_the_action_bar(self, extract_page):
        html = self._message("Hey — I'm here and listening. What do you want to test?")
        result = _extract(extract_page, html, site_id="meta")
        assert result["found"] is True
        assert (
            result["content"]
            == "Hey — I'm here and listening. What do you want to test?"
        ), result["content"]
        assert result["done"] is True
        assert result["busy"] is False
        # Only the message row itself; the action bar must not be selected.
        assert result["count"] == 1

    def test_streaming_state_reports_busy(self, extract_page):
        """While meta.ai is generating, its own streaming-state attributes
        are the positive busy signal; without them the site never reports
        busy and long answers end at the first idle window."""
        html = self._message("Partial ans", streaming=True)
        result = _extract(extract_page, html, site_id="meta")
        assert result["busy"] is True
        assert result["done"] is False
        assert result["content"] == "Partial ans"

    def test_newest_message_wins(self, extract_page):
        html = self._message("Older reply.") + self._message("Newest reply.")
        result = _extract(extract_page, html, site_id="meta")
        assert result["content"] == "Newest reply."
        assert result["count"] == 2
