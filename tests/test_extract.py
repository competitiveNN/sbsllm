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
    site = SITES[site_id]
    js = _response_js(
        site["response_selectors"],
        site["thinking_selectors"],
        site["loading_selectors"],
        site.get("login_wall_selectors", []),
        site.get("response_container"),
        site.get("response_exclude_selectors", []),
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


class TestHuggingFaceExtraction:
    """huggingface.co/chat markup, as rendered by chat-ui's
    ChatMessage.svelte / OpenReasoningResults.svelte.

    The assistant turn is marked `data-message-role="assistant"` (NOT
    `data-message-author-role`), the answer lives in a `div.prose`
    whose class list has no `prose-sm`, and the reasoning viewport's
    prose carries `prose-sm`. The old selectors
    (`data-message-author-role`, `.assistant-message`,
    `[class*="assistant"]`) never matched, so the answer was never
    extracted: only the thinking trace streamed and the local chat
    waited forever for the reply. The action bar (router metadata,
    copy/retry buttons) also sits inside the turn, so a whole-turn
    selector would leak "route with <model> via <org>" into the answer.
    """

    @staticmethod
    def _message(answer, *, reasoning=None, streaming=False):
        thinking = ""
        if reasoning is not None:
            thinking = f"""
              <div data-exclude-from-copy class="not-last:mb-1 has-[+.prose]:mb-2!">
                <div class="not-last:mb-1">
                  <button type="button" aria-label="Collapse" class="group/header">
                    <span class="text-sm thinking-shimmer">Thinking</span>
                  </button>
                  <div class="thinking-viewport mt-2 flex max-h-56 flex-col justify-end overflow-hidden md:max-h-80">
                    <div class="prose prose-sm max-w-none text-sm leading-relaxed">
                      <p>{reasoning}</p>
                    </div>
                  </div>
                </div>
              </div>"""
        stop = (
            '<button type="button" class="stop-generating-btn" aria-label="Stop generating">'
            '<span class="sr-only">Stop generating</span></button>'
            if streaming
            else ""
        )
        # The answer prose only renders once text exists, so a
        # thinking-only turn has no prose block at all.
        answer_html = (
            f"""
            <div class="prose max-w-none text-smd dark:prose-invert prose-headings:font-semibold">
              <p>{answer}</p>
            </div>"""
            if answer
            else ""
        )
        return f"""
        <div data-message-id="m1" data-message-role="assistant" role="presentation"
             class="group relative -mb-4 flex w-fit max-w-full items-start justify-start gap-4 pb-4 leading-relaxed">
          <div class="mt-5 size-3.5 flex-none select-none rounded-full"></div>
          <div class="relative flex min-w-[60px] flex-col gap-2 rounded-2xl border px-5 py-3.5">
            <div>
              {answer_html}{thinking}
            </div>
          </div>
          <div class="absolute -bottom-3.5 right-1 flex max-w-[100%] items-center gap-0.5">
            <div class="mr-2 flex items-center gap-1.5 truncate text-gray-400">
              <span>router</span><span>with</span><span>model</span><span>via</span><span>provider</span>
            </div>
            <button class="btn" title="Copy"><svg></svg></button>
            <button class="btn" title="Retry"><svg></svg></button>
          </div>
        </div>
        {stop}
        """

    def test_answer_extracted_without_thinking_or_metadata(self, extract_page):
        result = _extract(
            extract_page,
            self._message("The answer is 4.", reasoning="Let me add 2 and 2."),
            site_id="huggingface",
        )
        assert result["found"] is True
        assert result["content"] == "The answer is 4.", result["content"]
        # The reasoning trace streams as thinking, not as the answer...
        assert result["thinking"] == "Let me add 2 and 2.", result["thinking"]
        # ...and the action bar's router metadata never leaks in.
        assert "via" not in result["content"]
        assert "router" not in result["content"]
        assert result["done"] is True
        assert result["busy"] is False
        assert result["count"] == 1

    @staticmethod
    def _pending_message(*, reasoning=None):
        """The newest turn while it is still pending, exactly as a
        live probe captured it: the container (with the loading
        ball) is already in the DOM but no answer prose has
        rendered yet -- and once reasoning streams, its viewport
        renders before the answer prose does."""
        thinking = ""
        if reasoning is not None:
            thinking = (
                """
              <div class="thinking-viewport mt-2 flex max-h-56 flex-col justify-end overflow-hidden md:max-h-80">
                <div class="prose prose-sm max-w-none text-sm leading-relaxed">
                  <p>"""
                + reasoning
                + """</p>
                </div>
              </div>"""
            )
        return f"""
        <div data-message-id="m2" data-message-role="assistant" role="presentation"
             class="group relative -mb-4 flex w-fit max-w-full items-start justify-start gap-4 pb-4 leading-relaxed">
          <svg id="ball" width="1em" height="1em" viewBox="0 0 12 12"></svg>
          <div class="relative flex min-w-[60px] flex-col gap-2 rounded-2xl border px-5 py-3.5">
            <div>{thinking}</div>
          </div>
        </div>
        """

    def test_pending_turn_reports_no_content(self, extract_page):
        """Regression: while the newest turn is still pending it has
        no prose, so a document-wide selector's last match was the
        PREVIOUS turn's answer -- the first response leaked into the
        second. The extraction must report nothing until the newest
        turn renders its own prose."""
        html = self._message("First answer.") + self._pending_message()
        result = _extract(extract_page, html, site_id="huggingface")
        assert result["found"] is False, result
        assert result["content"] == "", result
        assert result["done"] is False, result

    def test_pending_turn_thinking_scoped_to_newest(self, extract_page):
        """The previous turn's reasoning block stays in the DOM; only
        the newest turn's trace may stream as thinking."""
        html = self._message(
            "First answer.", reasoning="Old reasoning."
        ) + self._pending_message(reasoning="New reasoning.")
        result = _extract(extract_page, html, site_id="huggingface")
        assert result["found"] is False, result
        assert result["content"] == "", result
        assert result["thinking"] == "New reasoning.", result

    def test_thinking_only_phase_yields_no_content(self, extract_page):
        """While the model is still thinking there is no answer prose,
        so extraction must report no content (the trace streams as
        thinking) instead of rendering the reasoning as the answer."""
        result = _extract(
            extract_page,
            self._message("", reasoning="Still reasoning about it."),
            site_id="huggingface",
        )
        assert result["found"] is False
        assert result["content"] == ""
        assert result["thinking"] == "Still reasoning about it."
        assert result["busy"] is False

    def test_stop_control_keeps_stream_open(self, extract_page):
        result = _extract(
            extract_page,
            self._message("Partial ans", streaming=True),
            site_id="huggingface",
        )
        assert result["content"] == "Partial ans"
        assert result["busy"] is True
        assert result["done"] is False

    def test_newest_turn_wins(self, extract_page):
        html = self._message("Older reply.") + self._message("Newest reply.")
        result = _extract(extract_page, html, site_id="huggingface")
        assert result["content"] == "Newest reply."
        # Count is scoped to the response container (the newest
        # turn), so it is the prose matches inside that turn.
        assert result["count"] == 1


class TestQwenExtraction:
    """chat.qwen.ai markup, as captured live (2026-10).

    The whole turn -- status cards, answer, footer -- lives inside
    ``.qwen-chat-message.qwen-chat-message-assistant``. The OLD response
    selector ``[class*="assistant"]`` matched that whole message, so the
    in-flow thinking/status cards leaked into the answer that the
    local chat displays:

    - ``.qwen-chat-thinking-status-card`` ("Thinking completed")
    - ``.qwen-chat-status-card`` with a title
      ("Analyzing user input to determine intent and tone",
      "Refining poetic expressions to enhance elegance") and a
      ``.qwen-chat-status-card-answer-now`` button ("Skip")

    The real reasoning lives in a collapsible "Thinking and Search"
    sidebar that is hidden by default and absent from the flow DOM, so
    there is nothing to stream as thinking. The answer is the markdown
    under the answer phase.
    """

    @staticmethod
    def _message(answer, *, title=None, thinking_completed=False, skip=False):
        title_html = ""
        if title is not None:
            title_html = f"""
              <div class="qwen-chat-status-card ant-flex">
                <div class="qwen-chat-status-card-title">
                  <div class="qwen-chat-status-card-title-text">{title}</div>
                </div>
                <div class="qwen-chat-status-card-answer-now">Skip</div>
              </div>"""
        thinking_html = ""
        if thinking_completed:
            thinking_html = """
              <div class="qwen-chat-thinking-tool-status-card-wraper">
                <div class="qwen-chat-thinking-status-card-completed">
                  <div class="qwen-chat-thinking-status-card-title-text">
                    Thinking completed
                  </div>
                </div>
              </div>"""
        # The answer markdown only renders once text exists; a
        # thinking/refining turn has no .custom-qwen-markdown at all.
        answer_html = ""
        if answer:
            answer_html = f"""
              <div class="response-message-content t2t phase-answer">
                <div class="custom-qwen-markdown">
                  <div class="qwen-markdown">
                    <p>{answer}</p>
                  </div>
                </div>
              </div>"""
        return f"""
        <div class="qwen-chat-message qwen-chat-message-assistant">
          <div class="chat-response-message">
            <div class="chat-response-message-right">
              {thinking_html}{title_html}
              {answer_html}
            </div>
          </div>
        </div>
        """

    def test_answer_only_no_status_titles(self, extract_page):
        """The thinking/status card titles and the "Skip" button must
        never reach the answer that the local chat displays."""
        result = _extract(
            extract_page,
            self._message(
                "A poem about dogs.",
                title="Analyzing user input to determine intent and tone",
                thinking_completed=True,
                skip=True,
            ),
            site_id="qwen",
        )
        assert result["found"] is True, result
        assert result["content"] == "A poem about dogs.", result["content"]
        assert "Skip" not in result["content"], result["content"]
        assert "Analyzing" not in result["content"], result["content"]
        assert "Thinking completed" not in result["content"], result["content"]
        assert result["thinking"] in (None, ""), result["thinking"]

    def test_thinking_only_phase_yields_no_content(self, extract_page):
        """While the model is refining (no answer markdown yet) the
        extraction must report no content -- the status card is UI
        chrome, not a reply. ``found`` is False because the answer
        markdown has not rendered yet."""
        result = _extract(
            extract_page,
            self._message(
                "",
                title="Refining poetic expressions to enhance elegance",
                thinking_completed=True,
            ),
            site_id="qwen",
        )
        assert result["content"] == "", result
        assert result["thinking"] in (None, ""), result["thinking"]
        assert result["done"] is False, result

    def test_newest_turn_wins(self, extract_page):
        html = self._message("Older reply.") + self._message("Newest reply.")
        result = _extract(extract_page, html, site_id="qwen")
        assert result["content"] == "Newest reply.", result["content"]
        assert result["count"] == 1

    def test_loading_signal_during_generation(self, extract_page):
        """The "Stop" button and the loading lottie keep the stream
        open while the answer is still arriving."""
        result = _extract(
            extract_page,
            self._message("")
            + """
            <div class="response-loading"><div class="qwen-lottie-web"></div></div>
            <button aria-label="Stop">Stop</button>
            """,
            site_id="qwen",
        )
        assert result["content"] == "", result
        assert result["busy"] is True, result
        assert result["done"] is False, result


class TestGrokExtraction:
    """grok.com markup, as captured live (2026-10).

    Grok renders user bubbles with the SAME ``prose-chat`` class as
    assistant answers:

    ``div.message-bubble ... chat-md prose prose-chat ...
    bg-surface-user-bubble [data-testid="user-message"]``

    The OLD fallback selector ``div[class*="prose-chat"]`` matched
    that user bubble, so the user's own prompt was streamed back as
    the "assistant answer" (found=true, content="what is 2 plus 2").
    The fallback now excludes the user bubble via
    ``:not([data-testid="user-message"]):not([aria-label="You"])``.
    """

    @staticmethod
    def _user(prompt):
        return f"""
        <div class="message-bubble relative text-fg-primary min-h-7 chat-md
          prose prose-chat dark:prose-invert break-words max-w-[100%]
          bg-surface-user-bubble rounded-2xl"
          data-testid="user-message" aria-label="You">
          <p class="py-2">{prompt}</p>
        </div>"""

    @staticmethod
    def _assistant(answer):
        return f"""
        <div class="message-bubble relative text-fg-primary min-h-7 chat-md
          prose prose-chat dark:prose-invert break-words max-w-[100%]
          bg-surface-assistant-bubble rounded-2xl"
          data-testid="assistant-message">
          <p class="py-2">{answer}</p>
        </div>"""

    def test_user_prompt_not_returned_as_answer(self, extract_page):
        """A bare user bubble (no assistant reply yet) must not be
        extracted as an answer -- this is the regression that
        streamed the prompt back as the reply."""
        result = _extract(
            extract_page,
            self._user("what is 2 plus 2"),
            site_id="grok",
        )
        assert result["found"] is False, result
        assert result["content"] == "", result["content"]

    def test_assistant_answer_extracted(self, extract_page):
        result = _extract(
            extract_page,
            self._user("what is 2 plus 2") + self._assistant("Four."),
            site_id="grok",
        )
        assert result["found"] is True, result
        assert result["content"] == "Four.", result["content"]

    def test_newest_turn_wins(self, extract_page):
        html = (
            self._user("q1")
            + self._assistant("First reply.")
            + self._user("q2")
            + self._assistant("Second reply.")
        )
        result = _extract(extract_page, html, site_id="grok")
        assert result["content"] == "Second reply.", result["content"]


TENCENT_TURN = """
<div class="agent-chat__list">
  <div class="agent-chat__list__item agent-chat__list__item--ai agent-chat__list__item--last">
    <div class="agent-chat__list__item__content">
      <div class="agent-chat__list__item__checkbox"><label class="t-checkbox t-is-disabled"><span class="t-checkbox__input"></span></label></div>
      <div class="agent-chat__bubble agent-chat__bubble--ai">
        <div class="agent-chat__bubble__content">
          <div class="agent-chat__conv--ai__speech_show">
            <div class="hy-collapse">
              <div class="hy-collapse-header"><span class="hy-collapse-chevron"></span></div>
              <div class="hy-collapse-body"><div class="hy-collapse-content"><div class="scroll-content">
                <div class="hy-detail-block hy-think">
                  <div class="hy-detail-block-header">
                    <span class="hy-detail-block-header-title">Deep thinking completed (Ran for 1.2s)</span>
                  </div>
                  <div class="hy-detail-block-body"><div class="hy-detail-block-content">
                    <div class="hy-cherry-markdown"><p>First I check the arithmetic, then I answer plainly.</p></div>
                  </div></div>
                </div>
              </div></div>
            </div>
            <div class="hyc-content-md">
              <div class="hyc-common-markdown hyc-common-markdown-style">
                <p>The answer is 16.</p>
              </div>
            </div>
          </div>
          <div class="agent-chat__conv--ai__toolbar">
            <div class="agent-chat__toolbar__item agent-chat__toolbar__copy">Copy</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""

TENCENT_PENDING_THINKING = """
<div class="agent-chat__list">
  <div class="agent-chat__list__item agent-chat__list__item--ai">
    <div class="agent-chat__list__item__content">
      <div class="agent-chat__bubble agent-chat__bubble--ai">
        <div class="agent-chat__bubble__content">
          <div class="agent-chat__conv--ai__speech_show">
            <div class="hy-collapse">
              <div class="hy-collapse-header"><span class="hy-collapse-chevron"></span></div>
              <div class="hy-collapse-body" style="display:none">
                <div class="hy-collapse-content"><div class="scroll-content">
                  <div class="hy-detail-block hy-think">
                    <div class="hy-detail-block-header">
                      <span class="hy-detail-block-header-title">Thinking...</span>
                    </div>
                    <div class="hy-detail-block-body"><div class="hy-detail-block-content">
                      <div class="hy-cherry-markdown"><p>First I check the arithmetic, then I answer.</p></div>
                    </div></div>
                  </div>
                </div></div>
              </div>
            </div>
            <span class="t-loading"></span>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""

TENCENT_TURN_LOADING = """
<div class="agent-chat__list">
  <div class="agent-chat__list__item agent-chat__list__item--ai agent-chat__list__item--last">
    <div class="agent-chat__list__item__content">
      <div class="agent-chat__bubble agent-chat__bubble--ai">
        <div class="agent-chat__bubble__content">
          <div class="agent-chat__conv--ai__speech_show">
            <div class="hy-collapse">
              <div class="hy-collapse-header"><span class="hy-collapse-chevron"></span></div>
              <div class="hy-collapse-body"><div class="hy-collapse-content"><div class="scroll-content">
                <div class="hy-detail-block hy-think">
                  <div class="hy-detail-block-header">
                    <span class="hy-detail-block-header-title">Thinking...</span>
                  </div>
                  <div class="hy-detail-block-body"><div class="hy-detail-block-content">
                    <div class="hy-cherry-markdown"><p>First I check the arithmetic.</p></div>
                  </div></div>
                </div>
              </div></div>
            </div>
            <div class="hyc-content-md">
              <div class="hyc-common-markdown hyc-common-markdown-style">
                <p>The answer is </p><span class="hyc-common-markdown__loading"></span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
"""

# The newest-turn item (TENCENT_TURN's item) must be the LAST
# match inside the list, so an unscoped selector cannot return the
# older turn's answer. Build it programmatically: strip the list
# wrapper off TENCENT_TURN and sandwich the older "First reply."
# turn in front of it.
_TENCENT_TURN_ITEM = (
    TENCENT_TURN.strip()
    .removeprefix('<div class="agent-chat__list">')
    .removesuffix("</div>")
    .strip()
)
TENCENT_TWO_TURNS = (
    '<div class="agent-chat__list">'
    + '\n  <div class="agent-chat__list__item agent-chat__list__item--ai">'
    + '\n    <div class="agent-chat__list__item__content">'
    + '\n      <div class="agent-chat__bubble agent-chat__bubble--ai">'
    + '\n        <div class="agent-chat__bubble__content">'
    + '\n          <div class="agent-chat__conv--ai__speech_show">'
    + '\n            <div class="hyc-content-md">'
    + '\n              <div class="hyc-common-markdown hyc-common-markdown-style">'
    + "\n                <p>First reply.</p>"
    + "\n              </div>"
    + "\n            </div>"
    + "\n          </div>"
    + "\n        </div>"
    + "\n      </div>"
    + "\n    </div>"
    + "\n  </div>"
    + _TENCENT_TURN_ITEM
    + "\n</div>"
)

TENCENT_MODEL_DETAILS = """
<div class="all-chatlist-wrapper">
  <div class="model-desc-wrapper">
    <div class="mvfqmiYoVreC1bYK_YyC VqtPtlVR7qtmDEsq01_H">
      <div class="t0RzcsGrBHWFkFBdssj6"><h3>Model Details</h3></div>
      <div class="qjrgrCxYd76NfBWeWOem">
        <p class="nOo3ZxtBTAzyo04FTbsu">Hy4 preview features 770B total parameters
        with 49B active parameters and is optimized for agentic coding scenarios.</p>
      </div>
    </div>
  </div>
</div>
"""

TENCENT_TURN_WITH_MODEL_DETAILS = TENCENT_TURN + TENCENT_MODEL_DETAILS


class TestTencentExtraction:
    def test_answer_and_thinking_are_separated(self, extract_page):
        """Streaming regression: thinking must not leak into the answer."""
        result = _extract(extract_page, TENCENT_TURN, "tencent")
        assert result["found"] is True
        assert result["content"] == "The answer is 16."
        assert "Deep thinking completed (Ran for 1.2s)" in (result["thinking"] or "")
        assert "First I check the arithmetic" in (result["thinking"] or "")
        # Turn chrome must never be the reply.
        assert "Copy" not in result["content"]
        assert "checkbox" not in result["content"]

    def test_reasoning_phase_reports_busy(self, extract_page):
        """A turn that is still reasoning (no answer yet) reports
        found=true with empty content, no thinking and busy=true.

        This is what a fresh tencent tab shows for several seconds
        while the model reasons. The speech-area fallback in
        response_selectors keeps `found` true so the reasoning-phase
        spinner counts as a node INSIDE the response and `busy`
        stays true -- the poller then keeps waiting (neither the
        idle rule nor the done rule fires while content/thinking
        are empty) instead of ending the turn early. The
        response_container scoping is what stops a previous turn's
        answer from being pulled as this turn's reply here: an
        unscoped scan would match the older turn's prose.
        """
        result = _extract(extract_page, TENCENT_PENDING_THINKING, "tencent")
        assert result["found"] is True
        assert result["content"] == ""
        assert result["thinking"] is None
        assert result["done"] is False
        assert result["busy"] is True

    def test_thinking_keeps_busy_during_reasoning(self, extract_page):
        """The reasoning-phase spinner sits inside the speech area,
        so busy stays true and the poller keeps waiting instead of
        ending the turn at the idle threshold. `found` is true and
        `content` empty during the reasoning phase; a non-blank
        capture only appears once the answer starts streaming.
        """
        result = _extract(extract_page, TENCENT_PENDING_THINKING, "tencent")
        assert result["busy"] is True

    def test_answer_loading_dot_keeps_busy(self, extract_page):
        """The cursor dot lives inside .hyc-content-md while the answer
        streams, so busy stays true until the answer is fully on screen."""
        result = _extract(extract_page, TENCENT_TURN_LOADING, "tencent")
        assert result["found"] is True
        assert result["busy"] is True
        assert result["done"] is False

    def test_model_details_panel_is_not_thinking(self, extract_page):
        """The page's obfuscated .mvfqmiYoVreC1bYK panel is a model-spec sidebar
        (770B parameters), NOT a thinking block. It must never pollute
        thinking_content, so we never match the hy-think element by accident.

        This is the exact shape the user reported as "thinking block" and it
        is deliberately NOT in thinking_selectors -- the real trace is
        .hy-detail-block.hy-think.
        """
        result = _extract(extract_page, TENCENT_MODEL_DETAILS, "tencent")
        assert result["found"] is False
        assert "770B" not in (result["content"] or "")
        assert "770B" not in (result["thinking"] or "")

    def test_model_details_mixed_with_a_turn_is_excluded(self, extract_page):
        """When the model-spec sidebar coexists with a turn, extraction must
        return that turn's answer + thinking and nothing from the sidebar."""
        result = _extract(extract_page, TENCENT_TURN_WITH_MODEL_DETAILS, "tencent")
        assert result["found"] is True
        assert result["content"] == "The answer is 16."
        assert "770B" not in result["content"]
        assert "770B" not in result["thinking"]
        assert "First I check the arithmetic" in (result["thinking"] or "")

    def test_newest_turn_wins_no_leak(self, extract_page):
        """A turn container keeps every turn in the DOM, so the last match must
        be the newest turn's answer (this is exactly the regression the
        response_container=.agent-chat__list__item--ai fix addresses)."""
        result = _extract(extract_page, TENCENT_TWO_TURNS, "tencent")
        assert result["content"] == "The answer is 16.", result["content"]
        assert "First reply." not in result["content"]
        assert "First reply." not in (result["thinking"] or "")

    def test_thinking_contains_header_label(self, extract_page):
        """The disclosure header label (e.g. "Deep thinking completed (Ran
        for 1.2s)") is part of the relayed trace, not dropped. It is
        informative rather than chrome."""
        result = _extract(extract_page, TENCENT_TURN, "tencent")
        assert "Deep thinking completed" in (result["thinking"] or "")
