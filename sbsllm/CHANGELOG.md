# Changelog

## Unreleased

### Fixed

- **Non-streaming path had no test coverage for success or error branches.**
  The `wait_for_response` → `_handle_chat_completions` path (non-`stream`
  requests) was only covered for inject/submit failures. Five new tests now
  exercise the success branch (200 with content, 200 with thinking) and the
  error branches (504 timeout, 504 no-response, 502 login wall), matching the
  streaming path's coverage. Coverage for `server.py` rose from 77% to 80%.
- **z.ai double-send.** The `post_inject_js` block dispatched Enter key
  events on the textarea, which triggered the form's native submit handler;
  then `submit_js` clicked the send button again, so the prompt was sent
  twice. `post_inject_js` now only verifies the marked input exists and
  returns `OK` without dispatching any key events. The `submit_js` comment
  documents the trap for future maintainers. Verified with a regression test
  asserting the form's submit handler is not triggered by `post_inject_js`,
  plus an end-to-end streaming test that exercises inject → submit → growing
  thinking → answer extraction against a real browser DOM.
- **Grok contenteditable editor not receiving prompts.** The inject template
  now dispatches a `beforeinput` event for contenteditable/TipTap editors,
  which ProseMirror listens on to update its internal document model. A
  `post_inject_js` block was also added to grok so a fresh `input` event fires
  after the main inject IIFE, which some TipTap versions ignore. Verified with
  a mock grok page (TipTap contenteditable + send button + response bubble).
- **Google AI Studio content inserted but Run button stayed disabled.** The
  `ms-autosize-textarea` wrapper's internal `data-value` attribute was never
  synced when `.value` was set via the property setter, so the Run button
  stayed disabled and the prompt was never sent. A `post_inject_js` block now
  syncs the wrapper's `data-value` and re-dispatches `input`/`change` events.
  Verified with a mock Google AI Studio page (ms-* web components).
- **`post_inject_js` separator missing.** When a site defines a
  `post_inject_js` block, the two IIFEs were concatenated without a
  semicolon, so the post-inject IIFE was parsed as calling the return value
  of the main inject IIFE (`})()` followed by `(() =>`), which threw
  `TypeError: ... is not a function`. A semicolon separator is now inserted
  between the two blocks.
- **Non-streaming path returned a fake "Failed to inject/submit" assistant
  message** when injection or submission failed, which the local chat
  rendered as the model's answer. The streaming path already returned a 502;
  the non-streaming path now does too, with the same `inject_submit_failed`
  error type. Verified with unit tests asserting the 502 and the error
  message.
- **Thinking-only stalls were truncated at 20s.** A site that is still
  generating with an empty answer but a growing reasoning trace used the
  generic `busy_patience` (20s), so a thinking model (o1, deepseek-reasoner,
  Claude-thinking) that thinks for minutes got cut off mid-thought. A new
  `thinking_patience` (default 120s) is used for stalls where `content` is
  empty but `thinking` is present; answer stalls still use `busy_patience`.
  New `thinking_timeout` stop reason is surfaced in the empty-stream detail
  message. Both the streaming and non-streaming paths use it.
- **z.ai thinking leaked into content.** The extraction prune fallback
  (`if prunedText) content = prunedText`) fell back to the unpruned text during
  z.ai's empty thinking phase, injecting reasoning into `content`. The local chat
  rendered thinking twice. Pruned text is now used whenever anything was pruned
  (empty included); the over-broad safety net is replaced with an explicit
  answer-host check.
- **Silent hang.** `browser_timeout` was 600s with no first-token or busy-patience
  guards, and the idle fallback was dead code. A selector mismatch or stuck
  spinner meant 10 minutes of silence holding the tab lock. The streaming loop
  now guarantees termination via `stop_reason` (`site_done|idle|busy_timeout|
  no_output|budget_exhausted`) and emits SSE keepalives so clients never time out.
- **`busy_patience` guard couldn't fire.** The change timestamp was updated on
  `busy` as well as real changes, so a stuck spinner kept `stable_for ≈ 0` forever.
- **Selectors overridden.** Extraction merged all response selectors and took the
  LAST match, so catch-all selectors (`div[id*=message]`) beat the precise
  `#response-content-container .markdown-prose`. Ordered fallbacks now take the
  first match.
- **z.ai loading dots not detected.** The animated dot loader (`.dot` elements
  inside `#response-content-container`) matched no loading selector, so
  `busy=False` even while z.ai was generating. This made the server see
  `done=True` immediately and terminate the stream before any content arrived.
  Added z.ai-specific loading selectors (`#response-content-container .dot`,
  `.skeleton.loading`).
- **Second divergent `_is_new_response` in `browser.py`.** It missed the thinking
  check, so the non-streaming path never recognised thinking-only turns. The
  predicate is now shared and thinking-aware.

### Added

- **Retry/backoff in `capture_response`.** A transient Playwright error
  (page mid-navigation, connection retry, detached frame) used to abort the
  whole request with a 502. The poll loop now retries inside the capture call
  with a short exponential backoff, so momentary hiccups don't surface to the
  client. `BrowserOperationTimeout` (wedged worker) and non-transient errors
  are not retried — they propagate immediately so the request ends.
- **Headless CI integration test** (`tests/test_integration_browser.py`).
  Boots a real Chromium against a mock z.ai-style page and runs the *real*
  inject/submit/extraction JS against live DOM APIs. This closes the
  end-to-end verification gap that CAPTCHA and login walls left on live AI
  sites, covering: page load, inject template, submit template, extraction of
  answers, thinking/answer separation, loading-dots busy detection, and a full
  inject→submit→extract cycle.
- **`capture_response` retry unit tests** (`tests/test_browser.py`):
  transient-error retry, connection-error retry, no-retry on
  `BrowserOperationTimeout`, no-retry on non-transient errors, and
  exhaustion-then-raise.
- **`SBSLLM_HEADLESS` env var** (`sbsllm/browser.py`): defaults to a visible
  browser so users can log in interactively, but CI / container environments
  can set `SBSLLM_HEADLESS=true` to run headless. Verified with unit tests.
- **README kimi URL corrected** (`README.md`): was `https://www.kimi.com/`,
  now `https://kimi.ai/` to match `sites.py`.

- **Per-tab locks** (`_ModelLockRegistry`): different chats run in parallel,
  same chat serializes.
- **New knobs**: `first_token_timeout=60s`, `busy_patience=20s`,
  `keepalive_interval=10s`, `browser_timeout=180s`, `browser_lock_timeout=300s`,
  `response_idle_timeout=3.0s`.
- **Reasoning-leak test coverage** (`tests/test_extract.py`, `tests/test_streaming.py`):
  complete thinking+answer flow, prune-not-replace, z.ai loading dots.

## Previous

- Initial public release.