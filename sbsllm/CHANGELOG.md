# Changelog

## Unreleased

### Changed

- **Blocking popups now fail fast with a "temporarily
  unavailable" error (all services).** A visible modal dialog
  (`role="dialog"[aria-modal="true"]` or `role="alertdialog"`)
  blocks the composer, so a request that arrives while one is
  up used to inject into a page the user cannot see and hang
  until the timeout. Every service now carries the shared
  `popup_selectors` default (extend it per-site, or set
  `"popup_selectors": []` to opt out), checked twice:
  (1) before inject+submit — the request ends immediately
  with a 503 naming the popup's text; (2) on every poll —
  the shared `ResponsePoller` stops with a `popup` stop
  reason the moment a dialog appears mid-generation (a
  503 on the non-streaming path; an SSE error payload on
  the streaming path, whose HTTP status is already
  committed; a per-tab 503 that fails the whole `multi`
  request, matching the strict fan-out failure semantics).
  A popup that arrives after the turn completed (a cookie
  banner popping up once the answer is on screen) blocks
  nothing, so that answer is returned rather than failed.
  Hidden-but-blocking dialogs (copilot's security check,
  whose enter transition never completes under automation)
  stay in `login_wall_modal_selectors` and keep the 502
  login-wall error.

- **Chromium now launches with GPU acceleration disabled (no VRAM).**
  The chat tabs only render text, images and video, so the browser
  runs fully on the CPU: `--disable-gpu`,
  `--disable-gpu-compositing`, `--disable-gpu-rasterization`,
  `--disable-accelerated-2d-canvas`,
  `--disable-accelerated-video-decode`,
  `--disable-features=VaapiVideoDecoder,VaapiVideoEncoder` and
  `--use-gl=swiftshader` are passed on every launch. WebGL then
  reports the SwiftShader software device and WebGPU is unavailable,
  so nothing is allocated on the graphics card. SwiftShader itself
  stays enabled (disabling the software rasterizer too would leave
  pages unpaintable). Set `SBSLLM_GPU=1` to restore hardware
  acceleration. Verified by launching Chromium with the flags and
  reading the unmasked WebGL renderer, plus launch-arg tests.

### Fixed

- **Every request force-switched the visible browser tab.**
  `_do_inject_and_submit` called `page.bring_to_front()`
  before injecting, so with several chats open each request
  yanked its own tab to the front — the "copilot steals
  focus" report was our own code, not the site (copilot.com
  never calls `window.focus()`). Playwright drives
  background pages fine (the inject JS already focuses the
  composer element itself), so the call and its settle
  sleep are gone; tabs are now driven in the background.

- **The login-wall 502 could never fire.** The extraction
  JS reports `login_wall`, but `_normalize_response`
  whitelisted fields and dropped it, so the server's
  `response.get("login_wall")` was always false and a
  logged-out chat degraded to a generic 504 instead of
  the actionable "requires a signed-in session" error.
  `login_wall` (and the new `popup`/`popup_text`) are
  now passed through normalization.

- **Copilot sometimes never spawned a new chat.** copilot.com
  intermittently raises a bot check: a modal
  `role="dialog" aria-modal="true" aria-label="Security check
  required"` ("Verification required"). While it is up, the
  composer `div[data-testid="composer-input"]` and the send
  button are absent from the DOM (the editor that remains is a
  `span.fai-EditorInput__input`), so inject and submit still
  reported `OK` against unrelated elements and no turn ever
  started -- the request hung until the timeout and returned a
  bare 504. Two fixes: (1) the dialog is now reported as a
  login wall, so the server fails fast with the actionable 502
  ("complete any security check") instead. The dialog keeps
  computed `visibility: hidden` (its enter transition never
  completes under automation), so the new
  `login_wall_modal_selectors` site key matches it WITHOUT the
  visibility gate -- only a dialog whose mere presence blocks
  the page may be listed there. (2) The inject cascade now
  targets the live editor (`span.fai-EditorInput__input`, then
  the tag-agnostic `[contenteditable="true"][role="textbox"]`)
  before the generic `[contenteditable]` fallback, which could
  otherwise grab whatever contenteditable comes first in
  document order (sidebar search, a "new chat" title input)
  depending on page state. Verified against the live page
  (extraction reports `login_wall: true` while the check is
  up, inject marks the editor span) plus extraction and
  inject-cascade regression tests.
- **meta.ai stopped forwarding prompts after the first message.**
  After the first send, meta.ai navigates to `/prompt/<uuid>`, which
  renders a "Conversation title" input (`input[type="text"]`) next to
  the composer. The meta inject selector list contained that generic
  `input[type="text"]`, and it matched the title field *before* the
  `textarea` / `[contenteditable]` fallbacks — so the second prompt
  was typed into the conversation title instead of the composer. The
  site then re-sent the previous turn's still-present text (a
  duplicated answer), and once the composer had cleared, the third
  prompt was never sent at all: Send stayed disabled and the local
  chat read the stale answer (or waited it out). The selector list now
  only matches the composer: the aria-labelled input, the
  "Ask Meta AI" placeholder on input/textarea, the visible
  contenteditable, then generic textareas. Verified against the live
  site (three sequential turns each forwarded and extracted their own
  answer) and with a mock `/prompt/<uuid>` page regression test
  asserting the prompt never reaches the title input.
- **Meta AI replies were never relayed to the local chat.** The assistant
  message row (`group/assistant-message`,
  `data-testid="assistant-message"`) shares its class substring with the
  action bar (`group/assistant-message-actions`), so the
  `[class*="assistant-message"]` catch-all matched both and selection —
  which takes the last match — landed on the icon-only Like/Dislike/Copy
  buttons, yielding empty content. Response selection now prefers the
  precise `data-testid` and the catch-all excludes the action bar, and the
  message row's own `data-streaming-state` / `data-streaming-complete`
  attributes now drive `busy` so long answers are not cut at the first idle
  window. Verified with extraction tests built from the live-site markup
  (answer extraction, streaming busy signal, newest-message selection).
- **Extracted answers lost their paragraph breaks.** The extraction pipeline
  squashed every whitespace run into a single space, and the pruning path
  read a detached clone with `textContent`, which carries no layout-derived
  newlines — so a multi-paragraph reply reached the local chat as one run-on
  line (`...meet you!Is there...`). Extraction now normalises whitespace per
  line (`normalize`), collapsing blank-line runs instead of destroying them,
  and reads detached clones with a block-boundary walker (`blockTextOf`) that
  emits real newlines for block elements while collapsing source line breaks
  inside a text node. Verified with Chromium-backed extraction tests covering
  the pruned and unpruned paths, `<br>` handling, and source-wrapped
  paragraphs.
- **Site re-renders duplicated everything already streamed.** Content deltas
  used `str.removeprefix`, which does not match when a site re-renders
  already-sent text mid-stream (z.ai dropped markdown emphasis: `I'm **GLM`
  became `I'm GLM`); the whole new content was then appended after what the
  client already had, doubling the reply. `_stream_delta` now skips the
  shared prefix plus the longest run of new content that already occurs in
  the old content (binary-searched; the predicate is monotonic), so only
  unseen text is emitted and nothing is lost. Verified with unit tests for
  the diff and streaming tests replaying a re-render sequence.
- **Duplicate prompts were re-posted to the same tab in a loop.** One
  `POST /v1/chat/completions` injects and submits exactly once, but clients
  that retry silently (an SSE reconnect behind a proxy, a fetch/proxy retry,
  a frontend retry policy) produce a train of identical POSTs — each one
  re-injecting and re-submitting the same prompt, which surfaced as the
  prompt repeating forever in the browser tab. A per-tab duplicate-submit
  guard (`_TurnRegistry`) now records the last successful submission per
  model; an identical prompt within `duplicate_prompt_cooldown` (default
  60s, `0` disables) attaches to the turn already on screen instead of
  being sent again: its finished answer is returned directly (streaming and
  non-streaming), and an in-flight one is followed without re-injecting.
  Each suppressed duplicate slides the cooldown window forward, so a retry
  train cannot out-wait the guard, while a deliberate re-send of the same
  prompt works again once the retries stop. Verified with unit tests for
  the registry (per-model scope, cooldown expiry, window sliding) and
  handler tests asserting one submission total across a retry train in
  both the streaming and non-streaming paths.
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
- **`capture_response` retries `BrowserWorkerStopped`.** A dead worker
  mid-dispatch used to abort the whole request with a 502, even though the
  operation is read-only (no prompt submission) and safe to re-run: the old
  queue item was drained on restart and a fresh worker is started
  transparently. The retry loop now also catches `BrowserWorkerStopped` and
  retries with backoff, turning a transient worker death into a recoverable
  blip. `BrowserOperationTimeout` (wedged, still-alive worker) is still never
  retried. New tests: `test_browser_worker_stopped_retried`,
  `test_browser_worker_stopped_exhausts_retries`.
- **`SBSLLM_HEADLESS` env var** (`sbsllm/browser.py`): defaults to a visible
  browser so users can log in interactively, but CI / container environments
  can set `SBSLLM_HEADLESS=true` to run headless. Verified with unit tests.
- **README kimi URL corrected** (`README.md`): was `https://www.kimi.com/`,
  now `https://kimi.ai/` to match `sites.py`.

- **Microsoft Copilot support** (`sbsllm/sites.py` + tests): a new `copilot`
  site (https://copilot.com/) was added — textarea composer, send-button
  submit, turn-marker response selection (`data-content` /
  `data-message-author-role` fallback cascade), exclusion of citation /
  feedback / suggested-action chrome, thinking/answer separation, and
  `TestCopilotExtraction` (5 Chromium-backed extraction tests). Note: the
  consumer chat is auth-gated, so the site is enabled in `config.yaml`
  after signing in.

- **Per-tab locks** (`_ModelLockRegistry`): different chats run in parallel,
  same chat serializes.
- **New knobs**: `first_token_timeout=60s`, `busy_patience=20s`,
  `keepalive_interval=10s`, `browser_timeout=180s`, `browser_lock_timeout=300s`,
  `response_idle_timeout=3.0s`.
- **Reasoning-leak test coverage** (`tests/test_extract.py`, `tests/test_streaming.py`):
  complete thinking+answer flow, prune-not-replace, z.ai loading dots.

## Previous

- Initial public release.