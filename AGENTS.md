sbsllm (side by side llm), is an app that lets the user test the same prompt against multiple llm chats (websites) at the same time.

The app starts a very minimal web browser (Chromium via Playwright) to display AI chat websites.
The browser is controlled using Playwright to open the predefined ai chat apps.
The user is prompted to login in those ai chat apps.
The prompt is sent from cli with fanout to all the opened browser pages.
The supported chats are selected using a config file.

The apps supports:
# chatgpt
https://chatgpt.com/
# claude
https://claude.ai/
# deepseek
https://chat.deepseek.com
# qwen
https://chat.qwen.ai/
# grok
https://grok.com/
# google
https://aistudio.google.com/
# mistral
https://chat.mistral.ai
# kimi
https://www.kimi.ai/
# zai
https://chat.z.ai/
# meta
https://meta.ai/
# huggingface
https://huggingface.co/chat
# copilot
https://copilot.com/
# tencent
https://aistudio.tencent.ai/

## Architecture

sbsllm exposes an OpenAI-compatible HTTP server (`sbsllm/server.py`) that accepts
`POST /v1/chat/completions` requests. Each model name in the config maps to a
browser tab on a specific chat website (configured via `model_map`).

### Prompt flow

1. The server extracts only the **latest user message** from the OpenAI messages
   array (`_build_web_prompt`).
2. Meta-instruction wrappers from local chats (open-webui, ChatGPT web, etc.)
   are stripped by `_strip_meta_tags`, which recognises:
   - `<chat_history>USER: ... ASSISTANT: ... USER: ...` blocks
   - `[INST] <<SYS>>...<</SYS>> ... [/INST]` (open-webui LLaMA format)
   - Plain `USER:` / `ASSISTANT:` turn blocks
3. Only the actual user message is sent to the web chat via Playwright DOM
   injection (`inject_prompt`). The browser tab retains its own conversation
   history, so prior turns are not re-sent.
4. The web chat's response is extracted via site-specific selectors
   (`response_selectors`, `thinking_selectors`, `loading_selectors`) and
   streamed back to the caller.

### Streaming

- `_stream_web_chat` polls the browser DOM at intervals (`poll_interval`).
- New content deltas are sent as SSE `chat.completion.chunk` events.
- Thinking content is streamed as both `thinking` and `reasoning_content`
  (OpenAI-compatible) deltas.
- Completion is detected via the site's `done` flag, idle timeout, busy
  patience, thinking patience, and first-token timeout.

### Known fixes

- **Meta chat**: response selection prefers `data-testid="assistant-message"`.
  The `[class*="assistant-message"]` catch-all also matched the action bar
  (`group/assistant-message-actions`) and, being the last match, won — so
  content was read from the icon-only action buttons (always empty) and
  replies never reached the local chat. The catch-all now excludes the action
  bar, and `data-streaming-state` / `data-streaming-complete="false"`
  attributes on the message row signal `busy` while generating.
- **Meta chat**: after the first send the site navigates to
  `/prompt/<uuid>`, which renders a "Conversation title" input
  (`input[type="text"]`). The inject selector list contained that generic
  `input[type="text"]`, which matched the title field before the
  `textarea`/`[contenteditable]` fallbacks — from the second message on,
  the prompt was typed into the conversation title (the site then
  re-sent the previous turn's text, and later turns sent nothing).
  The selector list now matches only the composer (aria-labelled
  input, "Ask Meta AI" placeholders, the visible contenteditable,
  generic textareas).
- **Extraction keeps paragraph breaks**: answers are normalised with
  `normalize()` (per-line squash, blank runs collapsed) instead of a
  whole-text `squash()`, and detached clones are read with `blockTextOf()`
  (block-boundary newlines, whitespace collapsed inside text nodes) because
  `textContent` carries no layout-derived newlines. Multi-paragraph replies
  no longer arrive as one run-on line.
- **Streaming deltas survive site re-renders**: `_stream_delta` in
  `server.py` replaces `str.removeprefix`. When a site re-renders
  already-streamed text (z.ai dropped `**` mid-stream: `I'm **GLM` became
  `I'm GLM`), the diff skips the shared prefix plus the longest run of new
  text already present in the old content instead of re-sending the whole
  message.
- **Google AI Studio**: `inject` and `submit_js` now traverse Shadow DOM roots
  of `ms-prompt-box` / `ms-autosize-textarea` / `ms-run-button` web components.
- **Google AI Studio send**: the prompt was inserted but never sent. Two
  compounding bugs: (1) `post_inject_js` located the `ms-autosize-textarea`
  host with `el.closest()`, which cannot cross shadow boundaries, so the
  `data-value` sync silently no-oped and the Run button stayed disabled;
  the host is now found via a depth-first shadow scan (`shadowRoot.contains(el)`),
  because the marked textarea lives in `ms-autosize-textarea`'s shadow
  root, itself nested inside `ms-prompt-box`'s — a single-level scan of
  the document's shadow hosts never reaches it. The same recursive scan
  replaces the single-level marker search in the generic submit template,
  the contenteditable/Lexical post-inject blocks, and Google's own
  `post_inject_js`/`submit_js`. (2) With the button
  disabled, the generic submit template fell back to dispatching *plain*
  Enter, which only inserts a newline in AI Studio (its send key is
  Ctrl+Enter). Google's `submit_js` now clicks the enabled Run button
  (shadow-aware) and otherwise falls back to a `composed` Ctrl+Enter
  keydown on the marked textarea. Response relay verified: extraction
  selectors target `ms-chat-turn` Model turns.
- **Google AI Studio turn chrome in answers**: the relayed answer
  carried the turn's UI chrome (`edit`, `more_vert`, `Model
  1:32 PM`, `thumb_up`, `thumb_down`). Two layers fix it: (1) the
  primary `response_selectors` entry scopes to
  `.chat-turn-container.model .turn-content` — the turn header
  (model name + timestamp), options menu and feedback bar render
  outside it; (2) a new per-site `response_exclude_selectors`
  (threaded through `_response_js`/`extract_js` into
  `_RESPONSE_TEMPLATE`) prunes buttons, `ms-chat-turn-options`,
  `.turn-footer`, Material Symbols icon ligatures and
  header/timestamp elements from a clone of the response before its
  text is read — covering the fallback path where the whole turn
  container matches. Chrome is never the answer, so unlike the
  reasoning prune there is no `wrapsAnswer` guard.
- **Google AI Studio "Thinking" label in answers**: the relayed
  answer started with a glued `Thinking` word. The reasoning trace
  lives in the `ms-thought-chunk` web component (a tag, not a
  class), so the old `thinking_selectors`
  (`[class*="thinking"]`/`[class*="reasoning"]`) never matched it
  and a collapsed chunk — which renders only its label — leaked
  into the answer. Google's `thinking_selectors` now lead with
  `ms-thought-chunk`/`.thought-panel`, so the chunk is captured as
  thinking (the bare label is filtered by the existing
  label-only check) and pruned from the answer clone. A template
  safety net also handles labels that survive the prune: a
  leading `Thinking`/`Thoughts`/`Thought`/`Reasoning` label is
  stripped when the next non-space character is an uppercase
  letter, a digit, or nothing at all; and a line that is nothing
  but a disclosure label (a lingering "Thinking" status chip
  rendered outside any thinking selector) is dropped whole. The
  leading-label regex carries no `/i` flag on purpose: with it,
  the lookahead's `[A-Z0-9]` matches case-insensitively and a
  legitimate answer like "Thinking about the sun" would lose
  its first word — a lowercase continuation marks a real answer
  and is left untouched.
- **Zai chat**: URL changed to `https://chat.z.ai/` to avoid auth wall;
  `login_wall_selectors` narrowed to exclude nav sign-in button false-positive;
  double-send from Enter key dispatch in `post_inject_js` eliminated.
- **Grok chat**: `inject` and `submit_js` now fall back to scanning shadow roots
  for the TipTap/ProseMirror editor and send button.
- **Kimi URL**: Corrected to `https://www.kimi.ai/`.
- **HuggingFace chat**: chat-ui marks assistant turns with
  `data-message-role="assistant"` (NOT `data-message-author-role`)
  and renders the answer in a `div.prose` whose class list carries no
  `prose-sm`; the reasoning viewport's prose does. The old selectors
  (`data-message-author-role`, `.assistant-message`,
  `[class*="assistant"])` never matched, so only the thinking trace
  streamed and the answer never reached the local chat. The action
  bar (router metadata, copy/retry) lives inside the turn, so a
  whole-turn selector would leak "route with <model> via <org>" into
  every answer. The extraction is also scoped to the newest turn via
  `response_container` (`[data-message-role="assistant"]`): chat-ui
  keeps every turn in the DOM, so while the newest turn is still
  pending (container rendered, no prose yet) an unscoped last-match
  selector returned the PREVIOUS turn's finished answer -- the first
  response leaked into the second. With the container, a pending turn
  reports `found=false` until its own prose appears, and reasoning
  blocks are read only inside the newest turn. Also: the composer
  placeholder is "Ask anything", and
  the first-load welcome overlay's only button ("Start chatting")
  redirects to login, so `setup_js` dismisses it with Escape (Modal
     listens for keydown on window) instead of clicking.
- **Qwen chat**: the whole turn (status cards + answer + footer) sits
  inside `.qwen-chat-message-assistant`. The old response selector
  (`[class*="assistant"]`) matched that whole message, so the in-flow
  thinking/status cards -- "Analyzing user input to determine intent
  and tone", "Refining poetic expressions...", "Thinking completed" --
  and the "Skip" control (`.qwen-chat-status-card-answer-now`) leaked
  into the answer shown in the local chat. Those cards are UI chrome
  (the real reasoning is in a collapsible "Thinking and Search"
  sidebar that is hidden by default and absent from the flow DOM), so
  they are not streamed as thinking. The extraction is scoped to the
  answer markdown under the answer phase
  (`.response-message-content.phase-answer .custom-qwen-markdown`) and
  to the newest turn's container; the busy signal is the "Stop" button
  (`button[aria-label*="Stop"]`, already in the shared loading set),
  which Qwen shows for the whole generation including the long
  thinking/refining phase that precedes the answer.
- **Browser worker lifecycle**: `_stop_browser_worker` no longer enqueues a
  `None` shutdown sentinel when the worker is already dead (it left the
  sentinel in the shared `_browser_queue`, so the next worker dequeued it and
  exited immediately, surfacing as `RuntimeError("Browser worker stopped
  before completing the operation")`). `_start_browser_worker` now drains stale
  queue items before spawning a fresh worker, so a dead worker is restarted
  transparently instead of failing the caller.
- **Duplicate prompt loop**: one HTTP request injects and submits exactly
  once, but clients that retry silently (SSE reconnect, fetch/proxy retry)
  re-POST the identical request, and each POST re-sent the prompt — visible
  as the same prompt repeating forever in the tab. A per-tab duplicate-submit
  guard (`_TurnRegistry`, `duplicate_prompt_cooldown`, default 60s, `0`
  disables) now attaches an identical prompt within the cooldown window to
  the turn already on screen (returning its finished answer directly, or
  following an in-flight one) instead of re-injecting. Suppressed duplicates
  slide the window forward, so a retry train cannot re-send the prompt; a
  deliberate re-send works again once the retries stop.

- **Grok user-bubble leak**: Grok renders user bubbles with the same
  `prose-chat` class as assistant answers
  (`div.message-bubble ... prose prose-chat [data-testid="user-message"]`),
  so the `div[class*="prose-chat"]` fallback returned the user's own
  prompt as the "assistant answer". The fallback now excludes the user
  bubble (`:not([data-testid="user-message"]):not([aria-label="You" i])`).
- **Kimi Lexical composer**: Kimi's editor is Lexical, which keeps its
  own document model — the shared contenteditable sync
  (innerText clear + `execCommand('insertText')`) appended to the model
  instead of replacing it (prompt duplicated 6x), and Lexical ignores
  synthetic `beforeinput insertText`. Kimi now has a custom
  `post_inject_js` (`_KIMI_POST_INJECT_LEXICAL`) that clears the model
  with a synthetic Ctrl+A + Delete keydown (Lexical's keymap handles the
  native selection-clearing; `execCommand('delete')` alone does NOT clear
  Lexical — it returns true but the model keeps its nodes) and then
  writes exactly one copy via `execCommand('insertText')`. The earlier
  paste-event approach was silently ignored by headless Chrome
  (`ClipboardEvent('paste', { clipboardData })` is a no-op), so the paste
  handler fired but inserted nothing and the prompt still accumulated.
  Verified against the live editor: running the full inject+post_inject
  cycle five times in a row leaves the composer holding the prompt
  exactly once. The clear runs both paths (real selection +
  Ctrl+A+Delete) so it also works on plain contenteditable editors and
  the mock tests. Kimi's send control is also a div
- **Kimi thinking trace leaking into the answer**: Kimi renders the
  thinking chain in a `.markdown-container.toolcall-content-text` inside
  `.thinking-container`, and the answer in a sibling `.markdown-container`
  (both hold a `.markdown` div). The old `response_selectors` fell back to
  `[class*="assistant"] .markdown`, which matched the thinking node first
  (it has no `.thinking-container` ancestor relative to itself), so the
  thinking text was relayed as the answer. Fixed: the fallback is now
  `.markdown-container:not(.toolcall-content-text) .markdown`, which
  matches the answer's container and skips the thinking one. Verified
  against the live page across a full streaming poll: the answer and
  thinking stay separate at every tick, including mid-stream when only
  the thinking `.markdown` exists yet.
- **Kimi thinking trace relayed once per step**: Kimi renders the
  thinking chain as a sequence of sibling `.toolcall-content-text`
  blocks inside one `.thinking-container`. The old `thinking_selectors`
  (`[class*="thinking"]` etc.) matched each block separately, so the trace
  was forwarded once per step and the local chat showed the same
  paragraph repeated. Fixed: the selectors now lead with
  `.toolcall-container.thinking-container`, whose whole subtree is read
  in one pass. (The container's title span — "Thinking complete" vs
  "Thinking" — distinguishes multi-step traces, so two sibling containers
  with different titles are correctly relayed as two parts; only
  identical siblings are deduplicated by the existing `thinkingParts`
  check.)
- **Kimi previous-turn answer leaking into the next turn**: Kimi keeps
  every turn in the DOM, so the last answer `.markdown` belongs to the
  PREVIOUS turn while the current one is still thinking (it has a thinking
  block but no answer yet) — the previous turn's poem was relayed as this
  turn's answer, which is why the CIA turn's reply carried the Majin Buu
  poem. Fixed: Kimi now sets `response_container` to `.chat-content-item`,
  so the extraction scopes to the newest turn and a pending turn reports
  `found=false` until its own answer `.markdown` appears. This is the same
  framework mechanism qwen uses for the "first response leaks into the
  second" bug.
  (`.send-button-container`), not a `<button>`, so `submit_js` now clicks
  it directly. Note: kimi.ai now requires sign-in to send (the send
  handler opens a login modal).
- **Tencent send control**: the send control is a div
  (`div.hy-chat-input-send-btn`), not a `<button>`, so the shared
  submit template's button candidates never matched and every request
  fell back to a synthetic Enter the composer ignores. `submit_js` now
  prefers the enabled div (`:not(.hy-chat-input-send-btn--disabled)`)
  and otherwise clicks the div.
- **Tencent AI Studio extraction (nothing streamed back)**: the old
  selectors (`[data-message-author-role="assistant"]`,
  `[class*="assistant"]`) match nothing on aistudio.tencent.ai, so
  every turn returned `found=false` and no answer or reasoning ever
  reached the local chat. The real DOM: one turn = one
  `div.agent-chat__list__item--ai`; the answer markdown renders in
  `.hyc-content-md` (only once the answer starts streaming), wrapped
  by the turn's speech area `.agent-chat__conv--ai__speech_show`,
  which also wraps the reasoning disclosure. The extraction now leads
  with `.hyc-content-md` (answer) and falls back to the speech area,
  scoped to the newest turn via `response_container`
  (`.agent-chat__list__item--ai`) so a pending turn reports
  `found=true` with empty content and `busy=true` (the reasoning
  spinner `.t-loading` sits inside the speech area) instead of
  leaking the previous turn's answer. The reasoning trace is
  `div.hy-detail-block.hy-think` inside the `hy-collapse`
  disclosure; the speech-area fallback would carry it, so
  `response_exclude_selectors` prunes `.hy-collapse` /
  `.hy-detail-block` / the action toolbar / checkbox from the answer
  clone. The page's obfuscated "Model Details" sidebar
  (`.mvfqmiYoVreC1bYK_YyC`, 770B parameters) is NOT a thinking
  block and must never be added to `thinking_selectors` — it sits
  outside the turn items, so the container scoping already keeps it
  out. Because the site collapses its reasoning disclosure when the
  answer completes, the final capture reports no thinking even though
  the full trace streamed a moment earlier; `_with_last_thinking` in
  `browser.py` re-attaches the last non-empty trace seen while the
  turn was live so the non-streaming path still returns it.
- **Extraction no longer reads hidden text**: the shared
  `textOf` helper used `innerText || textContent`, so a response
  element whose visible text was empty (e.g. the speech-area
  fallback during Tencent's reasoning phase, where every child is
  inside a collapsed disclosure) fell back to `textContent` and
  leaked the hidden reasoning trace as the answer. `textOf` now uses
  `innerText` and only falls back to `textContent` when `innerText`
  is unavailable (exotic environments), never merely empty.
- **DeepSeek sign-in page**: the login-wall selectors now also match the
  Cloudflare sign-in page (`#cf-turnstile`).
- **Copilot support added** (`sbsllm/sites.py`): a new `copilot` site for
  https://copilot.com/ ships with a textarea composer, a send-button
  submit handler, a multi-variant turn-marker cascade (`data-content` /
  `data-message-author-role` / testid), exclusion of citation, feedback,
  suggested-action and status-chip chrome, and thinking/answer separation.
  Because the consumer chat is auth-gated, enable it in `config.yaml` only
  after signing in; 5 Chromium-backed extraction tests exercise
  newest-turn selection, chrome pruning, paragraph structure and the
  thinking-only busy signal.
- **Google AI Studio welcome page**: logged-out visitors land on the
  `/welcome` marketing page; its CTAs (`a.nav__cta`, `a.hero__cta`)
  only exist there and now mark the pre-login state in
  `login_wall_selectors`.

### Duplicate prompt guard

`duplicate_prompt_cooldown` (config, default 60s, `0` disables) arms the
guard only on a *successful* submit. An identical prompt on the same tab
within the window skips inject+submit entirely: if the answer is already on
screen it is handed straight back; if the turn is still generating (or
nothing has appeared yet) the request polls from the current capture onward.
Each suppressed duplicate slides the window forward, so a retry train
cannot out-wait the guard; once retries stop the entry ages out after one
cooldown and re-sending the same prompt works again.

### Non-streaming timeout fallback

The non-streaming path returns the full response at once. If the browser
budget runs out but a non-blank answer (or reasoning trace) is already on
screen, it now returns the truncated answer with `finish_reason="length"` (a
200, not an error) rather than a bare 504 — mirroring the streaming path and
how OpenAI signals a `max_tokens` cutoff. A timeout that captured nothing (or
only whitespace) still returns 504.

### Token usage

`prompt_tokens` / `completion_tokens` / `total_tokens` are estimated by
`_estimate_tokens` in `server.py`: CJK characters (Han / Hiragana /
Katakana / Hangul) count ~1 token each, everything else ~4 characters per
token. Web chats expose no tokenizer, so these are ballpark figures for
cost/progress display, not billing. Included on the final SSE chunk
(streaming) and on the completion object (non-streaming).

### Multi model

When two or more chats are configured, the server registers a virtual
`multi` model (`MULTI_MODEL_ID` in `server.py`, appended to `model_map`
by `Server.__init__`). A non-streaming request to `multi` submits the
prompt on every tab concurrently (`_handle_multi_model` →
`_fan_out_to_tabs`, one thread per tab; the Playwright calls inside are
marshalled onto the single browser worker thread, so the answer waits
overlap), waits for all of them, and returns one completion whose
content is each answer under a `### <model>` heading (thinking traces
aggregated the same way). Every tab's lock is acquired upfront under one
shared `browser_lock_timeout` deadline. Failure semantics are strict:
any tab failing (timeout, no output, login wall, inject/submit failure,
unsupported extraction) fails the whole request with that tab's status
code plus `[multi: failed on <models>]` — no partial aggregates.
`"stream": true` with `multi` is rejected with 400: there is no sane
ordering for multiplexing concurrent browser streams into one SSE
stream. The per-tab duplicate-submit guard applies unchanged, keyed on
each real tab's model name.
