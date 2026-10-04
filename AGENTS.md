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
`_estimate_tokens` in `server.py`: CJK characters (Han / Hiragana / Katakana /
Hangul) count ~1 token each, everything else ~4 characters per token. Web chats
expose no tokenizer, so these are ballpark figures for cost/progress display,
not billing. Included on the final SSE chunk (streaming) and on the completion
object (non-streaming).
