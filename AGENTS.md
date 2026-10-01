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

- **Google AI Studio**: `inject` and `submit_js` now traverse Shadow DOM roots
  of `ms-prompt-box` / `ms-autosize-textarea` / `ms-run-button` web components.
- **Zai chat**: URL changed to `https://chat.z.ai/` to avoid auth wall;
  `login_wall_selectors` narrowed to exclude nav sign-in button false-positive;
  double-send from Enter key dispatch in `post_inject_js` eliminated.
- **Grok chat**: `inject` and `submit_js` now fall back to scanning shadow roots
  for the TipTap/ProseMirror editor and send button.
- **Kimi URL**: Corrected to `https://www.kimi.ai/`.

### Design TODO

- Token usage counts (`prompt_tokens`, `completion_tokens`) are always 0.
  A simple word-count heuristic could provide rough estimates.
- The non-streaming path returns the full response at once; no chunked fallback.
