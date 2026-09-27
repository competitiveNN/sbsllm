# Changelog

## Unreleased

### Fixed

- **Non-streaming path returned a fake "Prompt sent to X successfully" message**
  when extraction was unsupported, which the local chat rendered as the
  assistant's answer. Now returns a visible 502 error instead.
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

- **Per-tab locks** (`_ModelLockRegistry`): different chats run in parallel,
  same chat serializes.
- **New knobs**: `first_token_timeout=60s`, `busy_patience=20s`,
  `keepalive_interval=10s`, `browser_timeout=180s`, `browser_lock_timeout=300s`,
  `response_idle_timeout=3.0s`.
- **Reasoning-leak test coverage** (`tests/test_extract.py`, `tests/test_streaming.py`):
  complete thinking+answer flow, prune-not-replace, z.ai loading dots.

## Previous

- Initial public release.