# sbsllm — Side-by-Side LLM Testing

Test the same prompt against multiple AI chat websites simultaneously.

## Installation

```bash
pip install -e .
python -m playwright install chromium
```

## Quick Start

1. Create a config file (`config.yaml` or `~/.config/sbsllm/config.yaml`):

```yaml
chats:
  - claude
  - deepseek

login_wait: 30
```

2. Run:
```bash
sbsllm
```

3. Log in to each chat when the browser opens.

4. Enter your prompt when prompted.

5. The prompt is sent to all chats simultaneously.

## Usage

```bash
sbsllm                                      # Interactive mode
sbsllm -p "Hello, world!"                  # Prompt via CLI
sbsllm -c myconfig.yaml                    # Custom config
sbsllm --login-wait 60                     # Override the login wait
sbsllm --chrome-bin /usr/bin/chromium      # Use a specific browser binary
sbsllm --list-sites                        # List available chats
```

## Config

| Key | Default | Description |
|-----|---------|-------------|
| `chats` | (required) | List of chat site IDs to open |
| `login_wait` | `30` | Seconds to wait for login; `0` waits for Enter |
| `chrome_bin` | Playwright Chromium | Path to Chrome/Chromium binary (auto-detected if omitted) |
| `log_level` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `log_file` | stdout | Path to log file (optional) |
| `user_data_dir` | `~/.local/share/sbsllm/chrome` | Chromium profile directory. Holds login sessions and the browser cache between runs, so it must live on a filesystem that survives reboots. Override with `SBSLLM_USER_DATA_DIR` or `--user-data-dir`. |

## Persistent logins

The Chromium profile is stored under `~/.local/share/sbsllm/chrome` so that
login sessions and the browser cache survive restarts. The original default
(`/tmp/sbsllm-chrome`) sat on a volatile tmpfs and was wiped on every
container/boot restart, which is why logins never persisted. Point it
elsewhere with:

```bash
SBSLLM_USER_DATA_DIR=$HOME/.local/share/sbsllm/chrome sbsllm --server
# or
sbsllm --user-data-dir $HOME/.local/share/sbsllm/chrome --server
# or in config.yaml:
# user_data_dir: ~/.local/share/sbsllm/chrome
```

## OpenAI-Compatible Server

Start server mode with:

```bash
sbsllm --server --host 127.0.0.1 --port 8080
```

The browser tabs open first, and the server URL is printed before login or
prompt collection begins. Use `--port 0` to let the operating system choose an
available port; the printed URL contains the actual bound port.

Endpoints:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /health`

The server runs until interrupted. Stopping it also closes the Chromium pages.

## Supported Chats

- `claude` — https://claude.ai/
- `copilot` — https://copilot.com/
- `deepseek` — https://chat.deepseek.com/
- `qwen` — https://chat.qwen.ai/
- `grok` — https://grok.com/
- `google` — https://aistudio.google.com/
- `mistral` — https://chat.mistral.ai
- `kimi` — https://kimi.ai/
- `perplexity` — https://www.perplexity.ai/
- `poe` — https://poe.com/
- `cohere` — https://cohere.com/chat
- `zai` — https://chat.z.ai/auth
- `meta` — https://meta.ai/
- `huggingface` — https://huggingface.co/chat
- `tencent` — https://aistudio.tencent.ai/

## Unsupported

- `chatgpt` — the old selectors no longer hit the composer on chatgpt.com; support is pending a DOM update.

## How It Works

sbsllm launches an isolated Chromium profile via Playwright, so login
sessions persist between runs. The profile lives at
`~/.local/share/sbsllm/chrome` (overridable via `user_data_dir` /
`SBSLLM_USER_DATA_DIR` / `--user-data-dir`) so it survives reboots.
opens each chat in a separate page/tab, waits for you to log in, then uses
JavaScript injection to type and send your prompt into each chat's input field.

> **Note**: The DOM selectors used for injection are fragile — chat sites change
> their markup frequently. If a chat stops working, the selectors in `sites.py`
> may need updating.

## TODO

- [ ] **File attachments** — the next big feature: support uploading files with prompts and forwarding them to every configured chat.
