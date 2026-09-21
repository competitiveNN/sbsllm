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
  - chatgpt
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

- `chatgpt` — https://chatgpt.com/
- `claude` — https://claude.ai/
- `deepseek` — https://chat.deepseek.com/
- `qwen` — https://chat.qwen.ai/
- `grok` — https://grok.com/
- `google` — https://aistudio.google.com/
- `mistral` — https://chat.mistral.ai
- `kimi` — https://www.kimi.com/
- `perplexity` — https://www.perplexity.ai/
- `poe` — https://poe.com/
- `cohere` — https://cohere.com/chat
- `zai` — https://chat.z.ai/auth
- `meta` — https://meta.ai/
- `huggingface` — https://huggingface.co/chat
- `tencent` — https://aistudio.tencent.ai/

## How It Works

sbsllm launches an isolated Chromium profile via Playwright,
opens each chat in a separate page/tab, waits for you to log in, then uses
JavaScript injection to type and send your prompt into each chat's input field.

> **Note**: The DOM selectors used for injection are fragile — chat sites change
> their markup frequently. If a chat stops working, the selectors in `sites.py`
> may need updating.
