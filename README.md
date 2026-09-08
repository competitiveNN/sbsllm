# sbsllm — Side-by-Side LLM Testing

Test the same prompt against multiple AI chat websites simultaneously.

## Installation

```bash
pip install -e .
```

Requires [qutebrowser](https://qutebrowser.org/) to be installed:
```bash
pip install qutebrowser
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
sbsllm                          # Interactive mode
sbsllm -p "Hello, world!"       # Prompt via CLI
sbsllm -c myconfig.yaml         # Custom config
sbsllm --list-sites             # List available chats
```

## Config

| Key | Default | Description |
|-----|---------|-------------|
| `chats` | (required) | List of chat site IDs to open |
| `login_wait` | `30` | Seconds to wait for login |
| `qutebrowser_bin` | auto-detect | Path to qutebrowser binary |

## Supported Chats

- `chatgpt` — https://chatgpt.com/
- `claude` — https://claude.ai/
- `deepseek` — https://chat.deepseek.com
- `qwen` — https://chat.qwen.ai/
- `grok` — https://grok.com/
- `google` — https://aistudio.google.com/
- `mistral` — https://chat.mistral.ai
- `kimi` — https://www.kimi.com/

## How It Works

sbsllm launches an isolated qutebrowser instance (`--basedir /tmp/sbsllm-qb`),
opens each chat in a tab, waits for you to log in, then uses JavaScript injection
to type and send your prompt into each chat's input field.

> **Note**: The DOM selectors used for injection are fragile — chat sites change
> their markup frequently. If a chat stops working, the selectors in `sites.py`
> may need updating.
