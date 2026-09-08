"""CLI entry point — orchestrates the sbsllm flow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .browser import (
    BrowserError,
    ensure_qutebrowser,
    inject_and_submit,
    open_tabs,
)
from .config import load_config
from .inject import inject_prompt, submit_js
from .server import create_server
from .sites import get_site


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sbsllm",
        description="Side-by-side LLM testing — test prompts against multiple AI chats at once.",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml or ~/.config/sbsllm/config.yaml)",
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Prompt to send (if omitted, you'll be prompted interactively)",
    )
    parser.add_argument(
        "--login-wait",
        type=int,
        default=None,
        help="Seconds to wait for login (overrides config)",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List available chat sites and exit",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start OpenAI-compatible server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )
    return parser


def list_sites() -> None:
    """Print available sites and exit."""
    from .sites import list_sites as _list
    print("Available chat sites:")
    for site_id in _list():
        site = get_site(site_id)
        print(f"  {site_id:12} {site['url']}")


def get_prompt() -> str:
    """Get prompt from user interactively."""
    print("Enter your prompt (end with an empty line):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def run(config, prompt: str | None, login_wait: int | None) -> int:
    """Main orchestration. Returns exit code."""
    qb_bin = config.qb_bin
    chats = config.chats
    wait = login_wait if login_wait is not None else config.login_wait

    # Step 1: Ensure qutebrowser is running
    try:
        ensure_qutebrowser(qb_bin)
    except BrowserError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Step 2: Open tabs
    urls = [get_site(chat)["url"] for chat in chats]
    print(f"Opening {len(urls)} tabs...")
    results = open_tabs(qb_bin, urls)

    for chat, url, ok in zip(chats, urls, results):
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {chat}: {url}")

    if not all(results):
        print("\nSome tabs failed to open. Continue anyway? [y/N]", end=" ")
        if input().lower() != "y":
            return 1

    # Step 3: Wait for login
    print(f"\n{wait} seconds to log in to each chat...")
    print("Press Enter when ready (or wait for timeout)...")

    # Simple input with timeout would require threading; for now just use input
    # A more robust version could use select or signal
    try:
        input()
    except EOFError:
        pass

    # Step 4: Get prompt if not provided
    if prompt is None:
        prompt = get_prompt()

    if not prompt.strip():
        print("No prompt provided. Exiting.", file=sys.stderr)
        return 1

    # Step 5: Fan out prompt to all tabs
    print(f"\nSending prompt to {len(chats)} chats...\n")

    for i, chat in enumerate(chats, start=1):
        print(f"[{i}/{len(chats)}] {chat}...", end=" ", flush=True)

        inject_js = inject_prompt(chat, prompt)
        submit_js_val = submit_js(chat)

        status = inject_and_submit(qb_bin, i, inject_js, submit_js_val)

        inject_ok = status["inject"] == "OK"
        submit_ok = status["submit"] == "OK"

        if inject_ok and submit_ok:
            print("OK")
        elif not inject_ok:
            print(f"INJECT FAILED: {status['inject']}")
        else:
            print(f"SUBMIT FAILED: {status['submit']}")

    print("\nDone! Check qutebrowser for responses.")
    return 0


def run_server(config, host: str, port: int) -> int:
    """Run the OpenAI-compatible server."""
    qb_bin = config.qb_bin
    chats = config.chats

    # Ensure qutebrowser is running
    try:
        ensure_qutebrowser(qb_bin)
    except BrowserError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Open tabs
    urls = [get_site(chat)["url"] for chat in chats]
    print(f"Opening {len(urls)} tabs...")
    results = open_tabs(qb_bin, urls)

    for chat, url, ok in zip(chats, urls, results):
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {chat}: {url}")

    # Build model map: model name -> site_id
    # Use site_id as model name by default
    model_map = {chat: chat for chat in chats}
    tab_map = {chat: i for i, chat in enumerate(chats, start=1)}

    # Create and start server
    server = create_server(
        qb_bin=qb_bin,
        model_map=model_map,
        tab_map=tab_map,
        host=host,
        port=port,
    )

    print(f"\nServer ready! Send requests to:")
    print(f"  POST http://{host}:{port}/v1/chat/completions")
    print(f"  GET  http://{host}:{port}/v1/models")
    print(f"\nExample:")
    print(f'  curl -X POST http://{host}:{port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{chats[0]}", "messages": [{{"role": "user", "content": "Hello!"}}]}}\'')
    print()

    server.start()
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_sites:
        list_sites()
        return

    config = load_config(args.config)

    if args.server:
        exit_code = run_server(config, args.host, args.port)
    else:
        exit_code = run(config, args.prompt, args.login_wait)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
