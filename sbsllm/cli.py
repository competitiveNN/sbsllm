"""CLI entry point — orchestrates the sbsllm flow."""

from __future__ import annotations

import argparse
import logging
import select
import sys
import threading
from pathlib import Path

from .browser import (
    BrowserError,
    close_browser,
    ensure_browser,
    inject_and_submit,
    open_page,
    setup_logging,
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
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: ./config.yaml or ~/.config/sbsllm/config.yaml)",
    )
    parser.add_argument(
        "-p",
        "--prompt",
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
    parser.add_argument(
        "--chrome-bin",
        type=str,
        default=None,
        help="Path to Chrome/Chromium binary (auto-detected if omitted)",
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


def wait_for_login(wait: int) -> None:
    """Wait for login, allowing Enter to continue before the timeout."""
    print(f"\n{wait} seconds to log in to each chat...")
    print("Press Enter when ready (or wait for timeout)...")

    if wait <= 0:
        try:
            input()
        except EOFError:
            pass
        return

    try:
        ready, _, _ = select.select([sys.stdin], [], [], wait)
        if ready:
            try:
                input()
            except EOFError:
                pass
    except (OSError, ValueError) as e:
        try:
            input()
        except EOFError:
            pass
        logger = logging.getLogger(__name__)
        logger.debug(f"stdin is not selectable; waiting for login interactively: {e}")


def _open_tabs(chats: list[str], urls: list[str], browser_bin: str | None) -> list:
    """Open all tabs sequentially."""
    pages: list = [None] * len(chats)
    for i, (chat, url) in enumerate(zip(chats, urls)):
        try:
            page = open_page(url, browser_bin)
            pages[i] = page
            print(f"  [OK] {chat}: {url}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAILED] {chat}: {url} - {e}", file=sys.stderr)
            pages[i] = None
    return pages


def run(
    config,
    prompt: str | None,
    login_wait: int | None,
    chrome_bin: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> int:
    """Main orchestration. Returns exit code."""
    # Setup logging
    setup_logging(level=config.log_level, log_file=config.log_file)

    chats = config.chats
    wait = login_wait if login_wait is not None else config.login_wait
    browser_bin = chrome_bin or config.chrome_bin

    try:
        # Step 1: Ensure browser is running
        try:
            ensure_browser(browser_bin)
        except BrowserError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Step 2: Open tabs
        urls = [get_site(chat)["url"] for chat in chats]
        print(f"Opening {len(urls)} tabs...")

        pages = _open_tabs(chats, urls, browser_bin)

        if any(p is None for p in pages):
            print("\nSome tabs failed to open. Continue anyway? [y/N]", end=" ")
            if input().lower() != "y":
                return 1

        # Step 3: Wait for login
        wait_for_login(wait)

        # Step 4: Get prompt if not provided
        if prompt is None:
            prompt = get_prompt()

        if not prompt.strip():
            print("No prompt provided. Exiting.", file=sys.stderr)
            return 1

        # Step 5: Fan out prompt to all tabs
        print(f"\nSending prompt to {len(chats)} chats...\n")

        for i, (chat, page) in enumerate(zip(chats, pages), start=1):
            if page is None:
                print(f"[{i}/{len(chats)}] {chat}... SKIPPED")
                continue

            print(f"[{i}/{len(chats)}] {chat}...", end=" ", flush=True)

            inject_js = inject_prompt(chat, prompt)
            submit_js_val = submit_js(chat)

            status = inject_and_submit(page, inject_js, submit_js_val, i)

            inject_ok = status["inject"] == "OK"
            submit_ok = status["submit"] == "OK"

            if inject_ok and submit_ok:
                print("OK")
            elif not inject_ok:
                print(f"INJECT FAILED: {status['inject']}")
            else:
                print(f"SUBMIT FAILED: {status['submit']}")

        # Step 6: Start OpenAI-compatible server so responses remain accessible
        server = create_server(
            model_map={chat: chat for chat in chats},
            tab_map={chat: page for chat, page in zip(chats, pages) if page is not None},
            host=host,
            port=port,
        )
        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        print(f"\nOpenAI-compatible server URL: http://{host}:{port}/", flush=True)
        print(f"Server listening on http://{host}:{port}", flush=True)
        print("Responses are open in the browser.", flush=True)

        try:
            server_thread.join()
            return 0
        except KeyboardInterrupt:
            print("\nShutting down server...")
            return 0
    finally:
        close_browser()


def run_server(config, host: str, port: int, chrome_bin: str | None = None) -> int:
    """Run the OpenAI-compatible server."""
    chats = config.chats
    browser_bin = chrome_bin or config.chrome_bin

    # Ensure browser is running
    try:
        ensure_browser(browser_bin)
    except BrowserError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Open tabs
    urls = [get_site(chat)["url"] for chat in chats]
    print(f"Opening {len(urls)} tabs...", flush=True)

    pages = _open_tabs(chats, urls, browser_bin)

    # Build model map: model name -> site_id + page reference
    model_map = {chat: chat for chat in chats}
    tab_map = {chat: page for chat, page in zip(chats, pages) if page is not None}

    # Create and start server
    server = create_server(
        model_map=model_map,
        tab_map=tab_map,
        host=host,
        port=port,
    )

    print(f"\nOpenAI-compatible server URL: http://{host}:{port}/", flush=True)
    print("Server ready! Send requests to:", flush=True)
    print(f"  POST http://{host}:{port}/v1/chat/completions", flush=True)
    print(f"  GET  http://{host}:{port}/v1/models", flush=True)
    print("\nExample:", flush=True)
    print(f"  curl -X POST http://{host}:{port}/v1/chat/completions \\", flush=True)
    print('    -H "Content-Type: application/json" \\', flush=True)
    print(
        f'    -d \'{{"model": "{chats[0]}", "messages": [{{"role": "user", "content": "Hello!"}}]}}\'',
        flush=True,
    )
    print(flush=True)

    try:
        server.start()
        return 0
    finally:
        close_browser()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_sites:
        list_sites()
        return

    config = load_config(args.config)

    if args.server:
        exit_code = run_server(config, args.host, args.port, args.chrome_bin)
    else:
        exit_code = run(config, args.prompt, args.login_wait, args.chrome_bin)

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
