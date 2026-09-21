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


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("must be an integer") from e
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def server_port(value: str) -> int:
    """Parse a valid TCP port, allowing zero for an ephemeral port."""
    port = non_negative_int(value)
    if port > 65535:
        raise argparse.ArgumentTypeError("must be at most 65535")
    return port


def login_wait(value: str) -> int:
    """Parse a valid login wait time, capped at 24 hours."""
    wait = non_negative_int(value)
    max_wait = 24 * 60 * 60  # 24 hours
    if wait > max_wait:
        raise argparse.ArgumentTypeError(f"must be at most {max_wait} (24 hours)")
    return wait


def _server_url(server, host: str, port: int) -> str:
    """Return the server URL, using the bound port when available."""
    url = getattr(server, "url", None)
    if isinstance(url, str):
        return url
    return f"http://{host}:{port}/"


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
        type=login_wait,
        default=None,
        help="Seconds to wait for login (overrides config; 0 waits for Enter; max 86400)",
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
        type=server_port,
        default=8080,
        help="Server port (default: 8080)",
    )
    parser.add_argument(
        "--chrome-bin",
        type=str,
        default=None,
        help="Path to Chrome/Chromium binary (auto-detected if omitted)",
    )
    parser.add_argument(
        "--json-log",
        action="store_true",
        help="Output logs in JSON format for structured logging",
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

    # Cap the wait to avoid select overflow on some platforms.
    MAX_WAIT = 24 * 60 * 60  # 24 hours
    remaining = min(wait, MAX_WAIT)

    while remaining > 0:
        chunk = min(remaining, 3600)  # max 1 hour per select call
        try:
            ready, _, _ = select.select([sys.stdin], [], [], chunk)
            if ready:
                try:
                    input()
                except EOFError:
                    pass
                return
        except (OSError, ValueError, OverflowError) as e:
            logger = logging.getLogger(__name__)
            logger.debug(f"stdin select failed: {e}")
            # If stdin is not a TTY, there is no point in blocking on input().
            if not sys.stdin.isatty():
                return
            try:
                input()
            except EOFError:
                pass
            return
        remaining -= chunk


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


def _start_server(server, announce: bool = False) -> threading.Thread | None:
    """Start a server thread and wait until its socket is bound."""
    install_handlers = getattr(server, "install_signal_handlers", None)
    if install_handlers is not None:
        install_handlers()

    def run_server() -> None:
        try:
            server.start(announce=announce)
        except Exception as e:  # noqa: BLE001
            print(f"Server thread stopped before readiness: {e}", file=sys.stderr)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    wait_ready = getattr(server, "wait_until_ready", None)
    if wait_ready is not None:
        try:
            wait_ready(timeout=5)
        except Exception as e:  # noqa: BLE001
            print(f"Error: failed to start server: {e}", file=sys.stderr)
            server.stop()
            thread.join(timeout=1)
            return None
    return thread


def _wait_for_server_shutdown(
    server_thread: threading.Thread | None, timeout: float = 5.0
) -> None:
    """Wait for server thread to finish, with timeout to avoid hanging."""
    if server_thread is None:
        return
    server_thread.join(timeout=timeout)


def run(
    config,
    prompt: str | None,
    login_wait: int | None,
    chrome_bin: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8080,
    json_log: bool = False,
) -> int:
    """Main orchestration. Returns exit code."""
    # Setup logging
    setup_logging(
        level=config.log_level,
        log_file=config.log_file,
        json_format=json_log,
    )

    chats = config.chats
    wait = login_wait if login_wait is not None else config.login_wait
    browser_bin = chrome_bin or config.chrome_bin

    server = None
    server_thread = None

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
            if not any(pages):
                print("No browser tabs are available. Exiting.", file=sys.stderr)
                return 1

        successful_tabs = {
            chat: page for chat, page in zip(chats, pages) if page is not None
        }

        # Step 3: Start the server before login/prompt so its URL is visible immediately
        # Only advertise models for tabs that actually opened successfully
        model_map = {chat: chat for chat in successful_tabs}
        server = create_server(
            model_map=model_map,
            tab_map=successful_tabs,
            host=host,
            port=port,
            browser_timeout=config.browser_timeout,
            browser_lock_timeout=config.browser_lock_timeout,
        )
        server_thread = _start_server(server)
        if server_thread is None:
            return 1
        print(
            f"\nOpenAI-compatible server URL: {_server_url(server, host, port)}",
            flush=True,
        )
        print(
            f"Server listening on {_server_url(server, host, port).rstrip('/')}",
            flush=True,
        )
        print("Responses are open in the browser.", flush=True)

        # Step 4: Wait for login
        wait_for_login(wait)

        # Step 5: Get prompt if not provided
        if prompt is None:
            prompt = get_prompt()

        if not prompt.strip():
            print("No prompt provided. Exiting.", file=sys.stderr)
            return 1

        # Step 6: Fan out prompt to all tabs
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
            submit_ok = status["submit"] in (
                "OK",
                "ENTER_SENT",
                "ENTER_SENT_UNVERIFIED",
            )

            if inject_ok and submit_ok:
                if status["submit"] in ("ENTER_SENT", "ENTER_SENT_UNVERIFIED"):
                    print("ENTER SENT (verify in browser)")
                else:
                    print("OK")
            elif not inject_ok:
                print(f"INJECT FAILED: {status['inject']}")
            else:
                print(f"SUBMIT FAILED: {status['submit']}")

        try:
            server_thread.join()
            return 0
        except KeyboardInterrupt:
            print("\nShutting down server...")
            return 0
        finally:
            if server is not None:
                server.stop()
            _wait_for_server_shutdown(server_thread)
    finally:
        if server is not None:
            server.stop()
        _wait_for_server_shutdown(server_thread)
        close_browser()


def run_server(
    config, host: str, port: int, chrome_bin: str | None = None, json_log: bool = False
) -> int:
    """Run the OpenAI-compatible server."""
    chats = config.chats
    browser_bin = chrome_bin or config.chrome_bin
    server = None
    server_thread = None

    try:
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
        successful_tabs = {
            chat: page for chat, page in zip(chats, pages) if page is not None
        }
        if not successful_tabs:
            print("No browser tabs are available. Exiting.", file=sys.stderr)
            return 1

        # Setup logging
        setup_logging(
            level=config.log_level,
            log_file=config.log_file,
            json_format=json_log,
        )

        # Build model map: model name -> site_id + page reference
        model_map = {chat: chat for chat in successful_tabs}
        tab_map = successful_tabs

        # Create and start server
        server = create_server(
            model_map=model_map,
            tab_map=tab_map,
            host=host,
            port=port,
            browser_timeout=config.browser_timeout,
            browser_lock_timeout=config.browser_lock_timeout,
        )

        server_thread = _start_server(server)
        if server_thread is None:
            return 1

        print(
            f"\nOpenAI-compatible server URL: {_server_url(server, host, port)}",
            flush=True,
        )
        print("Server ready! Send requests to:", flush=True)
        base_url = _server_url(server, host, port).rstrip("/")
        print(f"  POST {base_url}/v1/chat/completions", flush=True)
        print(f"  GET  {base_url}/v1/models", flush=True)
        print("\nExample:", flush=True)
        print(f"  curl -X POST {base_url}/v1/chat/completions \\", flush=True)
        print('    -H "Content-Type: application/json" \\', flush=True)
        print(
            f'    -d \'{{"model": "{chats[0]}", "messages": [{{"role": "user", "content": "Hello!"}}]}}\'',
            flush=True,
        )
        print(flush=True)

        try:
            server_thread.join()
            return 0
        except KeyboardInterrupt:
            print("\nShutting down server...")
            return 0
        finally:
            if server is not None:
                server.stop()
            _wait_for_server_shutdown(server_thread)
    finally:
        if server is not None:
            server.stop()
        _wait_for_server_shutdown(server_thread)
        close_browser()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_sites:
        list_sites()
        return

    config = load_config(args.config)

    if args.server:
        exit_code = run_server(
            config, args.host, args.port, args.chrome_bin, args.json_log
        )
    else:
        exit_code = run(
            config,
            args.prompt,
            args.login_wait,
            args.chrome_bin,
            args.host,
            args.port,
            args.json_log,
        )

    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover
    main()
