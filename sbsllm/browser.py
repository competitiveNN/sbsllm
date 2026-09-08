"""qutebrowser launch and IPC control."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

# Isolated profile directory for sbsllm's qutebrowser instance
BASEDIR = Path("/tmp/sbsllm-qb")

# Timeout for IPC commands (seconds)
IPC_TIMEOUT = 5

# Timeout for launching qutebrowser (seconds)
LAUNCH_TIMEOUT = 15


class BrowserError(Exception):
    """Raised when a browser operation fails."""
    pass


def run_command(qb_bin: str, *args: str, timeout: int = IPC_TIMEOUT) -> subprocess.CompletedProcess:
    """Run a qutebrowser command via CLI (sends via IPC if running)."""
    cmd = [qb_bin, "--basedir", str(BASEDIR), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def is_running(qb_bin: str) -> bool:
    """Check if a qutebrowser instance is running for our basedir."""
    try:
        result = run_command(qb_bin, ":jseval", "--quiet", "1", timeout=2)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def ensure_qutebrowser(qb_bin: str) -> None:
    """Ensure qutebrowser is running. Launch if not."""
    if is_running(qb_bin):
        print("qutebrowser is already running.")
        return

    print("Starting qutebrowser...")
    # Launch qutebrowser in background with no initial URL
    # We'll open tabs via IPC once it's ready
    subprocess.Popen(
        [qb_bin, "--basedir", str(BASEDIR)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for it to be ready
    start = time.time()
    while time.time() - start < LAUNCH_TIMEOUT:
        if is_running(qb_bin):
            print("qutebrowser is ready.")
            # Give it a moment to fully initialize
            time.sleep(1)
            return
        time.sleep(0.5)

    raise BrowserError(
        f"qutebrowser did not start within {LAUNCH_TIMEOUT}s. "
        "Is it installed? Try: pip install qutebrowser"
    )


def open_tab(qb_bin: str, url: str) -> bool:
    """Open a URL in a new tab. Returns True on success."""
    result = run_command(qb_bin, ":open", "-t", url)
    if result.returncode != 0:
        print(f"Failed to open tab: {url}", file=sys.stderr)
        print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def open_tabs(qb_bin: str, urls: list[str]) -> list[bool]:
    """Open multiple URLs in new tabs. Returns list of success flags."""
    results = []
    for url in urls:
        results.append(open_tab(qb_bin, url))
        # Small delay between tab opens to avoid race conditions
        time.sleep(0.3)
    return results


def focus_tab(qb_bin: str, index: int) -> bool:
    """Focus a tab by index (1-based). Returns True on success."""
    result = run_command(qb_bin, ":tab-focus", str(index))
    return result.returncode == 0


def run_js(qb_bin: str, js: str, quiet: bool = True) -> str:
    """Execute JS in the current tab. Returns stdout (JS return value)."""
    args = [":jseval"]
    if quiet:
        args.append("--quiet")
    args.append(js)

    result = run_command(qb_bin, *args)
    return result.stdout.strip()


def inject_and_submit(qb_bin: str, tab_index: int, inject_js: str, submit_js: str) -> dict:
    """Focus a tab, inject prompt, and submit. Returns status dict."""
    status = {"tab": tab_index, "inject": None, "submit": None}

    # Focus the tab
    if not focus_tab(qb_bin, tab_index):
        status["inject"] = "FAILED_TAB_FOCUS"
        return status

    # Small delay for tab switch to complete
    time.sleep(0.5)

    # Inject prompt
    inject_result = run_js(qb_bin, inject_js)
    status["inject"] = inject_result

    if inject_result != "OK":
        return status

    # Small delay before submit
    time.sleep(0.3)

    # Submit
    submit_result = run_js(qb_bin, submit_js)
    status["submit"] = submit_result

    return status
