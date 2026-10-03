"""Live-DOM verification: run the real inject JS against real pages.

Inject only FILLS the input field (it does not click send), so it is safe to
run against live sites. Reports the inject status and, on failure, what
selectors/structures the page currently exposes.
"""

from __future__ import annotations

import sys
import time
import traceback

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from sbsllm.inject import extract_js, inject_prompt
from sbsllm.sites import SITES

TARGETS = ["grok", "zai", "google", "chatgpt", "claude", "deepseek", "qwen", "kimi"]
VERIFY_PROMPT = "sbsllm-live-verify"


def safe_eval(page, expr):
    try:
        return page.evaluate(expr)
    except Exception as e:  # noqa: BLE001
        return f"EVAL_ERROR: {e!r}"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        for site_id in TARGETS:
            site = SITES[site_id]
            url = site["url"]
            print(f"\n=== {site_id} ({url}) ===", flush=True)
            page = ctx.new_page()
            try:
                try:
                    resp = page.goto(url, wait_until="domcontentloaded", timeout=25000)
                    print(
                        f"  http_status={resp.status if resp else 'no-resp'}",
                        flush=True,
                    )
                except PlaywrightTimeoutError as e:
                    print(f"  NAV_TIMEOUT: {e!r}", flush=True)
                    results_summary(site_id, {"error": "nav_timeout"})
                    page.close()
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"  NAV_ERROR: {e!r}", flush=True)
                    page.close()
                    continue

                # Let the page settle before running the inject probe.
                time.sleep(1.5)

                # Run the real inject JS (fills input only — no submit).
                inject_js = inject_prompt(site_id, VERIFY_PROMPT)
                status = safe_eval(page, inject_js)
                print(f"  inject_status={status!r}", flush=True)

                # For OK cases, confirm the value was actually placed on the
                # marked input AND read its nature (tag/classes).
                if status == "OK":
                    marked = safe_eval(
                        page,
                        """(() => {
                            const el = document.querySelector('[data-sbsllm-input="true"]');
                            if (!el) return null;
                            return {
                                tag: el.tagName,
                                isCE: el.isContentEditable,
                                class: (el.className || '').slice(0, 60),
                                text: (el.innerText || el.value || el.textContent || '').slice(0, 40),
                            };
                        })()""",
                    )
                    print(f"  marked_input={marked!r}", flush=True)
                    # Confirm the filled value is observable.
                    saw_value = safe_eval(
                        page,
                        f"!!(document.body.innerText.includes({VERIFY_PROMPT!r}) "
                        "|| (document.querySelector('[data-sbsllm-input=\"true\"]')"
                        f"?.value?.includes({VERIFY_PROMPT!r})"
                        "|| document.querySelector('[data-sbsllm-input=\"true\"]')"
                        f"?.textContent?.includes({VERIFY_PROMPT!r}))",
                    )
                    print(f"  value_observed_on_page={saw_value}", flush=True)

                if status != "OK":
                    # Probe which input shapes exist on the live page.
                    probes = {
                        "textarea": "document.querySelector('textarea')",
                        "contenteditable": "document.querySelector('div[contenteditable=\"true\"]')",
                        "tiptap": "document.querySelector('.tiptap')",
                        "lexical": "document.querySelector('[data-lexical-editor]')",
                        "ms-textarea": "document.querySelector('ms-prompt-box textarea')",
                        "input_text": "document.querySelector('input[type=\"text\"]')",
                    }
                    for label, expr in probes.items():
                        print(
                            f"  probe[{label}]={safe_eval(page, f'!!{expr}')}",
                            flush=True,
                        )
                    # Run login-wall detection if extraction JS available.
                    ej = extract_js(site_id)
                    if ej:
                        ex = safe_eval(page, ej)
                        print(
                            f"  login_wall={ex.get('login_wall') if isinstance(ex, dict) else ex} "
                            f"found={ex.get('found') if isinstance(ex, dict) else None}",
                            flush=True,
                        )
                    body = safe_eval(
                        page, "(document.body && document.body.innerText) || ''"
                    )
                    if isinstance(body, str):
                        print(f"  body_preview={body[:160]!r}", flush=True)
                results_summary(site_id, {"inject": status})
            finally:
                page.close()
        browser.close()


def results_summary(site_id, info):
    # kept minimal; main output is the prints above
    pass


if __name__ == "__main__":
    try:
        main()
    # Top-level guard: surface any failure with a traceback and exit non-zero.
    # Catching Exception (not BaseException) deliberately lets KeyboardInterrupt
    # and SystemExit propagate. noqa: BLE001 -- a broad catch is intentional
    # here as the script's single entry-point error boundary.
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
