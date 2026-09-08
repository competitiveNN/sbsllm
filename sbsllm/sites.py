"""Site registry: URLs and DOM injection logic for each supported chat."""

SITES: dict[str, dict] = {
    "chatgpt": {
        "url": "https://chatgpt.com/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[data-id="root"]')
                    || document.querySelector('#prompt-textarea')
                    || document.querySelector('textarea[placeholder*="message"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[data-testid="send-button"]')
                    || document.querySelector('button[aria-label="Send prompt"]')
                    || document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('textarea')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "claude": {
        "url": "https://claude.ai/",
        "inject": """
            (() => {
                // Claude uses a contenteditable div
                const editor = document.querySelector('[contenteditable="true"]')
                    || document.querySelector('.ProseMirror')
                    || document.querySelector('[data-placeholder]');
                if (!editor) return 'NO_INPUT';
                editor.focus();
                editor.textContent = `PROMPT_PLACEHOLDER`;
                editor.dispatchEvent(new Event('input', { bubbles: true }));
                editor.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label="Send Message"]')
                    || document.querySelector('button[data-testid="send-button"]')
                    || document.querySelector('button[class*="send"]')
                    || document.querySelector('[contenteditable="true"]')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "deepseek": {
        "url": "https://chat.deepseek.com/",
        "inject": """
            (() => {
                const ta = document.querySelector('#chat-input')
                    || document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label="Send"]')
                    || document.querySelector('.send-btn')
                    || document.querySelector('button[class*="send"]')
                    || document.querySelector('textarea')?.parentElement?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "qwen": {
        "url": "https://chat.qwen.ai/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea[placeholder*="ask"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('button[class*="send"]')
                    || document.querySelector('textarea')?.closest('div')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "grok": {
        "url": "https://grok.com/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('button[class*="send"]')
                    || document.querySelector('textarea')?.parentElement?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "google": {
        "url": "https://aistudio.google.com/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="Enter"]')
                    || document.querySelector('textarea[placeholder*="prompt"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Run"]')
                    || document.querySelector('button[aria-label*="run"]')
                    || document.querySelector('button[class*="run"]')
                    || document.querySelector('textarea')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "mistral": {
        "url": "https://chat.mistral.ai/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea[placeholder*="Send"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('button[class*="send"]')
                    || document.querySelector('textarea')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "kimi": {
        "url": "https://www.kimi.com/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="输入"]')
                    || document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('button[class*="send"]')
                    || document.querySelector('textarea')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "perplexity": {
        "url": "https://www.perplexity.ai/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea[placeholder*="Search"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Submit"]')
                    || document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('textarea')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "poe": {
        "url": "https://poe.com/",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="Message"]')
                    || document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('button[class*="send"]')
                    || document.querySelector('textarea')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
    "cohere": {
        "url": "https://cohere.com/chat",
        "inject": """
            (() => {
                const ta = document.querySelector('textarea[placeholder*="Message"]')
                    || document.querySelector('textarea[placeholder*="Ask"]')
                    || document.querySelector('textarea');
                if (!ta) return 'NO_INPUT';
                const nativeSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value'
                ).set;
                nativeSetter.call(ta, `PROMPT_PLACEHOLDER`);
                ta.dispatchEvent(new Event('input', { bubbles: true }));
                ta.dispatchEvent(new Event('change', { bubbles: true }));
                return 'OK';
            })()
        """,
        "submit_js": """
            (() => {
                const btn = document.querySelector('button[aria-label*="Send"]')
                    || document.querySelector('button[type="submit"]')
                    || document.querySelector('textarea')?.closest('form')?.querySelector('button');
                if (btn) { btn.click(); return 'OK'; }
                return 'NO_BUTTON';
            })()
        """,
    },
}


def get_site(site_id: str) -> dict:
    """Get site config by ID. Raises ValueError if not found."""
    if site_id not in SITES:
        raise ValueError(
            f"Unknown site: {site_id!r}. "
            f"Available: {', '.join(SITES)}"
        )
    return SITES[site_id]


def list_sites() -> list[str]:
    """Return list of available site IDs."""
    return list(SITES.keys())
