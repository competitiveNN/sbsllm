"""Site registry: URLs and DOM injection logic for each supported chat."""

from __future__ import annotations

import json

_INJECT_TEMPLATE = """
    (() => {
        const input = __SELECTORS__;
        if (!input) return 'NO_INPUT';
        const value = `PROMPT_PLACEHOLDER`;
        const contentEditable = input.getAttribute?.('contenteditable');
        const isEditable = input.isContentEditable === true
            || contentEditable === ''
            || contentEditable?.toLowerCase() === 'true';
        const isTextInput = input instanceof HTMLInputElement
            && (!input.type || ['text', 'search', 'url', 'tel', 'email', 'password'].includes(input.type.toLowerCase()));
        if (!(input instanceof HTMLTextAreaElement || isTextInput || isEditable)) {
            return 'NO_INPUT';
        }
        // Prevent double submission: clear any previous sbsllm marker
        if (input.dataset.sbsllmInput === 'true') {
            return 'OK';
        }
        input.focus();
        if (isEditable) {
            try {
                const selection = window.getSelection();
                selection?.selectAllChildren(input);
                document.execCommand('delete', false, null);
                document.execCommand('insertText', false, value);
            } catch (_) {}
            if ((input.textContent || '') !== value) {
                input.textContent = value;
            }
        } else if (input instanceof HTMLTextAreaElement) {
            const setter = Object.getOwnPropertyDescriptor(
                HTMLTextAreaElement.prototype, 'value'
            )?.set;
            if (!setter) return 'NO_INPUT';
            setter.call(input, value);
        } else {
            const setter = Object.getOwnPropertyDescriptor(
                HTMLInputElement.prototype, 'value'
            )?.set;
            if (!setter) return 'NO_INPUT';
            setter.call(input, value);
        }
        input.dataset.sbsllmInput = 'true';
        try {
            input.dispatchEvent(new InputEvent('input', {
                bubbles: true, inputType: 'insertText', data: value
            }));
        } catch (_) {
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
        input.dispatchEvent(new Event('change', { bubbles: true }));
        return 'OK';
    })()
    """

_SUBMIT_TEMPLATE = """
    (() => {
        const input = document.querySelector('[data-sbsllm-input="true"]')
            || (__INPUT_SELECTOR__);
        if (input?.dataset?.sbsllmInput === 'true') {
            delete input.dataset.sbsllmInput;
        }
        const form = input?.closest('form');
        // Helper to check if a button looks like a file upload button
        const isUploadButton = (btn) => {
            const id = (btn.id || '').toLowerCase();
            const cls = (btn.className || '').toLowerCase();
            const aria = (btn.getAttribute('aria-label') || '').toLowerCase();
            const title = (btn.getAttribute('title') || '').toLowerCase();
            const text = (btn.textContent || '').toLowerCase();
            return id.includes('upload') || id.includes('attach') || id.includes('file')
                || cls.includes('upload') || cls.includes('attach')
                || aria.includes('upload') || aria.includes('attach')
                || title.includes('upload') || title.includes('attach')
                || text.includes('upload') || text.includes('attach')
                || text.includes('file');
        };
        const candidates = [
            __BUTTON_SELECTORS__,
            document.querySelector('button[type="submit"]:not([disabled])'),
            form?.querySelector('button:not([disabled])'),
            input?.parentElement?.querySelector('button:not([disabled])'),
        ];
        const btn = candidates.find((candidate) =>
            candidate && !candidate.disabled && !isUploadButton(candidate)
                && candidate.getAttribute?.('aria-disabled') !== 'true'
        );
        if (btn) {
            // Prevent double submission: mark as submitted and block form events
            window.sbsllm_submitted = true;
            // For submit-type buttons in forms, prevent the form's native
            // submit handler from also firing after the click.
            // We do this by temporarily disabling the button after click.
            try {
                btn.disabled = true;
                btn.setAttribute('aria-disabled', 'true');
            } catch (_) {}
            btn.click();
            return 'OK';
        }
        // If a button was already clicked, don't fall through to form submit
        // or Enter key events (prevents double-send on sites like meta.ai
        // where button click also triggers form submission)
        if (window.sbsllm_submitted) {
            return 'OK';
        }
        try {
            if (form?.requestSubmit && !window.sbsllm_submitted) { form.requestSubmit(); return 'OK'; }
        } catch (_) {}
        if (input) {
            try {
                // Try to submit via form if available
                if (form && !window.sbsllm_submitted) {
                    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                    return 'ENTER_SENT';
                }
                // Fallback: dispatch Enter key events
                input.dispatchEvent(new KeyboardEvent('keydown', {
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                    bubbles: true, cancelable: true
                }));
                input.dispatchEvent(new KeyboardEvent('keypress', {
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                    bubbles: true, cancelable: true
                }));
                input.dispatchEvent(new KeyboardEvent('keyup', {
                    key: 'Enter', code: 'Enter', keyCode: 13, which: 13,
                    bubbles: true, cancelable: true
                }));
                return 'ENTER_SENT_UNVERIFIED';
            } catch (_) {}
        }
        return 'NO_BUTTON';
    })()
    """


def _inject_js(selectors: str) -> str:
    return _INJECT_TEMPLATE.replace("__SELECTORS__", selectors)


def _submit_js(button_selectors: str, input_selector: str | None = None) -> str:
    selector = input_selector or (
        "document.querySelector('textarea, [contenteditable], input[type=\"text\"]')"
    )
    return _SUBMIT_TEMPLATE.replace("__INPUT_SELECTOR__", selector).replace(
        "__BUTTON_SELECTORS__", button_selectors
    )


_RESPONSE_TEMPLATE = """
    (() => {
        const responseSelectors = __RESPONSE_SELECTORS__;
        const thinkingSelectors = __THINKING_SELECTORS__;
        const loadingSelectors = __LOADING_SELECTORS__;
        const textOf = (element) => {
            if (!element) return '';
            const text = element.innerText || element.textContent || '';
            return text.replace(/\\u00a0/g, ' ').trim();
        };
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            return style && style.visibility !== 'hidden' && style.display !== 'none'
                && element.getClientRects().length > 0;
        };
        const matches = (selectors) => {
            const elements = [];
            const seen = new Set();
            for (const selector of selectors) {
                try {
                    for (const element of document.querySelectorAll(selector)) {
                        if (!seen.has(element)) {
                            seen.add(element);
                            elements.push(element);
                        }
                    }
                } catch (_) {}
            }
            return elements;
        };
        const responses = matches(responseSelectors).filter(isVisible);
        const thinking = matches(thinkingSelectors).filter(isVisible);
        const loading = matches(loadingSelectors).filter(isVisible);
        const response = responses.at(-1) || null;
        const thinkingText = thinking.map(textOf).filter(Boolean).join('\\n\\n').trim();
        // Strip "Working for Xs" prefix that appears during streaming.
        // Also treat content starting with "Working for" as still loading.
        let content = textOf(response);
        const workingMatch = content.match(/^Working for \\d+s/);
        const isWorking = workingMatch !== null;
        content = content.replace(/^Working for \\d+s\\s*/, '');
        // Also strip any remaining "Worked for Xs" or similar timing prefixes
        content = content.replace(/^Worked for \\d+s\\s*/, '');
        return {
            found: response !== null,
            content: content,
            thinking: thinkingText || null,
            done: response !== null && loading.length === 0 && !isWorking,
            count: responses.length,
        };
    })()
    """


def _response_js(
    response_selectors: list[str] | tuple[str, ...] | None,
    thinking_selectors: list[str] | tuple[str, ...] | None = None,
    loading_selectors: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Build JS that extracts the newest assistant response from a page."""
    return (
        _RESPONSE_TEMPLATE.replace(
            "__RESPONSE_SELECTORS__", json.dumps(list(response_selectors or []))
        )
        .replace(
            "__THINKING_SELECTORS__", json.dumps(list(thinking_selectors or []))
        )
        .replace(
            "__LOADING_SELECTORS__", json.dumps(list(loading_selectors or []))
        )
    )


SITES: dict[str, dict] = {
    "chatgpt": {
        "url": "https://chatgpt.com/",
        "inject": _inject_js("""
            document.querySelector('textarea[data-id="root"]')
                || document.querySelector('#prompt-textarea')
                || document.querySelector('textarea[placeholder*="message"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[data-testid="send-button"]')
                || document.querySelector('button[aria-label="Send prompt"]')
                || document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
    },
    "claude": {
        "url": "https://claude.ai/",
        "inject": _inject_js("""
            document.querySelector('[contenteditable="true"]')
                || document.querySelector('.ProseMirror')
                || document.querySelector('[data-placeholder]')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label="Send Message"]')
                || document.querySelector('button[data-testid="send-button"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('[contenteditable="true"]')?.closest('form')?.querySelector('button')
        """),
    },
    "deepseek": {
        "url": "https://chat.deepseek.com/",
        "inject": _inject_js("""
            document.querySelector('#chat-input')
                || document.querySelector('textarea')
                || document.querySelector('div[contenteditable="true"]')
                || document.querySelector('[contenteditable]')
                || document.querySelector('textarea[placeholder*="Ask"]')
        """),
        "submit_js": _submit_js(
            """
                document.querySelector('button[data-testid*="send" i]')
                    || document.querySelector('button[aria-label*="send" i]')
                    || document.querySelector('button[aria-label*="submit" i]')
                    || document.querySelector('button[aria-label*="run" i]')
                    || document.querySelector('button[title*="send" i]')
                    || document.querySelector('button[title*="submit" i]')
                    || document.querySelector('button[class*="send" i]')
                    || document.querySelector('button[type="submit"]:not([disabled])')
                    || input?.parentElement?.querySelector('button:not([disabled])')
            """,
            "document.querySelector('textarea, #chat-input, [contenteditable]')",
        ),
    },
    "qwen": {
        "url": "https://chat.qwen.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="ask"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label="Send"]:not([disabled])')
                || document.querySelector('button[aria-label*="Send"]:not([disabled])')
                || document.querySelector('button[class*="send"]:not([disabled])')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button:not([disabled])')
        """),
    },
    "grok": {
        "url": "https://grok.com/",
        "inject": _inject_js("""
            document.querySelector('.tiptap.ProseMirror[contenteditable="true"]')
                || document.querySelector('div[contenteditable="true"][data-lexical-editor="true"]')
                || document.querySelector('div[role="textbox"][contenteditable="true"]')
                || document.querySelector('div[contenteditable="true"]')
                || document.querySelector('textarea[placeholder*="Ask" i]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[data-testid="chat-submit"]:not([disabled])')
                || document.querySelector('button[aria-label="Submit"]:not([disabled])')
                || document.querySelector('button[data-testid*="send" i]:not([disabled])')
                || document.querySelector('button[data-testid*="submit" i]:not([disabled])')
                || document.querySelector('button[aria-label*="Send" i]:not([disabled])')
                || document.querySelector('button[type="submit"]:not([disabled])')
                || document.querySelector('.tiptap, [contenteditable], textarea')?.closest('form')?.querySelector('button:not([disabled])')
        """, "document.querySelector('.tiptap, [contenteditable], textarea')"),
        "response_selectors": [
            '[data-testid="user-message"] ~ .message-bubble',
            '.message-bubble:not([data-testid="user-message"])',
            'div[class*="prose-chat"]',
        ],
        "thinking_selectors": [
            '[data-testid*="thinking"]',
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            '[class*="loading"]',
            '[class*="typing"]',
            '[class*="generating"]',
            'button[aria-label*="Stop" i]',
            '[class*="working"]',
            '[class*="streaming"]',
        ],
    },
    "google": {
        "url": "https://aistudio.google.com/",
        "inject": _inject_js("""
            document.querySelector('ms-prompt-box ms-autosize-textarea textarea')
                || document.querySelector('ms-prompt-box textarea[aria-label="Enter a prompt"]')
                || document.querySelector('ms-prompt-box textarea[aria-label="Type something"]')
                || document.querySelector('ms-prompt-box textarea')
                || document.querySelector('textarea[aria-label="Enter a prompt"]')
                || document.querySelector('textarea[aria-label="Type something"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('ms-prompt-box ms-run-button button[aria-label="Run"]')
                || document.querySelector('ms-prompt-box button[aria-label="Run"][type="submit"]')
                || document.querySelector('button[aria-label="Run"].run-button')
                || document.querySelector('ms-run-button button[type="submit"].run-button')
                || document.querySelector('button[aria-label*="Run" i]')
                || document.querySelector('button[aria-label*="Send" i]')
                || document.querySelector('button[aria-label*="Submit" i]')
                || document.querySelector('button[class*="build-button"]:not([disabled])')
                || document.querySelector('button[class*="ms-button-primary"]:not([disabled])')
                || document.querySelector('button[type="submit"]:not([disabled])')
                || document.querySelector('ms-prompt-box textarea')?.closest('form')?.querySelector('button:not([disabled])')
                || document.querySelector('ms-prompt-box textarea')?.parentElement?.querySelector('button:not([disabled])')
                || document.querySelector('ms-prompt-box textarea')?.parentElement?.parentElement?.querySelector('button:not([disabled])')
        """, "document.querySelector('ms-prompt-box textarea, textarea')"),
        "response_selectors": [
            'ms-chat-turn .chat-turn-container.model',
            'ms-chat-turn:has([data-turn-role="Model"])',
            'ms-chat-turn [data-turn-role="Model"]',
        ],
        "thinking_selectors": [
            'ms-chat-turn [class*="thinking"]',
            'ms-chat-turn [class*="reasoning"]',
        ],
        "loading_selectors": [
            'ms-run-button .stoppable-spinner',
            'ms-run-button button[aria-label="Run"] svg',
            '[class*="loading"]',
            'button[aria-label*="Stop" i]',
        ],
    },
    "mistral": {
        "url": "https://chat.mistral.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Send"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
    },
    "kimi": {
        "url": "https://kimi.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea.ph')
                || document.querySelector('textarea[name="message"]')
                || document.querySelector('textarea[placeholder*="What would you like to know"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
                || document.querySelector('div[contenteditable="true"]')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label="Submit"]')
                || document.querySelector('button.send')
                || document.querySelector('button[type="submit"]:not([disabled])')
                || document.querySelector('textarea.ph, textarea[name="message"], textarea')?.closest('form')?.querySelector('button:not([disabled])')
        """, "document.querySelector('textarea.ph, textarea[name=\"message\"], textarea')"),
        "response_selectors": [
            '[data-role="assistant"]',
            '[data-message-author-role="assistant"]',
            '.message.assistant',
            '[class*="assistant"] .markdown',
            'article .markdown',
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
            '[data-testid*="thinking"]',
        ],
        "loading_selectors": [
            '[class*="loading"]',
            '[class*="typing"]',
            '[class*="generating"]',
            'button[aria-label*="Stop" i]',
        ],
    },
    "perplexity": {
        "url": "https://www.perplexity.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Search"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Submit"]')
                || document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
    },
    "poe": {
        "url": "https://poe.com/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
    },
    "cohere": {
        "url": "https://cohere.com/chat",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[type="submit"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
    },
    "zai": {
        "url": "https://chat.z.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea#chat-input')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            // Must check for actual send button first to avoid file upload buttons
            document.querySelector('button#send-message-button:not([disabled])')
                || document.querySelector('button#send-message-button:not([class*="upload"]):not([class*="image"])')
                || document.querySelector('button[aria-label="Send"]:not([disabled]):not([class*="upload"]):not([class*="image"])')
                || document.querySelector('button[data-testid="send-message-button"]:not([disabled])')
                || document.querySelector('button[class*="send"]:not([disabled]):not([class*="upload"]):not([class*="image"])')
                // Fallback: find submit button in form that is not upload/attach
                || form?.querySelector('button[type="submit"]:not([disabled]):not([id*="upload"]):not([id*="attach"]):not([id*="file"])')
                || form?.querySelector('button[type="submit"]:not([disabled])')
        """, "document.querySelector('textarea#chat-input, textarea')"),
        "response_selectors": [
            '#response-content-container',
            '[data-message-author-role="assistant"]',
            '.message.assistant .markdown',
            '.chat-assistant .markdown',
            '.assistant-message .markdown',
            'article .markdown',
        ],
        "thinking_selectors": [
            '.thinking-block',
            '.thinking-chain-container',
            '[class*="thinking"]',
            '[class*="reasoning"]',
            '[data-testid*="thinking"]',
        ],
        "loading_selectors": [
            '[class*="loading"]',
            '[class*="typing"]',
            '[class*="generating"]',
            'button[aria-label*="Stop" i]',
            '#send-message-button[disabled]',
        ],
    },
    "meta": {
        "url": "https://meta.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "response_selectors": [
            '[data-testid="ai-message"]',
            '[data-message-author-role="assistant"]',
            '.assistant-message',
            '[class*="assistant-message"]',
            'article .markdown',
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
            '[data-testid*="thinking"]',
        ],
        "loading_selectors": [
            '[class*="loading"]',
            '[class*="typing"]',
            '[class*="generating"]',
            'button[aria-label*="Stop" i]',
        ],
    },
    "huggingface": {
        "url": "https://huggingface.co/chat",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[aria-label*="Submit"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
    },
    "tencent": {
        "url": "https://aistudio.tencent.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('div[contenteditable="true"]')
                || document.querySelector('[contenteditable]')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[data-testid="send-button"]')
                || document.querySelector('button[aria-label="Send"]')
                || document.querySelector('button[aria-label="Submit"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('button[class*="Send"]')
                || document.querySelector('button[type="submit"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
    },
}


def get_site(site_id: str) -> dict:
    """Get site config by ID. Raises ValueError if not found."""
    if site_id not in SITES:
        raise ValueError(f"Unknown site: {site_id!r}. Available: {', '.join(SITES)}")
    return SITES[site_id]


def list_sites() -> list[str]:
    """Return list of available site IDs."""
    return list(SITES.keys())
