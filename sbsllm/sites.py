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
        // Always (re)insert the fresh prompt value. Relying on a marker to skip
        // injection would leave the previous prompt in the box on repeat requests.
        input.focus();
        if (isEditable) {
            // Store the value for a post-inject hook that may need to
            // re-apply it via editor-specific APIs (e.g. TipTap/ProseMirror).
            try { input.dataset.sbsllmValue = value; } catch (_) {}
            // For contenteditable/TipTap editors, dispatch beforeinput first.
            // ProseMirror editors (including TipTap) listen for beforeinput
            // to update their internal document model. Simply setting
            // textContent does not trigger the editor's update cycle.
            try {
                const beforeInputEvent = new InputEvent('beforeinput', {
                    bubbles: true, cancelable: true,
                    inputType: 'insertText', data: value
                });
                input.dispatchEvent(beforeInputEvent);
            } catch (_) {}
            try {
                const selection = window.getSelection();
                selection?.selectAllChildren(input);
                document.execCommand('delete', false, null);
                document.execCommand('insertText', false, value);
            } catch (_) {}
            // execCommand may not work or may be ignored by editors; ensure textContent is set
            if ((input.textContent || '') !== value) {
                input.textContent = value;
            }
            // contenteditable divs don't fire `input` events the way
            // textarea/input do; dispatch both to be safe.
            try {
                input.dispatchEvent(new InputEvent('input', {
                    bubbles: true, inputType: 'insertText', data: value
                }));
            } catch (_) {
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }
            input.dispatchEvent(new Event('change', { bubbles: true }));
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
        // Mark so the submit step can locate this input, then clean up afterwards.
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
            if (!btn || typeof btn.getAttribute !== 'function') return false;
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
            candidate && typeof candidate.click === 'function' && !candidate.disabled
                && !isUploadButton(candidate)
                && candidate.getAttribute?.('aria-disabled') !== 'true'
        );
        if (btn && typeof btn.click === 'function') {
            // Click first so submit-type buttons (e.g. z.ai, meta.ai) still fire
            // their activation behavior; disabling before click would block it.
            btn.click();
            // Temporarily disable to prevent the form's native submit handler
            // from also firing after the click (double-send on meta.ai).
            // Restore on next tick so subsequent requests can still find it.
            try {
                btn.disabled = true;
                btn.setAttribute('aria-disabled', 'true');
                window.setTimeout(() => {
                    try { btn.disabled = false; btn.removeAttribute('aria-disabled'); } catch (_) {}
                }, 1500);
            } catch (_) {}
            return 'OK';
        }
        if (input && (input.tagName === 'TEXTAREA' || input.tagName === 'INPUT' || (input.isContentEditable === true) || input.getAttribute?.('contenteditable') === 'true')) {
            try {
                // For contenteditable editors, dispatch Enter via key events
                // (no form submit available). For textarea/input, prefer the
                // form's submit event.
                if (input.isContentEditable === true || input.getAttribute?.('contenteditable') === 'true') {
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
                }
                // Try to submit via form if available
                if (form) {
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


# Shared post-inject JS for contenteditable-based editors (ProseMirror/TipTap).
# These editors keep an internal document model that plain textContent/execCommand
# mutations in _INJECT_TEMPLATE may not fully sync. This block performs a second
# pass: focus, clear via innerText, re-insert via execCommand('insertText'), and
# fire beforeinput/input events so the editor's listeners catch up.
_POST_INJECT_CONTENTEDITABLE = """
    (() => {
        const el = document.querySelector('[data-sbsllm-input="true"]');
        if (!el) return 'NO_MARKED_INPUT';
        try {
            el.focus();
            el.innerText = '';
            document.execCommand('insertText', false, el.dataset.sbsllmValue || '');
            el.dispatchEvent(new InputEvent('beforeinput', {
                bubbles: true, cancelable: true,
                inputType: 'insertText', data: el.dataset.sbsllmValue || ''
            }));
            el.dispatchEvent(new InputEvent('input', {
                bubbles: true, inputType: 'insertText'
            }));
        } catch (_) {}
        return 'OK';
    })()
"""

# Selectors that positively indicate the site is still generating a reply.
# Bare `[class*="loading"]` is deliberately excluded: sites keep decorative
# skeletons and spinners in the DOM (often at zero opacity) long after
# generation ends, which pinned `done` to false and left the local chat
# streaming until the request deadline.
_LOADING_SELECTORS = [
    'button[aria-label*="Stop" i]',
    'button[aria-label*="Stop generating" i]',
    '[data-testid*="stop-button"]',
    '[aria-busy="true"]',
]

_RESPONSE_TEMPLATE = """
    (() => {
        const responseSelectors = __RESPONSE_SELECTORS__;
        const thinkingSelectors = __THINKING_SELECTORS__;
        const loadingSelectors = __LOADING_SELECTORS__;
        const isVisible = (element) => {
            if (!element) return false;
            const style = window.getComputedStyle(element);
            if (!style) return true;
            if (style.display === 'none') return false;
            if (style.visibility === 'hidden' || style.visibility === 'collapse') return false;
            // Skeletons and placeholders are laid out at zero opacity; treat
            // them as absent so they cannot keep a finished answer "loading".
            if (style.opacity !== '' && Number(style.opacity) < 0.05) return false;
            return element.getClientRects().length > 0;
        };
        const textOf = (element) => {
            if (!element) return '';
            const text = element.innerText || element.textContent || '';
            return text.replace(/\\u00a0/g, ' ');
        };
        // Detached nodes have no layout, so innerText is empty for clones.
        const rawTextOf = (element) => {
            if (!element) return '';
            return (element.textContent || '').replace(/\\u00a0/g, ' ');
        };
        const squash = (text) => (text || '').replace(/\\s+/g, ' ').trim();
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
        // A collapsed disclosure only renders its label ("Thought Process").
        // That is UI chrome, not a reasoning trace, so it must not be
        // forwarded to the local chat as thinking content.
        const ANSWER_HOST = 'div[class*="prose"], .markdown, [class*="markdown"]';
        const LABEL_ONLY = /^(?:thought\\s*process|thought|thinking|reasoning|deep\\s*think|chain\\s*of\\s*thought|思考(?:过程|中)?)[\\s:：0-9smh秒.,-]*$/i;
        const isLabelOnly = (text) => {
            const t = squash(text);
            return t === '' || LABEL_ONLY.test(t);
        };
        // Selectors are ordered fallbacks: use the FIRST one that matches.
        // Merging every selector and taking the last match let a catch-all
        // further down the list (div[id*=message] and friends) override the
        // precise selector and win for the wrong element entirely.
        let response = null;
        let responseCount = 0;
        for (const selector of responseSelectors) {
            let nodes;
            try {
                nodes = Array.from(document.querySelectorAll(selector)).filter(isVisible);
            } catch (_) {
                continue;
            }
            if (nodes.length) {
                response = nodes[nodes.length - 1];
                responseCount = nodes.length;
                break;
            }
        }

        // Reasoning: read each thinking container from a detached copy with
        // its collapsible header removed, and skip containers that only hold
        // the collapsed label.
        const thinkingParts = [];
        for (const node of matches(thinkingSelectors)) {
            if (!isVisible(node)) continue;
            const clone = node.cloneNode(true);
            for (const header of Array.from(clone.querySelectorAll(
                'button, [role="button"], summary, [aria-expanded]'
            ))) {
                if (header.parentNode) header.parentNode.removeChild(header);
            }
            const text = squash(textOf(clone) || rawTextOf(clone));
            if (isLabelOnly(text) || thinkingParts.indexOf(text) !== -1) continue;
            thinkingParts.push(text);
        }
        const thinkingText = thinkingParts.join('\\n\\n').trim();

        // Answer: prune the reasoning subtree from a copy of the response
        // instead of string-replacing its text. z.ai nests the disclosure
        // inside the same container as the answer, so replacing text there
        // either left the label behind or ate the answer.
        let content = textOf(response);
        if (response && thinkingSelectors.length) {
            // A broad thinking selector can match a node that also wraps the
            // answer. z.ai keeps the answer in a .markdown-prose block, so
            // treat any candidate containing one as an answer host and leave
            // it alone rather than deleting the answer with the reasoning.
            const wrapsAnswer = (node) => {
                try {
                    return node.matches(ANSWER_HOST) ||
                        node.querySelector(ANSWER_HOST) !== null;
                } catch (_) {
                    return false;
                }
            };
            const clone = response.cloneNode(true);
            let pruned = false;
            for (const selector of thinkingSelectors) {
                let nodes;
                try {
                    nodes = Array.from(clone.querySelectorAll(selector));
                } catch (_) {
                    continue;
                }
                for (const node of nodes) {
                    if (!node.parentNode || wrapsAnswer(node)) continue;
                    node.parentNode.removeChild(node);
                    pruned = true;
                }
            }
            if (pruned) {
                // Use the pruned text even when empty. During the thinking
                // phase the answer is legitimately empty, and falling back to
                // the unpruned text there re-injected the reasoning into the
                // answer, so the local chat rendered the thinking twice.
                content = squash(rawTextOf(clone));
            }
        }
        // Safety net for sites that render the disclosure label outside any
        // element matched by thinking_selectors.
        content = content.replace(/Thought Process\\s*/gi, '');
        content = squash(content);
        // "Working for 12s" / "Worked for 12s" appear while a reply streams.
        const isWorking = /^Work(?:ing|ed) for \\d+s/.test(content);
        content = content.replace(/^Work(?:ing|ed) for \\d+s\\s*/, '');
        content = content.trim();

        // Still generating? Require a positive signal: a stop control, an
        // aria-busy region, or a busy node inside the answer itself.
        const loading = matches(loadingSelectors).filter((element) => {
            if (!isVisible(element)) return false;
            if (response && (response === element || response.contains(element))) {
                return true;
            }
            if (element.getAttribute('aria-busy') === 'true') return true;
            const tag = (element.tagName || '').toLowerCase();
            return tag === 'button' || element.getAttribute('role') === 'button';
        });
        const busy = loading.length > 0;
        // Login-wall detection: a positive signal that the chat requires a
        // signed-in session. Checked AFTER a submit attempt so that a
        // logged-out home page (where `found` is legitimately false) does
        // not trigger a false positive. The page text is the most reliable
        // cross-site signal; specific selectors are site-specific bonuses.
        let loginWall = false;
        const bodyText = (document.body && document.body.innerText) || '';
        const loginPatterns = [
            /sign up to continue/i,
            /sign in to continue/i,
            /please log in to continue/i,
            /login to continue/i,
            /continue your conversation/i,
            /you must be logged in/i,
        ];
        for (const pattern of loginPatterns) {
            if (pattern.test(bodyText)) {
                loginWall = true;
                break;
            }
        }
        return {
            found: response !== null,
            content: content,
            thinking: thinkingText || null,
            // `busy` is the raw "still generating" signal. `done` alone is
            // not trustworthy: sites render their stop control a moment after
            // the first token, so `done` is briefly true while the answer is
            // still arriving. Callers must use `busy` to tell a real finish
            // from the gap between the start of a turn and its spinner.
            busy: busy,
            done: response !== null && !isWorking && !busy,
            count: responseCount,
            // `login_wall` is a positive signal that the chat requires a
            // signed-in session. It is checked AFTER a submit attempt so
            // that a logged-out home page (where `found` is legitimately
            // false) does not trigger a false positive.
            login_wall: loginWall,
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
        .replace("__THINKING_SELECTORS__", json.dumps(list(thinking_selectors or [])))
        .replace("__LOADING_SELECTORS__", json.dumps(list(loading_selectors or [])))
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
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "submit_js": _submit_js("""
            document.querySelector('button[data-testid="send-button"]')
                || document.querySelector('button[aria-label="Send prompt"]')
                || document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "response_selectors": [
            'div[data-message-author-role="assistant"]',
            ".markdown",
            'div[data-testid="conversation"] > div > div',
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
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
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"] .message',
        ],
        "post_inject_js": """
            (() => {
                // Claude uses ProseMirror/contenteditable. The main inject sets
                // textContent, but that does NOT update the editor's internal
                // document model. We must use innerText + execCommand +
                // beforeinput to make the editor register the new value.
                const el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) return 'NO_MARKED_INPUT';
                try {
                    el.focus();
                    el.innerText = '';
                    document.execCommand('insertText', false, el.dataset.sbsllmValue || '');
                    el.dispatchEvent(new InputEvent('beforeinput', {
                        bubbles: true, cancelable: true,
                        inputType: 'insertText', data: el.dataset.sbsllmValue || ''
                    }));
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true, inputType: 'insertText'
                    }));
                } catch (_) {}
                return 'OK';
            })()
        """,
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "deepseek": {
        "url": "https://chat.deepseek.com/",
        "inject": _inject_js("""
            document.querySelector('#chat-input')
                || document.querySelector('textarea')
                || document.querySelector('div[contenteditable="true"]')
                || document.querySelector('[contenteditable]')
                || document.querySelector('input[type="text"]:not([placeholder*="Phone"]):not([placeholder*="Email"])')
                || document.querySelector('input[type="text"]')
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
            "document.querySelector('textarea, #chat-input, [contenteditable], input[type=\"text\"]')",
        ),
        "post_inject_js": """
            (() => {
                // DeepSeek uses contenteditable. The main inject sets textContent,
                // but that does NOT update the editor's internal document model.
                // We must use innerText + execCommand + beforeinput to make the
                // editor register the new value.
                const el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) return 'NO_MARKED_INPUT';
                try {
                    el.focus();
                    el.innerText = '';
                    document.execCommand('insertText', false, el.dataset.sbsllmValue || '');
                    el.dispatchEvent(new InputEvent('beforeinput', {
                        bubbles: true, cancelable: true,
                        inputType: 'insertText', data: el.dataset.sbsllmValue || ''
                    }));
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true, inputType: 'insertText'
                    }));
                } catch (_) {}
                return 'OK';
            })()
        """,
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "qwen": {
        "url": "https://chat.qwen.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="ask"]')
                || document.querySelector('textarea')
        """),
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label="Send"]:not([disabled])')
                || document.querySelector('button[aria-label*="Send"]:not([disabled])')
                || document.querySelector('button[class*="send"]:not([disabled])')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button:not([disabled])')
        """),
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "grok": {
        "url": "https://grok.com/",
        "inject": _inject_js("""
            document.querySelector('.tiptap.ProseMirror[contenteditable="true"]')
                || document.querySelector('div[contenteditable="true"][data-lexical-editor="true"]')
                || document.querySelector('div[role="textbox"][contenteditable="true"]')
                || document.querySelector('div[contenteditable="true"]')
                || document.querySelector('textarea[class*="prose"]')
                || document.querySelector('textarea[placeholder*="Ask" i]')
                || document.querySelector('textarea')
        """),
        "post_inject_js": """
            (() => {
                // Grok uses TipTap/ProseMirror. The main inject above sets
                // textContent, but that does NOT update the editor's internal
                // document model. We must use innerText + execCommand +
                // beforeinput to make the editor register the new value.
                // This matches the pattern used by working grok automation scripts.
                const el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) return 'NO_MARKED_INPUT';
                try {
                    // Clear existing content
                    el.focus();
                    el.innerText = '';
                    // Use execCommand to insert text (works with ProseMirror)
                    document.execCommand('insertText', false, el.dataset.sbsllmValue || '');
                    // Dispatch beforeinput for TipTap listeners
                    el.dispatchEvent(new InputEvent('beforeinput', {
                        bubbles: true, cancelable: true,
                        inputType: 'insertText', data: el.dataset.sbsllmValue || ''
                    }));
                    // Dispatch input event
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true, inputType: 'insertText'
                    }));
                } catch (_) {}
                return 'OK';
            })()
        """,
        "submit_js": _submit_js(
            """
            document.querySelector('button[data-testid="chat-submit"]:not([disabled])')
                || document.querySelector('button[aria-label="Submit"]:not([disabled])')
                || document.querySelector('button[data-testid*="send" i]:not([disabled])')
                || document.querySelector('button[data-testid*="submit" i]:not([disabled])')
                || document.querySelector('button[aria-label*="Send" i]:not([disabled])')
                || document.querySelector('button[type="submit"]:not([disabled])')
                || document.querySelector('.tiptap, [contenteditable], textarea')?.closest('form')?.querySelector('button:not([disabled])')
        """,
            "document.querySelector('.tiptap, [contenteditable], textarea')",
        ),
        "response_selectors": [
            '.message-bubble:not([data-testid="user-message"])',
            '[data-testid="assistant-message"]',
            '[data-message-author-role="assistant"]',
            'div[class*="prose-chat"]',
        ],
        "thinking_selectors": [
            '[data-testid*="thinking"]',
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
            '[class*="working"]',
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
        "post_inject_js": """
            (() => {
                // Google AI Studio's ms-autosize-textarea web component keeps
                // internal state (data-value attribute, disabled flag on the
                // Run button) that does not sync when .value is set via the
                // property setter. Sync the wrapper so the Run button enables.
                const el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) return 'NO_MARKED_INPUT';
                const autosize = el.closest('ms-autosize-textarea');
                if (autosize) {
                    autosize.setAttribute('data-value', el.value || '');
                }
                try {
                    // Dispatch a proper InputEvent (not just Event) so the
                    // web component's internal listener updates its state.
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true, cancelable: true,
                        inputType: 'insertText', data: el.value
                    }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                } catch (_) {
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }
                // The Run button lives inside ms-run-button; enable it directly
                // so the submit step can click it. The component may keep its
                // own disabled flag, but removing ours helps.
                const runButton = document.querySelector('ms-run-button');
                if (runButton) {
                    try {
                        runButton.removeAttribute('disabled');
                        const inner = runButton.querySelector('button');
                        if (inner) {
                            inner.removeAttribute('disabled');
                            inner.setAttribute('aria-disabled', 'false');
                        }
                    } catch (_) {}
                }
                return 'OK';
            })()
        """,
        "submit_js": _submit_js(
            """
            // Google AI Studio: click the Run button inside ms-run-button or the
            // ms-prompt-box. If no button is found, dispatch Enter on the
            // textarea — AI Studio submits on Enter.
            document.querySelector('ms-run-button button[aria-label="Run"]')
                || document.querySelector('ms-prompt-box ms-run-button button[aria-label="Run"]')
                || document.querySelector('ms-prompt-box ms-run-button button[type="submit"]')
                || document.querySelector('ms-run-button button[type="submit"].run-button')
                || document.querySelector('button[aria-label="Run"].run-button')
                || document.querySelector('button[aria-label*="Run" i]')
                || document.querySelector('button[aria-label*="Send" i]')
                || document.querySelector('button[aria-label*="Submit" i]')
                || document.querySelector('button[class*="build-button"]:not([disabled])')
                || document.querySelector('button[class*="ms-button-primary"]:not([disabled])')
                || document.querySelector('button[type="submit"]:not([disabled])')
                || document.querySelector('ms-prompt-box textarea')?.closest('form')?.querySelector('button:not([disabled])')
                || document.querySelector('ms-prompt-box textarea')?.parentElement?.querySelector('button:not([disabled])')
                || document.querySelector('ms-prompt-box textarea')?.parentElement?.parentElement?.querySelector('button:not([disabled])')
        """,
            "document.querySelector('ms-prompt-box textarea, textarea')",
        ),
        "response_selectors": [
            "ms-chat-turn .chat-turn-container.model",
            'ms-chat-turn:has([data-turn-role="Model"])',
            'ms-chat-turn [data-turn-role="Model"]',
        ],
        "thinking_selectors": [
            'ms-chat-turn [class*="thinking"]',
            'ms-chat-turn [class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
            "ms-run-button .stoppable-spinner",
        ],
    },
    "mistral": {
        "url": "https://chat.mistral.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Send"]')
                || document.querySelector('textarea')
                || document.querySelector('.ProseMirror[contenteditable="true"]')
                || document.querySelector('[contenteditable="true"]')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
                || document.querySelector('.ProseMirror')?.closest('form')?.querySelector('button')
        """),
        "post_inject_js": """
            (() => {
                // Mistral uses ProseMirror/contenteditable. The main inject sets
                // textContent, but that does NOT update the editor's internal
                // document model. We must use innerText + execCommand +
                // beforeinput to make the editor register the new value.
                const el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) return 'NO_MARKED_INPUT';
                try {
                    el.focus();
                    el.innerText = '';
                    document.execCommand('insertText', false, el.dataset.sbsllmValue || '');
                    el.dispatchEvent(new InputEvent('beforeinput', {
                        bubbles: true, cancelable: true,
                        inputType: 'insertText', data: el.dataset.sbsllmValue || ''
                    }));
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true, inputType: 'insertText'
                    }));
                } catch (_) {}
                return 'OK';
            })()
        """,
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "kimi": {
        "url": "https://kimi.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea.ph')
                || document.querySelector('textarea[name="message"]')
                || document.querySelector('textarea[placeholder*="What would you like to know"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
                || document.querySelector('div.chat-input-editor[contenteditable="true"]')
                || document.querySelector('div[contenteditable="true"]')
        """),
        "submit_js": _submit_js(
            """
            document.querySelector('button[aria-label="Submit"]')
                || document.querySelector('button.send')
                || document.querySelector('button[type="submit"]:not([disabled])')
                || document.querySelector('textarea.ph, textarea[name="message"], textarea, div.chat-input-editor')?.closest('form')?.querySelector('button:not([disabled])')
                || document.querySelector('.chat-input-editor')?.parentElement?.querySelector('button:not([disabled])')
        """,
            "document.querySelector('textarea.ph, textarea[name=\"message\"], textarea, div.chat-input-editor')",
        ),
        "post_inject_js": """
            (() => {
                // Kimi uses a contenteditable div (chat-input-editor) as a
                // fallback. The main inject sets textContent, but that does
                // NOT update the editor's internal document model. We must
                // use innerText + execCommand + beforeinput to make the
                // editor register the new value.
                const el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) return 'NO_MARKED_INPUT';
                try {
                    el.focus();
                    el.innerText = '';
                    document.execCommand('insertText', false, el.dataset.sbsllmValue || '');
                    el.dispatchEvent(new InputEvent('beforeinput', {
                        bubbles: true, cancelable: true,
                        inputType: 'insertText', data: el.dataset.sbsllmValue || ''
                    }));
                    el.dispatchEvent(new InputEvent('input', {
                        bubbles: true, inputType: 'insertText'
                    }));
                } catch (_) {}
                return 'OK';
            })()
        """,
        "response_selectors": [
            '[data-role="assistant"]',
            '[data-message-author-role="assistant"]',
            ".message.assistant",
            '[class*="assistant"] .markdown',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
            '[data-testid*="thinking"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "perplexity": {
        "url": "https://www.perplexity.ai/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Search"]')
                || document.querySelector('textarea')
                || document.querySelector('[contenteditable="true"]')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Submit"]')
                || document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "poe": {
        "url": "https://poe.com/",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
                || document.querySelector('[contenteditable="true"]')
        """),
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "cohere": {
        "url": "https://cohere.com/chat",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
                || document.querySelector('[contenteditable="true"]')
        """),
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*=\"Send\"]')
                || document.querySelector('button[type=\"submit\"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "zai": {
        "url": "https://chat.z.ai/auth",
        "inject": _inject_js("""
            document.querySelector('textarea#chat-input')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea')
        """),
        "post_inject_js": """
            (() => {
                // z.ai's textarea value is already set by the main inject IIFE.
                // No post-inject sync needed here. Previously this block
                // dispatched Enter key events, which triggered the form's
                // native submit handler -- then submit_js clicked the send
                // button again, causing the prompt to be sent twice.
                const el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) return 'NO_MARKED_INPUT';
                return 'OK';
            })()
        """,
        "submit_js": _submit_js(
            """
            // z.ai's send button is a custom element. Click it directly --
            // the _SUBMIT_TEMPLATE already disables the button after clicking
            // to prevent the form's native submit handler from firing a
            // second time. Do NOT add Enter-key dispatch here: post_inject_js
            // must also stay free of Enter events, otherwise the form submits
            // once via the key listener and again via the button click.
            document.querySelector('button#send-message-button:not([disabled])')
                || document.querySelector('button#send-message-button:not([class*="upload"]):not([class*="image"])')
                || document.querySelector('button[aria-label="Send"]:not([disabled]):not([class*="upload"]):not([class*="image"])')
                || document.querySelector('button[data-testid="send-message-button"]:not([disabled])')
                || document.querySelector('button[class*="send"]:not([disabled]):not([class*="upload"]):not([class*="image"])')
                // Fallback: find submit button in form that is not upload/attach
                || form?.querySelector('button[type="submit"]:not([disabled]):not([id*="upload"]):not([id*="attach"]):not([id*="file"])')
                || form?.querySelector('button[type="submit"]:not([disabled])')
        """,
            "document.querySelector('textarea#chat-input, textarea')",
        ),
        "setup_js": """
            (() => {
                try {
                    // Click the Deep Think dropdown and select "High" if not already set.
                    const trigger = document.querySelector('#bits-c286[aria-haspopup="menu"]');
                    if (!trigger || trigger.getAttribute('aria-expanded') === 'true') return 'SKIP';
                    trigger.click();
                    return new Promise(resolve => {
                        setTimeout(() => {
                            const items = document.querySelectorAll('[role="menu"] button');
                            for (const item of items) {
                                if (item.textContent.trim().toLowerCase() === 'high') {
                                    item.click();
                                    resolve('SET_HIGH');
                                    return;
                                }
                            }
                            resolve('NO_HIGH_FOUND');
                        }, 300);
                    });
                } catch (e) {
                    return 'SKIP_ERROR: ' + e.message;
                }
            })()
        """,
        "response_selectors": [
            "#response-content-container .markdown-prose",
            "#response-content-container",
            '[data-message-author-role="assistant"]',
            ".message.assistant .markdown",
            ".chat-assistant .markdown",
            ".assistant-message .markdown",
            "article .markdown",
            # Fallbacks from external automation scripts
            'div[class*="prose"]',
            'div[class*="markdown"]',
            'div[class*="chat-assistant"]',
            'div[class*="response"]',
            '[data-message-role="assistant"]',
            "div.chat-message",
            'div[id*="message"]',
        ],
        "thinking_selectors": [
            ".thinking-block",
            ".thinking-chain-container",
            '[class*="thinking"]',
            '[class*="reasoning"]',
            '[data-testid*="thinking"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
            # z.ai uses an animated dot loader inside the response container
            # while a reply is being generated. Without this, `busy` is always
            # False and the server terminates the stream before any content
            # arrives.
            "#response-content-container .dot",
            ".skeleton.loading",
        ],
    },
    "meta": {
        "url": "https://meta.ai/",
        "inject": _inject_js("""
            document.querySelector('input[aria-label="Ask Meta AI"]')
                || document.querySelector('input[placeholder*="Ask Meta AI"]')
                || document.querySelector('input[placeholder*="Ask"]')
                || document.querySelector('input[placeholder*="Message"]')
                || document.querySelector('input[type="text"]')
                || document.querySelector('textarea')
                || document.querySelector('[contenteditable="true"]')
        """),
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "setup_js": """
            (() => {
                // Meta AI shows a welcome screen on first load. Dismiss it so
                // the input is ready for the prompt. The "Get Started" button
                // appears in the welcome overlay; clicking it or pressing Escape
                // closes the overlay.
                try {
                    const overlay = document.querySelector('[role="dialog"]')
                        || document.querySelector('.welcome')
                        || document.querySelector('[class*="onboarding"]');
                    if (overlay) {
                        const dismiss = overlay.querySelector('button')
                            || document.querySelector('button[aria-label="Dismiss"]')
                            || document.querySelector('button[aria-label="Close"]');
                        if (dismiss) { dismiss.click(); return 'DISMISSED'; }
                    }
                } catch (_) {}
                return 'OK';
            })()
        """,
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*="Send"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "response_selectors": [
            '[data-testid="ai-message"]',
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant-message"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
            '[data-testid*="thinking"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "huggingface": {
        "url": "https://huggingface.co/chat",
        "inject": _inject_js("""
            document.querySelector('textarea[placeholder*="Message"]')
                || document.querySelector('textarea[placeholder*="Ask"]')
                || document.querySelector('textarea')
                || document.querySelector('[contenteditable="true"]')
        """),
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "submit_js": _submit_js("""
            document.querySelector('button[aria-label*=\"Send\"]')
                || document.querySelector('button[aria-label*=\"Submit\"]')
                || document.querySelector('button[class*=\"send\"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
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
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "submit_js": _submit_js("""
            document.querySelector('button[data-testid="send-button"]')
                || document.querySelector('button[aria-label="Send"]')
                || document.querySelector('button[aria-label="Submit"]')
                || document.querySelector('button[class*="send"]')
                || document.querySelector('button[class*="Send"]')
                || document.querySelector('button[type="submit"]')
                || document.querySelector('textarea')?.closest('form')?.querySelector('button')
        """),
        "response_selectors": [
            '[data-message-author-role="assistant"]',
            ".assistant-message",
            '[class*="assistant"]',
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
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
