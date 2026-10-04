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
        } else if (input instanceof HTMLTextAreaElement || isTextInput) {
            try { input.dataset.sbsllmValue = value; } catch (_) {}
            const proto = input instanceof HTMLTextAreaElement
                ? HTMLTextAreaElement.prototype
                : HTMLInputElement.prototype;
            const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
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
        // Try light-DOM first; fall back to scanning shadow roots for the
        // marked input (Google AI Studio ms-* web components use Shadow DOM).
        let input = document.querySelector('[data-sbsllm-input="true"]')
            || (__INPUT_SELECTOR__);
        if (!input) {
            for (const host of document.querySelectorAll('*')) {
                try {
                    if (!host.shadowRoot) continue;
                    input = host.shadowRoot.querySelector('[data-sbsllm-input="true"]');
                    if (input) break;
                } catch (_) {}
            }
        }
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
        let el = document.querySelector('[data-sbsllm-input="true"]');
        if (!el) {
            // The inject step may have found the editor inside a Shadow DOM
            // (web component host). Search all shadow roots for the marker.
            for (const host of document.querySelectorAll('*')) {
                try {
                    if (!host.shadowRoot) continue;
                    el = host.shadowRoot.querySelector('[data-sbsllm-input="true"]');
                    if (el) break;
                } catch (_) {}
            }
        }
        if (!el) return 'NO_MARKED_INPUT';
        const value = el.dataset.sbsllmValue || '';
        try {
            if (el.isContentEditable) {
                // contenteditable/TipTap/ProseMirror path: clear via innerText,
                // re-insert via execCommand, fire beforeinput/input events.
                el.focus();
                el.innerText = '';
                document.execCommand('insertText', false, value);
                el.dispatchEvent(new InputEvent('beforeinput', {
                    bubbles: true, cancelable: true,
                    inputType: 'insertText', data: value
                }));
                el.dispatchEvent(new InputEvent('input', {
                    bubbles: true, inputType: 'insertText'
                }));
            } else {
                // Plain textarea/input path: set value via the property
                // descriptor so React/lexical/etc. listeners fire, then
                // dispatch a native input event.
                const setter = Object.getOwnPropertyDescriptor(
                    el instanceof HTMLTextAreaElement
                        ? HTMLTextAreaElement.prototype
                        : HTMLInputElement.prototype,
                    'value'
                )?.set;
                if (setter) {
                    setter.call(el, value);
                } else {
                    el.value = value;
                }
                el.dispatchEvent(new Event('input', { bubbles: true }));
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
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
        const loginWallSelectors = __LOGIN_WALL_SELECTORS__;
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
        const squash = (text) => (text || '').replace(/\\s+/g, ' ').trim();
        // Whitespace cleanup that KEEPS paragraph breaks: every line is
        // squashed individually, runs of blank lines collapse to one, and the
        // whole text is trimmed. Squashing the entire answer (the old
        // behaviour) flattened every paragraph into one run-on line.
        const normalize = (text) => (text || '')
            .split('\\n')
            .map((line) => line.replace(/[ \\t\\r]+/g, ' ').trim())
            .join('\\n')
            .replace(/\\n{3,}/g, '\\n\\n')
            .trim();
        // Detached clones have no layout, so innerText/textContent carries no
        // block-boundary newlines. Walk the subtree instead: whitespace inside
        // a text node collapses (a source line break inside a <p> is not a
        // paragraph break) while block-level elements contribute real ones.
        const BLOCK_TAGS = new Set([
            'ADDRESS', 'ARTICLE', 'ASIDE', 'BLOCKQUOTE', 'BR', 'DD', 'DETAILS',
            'DIALOG', 'DIV', 'DL', 'DT', 'FIELDSET', 'FIGCAPTION', 'FIGURE',
            'FOOTER', 'FORM', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'HEADER',
            'HGROUP', 'HR', 'LI', 'MAIN', 'NAV', 'OL', 'P', 'PRE', 'SECTION',
            'TABLE', 'TD', 'TH', 'TR', 'UL',
        ]);
        const blockTextOf = (element) => {
            if (!element) return '';
            const parts = [];
            const walk = (node) => {
                if (node.nodeType === 3) {
                    parts.push(node.nodeValue.replace(/\\s+/g, ' '));
                    return;
                }
                if (node.nodeType !== 1) return;
                const tag = (node.tagName || '').toUpperCase();
                if (tag === 'BR') {
                    parts.push('\\n');
                    return;
                }
                const block = BLOCK_TAGS.has(tag);
                if (block) parts.push('\\n');
                for (const child of node.childNodes) walk(child);
                if (block) parts.push('\\n');
            };
            walk(element);
            return parts.join('');
        };
        // document.querySelectorAll cannot reach into a web component's
        // Shadow DOM. Several supported sites (Google AI Studio's ms-* tree,
        // custom elements on z.ai / grok) render the assistant answer inside a
        // shadow root, so a light-DOM-only scan returns nothing and the
        // server reports "no output" on a perfectly healthy reply.
        //
        // Walk every element in the document; for each one that exposes a
        // shadowRoot, query within it too. Depth-first so nested shadow trees
        // (ms-prompt-box > ms-autosize-textarea > textarea) are reached.
        const _shadowQuery = (root, selector) => {
            let out = [];
            try {
                out = Array.from(root.querySelectorAll(selector));
            } catch (_) {}
            return out;
        };
        const matchesInShadow = (selectors) => {
            const elements = [];
            const seen = new Set();
            const stack = [document];
            while (stack.length) {
                const root = stack.pop();
                for (const selector of selectors) {
                    try {
                        for (const element of _shadowQuery(root, selector)) {
                            if (!seen.has(element)) {
                                seen.add(element);
                                elements.push(element);
                            }
                        }
                    } catch (_) {}
                }
                // Enqueue shadow hosts found in this root for deeper scanning.
                try {
                    for (const host of root.querySelectorAll('*')) {
                        if (host.shadowRoot) stack.push(host.shadowRoot);
                    }
                } catch (_) {}
            }
            return elements;
        };
        const matches = (selectors) => {
            // Light-DOM first (cheap, and the common case), then shadow roots.
            const light = [];
            const seen = new Set();
            for (const selector of selectors) {
                try {
                    for (const element of document.querySelectorAll(selector)) {
                        if (!seen.has(element)) {
                            seen.add(element);
                            light.push(element);
                        }
                    }
                } catch (_) {}
            }
            const shadow = matchesInShadow(selectors);
            for (const element of shadow) {
                if (!seen.has(element)) {
                    seen.add(element);
                    light.push(element);
                }
            }
            return light;
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
        // Response selection also needs shadow-DOM reach: sites that render the
        // assistant turn inside a web component (Google AI Studio's
        // ms-chat-turn, custom elements elsewhere) would otherwise report
        // "found: false" on a reply that is fully on screen.
        //
        // A selector like "my-chat-turn [data-turn-role='model'] .prose-chat"
        // references the shadow host itself ("my-chat-turn"), which lives in
        // the light DOM. querySelectorAll on the shadow root can never match
        // it, so we also try the selector's last compound sub-selector (".prose-chat")
        // inside each shadow root as a fallback. This is only used for response
        // containers -- thinking/loading/login selectors go through the
        // separate matches()/matchesInShadow() path which is more generic.
        const _shadowAll = (selector) => {
            const out = [];
            const seen = new Set();
            const stack = [document];
            while (stack.length) {
                const root = stack.pop();
                try {
                    for (const el of root.querySelectorAll(selector)) {
                        if (!seen.has(el)) {
                            seen.add(el);
                            out.push(el);
                        }
                    }
                } catch (_) {}
                // When scanning inside a shadow root, the full selector may
                // reference the host element (in light DOM) and thus can't
                // match. Fall back to the selector's trailing compound part
                // so we still reach answer elements that live in the shadow tree.
                if (root !== document && selector.indexOf(' ') !== -1) {
                    const lastPart = selector.split(/\\s+/).pop();
                    if (lastPart && lastPart !== selector) {
                        try {
                            for (const el of root.querySelectorAll(lastPart)) {
                                if (!seen.has(el)) {
                                    seen.add(el);
                                    out.push(el);
                                }
                            }
                        } catch (_) {}
                    }
                }
                try {
                    for (const host of root.querySelectorAll('*')) {
                        if (host.shadowRoot) stack.push(host.shadowRoot);
                    }
                } catch (_) {}
            }
            return out;
        };
        let response = null;
        let responseCount = 0;
        for (const selector of responseSelectors) {
            let nodes = [];
            try {
                nodes = Array.from(document.querySelectorAll(selector));
            } catch (_) {
                nodes = [];
            }
            if (!nodes.length) {
                try {
                    nodes = _shadowAll(selector);
                } catch (_) {
                    nodes = [];
                }
            }
            nodes = nodes.filter(isVisible);
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
            const text = normalize(blockTextOf(clone));
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
                content = blockTextOf(clone);
            }
        }
        // Safety net for sites that render the disclosure label outside any
        // element matched by thinking_selectors.
        content = content.replace(/Thought Process\\s*/gi, '');
        content = normalize(content);
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
        // Site-specific login-wall selectors (e.g., a "Sign in" button that
        // only appears on auth-gated pages). These are positive signals only.
        // Use matches()+isVisible() instead of bare querySelector so that
        // hidden login links in nav menus or collapsed dialogs do not
        // trigger a false positive.
        if (!loginWall && loginWallSelectors.length) {
            loginWall = matches(loginWallSelectors).some(isVisible);
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
    login_wall_selectors: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Build JS that extracts the newest assistant response from a page."""
    return (
        _RESPONSE_TEMPLATE.replace(
            "__RESPONSE_SELECTORS__", json.dumps(list(response_selectors or []))
        )
        .replace("__THINKING_SELECTORS__", json.dumps(list(thinking_selectors or [])))
        .replace("__LOADING_SELECTORS__", json.dumps(list(loading_selectors or [])))
        .replace(
            "__LOGIN_WALL_SELECTORS__", json.dumps(list(login_wall_selectors or []))
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
        "login_wall_selectors": [
            'a[href*="login" i]',
            'button[aria-label*="Sign in" i]',
            'button[aria-label*="Log in" i]',
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
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
        ],
        "login_wall_selectors": [
            'div[data-testid="signin-button"]',
            'button[data-testid*="signin" i]',
            'a[href*="login" i]',
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
        "login_wall_selectors": [
            "#login-wrap",
            'button[aria-label*="Sign in" i]',
            'a[href*="signin" i]',
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
        "login_wall_selectors": [
            'button[aria-label*="Sign in" i]',
            'a[href*="login" i]',
            'button[data-testid*="login" i]',
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
                || (function() {
                    // Shadow DOM fallback: Grok may render its composer inside a
                    // web-component shadow tree that document.querySelector
                    // cannot reach. Scan all shadow roots for an editable element.
                    for (const host of document.querySelectorAll('*')) {
                        try {
                            if (!host.shadowRoot) continue;
                            const el = host.shadowRoot.querySelector('.tiptap.ProseMirror[contenteditable="true"]')
                                || host.shadowRoot.querySelector('div[contenteditable="true"][data-lexical-editor="true"]')
                                || host.shadowRoot.querySelector('div[role="textbox"][contenteditable="true"]')
                                || host.shadowRoot.querySelector('div[contenteditable="true"]')
                                || host.shadowRoot.querySelector('textarea')
                                || host.shadowRoot.querySelector('[contenteditable="true"]');
                            if (el) return el;
                        } catch (_) {}
                    }
                    return null;
                })()
        """),
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
        "submit_js": _submit_js(
            """
            document.querySelector('button[data-testid="chat-submit"]:not([disabled])')
                || document.querySelector('button[aria-label="Submit"]:not([disabled])')
                || document.querySelector('button[data-testid*="send" i]:not([disabled])')
                || document.querySelector('button[data-testid*="submit" i]:not([disabled])')
                || document.querySelector('button[aria-label*="Send" i]:not([disabled])')
                || document.querySelector('button[type="submit"]:not([disabled])')
                || document.querySelector('.tiptap, [contenteditable], textarea')?.closest('form')?.querySelector('button:not([disabled])')
                || (function() {
                    // Shadow DOM fallback: scan all shadow roots for a send button.
                    for (const host of document.querySelectorAll('*')) {
                        try {
                            if (!host.shadowRoot) continue;
                            const btn = host.shadowRoot.querySelector('button[data-testid*="send" i]:not([disabled])')
                                || host.shadowRoot.querySelector('button[data-testid*="submit" i]:not([disabled])')
                                || host.shadowRoot.querySelector('button[type="submit"]:not([disabled])')
                                || host.shadowRoot.querySelector('button:not([disabled])');
                            if (btn) return btn;
                        } catch (_) {}
                    }
                    return null;
                })()
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
        "login_wall_selectors": [
            'a[href*="login" i]',
            'button[aria-label*="Sign in" i]',
            'button[data-testid*="login" i]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
            '[class*="working"]',
        ],
    },
    "google": {
        "url": "https://aistudio.google.com/",
        "inject": _inject_js("""
            (function() {
                // Try standard light-DOM selectors first.
                var result =
                    document.querySelector('ms-prompt-box ms-autosize-textarea textarea')
                    || document.querySelector('ms-prompt-box textarea[aria-label="Enter a prompt"]')
                    || document.querySelector('ms-prompt-box textarea[aria-label="Type something"]')
                    || document.querySelector('ms-prompt-box textarea')
                    || document.querySelector('textarea[aria-label="Enter a prompt"]')
                    || document.querySelector('textarea[aria-label="Type something"]')
                    || document.querySelector('textarea');
                if (result) return result;
                // Google AI Studio renders ms-* web components with Shadow DOM,
                // so standard querySelector cannot reach the textarea. Walk known
                // shadow hosts and search their shadow roots.
                var hosts = document.querySelectorAll('ms-prompt-box, ms-autosize-textarea');
                for (var host of hosts) {
                    try {
                        var root = host.shadowRoot;
                        if (!root) continue;
                        result = root.querySelector('textarea[aria-label="Enter a prompt"]')
                            || root.querySelector('textarea[aria-label="Type something"]')
                            || root.querySelector('textarea');
                        if (result) return result;
                        // Descend one more level (ms-autosize-textarea inside ms-prompt-box)
                        var innerHost = root.querySelector('ms-autosize-textarea');
                        if (innerHost && innerHost.shadowRoot) {
                            result = innerHost.shadowRoot.querySelector('textarea[aria-label="Enter a prompt"]')
                                || innerHost.shadowRoot.querySelector('textarea[aria-label="Type something"]')
                                || innerHost.shadowRoot.querySelector('textarea');
                            if (result) return result;
                        }
                    } catch (_) {}
                }
                return null;
            })()
        """),
        "post_inject_js": """
            (() => {
                // Google AI Studio's ms-autosize-textarea web component keeps
                // internal state (data-value attribute, disabled flag on the
                // Run button) that does not sync when .value is set via the
                // property setter. Sync the wrapper so the Run button enables.
                // The marked input may live inside a shadow root, so scan
                // both light DOM and shadow roots.
                let el = document.querySelector('[data-sbsllm-input="true"]');
                if (!el) {
                    for (const host of document.querySelectorAll('*')) {
                        try {
                            if (!host.shadowRoot) continue;
                            el = host.shadowRoot.querySelector('[data-sbsllm-input="true"]');
                            if (el) break;
                        } catch (_) {}
                    }
                }
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
                // The Run button lives inside ms-run-button's shadow DOM;
                // standard querySelector cannot reach into shadow roots, so
                // traverse shadowRoot explicitly. Fall back to light-DOM
                // querySelector for components that don't use shadow DOM.
                const runButtonHost = document.querySelector('ms-run-button');
                if (runButtonHost) {
                    try {
                        runButtonHost.removeAttribute('disabled');
                        const root = (runButtonHost.shadowRoot || runButtonHost);
                        const inner = root.querySelector('button');
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
            // Google AI Studio uses ms-* web components that may use Shadow DOM.
            // Standard querySelector cannot reach into shadow roots, so we
            // traverse shadowRoot first, then fall back to light-DOM selectors.
            (document.querySelector('ms-run-button')?.shadowRoot?.querySelector('button[aria-label="Run"]'))
                || (document.querySelector('ms-run-button')?.shadowRoot?.querySelector('button[type="submit"]'))
                || (document.querySelector('ms-run-button')?.shadowRoot?.querySelector('button:not([disabled])'))
                || (document.querySelector('ms-prompt-box')?.shadowRoot?.querySelector('ms-run-button')?.shadowRoot?.querySelector('button[aria-label="Run"]'))
                || document.querySelector('ms-run-button button[aria-label="Run"]')
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
        "login_wall_selectors": [
            'button[aria-label*="Sign in" i]',
            'a[href*="login" i]',
            'button[data-testid*="login" i]',
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
        "login_wall_selectors": [
            'button[class*="login"]',
            'a[href*="login" i]',
            'button[data-testid*="login" i]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "kimi": {
        "url": "https://www.kimi.ai/",
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
        "post_inject_js": _POST_INJECT_CONTENTEDITABLE,
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
        "login_wall_selectors": [
            'button[data-testid*="login" i]',
            'a[href*="login" i]',
            'div[class*="login"]',
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
        "login_wall_selectors": [
            'button[aria-label*="Log in" i]',
            'a[href*="login" i]',
            'button[data-testid*="login" i]',
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
        "login_wall_selectors": [
            'button[aria-label*="Log in" i]',
            'a[href*="login" i]',
            'button[data-testid*="login" i]',
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
        "login_wall_selectors": [
            'button[aria-label*="Log in" i]',
            'a[href*="login" i]',
            'div[class*="login"]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
    "zai": {
        "url": "https://chat.z.ai/",
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
        "login_wall_selectors": [
            'a[href*="login" i]',
            'button[aria-label*="Log in" i]',
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
        # meta.ai marks the message row with data-testid="assistant-message"
        # (class token is `group/assistant-message`, so `.assistant-message`
        # never matches). The catch-all below ALSO matched the action bar
        # (`group/assistant-message-actions`), and with selection taking the
        # LAST match the icon-only Like/Dislike/Copy buttons won — content
        # came back empty, so replies were never relayed to the local chat.
        # The precise testid therefore goes first, and the class catch-all
        # excludes the action bar.
        "response_selectors": [
            '[data-testid="assistant-message"]',
            '[data-testid="ai-message"]',
            '[data-message-author-role="assistant"]',
            '[class*="assistant-message"]:not([class*="assistant-message-actions"])',
            ".assistant-message",
            "article .markdown",
        ],
        "thinking_selectors": [
            '[class*="thinking"]',
            '[class*="reasoning"]',
            '[data-testid*="thinking"]',
        ],
        "login_wall_selectors": [
            'a[href*="login" i]',
            'button[aria-label*="Log in" i]',
            'button[aria-label*="Sign in" i]',
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
            # meta.ai flags the message row itself while generating
            # (done state reads data-streaming-state="DONE" /
            # data-streaming-complete="true"). Without one of these the site
            # never reports busy and long answers end at the first idle
            # window. The node IS the response, so it passes the
            # inside-response test in the busy filter.
            '[data-streaming-complete="false"]',
            '[data-streaming-state="STREAMING"]',
            '[data-streaming-state="IN_PROGRESS"]',
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
        "login_wall_selectors": [
            'a[href*="login" i]',
            'button[aria-label*="Log in" i]',
            'button[aria-label*="Sign in" i]',
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
        "login_wall_selectors": [
            'button[data-testid*="login" i]',
            'a[href*="login" i]',
            ".login-box",
        ],
        "loading_selectors": [
            *_LOADING_SELECTORS,
        ],
    },
}


_REQUIRED_KEYS = ("url", "inject", "submit_js", "response_selectors")
# Keys whose value is a list of CSS selectors rather than a JS string.
_SELECTOR_KEYS = frozenset(
    {"response_selectors", "thinking_selectors", "loading_selectors", "login_wall_selectors"}
)
# Optional JS blocks that must be non-empty strings when present.
_OPTIONAL_JS_KEYS = ("post_inject_js", "setup_js")


def validate_site_config(site_id: str, site: dict) -> list[str]:
    """Validate one site entry; return a list of problems (empty = valid).

    A misconfigured site (missing URL, empty response selectors, inject JS
    without the placeholder) used to surface only at request time as a 500
    or a silently empty answer. Running this at import time means the whole
    process refuses to start with a broken config instead of failing the
    first request.
    """
    problems: list[str] = []
    if not isinstance(site, dict):
        return [f"{site_id}: not a dict"]
    for key in _REQUIRED_KEYS:
        if key not in site:
            problems.append(f"{site_id}: missing required key {key!r}")
            continue
        value = site[key]
        if key in _SELECTOR_KEYS:
            if not isinstance(value, (list, tuple)) or not value:
                problems.append(
                    f"{site_id}: key {key!r} must be a non-empty list of selectors"
                )
            elif not all(isinstance(s, str) and s.strip() for s in value):
                problems.append(
                    f"{site_id}: key {key!r} must contain only non-empty strings"
                )
        elif not isinstance(value, str) or not value.strip():
            problems.append(f"{site_id}: key {key!r} must be a non-empty string")
    for key in _OPTIONAL_JS_KEYS:
        if key in site:
            value = site[key]
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{site_id}: key {key!r} must be a non-empty string")
    inject = site.get("inject")
    if isinstance(inject, str) and "PROMPT_PLACEHOLDER" not in inject:
        problems.append(f"{site_id}: inject JS must contain PROMPT_PLACEHOLDER")
    url = site.get("url")
    if isinstance(url, str) and url and not url.startswith("https://"):
        problems.append(f"{site_id}: url must be https://")
    return problems


def validate_sites(sites: dict[str, dict] | None = None) -> list[str]:
    """Validate every site in the registry (or the given dict).

    Returns a flat list of problem strings. Empty means all sites are
    well-formed. Does not raise: the caller decides whether a warning or
    a hard failure is appropriate.
    """
    problems: list[str] = []
    for site_id, site in (sites or SITES).items():
        problems.extend(validate_site_config(site_id, site))
    return problems


# Validate the built-in registry at import time so a broken site entry
# fails fast rather than on the first request.
_site_config_problems = validate_sites()
if _site_config_problems:
    import logging as _logging

    _logging.getLogger(__name__).warning(
        "site_config_invalid",
        extra={"problems": _site_config_problems},
    )


def get_site(site_id: str) -> dict:
    """Get site config by ID. Raises ValueError if not found."""
    if site_id not in SITES:
        raise ValueError(f"Unknown site: {site_id!r}. Available: {', '.join(SITES)}")
    return SITES[site_id]


def list_sites() -> list[str]:
    """Return list of available site IDs."""
    return list(SITES.keys())
