"""Prompt escaping and JS template filling."""

from __future__ import annotations

import json

from .sites import _POPUP_SELECTORS, _response_js, get_site

# Template placeholder used in injection JS
PLACEHOLDER = "PROMPT_PLACEHOLDER"


def escape_prompt(prompt: str) -> str:
    """Escape a prompt for safe embedding in a JS template literal.

    Escapes backslashes, backticks, and ${...} interpolation sequences.
    """
    # Order matters: escape backslashes first
    escaped = prompt.replace("\\", "\\\\")
    escaped = escaped.replace("`", "\\`")
    escaped = escaped.replace("${", "\\${")
    return escaped


def inject_prompt(site_id: str, prompt: str) -> str:
    """Get the injection JS for a site with the prompt filled in.

    If the site defines a ``post_inject_js`` block, it is appended after the
    main inject IIFE. This lets sites that use composite editors
    (TipTap/ProseMirror, Google ``ms-autosize-textarea``) sync internal state
    that a plain ``value``/``textContent`` assignment does not reach.
    """
    site = get_site(site_id)
    # site["inject"] is already a fully-resolved IIFE (selectors filled in by
    # _inject_js at config-build time). Only the prompt value still needs
    # substitution — do NOT re-wrap with _inject_js or you get a nested IIFE
    # that returns NO_INPUT.
    js = site["inject"]
    escaped = escape_prompt(prompt)
    result = js.replace(PLACEHOLDER, escaped)
    post = site.get("post_inject_js")
    if post:
        # Add a semicolon so the post-inject IIFE is a separate statement
        # rather than being parsed as calling the return value of the main IIFE.
        result = result + ";\n" + post
    return result


def submit_js(site_id: str) -> str:
    """Get the submit JS for a site."""
    site = get_site(site_id)
    return site["submit_js"]


def extract_js(site_id: str) -> str | None:
    """Get the response extraction JS for a site, if supported."""
    site = get_site(site_id)
    if "response_selectors" not in site:
        return None
    return _response_js(
        site.get("response_selectors"),
        site.get("thinking_selectors"),
        site.get("loading_selectors"),
        site.get("login_wall_selectors"),
        site.get("response_container"),
        site.get("response_exclude_selectors"),
        site.get("login_wall_modal_selectors"),
        site.get("popup_selectors"),
    )


# Standalone popup check, run BEFORE inject+submit so a request that
# arrives while a blocking modal is up fails fast with a clear error
# instead of injecting into a page the user cannot see and hanging
# until the timeout. Mirrors the visibility gate of the extraction
# template: hidden-but-blocking dialogs stay in the site's
# login_wall_modal_selectors, which the extraction reports as a
# login wall after the submit attempt.
_POPUP_CHECK_TEMPLATE = """
(() => {
    const selectors = __POPUP_SELECTORS__;
    const isVisible = (element) => {
        if (!element) return false;
        const style = window.getComputedStyle(element);
        if (!style) return true;
        if (style.display === 'none') return false;
        if (style.visibility === 'hidden' || style.visibility === 'collapse') return false;
        if (style.opacity !== '' && Number(style.opacity) < 0.05) return false;
        return element.getClientRects().length > 0;
    };
    const nodes = [];
    for (const selector of selectors) {
        try {
            for (const element of document.querySelectorAll(selector)) {
                if (isVisible(element) && !nodes.includes(element)) {
                    nodes.push(element);
                }
            }
        } catch (_) {}
    }
    const squash = (text) => (text || '').replace(/\\s+/g, ' ').trim();
    return {
        popup: nodes.length > 0,
        popup_text: nodes
            .map((node) => squash(node.innerText || node.textContent || ''))
            .filter(Boolean)
            .slice(0, 3)
            .join(' | ')
    };
})()
"""


def popup_check_js(site_id: str) -> str:
    """Build the standalone pre-submit popup check for a site.

    Uses the site's ``popup_selectors`` when it defines them and the
    shared default set otherwise; ``[]`` disables the check.
    """
    site = get_site(site_id)
    selectors = site.get("popup_selectors")
    if selectors is None:
        selectors = _POPUP_SELECTORS
    return _POPUP_CHECK_TEMPLATE.replace(
        "__POPUP_SELECTORS__", json.dumps(list(selectors))
    )
