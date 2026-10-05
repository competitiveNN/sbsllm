"""Prompt escaping and JS template filling."""

from __future__ import annotations

from .sites import _response_js, get_site

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
    )
