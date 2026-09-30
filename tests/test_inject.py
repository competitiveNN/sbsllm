"""Tests for inject.py."""

import pytest

from sbsllm.inject import escape_prompt, inject_prompt, submit_js


class TestEscapePrompt:
    def test_empty_string(self):
        assert escape_prompt("") == ""

    def test_simple_string(self):
        assert escape_prompt("hello world") == "hello world"

    def test_escape_backslash(self):
        assert escape_prompt("\\") == "\\\\"

    def test_escape_backtick(self):
        assert escape_prompt("`") == "\\`"

    def test_escape_dollar_brace(self):
        assert escape_prompt("${}") == "\\${}"

    def test_escape_all_special_chars(self):
        result = escape_prompt("path\\to`file${var}")
        assert result == "path\\\\to\\`file\\${var}"

    def test_multiline_prompt(self):
        result = escape_prompt("line1\nline2")
        assert result == "line1\nline2"

    def test_unicode_preserved(self):
        assert escape_prompt("héllo 世界") == "héllo 世界"

    def test_backslash_before_backtick(self):
        # Backslash should be escaped first, then backtick
        result = escape_prompt("\\`")
        assert result == "\\\\\\`"


class TestInjectPrompt:
    def test_returns_string(self):
        result = inject_prompt("chatgpt", "hello")
        assert isinstance(result, str)

    def test_contains_prompt(self):
        result = inject_prompt("chatgpt", "test prompt")
        assert "test prompt" in result

    def test_placeholder_replaced(self):
        result = inject_prompt("chatgpt", "hello")
        assert "PROMPT_PLACEHOLDER" not in result

    def test_escapes_prompt_injection(self):
        # Prompt with backtick should be escaped
        result = inject_prompt("chatgpt", "`evil`")
        assert "`evil`" not in result
        assert "\\`evil\\`" in result

    def test_escapes_dollar_brace(self):
        result = inject_prompt("chatgpt", "${alert(1)}")
        # The ${ should be escaped to \${, so the raw string should not contain
        # the exact sequence ${alert(1)} - but it will contain \${alert(1)}
        # The key check is that ${ is escaped
        assert "\\${alert(1)}" in result

    def test_works_for_all_sites(self):
        from sbsllm.sites import list_sites

        for site_id in list_sites():
            result = inject_prompt(site_id, "test")
            assert "PROMPT_PLACEHOLDER" not in result
            assert "test" in result

    def test_unknown_site_raises_error(self):
        with pytest.raises(ValueError):
            inject_prompt("nonexistent", "hello")

    def test_multiline_prompt(self):
        prompt = "line1\nline2\nline3"
        result = inject_prompt("chatgpt", prompt)
        assert "line1\nline2\nline3" in result

    def test_empty_prompt(self):
        result = inject_prompt("chatgpt", "")
        assert "PROMPT_PLACEHOLDER" not in result

    def test_post_inject_js_appended_for_grok(self):
        """Sites with post_inject_js should have it appended."""
        result = inject_prompt("grok", "test")
        assert "post_inject_js" not in result  # not the literal key
        # The post_inject block dispatches an input event; verify it's present.
        assert "data-sbsllm-input" in result

    def test_post_inject_js_appended_for_google(self):
        result = inject_prompt("google", "test")
        assert "data-value" in result  # google post_inject syncs data-value

    def test_post_inject_js_appended_for_zai(self):
        result = inject_prompt("zai", "test")
        assert "NO_MARKED_INPUT" in result  # zai post_inject checks for marked input

    def test_post_inject_separated_by_semicolon(self):
        """post_inject_js must be separated from the main IIFE by a semicolon."""
        result = inject_prompt("grok", "test")
        # The main IIFE ends with })() and the post_inject starts with (() =>
        # Without a semicolon separator, the two IIFEs are parsed as a call chain.
        assert "});\n" in result or "});\n\n" in result or "})();\n" in result
        assert "(() =>" in result  # post_inject starts a new IIFE

    def test_grok_post_inject_uses_execcommand_for_tipTap(self):
        """Grok's post_inject_js must use execCommand for TipTap/ProseMirror editors."""
        result = inject_prompt("grok", "test")
        # TipTap/ProseMirror editors don't accept textContent assignment;
        # execCommand('insertText') is needed to update the editor state.
        assert "execCommand('insertText'" in result
        assert "innerText" in result
        assert "beforeinput" in result

    def test_grok_inject_stores_value_for_post_inject(self):
        """Grok's inject template stores the prompt value in a data attribute."""
        result = inject_prompt("grok", "my prompt")
        assert "sbsllmValue" in result

    def test_zai_setup_js_has_error_handling(self):
        """Zai's setup_js must be wrapped in try/catch to avoid silent failures."""
        from sbsllm.sites import SITES

        setup_js = SITES["zai"].get("setup_js", "")
        assert "try" in setup_js
        assert "catch" in setup_js

    def test_zai_url_uses_auth_path(self):
        """Zai URL should point to the auth page for login flow."""
        from sbsllm.sites import SITES

        assert SITES["zai"]["url"] == "https://chat.z.ai/auth"

    def test_google_post_inject_syncs_data_value(self):
        """Google's post_inject_js syncs data-value on ms-autosize-textarea."""
        result = inject_prompt("google", "hello google")
        assert "data-value" in result
        assert "ms-autosize-textarea" in result

    def test_google_post_inject_dispatches_input_event(self):
        """Google's post_inject_js dispatches an InputEvent for web component listeners."""
        result = inject_prompt("google", "test")
        assert "InputEvent" in result
        assert "inputType" in result

    def test_google_post_inject_enables_run_button(self):
        """Google's post_inject_js enables the ms-run-button for submit."""
        result = inject_prompt("google", "test")
        assert "ms-run-button" in result
        assert "removeAttribute" in result

    def test_google_submit_targets_ms_run_button(self):
        """Google's submit_js targets ms-run-button as primary candidate."""
        result = submit_js("google")
        assert "ms-run-button" in result

    def test_kimi_url_uses_kimi_ai(self):
        """Kimi URL should be kimi.ai, not kimi.com."""
        from sbsllm.sites import SITES

        assert SITES["kimi"]["url"] == "https://kimi.ai/"
        assert "kimi.com" not in SITES["kimi"]["url"]

    @pytest.mark.parametrize(
        "site_id",
        [
            "chatgpt",
            "claude",
            "deepseek",
            "qwen",
            "grok",
            "google",
            "mistral",
            "kimi",
            "perplexity",
            "poe",
            "cohere",
            "zai",
            "meta",
            "huggingface",
            "tencent",
        ],
    )
    def test_all_sites_have_post_inject_js(self, site_id):
        """Every site must define post_inject_js for robust editor sync."""
        from sbsllm.sites import SITES

        assert "post_inject_js" in SITES[site_id]

    @pytest.mark.parametrize(
        "site_id",
        [
            "claude",
            "deepseek",
            "mistral",
            "kimi",
            "perplexity",
            "poe",
            "cohere",
            "meta",
            "huggingface",
            "tencent",
            "chatgpt",
            "qwen",
        ],
    )
    def test_contenteditable_sites_use_execcommand(self, site_id):
        """Sites with contenteditable inputs must use execCommand for TipTap."""
        from sbsllm.sites import SITES

        post_inject = SITES[site_id].get("post_inject_js", "")
        assert "execCommand('insertText'" in post_inject
        assert "beforeinput" in post_inject
        assert "innerText" in post_inject

    def test_zai_post_inject_no_enter_dispatch(self):
        """Zai post_inject must NOT dispatch Enter key (prevents double-send)."""
        from sbsllm.sites import SITES

        post_inject = SITES["zai"].get("post_inject_js", "")
        assert "keydown" not in post_inject
        assert "KeyboardEvent" not in post_inject

    def test_meta_setup_js_dismisses_welcome(self):
        """Meta setup_js should handle welcome screen dismissal."""
        from sbsllm.sites import SITES

        setup_js = SITES["meta"].get("setup_js", "")
        assert "welcome" in setup_js.lower()
        assert "dialog" in setup_js.lower()

    def test_all_setup_js_have_error_handling(self):
        """All setup_js blocks must be wrapped in try/catch."""
        from sbsllm.sites import SITES

        for site_id, cfg in SITES.items():
            if "setup_js" in cfg:
                assert "try" in cfg["setup_js"], f"{site_id} setup_js missing try/catch"
                assert "catch" in cfg["setup_js"], (
                    f"{site_id} setup_js missing try/catch"
                )

    def test_post_inject_all_sites_contain_marked_input_selector(self):
        """All post_inject_js must reference the marked input element."""
        from sbsllm.sites import SITES

        for site_id, cfg in SITES.items():
            post_inject = cfg.get("post_inject_js", "")
            assert "data-sbsllm-input" in post_inject, (
                f"{site_id} post_inject missing marker selector"
            )

    @pytest.mark.parametrize(
        "site_id", list(__import__("sbsllm.sites", fromlist=["SITES"]).SITES.keys())
    )
    def test_all_sites_have_login_wall_selectors(self, site_id):
        """Every site must define login_wall_selectors for auth detection."""
        from sbsllm.sites import SITES

        assert "login_wall_selectors" in SITES[site_id]
        assert isinstance(SITES[site_id]["login_wall_selectors"], list)
        assert len(SITES[site_id]["login_wall_selectors"]) > 0

    def test_login_wall_selectors_embedded_in_extract_js(self):
        """extract_js must include loginWallSelectors in the output."""
        from sbsllm.inject import extract_js

        result = extract_js("chatgpt")
        assert "loginWallSelectors" in result

    def test_login_wall_selectors_empty_when_not_defined(self):
        """Sites without login_wall_selectors get an empty array."""
        import sbsllm.inject as inject_module
        from sbsllm.inject import extract_js

        original = inject_module.get_site
        try:
            inject_module.get_site = lambda sid: {
                "url": "https://example.com/",
                "response_selectors": ["div.reply"],
                "thinking_selectors": [],
                "loading_selectors": [],
            }
            result = extract_js("fake")
            assert "loginWallSelectors" in result
            assert "[]" in result
        finally:
            inject_module.get_site = original


class TestSubmitJs:
    def test_returns_string(self):
        result = submit_js("chatgpt")
        assert isinstance(result, str)

    def test_no_placeholder(self):
        result = submit_js("chatgpt")
        assert "PROMPT_PLACEHOLDER" not in result

    def test_works_for_all_sites(self):
        from sbsllm.sites import list_sites

        for site_id in list_sites():
            result = submit_js(site_id)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_unknown_site_raises_error(self):
        with pytest.raises(ValueError):
            submit_js("nonexistent")


class TestExtractJs:
    def test_returns_string_for_supported_site(self):
        from sbsllm.inject import extract_js

        result = extract_js("chatgpt")
        assert isinstance(result, str)
        assert "responseSelectors" in result

    def test_returns_none_for_site_without_extraction(self):
        """A site with no response_selectors must return None.

        Every site in the registry currently defines response_selectors, so we
        synthesize a fake site entry to exercise the branch.
        """
        # Monkeypatch get_site to return a site dict without response_selectors.
        import sbsllm.inject as inject_module
        from sbsllm.inject import extract_js

        original = inject_module.get_site
        try:
            inject_module.get_site = lambda site_id: {"url": "https://example.com/"}
            assert extract_js("fake") is None
        finally:
            inject_module.get_site = original
