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
