"""Tests for sites.py."""

import pytest

from sbsllm.sites import SITES, _response_js, get_site, list_sites


class TestListSites:
    def test_returns_list(self):
        result = list_sites()
        assert isinstance(result, list)

    def test_contains_all_expected_sites(self):
        sites = list_sites()
        expected = [
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
        ]
        for site in expected:
            assert site in sites, f"Missing site: {site}"

    def test_returns_strings_only(self):
        for site in list_sites():
            assert isinstance(site, str)


class TestGetSite:
    def test_returns_dict(self):
        site = get_site("chatgpt")
        assert isinstance(site, dict)

    def test_has_url_key(self):
        site = get_site("chatgpt")
        assert "url" in site
        assert site["url"].startswith("https://")

    def test_has_inject_key(self):
        site = get_site("chatgpt")
        assert "inject" in site
        assert isinstance(site["inject"], str)

    def test_has_submit_js_key(self):
        site = get_site("chatgpt")
        assert "submit_js" in site
        assert isinstance(site["submit_js"], str)

    def test_inject_contains_placeholder(self):
        site = get_site("chatgpt")
        assert "PROMPT_PLACEHOLDER" in site["inject"]

    def test_unknown_site_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown site"):
            get_site("nonexistent")

    def test_error_message_lists_available_sites(self):
        with pytest.raises(ValueError) as exc_info:
            get_site("nonexistent")
        assert "chatgpt" in str(exc_info.value)


class TestSiteStructure:
    """Verify all sites have consistent structure."""

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_site_has_url(self, site_id):
        assert "url" in SITES[site_id]
        assert SITES[site_id]["url"].startswith("https://")

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_site_has_inject(self, site_id):
        assert "inject" in SITES[site_id]
        assert "PROMPT_PLACEHOLDER" in SITES[site_id]["inject"]

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_site_has_submit_js(self, site_id):
        assert "submit_js" in SITES[site_id]

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_inject_is_valid_js_iife(self, site_id):
        inject = SITES[site_id]["inject"]
        # Should be an IIFE
        assert inject.strip().startswith("(() =>") or inject.strip().startswith(
            "(function"
        )

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_submit_is_valid_js_iife(self, site_id):
        submit = SITES[site_id]["submit_js"]
        assert submit.strip().startswith("(() =>") or submit.strip().startswith(
            "(function"
        )


class TestResponseSelectors:
    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_site_has_response_selectors(self, site_id):
        assert SITES[site_id]["response_selectors"]

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_loading_selectors_are_precise(self, site_id):
        """Bare `[class*="loading"]` matches decorative skeletons that linger
        after generation ends, which pinned `done` to false and left the
        local chat streaming until the request deadline."""
        for selector in SITES[site_id]["loading_selectors"]:
            assert '[class*="loading"]' not in selector
            assert '[class*="typing"]' not in selector

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_loading_selectors_include_shared_set(self, site_id):
        assert 'button[aria-label*="Stop" i]' in SITES[site_id]["loading_selectors"]

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_thinking_selectors_present(self, site_id):
        assert SITES[site_id]["thinking_selectors"]

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_loading_selector_lists_are_not_shared(self, site_id):
        """Each site gets its own list so a site-specific tweak cannot leak."""
        other = next(s for s in SITES if s != site_id)
        a = SITES[site_id]["loading_selectors"]
        a.append("sentinel")
        try:
            assert "sentinel" not in SITES[other]["loading_selectors"]
        finally:
            a.remove("sentinel")


class TestResponseJs:
    def test_embeds_all_selector_groups(self):
        js = _response_js(["#a"], [".t"], ["#l"])
        assert '["#a"]' in js
        assert '[".t"]' in js
        assert '["#l"]' in js

    def test_filters_collapsed_thinking_labels(self):
        js = _response_js(["#a"], [".t"], [])
        assert "LABEL_ONLY" in js
        assert "Thought Process" in js

    def test_prunes_thinking_from_answer(self):
        js = _response_js(["#a"], [".t"], [])
        assert "cloneNode" in js
        assert "removeChild" in js

    def test_ignores_invisible_loading_nodes(self):
        js = _response_js(["#a"], [".t"], ["#l"])
        assert "opacity" in js
        assert "aria-busy" in js

    def test_handles_empty_selector_groups(self):
        js = _response_js(None, None, None)
        assert "const responseSelectors = [];" in js

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_generated_js_is_a_valid_iife(self, site_id):
        js = _response_js(
            SITES[site_id]["response_selectors"],
            SITES[site_id]["thinking_selectors"],
            SITES[site_id]["loading_selectors"],
        )
        assert js.strip().startswith("(() =>")
        assert "__RESPONSE_SELECTORS__" not in js
        assert "__THINKING_SELECTORS__" not in js
        assert "__LOADING_SELECTORS__" not in js
