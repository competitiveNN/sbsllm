"""Tests for sites.py."""

import pytest

from sbsllm.sites import SITES, get_site, list_sites


class TestListSites:
    def test_returns_list(self):
        result = list_sites()
        assert isinstance(result, list)

    def test_contains_all_expected_sites(self):
        sites = list_sites()
        expected = ["chatgpt", "claude", "deepseek", "qwen", "grok", "google", "mistral", "kimi", "perplexity", "poe", "cohere"]
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
        assert inject.strip().startswith("(() =>") or inject.strip().startswith("(function")

    @pytest.mark.parametrize("site_id", list(SITES.keys()))
    def test_submit_is_valid_js_iife(self, site_id):
        submit = SITES[site_id]["submit_js"]
        assert submit.strip().startswith("(() =>") or submit.strip().startswith("(function")
