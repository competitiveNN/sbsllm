"""Tests for config.py."""

import pytest
import yaml

from sbsllm.config import Config, load_config, parse_config


class TestConfigModel:
    def test_creation(self):
        config = Config(chats=["chatgpt"], login_wait=30, chrome_bin=None)
        assert config.chats == ["chatgpt"]
        assert config.login_wait == 30
        assert config.chrome_bin is None

    def test_chrome_bin_default(self):
        config = Config(chats=["chatgpt"], login_wait=30, chrome_bin=None)
        assert config.chrome_bin is None

    def test_chrome_bin_custom(self):
        config = Config(
            chats=["chatgpt"], login_wait=30, chrome_bin="/usr/bin/google-chrome"
        )
        assert config.chrome_bin == "/usr/bin/google-chrome"

    def test_default_log_level(self):
        config = Config(chats=["chatgpt"])
        assert config.log_level == "INFO"

    def test_custom_log_level(self):
        config = Config(chats=["chatgpt"], log_level="DEBUG")
        assert config.log_level == "DEBUG"

    def test_log_level_case_insensitive(self):
        config = Config(chats=["chatgpt"], log_level="debug")
        assert config.log_level == "DEBUG"

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            Config(chats=["chatgpt"], log_level="INVALID")

    def test_negative_login_wait_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], login_wait=-1)

    def test_duplicate_prompt_cooldown_default(self):
        from sbsllm.server import DEFAULT_DUPLICATE_PROMPT_COOLDOWN

        config = Config(chats=["chatgpt"])
        assert config.duplicate_prompt_cooldown == DEFAULT_DUPLICATE_PROMPT_COOLDOWN

    def test_duplicate_prompt_cooldown_custom(self):
        config = Config(chats=["chatgpt"], duplicate_prompt_cooldown=15.0)
        assert config.duplicate_prompt_cooldown == 15.0

    def test_duplicate_prompt_cooldown_zero_allowed(self):
        config = Config(chats=["chatgpt"], duplicate_prompt_cooldown=0)
        assert config.duplicate_prompt_cooldown == 0

    def test_duplicate_prompt_cooldown_negative_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], duplicate_prompt_cooldown=-1)

    def test_default_thinking_patience(self):
        """thinking_patience defaults to DEFAULT_THINKING_PATIENCE (120s)."""
        from sbsllm.config import DEFAULT_THINKING_PATIENCE  # noqa: F401
        from sbsllm.server import DEFAULT_THINKING_PATIENCE as DEFAULT

        config = Config(chats=["chatgpt"])
        assert config.thinking_patience == DEFAULT

    def test_custom_thinking_patience(self):
        config = Config(chats=["chatgpt"], thinking_patience=60.0)
        assert config.thinking_patience == 60.0

    def test_thinking_patience_too_low_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], thinking_patience=0.05)

    def test_thinking_patience_too_high_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], thinking_patience=121.0)

    def test_default_json_log_format(self):
        config = Config(chats=["chatgpt"])
        assert config.json_log_format is False

    def test_custom_json_log_format(self):
        config = Config(chats=["chatgpt"], json_log_format=True)
        assert config.json_log_format is True

    def test_default_browser_timeout(self):
        config = Config(chats=["chatgpt"])
        assert config.browser_timeout == 180

    def test_custom_browser_timeout(self):
        config = Config(chats=["chatgpt"], browser_timeout=300)
        assert config.browser_timeout == 300

    def test_browser_timeout_too_low_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], browser_timeout=0)

    def test_browser_timeout_too_high_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], browser_timeout=1801)

    def test_default_browser_lock_timeout(self):
        config = Config(chats=["chatgpt"])
        assert config.browser_lock_timeout == 300

    def test_custom_browser_lock_timeout(self):
        config = Config(chats=["chatgpt"], browser_lock_timeout=600)
        assert config.browser_lock_timeout == 600

    def test_browser_lock_timeout_too_high_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], browser_lock_timeout=1801)

    def test_default_response_idle_timeout(self):
        from sbsllm.server import DEFAULT_RESPONSE_IDLE_TIMEOUT

        config = Config(chats=["chatgpt"])
        assert config.response_idle_timeout == DEFAULT_RESPONSE_IDLE_TIMEOUT

    def test_custom_response_idle_timeout(self):
        config = Config(chats=["chatgpt"], response_idle_timeout=5.0)
        assert config.response_idle_timeout == 5.0

    def test_response_idle_timeout_too_low_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], response_idle_timeout=0.05)

    def test_response_idle_timeout_too_high_raises(self):
        with pytest.raises(ValueError):
            Config(chats=["chatgpt"], response_idle_timeout=121.0)

    def test_extra_field_rejected(self):
        """Unknown config keys must be rejected (extra='forbid')."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Config(chats=["chatgpt"], unknown_field=1)


class TestParseConfig:
    def test_valid_config(self):
        data = {"chats": ["chatgpt", "claude"], "login_wait": 60}
        config = parse_config(data)
        assert config.chats == ["chatgpt", "claude"]
        assert config.login_wait == 60
        assert config.chrome_bin is None

    def test_default_login_wait(self):
        data = {"chats": ["chatgpt"]}
        config = parse_config(data)
        assert config.login_wait == 30

    def test_custom_chrome_bin(self):
        data = {"chats": ["chatgpt"], "chrome_bin": "/custom/path"}
        config = parse_config(data)
        assert config.chrome_bin == "/custom/path"

    def test_empty_chats_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({"chats": []})
        captured = capsys.readouterr()
        assert "Config error" in captured.err

    def test_missing_chats_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({})
        captured = capsys.readouterr()
        assert "Config error" in captured.err

    def test_invalid_chat_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({"chats": ["nonexistent"]})
        captured = capsys.readouterr()
        assert "Config error" in captured.err

    def test_partial_invalid_chat_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({"chats": ["chatgpt", "nonexistent"]})
        captured = capsys.readouterr()
        assert "Config error" in captured.err

    def test_multiple_chats(self):
        data = {"chats": ["chatgpt", "claude", "deepseek"]}
        config = parse_config(data)
        assert len(config.chats) == 3

    def test_login_wait_as_string_converted_to_int(self):
        data = {"chats": ["chatgpt"], "login_wait": "45"}
        config = parse_config(data)
        assert config.login_wait == 45
        assert isinstance(config.login_wait, int)

    def test_all_new_sites_valid(self):
        """Verify all newly added sites are accepted."""
        data = {
            "chats": [
                "perplexity",
                "poe",
                "cohere",
                "zai",
                "meta",
                "huggingface",
                "tencent",
            ]
        }
        config = parse_config(data)
        assert config.chats == data["chats"]

    def test_log_level_config(self):
        data = {"chats": ["chatgpt"], "log_level": "DEBUG"}
        config = parse_config(data)
        assert config.log_level == "DEBUG"

    def test_log_file_config(self):
        data = {"chats": ["chatgpt"], "log_file": "/tmp/sbsllm.log"}
        config = parse_config(data)
        assert config.log_file == "/tmp/sbsllm.log"


class TestLoadConfig:
    def test_load_from_explicit_path(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump({"chats": ["chatgpt"]}))
        config = load_config(config_file)
        assert config.chats == ["chatgpt"]

    def test_load_nonexistent_path_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_from_cwd(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.safe_dump({"chats": ["claude"]}))
        monkeypatch.chdir(tmp_path)
        config = load_config()
        assert config.chats == ["claude"]

    def test_load_from_home_config(self, tmp_path, monkeypatch):
        config_dir = tmp_path / ".config" / "sbsllm"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml.safe_dump({"chats": ["deepseek"]}))
        monkeypatch.setenv("HOME", str(tmp_path))
        # Also ensure cwd has no config
        other_dir = tmp_path / "other"
        other_dir.mkdir(exist_ok=True)
        monkeypatch.chdir(other_dir)
        config = load_config()
        assert config.chats == ["deepseek"]

    def test_no_config_found_exits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit):
            load_config()

    def test_empty_yaml_file(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        with pytest.raises(SystemExit):
            load_config(config_file)

    def test_yaml_with_comments_only(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("# just a comment\n")
        with pytest.raises(SystemExit):
            load_config(config_file)

    @pytest.mark.parametrize("content", ["- chatgpt\n", "chatgpt\n", "42\n"])
    def test_yaml_root_must_be_mapping(self, tmp_path, content):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(content)
        with pytest.raises(SystemExit):
            load_config(config_file)

    def test_parse_config_rejects_non_mapping(self, capsys):
        """parse_config must exit when the YAML root is not a mapping."""
        with pytest.raises(SystemExit):
            parse_config(["not", "a", "dict"])  # type: ignore[arg-type]
        captured = capsys.readouterr()
        assert "expected a YAML mapping" in captured.err

    def test_malformed_yaml_exits(self, tmp_path, capsys):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("chats: [unterminated\n")
        with pytest.raises(SystemExit):
            load_config(config_file)
        assert "invalid YAML" in capsys.readouterr().err
