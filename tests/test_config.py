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
        data = {"chats": ["perplexity", "poe", "cohere"]}
        config = parse_config(data)
        assert config.chats == ["perplexity", "poe", "cohere"]

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
