"""Tests for config.py."""

import tempfile
from pathlib import Path

import pytest
import yaml

from sbsllm.config import Config, load_config, parse_config


class TestConfigDataclass:
    def test_creation(self):
        config = Config(chats=["chatgpt"], login_wait=30, qutebrowser_bin=None)
        assert config.chats == ["chatgpt"]
        assert config.login_wait == 30
        assert config.qutebrowser_bin is None

    def test_qb_bin_default(self):
        config = Config(chats=["chatgpt"], login_wait=30, qutebrowser_bin=None)
        assert config.qb_bin == "qutebrowser"

    def test_qb_bin_custom(self):
        config = Config(chats=["chatgpt"], login_wait=30, qutebrowser_bin="/usr/bin/qutebrowser")
        assert config.qb_bin == "/usr/bin/qutebrowser"


class TestParseConfig:
    def test_valid_config(self):
        data = {"chats": ["chatgpt", "claude"], "login_wait": 60}
        config = parse_config(data)
        assert config.chats == ["chatgpt", "claude"]
        assert config.login_wait == 60
        assert config.qutebrowser_bin is None

    def test_default_login_wait(self):
        data = {"chats": ["chatgpt"]}
        config = parse_config(data)
        assert config.login_wait == 30

    def test_custom_qutebrowser_bin(self):
        data = {"chats": ["chatgpt"], "qutebrowser_bin": "/custom/path"}
        config = parse_config(data)
        assert config.qutebrowser_bin == "/custom/path"

    def test_empty_chats_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({"chats": []})
        captured = capsys.readouterr()
        assert "empty or missing" in captured.err

    def test_missing_chats_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({})
        captured = capsys.readouterr()
        assert "empty or missing" in captured.err

    def test_invalid_chat_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({"chats": ["nonexistent"]})
        captured = capsys.readouterr()
        assert "unknown chats" in captured.err

    def test_partial_invalid_chat_exits(self, capsys):
        with pytest.raises(SystemExit):
            parse_config({"chats": ["chatgpt", "nonexistent"]})
        captured = capsys.readouterr()
        assert "unknown chats" in captured.err

    def test_multiple_chats(self):
        data = {"chats": ["chatgpt", "claude", "deepseek"]}
        config = parse_config(data)
        assert len(config.chats) == 3

    def test_login_wait_as_string_converted_to_int(self):
        data = {"chats": ["chatgpt"], "login_wait": "45"}
        config = parse_config(data)
        assert config.login_wait == 45
        assert isinstance(config.login_wait, int)


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
