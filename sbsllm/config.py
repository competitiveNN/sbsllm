"""Config loading and validation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from .sites import list_sites


@dataclass
class Config:
    chats: list[str]
    login_wait: int
    qutebrowser_bin: str | None

    @property
    def qb_bin(self) -> str:
        return self.qutebrowser_bin or "qutebrowser"


def load_config(path: Path | None = None) -> Config:
    """Load config from YAML file.

    Order of precedence:
    1. Explicit path passed as argument
    2. ./config.yaml (current directory)
    3. ~/.config/sbsllm/config.yaml
    """
    if path is not None:
        if not path.exists():
            print(f"Config file not found: {path}", file=sys.stderr)
            sys.exit(1)
    else:
        candidates = [
            Path.cwd() / "config.yaml",
            Path.home() / ".config" / "sbsllm" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = candidate
                break
        else:
            print(
                "No config file found. Create config.yaml or ~/.config/sbsllm/config.yaml",
                file=sys.stderr,
            )
            sys.exit(1)

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    return parse_config(data, path)


def parse_config(data: dict, source: Path | None = None) -> Config:
    """Validate raw config dict and return Config dataclass."""
    available = list_sites()

    raw_chats = data.get("chats")
    if not raw_chats:
        chats_str = ", ".join(available)
        print(
            f"Config error: 'chats' is empty or missing. Available: {chats_str}",
            file=sys.stderr,
        )
        sys.exit(1)

    invalid = [c for c in raw_chats if c not in available]
    if invalid:
        chats_str = ", ".join(available)
        print(
            f"Config error: unknown chats: {invalid}. Available: {chats_str}",
            file=sys.stderr,
        )
        sys.exit(1)

    login_wait = data.get("login_wait", 30)
    qutebrowser_bin = data.get("qutebrowser_bin") or None

    return Config(
        chats=raw_chats,
        login_wait=int(login_wait),
        qutebrowser_bin=qutebrowser_bin,
    )
