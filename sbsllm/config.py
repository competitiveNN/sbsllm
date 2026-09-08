"""Config loading and validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from .sites import list_sites


class Config(BaseModel):
    """Configuration for sbsllm."""

    chats: list[str] = Field(..., min_length=1, description="List of chat sites to open")
    login_wait: int = Field(default=30, ge=0, description="Seconds to wait for login")
    qutebrowser_bin: str | None = Field(default=None, description="Path to qutebrowser binary")
    log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)")
    log_file: str | None = Field(default=None, description="Path to log file (optional)")
    retry_count: int = Field(default=3, ge=0, description="Number of retries for failed operations")
    retry_delay: float = Field(default=1.0, ge=0, description="Delay between retries in seconds")

    @field_validator("chats")
    @classmethod
    def validate_chats(cls, v: list[str]) -> list[str]:
        """Validate that all chat sites exist."""
        available = set(list_sites())
        invalid = [c for c in v if c not in available]
        if invalid:
            raise ValueError(
                f"Unknown chats: {invalid}. Available: {sorted(available)}"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of: {valid_levels}")
        return upper

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


def parse_config(data: dict[str, Any], source: Path | None = None) -> Config:
    """Validate raw config dict and return Config."""
    try:
        return Config(**data)
    except Exception as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
