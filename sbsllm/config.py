"""Config loading and validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .server import DEFAULT_RESPONSE_IDLE_TIMEOUT
from .sites import list_sites

MAX_LOGIN_WAIT = 24 * 60 * 60  # 24 hours
MAX_BROWSER_TIMEOUT = 1800  # 30 minutes
MAX_BROWSER_LOCK_TIMEOUT = 1800  # 30 minutes
MAX_RESPONSE_IDLE_TIMEOUT = 120


class Config(BaseModel):
    """Configuration for sbsllm."""

    model_config = ConfigDict(extra="forbid")

    chats: list[str] = Field(
        ..., min_length=1, description="List of chat sites to open"
    )
    login_wait: int = Field(
        default=30, ge=0, le=MAX_LOGIN_WAIT, description="Seconds to wait for login"
    )
    chrome_bin: str | None = Field(
        default=None, description="Path to Chrome/Chromium binary"
    )
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_file: str | None = Field(
        default=None, description="Path to log file (optional)"
    )
    browser_timeout: int = Field(
        default=180,
        ge=1,
        le=MAX_BROWSER_TIMEOUT,
        description=(
            "Hard cap on one answer (seconds). A chat with thinking enabled can "
            "run for minutes, and cutting here truncates the reply."
        ),
    )
    browser_lock_timeout: int = Field(
        default=300,
        ge=1,
        le=MAX_BROWSER_LOCK_TIMEOUT,
        description=(
            "How long a request waits for its chat tab to be free (seconds). "
            "Must exceed a full answer, or follow-up requests get a 503."
        ),
    )
    response_idle_timeout: float = Field(
        default=DEFAULT_RESPONSE_IDLE_TIMEOUT,
        ge=0.1,
        le=MAX_RESPONSE_IDLE_TIMEOUT,
        description=(
            "Seconds an unchanged answer is treated as finished when the site "
            "gives no 'still generating' signal."
        ),
    )
    json_log_format: bool = Field(
        default=False, description="Output logs in JSON format"
    )

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
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Config error: invalid YAML in {path}: {e}", file=sys.stderr)
            sys.exit(1)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        print(f"Config error: {path} must contain a YAML mapping", file=sys.stderr)
        sys.exit(1)

    return parse_config(data, path)


def parse_config(data: dict[str, Any], source: Path | None = None) -> Config:
    """Validate raw config dict and return Config."""
    if not isinstance(data, dict):
        print("Config error: expected a YAML mapping", file=sys.stderr)
        sys.exit(1)
    try:
        return Config(**data)
    except (ValidationError, TypeError) as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)
