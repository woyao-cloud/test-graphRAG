"""Environment variable management utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_dotenv(env_path: Optional[Path] = None) -> dict[str, str]:
    """Load environment variables from a .env file.

    Simple implementation that does NOT override existing env vars.
    Returns a dict of loaded variables.
    """
    if env_path is None:
        env_path = Path(".env")

    if not env_path.exists():
        return {}

    loaded: dict[str, str] = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Skip if no equals sign
            if "=" not in line:
                continue
            # Parse key=value
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # Only set if not already in environment
            if key not in os.environ:
                os.environ[key] = value
                loaded[key] = value

    return loaded


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable with a default."""
    return os.environ.get(key, default)


def require_env(key: str) -> str:
    """Get a required environment variable, raising if missing."""
    value = os.environ.get(key)
    if value is None:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Set it in .env or export it in your shell."
        )
    return value
