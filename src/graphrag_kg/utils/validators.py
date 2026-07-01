"""Input validation helpers for GraphRAG-KG."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def validate_directory(path: Path, create: bool = False) -> Path:
    """Validate that a path is an existing directory, optionally creating it.

    Args:
        path: Directory path to validate.
        create: If True, create the directory if it doesn't exist.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If path exists but is not a directory.
        NotADirectoryError: If path doesn't exist and create=False.
    """
    path = path.resolve()
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
    elif create:
        path.mkdir(parents=True, exist_ok=True)
    else:
        raise NotADirectoryError(f"Directory does not exist: {path}")
    return path


def validate_file(path: Path, must_exist: bool = True) -> Path:
    """Validate that a path is a file.

    Args:
        path: File path to validate.
        must_exist: If True, raise if file doesn't exist.

    Returns:
        The resolved Path object.

    Raises:
        ValueError: If path exists but is a directory.
        FileNotFoundError: If path doesn't exist and must_exist=True.
    """
    path = path.resolve()
    if path.exists():
        if path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {path}")
    elif must_exist:
        raise FileNotFoundError(f"File does not exist: {path}")
    return path


def validate_project_name(name: str) -> str:
    """Validate a project name.

    Must be 1-128 characters, containing only alphanumeric,
    hyphens, underscores, and dots.

    Raises:
        ValueError: If name is invalid.
    """
    if not name:
        raise ValueError("Project name cannot be empty.")
    if len(name) > 128:
        raise ValueError(f"Project name too long ({len(name)} > 128).")
    if not all(c.isalnum() or c in "-_." for c in name):
        raise ValueError(
            f"Project name '{name}' contains invalid characters. "
            f"Use only letters, numbers, hyphens, underscores, and dots."
        )
    return name


def validate_encoding(encoding: str) -> str:
    """Validate a text encoding name.

    Raises:
        ValueError: If encoding is not recognized.
    """
    common_encodings = {
        "utf-8", "utf-16", "utf-32", "ascii", "latin-1",
        "gbk", "gb2312", "gb18030", "big5", "shift_jis",
        "euc-jp", "euc-kr", "iso-8859-1", "windows-1252",
    }
    if encoding.lower() not in common_encodings:
        # Try to actually use it
        try:
            "test".encode(encoding)
        except LookupError:
            raise ValueError(f"Unknown encoding: {encoding}")
    return encoding


def validate_neo4j_uri(uri: str) -> str:
    """Validate a Neo4j Bolt URI.

    Raises:
        ValueError: If URI format is invalid.
    """
    if not uri.startswith(("bolt://", "neo4j://", "bolt+s://", "neo4j+s://")):
        raise ValueError(
            f"Invalid Neo4j URI: {uri}. "
            f"Must start with bolt://, neo4j://, bolt+s://, or neo4j+s://"
        )
    return uri
