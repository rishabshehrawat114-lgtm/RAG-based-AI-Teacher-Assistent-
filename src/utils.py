"""
src/utils.py
============
Reusable utility helper functions for the RAG AI Teacher Assistant.

Includes functions for:
- Creating and verifying directories
- Generating sanitized, cross-platform filesystem-safe filenames with path traversal protection
- Checking file and directory existence
- Centralized structured logging setup with secret masking
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Union

from src.config import ALL_DATA_DIRS, LOG_LEVEL

_LOGGING_INITIALIZED = False


def setup_logging(
    level: str | None = None,
    log_file: Path | str | None = None,
) -> None:
    """
    Configures centralized application logging with consistent formatting.

    Args:
        level: Optional log level override (e.g. 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional path to append log entries on disk.
    """
    global _LOGGING_INITIALIZED
    effective_level = (level or LOG_LEVEL or "INFO").upper()
    numeric_level = getattr(logging, effective_level, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not _LOGGING_INITIALIZED:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        _LOGGING_INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger configured with the application logging format."""
    if not _LOGGING_INITIALIZED:
        setup_logging()
    return logging.getLogger(name)


def mask_secret(secret: str | None, visible_prefix: int = 4, visible_suffix: int = 4) -> str:
    """
    Masks a sensitive string (such as an API key) for safe logging or display.

    Example:
        'sk-proj-123456789abcdef' -> 'sk-p...cdef'
    """
    if not secret:
        return "[NOT CONFIGURED]"
    s = str(secret).strip()
    if len(s) <= (visible_prefix + visible_suffix):
        return "***"
    return f"{s[:visible_prefix]}...{s[-visible_suffix:]}"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensures that a directory exists. If it does not exist, it is created
    along with any necessary parent directories.

    Args:
        path: Path or string representing the directory path.

    Returns:
        Path: The resolved Path object pointing to the directory.
    """
    directory_path = Path(path).resolve()
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def file_exists(path: Union[str, Path]) -> bool:
    """
    Checks if a path exists and points to an actual file.

    Args:
        path: Path or string representing the file path.

    Returns:
        bool: True if file exists and is a file, False otherwise.
    """
    target = Path(path)
    return target.exists() and target.is_file()


def dir_exists(path: Union[str, Path]) -> bool:
    """
    Checks if a path exists and points to a directory.

    Args:
        path: Path or string representing the directory path.

    Returns:
        bool: True if directory exists and is a directory, False otherwise.
    """
    target = Path(path)
    return target.exists() and target.is_dir()


def sanitize_filename(filename: str, replacement: str = "_") -> str:
    """
    Sanitizes a filename by removing path traversal patterns and characters that are
    invalid or problematic across Windows, Linux, and macOS filesystems.

    Args:
        filename: Raw input string to convert into a safe filename.
        replacement: Character used to replace invalid characters (default: "_").

    Returns:
        str: A clean, safe basename without directory components.
    """
    if not filename or not isinstance(filename, str):
        return "untitled"

    # Extract basename to neutralize directory traversal attacks (e.g. '../../etc/passwd')
    raw = Path(filename).name.strip()
    if not raw:
        return "untitled"

    # Remove null bytes and control characters
    raw = "".join(ch for ch in raw if ord(ch) >= 32 and ch not in '\x00\r\n\t')

    # Extract clean file extension if present at end of string
    ext_match = re.search(r"\.([a-zA-Z0-9]+)$", raw)
    if ext_match:
        ext = f".{ext_match.group(1).lower()}"
        stem = raw[: ext_match.start()]
    else:
        ext = ""
        stem = raw

    # Replace all non-alphanumeric/non-hyphen characters in the stem with the replacement
    cleaned_stem = re.sub(r"[^\w\-]+", replacement, stem)

    # Collapse consecutive replacement characters
    cleaned_stem = re.sub(rf"{re.escape(replacement)}+", replacement, cleaned_stem)

    # Strip leading and trailing replacement characters, dots, and hyphens
    cleaned_stem = cleaned_stem.strip(f"{replacement}-.")

    # Fallback if stem is empty
    if not cleaned_stem:
        cleaned_stem = "untitled"

    return f"{cleaned_stem}{ext}"


def create_project_directories() -> list[Path]:
    """
    Creates and verifies all essential data directories configured in the project.

    Returns:
        list[Path]: List of Path objects for all verified/created directories.
    """
    verified_dirs: list[Path] = []
    for directory in ALL_DATA_DIRS:
        verified_dirs.append(ensure_dir(directory))
    return verified_dirs
