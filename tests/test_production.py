"""
tests/test_production.py
========================
Automated production readiness, configuration, portability, security hygiene,
and error handling test suite for Module 14.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    ALL_DATA_DIRS,
    ALL_SUPPORTED_MEDIA_EXTENSIONS,
    APP_ENV,
    BASE_DIR,
    COURSE_CATALOG_PATH,
    DATA_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_WHISPER_MODEL,
    EVALUATION_DATASET_PATH,
    MAX_UPLOAD_SIZE_MB,
    RESULTS_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_VIDEO_EXTENSIONS,
    get_config_summary,
    is_production,
)
from src.data.course_catalog import CourseCatalog
from src.health import check_directories, check_system_health, check_configuration, check_python_runtime
from src.llm import LLMClient
from src.utils import dir_exists, ensure_dir, mask_secret, sanitize_filename


# =============================================================================
# 1. CONFIGURATION TESTS
# =============================================================================

class TestConfigurationManagement:
    """Verifies that the centralized configuration system behaves correctly."""

    def test_default_constants(self) -> None:
        """Verifies core default configuration values match project standards."""
        assert DEFAULT_RELEVANCE_THRESHOLD == 0.32
        assert DEFAULT_TOP_K >= 1
        assert DEFAULT_CHUNK_SIZE > 0
        assert DEFAULT_CHUNK_OVERLAP >= 0
        assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE
        assert DEFAULT_EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert DEFAULT_WHISPER_MODEL in ("tiny", "base", "small", "medium", "large")
        assert MAX_UPLOAD_SIZE_MB > 0

    def test_supported_media_extensions(self) -> None:
        """Verifies that supported video and audio formats are properly configured."""
        assert ".mp3" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".wav" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".m4a" in SUPPORTED_AUDIO_EXTENSIONS
        assert ".mp4" in SUPPORTED_VIDEO_EXTENSIONS
        assert ".mkv" in SUPPORTED_VIDEO_EXTENSIONS
        assert ALL_SUPPORTED_MEDIA_EXTENSIONS == (SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS)

    def test_config_summary_masking(self) -> None:
        """Verifies get_config_summary returns valid dictionary without leaking unmasked keys."""
        summary = get_config_summary()
        assert isinstance(summary, dict)
        assert "BASE_DIR" in summary
        assert "RELEVANCE_THRESHOLD" in summary
        assert "OPENAI_API_KEY_CONFIGURED" in summary
        # Ensure raw secret string is never exposed as a dictionary key or value
        for k, v in summary.items():
            assert not str(v).startswith("sk-proj-")
            assert not str(k).startswith("sk-proj-")

    def test_is_production_flag(self) -> None:
        """Verifies is_production helper responds appropriately to environment."""
        with patch.dict(os.environ, {"APP_ENV": "production"}):
            import importlib
            import src.config.settings as s
            importlib.reload(s)
            assert s.is_production() is True

        with patch.dict(os.environ, {"APP_ENV": "development"}):
            importlib.reload(s)
            assert s.is_production() is False


# =============================================================================
# 2. PATH PORTABILITY TESTS
# =============================================================================

class TestPathPortability:
    """Verifies that paths are dynamic Path objects and portable across machines."""

    def test_base_dir_resolution(self) -> None:
        """Verifies BASE_DIR resolves to project root containing app.py and requirements.txt."""
        assert isinstance(BASE_DIR, Path)
        assert BASE_DIR.is_dir()
        assert (BASE_DIR / "app.py").exists()
        assert (BASE_DIR / "requirements.txt").exists()

    def test_data_paths_structure(self) -> None:
        """Verifies all essential data paths are subpaths of DATA_DIR or BASE_DIR."""
        for d in ALL_DATA_DIRS:
            assert isinstance(d, Path)
            assert str(d).startswith(str(BASE_DIR)) or str(d).startswith(str(DATA_DIR))

    def test_course_catalog_portable_resolution(self, tmp_path: Path) -> None:
        """Verifies CourseCatalog.resolve_file_path recovers from alien machine paths."""
        local_dir = tmp_path / "chunks"
        local_dir.mkdir(parents=True, exist_ok=True)
        sample_chunk = local_dir / "lecture_01_chunks.json"
        sample_chunk.write_text('{"chunks": []}', encoding="utf-8")

        alien_path = "/foreign/machine/data/chunks/lecture_01_chunks.json"
        resolved = CourseCatalog.resolve_file_path(alien_path, fallback_dir=local_dir)
        assert resolved is not None
        assert resolved.exists()
        assert resolved == sample_chunk

    def test_course_catalog_resolve_nonexistent(self, tmp_path: Path) -> None:
        """Verifies CourseCatalog.resolve_file_path returns None for missing files."""
        res = CourseCatalog.resolve_file_path("totally_nonexistent_file_9999.json", fallback_dir=tmp_path)
        assert res is None


# =============================================================================
# 3. SECRETS & SECURITY HYGIENE TESTS
# =============================================================================

class TestSecurityHygiene:
    """Verifies secret masking, .env.example hygiene, and path traversal protection."""

    def test_env_example_contains_no_real_secrets(self) -> None:
        """Verifies .env.example contains only placeholders, never real keys."""
        env_example_path = BASE_DIR / ".env.example"
        assert env_example_path.exists()
        content = env_example_path.read_text(encoding="utf-8")

        assert "sk-proj-" not in content
        assert "OPENAI_API_KEY=your_openai_api_key_here" in content or "your_" in content
        assert "RELEVANCE_THRESHOLD=0.32" in content

    def test_gitignore_covers_sensitive_files(self) -> None:
        """Verifies .gitignore rules exclude .env, caches, venvs, and logs."""
        gitignore_path = BASE_DIR / ".gitignore"
        assert gitignore_path.exists()
        content = gitignore_path.read_text(encoding="utf-8")

        assert ".env" in content
        assert "!.env.example" in content
        assert ".venv" in content or "venv" in content
        assert "__pycache__" in content
        assert ".pytest_cache" in content
        assert ".streamlit/secrets.toml" in content
        # Ensure rag_benchmark.json is explicitly preserved
        assert "!data/evaluation/rag_benchmark.json" in content

    def test_mask_secret_helper(self) -> None:
        """Verifies mask_secret correctly masks keys and handles edge cases."""
        assert mask_secret("sk-proj-1234567890abcdef") == "sk-p...cdef"
        assert mask_secret(None) == "[NOT CONFIGURED]"
        assert mask_secret("") == "[NOT CONFIGURED]"
        assert mask_secret("short") == "***"

    def test_sanitize_filename_traversal_neutralization(self) -> None:
        """Verifies sanitize_filename blocks directory traversal attacks."""
        dangerous_inputs = [
            ("../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\cmd.exe", "cmd.exe"),
            ("/var/log/syslog", "syslog"),
            ("C:\\Users\\victim\\secret.txt", "secret.txt"),
            ("lecture\x00file.mp4", "lecturefile.mp4"),
            ("....///test.wav", "test.wav"),
            ("", "untitled"),
            (None, "untitled"),
        ]
        for raw, expected_name in dangerous_inputs:
            cleaned = sanitize_filename(raw)  # type: ignore
            assert "/" not in cleaned
            assert "\\" not in cleaned
            assert ".." not in cleaned
            assert "\x00" not in cleaned
            assert cleaned == expected_name


# =============================================================================
# 4. ERROR HANDLING & OFFLINE MODE TESTS
# =============================================================================

class TestErrorHandlingAndOfflineMode:
    """Verifies that missing optional credentials do not crash the application."""

    def test_llm_client_offline_fallback(self) -> None:
        """Verifies LLMClient raises clean LLMError without unhandled crash when key is absent."""
        from src.llm import LLMError
        client = LLMClient(api_key="")
        assert not client.is_configured()
        with pytest.raises(LLMError) as exc_info:
            client.generate([{"role": "user", "content": "Hello test prompt"}])
        assert "OpenAI API key is missing" in str(exc_info.value)

    def test_llm_client_placeholder_key_detected(self) -> None:
        """Verifies LLMClient recognizes placeholder keys as unconfigured."""
        client = LLMClient(api_key="your_openai_api_key_here")
        assert not client.is_configured()

    def test_system_health_checks(self) -> None:
        """Verifies system health checks execute and return a valid report."""
        report = check_system_health()
        assert report.is_healthy is True
        assert report.status in ("OK", "WARNING")
        assert len(report.checks) >= 5
        report_dict = report.to_dict()
        assert "status" in report_dict
        assert "checks" in report_dict

    def test_check_configuration_boundaries(self) -> None:
        """Verifies check_configuration catches invalid thresholds."""
        item = check_configuration()
        assert item.status == "OK"

        with patch("src.health.DEFAULT_RELEVANCE_THRESHOLD", 0.0):
            bad_item = check_configuration()
            assert bad_item.status == "ERROR"

    def test_check_directories_success(self) -> None:
        """Verifies check_directories validates existing directories."""
        item = check_directories()
        assert item.status == "OK"
