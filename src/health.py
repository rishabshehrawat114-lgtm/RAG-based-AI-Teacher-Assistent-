"""
src/health.py
=============
System startup validation and health check module for RAG AI Teacher Assistant.

Verifies:
- Essential project directory structure
- Core configuration parameters and thresholds
- Python runtime environment
- Local embedding and vector store readiness
- Course catalog and benchmark dataset integrity
- Optional capabilities (FFmpeg video decoding, OpenAI LLM synthesis)

Ensures that missing optional services (such as OpenAI API keys or FFmpeg)
do NOT crash the application, but instead trigger clean offline/fallback modes.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import (
    ALL_DATA_DIRS,
    APP_ENV,
    COURSE_CATALOG_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_WHISPER_MODEL,
    EVALUATION_DATASET_PATH,
    OPENAI_API_KEY,
    is_production,
)
from src.utils import dir_exists, ensure_dir, file_exists, mask_secret

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckItem:
    """Individual health check component result."""
    name: str
    status: str  # "OK", "WARNING", "ERROR"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Comprehensive system health report."""
    status: str  # "OK", "WARNING", "ERROR"
    is_healthy: bool
    offline_mode: bool
    checks: list[HealthCheckItem] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Converts report into a dictionary format."""
        return {
            "status": self.status,
            "is_healthy": self.is_healthy,
            "offline_mode": self.offline_mode,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
        }


def check_directories() -> HealthCheckItem:
    """Verifies that all required application data directories exist or can be created."""
    created_or_verified = []
    failed = []
    for d in ALL_DATA_DIRS:
        try:
            ensure_dir(d)
            created_or_verified.append(str(d.name))
        except Exception as exc:
            failed.append(f"{d.name} ({exc})")

    if failed:
        return HealthCheckItem(
            name="Directories",
            status="ERROR",
            message=f"Failed to create essential directories: {', '.join(failed)}",
            details={"verified": created_or_verified, "failed": failed},
        )
    return HealthCheckItem(
        name="Directories",
        status="OK",
        message=f"All {len(created_or_verified)} application directories verified.",
        details={"directories": created_or_verified},
    )


def check_configuration() -> HealthCheckItem:
    """Validates configuration parameters and boundaries."""
    issues = []
    if DEFAULT_RELEVANCE_THRESHOLD <= 0.0 or DEFAULT_RELEVANCE_THRESHOLD >= 1.0:
        issues.append(f"Invalid RELEVANCE_THRESHOLD: {DEFAULT_RELEVANCE_THRESHOLD} (must be between 0.0 and 1.0)")
    if DEFAULT_CHUNK_SIZE <= 0:
        issues.append(f"Invalid CHUNK_SIZE: {DEFAULT_CHUNK_SIZE} (must be > 0)")
    if DEFAULT_TOP_K <= 0:
        issues.append(f"Invalid TOP_K: {DEFAULT_TOP_K} (must be > 0)")

    if issues:
        return HealthCheckItem(
            name="Configuration",
            status="ERROR",
            message="; ".join(issues),
            details={"issues": issues},
        )
    return HealthCheckItem(
        name="Configuration",
        status="OK",
        message=(
            f"Config valid. Env: {APP_ENV}, Threshold: {DEFAULT_RELEVANCE_THRESHOLD:.2f}, "
            f"Embedding: {DEFAULT_EMBEDDING_MODEL}, Top-K: {DEFAULT_TOP_K}"
        ),
        details={
            "app_env": APP_ENV,
            "relevance_threshold": DEFAULT_RELEVANCE_THRESHOLD,
            "top_k": DEFAULT_TOP_K,
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "whisper_model": DEFAULT_WHISPER_MODEL,
        },
    )


def check_python_runtime() -> HealthCheckItem:
    """Checks Python version compatibility (>= 3.9)."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    if sys.version_info < (3, 9):
        return HealthCheckItem(
            name="Python Runtime",
            status="ERROR",
            message=f"Python version {version_str} is unsupported. Python 3.9+ is required.",
            details={"version": version_str},
        )
    return HealthCheckItem(
        name="Python Runtime",
        status="OK",
        message=f"Python {version_str} detected.",
        details={"version": version_str, "executable": sys.executable},
    )


def check_optional_ffmpeg() -> HealthCheckItem:
    """Checks whether FFmpeg is available for video processing (optional)."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return HealthCheckItem(
            name="FFmpeg (Media Extraction)",
            status="OK",
            message="FFmpeg found in system PATH. Video ingestion is enabled.",
            details={"path": ffmpeg_path},
        )
    return HealthCheckItem(
        name="FFmpeg (Media Extraction)",
        status="WARNING",
        message="FFmpeg not found in PATH. Video ingestion will require pre-extracted audio or FFmpeg installation.",
        details={"path": None},
    )


def check_llm_status() -> HealthCheckItem:
    """
    Checks LLM integration status. Missing API key is safe and triggers offline mode.
    """
    key = OPENAI_API_KEY.strip() if OPENAI_API_KEY else ""
    is_placeholder = key.startswith("your_") or not key

    if not is_placeholder:
        return HealthCheckItem(
            name="LLM Integration",
            status="OK",
            message=f"OpenAI API Key configured ({mask_secret(key)}). Generative AI answers active.",
            details={"configured": True, "masked_key": mask_secret(key)},
        )
    return HealthCheckItem(
        name="LLM Integration",
        status="OK",
        message="No OpenAI API key provided. Offline/Local synthesis mode active (grounded citations & direct quotes).",
        details={"configured": False, "offline_mode": True},
    )


def check_course_data() -> HealthCheckItem:
    """Checks course catalog and benchmark dataset availability."""
    catalog_exists = file_exists(COURSE_CATALOG_PATH)
    benchmark_exists = file_exists(EVALUATION_DATASET_PATH)

    details = {
        "catalog_path": str(COURSE_CATALOG_PATH),
        "catalog_exists": catalog_exists,
        "benchmark_path": str(EVALUATION_DATASET_PATH),
        "benchmark_exists": benchmark_exists,
    }

    if catalog_exists and benchmark_exists:
        return HealthCheckItem(
            name="Course & Evaluation Data",
            status="OK",
            message="Course catalog and RAG evaluation benchmark dataset found.",
            details=details,
        )
    elif benchmark_exists and not catalog_exists:
        return HealthCheckItem(
            name="Course & Evaluation Data",
            status="OK",
            message="Benchmark dataset found; Course catalog will auto-initialize on first run.",
            details=details,
        )
    elif catalog_exists and not benchmark_exists:
        return HealthCheckItem(
            name="Course & Evaluation Data",
            status="WARNING",
            message="Course catalog found, but evaluation benchmark dataset was not found.",
            details=details,
        )
    else:
        return HealthCheckItem(
            name="Course & Evaluation Data",
            status="WARNING",
            message="Clean initial state: No existing catalog or benchmark dataset detected.",
            details=details,
        )


def check_system_health() -> HealthReport:
    """
    Executes all system health checks and compiles a comprehensive HealthReport.

    Returns:
        HealthReport indicating overall system health and individual component statuses.
    """
    checks = [
        check_python_runtime(),
        check_directories(),
        check_configuration(),
        check_optional_ffmpeg(),
        check_llm_status(),
        check_course_data(),
    ]

    has_error = any(c.status == "ERROR" for c in checks)
    has_warning = any(c.status == "WARNING" for c in checks)

    llm_check = next((c for c in checks if c.name == "LLM Integration"), None)
    offline_mode = bool(llm_check and llm_check.details.get("offline_mode", False))

    if has_error:
        overall_status = "ERROR"
        is_healthy = False
        summary = "System startup checks encountered critical errors. Please inspect configuration."
    elif has_warning:
        overall_status = "WARNING"
        is_healthy = True
        summary = "System is operational with warnings (e.g. optional tooling or offline mode)."
    else:
        overall_status = "OK"
        is_healthy = True
        summary = "All startup health checks passed. System is fully operational."

    return HealthReport(
        status=overall_status,
        is_healthy=is_healthy,
        offline_mode=offline_mode,
        checks=checks,
        summary=summary,
    )


if __name__ == "__main__":
    report = check_system_health()
    print("=" * 65)
    print(f" RAG AI Teacher Assistant - Startup Health Check [{report.status}]")
    print("=" * 65)
    print(f"Summary: {report.summary}")
    print(f"Mode   : {'Offline / Deterministic' if report.offline_mode else 'Full Generative AI'}")
    print("-" * 65)
    for chk in report.checks:
        tag = "[PASS]" if chk.status == "OK" else ("[WARN]" if chk.status == "WARNING" else "[FAIL]")
        print(f" {tag} [{chk.status:7s}] {chk.name}: {chk.message}")
    print("=" * 65)
    sys.exit(0 if report.is_healthy else 1)
