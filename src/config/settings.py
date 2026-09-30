"""
src/config/settings.py
======================
Central configuration module for the RAG AI Teacher Assistant.

Handles path resolutions across Windows, macOS, and Linux using Python's pathlib.
Loads optional environment variables from a .env file located at the project root.
Provides production-ready configuration defaults, environment overrides, and diagnostic summaries.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Base Directory: root folder of the project ('rag-ai-teacher')
# Using resolve() ensures cross-platform absolute path resolution without hardcoding
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file if it exists at project root
ENV_FILE: Path = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE)

# -----------------------------------------------------------------------------
# Application Environment & Logging
# -----------------------------------------------------------------------------
APP_ENV: str = os.getenv("APP_ENV", "development").lower()
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def is_production() -> bool:
    """Returns True if the application is running in production mode."""
    return APP_ENV in ("production", "prod")


# -----------------------------------------------------------------------------
# File & Directory Paths
# -----------------------------------------------------------------------------
# Primary Data Directory
DATA_DIR: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))

# Configurable data subdirectories (can be overridden via environment variables)
VIDEO_DIR: Path = Path(os.getenv("VIDEO_DIR", DATA_DIR / "videos"))
AUDIO_DIR: Path = Path(os.getenv("AUDIO_DIR", DATA_DIR / "audio"))
TRANSCRIPT_DIR: Path = Path(os.getenv("TRANSCRIPT_DIR", DATA_DIR / "transcripts"))
PROCESSED_TRANSCRIPT_DIR: Path = Path(
    os.getenv("PROCESSED_TRANSCRIPT_DIR", TRANSCRIPT_DIR / "processed")
)
CHUNK_DIR: Path = Path(os.getenv("CHUNK_DIR", DATA_DIR / "chunks"))
EMBEDDING_DIR: Path = Path(os.getenv("EMBEDDING_DIR", DATA_DIR / "embeddings"))
PROCESSED_DATA_DIR: Path = Path(os.getenv("PROCESSED_DATA_DIR", DATA_DIR / "processed"))
CONSOLIDATED_DATASET_PATH: Path = Path(
    os.getenv("DATASET_PATH", PROCESSED_DATA_DIR / "teacher_embeddings.joblib")
)

# Course Catalog Path (Module 10)
COURSE_CATALOG_PATH: Path = Path(
    os.getenv("COURSE_CATALOG_PATH", DATA_DIR / "course_catalog.json")
)

# Evaluation Paths (Module 13)
EVALUATION_DIR: Path = Path(os.getenv("EVALUATION_DIR", DATA_DIR / "evaluation"))
EVALUATION_DATASET_PATH: Path = Path(
    os.getenv("EVALUATION_DATASET_PATH", EVALUATION_DIR / "rag_benchmark.json")
)
RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR", EVALUATION_DIR / "results"))

# List of all essential data directories used by the application pipeline
ALL_DATA_DIRS: list[Path] = [
    VIDEO_DIR,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    PROCESSED_TRANSCRIPT_DIR,
    CHUNK_DIR,
    EMBEDDING_DIR,
    PROCESSED_DATA_DIR,
    EVALUATION_DIR,
    RESULTS_DIR,
]

# -----------------------------------------------------------------------------
# Media Ingestion Settings
# -----------------------------------------------------------------------------
SUPPORTED_AUDIO_EXTENSIONS: set[str] = {".mp3", ".wav", ".m4a"}
SUPPORTED_VIDEO_EXTENSIONS: set[str] = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
ALL_SUPPORTED_MEDIA_EXTENSIONS: set[str] = SUPPORTED_AUDIO_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "200"))

# Audio transcription / Whisper settings (Module 3)
DEFAULT_WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# -----------------------------------------------------------------------------
# Chunking Settings (Module 5)
# -----------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "700"))
DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# -----------------------------------------------------------------------------
# Vector Embedding Settings (Module 6)
# -----------------------------------------------------------------------------
DEFAULT_EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
DEFAULT_NORMALIZE_EMBEDDINGS: bool = (
    os.getenv("NORMALIZE_EMBEDDINGS", "True").lower() in ("true", "1", "yes")
)

# -----------------------------------------------------------------------------
# Semantic Retrieval Settings (Module 7)
# -----------------------------------------------------------------------------
DEFAULT_TOP_K: int = int(os.getenv("TOP_K", "5"))
_raw_threshold = os.getenv("SIMILARITY_THRESHOLD")
DEFAULT_SIMILARITY_THRESHOLD: float | None = (
    float(_raw_threshold) if _raw_threshold and _raw_threshold.strip() else None
)

# Relevance / Abstention threshold (Module 9)
# Minimum similarity required for top retrieved chunk to be considered relevant.
# Queries below this threshold are marked NOT RELEVANT and rejected with abstention response.
DEFAULT_RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.32"))

# -----------------------------------------------------------------------------
# LLM Provider & Generation Settings (Module 9)
# -----------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
DEFAULT_LLM_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "").strip()
DEFAULT_LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
DEFAULT_MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
DEFAULT_CONVERSATION_TURNS: int = int(os.getenv("CONVERSATION_TURNS", "4"))


def get_config_summary() -> dict[str, Any]:
    """
    Returns a dictionary summary of key configuration paths and settings.
    Useful for debugging, logging, and diagnostics in tests or UI.
    Masks sensitive values such as API keys.
    """
    api_key_configured = bool(OPENAI_API_KEY.strip()) and not OPENAI_API_KEY.startswith("your_")
    return {
        "APP_ENV": APP_ENV,
        "LOG_LEVEL": LOG_LEVEL,
        "BASE_DIR": str(BASE_DIR),
        "DATA_DIR": str(DATA_DIR),
        "VIDEO_DIR": str(VIDEO_DIR),
        "AUDIO_DIR": str(AUDIO_DIR),
        "TRANSCRIPT_DIR": str(TRANSCRIPT_DIR),
        "PROCESSED_TRANSCRIPT_DIR": str(PROCESSED_TRANSCRIPT_DIR),
        "CHUNK_DIR": str(CHUNK_DIR),
        "EMBEDDING_DIR": str(EMBEDDING_DIR),
        "PROCESSED_DATA_DIR": str(PROCESSED_DATA_DIR),
        "CONSOLIDATED_DATASET_PATH": str(CONSOLIDATED_DATASET_PATH),
        "COURSE_CATALOG_PATH": str(COURSE_CATALOG_PATH),
        "EVALUATION_DATASET_PATH": str(EVALUATION_DATASET_PATH),
        "RESULTS_DIR": str(RESULTS_DIR),
        "WHISPER_MODEL": DEFAULT_WHISPER_MODEL,
        "SUPPORTED_AUDIO_EXTENSIONS": ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS)),
        "SUPPORTED_VIDEO_EXTENSIONS": ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS)),
        "CHUNK_SIZE": str(DEFAULT_CHUNK_SIZE),
        "CHUNK_OVERLAP": str(DEFAULT_CHUNK_OVERLAP),
        "EMBEDDING_MODEL": DEFAULT_EMBEDDING_MODEL,
        "EMBEDDING_BATCH_SIZE": str(DEFAULT_EMBEDDING_BATCH_SIZE),
        "NORMALIZE_EMBEDDINGS": str(DEFAULT_NORMALIZE_EMBEDDINGS),
        "TOP_K": str(DEFAULT_TOP_K),
        "SIMILARITY_THRESHOLD": str(DEFAULT_SIMILARITY_THRESHOLD),
        "RELEVANCE_THRESHOLD": str(DEFAULT_RELEVANCE_THRESHOLD),
        "LLM_PROVIDER": LLM_PROVIDER,
        "LLM_MODEL": DEFAULT_LLM_MODEL,
        "LLM_TEMPERATURE": str(DEFAULT_LLM_TEMPERATURE),
        "MAX_CONTEXT_CHUNKS": str(DEFAULT_MAX_CONTEXT_CHUNKS),
        "CONVERSATION_TURNS": str(DEFAULT_CONVERSATION_TURNS),
        "MAX_UPLOAD_SIZE_MB": str(MAX_UPLOAD_SIZE_MB),
        "ENV_FILE_FOUND": str(ENV_FILE.exists()),
        "OPENAI_API_KEY_CONFIGURED": "Yes" if api_key_configured else "No",
    }
