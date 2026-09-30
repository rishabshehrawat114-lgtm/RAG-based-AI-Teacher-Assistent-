"""
src/embeddings/embedder.py
==========================
Dense vector embedding generation and persistence for the RAG AI Teacher Assistant.

Converts transcript chunks generated in Module 5 into numerical embedding vectors
using local Sentence Transformers models (e.g. 'all-MiniLM-L6-v2').

Architecture & Core Principles:
-------------------------------
1. Local & Free: Operates entirely locally with Sentence Transformers; no paid API keys required.
2. Configurable: Model selection, batch sizes, and cosine normalization are managed centrally
   via `src/config.py` and environment variables.
3. Automatic Dimension Detection: The embedding dimension (e.g. 384) is determined dynamically
   from the loaded model rather than hardcoded.
4. Strict Deterministic Alignment: Embedding row `i` strictly matches metadata chunk `i`
   with explicit `embedding_index: i` recorded in the metadata JSON.
5. Optimized Persistence: Numerical vectors are stored in compressed NumPy matrices (.npz)
   and metadata is stored in human-readable JSON (.json).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    EMBEDDING_DIR,
)

logger = logging.getLogger(__name__)

# In-memory model cache to avoid repeated model loading across multiple files
_MODEL_CACHE: dict[str, Any] = {}


class EmbeddingError(Exception):
    """Custom exception raised when embedding generation, loading, or validation fails."""
    pass


def load_embedding_model(
    model_name: str | None = None,
    device: str | None = None,
) -> Any:
    """
    Loads and caches a Sentence Transformers embedding model.

    The model is downloaded automatically to the local Hugging Face cache on first use.
    Subsequent calls with the same model name return the cached in-memory instance.

    Args:
        model_name: Name or path of the Sentence Transformers model.
                    Defaults to DEFAULT_EMBEDDING_MODEL ('all-MiniLM-L6-v2').
        device: Device to load the model on ('cpu', 'cuda', etc.). If None, automatically detected.

    Returns:
        Loaded SentenceTransformer instance.

    Raises:
        EmbeddingError: If sentence-transformers is not installed or model loading fails.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingError(
            "The 'sentence-transformers' package is required for Module 6. "
            "Please install it using: pip install sentence-transformers"
        ) from exc

    resolved_model_name = (model_name or DEFAULT_EMBEDDING_MODEL).strip()
    cache_key = f"{resolved_model_name}_{device or 'auto'}"

    if cache_key in _MODEL_CACHE:
        logger.debug("Using cached embedding model: %s", resolved_model_name)
        return _MODEL_CACHE[cache_key]

    logger.info("Loading embedding model: '%s'...", resolved_model_name)
    try:
        model = SentenceTransformer(model_name_or_path=resolved_model_name, device=device)
        _MODEL_CACHE[cache_key] = model
        logger.info(
            "Successfully loaded '%s' (Embedding Dimension: %d)",
            resolved_model_name,
            get_embedding_dimension(model),
        )
        return model
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to load embedding model '{resolved_model_name}'. "
            f"Ensure network connectivity for first download or verify the model name. Error: {exc}"
        ) from exc


def get_embedding_dimension(model: Any) -> int:
    """
    Dynamically detects the output embedding dimension of a loaded model.

    Avoids hardcoding dimension values (e.g. 384 for all-MiniLM-L6-v2) so any
    supported Sentence Transformers model can be used seamlessly.

    Args:
        model: Loaded SentenceTransformer instance or compatible object.

    Returns:
        Integer embedding dimension (e.g. 384, 768, 1024).

    Raises:
        EmbeddingError: If dimension cannot be determined.
    """
    # 1. Standard SentenceTransformer API (supports modern and legacy methods)
    for method_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        if hasattr(model, method_name):
            try:
                dim = getattr(model, method_name)()
                if dim and isinstance(dim, int) and dim > 0:
                    return dim
            except Exception:
                pass

    # 2. Inspect underlying model configuration if present
    if hasattr(model, "tokenizer") and hasattr(model, "modules"):
        for module in model.modules():
            if hasattr(module, "word_embedding_dimension"):
                return int(module.word_embedding_dimension)

    # 3. Fallback: encode a short probe text and inspect vector shape
    try:
        sample_vec = model.encode(["dimension_probe"], convert_to_numpy=True)
        if hasattr(sample_vec, "shape") and len(sample_vec.shape) == 2:
            return int(sample_vec.shape[1])
    except Exception as exc:
        raise EmbeddingError(
            f"Unable to automatically determine embedding dimension from model: {exc}"
        ) from exc

    raise EmbeddingError("Could not detect embedding dimension from the provided model.")


def generate_embeddings(
    texts: list[str],
    model: Any = None,
    model_name: str | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Generates dense vector embeddings for a list of text strings in batches.

    Args:
        texts: List of text strings to embed.
        model: Pre-loaded SentenceTransformer model. If None, loads `model_name`.
        model_name: Model name to load if `model` is not passed.
        batch_size: Batch size for model inference.
        normalize: If True, L2-normalizes embeddings for cosine similarity.
        show_progress_bar: Whether to display a progress bar during encoding.

    Returns:
        2D NumPy array of shape (len(texts), embedding_dimension) with float32 values.

    Raises:
        EmbeddingError: If model encoding fails.
    """
    if model is None:
        model = load_embedding_model(model_name)

    dim = get_embedding_dimension(model)

    if not texts:
        return np.empty((0, dim), dtype=np.float32)

    try:
        # Pass inputs= or sentences= depending on sentence-transformers version
        import inspect
        sig = inspect.signature(model.encode)
        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            "normalize_embeddings": normalize,
            "convert_to_numpy": True,
        }
        if "inputs" in sig.parameters:
            embeddings = model.encode(inputs=texts, **kwargs)
        else:
            embeddings = model.encode(sentences=texts, **kwargs)

        embeddings = np.asarray(embeddings, dtype=np.float32)
        return embeddings
    except Exception as exc:
        raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc


def embed_chunks_data(
    chunks_data: dict[str, Any],
    model: Any = None,
    model_name: str | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
    show_progress_bar: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Converts chunk JSON data into an embedding matrix and paired metadata dictionary.

    Validates chunks, skips empty texts with a warning, maintains deterministic 1-to-1
    mapping between embedding rows and metadata records, and enriches metadata with
    model information and explicit row indices.

    Args:
        chunks_data: Dictionary conforming to Module 5 chunk JSON structure.
        model: Pre-loaded SentenceTransformer model or None.
        model_name: Model name to load if model is None.
        batch_size: Batch size for encoding.
        normalize: Whether to normalize embedding vectors.
        show_progress_bar: Show progress bar during batch encoding.

    Returns:
        Tuple of:
          - embeddings: 2D NumPy array of shape (valid_chunks_count, dimension)
          - metadata: Dictionary containing top-level info and aligned chunks list

    Raises:
        EmbeddingError: If input structure is invalid or embedding fails.
    """
    if not isinstance(chunks_data, dict):
        raise EmbeddingError(f"Expected chunks_data to be a dict, got {type(chunks_data).__name__}.")

    raw_chunks = chunks_data.get("chunks")
    if raw_chunks is None or not isinstance(raw_chunks, list):
        raise EmbeddingError("Invalid chunk data: missing or non-list 'chunks' field.")

    if model is None:
        model = load_embedding_model(model_name)

    dim = get_embedding_dimension(model)
    resolved_model_name = getattr(model, "model_card_data", None)
    if resolved_model_name and hasattr(resolved_model_name, "base_model"):
        resolved_name = str(resolved_model_name.base_model or model_name or DEFAULT_EMBEDDING_MODEL)
    else:
        resolved_name = str(model_name or DEFAULT_EMBEDDING_MODEL)

    # Validate and filter chunks, discarding empty/whitespace text segments
    valid_chunks: list[dict[str, Any]] = []
    valid_texts: list[str] = []
    seen_chunk_ids: set[str] = set()

    for idx, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, dict):
            logger.warning("Skipping non-dict chunk at index %d: %s", idx, type(chunk).__name__)
            continue

        chunk_id = str(chunk.get("chunk_id", f"chunk_{idx:04d}")).strip()
        if not chunk_id:
            chunk_id = f"chunk_{idx:04d}"

        if chunk_id in seen_chunk_ids:
            logger.warning("Duplicate chunk_id '%s' found at index %d; preserving duplicate.", chunk_id, idx)
        seen_chunk_ids.add(chunk_id)

        text = chunk.get("text")
        if text is None or not isinstance(text, str) or not text.strip():
            logger.warning("Skipping chunk '%s' at index %d due to empty or missing text.", chunk_id, idx)
            continue

        valid_chunks.append(chunk)
        valid_texts.append(text.strip())

    # Generate embeddings
    if not valid_texts:
        logger.warning("No valid chunks with text found to embed.")
        embeddings = np.empty((0, dim), dtype=np.float32)
    else:
        embeddings = generate_embeddings(
            texts=valid_texts,
            model=model,
            batch_size=batch_size,
            normalize=normalize,
            show_progress_bar=show_progress_bar,
        )

    # Build aligned metadata
    # CRITICAL: Row i of embeddings matrix strictly corresponds to metadata["chunks"][i]
    metadata_chunks: list[dict[str, Any]] = []
    for row_idx, chunk in enumerate(valid_chunks):
        chunk_meta = {
            "chunk_id": chunk.get("chunk_id", f"chunk_{row_idx:04d}"),
            "embedding_index": row_idx,  # Explicit 1-to-1 mapping to embedding matrix row
            # Module 10 multi-lecture course metadata
            "course_id": chunk.get("course_id", chunks_data.get("course_id", "python_course")),
            "lecture_id": chunk.get("lecture_id", chunks_data.get("lecture_id", "")),
            "lecture_title": chunk.get("lecture_title", chunks_data.get("lecture_title", "")),
            "source_filename": chunk.get("source_filename", chunk.get("source_file", chunks_data.get("source_file", ""))),
            "timestamp_start": chunk.get("timestamp_start", ""),
            "timestamp_end": chunk.get("timestamp_end", ""),
            "timestamp": chunk.get("timestamp", ""),
            # Core fields
            "source_file": chunk.get("source_file", chunks_data.get("source_file", "")),
            "source_transcript": chunk.get("source_transcript", ""),
            "language": chunk.get("language", chunks_data.get("language", "en")),
            "start_time": float(chunk.get("start_time", 0.0)),
            "end_time": float(chunk.get("end_time", 0.0)),
            "word_count": int(chunk.get("word_count", len(chunk.get("text", "").split()))),
            "segment_start_id": chunk.get("segment_start_id"),
            "segment_end_id": chunk.get("segment_end_id"),
            "text": chunk.get("text", ""),
        }
        metadata_chunks.append(chunk_meta)

    metadata: dict[str, Any] = {
        "course_id": chunks_data.get("course_id", "python_course"),
        "lecture_id": chunks_data.get("lecture_id", ""),
        "lecture_title": chunks_data.get("lecture_title", ""),
        "source_filename": chunks_data.get("source_filename", chunks_data.get("source_file", "")),
        "source_file": chunks_data.get("source_file", ""),
        "language": chunks_data.get("language", "en"),
        "source_chunk_file": chunks_data.get("source_chunk_file", ""),
        "embedding_model": resolved_name,
        "embedding_dimension": dim,
        "chunk_count": len(metadata_chunks),
        "normalized": bool(normalize),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": metadata_chunks,
    }

    return embeddings, metadata


def save_embeddings(
    embeddings: np.ndarray,
    metadata: dict[str, Any],
    output_stem: str,
    output_dir: Path | str = EMBEDDING_DIR,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """
    Saves numerical embeddings into a compressed .npz file and metadata into a JSON file.

    Files generated:
      - {output_stem}_embeddings.npz: compressed NumPy archive with 'embeddings' and 'chunk_ids'
      - {output_stem}_metadata.json: formatted JSON with full metadata and aligned chunk details

    Args:
        embeddings: 2D NumPy array of shape (N, dimension).
        metadata: Metadata dictionary containing chunk records.
        output_stem: Base name for output files (e.g. 'python_lecture').
        output_dir: Target directory path (defaults to EMBEDDING_DIR).
        overwrite: If False, raises error or skips if files exist.

    Returns:
        Tuple of (npz_path, json_path).

    Raises:
        EmbeddingError: If file writing fails or files exist when overwrite=False.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize output stem (remove '_chunks' suffix if present to keep naming clean)
    clean_stem = output_stem.replace("_chunks", "")
    npz_path = target_dir / f"{clean_stem}_embeddings.npz"
    json_path = target_dir / f"{clean_stem}_metadata.json"

    if not overwrite and npz_path.exists() and json_path.exists():
        logger.info("Embeddings already exist for '%s'; skipping.", clean_stem)
        return npz_path, json_path

    # Verify alignment before writing
    if embeddings.shape[0] != len(metadata.get("chunks", [])):
        raise EmbeddingError(
            f"Embedding row count ({embeddings.shape[0]}) does not match metadata chunk count "
            f"({len(metadata.get('chunks', []))}). Aborting save."
        )

    try:
        chunk_ids = np.array([c["chunk_id"] for c in metadata.get("chunks", [])], dtype=object)
        # Save compressed NPZ
        np.savez_compressed(
            npz_path,
            embeddings=embeddings,
            chunk_ids=chunk_ids,
        )
        logger.info("Saved numerical embeddings to: %s", npz_path)

        # Save metadata JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Saved metadata JSON to: %s", json_path)

        return npz_path, json_path
    except Exception as exc:
        raise EmbeddingError(f"Failed to save embeddings to disk: {exc}") from exc


def load_embeddings(
    input_stem_or_path: Path | str,
    embeddings_dir: Path | str = EMBEDDING_DIR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Loads embeddings and paired metadata for a given lecture stem or file path.

    Verifies 1-to-1 deterministic alignment between the loaded embedding matrix
    and metadata records.

    Args:
        input_stem_or_path: File path or base stem (e.g. 'python_lecture').
        embeddings_dir: Directory containing embedding files.

    Returns:
        Tuple of (embeddings_matrix, metadata_dict).

    Raises:
        EmbeddingError: If files are missing, malformed, or misaligned.
    """
    input_path = Path(input_stem_or_path)
    base_dir = Path(embeddings_dir)

    if input_path.is_file():
        stem = input_path.stem.replace("_embeddings", "").replace("_metadata", "").replace("_chunks", "")
        base_dir = input_path.parent
    else:
        stem = input_path.stem.replace("_embeddings", "").replace("_metadata", "").replace("_chunks", "")

    npz_path = base_dir / f"{stem}_embeddings.npz"
    json_path = base_dir / f"{stem}_metadata.json"

    if not npz_path.exists():
        raise EmbeddingError(f"Embedding file not found: {npz_path}")
    if not json_path.exists():
        raise EmbeddingError(f"Metadata file not found: {json_path}")

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "embeddings" not in data:
                raise EmbeddingError(f"Key 'embeddings' not found in {npz_path}")
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)

        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    except Exception as exc:
        raise EmbeddingError(f"Failed to load embedding files for '{stem}': {exc}") from exc

    # Validate alignment
    chunks = metadata.get("chunks", [])
    expected_rows = len(chunks)
    if embeddings.shape[0] != expected_rows:
        raise EmbeddingError(
            f"Alignment error in '{stem}': embeddings matrix has {embeddings.shape[0]} rows "
            f"but metadata has {expected_rows} chunks."
        )

    expected_dim = metadata.get("embedding_dimension")
    if expected_dim and embeddings.shape[1] != expected_dim:
        raise EmbeddingError(
            f"Dimension mismatch in '{stem}': matrix has dimension {embeddings.shape[1]} "
            f"but metadata specifies {expected_dim}."
        )

    return embeddings, metadata


def embed_chunk_file(
    chunk_file: Path | str,
    output_dir: Path | str = EMBEDDING_DIR,
    model: Any = None,
    model_name: str | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
    overwrite: bool = False,
    show_progress_bar: bool = False,
) -> tuple[Path, Path] | None:
    """
    Generates embeddings and metadata for a single chunk JSON file.

    Args:
        chunk_file: Path to the input chunk JSON file.
        output_dir: Target directory for embeddings and metadata.
        model: Pre-loaded model or None.
        model_name: Model name to load if model is None.
        batch_size: Batch size for encoding.
        normalize: Whether to normalize embeddings.
        overwrite: Whether to overwrite existing embedding files.
        show_progress_bar: Whether to display encoding progress bar.

    Returns:
        Tuple of (npz_path, json_path), or None if skipped because it already exists.

    Raises:
        EmbeddingError: If file loading or processing fails.
    """
    path = Path(chunk_file)
    if not path.exists():
        raise EmbeddingError(f"Chunk file does not exist: {path}")

    clean_stem = path.stem.replace("_chunks", "")
    target_dir = Path(output_dir)
    npz_path = target_dir / f"{clean_stem}_embeddings.npz"
    json_path = target_dir / f"{clean_stem}_metadata.json"

    if not overwrite and npz_path.exists() and json_path.exists():
        logger.info("Embeddings already exist for '%s'; skipping (use --overwrite to recreate).", path.name)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
    except Exception as exc:
        raise EmbeddingError(f"Failed to read chunk JSON from '{path}': {exc}") from exc

    chunks_data["source_chunk_file"] = path.name

    embeddings, metadata = embed_chunks_data(
        chunks_data=chunks_data,
        model=model,
        model_name=model_name,
        batch_size=batch_size,
        normalize=normalize,
        show_progress_bar=show_progress_bar,
    )

    saved_npz, saved_json = save_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        output_stem=clean_stem,
        output_dir=target_dir,
        overwrite=overwrite,
    )

    return saved_npz, saved_json


def batch_embed_chunks(
    chunks_dir: Path | str,
    output_dir: Path | str = EMBEDDING_DIR,
    model_name: str | None = None,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize: bool = DEFAULT_NORMALIZE_EMBEDDINGS,
    overwrite: bool = False,
    show_progress_bar: bool = False,
) -> dict[str, Any]:
    """
    Processes all chunk JSON files in chunks_dir and creates embedding files.

    Args:
        chunks_dir: Directory containing *_chunks.json files.
        output_dir: Target directory for embeddings and metadata.
        model_name: Model name to load.
        batch_size: Batch size for encoding.
        normalize: Whether to normalize embeddings.
        overwrite: Whether to overwrite existing embedding files.
        show_progress_bar: Whether to display encoding progress bar.

    Returns:
        Summary dict containing counts of processed, skipped, failed files, and total chunks.
    """
    src_dir = Path(chunks_dir)
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    if not src_dir.exists():
        raise EmbeddingError(f"Chunks directory not found: {src_dir}")

    chunk_files = sorted(
        [f for f in src_dir.glob("*.json") if not f.name.startswith(".")]
    )

    summary: dict[str, Any] = {
        "total_files": len(chunk_files),
        "embedded_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "total_chunks_embedded": 0,
        "files_detail": [],
    }

    if not chunk_files:
        logger.info("No chunk files found in '%s'.", src_dir)
        return summary

    logger.info("Found %d chunk file(s) in '%s'.", len(chunk_files), src_dir)
    model = load_embedding_model(model_name)

    for idx, cfile in enumerate(chunk_files, start=1):
        file_info: dict[str, Any] = {"file": cfile.name, "status": "unknown"}
        logger.info("[%d/%d] Embedding %s...", idx, len(chunk_files), cfile.name)

        try:
            result = embed_chunk_file(
                chunk_file=cfile,
                output_dir=dst_dir,
                model=model,
                model_name=model_name,
                batch_size=batch_size,
                normalize=normalize,
                overwrite=overwrite,
                show_progress_bar=show_progress_bar,
            )

            if result is None:
                file_info["status"] = "skipped"
                summary["skipped_files"] += 1
                logger.info("  Skipped (already exists).")
            else:
                npz_p, json_p = result
                # Load metadata to get chunk count
                with open(json_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                chunks_count = meta.get("chunk_count", 0)
                file_info["status"] = "embedded"
                file_info["chunks"] = chunks_count
                summary["embedded_files"] += 1
                summary["total_chunks_embedded"] += chunks_count
                logger.info("  Embedded %d chunk(s) -> %s", chunks_count, npz_p.name)

        except Exception as exc:
            file_info["status"] = "failed"
            file_info["error"] = str(exc)
            summary["failed_files"] += 1
            logger.error("  Failed to embed '%s': %s", cfile.name, exc)

        summary["files_detail"].append(file_info)

    logger.info(
        "Finished embedding batch. Total files: %d | Embedded: %d | Skipped: %d | Total chunks: %d",
        summary["total_files"],
        summary["embedded_files"],
        summary["skipped_files"],
        summary["total_chunks_embedded"],
    )

    return summary
