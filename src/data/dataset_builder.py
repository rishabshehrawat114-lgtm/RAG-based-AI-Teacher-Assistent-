"""
src/data/dataset_builder.py
===========================
Bulk embedding dataset builder and Joblib persistence for the RAG AI Teacher Assistant.

Consolidates lecture chunk metadata from `data/chunks/` with precomputed numerical
embedding vectors from `data/embeddings/` into a single, validated Pandas DataFrame,
and serializes it to disk using Joblib.

Key Advantages:
---------------
1. Fast O(1) Deserialization: Eliminates the need to parse dozens of JSON files and .npz
   archives repeatedly during runtime.
2. Guaranteed Alignment: Ensures every embedding vector strictly matches its respective
   chunk metadata row (source video, timestamps, transcript text).
3. Data Integrity & Validation: Validates non-empty text, unique chunk IDs, finite numeric
   arrays, and consistent embedding dimensions across all rows.
4. Flexible Retrieval: Directly consumable by `LectureRetriever` in Module 7.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CHUNK_DIR,
    CONSOLIDATED_DATASET_PATH,
    EMBEDDING_DIR,
    PROCESSED_DATA_DIR,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "chunk_id",
    "text",
    "source_file",
    "source_transcript",
    "language",
    "start_time",
    "end_time",
    "word_count",
    "embedding",
]

MODULE10_REQUIRED_COLUMNS: list[str] = [
    "course_id",
    "lecture_id",
    "lecture_title",
    "source_filename",
    "timestamp_start",
    "timestamp_end",
    "chunk_id",
    "text",
]


class DatasetBuilderError(Exception):
    """Custom exception raised when dataset construction, validation, or persistence fails."""
    pass


def load_chunk_data(chunks_dir: Path | str = CHUNK_DIR) -> dict[str, dict[str, Any]]:
    """
    Loads all chunk JSON files from the chunks directory.

    Args:
        chunks_dir: Path to the directory containing *_chunks.json files.

    Returns:
        Dictionary mapping clean stem (e.g. 'python_lecture') to parsed chunk data dictionary.

    Raises:
        DatasetBuilderError: If chunks_dir does not exist or JSON parsing fails.
    """
    src_dir = Path(chunks_dir)
    if not src_dir.exists():
        raise DatasetBuilderError(f"Chunks directory does not exist: {src_dir}")

    chunk_files = sorted(
        [f for f in src_dir.glob("*.json") if not f.name.startswith(".")]
    )

    chunks_map: dict[str, dict[str, Any]] = {}
    for cfile in chunk_files:
        clean_stem = cfile.stem.replace("_chunks", "")
        try:
            with open(cfile, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks_map[clean_stem] = data
            logger.debug("Loaded chunk file '%s' (%s chunks)", cfile.name, len(data.get("chunks", [])))
        except Exception as exc:
            raise DatasetBuilderError(f"Failed to read chunk JSON '{cfile.name}': {exc}") from exc

    return chunks_map


def load_embedding_data(
    embeddings_dir: Path | str = EMBEDDING_DIR,
) -> dict[str, dict[str, Any]]:
    """
    Loads all embedding archives (.npz) and paired metadata (.json) from embeddings_dir.

    Args:
        embeddings_dir: Path to directory containing *_embeddings.npz and *_metadata.json.

    Returns:
        Dictionary mapping clean stem to dict containing:
          - 'embeddings': 2D NumPy array
          - 'metadata': metadata dictionary
          - 'chunk_ids': list of chunk IDs
          - 'model_name': embedding model string
          - 'dimension': integer embedding dimension

    Raises:
        DatasetBuilderError: If directory does not exist, files are missing, or row counts mismatch.
    """
    src_dir = Path(embeddings_dir)
    if not src_dir.exists():
        raise DatasetBuilderError(f"Embeddings directory does not exist: {src_dir}")

    npz_files = sorted(
        [f for f in src_dir.glob("*_embeddings.npz") if not f.name.startswith(".")]
    )

    if not npz_files:
        raise DatasetBuilderError(
            f"No embedding archives (*_embeddings.npz) found in '{src_dir}'. "
            "Run 'python scripts/create_embeddings.py' first."
        )

    embeddings_map: dict[str, dict[str, Any]] = {}
    for npz_path in npz_files:
        clean_stem = npz_path.stem.replace("_embeddings", "")
        json_path = src_dir / f"{clean_stem}_metadata.json"

        if not json_path.exists():
            raise DatasetBuilderError(
                f"Missing metadata JSON for embedding archive '{npz_path.name}'. "
                f"Expected '{json_path.name}' in '{src_dir}'."
            )

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "embeddings" not in data:
                    raise DatasetBuilderError(f"Archive '{npz_path.name}' is missing 'embeddings' array.")
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)

            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        except Exception as exc:
            raise DatasetBuilderError(f"Failed to load embedding pair for '{clean_stem}': {exc}") from exc

        meta_chunks = metadata.get("chunks", [])
        if embeddings.shape[0] != len(meta_chunks):
            raise DatasetBuilderError(
                f"Index alignment error in '{clean_stem}': {embeddings.shape[0]} embedding rows "
                f"vs {len(meta_chunks)} metadata chunks."
            )

        dim = embeddings.shape[1] if embeddings.shape[0] > 0 else int(metadata.get("embedding_dimension", 384))
        embeddings_map[clean_stem] = {
            "embeddings": embeddings,
            "metadata": metadata,
            "chunks": meta_chunks,
            "model_name": metadata.get("embedding_model", ""),
            "dimension": dim,
        }

    return embeddings_map


def merge_chunks_and_embeddings(
    chunks_data_map: dict[str, Any],
    embeddings_data_map: dict[str, Any],
) -> pd.DataFrame:
    """
    Merges chunk metadata with corresponding embedding vectors into a flat Pandas DataFrame.

    Each row represents one lecture chunk with its exact embedding vector, timestamps,
    and video provenance information.

    Args:
        chunks_data_map: Mapping of stem to chunk JSON data (from load_chunk_data).
        embeddings_data_map: Mapping of stem to embedding records (from load_embedding_data).

    Returns:
        Consolidated Pandas DataFrame containing all chunks across all lectures.

    Raises:
        DatasetBuilderError: If no data exists or merging encounters fatal misalignment.
    """
    if not embeddings_data_map:
        raise DatasetBuilderError("No embedding data provided to merge.")

    rows: list[dict[str, Any]] = []

    for stem, emb_info in embeddings_data_map.items():
        embeddings: np.ndarray = emb_info["embeddings"]
        meta_chunks: list[dict[str, Any]] = emb_info["chunks"]
        model_name: str = emb_info.get("model_name", "")
        top_source_file = emb_info.get("metadata", {}).get("source_file", f"{stem}.mp3")

        # Optional raw chunk lookup for auxiliary fields
        raw_chunk_file_data = chunks_data_map.get(stem, {})
        raw_chunks_list = raw_chunk_file_data.get("chunks", [])
        raw_by_id = {c["chunk_id"]: c for c in raw_chunks_list if isinstance(c, dict) and "chunk_id" in c}

        num_chunks = len(meta_chunks)
        for idx in range(num_chunks):
            chunk_record = meta_chunks[idx]
            vector = embeddings[idx]

            chunk_id = str(chunk_record.get("chunk_id", f"{stem}_{idx:04d}")).strip()
            text = str(chunk_record.get("text", "")).strip()

            # Cross-reference with raw chunk file if auxiliary segment IDs are missing
            raw_match = raw_by_id.get(chunk_id, {})
            seg_start = chunk_record.get("segment_start_id", raw_match.get("segment_start_id"))
            seg_end = chunk_record.get("segment_end_id", raw_match.get("segment_end_id"))

            start_t = float(chunk_record.get("start_time", 0.0))
            end_t = float(chunk_record.get("end_time", 0.0))

            # Format HH:MM:SS timestamps
            s_int = int(round(max(0.0, start_t)))
            e_int = int(round(max(0.0, end_t)))
            hhmmss_start = f"{s_int // 3600:02d}:{(s_int % 3600) // 60:02d}:{s_int % 60:02d}"
            hhmmss_end = f"{e_int // 3600:02d}:{(e_int % 3600) // 60:02d}:{e_int % 60:02d}"

            # Format MM:SS or HH:MM:SS range
            mmss_start = f"{s_int // 3600:02d}:{(s_int % 3600) // 60:02d}:{s_int % 60:02d}" if s_int >= 3600 else f"{(s_int % 3600) // 60:02d}:{s_int % 60:02d}"
            mmss_end = f"{e_int // 3600:02d}:{(e_int % 3600) // 60:02d}:{e_int % 60:02d}" if e_int >= 3600 else f"{(e_int % 3600) // 60:02d}:{e_int % 60:02d}"

            cid = str(
                chunk_record.get("course_id")
                or raw_match.get("course_id")
                or emb_info.get("metadata", {}).get("course_id")
                or "python_course"
            )
            lid = str(
                chunk_record.get("lecture_id")
                or raw_match.get("lecture_id")
                or emb_info.get("metadata", {}).get("lecture_id")
                or stem
            )
            ltitle = str(
                chunk_record.get("lecture_title")
                or raw_match.get("lecture_title")
                or emb_info.get("metadata", {}).get("lecture_title")
                or stem.replace("_", " ").title()
            )
            sfilename = str(
                chunk_record.get("source_filename")
                or raw_match.get("source_filename")
                or chunk_record.get("source_file")
                or top_source_file
            )
            t_start = str(chunk_record.get("timestamp_start") or raw_match.get("timestamp_start") or hhmmss_start)
            t_end = str(chunk_record.get("timestamp_end") or raw_match.get("timestamp_end") or hhmmss_end)
            t_range = str(chunk_record.get("timestamp") or raw_match.get("timestamp") or f"{mmss_start} - {mmss_end}")

            row = {
                "chunk_id": chunk_id,
                "text": text,
                "source_file": str(chunk_record.get("source_file", top_source_file)),
                "source_transcript": str(chunk_record.get("source_transcript", f"{stem}.json")),
                "language": str(chunk_record.get("language", "en")),
                "start_time": start_t,
                "end_time": end_t,
                "word_count": int(chunk_record.get("word_count", len(text.split()))),
                "segment_start_id": seg_start,
                "segment_end_id": seg_end,
                "embedding_model": model_name,
                "embedding": vector,
                # Module 10 Multi-Lecture Metadata
                "course_id": cid,
                "lecture_id": lid,
                "lecture_title": ltitle,
                "source_filename": sfilename,
                "timestamp_start": t_start,
                "timestamp_end": t_end,
                "timestamp": t_range,
            }
            rows.append(row)

    if not rows:
        raise DatasetBuilderError("Zero valid chunk records found across provided datasets.")

    df = pd.DataFrame(rows)
    return df


def validate_dataset(
    df: pd.DataFrame,
    expected_dim: int | None = None,
) -> dict[str, Any]:
    """
    Validates data integrity, alignment, and schema conformance of a consolidated DataFrame.

    Checks:
      1. Instance is a non-empty DataFrame.
      2. All required columns are present.
      3. Number of chunks == number of embeddings.
      4. Chunk IDs are strictly unique (no duplicate IDs).
      5. No empty or whitespace-only text segments.
      6. Timestamps are valid finite numbers with end_time >= start_time >= 0.
      7. All embeddings are 1D NumPy arrays with finite floats (no NaN or Inf).
      8. Embedding dimensions are consistent across all rows.

    Args:
        df: Pandas DataFrame to validate.
        expected_dim: Optional expected embedding dimension to enforce.

    Returns:
        Summary dictionary containing row count, column list, and dimension.

    Raises:
        DatasetBuilderError: If any validation rule is violated.
    """
    if not isinstance(df, pd.DataFrame):
        raise DatasetBuilderError(f"Expected a pandas DataFrame, got {type(df).__name__}.")

    if df.empty:
        raise DatasetBuilderError("Dataset validation failed: DataFrame is empty (0 rows).")

    # 1. Required columns check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetBuilderError(f"Dataset validation failed: missing required column(s): {missing_cols}")

    row_count = len(df)

    # 2. Number of chunks == number of embeddings
    if len(df["embedding"]) != row_count:
        raise DatasetBuilderError(
            f"Dataset validation failed: row count ({row_count}) does not match "
            f"embedding count ({len(df['embedding'])})."
        )

    # 3. Unique chunk_id check
    if not df["chunk_id"].is_unique:
        duplicates = df[df["chunk_id"].duplicated(keep=False)]["chunk_id"].tolist()
        raise DatasetBuilderError(
            f"Dataset validation failed: found duplicate chunk_id values: {duplicates[:10]}"
        )

    # 4. Non-empty text check
    empty_text_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    if empty_text_mask.any():
        bad_ids = df[empty_text_mask]["chunk_id"].tolist()
        raise DatasetBuilderError(
            f"Dataset validation failed: found {empty_text_mask.sum()} chunk(s) with empty text: {bad_ids[:5]}"
        )

    # 5. Timestamp validation
    if (df["start_time"] < 0).any():
        bad_starts = df[df["start_time"] < 0]["chunk_id"].tolist()
        raise DatasetBuilderError(f"Dataset validation failed: negative start_time in chunks: {bad_starts[:5]}")

    if (df["end_time"] < df["start_time"]).any():
        inverted = df[df["end_time"] < df["start_time"]]["chunk_id"].tolist()
        raise DatasetBuilderError(f"Dataset validation failed: end_time < start_time in chunks: {inverted[:5]}")

    # 6. Embedding vector validation
    first_vec = df["embedding"].iloc[0]
    if not isinstance(first_vec, np.ndarray):
        raise DatasetBuilderError(
            f"Dataset validation failed: embedding column must contain NumPy arrays, got {type(first_vec).__name__}."
        )

    detected_dim = first_vec.shape[0] if first_vec.ndim == 1 else first_vec.shape[-1]
    if expected_dim is not None and detected_dim != expected_dim:
        raise DatasetBuilderError(
            f"Dataset validation failed: expected dimension {expected_dim}, but detected {detected_dim}."
        )

    for idx, (cid, vec) in enumerate(zip(df["chunk_id"], df["embedding"])):
        if not isinstance(vec, np.ndarray):
            raise DatasetBuilderError(f"Chunk '{cid}' at row {idx} has non-numpy embedding: {type(vec).__name__}.")

        if vec.ndim != 1:
            raise DatasetBuilderError(f"Chunk '{cid}' at row {idx} has vector with ndim={vec.ndim} (expected 1D).")

        if vec.shape[0] != detected_dim:
            raise DatasetBuilderError(
                f"Dimension mismatch at row {idx} (chunk '{cid}'): "
                f"expected {detected_dim}, got {vec.shape[0]}."
            )

        if not np.all(np.isfinite(vec)):
            raise DatasetBuilderError(f"Chunk '{cid}' at row {idx} contains NaN or Infinite values in embedding.")

    summary = {
        "valid": True,
        "row_count": row_count,
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "embedding_dimension": detected_dim,
        "source_files": df["source_file"].unique().tolist(),
        "source_counts": df["source_file"].value_counts().to_dict(),
    }
    return summary


def save_dataset(
    df: pd.DataFrame,
    dataset_path: Path | str = CONSOLIDATED_DATASET_PATH,
    overwrite: bool = True,
) -> Path:
    """
    Validates and serializes the consolidated DataFrame to disk using Joblib.

    Args:
        df: Pandas DataFrame containing chunks and embeddings.
        dataset_path: Target path for the .joblib file (defaults to config path).
        overwrite: If False, raises error if dataset already exists.

    Returns:
        Path to the saved Joblib file.

    Raises:
        DatasetBuilderError: If validation fails or saving encounters errors.
    """
    target = Path(dataset_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and target.exists():
        raise DatasetBuilderError(
            f"Dataset file already exists at '{target}'. Use overwrite=True to replace."
        )

    # 1. Run strict validation before writing to disk
    validate_dataset(df)

    # 2. Serialize DataFrame using Joblib (compress=3 provides balanced speed/disk ratio)
    try:
        joblib.dump(df, target, compress=3)
        logger.info("Successfully serialized dataset to: %s (%d rows)", target, len(df))
        return target
    except Exception as exc:
        raise DatasetBuilderError(f"Failed to serialize dataset to '{target}': {exc}") from exc


def load_dataset(
    dataset_path: Path | str = CONSOLIDATED_DATASET_PATH,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Loads and optionally validates a consolidated DataFrame from a Joblib file.

    Args:
        dataset_path: Path to the .joblib dataset file.
        validate: If True, executes schema and vector validation on the loaded DataFrame.

    Returns:
        Deserialized Pandas DataFrame.

    Raises:
        DatasetBuilderError: If file is missing, corrupt, or fails validation.
    """
    target = Path(dataset_path)
    if not target.exists():
        raise DatasetBuilderError(
            f"Consolidated dataset not found at '{target}'. "
            "Please run 'python scripts/build_dataset.py' to generate it."
        )

    try:
        obj = joblib.load(target)
    except Exception as exc:
        raise DatasetBuilderError(f"Failed to deserialize Joblib dataset from '{target}': {exc}") from exc

    if not isinstance(obj, pd.DataFrame):
        raise DatasetBuilderError(
            f"Loaded Joblib object is not a pandas DataFrame: got {type(obj).__name__}."
        )

    if validate:
        validate_dataset(obj)

    logger.debug("Loaded dataset from '%s': %d rows, %d columns.", target, len(obj), len(obj.columns))
    return obj


def build_dataset(
    chunks_dir: Path | str = CHUNK_DIR,
    embeddings_dir: Path | str = EMBEDDING_DIR,
    dataset_path: Path | str = CONSOLIDATED_DATASET_PATH,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    High-level orchestrator: loads chunks, loads embeddings, merges, validates, and persists.

    Args:
        chunks_dir: Directory containing chunk JSON files.
        embeddings_dir: Directory containing embedding archives and metadata.
        dataset_path: Destination path for the output Joblib file.
        overwrite: Whether to overwrite existing file.

    Returns:
        Constructed and saved Pandas DataFrame.

    Raises:
        DatasetBuilderError: If any pipeline stage fails.
    """
    logger.info("Loading chunk files from: %s", chunks_dir)
    chunks_map = load_chunk_data(chunks_dir)

    logger.info("Loading embedding archives from: %s", embeddings_dir)
    embeddings_map = load_embedding_data(embeddings_dir)

    logger.info("Merging chunks and embeddings into DataFrame...")
    df = merge_chunks_and_embeddings(chunks_map, embeddings_map)

    logger.info("Validating consolidated DataFrame...")
    val_info = validate_dataset(df)
    logger.info(
        "Validation passed: %d rows, dimension %d, sources: %s",
        val_info["row_count"],
        val_info["embedding_dimension"],
        list(val_info["source_counts"].keys()),
    )

    logger.info("Saving dataset to: %s", dataset_path)
    save_dataset(df, dataset_path, overwrite=overwrite)

    return df
