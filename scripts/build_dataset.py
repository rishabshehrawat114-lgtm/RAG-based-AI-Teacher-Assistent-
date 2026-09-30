#!/usr/bin/env python3
"""
scripts/build_dataset.py
========================
CLI script to compile lecture chunks and numerical embeddings into a consolidated
Pandas DataFrame and serialize it to disk using Joblib (Module 8).

Pipelines:
----------
data/chunks/*_chunks.json
+
data/embeddings/*_embeddings.npz
↓
Pandas DataFrame (1 row per chunk)
↓
Strict Schema & Vector Validation
↓
Joblib Serialization (data/processed/teacher_embeddings.joblib)
↓
Fast verified load test

Usage:
------
python scripts/build_dataset.py
python scripts/build_dataset.py --output data/processed/custom_dataset.joblib
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHUNK_DIR,
    CONSOLIDATED_DATASET_PATH,
    EMBEDDING_DIR,
)
from src.data import (
    DatasetBuilderError,
    REQUIRED_COLUMNS,
    load_chunk_data,
    load_dataset,
    load_embedding_data,
    merge_chunks_and_embeddings,
    save_dataset,
    validate_dataset,
)

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for dataset builder CLI."""
    parser = argparse.ArgumentParser(
        description="Consolidate lecture chunks and embeddings into a Joblib dataset (Module 8)."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNK_DIR,
        help=f"Directory containing chunk JSON files (default: {CHUNK_DIR}).",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDING_DIR,
        help=f"Directory containing embedding archives (default: {EMBEDDING_DIR}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=CONSOLIDATED_DATASET_PATH,
        help=f"Destination path for serialized Joblib file (default: {CONSOLIDATED_DATASET_PATH}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing dataset if present (enabled by default).",
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI entrypoint."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        except Exception:
            pass

    args = parse_arguments()
    check = "✓" if sys.stdout.encoding and "utf" in sys.stdout.encoding.lower() else "[OK]"

    print("==================================================")
    print("RAG AI Teacher Assistant — Module 8: Dataset Builder")
    print("==================================================")

    # 1. Locate chunk and embedding files
    print("Loading chunk data...")
    try:
        chunks_map = load_chunk_data(args.chunks_dir)
        total_raw_chunks = sum(len(d.get("chunks", [])) for d in chunks_map.values())
        print(f"Found {total_raw_chunks} chunk(s) across {len(chunks_map)} file(s).")
    except DatasetBuilderError as exc:
        print(f"Error loading chunks: {exc}", file=sys.stderr)
        return 1

    print("\nLoading embeddings...")
    try:
        embeddings_map = load_embedding_data(args.embeddings_dir)
        total_embeddings = sum(e["embeddings"].shape[0] for e in embeddings_map.values())
        print(f"Found {total_embeddings} embedding(s) across {len(embeddings_map)} archive(s).")
    except DatasetBuilderError as exc:
        print(f"Error loading embeddings: {exc}", file=sys.stderr)
        return 1

    # 2. Build the Pandas DataFrame
    print("\nBuilding DataFrame...")
    try:
        df = merge_chunks_and_embeddings(chunks_map, embeddings_map)
    except DatasetBuilderError as exc:
        print(f"Error building DataFrame: {exc}", file=sys.stderr)
        return 1

    # 3. Validate the dataset
    print("\nValidation:")
    try:
        val_summary = validate_dataset(df)
        print(f"{check} Chunk count matches embedding count")
        print(f"{check} Unique chunk IDs")
        print(f"{check} Embedding dimensions consistent")
        print(f"{check} Metadata aligned")
    except DatasetBuilderError as exc:
        print(f"✗ Validation Failed: {exc}", file=sys.stderr)
        return 1

    # 4. Save DataFrame using Joblib
    try:
        saved_path = save_dataset(df, args.output, overwrite=args.overwrite)
    except DatasetBuilderError as exc:
        print(f"Error saving dataset: {exc}", file=sys.stderr)
        return 1

    dim = val_summary["embedding_dimension"]
    print("\nDataset created successfully.")
    print(f"Rows:                 {len(df)}")
    print(f"Columns:              {len(df.columns)}")
    print(f"Embedding dimension:  {dim}")
    print(f"\nSaved to:\n{saved_path}")

    # 5. Pandas Inspection
    print("\n--------------------------------------------------")
    print("DataFrame Inspection:")
    print("--------------------------------------------------")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSource File Distribution:")
    for source, count in df["source_file"].value_counts().items():
        print(f"  {source} -> {count}")

    print("\nSample Chunks:")
    sample_df = df[["chunk_id", "source_file", "start_time", "end_time", "word_count"]].head(3)
    print(sample_df.to_string(index=False))

    # 6. Joblib Verification (Load and verify)
    print("\n--------------------------------------------------")
    print("Joblib Verification:")
    print("--------------------------------------------------")
    try:
        loaded_df = load_dataset(saved_path, validate=True)
        assert len(loaded_df) == len(df), "Row count mismatch on reload"
        assert list(loaded_df.columns) == list(df.columns), "Column mismatch on reload"
        assert loaded_df["embedding"].iloc[0].shape[0] == dim, "Dimension mismatch on reload"
        print("Joblib save/load verification successful.")
    except Exception as exc:
        print(f"Joblib verification failed: {exc}", file=sys.stderr)
        return 1

    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
