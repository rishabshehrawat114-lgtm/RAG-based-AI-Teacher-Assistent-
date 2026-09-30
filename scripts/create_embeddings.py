#!/usr/bin/env python3
"""
scripts/create_embeddings.py
============================
CLI script to generate numerical dense vector embeddings for lecture transcript chunks.

Reads chunk JSON files from 'data/chunks/', loads the local Sentence Transformers
model (default: 'all-MiniLM-L6-v2'), generates normalized numerical vectors in batches,
and saves the results into 'data/embeddings/' as:
  - {lecture}_embeddings.npz  (compressed NumPy matrix)
  - {lecture}_metadata.json    (human-readable chunk metadata strictly aligned by row)

Usage:
------
# Process all chunk files in data/chunks/
python scripts/create_embeddings.py

# Overwrite existing embedding files
python scripts/create_embeddings.py --overwrite

# Process a specific chunk file
python scripts/create_embeddings.py --file data/chunks/python_lecture_chunks.json

# Use custom model and batch size
python scripts/create_embeddings.py --model all-MiniLM-L6-v2 --batch-size 64
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHUNK_DIR,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NORMALIZE_EMBEDDINGS,
    EMBEDDING_DIR,
)
from src.embeddings import (
    EmbeddingError,
    embed_chunk_file,
    get_embedding_dimension,
    load_embedding_model,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("create_embeddings")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the embedding generation CLI."""
    parser = argparse.ArgumentParser(
        description="Generate dense vector embeddings for lecture transcript chunks (Module 6)."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Path to a single chunk JSON file to embed. If omitted, embeds all files in chunks-dir.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=CHUNK_DIR,
        help=f"Directory containing input chunk JSON files (default: {CHUNK_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EMBEDDING_DIR,
        help=f"Directory to save generated embedding files (default: {EMBEDDING_DIR}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence Transformers embedding model name (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=f"Batch size for embedding generation (default: {DEFAULT_EMBEDDING_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable L2 normalization of embedding vectors (normalization enabled by default).",
    )
    parser.set_defaults(normalize=DEFAULT_NORMALIZE_EMBEDDINGS)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate and overwrite existing embedding files in the output directory.",
    )
    return parser.parse_args()


def main() -> int:
    """Main CLI entrypoint for batch chunk embedding."""
    args = parse_arguments()

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    # 1. Determine list of files to process
    if args.file:
        if not args.file.exists():
            print(f"Error: Specified chunk file not found: {args.file}", file=sys.stderr)
            return 1
        target_files = [args.file]
    else:
        if not args.chunks_dir.exists():
            print(f"Error: Chunks directory not found: {args.chunks_dir}", file=sys.stderr)
            return 1
        target_files = sorted(
            [f for f in args.chunks_dir.glob("*.json") if not f.name.startswith(".")]
        )

    if not target_files:
        print(f"No chunk files found to process in: {args.chunks_dir}")
        return 0

    print("==================================================")
    print("RAG AI Teacher Assistant — Module 6: Embeddings")
    print("==================================================")
    print(f"Found {len(target_files)} chunk file(s).")

    # Check which files actually require embedding
    pending_files = [
        f
        for f in target_files
        if args.overwrite
        or not (
            (args.output_dir / f"{f.stem.replace('_chunks', '')}_embeddings.npz").exists()
            and (args.output_dir / f"{f.stem.replace('_chunks', '')}_metadata.json").exists()
        )
    ]

    model = None
    if pending_files:
        print(f"Loading embedding model:\n{args.model}")
        try:
            model = load_embedding_model(args.model)
            dim = get_embedding_dimension(model)
            print(f"Model loaded successfully. Embedding dimension: {dim}")
            print(f"L2 Normalization: {'Enabled' if args.normalize else 'Disabled'}")
            print(f"Batch size: {args.batch_size}")
            print("--------------------------------------------------")
        except EmbeddingError as exc:
            print(f"\nError initializing embedding model: {exc}", file=sys.stderr)
            return 1
    else:
        print("All target files already have embeddings. Use --overwrite to recreate.")
        print("--------------------------------------------------")

    total_files = len(target_files)
    embedded_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks_embedded = 0

    # 3. Process each chunk file
    for idx, chunk_path in enumerate(target_files, start=1):
        clean_stem = chunk_path.stem.replace("_chunks", "")
        expected_npz = args.output_dir / f"{clean_stem}_embeddings.npz"
        expected_json = args.output_dir / f"{clean_stem}_metadata.json"

        print(f"\n[{idx}/{total_files}] {chunk_path.name}")

        if not args.overwrite and expected_npz.exists() and expected_json.exists():
            print("  Embeddings already exist. Skipping (use --overwrite to recreate).")
            skipped_count += 1
            continue

        try:
            # Read chunk count for terminal reporting
            with open(chunk_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            chunks_list = raw_data.get("chunks", [])
            print(f"Chunks: {len(chunks_list)}")
            print("Generating embeddings...")

            result = embed_chunk_file(
                chunk_file=chunk_path,
                output_dir=args.output_dir,
                model=model,
                model_name=args.model,
                batch_size=args.batch_size,
                normalize=args.normalize,
                overwrite=args.overwrite,
            )

            if result is not None:
                npz_p, json_p = result
                with open(json_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                num_embedded = meta.get("chunk_count", 0)
                total_chunks_embedded += num_embedded
                embedded_count += 1
                print("Saved embeddings.")
                print(f"  -> Vectors:  {npz_p}")
                print(f"  -> Metadata: {json_p}")
            else:
                skipped_count += 1
                print("  Skipped.")

        except Exception as exc:
            failed_count += 1
            print(f"  Error processing {chunk_path.name}: {exc}", file=sys.stderr)

    print("\n==================================================")
    print("Finished.")
    print("==================================================")
    print(f"Total chunk files:         {total_files}")
    print(f"Successfully embedded:     {embedded_count}")
    print(f"Skipped (already exists):  {skipped_count}")
    print(f"Failed:                    {failed_count}")
    print(f"Total chunks embedded:     {total_chunks_embedded}")
    print(f"Output directory:          {args.output_dir}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
