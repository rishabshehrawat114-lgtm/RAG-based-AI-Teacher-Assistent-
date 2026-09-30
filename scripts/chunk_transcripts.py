"""
scripts/chunk_transcripts.py
============================
Command-line execution script for batch chunking processed educational transcripts with timestamps and metadata.

Usage:
    python scripts/chunk_transcripts.py
    python scripts/chunk_transcripts.py --file "data/transcripts/processed/python_lecture.json"
    python scripts/chunk_transcripts.py --overwrite
    python scripts/chunk_transcripts.py --chunk-size 700 --chunk-overlap 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CHUNK_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PROCESSED_TRANSCRIPT_DIR,
)
from src.chunking import (
    batch_chunk_transcripts,
    chunk_transcript_file,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for transcript chunking."""
    parser = argparse.ArgumentParser(
        description="Chunk processed educational transcripts into timestamped text chunks (Module 5)."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional path to an individual processed transcript JSON file to chunk.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_TRANSCRIPT_DIR,
        help=f"Directory containing processed transcripts (default: {PROCESSED_TRANSCRIPT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CHUNK_DIR,
        help=f"Directory to save chunk JSON files (default: {CHUNK_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Target words per chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Target overlapping words between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-generate chunks even if output chunk files already exist.",
    )
    return parser.parse_args()


def process_single_file(
    file_path: Path,
    output_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    overwrite: bool,
) -> int:
    """Chunks an explicitly specified single transcript file."""
    source = file_path.resolve()
    target_file = output_dir / f"{source.stem}_chunks.json"

    print(f"Source Transcript: {source}")
    print(f"Target Chunks    : {target_file}")
    print(f"Chunk Size       : {chunk_size} words")
    print(f"Chunk Overlap    : {chunk_overlap} words")
    print(f"Overwrite Mode   : {'ENABLED' if overwrite else 'DISABLED'}")
    print("-" * 65)

    if not source.exists():
        print(f"[ERROR] Processed transcript does not exist: {source}")
        return 1

    if target_file.exists() and not overwrite:
        print(f"[SKIPPED] Chunk file already exists at '{target_file}'. Use --overwrite to re-generate.")
        return 0

    try:
        chunks_data, saved_path = chunk_transcript_file(
            input_path=source,
            output_path=target_file,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overwrite=overwrite,
        )
    except Exception as exc:
        print(f"[ERROR] Chunking failed: {exc}")
        return 1

    count = chunks_data.get("chunk_count", 0)
    print("\n[SUCCESS] Chunking complete!")
    print(f"Saved: {saved_path}")
    print(f"Source Audio  : {chunks_data.get('source_file')}")
    print(f"Total Chunks  : {count}")

    chunks = chunks_data.get("chunks", [])
    if chunks:
        print("\nChunk Preview (First up to 2):")
        for chunk in chunks[:2]:
            print(f"  [{chunk['chunk_id']}] {chunk['start_time']:6.2f}s -> {chunk['end_time']:6.2f}s ({chunk['word_count']} words, segments {chunk['segment_start_id']}-{chunk['segment_end_id']}):")
            preview_text = chunk['text'][:120] + ("..." if len(chunk['text']) > 120 else "")
            print(f"    \"{preview_text}\"")

    print("=" * 65)
    return 0


def main() -> int:
    """Main execution entry point for batch or single transcript chunking."""
    args = parse_arguments()

    print("=" * 65)
    print(" RAG AI Teacher Assistant - Transcript Chunking (Module 5)")
    print("=" * 65)

    # Validate parameters
    if args.chunk_size <= 0:
        print(f"[ERROR] --chunk-size must be greater than 0, got {args.chunk_size}")
        return 1
    if args.chunk_overlap < 0:
        print(f"[ERROR] --chunk-overlap cannot be negative, got {args.chunk_overlap}")
        return 1
    if args.chunk_overlap >= args.chunk_size:
        print(f"[ERROR] --chunk-overlap ({args.chunk_overlap}) must be strictly less than --chunk-size ({args.chunk_size})")
        return 1

    # Single-file mode
    if args.file is not None:
        return process_single_file(
            file_path=args.file,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            overwrite=args.overwrite,
        )

    # Batch scan mode
    src_dir = args.input_dir.resolve()
    dst_dir = args.output_dir.resolve()

    if not src_dir.exists():
        print(f"[ERROR] Processed transcript directory does not exist: {src_dir}")
        return 1

    all_json = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() == ".json"]
    all_json.sort(key=lambda p: p.name.lower())

    print(f"Found {len(all_json)} transcript files.\n")

    if not all_json:
        print("Finished.\n")
        print("Total transcripts: 0")
        print("Total chunks: 0")
        print("=" * 65)
        return 0

    chunked_count = 0
    skipped_count = 0
    failed_count = 0
    total_chunks = 0

    for idx, trans_file in enumerate(all_json, start=1):
        target_chunk_file = dst_dir / f"{trans_file.stem}_chunks.json"
        print(f"[{idx}/{len(all_json)}] Chunking {trans_file.name}")

        if target_chunk_file.exists() and not args.overwrite:
            print("Skipped: already chunked.\n")
            skipped_count += 1
            continue

        try:
            chunks_data, _saved = chunk_transcript_file(
                input_path=trans_file,
                output_path=target_chunk_file,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                overwrite=True,
            )
            count = chunks_data.get("chunk_count", 0)
            print(f"Generated {count} chunks\n")
            chunked_count += 1
            total_chunks += count
        except Exception as exc:
            print(f"Failed to chunk {trans_file.name}: {exc}\n")
            failed_count += 1

    print("Finished.\n")
    print(f"Total transcripts: {len(all_json)}")
    print(f"Successfully processed: {chunked_count}")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    print(f"Total chunks: {total_chunks}")
    print("=" * 65)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
