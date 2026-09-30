"""
scripts/process_transcripts.py
==============================
Command-line execution script for batch validating, cleaning, and normalizing raw Whisper transcripts.

Usage:
    python scripts/process_transcripts.py
    python scripts/process_transcripts.py --file "data/transcripts/python_lecture.json"
    python scripts/process_transcripts.py --overwrite
    python scripts/process_transcripts.py --transcript-dir "path/to/transcripts" --output-dir "path/to/processed"
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_TRANSCRIPT_DIR, TRANSCRIPT_DIR
from src.transcription import (
    batch_process_transcripts,
    process_transcript_file,
)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for transcript processing."""
    parser = argparse.ArgumentParser(
        description="Clean, validate, and normalize raw Whisper transcripts into structured JSON (Module 4)."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional path to an individual transcript JSON file to process.",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=TRANSCRIPT_DIR,
        help=f"Directory containing raw transcript JSON files (default: {TRANSCRIPT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_TRANSCRIPT_DIR,
        help=f"Directory to save processed JSON transcripts (default: {PROCESSED_TRANSCRIPT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process transcripts even if corresponding processed JSON files already exist.",
    )
    return parser.parse_args()


def process_single_file(file_path: Path, output_dir: Path, overwrite: bool) -> int:
    """Processes a single transcript file and outputs progress and diagnostics."""
    source = file_path.resolve()
    target_file = output_dir / source.name

    print(f"Source Transcript: {source}")
    print(f"Target Output    : {target_file}")
    print(f"Overwrite Mode   : {'ENABLED' if overwrite else 'DISABLED'}")
    print("-" * 65)

    if not source.exists():
        print(f"[ERROR] Transcript file does not exist: {source}")
        return 1

    if target_file.exists() and not overwrite:
        print(f"[SKIPPED] Processed file already exists at '{target_file}'. Use --overwrite to re-process.")
        return 0

    try:
        data, stats, saved_path = process_transcript_file(
            input_path=source,
            output_path=target_file,
            overwrite=overwrite,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to process transcript: {exc}")
        return 1

    print("\n[SUCCESS] Processing complete!")
    print(f"Saved: {saved_path}")
    print(f"Source Audio      : {data.get('source_file')}")
    print(f"Language          : {data.get('language')}")
    print(f"Duration          : {data.get('duration')}s")
    print(f"Valid Segments    : {stats['valid_segments']}")
    print(f"Invalid Segments  : {stats['invalid_segments']}")

    segments = data.get("segments", [])
    if segments:
        print("\nSegment Preview (First up to 3):")
        for seg in segments[:3]:
            print(f"  [{seg['id']}] {seg['start']:6.2f}s -> {seg['end']:6.2f}s: {seg['text']}")

    print("=" * 65)
    return 0


def main() -> int:
    """Main execution function for batch or single transcript processing."""
    args = parse_arguments()

    print("=" * 65)
    print(" RAG AI Teacher Assistant - Transcript Processing (Module 4)")
    print("=" * 65)

    # Single-file mode
    if args.file is not None:
        return process_single_file(
            file_path=args.file,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )

    # Batch scan mode
    src_dir = args.transcript_dir.resolve()
    dst_dir = args.output_dir.resolve()

    if not src_dir.exists():
        print(f"[ERROR] Source directory does not exist: {src_dir}")
        return 1

    # Discover candidate files in input directory (ignoring subdirectories like 'processed/')
    all_json = [
        f for f in src_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".json" and f.resolve() != dst_dir and dst_dir not in f.parents
    ]
    all_json.sort(key=lambda p: p.name.lower())

    print(f"Found {len(all_json)} transcript files.\n")

    if not all_json:
        print("Final summary:\n")
        print("Total files: 0")
        print("Successfully processed: 0")
        print("Skipped: 0")
        print("Failed: 0")
        print("Total valid segments: 0")
        print("Total invalid segments: 0")
        print("=" * 65)
        return 0

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    total_valid = 0
    total_invalid = 0

    for idx, trans_file in enumerate(all_json, start=1):
        target_file = dst_dir / trans_file.name
        print(f"[{idx}/{len(all_json)}] Processing {trans_file.name}")

        if target_file.exists() and not args.overwrite:
            print("Skipped: already processed.")
            skipped_count += 1
            print()
            continue

        try:
            _, stats, _saved = process_transcript_file(
                input_path=trans_file,
                output_path=target_file,
                overwrite=True,
            )
            print(f"Valid segments: {stats['valid_segments']}")
            print(f"Invalid segments: {stats['invalid_segments']}")
            print("Saved processed transcript.")
            processed_count += 1
            total_valid += stats["valid_segments"]
            total_invalid += stats["invalid_segments"]
        except Exception as exc:
            print(f"Error processing file: {exc}")
            failed_count += 1

        print()

    # Final summary report matching required format
    print("Final summary:\n")
    print(f"Total files: {len(all_json)}")
    print(f"Successfully processed: {processed_count}")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Total valid segments: {total_valid:,}")
    print(f"Total invalid segments: {total_invalid:,}")
    print("=" * 65)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
