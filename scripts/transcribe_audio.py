"""
scripts/transcribe_audio.py
===========================
Command-line execution script for batch transcribing lecture audio files to timestamped JSON transcripts.

Usage:
    python scripts/transcribe_audio.py
    python scripts/transcribe_audio.py --file "data/audio/python_lecture.mp3"
    python scripts/transcribe_audio.py --model tiny
    python scripts/transcribe_audio.py --overwrite
    python scripts/transcribe_audio.py --audio-dir "path/to/audio" --output-dir "path/to/transcripts"
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AUDIO_DIR, DEFAULT_WHISPER_MODEL, SUPPORTED_AUDIO_EXTENSIONS, TRANSCRIPT_DIR
from src.transcription import (
    SUPPORTED_AUDIO_EXTENSIONS,
    TranscriptionError,
    WhisperDependencyError,
    batch_transcribe_audio,
    check_dependencies,
    is_ffmpeg_available,
    transcribe_audio,
)


def format_status_message(status_type: str, message: str) -> None:
    """Formats and prints batch transcription events to the console."""
    symbols = {
        "INFO": "[INFO] ",
        "LOADING": "[MODEL] ",
        "TRANSCRIBING": "[TRANSCRIBING] ",
        "SUCCESS": "[SUCCESS] ",
        "SKIPPED": "[SKIPPED] ",
        "ERROR": "[ERROR] ",
    }
    prefix = symbols.get(status_type, "[-] ")
    print(f"{prefix}{message}")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the transcription runner."""
    parser = argparse.ArgumentParser(
        description="Transcribe educational audio files (.mp3, .wav, .m4a) to timestamped JSON using Whisper."
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Optional path to a single audio file to transcribe directly.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help=f"Directory containing audio files to transcribe (default: {AUDIO_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TRANSCRIPT_DIR,
        help=f"Directory to save output JSON transcripts (default: {TRANSCRIPT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model name: tiny, base, small, medium, large (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-transcribe audio files even if corresponding JSON transcripts already exist.",
    )
    return parser.parse_args()


def transcribe_single_file(audio_path: Path, output_dir: Path, model_name: str, overwrite: bool) -> int:
    """Handles transcription for an explicitly specified single audio file."""
    source = audio_path.resolve()
    target_json = output_dir / f"{source.stem}.json"

    print(f"Target Audio   : {source}")
    print(f"Target JSON    : {target_json}")
    print(f"Whisper Model  : {model_name}")
    print("-" * 65)

    if not source.exists():
        print(f"[ERROR] Audio file does not exist: {source}")
        return 1

    if target_json.exists() and not overwrite:
        print(f"[SKIPPED] Transcript already exists at '{target_json}'. Use --overwrite to re-generate.")
        return 0

    print("Loading Whisper model...")
    print("Transcribing...")
    try:
        result = transcribe_audio(
            audio_path=source,
            output_path=target_json,
            model_name=model_name,
        )
    except Exception as exc:
        print(f"[ERROR] Transcription failed: {exc}")
        return 1

    print("\n[SUCCESS] Transcription complete!")
    print(f"Saved: {target_json}")
    print(f"Detected Language : {result.get('language')}")
    print(f"Audio Duration    : {result.get('duration')}s")
    print(f"Speech Segments   : {len(result.get('segments', []))}")

    # Display short preview of first 3 segments if available
    segments = result.get("segments", [])
    if segments:
        print("\nSegment Preview (First up to 3):")
        for seg in segments[:3]:
            print(f"  [{seg['start']:6.2f}s -> {seg['end']:6.2f}s] {seg['text']}")

    print("=" * 65)
    return 0


def main() -> int:
    """Entry point for the Whisper audio transcription runner."""
    args = parse_arguments()

    print("=" * 65)
    print(" RAG AI Teacher Assistant - Whisper Transcription (Module 3)")
    print("=" * 65)

    # Dependency check for FFmpeg and Whisper
    if not is_ffmpeg_available():
        print("\n[ERROR] FFmpeg is not installed or not found in system PATH.")
        print("Whisper requires FFmpeg to decode audio streams.")
        print("\nInstallation on Windows:")
        print("  Option 1: winget install Gyan.FFmpeg")
        print("  Option 2: choco install ffmpeg")
        print("  Option 3: https://ffmpeg.org/download.html")
        print("=" * 65)
        return 1

    try:
        check_dependencies()
    except WhisperDependencyError as err:
        print(f"\n[ERROR] {err}")
        print("=" * 65)
        return 1

    # Single-file mode
    if args.file is not None:
        return transcribe_single_file(
            audio_path=args.file,
            output_dir=args.output_dir,
            model_name=args.model,
            overwrite=args.overwrite,
        )

    # Batch scan mode
    print(f"Whisper Model   : {args.model}")
    print(f"Source Directory: {args.audio_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Supported Types : {', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))}")
    print(f"Overwrite Mode  : {'ENABLED' if args.overwrite else 'DISABLED (skip existing)'}")
    print("-" * 65)

    try:
        results = batch_transcribe_audio(
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            model_name=args.model,
            overwrite=args.overwrite,
            status_callback=format_status_message,
        )
    except Exception as exc:
        print(f"\n[ERROR] Batch transcription process failed: {exc}")
        return 1

    # Print summary report matching required format
    print("\n" + "=" * 65)
    print(" Final summary:")
    print("=" * 65)
    print(f"Total files: {results['total_found']}")
    print(f"Successfully transcribed: {len(results['transcribed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Failed: {len(results['failed'])}")

    if results["transcribed"]:
        print("\nTranscribed JSON Files:")
        for item in results["transcribed"]:
            print(f"  - {item.name}")

    if results["failed"]:
        print("\nFailed Audio Files:")
        for failed_audio, err_msg in results["failed"]:
            print(f"  - {failed_audio.name}: {err_msg}")

    print("=" * 65)
    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
