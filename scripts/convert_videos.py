"""
scripts/convert_videos.py
=========================
Command-line execution script for batch converting educational videos to MP3 audio files.

Usage:
    python scripts/convert_videos.py
    python scripts/convert_videos.py --overwrite
    python scripts/convert_videos.py --video-dir "path/to/videos" --output-dir "path/to/audio"
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audio import (
    FFmpegNotFoundError,
    SUPPORTED_VIDEO_EXTENSIONS,
    batch_convert_videos,
    get_ffmpeg_version,
    is_ffmpeg_available,
)
from src.config import AUDIO_DIR, VIDEO_DIR


def format_status_message(status_type: str, message: str) -> None:
    """Formats and prints batch conversion events to the console."""
    symbols = {
        "INFO": "[INFO] ",
        "PROCESSING": "[CONVERTING] ",
        "SUCCESS": "[SUCCESS] ",
        "SKIPPED": "[SKIPPED] ",
        "ERROR": "[ERROR] ",
    }
    prefix = symbols.get(status_type, "[-] ")
    print(f"{prefix}{message}")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the video conversion runner."""
    parser = argparse.ArgumentParser(
        description="Extract speech-optimized MP3 audio from educational videos using FFmpeg."
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=VIDEO_DIR,
        help=f"Directory containing source video files (default: {VIDEO_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIO_DIR,
        help=f"Directory to save output MP3 files (default: {AUDIO_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert videos even if corresponding MP3 files already exist.",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point for the video-to-audio conversion script."""
    args = parse_arguments()

    print("=" * 65)
    print(" RAG AI Teacher Assistant - Video to Audio Converter (Module 2)")
    print("=" * 65)

    # Pre-check FFmpeg availability before scanning
    if not is_ffmpeg_available():
        print("\n[ERROR] FFmpeg is not installed or not found in your system PATH.\n")
        print("FFmpeg is required to extract audio from video files.")
        print("\nInstallation instructions for Windows:")
        print("  Option 1 (winget):   winget install Gyan.FFmpeg")
        print("  Option 2 (choco):    choco install ffmpeg")
        print("  Option 3 (manual):   Download from https://ffmpeg.org/download.html")
        print("                       Extract the archive and add the 'bin' folder to your System PATH")
        print("\nAfter installing, open a new terminal window and verify with:")
        print("  ffmpeg -version")
        print("=" * 65)
        return 1

    ffmpeg_version = get_ffmpeg_version()
    print(f"Detected FFmpeg: {ffmpeg_version}")
    print(f"Source Directory: {args.video_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Supported Types : {', '.join(sorted(SUPPORTED_VIDEO_EXTENSIONS))}")
    print(f"Overwrite Mode  : {'ENABLED' if args.overwrite else 'DISABLED (skip existing)'}")
    print("-" * 65)

    try:
        results = batch_convert_videos(
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            status_callback=format_status_message,
        )
    except FFmpegNotFoundError as err:
        print(f"\n[ERROR] {err}")
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Unexpected error during batch conversion: {exc}")
        return 1

    # Print summary report
    print("\n" + "=" * 65)
    print(" Batch Conversion Summary Report")
    print("=" * 65)
    print(f"  * Videos found           : {results['total_found']}")
    print(f"  * Successfully converted : {len(results['converted'])}")
    print(f"  * Skipped (already exist): {len(results['skipped'])}")
    print(f"  * Failed                 : {len(results['failed'])}")
    print(f"  * Output location        : {results['output_dir']}")

    if results["converted"]:
        print("\nConverted Files:")
        for item in results["converted"]:
            print(f"  - {item.name}")

    if results["failed"]:
        print("\nFailed Files:")
        for failed_file, err_msg in results["failed"]:
            print(f"  - {failed_file.name}: {err_msg}")

    print("=" * 65)
    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
