"""
src/audio/extractor.py
======================
Audio extraction module using FFmpeg.

Converts educational lecture and tutorial videos (.mp4, .mkv, .mov, .avi, .webm)
into compact MP3 audio files optimized for speech transcription with Whisper.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Union

from src.config import AUDIO_DIR, VIDEO_DIR
from src.utils import dir_exists, ensure_dir, file_exists

# Supported video extensions for ingestion
SUPPORTED_VIDEO_EXTENSIONS: set[str] = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

# Audio settings optimized for speech transcription (Whisper downsamples to 16kHz mono)
DEFAULT_SAMPLE_RATE: int = 16000   # 16 kHz sample rate (optimal for Whisper speech-to-text)
DEFAULT_CHANNELS: int = 1          # Mono channel (reduces file size by 50% without quality loss for speech)
DEFAULT_AUDIO_BITRATE: str = "64k" # 64 kbps MP3 bitrate (lightweight, voice-optimized)


class FFmpegNotFoundError(FileNotFoundError):
    """Raised when FFmpeg executable cannot be found in the system PATH."""
    pass


class AudioExtractionError(RuntimeError):
    """Raised when FFmpeg fails during the video-to-audio conversion process."""
    pass


def is_ffmpeg_available() -> bool:
    """
    Checks whether the FFmpeg executable is installed and reachable via system PATH.

    Returns:
        bool: True if FFmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def get_ffmpeg_version() -> Optional[str]:
    """
    Retrieves the first line of the FFmpeg version string.

    Returns:
        Optional[str]: FFmpeg version string if found, None otherwise.
    """
    if not is_ffmpeg_available():
        return None
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else ""
        return first_line.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def check_ffmpeg_installed() -> None:
    """
    Verifies FFmpeg presence. Raises FFmpegNotFoundError with clear installation guidance if missing.

    Raises:
        FFmpegNotFoundError: If FFmpeg is not found in PATH.
    """
    if not is_ffmpeg_available():
        error_message = (
            "FFmpeg was not found in your system PATH.\n"
            "FFmpeg is required to extract audio from educational videos.\n\n"
            "How to install FFmpeg:\n"
            "  - Windows (winget):   winget install Gyan.FFmpeg\n"
            "  - Windows (choco):    choco install ffmpeg\n"
            "  - Windows (manual):   Download from https://ffmpeg.org/download.html and add the 'bin' folder to PATH\n"
            "  - macOS (Homebrew):   brew install ffmpeg\n"
            "  - Linux (Ubuntu/Deb): sudo apt update && sudo apt install -y ffmpeg\n\n"
            "After installing, open a new terminal window and verify with:\n"
            "  ffmpeg -version"
        )
        raise FFmpegNotFoundError(error_message)


def convert_video_to_mp3(
    video_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    bitrate: str = DEFAULT_AUDIO_BITRATE,
) -> Path:
    """
    Converts a single educational video file to an MP3 audio file using FFmpeg.

    Audio parameters are tailored for speech recognition:
    - 16,000 Hz sample rate (matches Whisper's internal representation)
    - 1 channel (mono)
    - 64 kbps bitrate to preserve disk space while maintaining speech clarity

    Args:
        video_path: Path to the source video file (.mp4, .mkv, .mov, etc.).
        output_path: Destination path for the MP3. If None, saves to AUDIO_DIR with the same stem.
        overwrite: If True, replaces existing MP3 file. If False, raises FileExistsError.
        sample_rate: Audio sampling frequency in Hz (default: 16000).
        channels: Number of audio channels (1 for mono, 2 for stereo; default: 1).
        bitrate: Audio encoding bitrate (default: '64k').

    Returns:
        Path: Path to the generated MP3 audio file.

    Raises:
        FFmpegNotFoundError: If FFmpeg is not detected in system PATH.
        FileNotFoundError: If the source video file does not exist.
        ValueError: If the source video file has an unsupported format.
        FileExistsError: If output file exists and overwrite=False.
        AudioExtractionError: If FFmpeg returns a non-zero exit code during conversion.
    """
    check_ffmpeg_installed()

    source = Path(video_path).resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")

    if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported_list = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(
            f"Unsupported video format '{source.suffix}'. Supported formats are: {supported_list}"
        )

    # Resolve output MP3 path
    if output_path is None:
        target = AUDIO_DIR / f"{source.stem}.mp3"
    else:
        target = Path(output_path).resolve()

    # Create destination parent directory if missing
    ensure_dir(target.parent)

    # Check if target already exists and overwrite flag is False
    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Target audio file already exists: {target}. Set overwrite=True to replace it."
        )

    # Assemble FFmpeg command line
    # -y: force overwrite if specified, -n: fail if file exists
    # -i: input video path
    # -vn: disable video recording/processing (audio stream only)
    # -acodec libmp3lame: MP3 encoder
    # -ar: audio sampling rate (Hz)
    # -ac: audio channels (1 = mono)
    # -b:a: audio bitrate
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(source),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-b:a", str(bitrate),
        str(target),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as err:
        raise AudioExtractionError(f"Failed to execute FFmpeg process: {err}") from err

    if result.returncode != 0:
        # FFmpeg writes error and diagnostic information to stderr
        error_details = result.stderr.strip().splitlines()
        last_errors = "\n".join(error_details[-5:]) if error_details else "Unknown error"
        raise AudioExtractionError(
            f"FFmpeg conversion failed for '{source.name}' (exit code {result.returncode}):\n{last_errors}"
        )

    return target


def batch_convert_videos(
    video_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    bitrate: str = DEFAULT_AUDIO_BITRATE,
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> dict:
    """
    Scans the video directory for supported video files, converts each to an MP3,
    and returns a summary report. Continues processing even if an individual video fails.

    Args:
        video_dir: Directory containing input video files (defaults to config.VIDEO_DIR).
        output_dir: Directory where MP3s will be saved (defaults to config.AUDIO_DIR).
        overwrite: Whether to reconvert videos whose MP3 already exists.
        sample_rate: Target audio sample rate in Hz.
        channels: Target audio channels (1=mono, 2=stereo).
        bitrate: Target audio bitrate.
        status_callback: Optional callable taking (status_type, message) for reporting progress.

    Returns:
        dict with keys:
            - 'total_found': int
            - 'converted': list[Path]
            - 'skipped': list[Path]
            - 'failed': list[tuple[Path, str]]
            - 'output_dir': Path
    """
    check_ffmpeg_installed()

    src_dir = Path(video_dir or VIDEO_DIR).resolve()
    dst_dir = Path(output_dir or AUDIO_DIR).resolve()
    ensure_dir(dst_dir)

    def notify(status: str, msg: str) -> None:
        if status_callback:
            status_callback(status, msg)

    # Discover supported video files
    video_files: list[Path] = []
    if dir_exists(src_dir):
        for item in src_dir.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                video_files.append(item)

    # Sort files by name for deterministic order
    video_files.sort(key=lambda p: p.name.lower())

    results: dict = {
        "total_found": len(video_files),
        "converted": [],
        "skipped": [],
        "failed": [],
        "output_dir": dst_dir,
    }

    notify("INFO", f"Found {len(video_files)} supported video file(s) in '{src_dir}'.")

    for idx, video in enumerate(video_files, start=1):
        target_mp3 = dst_dir / f"{video.stem}.mp3"

        # Check for existing MP3
        if target_mp3.exists() and not overwrite:
            results["skipped"].append(target_mp3)
            notify(
                "SKIPPED",
                f"[{idx}/{len(video_files)}] Skipped: '{video.name}' -> '{target_mp3.name}' already exists.",
            )
            continue

        notify(
            "PROCESSING",
            f"[{idx}/{len(video_files)}] Converting: '{video.name}' -> '{target_mp3.name}'...",
        )

        try:
            output_file = convert_video_to_mp3(
                video_path=video,
                output_path=target_mp3,
                overwrite=overwrite,
                sample_rate=sample_rate,
                channels=channels,
                bitrate=bitrate,
            )
            results["converted"].append(output_file)
            notify("SUCCESS", f"[{idx}/{len(video_files)}] Converted: '{video.name}' successfully.")
        except Exception as exc:
            # Gracefully record error without halting the batch
            results["failed"].append((video, str(exc)))
            notify("ERROR", f"[{idx}/{len(video_files)}] Failed: '{video.name}' - {exc}")

    return results
