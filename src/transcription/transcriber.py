"""
src/transcription/transcriber.py
================================
Speech-to-text transcription engine using OpenAI Whisper.

Transcribes lecture and tutorial audio files (.mp3, .wav, .m4a) and generates
structured JSON transcripts with millisecond-precision timestamps (start, end)
for every speech segment, enabling downstream RAG retrieval to cite exact video timecodes.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Union

from src.config import (
    AUDIO_DIR,
    DEFAULT_WHISPER_MODEL,
    SUPPORTED_AUDIO_EXTENSIONS,
    TRANSCRIPT_DIR,
)
from src.utils import dir_exists, ensure_dir, file_exists

# Cache for loaded Whisper model instances across repeated transcription calls
_LOADED_MODELS: dict[str, Any] = {}


class WhisperDependencyError(ImportError):
    """Raised when OpenAI Whisper or required backends are not installed or cannot be loaded."""
    pass


class TranscriptionError(RuntimeError):
    """Raised when audio transcription fails during processing."""
    pass


def is_ffmpeg_available() -> bool:
    """
    Checks whether FFmpeg is reachable on system PATH.
    Whisper invokes FFmpeg under the hood to decode and resample audio streams.

    Returns:
        bool: True if FFmpeg is available, False otherwise.
    """
    return shutil.which("ffmpeg") is not None


def check_dependencies() -> None:
    """
    Verifies that both OpenAI Whisper and FFmpeg are available.

    Raises:
        WhisperDependencyError: If openai-whisper or torch cannot be imported.
        TranscriptionError: If FFmpeg is missing from system PATH.
    """
    try:
        import whisper  # noqa: F401
    except (ImportError, ModuleNotFoundError) as err:
        raise WhisperDependencyError(
            "OpenAI Whisper is not installed in the current Python environment.\n"
            "Install it via:\n"
            "  pip install openai-whisper"
        ) from err

    if not is_ffmpeg_available():
        raise TranscriptionError(
            "FFmpeg was not found in your system PATH.\n"
            "Whisper requires FFmpeg to load and process audio files.\n\n"
            "Installation options for Windows:\n"
            "  - winget: winget install Gyan.FFmpeg\n"
            "  - choco:  choco install ffmpeg\n"
            "  - manual: Download from https://ffmpeg.org/download.html and add the 'bin' folder to PATH."
        )


def get_whisper_model(model_name: Optional[str] = None) -> Any:
    """
    Loads and caches a Whisper model instance by name.
    Reuses cached models across calls to avoid costly repeated disk reads and weight initialization.

    Args:
        model_name: Name of the Whisper model ('tiny', 'base', 'small', 'medium', 'large').
                    Defaults to DEFAULT_WHISPER_MODEL configured in src/config.py.

    Returns:
        whisper.Whisper: The loaded Whisper model instance.

    Raises:
        WhisperDependencyError: If Whisper is not installed.
        TranscriptionError: If the model fails to load.
    """
    check_dependencies()
    import whisper

    target_name = (model_name or DEFAULT_WHISPER_MODEL).strip().lower()

    if target_name in _LOADED_MODELS:
        return _LOADED_MODELS[target_name]

    try:
        model = whisper.load_model(target_name)
        _LOADED_MODELS[target_name] = model
        return model
    except Exception as exc:
        available = whisper.available_models()
        raise TranscriptionError(
            f"Failed to load Whisper model '{target_name}': {exc}\n"
            f"Available models are: {', '.join(available)}"
        ) from exc


def format_transcription_result(
    audio_path: Union[str, Path],
    raw_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Converts raw Whisper output into a clean, standardized JSON-compatible structure
    containing segment timestamps, detected language, and duration.

    Args:
        audio_path: Path to the transcribed audio source file.
        raw_result: Raw dictionary returned by whisper.transcribe().

    Returns:
        dict: Standardized transcription structure:
            {
                "source_file": "lecture.mp3",
                "language": "en",
                "duration": 3600.5,
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 5.42,
                        "text": "Today we are going to study machine learning."
                    },
                    ...
                ]
            }
    """
    source_name = Path(audio_path).name
    language = raw_result.get("language", "unknown")
    raw_segments = raw_result.get("segments", [])

    clean_segments: list[dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments):
        start_time = round(float(seg.get("start", 0.0)), 2)
        end_time = round(float(seg.get("end", 0.0)), 2)
        text_content = str(seg.get("text", "")).strip()

        clean_segments.append({
            "id": int(seg.get("id", idx)),
            "start": start_time,
            "end": end_time,
            "text": text_content,
        })

    # Duration calculated from final segment timestamp, or 0.0 if audio has no detected segments
    duration = clean_segments[-1]["end"] if clean_segments else 0.0

    return {
        "source_file": source_name,
        "language": language,
        "duration": duration,
        "segments": clean_segments,
    }


def save_transcript(
    transcript_data: dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Saves a structured transcript dictionary to a JSON file.

    Args:
        transcript_data: Standardized transcript dictionary.
        output_path: Target JSON file path or destination directory.

    Returns:
        Path: Path to the saved JSON file.

    Raises:
        ValueError: If transcript_data is invalid or missing required keys.
        OSError: If writing to the destination fails.
    """
    target = Path(output_path).resolve()

    # If target is a directory or has no extension, place a <stem>.json inside it
    if target.is_dir() or target.suffix.lower() != ".json":
        source_name = transcript_data.get("source_file", "transcript")
        stem = Path(source_name).stem
        target = target / f"{stem}.json"

    ensure_dir(target.parent)

    with open(target, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, indent=2, ensure_ascii=False)

    return target


def load_transcript(transcript_path: Union[str, Path]) -> dict[str, Any]:
    """
    Loads and parses a transcript JSON file.

    Args:
        transcript_path: Path to the transcript .json file.

    Returns:
        dict: Parsed transcript dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    target = Path(transcript_path).resolve()
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Transcript file not found: {target}")

    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def transcribe_audio(
    audio_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    language: Optional[str] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Transcribes a single audio file (.mp3, .wav, .m4a) using OpenAI Whisper.

    Preserves segment timestamps (start, end in seconds) and detected language.
    Optionally writes the structured transcript to a JSON file if output_path is provided.

    Args:
        audio_path: Path to the input audio file.
        output_path: Optional destination path (JSON file or directory). If provided, saves the output.
        model_name: Whisper model to use ('tiny', 'base', 'small', 'medium', 'large').
                    Defaults to config.DEFAULT_WHISPER_MODEL ('base').
        language: Optional language code (e.g. 'en', 'hi') to force transcription language.
                  Defaults to None for automatic language detection.
        verbose: Whether to print Whisper's internal segment output to console. Default False.

    Returns:
        dict: Structured transcript data with timestamps, language, duration, and segments.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio format is unsupported.
        WhisperDependencyError: If Whisper or torch dependencies are missing.
        TranscriptionError: If FFmpeg is missing or Whisper fails during transcription.
    """
    source = Path(audio_path).resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Audio file not found: {source}")

    if source.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(
            f"Unsupported audio format '{source.suffix}'. Supported formats are: {supported_str}"
        )

    # Validate FFmpeg and load model
    check_dependencies()
    model = get_whisper_model(model_name)

    # Determine CPU vs GPU for fp16 precision (prevents UserWarning on CPU)
    import torch
    is_cuda = torch.cuda.is_available()

    try:
        # Transcribe audio file with automatic language detection unless explicitly forced
        transcribe_args: dict[str, Any] = {
            "audio": str(source),
            "fp16": is_cuda,
            "verbose": verbose,
        }
        if language:
            transcribe_args["language"] = language

        raw_result = model.transcribe(**transcribe_args)
    except Exception as exc:
        raise TranscriptionError(
            f"Transcription failed for '{source.name}': {exc}"
        ) from exc

    formatted = format_transcription_result(source, raw_result)

    # Optionally persist transcript JSON
    if output_path is not None:
        save_transcript(formatted, output_path)

    return formatted


def batch_transcribe_audio(
    audio_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    model_name: Optional[str] = None,
    overwrite: bool = False,
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """
    Scans the audio directory for supported audio files, transcribes each file with Whisper,
    and saves corresponding JSON transcripts to the output directory.

    Continues processing even if an individual audio file fails.
    Skips files that already have an existing transcript unless overwrite=True.

    Args:
        audio_dir: Directory containing audio files (defaults to config.AUDIO_DIR).
        output_dir: Directory where JSON transcripts will be saved (defaults to config.TRANSCRIPT_DIR).
        model_name: Whisper model to use ('tiny', 'base', 'small', 'medium', 'large').
        overwrite: If True, re-transcribes audio even if a JSON transcript already exists.
        status_callback: Optional callback callable(status_type, message) for reporting progress.

    Returns:
        dict: Summary of batch transcription:
            {
                "total_found": int,
                "transcribed": list[Path],
                "skipped": list[Path],
                "failed": list[tuple[Path, str]],
                "output_dir": Path,
                "model_name": str,
            }
    """
    check_dependencies()

    src_dir = Path(audio_dir or AUDIO_DIR).resolve()
    dst_dir = Path(output_dir or TRANSCRIPT_DIR).resolve()
    ensure_dir(dst_dir)

    selected_model_name = (model_name or DEFAULT_WHISPER_MODEL).strip().lower()

    def notify(status_type: str, msg: str) -> None:
        if status_callback:
            status_callback(status_type, msg)

    # Discover supported audio files
    audio_files: list[Path] = []
    if dir_exists(src_dir):
        for item in src_dir.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                audio_files.append(item)

    audio_files.sort(key=lambda p: p.name.lower())

    results: dict[str, Any] = {
        "total_found": len(audio_files),
        "transcribed": [],
        "skipped": [],
        "failed": [],
        "output_dir": dst_dir,
        "model_name": selected_model_name,
    }

    notify("INFO", f"Found {len(audio_files)} audio file(s) in '{src_dir}'.")

    if not audio_files:
        return results

    # Preload model once for the batch to avoid reloading per file
    notify("LOADING", f"Loading Whisper model '{selected_model_name}'...")
    try:
        get_whisper_model(selected_model_name)
    except Exception as exc:
        notify("ERROR", f"Failed to load Whisper model '{selected_model_name}': {exc}")
        for audio in audio_files:
            results["failed"].append((audio, str(exc)))
        return results

    for idx, audio in enumerate(audio_files, start=1):
        target_json = dst_dir / f"{audio.stem}.json"

        # Check if already transcribed
        if target_json.exists() and not overwrite:
            results["skipped"].append(target_json)
            notify(
                "SKIPPED",
                f"[{idx}/{len(audio_files)}] Skipped: '{audio.name}' -> '{target_json.name}' already exists.",
            )
            continue

        notify(
            "TRANSCRIBING",
            f"[{idx}/{len(audio_files)}] Transcribing: '{audio.name}'...",
        )

        try:
            transcript_data = transcribe_audio(
                audio_path=audio,
                output_path=target_json,
                model_name=selected_model_name,
            )
            results["transcribed"].append(target_json)

            lang = transcript_data.get("language", "unknown")
            duration = transcript_data.get("duration", 0.0)
            seg_count = len(transcript_data.get("segments", []))

            notify(
                "SUCCESS",
                f"[{idx}/{len(audio_files)}] Saved: '{target_json.name}' "
                f"(Language: {lang}, Duration: {duration}s, Segments: {seg_count})",
            )
        except Exception as exc:
            results["failed"].append((audio, str(exc)))
            notify(
                "ERROR",
                f"[{idx}/{len(audio_files)}] Failed: '{audio.name}' - {exc}",
            )

    return results
