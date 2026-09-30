"""
src/transcription/processor.py
==============================
Transcript post-processing and structured data normalization module.

Ingests raw timestamped Whisper transcripts (.json), validates segment integrity,
conservatively cleans spoken text, normalizes numerical timestamps to millisecond precision,
re-indexes sequential segment IDs, and outputs clean structured JSON files ready for
Module 5 semantic chunking.
"""

import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Optional, Union

from src.config import PROCESSED_TRANSCRIPT_DIR, TRANSCRIPT_DIR
from src.utils import dir_exists, ensure_dir, file_exists


class TranscriptProcessingError(Exception):
    """Raised when an error occurs during transcript parsing or processing."""
    pass


def clean_segment_text(text: Optional[str]) -> str:
    """
    Performs conservative text cleaning on a spoken transcript segment.

    Cleaning rules:
    - Normalizes zero-width and non-breaking space characters.
    - Collapses consecutive whitespace characters (spaces, tabs, newlines) into a single space.
    - Strips leading and trailing whitespace.
    - Retains all original words, capitalization, punctuation, numbers, and accents.
    - Does NOT rewrite, summarize, or alter the speaker's original words.

    Args:
        text: Raw text string from the transcript segment.

    Returns:
        str: Cleaned text string.
    """
    if text is None:
        return ""

    raw = str(text)
    # Remove zero-width spaces / byte order marks and replace non-breaking spaces
    cleaned = raw.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")

    # Collapse consecutive whitespace characters into a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def is_numeric_value(val: Any) -> bool:
    """
    Checks if a value is a valid numeric scalar (int, float, or parseable numeric string)
    that is neither NaN nor infinite.

    Args:
        val: The value to test.

    Returns:
        bool: True if val represents a finite number, False otherwise.
    """
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    if isinstance(val, str):
        try:
            num = float(val.strip())
            return not (math.isnan(num) or math.isinf(num))
        except (ValueError, TypeError):
            return False
    return False


def validate_segment(segment: Any) -> tuple[bool, Optional[str], Optional[dict[str, Any]]]:
    """
    Validates the structure and content of an individual transcript segment.

    Validation rules:
    - Segment must be a dictionary.
    - Must contain 'start', 'end', and 'text' fields.
    - 'start' must be numeric, finite, and non-negative.
    - 'end' must be numeric, finite, and >= 'start'.
    - 'text' must not be empty after conservative whitespace normalization.

    Args:
        segment: Dictionary representing a raw transcript segment.

    Returns:
        tuple containing:
            - is_valid (bool): True if segment meets all criteria, False otherwise.
            - error_reason (Optional[str]): Description of rejection if invalid, else None.
            - normalized_segment (Optional[dict]): Normalized segment dict with cleaned text
              and numeric float timestamps rounded to 3 decimals, or None if invalid.
    """
    if not isinstance(segment, dict):
        return False, "Segment must be a dictionary object.", None

    if "start" not in segment or "end" not in segment:
        return False, "Segment missing required 'start' or 'end' timestamp keys.", None

    raw_start = segment.get("start")
    raw_end = segment.get("end")

    if not is_numeric_value(raw_start):
        return False, f"Start timestamp '{raw_start}' is not a valid numeric value.", None

    if not is_numeric_value(raw_end):
        return False, f"End timestamp '{raw_end}' is not a valid numeric value.", None

    start_num = float(raw_start)
    end_num = float(raw_end)

    if start_num < 0:
        return False, f"Start timestamp ({start_num}s) cannot be negative.", None

    if end_num < start_num:
        return (
            False,
            f"End timestamp ({end_num}s) cannot precede start timestamp ({start_num}s).",
            None,
        )

    # Clean text conservatively
    raw_text = segment.get("text", "")
    cleaned_text = clean_segment_text(raw_text)

    if not cleaned_text:
        return False, "Segment text is empty after whitespace normalization.", None

    # Normalized segment dictionary with millisecond-precision float timestamps
    normalized = {
        "start": round(start_num, 3),
        "end": round(end_num, 3),
        "text": cleaned_text,
    }

    return True, None, normalized


def process_transcript(
    raw_data: dict[str, Any],
    fallback_source_name: str = "transcript.mp3",
) -> tuple[dict[str, Any], dict[str, int]]:
    """
    Processes a raw transcript data structure by validating all segments,
    cleaning text, standardizing timestamps, assigning sequential IDs,
    and compiling structured metadata.

    Args:
        raw_data: Dictionary loaded from a Whisper JSON transcript file.
        fallback_source_name: Fallback filename if 'source_file' is missing in raw_data.

    Returns:
        tuple containing:
            - processed_data (dict): Standardized transcript format:
                {
                    "source_file": "machine_learning.mp3",
                    "language": "en",
                    "duration": 3600.5,
                    "segment_count": 250,
                    "segments": [
                        {
                            "id": 0,
                            "start": 0.0,
                            "end": 5.42,
                            "text": "Today we are going to study machine learning."
                        }, ...
                    ]
                }
            - stats (dict): Summary statistics:
                {
                    "valid_segments": int,
                    "invalid_segments": int,
                    "total_raw_segments": int,
                }

    Raises:
        TranscriptProcessingError: If raw_data is not a valid dictionary or lacks 'segments'.
    """
    if not isinstance(raw_data, dict):
        raise TranscriptProcessingError("Transcript data must be a dictionary object.")

    source_file = str(raw_data.get("source_file") or fallback_source_name).strip()
    language = str(raw_data.get("language") or "unknown").strip()
    raw_segments = raw_data.get("segments")

    if not isinstance(raw_segments, list):
        raise TranscriptProcessingError(
            f"Transcript '{source_file}' must contain a 'segments' list, found: {type(raw_segments).__name__}"
        )

    valid_segments: list[dict[str, Any]] = []
    invalid_count: int = 0

    for raw_seg in raw_segments:
        is_valid, _reason, clean_seg = validate_segment(raw_seg)
        if is_valid and clean_seg is not None:
            # Assign consistent sequential 0-indexed ID
            clean_seg["id"] = len(valid_segments)
            # Reorder keys to id, start, end, text for clean, predictable JSON output
            ordered_seg = {
                "id": clean_seg["id"],
                "start": clean_seg["start"],
                "end": clean_seg["end"],
                "text": clean_seg["text"],
            }
            valid_segments.append(ordered_seg)
        else:
            invalid_count += 1

    # Duration: derived from the end time of the final valid segment, or the original duration
    if valid_segments:
        derived_duration = valid_segments[-1]["end"]
    else:
        orig_dur = raw_data.get("duration", 0.0)
        derived_duration = round(float(orig_dur), 3) if is_numeric_value(orig_dur) else 0.0

    processed_data: dict[str, Any] = {
        "source_file": source_file,
        "language": language,
        "duration": derived_duration,
        "segment_count": len(valid_segments),
        "segments": valid_segments,
    }

    stats: dict[str, int] = {
        "valid_segments": len(valid_segments),
        "invalid_segments": invalid_count,
        "total_raw_segments": len(raw_segments),
    }

    return processed_data, stats


def save_processed_transcript(
    processed_data: dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Saves a processed transcript dictionary to a JSON file inside the specified destination.
    Guarantees the original raw Whisper transcript is never overwritten.

    Args:
        processed_data: Cleaned transcript dictionary.
        output_path: Destination JSON file path or directory.

    Returns:
        Path: Resolved Path to the saved JSON file.

    Raises:
        TranscriptProcessingError: If writing to the file fails.
    """
    target = Path(output_path).resolve()

    # If target is a directory or lacks a .json extension, derive filename from source_file
    if target.is_dir() or target.suffix.lower() != ".json":
        source_name = processed_data.get("source_file", "processed_transcript")
        stem = Path(source_name).stem
        target = target / f"{stem}.json"

    ensure_dir(target.parent)

    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
    except OSError as err:
        raise TranscriptProcessingError(f"Failed to write processed transcript to '{target}': {err}") from err

    return target


def process_transcript_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> tuple[dict[str, Any], dict[str, int], Path]:
    """
    Loads, validates, cleans, and optionally saves an individual transcript JSON file.

    Args:
        input_path: Path to the raw transcript JSON file.
        output_path: Destination path for the processed JSON (defaults to PROCESSED_TRANSCRIPT_DIR / file.name).
        overwrite: If True, overwrites existing processed JSON file. If False, raises FileExistsError.

    Returns:
        tuple containing:
            - processed_data (dict): The cleaned transcript dictionary.
            - stats (dict): Segment counts and invalid statistics.
            - saved_path (Path): Path to the saved processed JSON file.

    Raises:
        FileNotFoundError: If input file does not exist.
        FileExistsError: If target file exists and overwrite=False.
        TranscriptProcessingError: If parsing or structure validation fails.
    """
    source = Path(input_path).resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Transcript file not found: {source}")

    target_dir = Path(output_path).resolve() if output_path else PROCESSED_TRANSCRIPT_DIR
    if target_dir.is_dir() or target_dir.suffix.lower() != ".json":
        target_file = target_dir / source.name
    else:
        target_file = target_dir

    if target_file.exists() and not overwrite:
        raise FileExistsError(
            f"Processed transcript already exists: '{target_file}'. Use overwrite=True to replace it."
        )

    try:
        with open(source, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        raise TranscriptProcessingError(f"Failed to parse JSON file '{source.name}': {err}") from err
    except OSError as err:
        raise TranscriptProcessingError(f"Failed to read file '{source.name}': {err}") from err

    processed_data, stats = process_transcript(raw_data, fallback_source_name=f"{source.stem}.mp3")
    saved_path = save_processed_transcript(processed_data, target_file)

    return processed_data, stats, saved_path


def batch_process_transcripts(
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """
    Scans the raw transcript directory, ignores subdirectories (specifically the 'processed/' folder),
    processes each transcript file, and writes normalized JSON output into the processed directory.

    Continues processing even if an individual transcript file fails.

    Args:
        input_dir: Source directory containing raw transcripts (defaults to config.TRANSCRIPT_DIR).
        output_dir: Target directory for processed transcripts (defaults to config.PROCESSED_TRANSCRIPT_DIR).
        overwrite: If True, re-processes and overwrites existing processed transcripts.
        status_callback: Optional callable(status_type, message) for reporting progress.

    Returns:
        dict: Summary statistics and file lists:
            {
                "total_found": int,
                "processed": list[Path],
                "skipped": list[Path],
                "failed": list[tuple[Path, str]],
                "total_valid_segments": int,
                "total_invalid_segments": int,
                "output_dir": Path,
            }
    """
    src_dir = Path(input_dir or TRANSCRIPT_DIR).resolve()
    dst_dir = Path(output_dir or PROCESSED_TRANSCRIPT_DIR).resolve()
    ensure_dir(dst_dir)

    def notify(status_type: str, message: str) -> None:
        if status_callback:
            status_callback(status_type, message)

    # Discover JSON files in input_dir while strictly ignoring subdirectories like 'processed/'
    transcript_files: list[Path] = []
    if dir_exists(src_dir):
        for item in src_dir.iterdir():
            # Only select regular files with .json extension directly inside src_dir
            if item.is_file() and item.suffix.lower() == ".json":
                # Guard against processing any file located inside the output directory
                if item.resolve() != dst_dir and dst_dir not in item.parents:
                    transcript_files.append(item)

    transcript_files.sort(key=lambda p: p.name.lower())

    results: dict[str, Any] = {
        "total_found": len(transcript_files),
        "processed": [],
        "skipped": [],
        "failed": [],
        "total_valid_segments": 0,
        "total_invalid_segments": 0,
        "output_dir": dst_dir,
    }

    notify("INFO", f"Found {len(transcript_files)} transcript file(s) in '{src_dir}'.")

    for idx, trans_file in enumerate(transcript_files, start=1):
        target_file = dst_dir / trans_file.name

        # Skip existing files unless overwrite is requested
        if target_file.exists() and not overwrite:
            results["skipped"].append(target_file)
            notify(
                "SKIPPED",
                f"[{idx}/{len(transcript_files)}] Skipped: '{trans_file.name}' -> already processed.",
            )
            continue

        notify(
            "PROCESSING",
            f"[{idx}/{len(transcript_files)}] Processing: '{trans_file.name}'...",
        )

        try:
            _, stats, saved_path = process_transcript_file(
                input_path=trans_file,
                output_path=target_file,
                overwrite=True,
            )
            results["processed"].append(saved_path)
            results["total_valid_segments"] += stats["valid_segments"]
            results["total_invalid_segments"] += stats["invalid_segments"]

            notify(
                "SUCCESS",
                f"[{idx}/{len(transcript_files)}] Saved processed transcript: '{saved_path.name}' "
                f"(Valid: {stats['valid_segments']}, Invalid: {stats['invalid_segments']})",
            )
        except Exception as exc:
            results["failed"].append((trans_file, str(exc)))
            notify(
                "ERROR",
                f"[{idx}/{len(transcript_files)}] Failed: '{trans_file.name}' - {exc}",
            )

    return results
