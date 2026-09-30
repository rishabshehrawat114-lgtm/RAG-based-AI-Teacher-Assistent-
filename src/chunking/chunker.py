"""
src/chunking/chunker.py
=======================
Transcript chunking and metadata attachment engine for the RAG AI Teacher Assistant.

Splits cleaned, timestamped transcript segments into cohesive text chunks tailored for
downstream vector embeddings and retrieval. Preserves exact segment boundary references,
continuous millisecond-precision timecodes (start_time, end_time), word counts, and
source provenance without rephrasing or losing educational content.
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional, Union

from src.config import (
    CHUNK_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    PROCESSED_TRANSCRIPT_DIR,
)
from src.utils import dir_exists, ensure_dir, file_exists


class ChunkingError(Exception):
    """Raised when an error occurs during transcript chunking or metadata generation."""
    pass


def count_words(text: Optional[str]) -> int:
    """
    Counts the number of whitespace-delimited words in a string.

    Args:
        text: Input text string.

    Returns:
        int: Total word count, or 0 if text is empty or None.
    """
    if not text:
        return 0
    return len(text.strip().split())


def split_oversized_segments(
    segments: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """
    Inspects segments and safely splits any individual segment that exceeds chunk_size
    into word-level sub-segments.

    Timestamp Handling:
    When a single speech segment without pauses exceeds chunk_size, words are split
    proportionally. Start and end timestamps are estimated by linear interpolation
    across the segment's total duration:
        sub_start = start + duration * (word_start_index / total_words)
        sub_end = start + duration * (word_end_index / total_words)
    This maintains chronological ordering and approximate timecodes without inventing false data.

    Args:
        segments: Cleaned segment dictionaries.
        chunk_size: Maximum target words per chunk.
        chunk_overlap: Desired word overlap between sub-segments.

    Returns:
        list[dict]: Segment list with oversized segments partitioned into valid bounds.
    """
    normalized_segments: list[dict[str, Any]] = []

    for seg in segments:
        text = seg.get("text", "").strip()
        words = text.split()
        total_words = len(words)

        # Standard segment fits comfortably within chunk_size
        if total_words <= chunk_size:
            normalized_segments.append(seg)
            continue

        # Segment exceeds chunk_size; partition into word slices
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        duration = max(0.0, seg_end - seg_start)
        seg_id = seg.get("id", len(normalized_segments))

        step = max(1, chunk_size - chunk_overlap)
        w_idx = 0

        while w_idx < total_words:
            sub_words = words[w_idx : w_idx + chunk_size]
            sub_text = " ".join(sub_words)

            start_ratio = w_idx / total_words
            end_ratio = min(total_words, w_idx + len(sub_words)) / total_words

            sub_start = round(seg_start + duration * start_ratio, 3)
            sub_end = round(seg_start + duration * end_ratio, 3)

            normalized_segments.append({
                "id": seg_id,
                "start": sub_start,
                "end": sub_end,
                "text": sub_text,
            })

            w_idx += step
            # If the remainder of the words was fully consumed, finish
            if w_idx + chunk_overlap >= total_words and (w_idx + len(sub_words)) >= total_words:
                break

    return normalized_segments


def format_time_hhmmss(seconds: float | int) -> str:
    """Formats numeric seconds into standard HH:MM:SS format (e.g. 00:10:25)."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_secs = int(round(float(seconds)))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_time_mmss(seconds: float | int) -> str:
    """Formats numeric seconds into MM:SS format, or HH:MM:SS if >= 1 hour."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_secs = int(round(float(seconds)))
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def create_chunks(
    segments: list[dict[str, Any]],
    source_file: str,
    source_transcript: str,
    language: str = "en",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    course_id: Optional[str] = None,
    lecture_id: Optional[str] = None,
    lecture_title: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Groups complete transcript segments into bounded text chunks with overlap and metadata.

    Algorithm Rules:
    1. Complete segments are kept intact whenever possible.
    2. Segments are accumulated until the accumulated word count reaches or exceeds chunk_size.
    3. The chunk's start_time and segment_start_id are derived strictly from the first segment.
    4. The chunk's end_time and segment_end_id are derived strictly from the last segment.
    5. The next chunk begins with an overlap of approximately chunk_overlap words from the end
       of the current chunk, preserving segment integrity.
    6. Forward progress is mathematically guaranteed (next starting index > current starting index)
       preventing duplicate chunks and infinite loops.
    7. Chunk IDs are deterministic and formatted as: <lecture_id>_0001, <lecture_id>_0002, etc.
    8. Includes full Module 10 multi-lecture course metadata: course_id, lecture_id,
       lecture_title, source_filename, timestamp_start, timestamp_end.

    Args:
        segments: Validated segment dictionaries from processed transcript.
        source_file: Audio/video source filename (e.g. 'machine_learning.mp3').
        source_transcript: Transcript JSON filename (e.g. 'machine_learning.json').
        language: Language code (e.g. 'en').
        chunk_size: Target words per chunk (default: 700).
        chunk_overlap: Target overlap words between consecutive chunks (default: 100).
        course_id: Optional course identifier (defaults to 'python_course').
        lecture_id: Optional unique lecture identifier.
        lecture_title: Optional human-readable lecture title.
        source_filename: Optional original source filename.

    Returns:
        list[dict]: List of chunk dictionaries enriched with timestamps and source metadata.
    """
    # Filter out empty or non-dict segments
    clean_segments = [
        s for s in segments
        if isinstance(s, dict) and s.get("text", "").strip()
    ]

    if not clean_segments:
        return []

    # Handle single segments that individually exceed chunk_size
    usable_segments = split_oversized_segments(clean_segments, chunk_size, chunk_overlap)
    total_segments = len(usable_segments)

    # Derive clean deterministic stem for chunk IDs
    stem = Path(source_transcript).stem
    if stem.endswith("_processed"):
        stem = stem[:-10]
    elif stem.endswith("_chunks"):
        stem = stem[:-7]

    cid = str(course_id or "python_course").strip()
    lid = str(lecture_id or stem).strip()
    ltitle = str(lecture_title or stem.replace("_", " ").title()).strip()
    sfilename = str(source_filename or source_file).strip()

    chunks: list[dict[str, Any]] = []
    chunk_index = 1
    i = 0

    while i < total_segments:
        # Accumulate segments starting from index i
        current_segments = [usable_segments[i]]
        current_words = count_words(usable_segments[i]["text"])
        j = i

        while j + 1 < total_segments and current_words < chunk_size:
            j += 1
            current_segments.append(usable_segments[j])
            current_words += count_words(usable_segments[j]["text"])

        # Construct combined text and word count
        combined_text = " ".join(s["text"].strip() for s in current_segments)
        actual_word_count = count_words(combined_text)

        start_s = round(float(current_segments[0]["start"]), 3)
        end_s = round(float(current_segments[-1]["end"]), 3)
        t_start_hh = format_time_hhmmss(start_s)
        t_end_hh = format_time_hhmmss(end_s)
        time_range = f"{format_time_mmss(start_s)} - {format_time_mmss(end_s)}"

        # Build chunk metadata
        chunk_entry = {
            # Module 10 required metadata
            "course_id": cid,
            "lecture_id": lid,
            "lecture_title": ltitle,
            "source_filename": sfilename,
            "timestamp_start": t_start_hh,
            "timestamp_end": t_end_hh,
            "timestamp": time_range,
            "chunk_id": f"{lid}_{chunk_index:04d}",
            "text": combined_text,
            # Backwards-compatibility and provenance metadata
            "source_file": source_file,
            "source_transcript": source_transcript,
            "language": language,
            "start_time": start_s,
            "end_time": end_s,
            "word_count": actual_word_count,
            "segment_start_id": int(current_segments[0]["id"]),
            "segment_end_id": int(current_segments[-1]["id"]),
        }
        chunks.append(chunk_entry)
        chunk_index += 1

        # If current chunk reached the final segment, finish processing
        if j >= total_segments - 1:
            break

        # Calculate next starting index (next_i) based on chunk_overlap
        if chunk_overlap <= 0:
            next_i = j + 1
        else:
            # Walk backward from j towards i+1 to accumulate up to chunk_overlap words
            overlap_words = 0
            next_i = j
            for k in range(j, i, -1):
                seg_w = count_words(usable_segments[k]["text"])
                if overlap_words + seg_w <= chunk_overlap:
                    overlap_words += seg_w
                    next_i = k
                else:
                    if next_i == j:
                        next_i = j
                    break

        # Guarantee strict forward progress: next_i must be > i
        next_i = max(i + 1, min(next_i, j))
        i = next_i

    return chunks


def chunk_transcript(
    transcript_data: dict[str, Any],
    transcript_filename: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    course_id: Optional[str] = None,
    lecture_id: Optional[str] = None,
    lecture_title: Optional[str] = None,
    source_filename: Optional[str] = None,
) -> dict[str, Any]:
    """
    Transforms a loaded processed transcript dictionary into a standardized chunking dataset.

    Args:
        transcript_data: Dictionary loaded from a processed transcript JSON file.
        transcript_filename: Name of the source transcript file (e.g. 'lecture.json').
        chunk_size: Configured target words per chunk.
        chunk_overlap: Configured word overlap between chunks.
        course_id: Optional course identifier.
        lecture_id: Optional unique lecture identifier.
        lecture_title: Optional human-readable lecture title.
        source_filename: Optional original source filename.

    Returns:
        dict: Standardized chunk dataset enriched with Module 10 course metadata:
            {
                "source_file": "machine_learning.mp3",
                "source_filename": "machine_learning.mp3",
                "course_id": "python_course",
                "lecture_id": "machine_learning",
                "lecture_title": "Machine Learning",
                "language": "en",
                "chunk_count": 57,
                "chunking_config": {
                    "chunk_size": 700,
                    "chunk_overlap": 100
                },
                "chunks": [ ... ]
            }

    Raises:
        ChunkingError: If transcript_data is malformed.
    """
    if not isinstance(transcript_data, dict):
        raise ChunkingError("Transcript data must be a dictionary.")

    source_file = str(transcript_data.get("source_file") or f"{Path(transcript_filename).stem}.mp3").strip()
    language = str(transcript_data.get("language") or "en").strip()
    segments = transcript_data.get("segments")

    if segments is None or not isinstance(segments, list):
        raise ChunkingError(
            f"Transcript '{transcript_filename}' does not contain a valid 'segments' list."
        )

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got: {chunk_size}")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got: {chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )

    cid = course_id or transcript_data.get("course_id") or "python_course"
    stem = Path(transcript_filename).stem.replace("_processed", "")
    lid = lecture_id or transcript_data.get("lecture_id") or stem
    ltitle = lecture_title or transcript_data.get("lecture_title") or stem.replace("_", " ").title()
    sfilename = source_filename or transcript_data.get("source_filename") or source_file

    chunks = create_chunks(
        segments=segments,
        source_file=source_file,
        source_transcript=transcript_filename,
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        course_id=cid,
        lecture_id=lid,
        lecture_title=ltitle,
        source_filename=sfilename,
    )

    return {
        "source_file": source_file,
        "source_filename": sfilename,
        "course_id": cid,
        "lecture_id": lid,
        "lecture_title": ltitle,
        "language": language,
        "chunk_count": len(chunks),
        "chunking_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "chunks": chunks,
    }


def save_chunks(
    chunks_data: dict[str, Any],
    output_path: Union[str, Path],
    overwrite: bool = True,
) -> Path:
    """
    Saves a chunk dataset dictionary to a JSON file.

    Args:
        chunks_data: Standardized chunk dataset dictionary.
        output_path: Destination JSON file path or directory.
        overwrite: Whether to replace existing chunk file if it already exists (default: True).

    Returns:
        Path: Resolved Path to the saved chunks JSON file.

    Raises:
        FileExistsError: If target file exists and overwrite=False.
        ChunkingError: If saving the file fails.
    """
    target = Path(output_path).resolve()

    # If target is a directory or lacks .json extension, derive '<stem>_chunks.json'
    if target.is_dir() or target.suffix.lower() != ".json":
        source_name = chunks_data.get("source_file", "transcript")
        stem = Path(source_name).stem
        target = target / f"{stem}_chunks.json"

    if target.exists() and not overwrite:
        raise FileExistsError(
            f"Chunk file already exists: '{target}'. Set overwrite=True to replace it."
        )

    ensure_dir(target.parent)

    try:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    except OSError as err:
        raise ChunkingError(f"Failed to write chunks to '{target}': {err}") from err

    return target


def load_chunks(chunk_path: Union[str, Path]) -> dict[str, Any]:
    """
    Loads and parses an existing chunks JSON file.

    Args:
        chunk_path: Path to the chunks .json file.

    Returns:
        dict: Parsed chunk dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If JSON parsing fails.
    """
    target = Path(chunk_path).resolve()
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"Chunk file not found: {target}")

    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_transcript_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    """
    Loads a processed transcript JSON file, converts it into chunks, and saves the result.

    Args:
        input_path: Path to the processed transcript JSON.
        output_path: Destination path for chunks JSON (defaults to CHUNK_DIR / '<stem>_chunks.json').
        chunk_size: Word count per chunk.
        chunk_overlap: Overlapping word count between chunks.
        overwrite: If True, replaces existing chunk file. If False, raises FileExistsError.

    Returns:
        tuple containing:
            - chunks_data (dict): Standardized chunk dataset.
            - saved_path (Path): Path to the saved chunks JSON file.

    Raises:
        FileNotFoundError: If input file does not exist.
        FileExistsError: If target file exists and overwrite=False.
        ChunkingError: If parsing or chunk generation fails.
    """
    source = Path(input_path).resolve()
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Processed transcript file not found: {source}")

    # Determine destination chunk file path
    target_dir = Path(output_path).resolve() if output_path else CHUNK_DIR
    if target_dir.is_dir() or target_dir.suffix.lower() != ".json":
        target_file = target_dir / f"{source.stem}_chunks.json"
    else:
        target_file = target_dir

    if target_file.exists() and not overwrite:
        raise FileExistsError(
            f"Chunk file already exists: '{target_file}'. Use overwrite=True to replace it."
        )

    try:
        with open(source, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except json.JSONDecodeError as err:
        raise ChunkingError(f"Failed to parse JSON file '{source.name}': {err}") from err
    except OSError as err:
        raise ChunkingError(f"Failed to read file '{source.name}': {err}") from err

    chunks_data = chunk_transcript(
        transcript_data=raw_data,
        transcript_filename=source.name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    saved_path = save_chunks(chunks_data, target_file, overwrite=overwrite)
    return chunks_data, saved_path


def batch_chunk_transcripts(
    input_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    overwrite: bool = False,
    status_callback: Optional[Callable[[str, str], None]] = None,
) -> dict[str, Any]:
    """
    Scans the processed transcript directory, chunks every transcript file,
    and saves corresponding chunk JSON files in the chunks directory.

    Continues processing even if an individual file fails.

    Args:
        input_dir: Directory containing processed transcripts (defaults to config.PROCESSED_TRANSCRIPT_DIR).
        output_dir: Directory where chunks are stored (defaults to config.CHUNK_DIR).
        chunk_size: Target words per chunk.
        chunk_overlap: Target overlap words.
        overwrite: If True, regenerates existing chunk files.
        status_callback: Optional callable(status_type, message) for reporting progress.

    Returns:
        dict: Summary of batch chunking results:
            {
                "total_found": int,
                "chunked": list[Path],
                "skipped": list[Path],
                "failed": list[tuple[Path, str]],
                "total_chunks": int,
                "output_dir": Path,
            }
    """
    src_dir = Path(input_dir or PROCESSED_TRANSCRIPT_DIR).resolve()
    dst_dir = Path(output_dir or CHUNK_DIR).resolve()
    ensure_dir(dst_dir)

    def notify(status_type: str, message: str) -> None:
        if status_callback:
            status_callback(status_type, message)

    transcript_files: list[Path] = []
    if dir_exists(src_dir):
        for item in src_dir.iterdir():
            if item.is_file() and item.suffix.lower() == ".json":
                transcript_files.append(item)

    transcript_files.sort(key=lambda p: p.name.lower())

    results: dict[str, Any] = {
        "total_found": len(transcript_files),
        "chunked": [],
        "skipped": [],
        "failed": [],
        "total_chunks": 0,
        "output_dir": dst_dir,
    }

    notify("INFO", f"Found {len(transcript_files)} transcript file(s) in '{src_dir}'.")

    for idx, trans_file in enumerate(transcript_files, start=1):
        target_chunk_file = dst_dir / f"{trans_file.stem}_chunks.json"

        # Check for existing chunk output
        if target_chunk_file.exists() and not overwrite:
            results["skipped"].append(target_chunk_file)
            notify(
                "SKIPPED",
                f"[{idx}/{len(transcript_files)}] Skipped: '{trans_file.name}' -> already chunked.",
            )
            continue

        notify(
            "PROCESSING",
            f"[{idx}/{len(transcript_files)}] Chunking: '{trans_file.name}'...",
        )

        try:
            chunks_data, saved_path = chunk_transcript_file(
                input_path=trans_file,
                output_path=target_chunk_file,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                overwrite=True,
            )
            count = chunks_data["chunk_count"]
            results["chunked"].append(saved_path)
            results["total_chunks"] += count

            notify(
                "SUCCESS",
                f"[{idx}/{len(transcript_files)}] Generated {count} chunks -> '{saved_path.name}'",
            )
        except Exception as exc:
            results["failed"].append((trans_file, str(exc)))
            notify(
                "ERROR",
                f"[{idx}/{len(transcript_files)}] Failed: '{trans_file.name}' - {exc}",
            )

    return results
