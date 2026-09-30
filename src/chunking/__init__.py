"""
src/chunking
============
Transcript chunking module for the RAG AI Teacher Assistant.

Exposes core functions for partitioning timestamped educational transcripts into
cohesive text chunks enriched with video references, start/end timestamps, and segment IDs.
"""

from src.chunking.chunker import (
    ChunkingError,
    batch_chunk_transcripts,
    chunk_transcript,
    chunk_transcript_file,
    count_words,
    create_chunks,
    load_chunks,
    save_chunks,
    split_oversized_segments,
)

__all__ = [
    "create_chunks",
    "chunk_transcript",
    "chunk_transcript_file",
    "batch_chunk_transcripts",
    "save_chunks",
    "load_chunks",
    "count_words",
    "split_oversized_segments",
    "ChunkingError",
]
