"""
src/rag/pipeline.py
===================
End-to-end RAG Teacher Assistant pipeline.

Connects and orchestrates:
1. Video Ingestion -> Audio Extraction -> Whisper Transcription -> Processing -> Chunking -> Embeddings -> Dataset Storage.
2. User Query -> Semantic Retrieval -> Grounded Prompt Construction -> LLM Synthesis -> Source Attribution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.audio import convert_video_to_mp3, is_ffmpeg_available
from src.chunking import (
    chunk_transcript,
    save_chunks,
)
from src.config import (
    AUDIO_DIR,
    CHUNK_DIR,
    CONSOLIDATED_DATASET_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CONVERSATION_TURNS,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_TRANSCRIPT_DIR,
    TRANSCRIPT_DIR,
    VIDEO_DIR,
)
from src.data.course_catalog import CourseCatalog, compute_file_hash
from src.data.dataset_builder import build_dataset, save_dataset
from src.embeddings import embed_chunk_file
from src.llm import LLMClient, LLMError
from src.rag.prompt import (
    DEFAULT_RESPONSE_STYLE,
    FALLBACK_NOT_FOUND_MESSAGE,
    RESPONSE_STYLES,
    build_rag_messages,
    build_retrieval_query,
    format_rag_context,
    is_follow_up_query,
    synthesize_grounded_answer,
)
from src.retrieval import LectureRetriever, RetrievalError, format_timestamp
from src.transcription import (
    process_transcript,
    save_processed_transcript,
    save_transcript,
    transcribe_audio,
)

logger = logging.getLogger(__name__)


def is_relevant(
    retrieved_chunks: list[dict[str, Any]],
    threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> bool:
    """
    Evaluates whether the top retrieved chunk meets the relevance threshold.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries containing 'similarity_score'.
        threshold: Minimum similarity score required for the top result.

    Returns:
        bool: True if top chunk similarity score >= threshold, False otherwise.
    """
    if not retrieved_chunks:
        return False
    top_score = float(retrieved_chunks[0].get("similarity_score", 0.0))
    return top_score >= threshold


@dataclass
class RAGAnswer:
    """Represents a grounded educational response produced by the RAG pipeline."""

    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    context_text: str = ""
    is_fallback: bool = False
    debug_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts the answer dataclass into a JSON-serializable dictionary."""
        return asdict(self)


class RAGTeacherPipeline:
    """
    Unified end-to-end RAG Teacher Assistant pipeline coordinator.

    Handles both:
    - Knowledge indexing (Video/Audio -> Chunks -> Embeddings -> Joblib dataset)
    - Question answering (Query -> Similarity search -> Grounded prompt -> LLM synthesis)
    """

    def __init__(
        self,
        retriever: LectureRetriever | None = None,
        llm_client: LLMClient | None = None,
        dataset_path: Path | str = CONSOLIDATED_DATASET_PATH,
        embeddings_dir: Path | str = EMBEDDING_DIR,
        catalog: CourseCatalog | None = None,
    ) -> None:
        """
        Initializes the RAG Teacher Assistant pipeline.

        Args:
            retriever: Optional pre-configured LectureRetriever instance.
            llm_client: Optional pre-configured LLMClient instance.
            dataset_path: Path to consolidated Joblib dataset file.
            embeddings_dir: Path to directory containing embedding archives.
            catalog: Optional pre-configured CourseCatalog instance.
        """
        self.dataset_path = Path(dataset_path)
        self.embeddings_dir = Path(embeddings_dir)

        self.retriever = retriever or LectureRetriever(
            embeddings_dir=self.embeddings_dir,
            dataset_path=self.dataset_path,
            auto_load=True,
        )
        self.llm_client = llm_client or LLMClient()
        self.catalog = catalog or CourseCatalog()

    def get_corpus_stats(self) -> dict[str, Any]:
        """
        Returns statistics regarding currently indexed lecture material.

        Returns:
            Dictionary containing chunk counts, indexed sources, catalog info, and model names.
        """
        metadata = self.retriever.chunks_metadata if self.retriever else []
        total_chunks = len(metadata)
        unique_sources = sorted(
            list({c.get("source_file") or c.get("source_transcript", "Unknown") for c in metadata})
        )
        total_words = sum(c.get("word_count", 0) for c in metadata)

        # Calculate total duration if timestamps available
        total_duration_secs = 0.0
        for c in metadata:
            end_t = float(c.get("end_time", 0.0))
            if end_t > total_duration_secs:
                total_duration_secs = end_t

        lectures = self.catalog.get_all_lectures()

        return {
            "total_chunks": total_chunks,
            "unique_sources": unique_sources,
            "source_count": len(unique_sources),
            "total_words": total_words,
            "max_timestamp": format_timestamp(total_duration_secs),
            "embedding_model": self.retriever.model_name if self.retriever else "Unknown",
            "llm_model": self.llm_client.model_name if self.llm_client else "Unknown",
            "llm_configured": self.llm_client.is_configured() if self.llm_client else False,
            "lectures": lectures,
            "lecture_count": len(lectures),
        }

    def get_catalog_lectures(self) -> list[dict[str, Any]]:
        """Returns all registered lectures in the course knowledge base catalog."""
        return self.catalog.get_all_lectures()

    def generate_summary(
        self,
        lecture_scope: str = "All Lectures",
        top_k: int = 6,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> Any:
        """Generates a structured, grounded summary of the specified lecture material."""
        from src.learning import generate_summary
        return generate_summary(
            retriever=self.retriever,
            llm_client=self.llm_client,
            lecture_scope=lecture_scope,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def generate_key_concepts(
        self,
        lecture_scope: str = "All Lectures",
        top_k: int = 6,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> Any:
        """Extracts key educational concepts grounded in the specified lecture material."""
        from src.learning import generate_key_concepts
        return generate_key_concepts(
            retriever=self.retriever,
            llm_client=self.llm_client,
            lecture_scope=lecture_scope,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def explain_lecture(
        self,
        lecture_scope: str = "All Lectures",
        top_k: int = 6,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> Any:
        """Produces a pedagogical, student-friendly explanation of the lecture material."""
        from src.learning import explain_lecture
        return explain_lecture(
            retriever=self.retriever,
            llm_client=self.llm_client,
            lecture_scope=lecture_scope,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def generate_quiz(
        self,
        lecture_scope: str = "All Lectures",
        num_questions: int = 5,
        top_k: int = 6,
        similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> Any:
        """Generates a validated multiple-choice quiz grounded in the lecture material."""
        from src.learning import generate_quiz
        return generate_quiz(
            retriever=self.retriever,
            llm_client=self.llm_client,
            lecture_scope=lecture_scope,
            num_questions=num_questions,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    def index_audio(
        self,
        audio_path: Path | str,
        overwrite: bool = True,
        whisper_model: str | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
        course_id: str | None = None,
        lecture_id: str | None = None,
        lecture_title: str | None = None,
        source_filename: str | None = None,
        file_hash: str | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """
        Processes an audio file end-to-end with duplicate detection and lecture metadata:
        Audio -> Whisper -> Processed Transcript -> Chunks -> Embeddings -> Dataset -> Reload Retriever.

        Args:
            audio_path: Path to .mp3, .wav, or .m4a file.
            overwrite: Whether to overwrite existing intermediate files.
            whisper_model: Optional Whisper model size override.
            progress_callback: Optional callback(stage_message, float_pct_0_to_1).
            course_id: Optional course identifier.
            lecture_id: Optional unique lecture identifier.
            lecture_title: Optional human-readable lecture title.
            source_filename: Optional original filename.
            file_hash: Optional precomputed content hash.
            force_reindex: If True, bypasses duplicate detection.

        Returns:
            Summary dictionary of indexing results.
        """
        def report(msg: str, pct: float) -> None:
            logger.info("Indexing progress: [%.0f%%] %s", pct * 100, msg)
            if progress_callback:
                progress_callback(msg, pct)

        audio_file = Path(audio_path).resolve()
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Duplicate detection by content hash (Section 5)
        content_hash = file_hash or compute_file_hash(audio_file)
        if not force_reindex and self.catalog.is_duplicate(content_hash):
            existing = self.catalog.get_lecture_by_hash(content_hash)
            existing_title = existing.get("lecture_title", audio_file.name) if existing else audio_file.name
            existing_id = existing.get("lecture_id", "") if existing else ""
            existing_count = existing.get("chunk_count", 0) if existing else 0
            msg = f"Lecture '{existing_title}' is already indexed (content hash: {content_hash[:8]}). Skipping duplicate."
            logger.info(msg)
            report(f"Already indexed: {existing_title}", 1.0)
            return {
                "source_file": audio_file.name,
                "lecture_id": existing_id,
                "lecture_title": existing_title,
                "chunk_count": existing_count,
                "total_corpus_chunks": len(self.retriever.chunks_metadata),
                "status": "skipped_duplicate",
                "message": msg,
            }

        stem = audio_file.stem
        cid = course_id or self.catalog.course_id
        lid = lecture_id or stem
        ltitle = lecture_title or stem.replace("_", " ").title()
        sfilename = source_filename or audio_file.name

        report(f"Transcribing audio with Whisper: {audio_file.name}", 0.15)
        raw_transcript = transcribe_audio(
            audio_path=audio_file,
            output_path=TRANSCRIPT_DIR / f"{stem}.json",
            model_name=whisper_model,
        )

        report("Cleaning and validating transcript segments", 0.40)
        processed_data, _ = process_transcript(raw_transcript, fallback_source_name=sfilename)
        save_processed_transcript(processed_data, PROCESSED_TRANSCRIPT_DIR / f"{stem}_processed.json")

        report("Segmenting transcript into semantic chunks", 0.60)
        chunks_data = chunk_transcript(
            transcript_data=processed_data,
            transcript_filename=f"{stem}.json",
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            course_id=cid,
            lecture_id=lid,
            lecture_title=ltitle,
            source_filename=sfilename,
        )
        chunks = chunks_data.get("chunks", [])
        num_created = len(chunks)
        logger.info("[Indexing] Created %d transcript chunks for '%s'.", num_created, stem)

        chunk_file = save_chunks(chunks_data, CHUNK_DIR / f"{stem}_chunks.json", overwrite=overwrite)
        logger.info("[Indexing] Successfully saved %d chunks to '%s'.", num_created, chunk_file)

        report(f"Generating dense vector embeddings ({num_created} chunks)", 0.80)
        embed_chunk_file(
            chunk_file=chunk_file,
            output_dir=EMBEDDING_DIR,
            overwrite=overwrite,
        )
        logger.info("[Indexing] Generated dense vector embeddings for '%s'.", stem)

        report("Consolidating dataset and updating vector storage", 0.95)
        raw_dataset = build_dataset(
            chunks_dir=CHUNK_DIR,
            embeddings_dir=EMBEDDING_DIR,
            dataset_path=self.dataset_path,
            overwrite=True,
        )
        df: pd.DataFrame = raw_dataset[0] if isinstance(raw_dataset, tuple) else raw_dataset

        num_records = len(df) if hasattr(df, "__len__") else 0
        num_cols = len(df.columns) if hasattr(df, "columns") else 0
        logger.info(
            "[Indexing] Consolidated dataset created: type=%s, rows=%d, columns=%d",
            type(df).__name__,
            num_records,
            num_cols,
        )

        save_dataset(df, dataset_path=self.dataset_path, overwrite=True)
        logger.info("[Indexing] Consolidated dataset updated with %d total chunks.", num_records)

        # Register lecture in course catalog
        self.catalog.register_lecture(
            lecture_id=lid,
            lecture_title=ltitle,
            course_id=cid,
            source_filename=sfilename,
            file_hash=content_hash,
            chunk_count=num_created,
            status="Indexed",
            chunk_file=str(chunk_file),
            embedding_file=str(EMBEDDING_DIR / f"{stem}_embeddings.npz"),
            metadata_file=str(EMBEDDING_DIR / f"{stem}_metadata.json"),
        )

        # Reload retriever corpus so new knowledge is immediately searchable
        self.retriever.load_corpus()
        report(
            f"Indexing complete! {num_created} chunks added. Total searchable corpus: {len(self.retriever.chunks_metadata)} chunks.",
            1.0,
        )
        logger.info(
            "[Indexing] Indexing complete: '%s' is now searchable (%d total chunks in memory).",
            sfilename,
            len(self.retriever.chunks_metadata),
        )

        return {
            "source_file": sfilename,
            "lecture_id": lid,
            "lecture_title": ltitle,
            "chunk_count": num_created,
            "total_corpus_chunks": len(self.retriever.chunks_metadata),
            "status": "success",
        }

    def index_video(
        self,
        video_path: Path | str,
        overwrite: bool = True,
        whisper_model: str | None = None,
        progress_callback: Callable[[str, float], None] | None = None,
        course_id: str | None = None,
        lecture_id: str | None = None,
        lecture_title: str | None = None,
        force_reindex: bool = False,
    ) -> dict[str, Any]:
        """
        Processes an educational video file end-to-end:
        Video -> FFmpeg Audio Extraction -> index_audio().

        Args:
            video_path: Path to video file (.mp4, .mkv, .mov, etc.).
            overwrite: Whether to overwrite existing intermediate files.
            whisper_model: Optional Whisper model size override.
            progress_callback: Optional callback(stage_message, float_pct_0_to_1).
            course_id: Optional course identifier.
            lecture_id: Optional unique lecture identifier.
            lecture_title: Optional human-readable lecture title.
            force_reindex: If True, bypasses duplicate detection.

        Returns:
            Summary dictionary of indexing results.
        """
        def report(msg: str, pct: float) -> None:
            logger.info("Video Indexing: [%.0f%%] %s", pct * 100, msg)
            if progress_callback:
                progress_callback(msg, pct)

        video_file = Path(video_path).resolve()
        if not video_file.exists():
            raise FileNotFoundError(f"Video file not found: {video_file}")

        video_hash = compute_file_hash(video_file)
        if not force_reindex and self.catalog.is_duplicate(video_hash):
            existing = self.catalog.get_lecture_by_hash(video_hash)
            existing_title = existing.get("lecture_title", video_file.name) if existing else video_file.name
            existing_id = existing.get("lecture_id", "") if existing else ""
            existing_count = existing.get("chunk_count", 0) if existing else 0
            msg = f"Lecture video '{existing_title}' is already indexed (content hash: {video_hash[:8]}). Skipping duplicate."
            logger.info(msg)
            report(f"Already indexed: {existing_title}", 1.0)
            return {
                "source_file": video_file.name,
                "lecture_id": existing_id,
                "lecture_title": existing_title,
                "chunk_count": existing_count,
                "total_corpus_chunks": len(self.retriever.chunks_metadata),
                "status": "skipped_duplicate",
                "message": msg,
            }

        report(f"Extracting audio from video: {video_file.name}", 0.05)
        mp3_path = AUDIO_DIR / f"{video_file.stem}.mp3"
        convert_video_to_mp3(
            video_path=video_file,
            output_path=mp3_path,
            overwrite=overwrite,
        )

        return self.index_audio(
            audio_path=mp3_path,
            overwrite=overwrite,
            whisper_model=whisper_model,
            progress_callback=progress_callback,
            course_id=course_id,
            lecture_id=lecture_id,
            lecture_title=lecture_title,
            source_filename=video_file.name,
            file_hash=video_hash,
            force_reindex=force_reindex,
        )

    def reindex_all(
        self,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Rebuilds the consolidated Joblib dataset from existing chunks and embeddings
        and refreshes the in-memory retriever corpus.

        Returns:
            Summary dictionary of reindexed dataset.
        """
        if progress_callback:
            progress_callback("Rebuilding dataset from disk...", 0.5)

        raw_dataset = build_dataset(
            chunks_dir=CHUNK_DIR,
            embeddings_dir=EMBEDDING_DIR,
            dataset_path=self.dataset_path,
            overwrite=True,
        )
        df: pd.DataFrame = raw_dataset[0] if isinstance(raw_dataset, tuple) else raw_dataset
        save_dataset(df, dataset_path=self.dataset_path, overwrite=True)
        self.retriever.load_corpus()

        if progress_callback:
            progress_callback("Dataset refresh complete!", 1.0)

        return {
            "total_chunks": len(self.retriever.chunks_metadata),
            "status": "success",
        }

    def answer_question(
        self,
        question: str,
        conversation_history: list[dict[str, Any]] | None = None,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float | None = DEFAULT_SIMILARITY_THRESHOLD,
        source_filter: str | None = None,
        lecture_scope: str | None = None,
        response_style: str = DEFAULT_RESPONSE_STYLE,
        retrieval_query: str | None = None,
    ) -> RAGAnswer:
        """
        Answers a student's question grounded strictly in retrieved lecture context.

        Args:
            question: The student's question string.
            conversation_history: Optional list of past chat messages for follow-up questions.
            top_k: Number of highest-scoring chunks to retrieve.
            similarity_threshold: Optional cosine similarity cutoff.
            source_filter: Optional source video/transcript filename filter.
            lecture_scope: Optional lecture scope ("All Lectures" or specific lecture name/ID).
            response_style: Pedagogical response mode ("Explain Clearly", "Short Answer", "Detailed Explanation").
            retrieval_query: Optional explicit retrieval query override (defaults to contextual follow-up builder).

        Returns:
            RAGAnswer dataclass containing answer, formatted sources with exact timestamps,
            context text, fallback status, and debug diagnostics.
        """
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            return RAGAnswer(
                question="",
                answer="Please enter a valid educational question.",
                is_fallback=True,
                debug_info={"error": "Empty question"},
            )

        # Resolve effective lecture filter
        effective_filter = (
            lecture_scope
            if lecture_scope and str(lecture_scope).strip() not in ("All Lectures", "All", "")
            else (source_filter if source_filter and str(source_filter).strip() not in ("All Lectures", "All", "") else None)
        )

        # Check if knowledge base is completely empty
        chunks_meta = getattr(self.retriever, "chunks_metadata", None)
        if chunks_meta is not None and len(chunks_meta) == 0:
            return RAGAnswer(
                question=cleaned_question,
                answer=(
                    "### Course Knowledge Base Empty\n\n"
                    "No course lectures are currently indexed. "
                    "Please upload a lecture video or audio file in the Course Knowledge Base section to begin asking questions."
                ),
                sources=[],
                context_text="",
                is_fallback=True,
                debug_info={
                    "error": "No lectures indexed",
                    "query": cleaned_question,
                    "selected_lecture": effective_filter or "All Lectures",
                    "lecture_scope": effective_filter or "All Lectures",
                    "retrieved_count": 0,
                    "llm_called": False,
                },
            )

        # Handle simple greetings supportively
        normalized_q = cleaned_question.lower().strip("!?,. ")
        greeting_words = {
            "hello", "hi", "hllo", "hey", "hola", "greetings", "good morning",
            "good afternoon", "good evening", "howdy", "sup", "yo"
        }
        if normalized_q in greeting_words:
            stats = self.get_corpus_stats()
            sources_summary = ", ".join(stats.get("unique_sources", [])) or "python_lecture.mp3"
            return RAGAnswer(
                question=cleaned_question,
                answer=(
                    "Hello! 👋 I am your **AI Teacher Assistant**.\n\n"
                    "I am ready to answer any questions grounded in your course lectures. "
                    f"Currently, I have **{stats['total_chunks']}** indexed lecture passages from **{sources_summary}**.\n\n"
                    "**Try asking me questions like:**\n"
                    "- 💡 *\"What is machine learning?\"*\n"
                    "- 💡 *\"What is the difference between supervised and unsupervised learning?\"*\n"
                    "- 💡 *\"What topics are introduced in the lecture?\"*\n\n"
                    "*(You can also upload more lecture videos or audio files in the sidebar to expand my knowledge base!)*"
                ),
                sources=[],
                context_text="",
                is_fallback=False,
                debug_info={"type": "greeting", "llm_called": False},
            )

        # Determine effective retrieval query (lightweight contextual follow-up)
        if retrieval_query and retrieval_query.strip():
            effective_retrieval_query = retrieval_query.strip()
            is_follow_up = True
        else:
            effective_retrieval_query, is_follow_up = build_retrieval_query(
                query=cleaned_question,
                conversation_history=conversation_history,
                current_scope=effective_filter,
            )

        # 1. Semantic Retrieval (Preserve top-K results and similarity scores)
        try:
            raw_retrieved_chunks = self.retriever.retrieve(
                query=effective_retrieval_query,
                top_k=top_k,
                similarity_threshold=None,
                source_file=effective_filter,
                lecture_filter=effective_filter,
                lecture_scope=effective_filter,
            )
        except RetrievalError as err:
            logger.warning(
                "Retrieval error for query '%s' (effective: '%s'): %s",
                cleaned_question,
                effective_retrieval_query,
                err,
            )
            raw_retrieved_chunks = []

        # 2. Determine effective relevance threshold and evaluate relevance
        effective_threshold = (
            float(similarity_threshold)
            if similarity_threshold is not None
            else DEFAULT_RELEVANCE_THRESHOLD
        )

        relevant = is_relevant(raw_retrieved_chunks, threshold=effective_threshold)
        top_score = (
            float(raw_retrieved_chunks[0].get("similarity_score", 0.0))
            if raw_retrieved_chunks
            else 0.0
        )
        decision = "RELEVANT" if relevant else "NOT RELEVANT"

        logger.info(
            "Query: '%s' | Effective Query: '%s' | Follow-up: %s | Scope: %s | Retrieved: %d chunks | Top score: %.4f | Threshold: %.4f | Decision: %s",
            cleaned_question,
            effective_retrieval_query,
            is_follow_up,
            effective_filter or "All Lectures",
            len(raw_retrieved_chunks),
            top_score,
            effective_threshold,
            decision,
        )

        # 3. Handle out-of-scope / irrelevant questions immediately (Abstention Layer)
        if not relevant:
            return RAGAnswer(
                question=cleaned_question,
                answer=(
                    f"### Course Lecture Answer\n\n"
                    f"{FALLBACK_NOT_FOUND_MESSAGE}"
                ),
                sources=[],
                context_text="",
                is_fallback=True,
                debug_info={
                    "query": cleaned_question,
                    "retrieval_query": effective_retrieval_query,
                    "is_follow_up": is_follow_up,
                    "response_style": response_style,
                    "selected_lecture": effective_filter or "All Lectures",
                    "lecture_scope": effective_filter or "All Lectures",
                    "retrieved_count": len(raw_retrieved_chunks),
                    "top_score": round(top_score, 4),
                    "relevance_decision": "NOT RELEVANT",
                    "relevance_threshold": round(effective_threshold, 4),
                    "similarity_scores": [
                        round(float(c.get("similarity_score", 0.0)), 4)
                        for c in raw_retrieved_chunks
                    ],
                    "chunk_ids": [c.get("chunk_id", "") for c in raw_retrieved_chunks],
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold,
                    "source_filter": effective_filter,
                    "history_length": len(conversation_history or []),
                    "llm_called": False,
                },
            )

        # 4. Relevant query: extract qualifying context chunks (preserving timestamps & metadata)
        retrieved_chunks = [
            c for c in raw_retrieved_chunks
            if float(c.get("similarity_score", 0.0)) >= effective_threshold
        ]
        if not retrieved_chunks:
            retrieved_chunks = [raw_retrieved_chunks[0]]

        sources: list[dict[str, Any]] = []
        for chunk in retrieved_chunks:
            source_file = chunk.get("source_file") or chunk.get("source_transcript") or "Lecture"
            source_filename = chunk.get("source_filename") or source_file
            start_s = float(chunk.get("start_time", 0.0))
            end_s = float(chunk.get("end_time", 0.0))
            score = chunk.get("similarity_score", 0.0)
            lecture_id = chunk.get("lecture_id") or ""
            lecture_title = chunk.get("lecture_title") or Path(source_file).stem.replace("_", " ").title()
            course_id = chunk.get("course_id") or "python_course"
            t_start = chunk.get("timestamp_start") or format_timestamp(start_s)
            t_end = chunk.get("timestamp_end") or format_timestamp(end_s)

            sources.append({
                "rank": chunk.get("rank", len(sources) + 1),
                "chunk_id": chunk.get("chunk_id", ""),
                "course_id": course_id,
                "lecture_id": lecture_id,
                "lecture_title": lecture_title,
                "source_file": source_file,
                "source_filename": source_filename,
                "start_time": start_s,
                "end_time": end_s,
                "timestamp_start": t_start,
                "timestamp_end": t_end,
                "start_formatted": format_timestamp(start_s),
                "end_formatted": format_timestamp(end_s),
                "timestamp_range": f"{format_timestamp(start_s)} - {format_timestamp(end_s)}",
                "similarity_score": score,
                "text": chunk.get("text", ""),
            })

        context_text = format_rag_context(retrieved_chunks)

        # 5. Assemble Grounded Prompt Messages
        messages = build_rag_messages(
            question=cleaned_question,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            max_history_turns=DEFAULT_CONVERSATION_TURNS,
            response_style=response_style,
        )

        top_source = sources[0]
        top_lecture = top_source.get("lecture_title") or top_source.get("source_filename") or top_source.get("source_file") or "Course Lecture"
        top_range = top_source["timestamp_range"]
        top_file = top_source.get("source_file") or top_source.get("source_filename") or ""

        # Header identifies lecture, filename, and exact timestamps (Section 10)
        header_file_part = f" ({top_file})" if top_file and top_file != top_lecture else ""
        answer_header = (
            f"### Course Lecture Answer\n\n"
            f"Based on:\n"
            f"**{top_lecture}**{header_file_part}\n\n"
            f"Timestamp:\n"
            f"**{top_range}**\n\n"
        )

        debug_info: dict[str, Any] = {
            "query": cleaned_question,
            "retrieval_query": effective_retrieval_query,
            "is_follow_up": is_follow_up,
            "response_style": response_style,
            "selected_lecture": effective_filter or "All Lectures",
            "lecture_scope": effective_filter or "All Lectures",
            "top_source_lecture": top_lecture,
            "top_source_timestamp": top_range,
            "retrieved_count": len(raw_retrieved_chunks),
            "top_score": round(top_score, 4),
            "relevance_decision": "RELEVANT",
            "relevance_threshold": round(effective_threshold, 4),
            "similarity_scores": [
                round(float(c.get("similarity_score", 0.0)), 4)
                for c in raw_retrieved_chunks
            ],
            "chunk_ids": [c.get("chunk_id", "") for c in retrieved_chunks],
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "source_filter": effective_filter,
            "history_length": len(conversation_history or []),
            "model": self.llm_client.model_name,
            "llm_called": True,
        }

        # 6. Generate Grounded LLM Response
        try:
            if not self.llm_client.is_configured():
                # Educational synthesis fallback when API key is not yet configured
                logger.info("OpenAI API key not configured; using local grounded context synthesis.")
                first_text = top_source["text"]

                body_content = synthesize_grounded_answer(
                    question=cleaned_question,
                    source_text=first_text,
                    lecture_title=top_lecture,
                    timestamp_range=top_range,
                    response_style=response_style,
                )

                answer = (
                    f"{answer_header}"
                    f"{body_content}\n\n"
                    f"---\n"
                    f"*(Note: OpenAI API Key is not configured; synthesized directly from verified course material)*"
                )

                debug_info["llm_called"] = False
                debug_info["offline_synthesis"] = True
                return RAGAnswer(
                    question=cleaned_question,
                    answer=answer,
                    sources=sources,
                    context_text=context_text,
                    is_fallback=False,
                    debug_info=debug_info,
                )

            # Cap generation tokens according to selected response style
            style_token_limits = {
                "Short Answer": 160,
                "Explain Clearly": 320,
                "Detailed Explanation": 520,
            }
            max_tokens = style_token_limits.get(response_style, 320)
            raw_answer = self.llm_client.generate(messages, max_tokens=max_tokens).strip()
            is_fallback = FALLBACK_NOT_FOUND_MESSAGE.lower() in raw_answer.lower()

            if not raw_answer.startswith("### Course Lecture Answer") and not is_fallback:
                answer = f"{answer_header}{raw_answer}"
            else:
                answer = raw_answer

            return RAGAnswer(
                question=cleaned_question,
                answer=answer,
                sources=sources,
                context_text=context_text,
                is_fallback=is_fallback,
                debug_info=debug_info,
            )

        except LLMError as err:
            logger.warning("LLM API call failed (%s); providing direct grounded synthesis.", err)
            body_content = synthesize_grounded_answer(
                question=cleaned_question,
                source_text=top_source["text"],
                lecture_title=top_lecture,
                timestamp_range=top_range,
                response_style=response_style,
            )
            answer = (
                f"{answer_header}"
                f"{body_content}\n\n"
                f"---\n"
                f"*(Note: OpenAI API returned: `{err}`. "
                f"The answer above was synthesized directly from your verified lecture knowledge base.)*"
            )
            debug_info["llm_error"] = str(err)
            debug_info["offline_synthesis"] = True
            return RAGAnswer(
                question=cleaned_question,
                answer=answer,
                sources=sources,
                context_text=context_text,
                is_fallback=False,
                debug_info=debug_info,
            )
