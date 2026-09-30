"""
src/learning/summary.py
=======================
Grounded Lecture Summary generator for the RAG AI Teacher Assistant.

Retrieves representative chunks from the selected lecture scope, enforces the 0.32
relevance threshold, and produces a structured summary with full source attribution.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config import DEFAULT_RELEVANCE_THRESHOLD
from src.learning.models import LearningResult
from src.learning.prompts import build_summary_messages
from src.llm import LLMClient, LLMError
from src.rag import FALLBACK_NOT_FOUND_MESSAGE, is_relevant
from src.retrieval import LectureRetriever, RetrievalError, format_timestamp

logger = logging.getLogger(__name__)


def generate_summary(
    retriever: LectureRetriever,
    llm_client: LLMClient | None = None,
    lecture_scope: str = "All Lectures",
    top_k: int = 6,
    similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> LearningResult:
    """
    Generates a structured, grounded lecture summary from indexed course material.

    Args:
        retriever: Pre-configured LectureRetriever instance.
        llm_client: Optional LLMClient instance for generative synthesis.
        lecture_scope: Target lecture filter or "All Lectures".
        top_k: Number of representative chunks to retrieve.
        similarity_threshold: Cosine similarity cutoff for relevance (default: 0.32).

    Returns:
        LearningResult object containing structured markdown summary and citations.
    """
    llm = llm_client or LLMClient()
    effective_scope = (
        None
        if lecture_scope in ("All Lectures", "All", "all", "", None)
        else lecture_scope
    )

    # 1. Guard against empty knowledge base
    chunks_meta = getattr(retriever, "chunks_metadata", None)
    if chunks_meta is not None and len(chunks_meta) == 0:
        return LearningResult(
            activity_type="summary",
            title=f"Summary — {lecture_scope}",
            content=(
                "### Course Knowledge Base Empty\n\n"
                "No course lectures are currently indexed. "
                "Please upload a lecture video or audio file in the Course Knowledge Base section to generate summaries."
            ),
            sources=[],
            lecture_scope=lecture_scope,
            is_fallback=True,
            debug_info={"error": "No lectures indexed"},
        )

    # 2. Scoped semantic retrieval of summary concepts
    target_title = ""
    if effective_scope:
        for c in getattr(retriever, "chunks_metadata", []):
            cid = str(c.get("lecture_id", "")).lower()
            ctitle = str(c.get("lecture_title", "")).lower()
            if effective_scope.lower() in cid or effective_scope.lower() in ctitle:
                target_title = c.get("lecture_title", "")
                break

    query = f"{target_title or lecture_scope} lecture overview main topics ideas concepts and key summary".strip()
    try:
        raw_chunks = retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=None,
            source_file=effective_scope,
            lecture_filter=effective_scope,
            lecture_scope=effective_scope,
        )
    except RetrievalError as err:
        logger.warning("Retrieval failed for lecture summary: %s", err)
        raw_chunks = []

    # 3. Grounding & Relevance Gate (strictly 0.32 threshold)
    relevant = is_relevant(raw_chunks, threshold=similarity_threshold)
    top_score = float(raw_chunks[0].get("similarity_score", 0.0)) if raw_chunks else 0.0

    if not relevant:
        logger.info(
            "Summary rejected: top score %.4f below threshold %.4f for scope '%s'",
            top_score,
            similarity_threshold,
            lecture_scope,
        )
        return LearningResult(
            activity_type="summary",
            title=f"Summary — {lecture_scope}",
            content=f"### Lecture Summary\n\n{FALLBACK_NOT_FOUND_MESSAGE}",
            sources=[],
            lecture_scope=lecture_scope,
            is_fallback=True,
            debug_info={
                "relevance_decision": "NOT RELEVANT",
                "top_score": top_score,
                "threshold": similarity_threshold,
                "retrieved_count": len(raw_chunks),
            },
        )

    # 4. Filter to relevant chunks and compile structured citations
    relevant_chunks = [
        c for c in raw_chunks
        if float(c.get("similarity_score", 0.0)) >= similarity_threshold
    ]
    if not relevant_chunks:
        relevant_chunks = [raw_chunks[0]]

    sources: list[dict[str, Any]] = []
    for chunk in relevant_chunks:
        s_file = chunk.get("source_filename") or chunk.get("source_file") or "Lecture"
        l_title = chunk.get("lecture_title") or s_file
        st_secs = float(chunk.get("start_time", 0.0))
        end_secs = float(chunk.get("end_time", 0.0))
        t_range = f"{format_timestamp(st_secs)} - {format_timestamp(end_secs)}"
        sources.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "lecture_title": l_title,
            "source_filename": s_file,
            "timestamp_range": t_range,
            "similarity_score": float(chunk.get("similarity_score", 0.0)),
            "text": chunk.get("text", ""),
        })

    # 5. Synthesis: Generative with LLM or Grounded Offline Fallback
    if llm.is_configured():
        try:
            messages = build_summary_messages(relevant_chunks, lecture_scope=lecture_scope)
            content = llm.generate(messages).strip()
            return LearningResult(
                activity_type="summary",
                title=f"Summary — {lecture_scope}",
                content=content,
                sources=sources,
                lecture_scope=lecture_scope,
                is_fallback=False,
                debug_info={"top_score": top_score, "llm_called": True},
            )
        except LLMError as err:
            logger.warning("LLM generation failed for summary (%s); using offline grounded synthesis.", err)

    # Offline Grounded Synthesis
    first_chunk = relevant_chunks[0]
    lead_text = first_chunk.get("text", "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", lead_text) if s.strip()]
    lead_sentence = sentences[0] if sentences else lead_text

    topics_list = []
    for c in relevant_chunks:
        c_title = c.get("lecture_title") or c.get("source_filename") or "Course Material"
        c_text = c.get("text", "").strip()
        first_few = " ".join(c_text.split()[:12])
        topics_list.append(f"* **{c_title}**: \"{first_few}...\"")

    all_takeaways = []
    for idx, c in enumerate(relevant_chunks[:4], start=1):
        s_list = [s.strip() for s in re.split(r"(?<=[.!?]) +", c.get("text", "")) if s.strip()]
        takeaway = s_list[-1] if len(s_list) > 1 else s_list[0] if s_list else "Fundamental principle covered."
        all_takeaways.append(f"{idx}. {takeaway}")

    offline_summary = (
        f"### Lecture Overview\n"
        f"Based on the course excerpts from **{lecture_scope}**, this material introduces core educational concepts. "
        f"{lead_sentence}\n\n"
        f"### Main Topics\n"
        + "\n".join(topics_list) + "\n\n"
        f"### Important Ideas\n"
        f"The lecture emphasizes key principles and hands-on understanding. "
        f"Key excerpt: \"{lead_text[:350]}...\"\n\n"
        f"### Examples Discussed\n"
        f"Practical illustrations and code demonstrations presented across the lecture passages.\n\n"
        f"### Key Takeaways\n"
        + "\n".join(all_takeaways) + "\n\n"
        f"---\n"
        f"*(Note: Synthesized directly from verified lecture excerpts)*"
    )

    return LearningResult(
        activity_type="summary",
        title=f"Summary — {lecture_scope}",
        content=offline_summary,
        sources=sources,
        lecture_scope=lecture_scope,
        is_fallback=False,
        debug_info={"top_score": top_score, "llm_called": False, "offline_synthesis": True},
    )
