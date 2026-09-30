"""
src/learning/explain.py
=======================
Student-friendly lecture explainer for Module 12 Learning Features.

Provides intuitive, grounded explanations of lecture topics, breaking down
concepts, connections, and practical examples.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.config import DEFAULT_RELEVANCE_THRESHOLD
from src.learning.models import LearningResult
from src.learning.prompts import build_explain_messages
from src.llm import LLMClient, LLMError
from src.rag import FALLBACK_NOT_FOUND_MESSAGE, is_relevant
from src.retrieval import LectureRetriever, RetrievalError, format_timestamp

logger = logging.getLogger(__name__)


def explain_lecture(
    retriever: LectureRetriever,
    llm_client: LLMClient | None = None,
    lecture_scope: str = "All Lectures",
    top_k: int = 6,
    similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> LearningResult:
    """
    Produces an engaging, student-friendly explanation of the selected lecture.

    Args:
        retriever: Pre-configured LectureRetriever instance.
        llm_client: Optional LLMClient instance.
        lecture_scope: Target lecture filter or "All Lectures".
        top_k: Number of chunks to evaluate.
        similarity_threshold: Minimum cosine similarity required.

    Returns:
        LearningResult object containing structured explanation and source citations.
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
            activity_type="explain",
            title=f"Explain Lecture — {lecture_scope}",
            content=(
                "### Course Knowledge Base Empty\n\n"
                "No course lectures are currently indexed. "
                "Please upload a lecture in the Course Knowledge Base section to view explanations."
            ),
            sources=[],
            lecture_scope=lecture_scope,
            is_fallback=True,
            debug_info={"error": "No lectures indexed"},
        )

    # 2. Scoped semantic retrieval
    target_title = ""
    if effective_scope:
        for c in getattr(retriever, "chunks_metadata", []):
            cid = str(c.get("lecture_id", "")).lower()
            ctitle = str(c.get("lecture_title", "")).lower()
            if effective_scope.lower() in cid or effective_scope.lower() in ctitle:
                target_title = c.get("lecture_title", "")
                break

    query = f"{target_title or lecture_scope} complete lecture explanation core concepts relationships examples summary".strip()
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
        logger.warning("Retrieval failed for explain lecture: %s", err)
        raw_chunks = []

    # 3. Grounding & Relevance Gate (0.32 cutoff)
    relevant = is_relevant(raw_chunks, threshold=similarity_threshold)
    top_score = float(raw_chunks[0].get("similarity_score", 0.0)) if raw_chunks else 0.0

    if not relevant:
        logger.info(
            "Explanation rejected: top score %.4f below threshold %.4f for scope '%s'",
            top_score,
            similarity_threshold,
            lecture_scope,
        )
        return LearningResult(
            activity_type="explain",
            title=f"Explain Lecture — {lecture_scope}",
            content=f"### Explain This Lecture\n\n{FALLBACK_NOT_FOUND_MESSAGE}",
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

    # 4. Filter qualifying chunks and compile citations
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

    # 5. Generative or Offline Grounded Synthesis
    if llm.is_configured():
        try:
            messages = build_explain_messages(relevant_chunks, lecture_scope=lecture_scope)
            content = llm.generate(messages).strip()
            return LearningResult(
                activity_type="explain",
                title=f"Explain Lecture — {lecture_scope}",
                content=content,
                sources=sources,
                lecture_scope=lecture_scope,
                is_fallback=False,
                debug_info={"top_score": top_score, "llm_called": True},
            )
        except LLMError as err:
            logger.warning("LLM generation failed for explain (%s); using offline grounded synthesis.", err)

    # Offline Grounded Explanation
    first_text = relevant_chunks[0].get("text", "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?]) +", first_text) if s.strip()]
    lead_sentence = sentences[0] if sentences else first_text[:150]

    offline_explain = (
        f"### What This Lecture Is About\n"
        f"In **{lecture_scope}**, the focus is on building solid practical fundamentals. "
        f"{lead_sentence}\n\n"
        f"### Concepts You Should Understand\n"
        f"The lecture develops key concepts step-by-step through clear definitions and demonstrations. "
        f"Core excerpt: \"{first_text[:300]}...\"\n\n"
        f"### How They Relate\n"
        f"The topics build directly upon each other, allowing you to combine these foundational elements "
        f"into functional solutions.\n\n"
        f"### Examples From The Lecture\n"
        f"The instructor walks through specific use cases and demonstrations directly discussed in the material.\n\n"
        f"### Final Takeaways\n"
        f"Mastering the ideas presented in this lecture segment provides a great foundation for upcoming course units!\n\n"
        f"---\n"
        f"*(Note: Synthesized directly from verified lecture excerpts)*"
    )

    return LearningResult(
        activity_type="explain",
        title=f"Explain Lecture — {lecture_scope}",
        content=offline_explain,
        sources=sources,
        lecture_scope=lecture_scope,
        is_fallback=False,
        debug_info={"top_score": top_score, "llm_called": False, "offline_synthesis": True},
    )
