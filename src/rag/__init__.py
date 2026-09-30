"""
src/rag
=======
Module responsible for orchestrating the end-to-end Retrieval-Augmented Generation pipeline.
Coordinates retrieval of lecture context and construction of grounded prompts for the LLM.
"""

from src.rag.pipeline import RAGAnswer, RAGTeacherPipeline, is_relevant
from src.rag.prompt import (
    DEFAULT_RESPONSE_STYLE,
    FALLBACK_NOT_FOUND_MESSAGE,
    RESPONSE_STYLES,
    RESPONSE_STYLE_PROMPTS,
    SYSTEM_TEACHER_PROMPT,
    build_rag_messages,
    build_retrieval_query,
    extract_example_topic,
    format_rag_context,
    is_follow_up_query,
    synthesize_grounded_answer,
)

__all__ = [
    "RAGTeacherPipeline",
    "RAGAnswer",
    "is_relevant",
    "FALLBACK_NOT_FOUND_MESSAGE",
    "SYSTEM_TEACHER_PROMPT",
    "build_rag_messages",
    "format_rag_context",
    "RESPONSE_STYLES",
    "RESPONSE_STYLE_PROMPTS",
    "DEFAULT_RESPONSE_STYLE",
    "is_follow_up_query",
    "extract_example_topic",
    "build_retrieval_query",
    "synthesize_grounded_answer",
]
