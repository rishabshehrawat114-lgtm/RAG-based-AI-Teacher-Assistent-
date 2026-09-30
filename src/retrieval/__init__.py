"""
src/retrieval
=============
Semantic retrieval module for the RAG AI Teacher Assistant.

Enables similarity-based search across precomputed lecture chunk embeddings.
Finds the top-K most semantically relevant lecture passages for student questions
using cosine similarity, timestamp mapping, and optional source lecture filtering.
"""

from src.retrieval.retriever import (
    LectureRetriever,
    RetrievalError,
    compute_cosine_similarity,
    format_retrieval_results,
    format_timestamp,
    load_all_lecture_embeddings,
    retrieve,
)

__all__ = [
    "LectureRetriever",
    "retrieve",
    "compute_cosine_similarity",
    "format_timestamp",
    "format_retrieval_results",
    "load_all_lecture_embeddings",
    "RetrievalError",
]
