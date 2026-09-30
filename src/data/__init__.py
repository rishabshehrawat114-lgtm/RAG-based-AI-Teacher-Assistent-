"""
src/data
========
Consolidated dataset management and Joblib persistence for the RAG AI Teacher Assistant.

Merges raw transcript chunk metadata with numerical Sentence Transformer embeddings
into a structured Pandas DataFrame, validates alignment, and serializes to disk
using Joblib for fast, one-step retrieval without parsing raw files repeatedly.
"""

from src.data.course_catalog import (
    CourseCatalog,
    CourseLecture,
    compute_file_hash,
)
from src.data.dataset_builder import (
    DatasetBuilderError,
    MODULE10_REQUIRED_COLUMNS,
    REQUIRED_COLUMNS,
    build_dataset,
    load_chunk_data,
    load_dataset,
    load_embedding_data,
    merge_chunks_and_embeddings,
    save_dataset,
    validate_dataset,
)

__all__ = [
    "build_dataset",
    "save_dataset",
    "load_dataset",
    "validate_dataset",
    "load_chunk_data",
    "load_embedding_data",
    "merge_chunks_and_embeddings",
    "REQUIRED_COLUMNS",
    "MODULE10_REQUIRED_COLUMNS",
    "DatasetBuilderError",
    "CourseCatalog",
    "CourseLecture",
    "compute_file_hash",
]
