"""
src/embeddings
==============
Vector embeddings module for the RAG AI Teacher Assistant.

Converts transcript chunks created in Module 5 into dense numerical embedding
vectors using local Sentence Transformers models (e.g. all-MiniLM-L6-v2).
Saves numerical vectors as compressed NumPy matrices (.npz) paired with
human-readable JSON metadata for seamless, deterministic retrieval.
"""

from src.embeddings.embedder import (
    EmbeddingError,
    batch_embed_chunks,
    embed_chunk_file,
    embed_chunks_data,
    generate_embeddings,
    get_embedding_dimension,
    load_embedding_model,
    load_embeddings,
    save_embeddings,
)

__all__ = [
    "load_embedding_model",
    "get_embedding_dimension",
    "generate_embeddings",
    "embed_chunks_data",
    "embed_chunk_file",
    "batch_embed_chunks",
    "save_embeddings",
    "load_embeddings",
    "EmbeddingError",
]
