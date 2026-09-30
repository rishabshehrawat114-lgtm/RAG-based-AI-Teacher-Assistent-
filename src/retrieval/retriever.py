"""
src/retrieval/retriever.py
==========================
Semantic similarity retrieval and chunk ranking for the RAG AI Teacher Assistant.

Retrieves the top-K most relevant lecture passages for student queries using
dense vector cosine similarity computed against pre-embedded lecture chunks.

Key Capabilities:
-----------------
1. Dense Semantic Matching: Converts query text into a vector using the exact same
   Sentence Transformers model configured in Module 6 (all-MiniLM-L6-v2).
2. Pure Cosine Similarity: Computes normalized vector dot products without requiring
   an external vector database.
3. Multi-Lecture Aggregation: Seamlessly searches across single or multiple lecture
   embedding archives in `data/embeddings/`.
4. Provenance Preservation: Returns source video filename, transcript references,
   millisecond start/end timestamps, and exact similarity scores.
5. Configurable Top-K & Threshold: Configurable number of return items and optional
   relevance cutoff thresholds.
6. Optional Source Filtering: Allows students to scope queries to a specific video/lecture.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    CONSOLIDATED_DATASET_PATH,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_DIR,
)
from src.embeddings import (
    EmbeddingError,
    generate_embeddings,
    load_embedding_model,
)

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Custom exception raised when retrieval, query embedding, or corpus loading fails."""
    pass


def format_timestamp(seconds: float | int, include_ms: bool = False) -> str:
    """
    Formats numeric seconds into human-readable MM:SS or HH:MM:SS format.

    Args:
        seconds: Time duration or offset in numeric seconds.
        include_ms: If True, appends tenths of a second (e.g. '02:05.5').

    Returns:
        Formatted time string (e.g. '02:05', '01:14:22').

    Examples:
        >>> format_timestamp(125.4)
        '02:05'
        >>> format_timestamp(3665.0)
        '01:01:05'
    """
    if seconds is None or seconds < 0:
        seconds = 0.0

    total_secs = float(seconds)
    hours = int(total_secs // 3600)
    minutes = int((total_secs % 3600) // 60)
    secs = int(total_secs % 60)
    fraction = int(round((total_secs - int(total_secs)) * 10))

    if hours > 0:
        base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        base = f"{minutes:02d}:{secs:02d}"

    if include_ms and fraction > 0:
        return f"{base}.{fraction}"
    return base


def compute_cosine_similarity(
    query_vector: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Computes cosine similarity between a 1D query vector and a 2D chunk embedding matrix.

    Formula:
        similarity(q, c) = (q · c) / (||q|| * ||c||)

    Guarantees numerical stability and prevents division by zero.
    Returns values strictly clipped to the [-1.0, 1.0] range.

    Args:
        query_vector: 1D NumPy array (shape: (D,)) or 2D array (shape: (1, D)).
        chunk_embeddings: 2D NumPy array of shape (N, D).

    Returns:
        1D NumPy array of shape (N,) containing float32 similarity scores.

    Raises:
        RetrievalError: If vector shapes or dimensions do not match.
    """
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    c = np.asarray(chunk_embeddings, dtype=np.float32)

    if c.ndim != 2:
        raise RetrievalError(f"Expected 2D chunk embeddings matrix, got shape {c.shape}.")

    if c.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    if c.shape[1] != q.shape[0]:
        raise RetrievalError(
            f"Embedding dimension mismatch: query vector has dimension {q.shape[0]}, "
            f"but chunk embeddings have dimension {c.shape[1]}."
        )

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        logger.warning("Query vector has zero norm; returning zero similarities.")
        return np.zeros(c.shape[0], dtype=np.float32)

    c_norms = np.linalg.norm(c, axis=1)
    # Prevent division by zero for zero-length chunk vectors
    c_norms = np.where(c_norms == 0.0, 1e-10, c_norms)

    # Dot product
    dots = np.dot(c, q)
    similarities = dots / (c_norms * q_norm)

    return np.clip(similarities, -1.0, 1.0).astype(np.float32)


def load_all_lecture_embeddings(
    embeddings_dir: Path | str = EMBEDDING_DIR,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """
    Loads and aggregates all embedding archives (.npz + .json) from the target directory.

    Enforces strict alignment between vector rows and chunk metadata records.
    Supports querying across multiple lectures simultaneously.

    Args:
        embeddings_dir: Directory containing *_embeddings.npz and *_metadata.json files.

    Returns:
        Tuple of:
          - all_embeddings: 2D NumPy array of shape (Total_Chunks, Dimension)
          - all_chunks: Flat list of chunk metadata dictionaries corresponding to each row

    Raises:
        RetrievalError: If directory is missing, no embeddings are found, or files are misaligned.
    """
    target_dir = Path(embeddings_dir)
    if not target_dir.exists():
        raise RetrievalError(f"Embeddings directory does not exist: {target_dir}")

    npz_files = sorted(
        [f for f in target_dir.glob("*_embeddings.npz") if not f.name.startswith(".")]
    )

    if not npz_files:
        raise RetrievalError(
            f"No embedding archives (*_embeddings.npz) found in '{target_dir}'. "
            "Please run 'python scripts/create_embeddings.py' first."
        )

    matrices: list[np.ndarray] = []
    all_chunks: list[dict[str, Any]] = []
    expected_dim: int | None = None

    for npz_path in npz_files:
        stem = npz_path.stem.replace("_embeddings", "")
        json_path = target_dir / f"{stem}_metadata.json"

        if not json_path.exists():
            raise RetrievalError(
                f"Missing metadata JSON for embedding archive '{npz_path.name}'. "
                f"Expected '{json_path.name}' in '{target_dir}'."
            )

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if "embeddings" not in data:
                    raise RetrievalError(f"Archive '{npz_path.name}' is missing 'embeddings' array.")
                embeddings = np.asarray(data["embeddings"], dtype=np.float32)

            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        except Exception as exc:
            raise RetrievalError(f"Failed to load embedding pair '{stem}': {exc}") from exc

        chunks = metadata.get("chunks", [])
        if embeddings.shape[0] != len(chunks):
            raise RetrievalError(
                f"Index alignment mismatch in '{stem}': {embeddings.shape[0]} vector rows "
                f"vs {len(chunks)} metadata chunks."
            )

        # Check dimension consistency across multiple lectures
        file_dim = embeddings.shape[1] if embeddings.shape[0] > 0 else metadata.get("embedding_dimension")
        if file_dim is not None:
            if expected_dim is None:
                expected_dim = int(file_dim)
            elif expected_dim != file_dim:
                raise RetrievalError(
                    f"Dimension mismatch between lecture archives: previous files have dimension "
                    f"{expected_dim}, but '{npz_path.name}' has dimension {file_dim}."
                )

        if embeddings.shape[0] > 0:
            for c in chunks:
                if not c.get("lecture_id"):
                    c["lecture_id"] = stem
                if not c.get("lecture_title"):
                    c["lecture_title"] = metadata.get("lecture_title") or stem.replace("_", " ").title()
            matrices.append(embeddings)
            all_chunks.extend(chunks)

    if not matrices:
        dim = expected_dim or 384
        return np.empty((0, dim), dtype=np.float32), []

    stacked_embeddings = np.vstack(matrices)
    logger.debug(
        "Loaded %d lecture archive(s): %d total chunks, dimension %d",
        len(npz_files),
        stacked_embeddings.shape[0],
        stacked_embeddings.shape[1],
    )

    return stacked_embeddings, all_chunks


class LectureRetriever:
    """
    In-memory semantic retriever for educational lecture transcripts.

    Loads precomputed chunk embeddings once, caches the Sentence Transformers model,
    and performs fast cosine similarity ranking for student queries.
    """

    def __init__(
        self,
        embeddings_dir: Path | str = EMBEDDING_DIR,
        dataset_path: Path | str | None = None,
        model_name: str | None = None,
        model: Any = None,
        auto_load: bool = True,
    ) -> None:
        """
        Initializes the retriever.

        Args:
            embeddings_dir: Directory containing generated embedding archives.
            dataset_path: Optional path to consolidated Joblib dataset file.
            model_name: Name of Sentence Transformers model (defaults to config).
            model: Optional pre-loaded model (useful for unit tests and custom embeddings).
            auto_load: If True, immediately loads all embeddings into memory.
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self._model = model
        self.embeddings: np.ndarray | None = None
        self.chunks_metadata: list[dict[str, Any]] = []

        if auto_load:
            self.load_corpus()

    @property
    def model(self) -> Any:
        """Lazy-loads and caches the embedding model if not provided at initialization."""
        if self._model is None:
            self._model = load_embedding_model(self.model_name)
        return self._model

    def warmup(self) -> None:
        """Pre-warms the SentenceTransformer model to eliminate cold-start latency on the first user query."""
        _ = self.model

    def load_corpus(self) -> None:
        """Loads or reloads all lecture embedding files from disk into memory."""
        # 1. Prefer consolidated Joblib dataset if explicitly provided,
        # or if using the default EMBEDDING_DIR and the consolidated dataset exists
        use_default_joblib = (
            self.dataset_path is None
            and self.embeddings_dir == EMBEDDING_DIR
            and CONSOLIDATED_DATASET_PATH.exists()
        )
        target_dataset = (
            self.dataset_path
            if self.dataset_path is not None
            else (CONSOLIDATED_DATASET_PATH if use_default_joblib else None)
        )

        if target_dataset and target_dataset.exists():
            try:
                from src.data.dataset_builder import load_dataset
                df = load_dataset(target_dataset, validate=False)
                if len(df) > 0:
                    self.embeddings = np.vstack(df["embedding"].values)
                    cols_to_keep = [c for c in df.columns if c != "embedding"]
                    self.chunks_metadata = df[cols_to_keep].to_dict(orient="records")
                    logger.info(
                        "Corpus loaded from Joblib dataset '%s': %d chunks.",
                        target_dataset.name,
                        len(self.chunks_metadata),
                    )
                    return
            except Exception as exc:
                logger.warning(
                    "Failed to load from Joblib dataset '%s' (%s). Falling back to raw embedding archives.",
                    target_dataset,
                    exc,
                )

        # 2. Fallback to scanning raw archives in embeddings_dir
        self.embeddings, self.chunks_metadata = load_all_lecture_embeddings(self.embeddings_dir)
        logger.info(
            "Corpus loaded: %d total chunk(s) across '%s'.",
            len(self.chunks_metadata),
            self.embeddings_dir,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encodes a single student query string into a normalized 1D vector.

        Args:
            query: The question text entered by the student.

        Returns:
            1D NumPy array representing the query embedding.

        Raises:
            RetrievalError: If query is empty or model encoding fails.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            raise RetrievalError("Query cannot be empty or whitespace only.")

        try:
            vec = generate_embeddings(
                texts=[cleaned],
                model=self.model,
                model_name=self.model_name,
                normalize=True,
                show_progress_bar=False,
            )
            return vec[0]
        except Exception as exc:
            raise RetrievalError(f"Failed to generate query embedding: {exc}") from exc

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float | None = DEFAULT_SIMILARITY_THRESHOLD,
        source_file: str | None = None,
        lecture_filter: str | None = None,
        lecture_id: str | None = None,
        lecture_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieves the top-K most semantically similar lecture chunks for a given query.

        Supports multi-lecture filtering across the unified course knowledge base.

        Args:
            query: The student's question string.
            top_k: Number of highest-scoring chunks to return (default: 5).
            similarity_threshold: Minimum cosine similarity score required (optional).
            source_file: Optional source video/transcript name to restrict search to.
            lecture_filter: Optional lecture ID or title to restrict search to.
            lecture_id: Optional specific lecture ID filter.
            lecture_scope: Optional lecture scope ("All Lectures" or specific lecture).

        Returns:
            List of result dictionaries sorted in descending order of similarity score.
            Each dictionary contains rank, chunk_id, similarity_score, text, timestamps,
            and Module 10 course/lecture metadata (course_id, lecture_id, lecture_title, etc.).

        Raises:
            RetrievalError: If query is invalid or corpus is unavailable.
        """
        if top_k <= 0:
            raise RetrievalError(f"top_k must be a positive integer, got {top_k}.")

        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise RetrievalError("Search query cannot be empty or whitespace only.")

        if self.embeddings is None:
            self.load_corpus()

        if self.embeddings is None or len(self.chunks_metadata) == 0:
            logger.warning("No lecture chunks available in corpus to retrieve from.")
            return []

        # 1. Optional lecture / source filtering before ranking
        filter_val = lecture_scope or lecture_filter or lecture_id or source_file
        if filter_val and str(filter_val).strip() not in ("All Lectures", "All", "all", ""):
            filter_norm = str(filter_val).strip().lower()
            target_source = Path(str(filter_val)).name.lower()

            candidate_indices: list[int] = []
            for i, c in enumerate(self.chunks_metadata):
                c_lid = str(c.get("lecture_id", "")).strip().lower()
                c_ltitle = str(c.get("lecture_title", "")).strip().lower()
                c_sfile = Path(str(c.get("source_file", ""))).name.lower()
                c_sfilename = Path(str(c.get("source_filename", ""))).name.lower()
                c_stranscript = Path(str(c.get("source_transcript", ""))).name.lower()

                c_stem = Path(c_sfile).stem.lower() if c_sfile else ""
                c_sfn_stem = Path(c_sfilename).stem.lower() if c_sfilename else ""

                match = (
                    filter_norm == c_lid
                    or filter_norm == c_ltitle
                    or filter_norm == c_stem
                    or filter_norm == c_sfn_stem
                    or target_source in (c_sfile, c_sfilename, c_stranscript, c_stem, c_sfn_stem)
                    or (c_ltitle and (filter_norm in c_ltitle or c_ltitle in filter_norm))
                    or (c_lid and filter_norm in c_lid)
                )
                if match:
                    candidate_indices.append(i)

            if not candidate_indices:
                logger.info("No chunks matched lecture filter: '%s'", filter_val)
                return []
            candidate_embeddings = self.embeddings[candidate_indices]
            candidate_chunks = [self.chunks_metadata[i] for i in candidate_indices]
        else:
            candidate_embeddings = self.embeddings
            candidate_chunks = self.chunks_metadata

        # 2. Generate query embedding
        query_vec = self.embed_query(cleaned_query)

        # 3. Calculate cosine similarities
        scores = compute_cosine_similarity(query_vec, candidate_embeddings)

        # 4. Filter by similarity threshold if specified
        if similarity_threshold is not None:
            qualifying_mask = scores >= float(similarity_threshold)
            if not np.any(qualifying_mask):
                logger.info(
                    "No chunks exceeded the similarity threshold of %.4f",
                    similarity_threshold,
                )
                return []
            candidate_indices_filtered = np.where(qualifying_mask)[0]
            scores = scores[candidate_indices_filtered]
            candidate_chunks = [candidate_chunks[i] for i in candidate_indices_filtered]

        if len(candidate_chunks) == 0:
            return []

        # 5. Sort candidates in descending order of similarity
        sorted_order = np.argsort(-scores)
        top_indices = sorted_order[:top_k]

        # 6. Construct structured results with provenance preserved
        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            chunk = candidate_chunks[idx]
            sim_score = float(scores[idx])
            start_s = float(chunk.get("start_time", 0.0))
            end_s = float(chunk.get("end_time", 0.0))

            result_item: dict[str, Any] = {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id", f"chunk_{rank:04d}"),
                "similarity_score": sim_score,
                "text": chunk.get("text", ""),
                "source_file": chunk.get("source_file", ""),
                "source_transcript": chunk.get("source_transcript", ""),
                "start_time": start_s,
                "end_time": end_s,
                "language": chunk.get("language", "en"),
                # Module 10 lecture provenance
                "course_id": chunk.get("course_id", "python_course"),
                "lecture_id": chunk.get("lecture_id", ""),
                "lecture_title": chunk.get("lecture_title", chunk.get("source_file", "Lecture")),
                "source_filename": chunk.get("source_filename", chunk.get("source_file", "")),
                "timestamp_start": chunk.get("timestamp_start", format_timestamp(start_s)),
                "timestamp_end": chunk.get("timestamp_end", format_timestamp(end_s)),
                "timestamp": chunk.get("timestamp", f"{format_timestamp(start_s)} - {format_timestamp(end_s)}"),
            }
            results.append(result_item)

        return results


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: float | None = DEFAULT_SIMILARITY_THRESHOLD,
    source_file: str | None = None,
    lecture_filter: str | None = None,
    lecture_id: str | None = None,
    lecture_scope: str | None = None,
    embeddings_dir: Path | str = EMBEDDING_DIR,
    dataset_path: Path | str | None = None,
    model_name: str | None = None,
    model: Any = None,
) -> list[dict[str, Any]]:
    """
    Convenience function for one-off retrieval without managing a persistent LectureRetriever.

    Args:
        query: The student's question string.
        top_k: Number of highest-scoring chunks to return.
        similarity_threshold: Optional minimum cosine similarity score.
        source_file: Optional source video/transcript filter.
        embeddings_dir: Target directory containing precomputed embeddings.
        dataset_path: Optional path to consolidated Joblib dataset file.
        model_name: Model name to load if model is not passed.
        model: Pre-loaded model or None.

    Returns:
        List of ranked result dictionaries.
    """
    retriever = LectureRetriever(
        embeddings_dir=embeddings_dir,
        dataset_path=dataset_path,
        model_name=model_name,
        model=model,
        auto_load=True,
    )
    return retriever.retrieve(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        source_file=source_file,
        lecture_filter=lecture_filter,
        lecture_id=lecture_id,
        lecture_scope=lecture_scope,
    )


def format_retrieval_results(
    results: list[dict[str, Any]],
    query: str = "",
    top_k: int | None = None,
) -> str:
    """
    Formats retrieved lecture chunks into a clean, human-readable terminal presentation.

    Args:
        results: List of result dictionaries returned by retrieve().
        query: Optional question text for the header.
        top_k: Optional requested top_k value for the header.

    Returns:
        Formatted multi-line display string.
    """
    lines: list[str] = []
    lines.append("==================================================")
    if query:
        lines.append("Query:")
        lines.append(query)
        lines.append("")

    if not results:
        lines.append("No sufficiently relevant lecture chunks were found.")
        lines.append("==================================================")
        return "\n".join(lines)

    display_k = len(results) if top_k is None else min(top_k, len(results))
    lines.append(f"Top {display_k} results:")
    lines.append("")

    for item in results[:display_k]:
        rank = item.get("rank", 1)
        score = item.get("similarity_score", 0.0)
        source = item.get("source_file", "unknown")
        start_fmt = format_timestamp(item.get("start_time", 0.0))
        end_fmt = format_timestamp(item.get("end_time", 0.0))
        text = item.get("text", "").strip()

        lines.append(f"[{rank}] Similarity: {score:.4f}")
        lines.append(f"Source: {source}")
        lines.append(f"Time: {start_fmt} - {end_fmt}")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("==================================================")
    return "\n".join(lines)
