# Module-by-Module Development History

This document outlines the evolutionary development of the **RAG-Based AI Teaching Assistant**, tracking how each module incrementally added verified capabilities to the system.

---

### Module 1 — Project Foundation & Architecture
- **Purpose:** Establish the project layout, directory scaffolding, path resolution, and dependency definitions.
- **Major Implementation:** Designed cross-platform `pathlib.Path` structures, created `.env.example`, `.gitignore`, and data directory scaffolding (`data/videos`, `data/audio`, `data/transcripts`, `data/chunks`, `data/embeddings`, `data/processed`).
- **Important Files:** `src/config/`, `src/utils.py`, `requirements.txt`.
- **Capability Added:** Structured, reproducible codebase skeleton running across Windows, macOS, and Linux.

---

### Module 2 — Media Ingestion & Audio Extraction
- **Purpose:** Convert educational video files into compact, speech-optimized audio files for speech-to-text processing.
- **Major Implementation:** Integrated FFmpeg subprocess wrapper to transcode video (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`) into 16,000 Hz, mono channel, 64 kbps MP3 files.
- **Important Files:** `src/audio/extractor.py`, `scripts/convert_videos.py`.
- **Capability Added:** Automated video audio extraction reducing file sizes by ~90% while standardizing acoustic inputs.

---

### Module 3 — Timestamped Speech Transcription
- **Purpose:** Transcribe spoken audio into text while retaining exact temporal intervals for every sentence.
- **Major Implementation:** Wrapped OpenAI Whisper (`base` model), capturing word- and segment-level timestamps (`start`, `end`, `text`).
- **Important Files:** `src/transcription/transcriber.py`, `scripts/transcribe_audio.py`.
- **Capability Added:** Automated speech-to-text generating timestamped JSON transcript files.

---

### Module 4 — Transcript Cleaning & Validation
- **Purpose:** Clean and normalize spoken transcripts to prepare them for NLP and semantic chunking.
- **Major Implementation:** Removed speech artifacts, normalized punctuation and whitespace, validated transcript continuity, and calculated audio duration metrics.
- **Important Files:** `src/transcription/processor.py`, `scripts/process_transcripts.py`.
- **Capability Added:** High-quality, normalized transcript representations preserving verbatim educational content.

---

### Module 5 — Semantic & Temporal Chunking
- **Purpose:** Segment lengthy transcripts into cohesive passages suitable for embedding models while maintaining temporal provenance.
- **Major Implementation:** Built a word-based sliding-window chunker (700 words, 100-word overlap). Reconstructed exact start and end timestamps (`MM:SS`) for each chunk based on underlying speech segments.
- **Important Files:** `src/chunking/chunker.py`, `scripts/chunk_transcripts.py`.
- **Capability Added:** Timestamped text chunks carrying full metadata (chunk ID, source file, timecodes).

---

### Module 6 — Dense Vector Embeddings
- **Purpose:** Map text chunks into dense continuous vector space for semantic similarity search.
- **Major Implementation:** Integrated Sentence-Transformers (`all-MiniLM-L6-v2`), generating 384-dimensional L2-normalized embeddings saved as compressed NumPy archives (`.npz`) and parallel metadata JSONs.
- **Important Files:** `src/embeddings/embedder.py`, `scripts/create_embeddings.py`.
- **Capability Added:** Dense semantic vector representations enabling cosine similarity operations.

---

### Module 7 — Semantic Vector Retrieval
- **Purpose:** Execute fast cosine similarity searches against the chunk vector store for incoming student questions.
- **Major Implementation:** Query vector encoding, dot-product similarity computation over normalized vectors, top-$K$ candidate extraction, and similarity score sorting.
- **Important Files:** `src/retrieval/retriever.py`, `scripts/query_retriever.py`.
- **Capability Added:** Sub-second retrieval of top-$K$ most relevant lecture passages.

---

### Module 8 — Bulk Dataset Consolidation
- **Purpose:** Consolidate distributed per-lecture embedding files into a single unified in-memory dataset.
- **Major Implementation:** Aggregated chunks, metadata, and embedding matrices into a compressed Joblib dataset (`teacher_embeddings.joblib`).
- **Important Files:** `src/data/dataset_builder.py`, `scripts/build_dataset.py`.
- **Capability Added:** Instant $O(1)$ startup vector loading without reading dozens of individual JSON files on every launch.

---

### Module 9 — Grounded RAG Pipeline & Relevance Gate
- **Purpose:** Enforce hallucination-free answer synthesis and implement a relevance abstention gate.
- **Major Implementation:** Built `RAGTeacherPipeline` with prompt engineering instructing the LLM to rely strictly on course material. Added the cosine similarity relevance gate (`DEFAULT_RELEVANCE_THRESHOLD = 0.3200`), returning a standard fallback message when queries are out-of-scope. Added local synthesis fallback when API keys are absent.
- **Important Files:** `src/rag/pipeline.py`, `src/rag/prompt.py`, `src/llm/client.py`.
- **Capability Added:** Grounded educational answers with timecode citations and reliable abstention on unsupported topics.

---

### Module 10 — Multi-Lecture Knowledge Base
- **Purpose:** Expand the system from a single recording to a full multi-lecture course repository.
- **Major Implementation:** Built `CourseCatalog` (`src/data/course_catalog.py`) with SHA-256 duplicate detection, lecture IDs, lecture titles, incremental indexing, and lecture-specific filtering.
- **Important Files:** `src/data/course_catalog.py`, `data/course_catalog.json`.
- **Capability Added:** Multi-lecture course management with duplicate detection and filtered or global ("All Lectures") retrieval.

---

### Module 11 — Student Chat Experience
- **Purpose:** Transform one-off question answering into an interactive, contextual educational dialogue.
- **Major Implementation:** Preserved dialogue turns for context-aware follow-up resolution, implemented customizable teaching styles (**Explain Clearly**, **Short Answer**, **Detailed Explanation**), added expandable developer diagnostics, and created session state controls (Clear/New Chat).
- **Important Files:** `app.py`, `src/rag/prompt.py`.
- **Capability Added:** Conversational educational chat assistant with follow-up awareness and style controls.

---

### Module 12 — Learning Center Features
- **Purpose:** Provide specialized study tooling beyond open-ended conversational chat.
- **Major Implementation:** Created four distinct pedagogical modules:
  1. *Lecture Summary:* Automated overview and key takeaways.
  2. *Key Concepts:* Important definitions and principles.
  3. *Explain This Lecture:* Progressive, beginner-friendly conceptual walkthrough.
  4. *Quiz & Practice:* Interactive 4-option multiple-choice quizzes with real-time scoring, answer keys, and citation links.
- **Important Files:** `src/learning/summary.py`, `src/learning/concepts.py`, `src/learning/explain.py`, `src/learning/quiz.py`.
- **Capability Added:** Integrated study center transforming raw lectures into interactive learning materials.

---

### Module 13 — RAG Evaluation & Benchmarking
- **Purpose:** Establish quantitative, reproducible benchmarking for retrieval accuracy, relevance gating, and answer validity.
- **Major Implementation:** Built `RAGEvaluator` running against a 20-question ground-truth benchmark (`data/evaluation/rag_benchmark.json`). Implemented metrics: Recall@K, Hit@K, MRR, Lecture Accuracy, Decision Accuracy, FAR, FRR, Citation Validity, and Deterministic Answer Checks. Built an interactive Streamlit evaluation dashboard.
- **Important Files:** `src/evaluation/evaluator.py`, `src/evaluation/metrics.py`, `src/evaluation/report.py`, `scripts/run_evaluation.py`.
- **Capability Added:** Objective quality measurement validating system performance (100% Hit@5, 0.00% FAR, 100% Citation Validity).

---

### Module 14 — Deployment & Production Cleanup
- **Purpose:** Transition the development prototype into a clean, portable, and deployment-ready software package.
- **Major Implementation:**
  - Centralized modular configuration in `src/config/`.
  - Added portable dynamic path resolution to `CourseCatalog` to heal cross-machine paths.
  - Implemented upload validation (empty check, 200MB size limit, path traversal defense via `sanitize_filename`).
  - Added secret masking (`mask_secret`) and safe Streamlit secrets support (`st.secrets` fallback).
  - Built startup health validation (`src/health.py`) and cross-platform launcher (`scripts/run_app.py`).
  - Created Docker specification (`Dockerfile`, `.dockerignore`, `requirements-prod.txt`).
  - Expanded test suite to 179 passing automated tests.
- **Important Files:** `src/config/settings.py`, `src/health.py`, `scripts/run_app.py`, `Dockerfile`, `tests/test_production.py`.
- **Capability Added:** Portable, secure, reproducible production deployment architecture.
