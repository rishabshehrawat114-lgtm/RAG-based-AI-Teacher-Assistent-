# Project Overview: RAG-Based AI Teaching Assistant

## Executive Summary

The **RAG-Based AI Teaching Assistant** is a multimodal, multi-lecture educational question-answering and study-support system. It transforms raw, unstructured lecture video recordings (`.mp4`, `.mov`, `.mkv`, etc.) and audio lectures into an indexed, searchable, and conversational course knowledge base.

Students can ask open-ended conceptual questions or specific factual questions, filter their scope to an individual lecture or the complete course ("All Lectures"), and receive pedagogical explanations grounded in transcript excerpts with exact timecodes (`MM:SS – MM:SS`).

To prevent the model from hallucinating or answering out-of-scope queries (e.g., questions not covered in the curriculum), the system implements a strict **Cosine Similarity Relevance Gate** (`threshold = 0.3200`). When the retrieved content does not meet this threshold, the assistant abstains from guessing and responds with a standard educational fallback message:
> *"I couldn't find enough information about this topic in the provided course material."*

In addition to question-answering, the application features an integrated **Learning Center** (generating lecture summaries, key concepts, step-by-step topic explanations, and interactive multiple-choice practice quizzes) and a dedicated **RAG Evaluation Suite** benchmarked against a 20-question course test set.

---

## Problem Statement

Educational videos and recorded lectures are among the most valuable digital learning resources, but they have fundamental usability limitations:
1. **Linear and Opaque:** Videos cannot be quickly skimmed, grepped, or indexed by standard search engines like text textbooks.
2. **High Search Cost:** A student looking for a specific explanation (e.g., *"How is memory allocated for dynamic arrays?"*) often spends 10–20 minutes scrubbing through video timelines.
3. **General-Purpose LLM Hallucinations:** When students query general-purpose commercial chatbots (e.g., ChatGPT), the models frequently hallucinate concepts not taught in the course, use terminology conflicting with the professor's syllabus, or provide inaccurate answers without attribution.
4. **Lack of Provenance:** Generic chatbots provide no verifiable citations linking directly back to the professor's spoken words or video timestamp.

---

## Target Users

1. **University & College Students:** Seeking quick clarification on lecture topics, revision summaries before exams, and practice quizzes directly grounded in their instructor's recordings.
2. **Professors & Teaching Assistants (TAs):** Offloading repetitive introductory Q&A while maintaining strict curriculum fidelity and preventing unverified AI responses.
3. **Online Course Platforms & Bootcamps:** Providing autonomous, 24/7 educational support for asynchronous learners.

---

## Main Objectives

- **Curriculum Grounding:** Ensure answers are derived strictly from the ingested lecture transcripts.
- **Accurate Temporal Attribution:** Pair every answer with source lecture titles, filenames, and second-level timestamp intervals (`MM:SS – MM:SS`).
- **Hallucination Suppression via Abstention:** Reject unsupported, out-of-scope queries using a calibrated cosine similarity threshold.
- **Multi-Lecture Scope:** Enable students to query either a specific lecture or the entire course catalog.
- **Study Tooling:** Provide instant automated summaries, key concept glossaries, lecture walkthroughs, and practice quizzes.
- **Empirical Evaluation:** Track objective retrieval and answer quality metrics (Recall@K, Hit@K, MRR, FAR, FRR, Citation Validity).
- **100% Offline Capability:** Support deterministic offline operation (direct quote synthesis with citations) when external LLM APIs are unavailable.

---

## Key Features

- **Media Ingestion & Deduplication:** Ingests video and audio files with SHA-256 content hashing to block redundant indexing regardless of filename changes.
- **Whisper Speech-to-Text:** Automatic speech recognition using OpenAI Whisper (`base` model by default), preserving word-level timing and speaker pacing.
- **Timestamped Sliding-Window Chunking:** Chunks transcript words (default: 700 words per chunk, 100-word overlap) while computing start and end timestamps.
- **Dense Vector Search:** Employs Sentence Transformers (`all-MiniLM-L6-v2`) generating 384-dimensional dense embeddings for cosine similarity retrieval.
- **Relevance / Abstention Gate:** Cosine threshold (`0.3200`) separating in-scope educational queries from unrelated distractions.
- **Student Chat with Contextual Awareness:** Preserves conversation history across turns, enabling natural follow-up queries (e.g., *"Can you give an example of that?"*).
- **Custom Response Styles:** Tailors output format across **Explain Clearly** (student-friendly), **Short Answer** (concise 2–3 sentences), and **Detailed Explanation** (in-depth breakdown).
- **Learning Center:**
  - *Lecture Summary:* Automated executive overview and core takeaways.
  - *Key Concepts:* Important terminology, definitions, and principles.
  - *Explain This Lecture:* Progressive conceptual walkthrough.
  - *Quiz Practice:* Interactive multiple-choice questions with real-time feedback and explanation citations.
- **Quality Benchmarking Suite:** Built-in 20-question evaluation benchmark measuring Recall@1/3/5, Hit@5, MRR, FAR, FRR, and Citation Validity.
- **Production-Ready Architecture:** Centralized settings, dynamic path portability, health check diagnostics, and container-ready configuration.

---

## End-to-End Workflow

```text
[Media Input] (.mp4 / .mp3)
       │
       ▼
[FFmpeg Audio Extraction] ──► 16kHz Mono MP3
       │
       ▼
[Whisper Transcription] ──► Raw Transcripts with Timecodes
       │
       ▼
[Text Cleaning & Validation] ──► Normalized Educational Transcript
       │
       ▼
[Timestamped Chunking] ──► Chunks tagged with lecture_id, source, timecodes
       │
       ▼
[Dense Embeddings] ──► MiniLM-L6-v2 Vector Representations
       │
       ▼
[Course Catalog Registration] ──► Multi-Lecture Registry & Deduplication
       │
═══════╪════════════════════════════════════════════════════════════════
       │
[Student Question]
       │
       ▼
[Query Encoding & Cosine Retrieval] ──► Top-K Relevant Chunks
       │
       ▼
[Relevance Gate Check (0.3200)]
  ├── Score < 0.3200 ──► Graceful Abstention ("I couldn't find enough info...")
  └── Score >= 0.3200 ──► Context Assembly (Passages + Follow-up History)
                               │
                               ▼
            [Pedagogical Synthesis (LLM or Local Mode)]
                               │
                               ▼
        [Educational Answer + Exact MM:SS Timestamp Citations]
```

---

## Major Technologies

| Technology | Purpose |
| :--- | :--- |
| **Python 3.9+** | Core language for data processing, retrieval, and application logic |
| **Streamlit** | Interactive web interface for chat, study hub, and evaluation dashboards |
| **OpenAI Whisper** | Timestamp-accurate automatic speech recognition |
| **FFmpeg** | Speech-optimized 16kHz mono audio extraction from videos |
| **Sentence-Transformers** | Dense semantic embedding generation (`all-MiniLM-L6-v2`, 384 dimensions) |
| **NumPy & Pandas** | High-performance matrix similarity operations and dataset management |
| **Joblib** | Compressed binary dataset caching for fast startup |
| **OpenAI API (gpt-4o-mini)** | Grounded educational synthesis (with deterministic local fallback) |
| **Pytest** | Automated test suite (179 tests, 100% pass rate) |
| **Docker** | Containerization specification for reproducible server deployment |

---

## Current Limitations

1. **Benchmark Size:** The current benchmark contains 20 representative course queries. While rigorous, broader curriculum coverage across multiple domains would benefit from 100+ questions.
2. **CPU Inference Latency:** Whisper transcription and Sentence-Transformers inference run on CPU by default. Transcribing a 1-hour lecture takes approximately 3–6 minutes on standard consumer hardware.
3. **Retrieval Boundary Case (`eval_012`):** Query `eval_012` (*"What department has she been lecturing in?"*) retrieves relevant context with a top similarity of `0.2970`. Because `0.2970 < 0.3200`, the relevance gate triggers a false rejection. The threshold is deliberately held constant to prevent overfitting.
4. **External LLM Quota Dependence:** Full generative natural language answers require an active OpenAI API key with credit balance; otherwise, the system operates in local deterministic mode (direct quotes and citations).

---

## Future Improvements

- **Cross-Encoder Reranking:** Adding a secondary reranker (e.g., `bge-reranker-base`) to re-score top-20 retrieved candidates before gating.
- **Hybrid Search:** Fusing BM25 sparse keyword scores with dense vector similarities via Reciprocal Rank Fusion (RRF).
- **Query Rewriting (HyDE):** Generating hypothetical passage answers to bridge vocabulary mismatches between student queries and spoken transcripts.
- **Dedicated Vector Database:** Transitioning from in-memory NumPy vector arrays to an embedded or standalone vector database (e.g., Qdrant, Chroma) for scaling to thousands of lectures.
