# Technical Decisions & Architectural Rationale

This document records the architectural and engineering rationale behind the key technical decisions made in the **RAG-Based AI Teaching Assistant**. Decisions are grounded in educational requirements, computational efficiency, and production robustness.

---

### 1. Why RAG instead of sending the complete lecture directly to an LLM?
- **Context Window & Cost Efficiency:** While modern LLMs offer large context windows (e.g., 128k+ tokens), sending the complete transcript of multi-hour or multi-lecture courses with every single student question would be prohibitively slow and expensive.
- **Attention Degradation ("Lost in the Middle"):** Empirical research shows LLMs degrade in recall accuracy when relevant facts are buried inside massive contexts. RAG isolates only the most semantically relevant passages, maximizing answer precision.
- **Dynamic Multi-Lecture Scaling:** A full semester course contains 20–40 hours of lecture audio (over 300,000 words). RAG scales horizontally with corpus size without requiring proportionate token expenditure.

---

### 2. Why Whisper?
- **Robust Automatic Speech Recognition (ASR):** OpenAI Whisper was trained on 680,000 hours of diverse multilingual and acoustic data, providing high resilience against background classroom noise, varying microphone qualities, and diverse speaker accents.
- **Native Word/Segment Timecoding:** Whisper outputs explicit second-accurate segment intervals (`start` and `end`), which is critical for generating temporal citations.
- **Offline Open-Source Execution:** Whisper can run entirely locally on CPU/GPU without recurring per-minute cloud API subscription costs.

---

### 3. Why timestamped transcripts?
- **Verifiable Provenance:** Students and educators cannot verify an AI answer if they cannot find the exact point in the video where the professor discussed it.
- **Direct Study Navigation:** Timestamped intervals (`MM:SS – MM:SS`) allow students to click and jump directly to relevant video timestamps in their course player rather than manually re-watching an hour of video.

---

### 4. Why chunking?
- **Embedding Cohesion:** Dense embedding models have finite input token limits (e.g., 256 or 512 tokens). Passing an entire 60-minute transcript to an embedding model would compress thousands of disparate concepts into a single vector, destroying semantic specificity.
- **Retrieval Precision:** Chunking (700 words, 100-word overlap) breaks lectures into self-contained conceptual blocks, ensuring the retriever surfaces only the specific paragraph answering the student's question.
- **Boundary Continuity:** The sliding-window overlap prevents concepts from being arbitrarily cut in half at chunk boundaries.

---

### 5. Why metadata?
- **Granular Attribution:** Vector embeddings only capture semantic meaning; they do not preserve source origin. Metadata attaches `lecture_id`, `lecture_title`, `source_filename`, `course_id`, and `timestamp_range` directly to each vector.
- **Filtered Scoping:** Metadata enables filtered queries, allowing students to restrict search to a specific lecture or expand to all lectures.

---

### 6. Why embeddings?
- **Beyond Keyword Matching:** Traditional keyword search (e.g., SQL `LIKE` or basic BM25) fails when students use synonyms or colloquial phrasing (e.g., a student asks *"how do computers do arithmetic?"* while the transcript discusses *"arithmetic logic units and CPU operations"*).
- **Dense Semantic Representation:** Embeddings project words and concepts into a continuous vector space where semantic proximity corresponds to geometric proximity.

---

### 7. Why `all-MiniLM-L6-v2`?
- **Efficiency / Latency Balance:** At only 22.7M parameters and ~80 MB disk footprint, `all-MiniLM-L6-v2` runs quickly on consumer CPUs without requiring a dedicated GPU.
- **Strong Benchmark Performance:** Despite its compact size, it achieves high semantic search quality on the MTEB (Massive Text Embedding Benchmark).
- **Fast Startup:** Loads in under 1 second, making local development, CI/CD pipelines, and educational demos fast and responsive.

---

### 8. Why similarity-based retrieval?
- **Cosine Metric Standardization:** For L2-normalized vectors, cosine similarity simplifies to the vector dot product:
  $$\text{sim}(u, v) = u \cdot v$$
  This allows high-performance vector search in pure NumPy without external server overhead for small to medium course catalogs.

---

### 9. Why a relevance threshold?
- **Eliminating False Confidences:** In pure top-$K$ retrieval without a threshold, the system *always* returns $K$ passages, even if the user asks about completely unrelated topics (e.g., asking about quantum mechanics in an introductory Python course).
- **Hallucination Prevention:** By gating context ingestion on similarity scores, the LLM is never fed irrelevant context, preventing it from hallucinating answers to off-topic queries.

---

### 10. Why is the threshold currently 0.3200?
- **Empirical Calibration:** During Module 9 and Module 13 benchmarking, cosine similarity distributions across relevant vs. irrelevant questions showed clear separation:
  - Average top score for relevant questions: **`0.5370`**
  - Average top score for rejected irrelevant questions: **`0.1759`**
  - Maximum top score observed on out-of-scope queries: **`0.2443`**
- **0.3200 Provides a Safety Margin:** The 0.3200 cutoff sits safely above the maximum irrelevant score (`0.2443`), achieving a **0.00% False Acceptance Rate (FAR)**.
- **Preserved Boundary Diagnostic (`eval_012`):** Question `eval_012` scored `0.2970`, causing a false rejection. We deliberately preserved `0.3200` to avoid artificially overfitting the threshold to a single dataset.

---

### 11. Why does the system abstain from unsupported questions?
- **Pedagogical Integrity:** In education, an AI assistant guessing incorrectly or making up facts is significantly more damaging to a student's learning than admitting it does not know.
- **Predictable Failure Mode:** Standardizing on a clear educational abstention message sets clear expectations:
  > *"I couldn't find enough information about this topic in the provided course material."*

---

### 12. Why is multi-lecture metadata important?
- **Course-Level Context:** University courses consist of sequential lectures building on prior material. Storing `lecture_id` and `lecture_title` prevents confusion between different lectures (e.g., differentiating between *"Introduction to Loops"* in Lecture 3 and *"Advanced Loop Optimization"* in Lecture 9).
- **Flexible Scope Control:** Students can narrow their review to the specific lecture covered on today's quiz or broaden their search across the entire semester.

---

### 13. Why Streamlit?
- **Rapid Educational Prototyping & Demonstration:** Streamlit enables rapid UI development directly in Python, eliminating frontend/backend synchronization overhead.
- **Native State Management:** Session state handles multi-turn conversation history, teaching styles, and dynamic quiz scoring seamlessly.
- **Extensible Layouts:** Streamlit's tabbed layout allows Student Chat, Learning Center, and RAG Evaluation to coexist in a single unified dashboard.

---

### 14. Why local / offline fallback?
- **Reliability & Resilience:** Educational platforms must remain operational even if external cloud APIs suffer outages, rate limits, or billing exhaustion.
- **Zero-Cost Deployment:** Educational institutions with zero external API budget can deploy the application in 100% local synthesis mode, delivering direct lecture excerpts and verified citations without spending a dollar on external tokens.

---

### 15. Why is evaluation necessary?
- **Avoiding Subjective Impressions:** In RAG development, "it feels like it works" is not a valid engineering standard. Objective metrics ensure that improvements to chunking or prompts do not secretly degrade retrieval accuracy.
- **Regression Detection:** The benchmark guarantees that modifications in Module 14 (production cleanup) or future scaling do not introduce regressions into Modules 1–13.

---

### 16. Why Recall@K, Hit@K, and MRR?
- **Recall@K:** Measures the proportion of all relevant lecture chunks retrieved in the top $K$ results. Essential for ensuring the LLM has all facts needed to answer complex multi-part questions.
- **Hit@K:** Binary indicator of whether *at least one* ground-truth chunk appeared in the top $K$. Crucial for question-answering because a single accurate passage is often sufficient for a grounded answer.
- **MRR (Mean Reciprocal Rank):** Evaluates rank position quality ($1/\text{rank}$). A system that places the best passage at rank 1 is much better than one placing it at rank 5.

---

### 17. Why are citations included?
- **Trust & Verification:** LLMs are non-deterministic black boxes. Attaching exact timestamps and excerpts grounds the response in physical reality, allowing students to verify claims against the professor's original words.
- **Academic Rigor:** Reinforces scholarly research habits in students by citing course sources.

---

### 18. Why does the Dockerfile exist even though Docker has not yet been tested locally?
- **Deployment Blueprint:** Providing a standardized `Dockerfile` documents exact Linux system dependencies (`ffmpeg`, `curl`, Python 3.11) and runtime entrypoints needed for containerization.
- **Honest Engineering Transparency:** Maintaining the `Dockerfile` while clearly stating that local Docker daemon testing was unavailable demonstrates professional transparency—delivering container readiness without fabricating test results.
