# Demonstration Guide: RAG-Based AI Teaching Assistant

This guide provides practical, step-by-step instructions for demonstrating the **RAG-Based AI Teaching Assistant** during live reviews, portfolio presentations, or technical interviews.

---

## Pre-Demo Checklist

1. **Verify Environment:** Ensure your virtual environment is active:
   ```bash
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   # macOS / Linux
   source .venv/bin/activate
   ```
2. **Launch Application:**
   ```bash
   python scripts/run_app.py
   ```
3. **Open Browser:** Navigate to `http://localhost:8501`.
4. **API Key Setup:**
   - If using **Generative AI Mode**, enter your OpenAI API key in the sidebar.
   - If demonstrating **100% Offline / Local Mode**, leave the key field blank. The system will demonstrate deterministic grounded synthesis with direct quotes.

---

## 3-Minute Lightning Pitch Demo

*Designed for fast showcase or initial interview screen.*

1. **Introduction (20s):**
   - *"This is an AI Teaching Assistant that converts lecture videos into an interactive, timestamped course knowledge base using Whisper, Sentence Transformers, and a calibrated relevance gate."*
2. **In-Scope Question & Citations (40s):**
   - In Tab 1 (**Student Chat Assistant**), click a quick-start chip or type:
     `"What is an algorithm according to the lecture?"`
   - Point out the grounded response.
   - Expand **📚 Sources & Citations** to show the exact lecture title and timecode interval (`MM:SS – MM:SS`).
3. **Conversational Follow-Up (30s):**
   - Type a contextual follow-up:
     `"Can you give a practical example of that?"`
   - Explain that dialogue context is retained without requiring the user to re-state the subject.
4. **Relevance Gate Abstention (40s):**
   - Type an out-of-scope query:
     `"How do quantum computers calculate prime factors?"`
   - Show the system's safe abstention:
     > *"I couldn't find enough information about this topic in the provided course material."*
   - Mention the cosine similarity cutoff (`0.3200`) preventing hallucinations.
5. **Learning Center & Evaluation (50s):**
   - Click Tab 2 (**Learning Center**), select **Quiz Practice**, and show an interactive MCQ generated directly from the lecture.
   - Click Tab 3 (**RAG Evaluation**) to show the live benchmark dashboard displaying 100% Hit@5, 0% False Acceptance Rate, and 100% Citation Validity across 20 test questions.

---

## 10-Minute Deep Dive Technical Demo

*Designed for hiring managers, senior engineers, or technical project reviews.*

### Step 1: Architecture & Knowledge Base Ingestion (2 Minutes)
- **Action:** Open the sidebar and scroll to **📊 Course Knowledge Base**. Expand **📚 Available Lectures** to show pre-indexed course lectures (`python_lecture`, `tmpfy6rr9mw`).
- **Explanation:**
  - *"Our ingestion pipeline accepts video or audio files. Videos are automatically extracted to 16kHz mono MP3 via FFmpeg, transcribed with OpenAI Whisper down to the second, chunked with a 700-word sliding window (100-word overlap), and encoded using `all-MiniLM-L6-v2`."*
  - Expand the **📤 Upload & Index New Lecture** section in the main view. Explain SHA-256 duplicate detection:
    *"If a student tries uploading the same lecture under a different filename, the content hash prevents redundant transcription and storage."*

### Step 2: Student Chat & Granular Attribution (2 Minutes)
- **Action:** In Tab 1, select `All Lectures` as the scope and ask:
  `"What is a program according to the lecture?"`
- **Explanation:**
  - Review the synthesized answer.
  - Expand the citation card. Highlight that each chunk includes the lecture title, source filename, similarity score, and precise `MM:SS` start/end timestamps.
  - Expand **🔍 Retrieval Diagnostics (Developer View)**. Explain the internal metrics displayed:
    - Original query vs. effective retrieval query.
    - Top similarity score (e.g., `0.5840`).
    - Gate decision (`RELEVANT` because score $\ge 0.3200$).
    - Vector chunk IDs retrieved.

### Step 3: Multi-Lecture Scoping & Contextual Follow-Up (2 Minutes)
- **Action 1:** Ask a follow-up query:
  `"Why is that important for developers?"`
  Explain how multi-turn conversation memory passes preceding turns to the prompt to maintain topic continuity.
- **Action 2:** Switch the **Lecture Scope** dropdown in the sidebar from `All Lectures` to a specific lecture (e.g., `Lecture 1 — Python Lecture`).
  Explain that metadata filtering narrows the candidate search space prior to scoring, ensuring questions intended for a specific lecture are not polluted by other modules.

### Step 4: Out-of-Scope Gating & Hallucination Prevention (1.5 Minutes)
- **Action:** Ask an off-topic question:
  `"What is Einstein's theory of general relativity?"`
- **Explanation:**
  - Show the abstention response: *"I couldn't find enough information about this topic in the provided course material."*
  - Expand the developer diagnostics to show that the top similarity score was ~`0.0800`, falling well below the `0.3200` threshold.
  - Discuss the design philosophy:
    *"In educational technology, admitting lack of context is far safer than generating plausible-sounding hallucinations."*

### Step 5: Learning Center & Pedagogical Tooling (1.5 Minutes)
- **Action:** Click Tab 2 (**🎓 Learning Center**).
  - Click **Lecture Summary** to show an instant high-level overview.
  - Click **Key Concepts** to show extracted technical terms and bulleted definitions.
  - Click **Explain This Lecture** to show a progressive conceptual walkthrough.
  - Click **Quiz & Practice Mode**. Answer an interactive multiple-choice question. Show the immediate scoring feedback and lecture citation.

### Step 6: Empirical RAG Benchmarking & Diagnostics (1 Minute)
- **Action:** Click Tab 3 (**📊 RAG Evaluation**).
- **Explanation:**
  - Show the benchmark dashboard displaying live metrics across the 20-question test set:
    - **Recall@5:** `89.29%`
    - **Hit@5:** `100.00%`
    - **MRR:** `0.7679`
    - **Decision Accuracy:** `95.00%`
    - **False Acceptance Rate:** `0.00%`
    - **Citation Validity:** `100.00%`
  - Address the known edge case transparently:
    *"We openly track failure cases. Question `eval_012` scored `0.2970`, causing a safe false rejection. We deliberately preserved the 0.3200 cutoff to maintain a 0.00% False Acceptance Rate rather than overfitting to the benchmark."*
