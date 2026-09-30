# RAG-Based AI Teaching Assistant

An intelligent, multi-lecture Retrieval-Augmented Generation (RAG) educational system designed to transform raw lecture recordings and tutorial videos into an interactive, grounded AI Teaching Assistant.

The system extracts speech-optimized audio, transcribes spoken content with second-level timestamps using OpenAI Whisper, generates dense vector embeddings via Sentence Transformers, applies a relevance abstention gate to prevent hallucinations, and provides grounded pedagogical explanations, interactive quizzes, lecture summaries, key concept extracts, and comprehensive RAG benchmarking.

---

## 📌 Overview

Educational videos and recorded lectures are rich in knowledge but difficult for students to search, review, and query interactively. The **RAG AI Teaching Assistant** bridges this gap:

1. **Course Ingestion:** Ingests educational videos (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`) and audio files (`.mp3`, `.wav`, `.m4a`).
2. **Audio & Speech Transcription:** Converts media to speech-optimized 16kHz mono audio and produces timecoded transcripts via Whisper.
3. **Chunking & Indexing:** Chunks spoken segments with sliding window overlap while preserving lecture titles, IDs, and second-level start/end timestamps.
4. **Vector Retrieval:** Encodes passages into 384-dimensional dense vectors using `all-MiniLM-L6-v2` for fast cosine similarity search.
5. **Relevance Gate & Abstention:** Rejects out-of-scope queries falling below cosine similarity cutoff (`0.3200`) with graceful educational abstention.
6. **Student Chat:** Retains conversational context for follow-up questions, supports customizable teaching styles, and provides clickable timestamp citations (`MM:SS – MM:SS`).
7. **Learning Center:** Generates instant lecture summaries, key conceptual terms with definitions, deep-dive explanations, and interactive multiple-choice practice quizzes.
8. **Quality Benchmarking:** Includes a 20-question course benchmark evaluating Recall@K, Hit@K, MRR, False Acceptance Rate (FAR), False Rejection Rate (FRR), and citation validity.

---

## ❗ Problem Statement

- **Opaque Media:** Video lectures cannot be skimmed or grepped like physical textbooks. Scrubbing through hours of video to locate a single explanation is tedious.
- **Commercial LLM Hallucinations:** Generic models (e.g., ChatGPT) invent answers or use terminology conflicting with the course syllabus.
- **No Verifiable Provenance:** Generic chatbots provide no timestamp citations linking back to the professor's spoken words.
- **Out-of-Scope Confabulation:** Unconstrained chatbots attempt to answer off-topic questions rather than admitting lack of knowledge.

---

## 🚀 Key Features

* **Multi-Format Media Ingestion:** Ingests video and audio recordings with SHA-256 duplicate detection to prevent redundant indexing.
* **Whisper Transcription with Timestamps:** Transcribes spoken lectures with second-accurate start and end intervals.
* **Timestamped Chunking & Provenance:** Chunks transcripts into cohesive passages tagged with lecture ID, title, source filename, and timecodes.
* **Dense Vector Embeddings:** Uses local Sentence Transformers (`all-MiniLM-L6-v2`) for fast, local semantic vector representation.
* **Relevance & Abstention Gate:** Employs a strict cosine similarity gate (`0.3200`) to abstain when course material lacks sufficient information:
  > *"I couldn't find enough information about this topic in the provided course material."*
* **Multi-Lecture Course Knowledge Base:** Search across the entire course ("All Lectures") or isolate questions to a single lecture.
* **Student Chat Assistant:** Context-aware dialogue maintaining recent conversation turns for follow-ups (e.g., *"Can you give an example of that?"*).
* **Pedagogical Response Styles:** Choose between **Explain Clearly**, **Short Answer**, and **Detailed Explanation**.
* **Source Attribution & Citations:** Every answer links directly to the source lecture, timestamp interval (`MM:SS`), similarity score, and excerpt.
* **Learning Center (Study Hub):**
  - **Lecture Summary:** High-level overview and key takeaways.
  - **Key Concepts:** Extracted technical terms, formulas, and definitions.
  - **Explain This Lecture:** Progressive, intuitive walkthrough of the lecture.
  - **Quiz Practice:** Interactive multiple-choice questions with instant scoring, feedback, and lecture source references.
* **RAG Evaluation Suite:** Comprehensive metrics including Recall@1/3/5, Hit@5, MRR, Lecture Retrieval Accuracy, Relevance Decision Accuracy, FAR, FRR, and Citation Validity.
* **100% Offline / Deterministic Fallback:** Operates without external API keys in local synthesis mode (direct quotes and timestamped citations).
* **Production Deployment Ready:** Centralized configuration, portable path handling, Docker containerization, health check validation, and logging.

---

## 🏗️ Architecture

```text
Lecture Video / Audio File (.mp4, .mov, .mp3, etc.)
                      │
                      ▼
            Audio Extraction (FFmpeg)
                      │
                      ▼
         Whisper Speech Transcription
                      │
                      ▼
         Timestamped Chunking & Catalog
                      │
                      ▼
      Dense Vector Embeddings (MiniLM-L6)
                      │
                      ▼
            Vector Knowledge Base
                      │
══════════════════════╪══════════════════════
                      │
              Student Question
                      │
                      ▼
            Dense Query Encoding
                      │
                      ▼
           Vector Retrieval (Top-K)
                      │
                      ▼
       Relevance Gate (Cosine Threshold 0.32)
        ├── Below Cutoff ──► Graceful Educational Abstention
        └── Above Cutoff ──► Grounded Context Assembly
                                    │
                                    ▼
                      LLM Synthesis (gpt-4o-mini / Local Mode)
                                    │
                                    ▼
                       Student Educational Answer
                   + Timestamp Citations (MM:SS – MM:SS)
```

For full architecture details and the interactive Mermaid diagram, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 💻 Tech Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11 / 3.9+ | Standard environment for data science and GenAI |
| **User Interface** | Streamlit | Rapid, responsive web dashboard with native session state |
| **Speech-to-Text** | OpenAI Whisper (`base`) | High noise robustness and native timecoded segment generation |
| **Media Processing** | FFmpeg | 16kHz mono audio extraction optimizing Whisper acoustic inputs |
| **Dense Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) | Lightweight 384-dim local embeddings running on CPU |
| **Vector Search** | NumPy | High-performance dot-product cosine similarity over normalized vectors |
| **Dataset Storage** | Joblib & JSON | Compressed binary caching for instant $O(1)$ startup |
| **LLM Synthesis** | OpenAI (`gpt-4o-mini` / `gpt-4o`) | Cost-effective grounded synthesis with local fallback |
| **Testing** | Pytest | Comprehensive test harness (**179 passed, 0 failed**) |
| **Deployment** | Docker | Standardized multi-platform container blueprint |

---

## 📂 Project Structure

```text
rag-ai-teacher/
│
├── app.py                         # Unified Streamlit Web Application
├── README.md                      # Comprehensive project documentation
├── requirements.txt               # Development & test dependencies (with pytest)
├── requirements-prod.txt          # Lean production runtime dependencies
├── .env.example                   # Environment variable template with placeholders
├── .gitignore                     # Git ignore rules (secrets, venvs, caches ignored)
├── Dockerfile                     # Production multi-platform Docker container
├── .dockerignore                  # Build exclusions for Docker context
├── pytest.ini                     # Pytest configuration
│
├── .streamlit/
│   └── config.toml                # Headless server, security, and UI theme config
│
├── docs/                          # In-depth architectural & technical documentation
│   ├── PROJECT_OVERVIEW.md        # Comprehensive project background & motivation
│   ├── ARCHITECTURE.md            # Detailed pipeline flows & Mermaid diagram
│   ├── TECHNICAL_DECISIONS.md     # 18 architectural decisions explained
│   ├── EVALUATION.md              # Empirical benchmark results & eval_012 analysis
│   ├── MODULES.md                 # Modules 1–14 chronological development history
│   └── DEMO_GUIDE.md              # 3-minute & 10-minute presentation guide
│
├── src/                           # Core modular application package
│   ├── __init__.py
│   ├── config/                    # Centralized configuration system
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── health.py                  # Startup validation and system diagnostics
│   ├── utils.py                   # Path helpers, logging setup, filename sanitization
│   ├── audio/                     # Video-to-audio extraction (FFmpeg)
│   │   └── extractor.py
│   ├── transcription/             # Whisper transcription & text normalization
│   │   ├── transcriber.py
│   │   └── processor.py
│   ├── chunking/                  # Semantic text chunking with timecodes
│   │   └── chunker.py
│   ├── embeddings/                # Sentence-Transformers dense embedding generation
│   │   └── embedder.py
│   ├── retrieval/                 # Semantic retrieval & multi-lecture filtering
│   │   └── retriever.py
│   ├── data/                      # Course catalog, lecture registry, dataset builder
│   │   ├── course_catalog.py
│   │   └── dataset_builder.py
│   ├── rag/                       # Grounded prompt engineering & RAG pipeline
│   │   ├── pipeline.py
│   │   └── prompt.py
│   ├── llm/                       # LLM client & error handling (OpenAI / Local)
│   │   └── client.py
│   ├── learning/                  # Learning Center features
│   │   ├── summary.py
│   │   ├── concepts.py
│   │   ├── explain.py
│   │   └── quiz.py
│   └── evaluation/                # RAG evaluation engine and metrics
│       ├── evaluator.py
│       ├── metrics.py
│       └── runner.py
│
├── scripts/                       # Reproducible CLI utilities & launchers
│   ├── run_app.py                 # Cross-platform application launcher with health check
│   ├── run_evaluation.py          # CLI RAG benchmark evaluation runner
│   ├── verify_functionality.py    # Functional smoke verification script
│   └── ...
│
├── tests/                         # Automated test suite (179 tests, 100% pass rate)
│   ├── test_production.py         # Module 14 deployment and configuration tests
│   ├── test_evaluation.py         # Module 13 evaluation framework tests
│   ├── test_learning_features.py  # Module 12 Learning Center tests
│   ├── test_student_chat.py       # Module 11 conversational chat tests
│   ├── test_multi_lecture.py      # Module 10 multi-lecture knowledge base tests
│   └── ...
│
└── data/                          # Persistent data directories (structure tracked via .gitkeep)
    ├── audio/
    ├── chunks/
    ├── course_catalog.json
    ├── embeddings/
    ├── evaluation/
    │   ├── rag_benchmark.json     # Ground-truth 20-question evaluation benchmark
    │   └── results/
    ├── processed/
    ├── transcripts/
    └── videos/
```

---

## 🛠️ Installation & Setup

### Prerequisites

* **Python 3.9+** (Python 3.10 or 3.11 recommended)
* **FFmpeg** (optional for video extraction; audio files can be ingested without FFmpeg)
  * Windows: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
  * macOS: `brew install ffmpeg`
  * Linux: `sudo apt-get install ffmpeg`

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/rag-ai-teacher.git
cd rag-ai-teacher
```

### Step 2: Create a Virtual Environment

#### Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ⚙️ Environment Configuration

Copy the example environment configuration template:

#### Windows PowerShell:
```powershell
Copy-Item .env.example .env
```

#### macOS / Linux:
```bash
cp .env.example .env
```

Edit `.env` to customize settings:

```dotenv
APP_ENV=development
LOG_LEVEL=INFO

# Relevance Abstention Threshold (Module 9)
RELEVANCE_THRESHOLD=0.32

# Vector Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Whisper Speech Transcription
WHISPER_MODEL=base

# Optional OpenAI API Key (Generative AI Mode)
# Leave blank for 100% offline local synthesis mode
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

> **Security Note:** `.env` and `.streamlit/secrets.toml` are explicitly ignored by `.gitignore`. Real credentials are never committed.

---

## 🚀 Running the Application

### Option 1: Using the Cross-Platform Launcher (Recommended)

```bash
python scripts/run_app.py
```
This launcher automatically validates system health, checks required directories, and launches Streamlit.

### Option 2: Direct Streamlit Execution

```bash
streamlit run app.py
```

Once running, navigate to `http://localhost:8501` in your browser.

The interface contains three core tabs:
1. **💬 Student Chat Assistant:** Conversational course Q&A with timestamped source citations.
2. **🎓 Learning Center:** Summaries, key concepts, detailed walkthroughs, and practice quizzes.
3. **📊 RAG Evaluation:** Live benchmark runner and evaluation metrics dashboard.

---

## 🔄 Example Workflow

1. **Ingest a Lecture:** Expand **📤 Upload & Index New Lecture** in the app and upload a course recording. The system transcribes, chunks, and indexes it with SHA-256 duplicate protection.
2. **Ask a Conceptual Question:** In Tab 1, ask *"What is an algorithm according to the lecture?"* View the synthesized answer and expand the citation card to inspect the exact `MM:SS` timecode.
3. **Follow-Up:** Type *"Can you give an example of that?"* The assistant uses conversation history to resolve context.
4. **Test Abstention:** Ask an off-topic query like *"How do black holes form?"* The relevance gate triggers safe educational abstention.
5. **Study Mode:** Switch to Tab 2 to generate instant summaries, glossaries, or take an interactive practice quiz.
6. **Evaluate:** Switch to Tab 3 to execute the live benchmark against `data/evaluation/rag_benchmark.json`.

---

## 🧪 Testing

Execute the automated test suite with `pytest`:

```bash
pytest -q
```
**Current Verified Baseline:** `179 passed, 0 failed in ~50s` across 13 test suites.

To run only the production deployment tests:

```bash
pytest tests/test_production.py -v
```

---

## 📊 RAG Evaluation & Benchmarking

Execute the evaluation benchmark from the command line:

```bash
python scripts/run_evaluation.py
```

Results are saved to `data/evaluation/results/latest_evaluation.json` and `latest_evaluation_report.md`.

### Benchmark Metrics Summary (20 Questions on Course Material)

| Metric | Score | Target | Description |
| :--- | :---: | :---: | :--- |
| **Recall@1** | 53.57% | > 50% | Relevant chunk ranked at position 1 |
| **Recall@3** | 82.14% | > 75% | Relevant chunk found within top 3 passages |
| **Recall@5** | 89.29% | > 80% | Relevant chunk found within top 5 passages |
| **Hit@5** | 100.00% | 100% | At least one relevant chunk found in top 5 |
| **MRR** | 0.7679 | > 0.70 | Mean Reciprocal Rank across relevant queries |
| **Lecture Accuracy** | 100.00% | 100% | Correct lecture identified in top rank |
| **Relevance Decision** | 95.00% | > 90% | Binary relevance decision accuracy |
| **FAR (False Accept)**| 0.00% | < 5% | Out-of-scope questions mistakenly accepted |
| **FRR (False Reject)**| 7.14% | < 10% | Valid in-scope questions rejected by gate |
| **Citation Validity** | 100.00% | 100% | Cited passages correctly map to lecture files |
| **Deterministic Answer**| 95.00% | > 90% | Grounded verification against expected answers |

For full evaluation analysis and the `eval_012` case study, see [docs/EVALUATION.md](docs/EVALUATION.md).

---

## 🔒 Security & Production Considerations

- **Secret Masking:** Keys are never printed in plain text in logs or UI summaries (`mask_secret()`).
- **Path Traversal Defense:** Uploaded filenames are sanitized (`sanitize_filename()`), stripping `../`, null bytes, and control characters.
- **Upload Guards:** Enforces 0-byte detection and a 200MB maximum upload limit.
- **Container Security:** Docker container runs under a non-root `appuser` (UID 1000).

---

## 🐳 Docker Deployment

The application includes a production-ready, CPU-compatible Dockerfile with FFmpeg pre-installed:

> **Important Note:** *The `Dockerfile` is included and follows production container standards, but Docker build/container execution has not been verified in the current local development environment due to Docker daemon unavailability on this host.*

### Build Container Image

```bash
docker build -t rag-ai-teacher .
```

### Run Container

```bash
docker run --rm -p 8501:8501 -e OPENAI_API_KEY="your_key_here" rag-ai-teacher
```

Navigate to `http://localhost:8501`.

---

## ⚠️ Current Limitations

* **Benchmark Size:** The current ground-truth benchmark consists of 20 representative course queries. Larger curricula will benefit from expanding to 100+ questions.
* **CPU Inference:** Whisper transcription and Sentence-Transformers inference run on CPU by default. Processing long video lectures (1+ hours) takes several minutes on standard hardware without a CUDA GPU.
* **Generative LLM Requirement:** Offline mode produces grounded direct excerpts with citations. High-level pedagogical explanations require an OpenAI API key.
* **Retrieval Threshold Edge Cases:** Query `eval_012` (*"What department has she been lecturing in?"*) achieves cosine similarity of `0.2970`, which sits just below the `0.3200` relevance cutoff, resulting in a false rejection. The threshold is deliberately held constant to prevent overfitting.

---

## 🔮 Future Improvements

* **Reranking:** Incorporating a cross-encoder reranker (e.g., `bge-reranker-base`) after initial retrieval.
* **Hybrid Search:** Combining dense embeddings with sparse BM25 keyword search.
* **Query Expansion & Rewriting:** Using HyDE (Hypothetical Document Embeddings) or multi-query expansion.
* **Distributed Vector Indexing:** Integrating dedicated vector databases (e.g., Qdrant, Chroma, Milvus) for institutional-scale multi-course deployments.
* **Agentic Workflows:** Autonomous student assignment grading and interactive code execution sandbox.

---

## 📚 Complete Documentation Index

For in-depth explanations, review the dedicated guides in [docs/](docs/):
- **[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md):** Executive summary, problem statement, and workflow.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md):** Complete pipeline breakdown and Mermaid architecture diagram.
- **[docs/TECHNICAL_DECISIONS.md](docs/TECHNICAL_DECISIONS.md):** Detailed rationale for 18 core technical decisions.
- **[docs/EVALUATION.md](docs/EVALUATION.md):** Empirical benchmark results, metrics formulas, and `eval_012` failure study.
- **[docs/MODULES.md](docs/MODULES.md):** Chronological implementation history across Modules 1–14.
- **[docs/DEMO_GUIDE.md](docs/DEMO_GUIDE.md):** 3-minute lightning pitch and 10-minute technical deep dive scripts.

---

## 📄 License

This project is developed for educational, portfolio, and demonstration purposes.
