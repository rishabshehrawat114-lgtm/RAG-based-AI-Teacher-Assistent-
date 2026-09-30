"""
app.py
======
Unified Streamlit Web Application for RAG AI Teacher Assistant (Module 10).

Module 10: Multi-Lecture Course Knowledge Base
- Ingest and index multiple course lectures incrementally.
- Content-hash based duplicate detection prevents redundant indexing.
- Search the complete course ("All Lectures") or filter to a specific lecture.
- Exact MM:SS lecture timestamps and clear lecture attribution in answers.
- Course Knowledge Base catalog and developer diagnostics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="RAG AI Teacher Assistant — Course Knowledge Base",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Core imports
from src.audio import SUPPORTED_VIDEO_EXTENSIONS, is_ffmpeg_available
from src.config import (
    ALL_DATA_DIRS,
    ALL_SUPPORTED_MEDIA_EXTENSIONS,
    AUDIO_DIR,
    BASE_DIR,
    CHUNK_DIR,
    CONSOLIDATED_DATASET_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_RELEVANCE_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    DEFAULT_WHISPER_MODEL,
    EMBEDDING_DIR,
    EVALUATION_DATASET_PATH,
    MAX_UPLOAD_SIZE_MB,
    OPENAI_API_KEY,
    RESULTS_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    TRANSCRIPT_DIR,
    VIDEO_DIR,
    get_config_summary,
)
from src.data import CourseCatalog, compute_file_hash
from src.evaluation import (
    DatasetError,
    EvaluationSummary,
    RAGEvaluator,
    load_benchmark_dataset,
    save_evaluation_results,
)
from src.health import check_system_health
from src.learning import (
    LearningResult,
    QuizQuestion,
    QuizResult,
    explain_lecture,
    generate_key_concepts,
    generate_quiz,
    generate_summary,
)
from src.llm import LLMClient
from src.rag import (
    DEFAULT_RESPONSE_STYLE,
    FALLBACK_NOT_FOUND_MESSAGE,
    RAGAnswer,
    RAGTeacherPipeline,
    RESPONSE_STYLES,
)
from src.retrieval import LectureRetriever, format_timestamp
from src.utils import dir_exists, ensure_dir, get_logger, mask_secret, sanitize_filename

logger = get_logger("app")


# Initialize cached RAG Pipeline
@st.cache_resource(show_spinner="Initializing Course Knowledge Engine & Vector Store...")
def get_rag_pipeline(api_key: str | None = None, model_name: str | None = None) -> RAGTeacherPipeline:
    """Creates and caches the persistent RAG Teacher Assistant pipeline."""
    retriever = LectureRetriever(
        embeddings_dir=EMBEDDING_DIR,
        dataset_path=CONSOLIDATED_DATASET_PATH,
        auto_load=True,
    )
    retriever.warmup()
    llm_client = LLMClient(api_key=api_key, model_name=model_name)
    catalog = CourseCatalog()
    return RAGTeacherPipeline(retriever=retriever, llm_client=llm_client, catalog=catalog)


def initialize_session_state() -> None:
    """Initializes Streamlit session state variables for chat history and settings."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "api_key" not in st.session_state:
        # Check st.secrets first if running in Streamlit cloud or with secrets.toml
        secret_key = ""
        try:
            if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
                secret_key = str(st.secrets["OPENAI_API_KEY"]).strip()
        except Exception:
            pass
        env_key = secret_key or (OPENAI_API_KEY.strip() if OPENAI_API_KEY else "")
        if env_key and not env_key.startswith("your_"):
            st.session_state.api_key = env_key
        else:
            st.session_state.api_key = ""

    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_LLM_MODEL

    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K

    if "threshold" not in st.session_state:
        st.session_state.threshold = None

    if "selected_source" not in st.session_state:
        st.session_state.selected_source = "All Lectures"

    if "lecture_scope" not in st.session_state:
        st.session_state.lecture_scope = "All Lectures"

    if "response_style" not in st.session_state:
        st.session_state.response_style = DEFAULT_RESPONSE_STYLE

    # Module 12: Learning Center session state
    if "learning_mode" not in st.session_state:
        st.session_state.learning_mode = None

    if "learning_scope" not in st.session_state:
        st.session_state.learning_scope = "All Lectures"

    if "learning_result" not in st.session_state:
        st.session_state.learning_result = None

    if "quiz_result" not in st.session_state:
        st.session_state.quiz_result = None

    if "quiz_current_index" not in st.session_state:
        st.session_state.quiz_current_index = 0

    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0

    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}

    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = {}

    if "quiz_complete" not in st.session_state:
        st.session_state.quiz_complete = False

    if "quiz_num_questions" not in st.session_state:
        st.session_state.quiz_num_questions = 5

    # Module 13: Evaluation session state
    if "evaluation_summary" not in st.session_state:
        st.session_state.evaluation_summary = None


def render_header() -> None:
    """Renders the top branding header and navigation bar."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
            <div>
                <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700;">
                    🎓 AI Teacher Assistant
                </h1>
                <p style="margin: 0.2rem 0 0 0; color: #6b7280; font-size: 1rem;">
                    Multi-Lecture Course Knowledge Base • Timestamped Grounded Retrieval • Educational Q&A
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


def get_lecture_scope_options(pipeline: RAGTeacherPipeline) -> list[str]:
    """Builds a formatted list of lecture options for the lecture scope filter."""
    stats = pipeline.get_corpus_stats()
    lectures = stats.get("lectures", [])
    options = ["All Lectures"]
    if lectures:
        for idx, lec in enumerate(lectures, start=1):
            title = lec.get("lecture_title") or f"Lecture {idx}"
            options.append(f"Lecture {idx} — {title}")
    elif stats.get("unique_sources"):
        for src in stats["unique_sources"]:
            stem = Path(src).stem.replace("_", " ").title()
            options.append(stem)
    return options


def render_sidebar(pipeline: RAGTeacherPipeline) -> None:
    """Renders the sidebar with configuration controls, content upload, and dataset stats."""
    with st.sidebar:
        st.header("⚙️ Configuration & Settings")

        # API Key Configuration
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="sk-proj-...",
            help="Your API key is never hardcoded. Enter your key to enable generative GPT explanations.",
        )
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            pipeline.llm_client = LLMClient(
                api_key=api_key_input,
                model_name=st.session_state.model_name,
            )

        if pipeline.llm_client.is_configured():
            st.caption("🟢 OpenAI API Key configured & active")
        else:
            st.caption("🟡 Operating in Local Synthesis Mode (direct lecture quotes). Enter API key above for generative AI answers.")

        # LLM Model Selector
        st.session_state.model_name = st.selectbox(
            "LLM Model",
            options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Model used to synthesize answers based on retrieved context.",
        )
        pipeline.llm_client.model_name = st.session_state.model_name

        # Response Style Selector (Section 8)
        current_style_idx = 0
        if st.session_state.get("response_style") in RESPONSE_STYLES:
            current_style_idx = list(RESPONSE_STYLES).index(st.session_state.response_style)
        st.session_state.response_style = st.selectbox(
            "Teaching / Response Style",
            options=list(RESPONSE_STYLES),
            index=current_style_idx,
            help=(
                "• Explain Clearly: Encouraging, student-friendly, moderate detail.\n"
                "• Short Answer: Concise direct answer in 2-4 sentences.\n"
                "• Detailed Explanation: In-depth coverage with structured sections."
            ),
        )

        st.markdown("---")
        st.subheader("🔍 Retrieval Parameters")

        # Top-K Slider
        st.session_state.top_k = st.slider(
            "Top-K Retrieved Chunks",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            help="Number of most semantically relevant lecture passages to retrieve.",
        )

        # Similarity Cutoff Threshold (Adaptive by default)
        use_threshold = st.checkbox("Manual Similarity Threshold (Override)", value=False)
        if use_threshold:
            st.session_state.threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=0.8,
                value=0.20,
                step=0.05,
                help="Manually override the adaptive filter with a strict cutoff.",
            )
        else:
            st.session_state.threshold = None

        # Lecture Scope filter
        scope_options = get_lecture_scope_options(pipeline)
        current_idx = 0
        if st.session_state.lecture_scope in scope_options:
            current_idx = scope_options.index(st.session_state.lecture_scope)

        selected_scope = st.selectbox(
            "Lecture Scope",
            options=scope_options,
            index=current_idx,
            help="Search all course lectures or scope question answering to a specific lecture.",
        )
        st.session_state.lecture_scope = selected_scope
        st.session_state.selected_source = selected_scope

        st.markdown("---")
        st.subheader("📊 Course Knowledge Base")
        stats = pipeline.get_corpus_stats()
        st.metric("Total Indexed Chunks", stats["total_chunks"])
        st.metric("Total Course Lectures", stats.get("lecture_count", stats["source_count"]))

        if stats.get("lectures"):
            with st.expander("📚 Available Lectures", expanded=False):
                for idx, lec in enumerate(stats["lectures"], start=1):
                    ltitle = lec.get("lecture_title", f"Lecture {idx}")
                    lchunks = lec.get("chunk_count", 0)
                    st.text(f"• Lecture {idx}: {ltitle} ({lchunks} chunks)")

        st.markdown("---")
        if st.button("🔄 Refresh Knowledge Base", use_container_width=True):
            pipeline.reindex_all()
            pipeline.catalog.sync_with_existing_data()
            st.success("Knowledge base refreshed from disk.")
            st.rerun()

        with st.expander("🩺 System Diagnostics & Health", expanded=False):
            health = check_system_health()
            badge = "🟢 Healthy" if health.status == "OK" else ("🟡 Operational (Warnings)" if health.status == "WARNING" else "🔴 Errors Detected")
            st.markdown(f"**System Status:** {badge}")
            st.caption(health.summary)
            for chk in health.checks:
                icon = "✅" if chk.status == "OK" else ("⚠️" if chk.status == "WARNING" else "❌")
                st.text(f"{icon} {chk.name}: {chk.status}")


def render_course_knowledge_base_section(pipeline: RAGTeacherPipeline) -> None:
    """Renders the Course Knowledge Base overview table and lecture upload interface."""
    st.markdown("## Course Knowledge Base")

    stats = pipeline.get_corpus_stats()
    lectures = stats.get("lectures", [])

    # Display Course Knowledge Base table (Section 6)
    if lectures:
        table_rows = []
        for idx, lec in enumerate(lectures, start=1):
            ltitle = lec.get("lecture_title", f"Lecture {idx}")
            lstatus = lec.get("status", "Indexed")
            lchunks = lec.get("chunk_count", 0)
            lsource = lec.get("source_filename", "N/A")
            table_rows.append({
                "Lecture": f"Lecture {idx}",
                "Title": ltitle,
                "Chunks": lchunks,
                "Status": lstatus,
                "Source File": lsource,
            })
        df_table = pd.DataFrame(table_rows)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
    else:
        st.info("No course lectures currently indexed. Upload your first course lecture below!")

    # Ingest / Upload lecture form
    with st.expander("📤 Upload & Index New Lecture", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Course Media File",
                type=["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a"],
                help="Upload an audio or video lecture to process and index into the course knowledge base.",
                key="lecture_file_uploader",
            )
        with col2:
            next_idx = len(lectures) + 1
            default_title = f"Lecture {next_idx} — Topics"
            lecture_title_input = st.text_input(
                "Lecture Title (Optional)",
                placeholder=default_title,
                help="Custom lecture title displayed in citations and course catalog.",
            )

        if uploaded_file is not None:
            if st.button("🚀 Process & Index Lecture", use_container_width=True):
                # 1. Empty file validation
                if uploaded_file.size == 0:
                    st.error("Uploaded file is empty (0 bytes). Please upload a valid course media file.")
                    st.stop()

                # 2. Maximum file size check
                max_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024
                if uploaded_file.size > max_bytes:
                    st.error(f"File exceeds maximum allowed upload size of {MAX_UPLOAD_SIZE_MB}MB.")
                    st.stop()

                # 3. Path traversal safety and extension check
                safe_name = sanitize_filename(uploaded_file.name)
                suffix = Path(safe_name).suffix.lower()
                is_video = suffix in SUPPORTED_VIDEO_EXTENSIONS
                is_audio = suffix in SUPPORTED_AUDIO_EXTENSIONS

                if not (is_video or is_audio):
                    st.error(
                        f"Unsupported file format '{suffix}'. "
                        f"Supported formats: {', '.join(sorted(ALL_SUPPORTED_MEDIA_EXTENSIONS))}"
                    )
                    st.stop()

                if is_video and not is_ffmpeg_available():
                    st.error("FFmpeg is not installed on this system. Audio cannot be extracted from video.")
                else:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def update_progress(msg: str, pct: float) -> None:
                        status_text.info(f"⏳ {msg}")
                        progress_bar.progress(min(max(pct, 0.0), 1.0))

                    try:
                        # Write buffer to disk in appropriate directory using safe_name
                        target_dir = VIDEO_DIR if is_video else AUDIO_DIR
                        target_dir.mkdir(parents=True, exist_ok=True)
                        dest_file = target_dir / safe_name
                        with open(dest_file, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Determine title and ID
                        resolved_title = lecture_title_input.strip() if lecture_title_input else safe_name
                        resolved_id = f"lecture_{next_idx:02d}"

                        if is_video:
                            update_progress("Starting Video Ingestion Pipeline...", 0.05)
                            res = pipeline.index_video(
                                video_path=dest_file,
                                progress_callback=update_progress,
                                lecture_id=resolved_id,
                                lecture_title=resolved_title,
                            )
                        else:
                            update_progress("Starting Audio Ingestion Pipeline...", 0.05)
                            res = pipeline.index_audio(
                                audio_path=dest_file,
                                progress_callback=update_progress,
                                lecture_id=resolved_id,
                                lecture_title=resolved_title,
                            )

                        if res.get("status") == "skipped_duplicate":
                            status_text.warning(f"⚠️ {res.get('message', 'Lecture is already indexed!')}")
                        else:
                            status_text.success(
                                f"✅ Successfully indexed `{resolved_title}`! ({res.get('chunk_count', 0)} chunks added)"
                            )
                            st.rerun()

                    except Exception as exc:
                        logger.exception("Failed to process and index uploaded lecture '%s': %s", safe_name, exc)
                        status_text.error(
                            "❌ We couldn't process this lecture. Please verify the media file is valid and try again."
                        )
                        with st.expander("Technical Error Details", expanded=False):
                            st.code(str(exc))

    st.divider()


def render_chat_interface(pipeline: RAGTeacherPipeline) -> None:
    """Renders the conversational chat interface, messages, source citations, and debug info."""
    st.subheader("💬 Ask Your AI Teacher Assistant")

    # Scope, Style, and Action toolbar
    col_scope, col_style, col_clear = st.columns([3, 2, 2])
    with col_scope:
        scope_options = get_lecture_scope_options(pipeline)
        current_idx = 0
        if st.session_state.lecture_scope in scope_options:
            current_idx = scope_options.index(st.session_state.lecture_scope)

        selected = st.selectbox(
            "Lecture Scope for Question",
            options=scope_options,
            index=current_idx,
            key="chat_lecture_scope_select",
            help="Select 'All Lectures' to search the entire course, or filter to a specific lecture.",
        )
        if selected != st.session_state.lecture_scope:
            st.session_state.lecture_scope = selected
            st.session_state.selected_source = selected

    with col_style:
        style_idx = 0
        if st.session_state.get("response_style") in RESPONSE_STYLES:
            style_idx = list(RESPONSE_STYLES).index(st.session_state.response_style)
        chosen_style = st.selectbox(
            "Response Style",
            options=list(RESPONSE_STYLES),
            index=style_idx,
            key="chat_response_style_select",
            help="Select the teaching style for answers (Explain Clearly, Short Answer, Detailed Explanation).",
        )
        if chosen_style != st.session_state.response_style:
            st.session_state.response_style = chosen_style

    with col_clear:
        st.write("")
        st.write("")
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🗑️ Clear", help="Clear conversation history while preserving indexed lecture data", use_container_width=True):
                st.session_state.messages = []
                keys_to_delete = [k for k in st.session_state.keys() if k.startswith(("msg_", "ctx_", "diag_"))]
                for k in keys_to_delete:
                    del st.session_state[k]
                st.rerun()
        with c_btn2:
            if st.button("➕ New", help="Start a new chat session", use_container_width=True):
                st.session_state.messages = []
                keys_to_delete = [k for k in st.session_state.keys() if k.startswith(("msg_", "ctx_", "diag_"))]
                for k in keys_to_delete:
                    del st.session_state[k]
                st.rerun()

    # Empty State (Section 14)
    if not st.session_state.messages:
        curr_scope = st.session_state.get("lecture_scope", "All Lectures")
        st.markdown(
            """
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; margin: 1rem 0;">
                <h3 style="margin-top: 0; margin-bottom: 0.3rem;">🎓 AI Teaching Assistant</h3>
                <p style="color: #64748b; margin-bottom: 0.8rem; font-size: 0.95rem;">
                    Ask questions grounded in your course lectures, explore concepts, and ask follow-ups.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 💡 Suggested Questions to Try:")
        if curr_scope != "All Lectures":
            starters = [
                "What topics are covered in this lecture?",
                "Explain the main concept discussed here.",
                "What example does the instructor give?",
                "Summarize the key takeaways from this lecture.",
            ]
        else:
            starters = [
                "What did the instructor explain about variables?",
                "What does Python do when it evaluates an expression?",
                "What is the difference between supervised and unsupervised learning?",
                "What topics are introduced in the lectures?",
            ]

        s_cols = st.columns(2)
        for idx, starter in enumerate(starters):
            col = s_cols[idx % 2]
            with col:
                if st.button(f"👉 {starter}", key=f"starter_{idx}", use_container_width=True):
                    process_user_query(pipeline, starter)
                    st.rerun()

    # Render conversation history (Section 4)
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])

            # Structured source citations (Section 9 & 10)
            citations = msg.get("citations") or msg.get("sources", [])
            if citations and role == "assistant":
                with st.expander(
                    f"📚 Sources & Citations ({len(citations)} passage{'s' if len(citations) > 1 else ''})",
                    expanded=False,
                ):
                    for idx, src in enumerate(citations, start=1):
                        chunk_id = src.get("chunk_id", f"chunk_{idx}")
                        timerange = src.get("timestamp_range", "00:00 - 00:00")
                        score = float(src.get("similarity_score", 0.0))
                        vid = src.get("source_filename") or src.get("source_file", "Lecture")
                        lec_title = src.get("lecture_title") or Path(vid).stem.replace("_", " ").title()
                        text = (src.get("text") or "").strip()

                        st.markdown(f"**{idx}. {lec_title}**")
                        st.markdown(f"- ⏱️ `{timerange}`")
                        st.markdown(f"- **Similarity:** `{score:.4f}`")
                        st.markdown(f"- **Source:** `{chunk_id}` (`{vid}`)")

                        excerpt = text[:260].strip() + ("..." if len(text) > 260 else "")
                        st.info(f"💬 *\"{excerpt}\"*")
                        if idx < len(citations):
                            st.markdown("---")

            # Developer Diagnostics (Section 10 & 18)
            debug_info = msg.get("diagnostics") or msg.get("debug_info")
            if debug_info and role == "assistant":
                with st.expander("🔍 Retrieval Diagnostics (Developer View)", expanded=False):
                    decision = debug_info.get("relevance_decision", "N/A")
                    decision_badge = (
                        "🟢 **RELEVANT**"
                        if decision == "RELEVANT"
                        else ("🔴 **NOT RELEVANT**" if decision == "NOT RELEVANT" else f"ℹ️ **{decision}**")
                    )
                    query_val = debug_info.get("query") or "N/A"
                    retrieval_q = debug_info.get("retrieval_query") or query_val
                    is_fu = debug_info.get("is_follow_up", False)
                    retrieved_count = debug_info.get("retrieved_count", 0)
                    top_score_val = debug_info.get("top_score")
                    top_score_str = f"{top_score_val:.4f}" if isinstance(top_score_val, (int, float)) else str(top_score_val or "N/A")
                    thresh_val = debug_info.get("relevance_threshold")
                    thresh_str = f"{thresh_val:.4f}" if isinstance(thresh_val, (int, float)) else str(thresh_val or "N/A")
                    sim_scores = debug_info.get("similarity_scores", [])
                    selected_lec = debug_info.get("selected_lecture") or debug_info.get("lecture_scope") or "All Lectures"
                    style_used = debug_info.get("response_style") or msg.get("response_style", "Explain Clearly")

                    st.markdown("### Retrieval Diagnostics")
                    st.markdown(f"**Original Query:** {query_val}")
                    if is_fu:
                        st.markdown(f"**Effective Retrieval Query (Follow-up Context):** `{retrieval_q}`")
                    st.markdown(f"**Selected lecture scope:** {selected_lec}")
                    st.markdown(f"**Response style:** {style_used}")
                    st.markdown(f"**Retrieved chunks:** {retrieved_count}")
                    st.markdown(f"**Top similarity score:** {top_score_str}")
                    st.markdown(f"**Relevance decision:** {decision_badge}")
                    st.markdown(f"**Threshold:** {thresh_str}")
                    if debug_info.get("top_source_lecture"):
                        st.markdown(f"**Top Source:** {debug_info.get('top_source_lecture')} ({debug_info.get('top_source_timestamp', 'N/A')})")
                    st.markdown(f"**Similarity Scores:** `{sim_scores}`")

                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Chunk IDs:** `{debug_info.get('chunk_ids', [])}`")
                        st.markdown(f"**Scope Filter:** `{selected_lec}`")
                    with col2:
                        st.markdown(f"**Model Used:** `{debug_info.get('model', 'Local Synthesis')}`")
                        st.markdown(f"**LLM Invoked:** `{debug_info.get('llm_called', False)}`")

                    context_content = msg.get("context_text") or debug_info.get("context_text")
                    if context_content and decision == "RELEVANT":
                        st.markdown("**Prompt Context Sent to LLM:**")
                        st.code(context_content, language="markdown")

    # Chat input box for new student questions (Section 15)
    user_input = st.chat_input("Ask a question about the course lectures...")
    if user_input:
        process_user_query(pipeline, user_input)
        st.rerun()


def process_user_query(pipeline: RAGTeacherPipeline, query: str) -> None:
    """Processes a student question through the RAG pipeline and records the conversation turn."""
    import uuid
    cleaned_query = query.strip()
    if not cleaned_query:
        return

    turn_id = uuid.uuid4().hex[:8]
    selected_scope = st.session_state.get("lecture_scope", "All Lectures")
    response_style = st.session_state.get("response_style", "Explain Clearly")

    # Add structured user message (Section 4)
    st.session_state.messages.append({
        "id": f"msg_u_{turn_id}",
        "role": "user",
        "content": cleaned_query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "lecture_scope": selected_scope,
    })

    # Prepare conversation history for follow-up context awareness
    history_for_llm = [
        {
            "role": m["role"],
            "content": m["content"],
            "lecture_scope": m.get("lecture_scope"),
        }
        for m in st.session_state.messages[:-1]
    ]

    # Resolve active lecture filter
    source_filter = (
        None if selected_scope == "All Lectures"
        else selected_scope
    )

    # Execute end-to-end RAG answer generation with progress state (Section 16)
    with st.status("Consulting Course Knowledge Base...", expanded=False) as status_box:
        status_box.write("🔍 Searching course material...")
        try:
            status_box.write("🧠 Generating grounded answer...")
            answer_result: RAGAnswer = pipeline.answer_question(
                question=cleaned_query,
                conversation_history=history_for_llm,
                top_k=int(st.session_state.top_k),
                similarity_threshold=st.session_state.threshold,
                source_filter=source_filter,
                lecture_scope=source_filter,
                response_style=response_style,
            )
            status_box.update(label="Answer ready!", state="complete", expanded=False)
        except Exception as exc:
            logger.exception("Error processing question '%s': %s", cleaned_query, exc)
            status_box.update(label="Processing failed", state="error", expanded=True)
            st.error("I encountered an issue while consulting the course knowledge base. Please try asking again.")
            with st.expander("Technical Error Details", expanded=False):
                st.code(str(exc))
            answer_result = RAGAnswer(
                question=cleaned_query,
                answer="I encountered an error while consulting the course knowledge base. Please try asking again.",
                is_fallback=True,
                debug_info={"error": str(exc)},
            )

    # Add structured assistant response to session history (Section 4)
    st.session_state.messages.append({
        "id": f"msg_a_{turn_id}",
        "role": "assistant",
        "content": answer_result.answer,
        "citations": answer_result.sources,
        "sources": answer_result.sources,
        "lecture_scope": selected_scope,
        "response_style": response_style,
        "diagnostics": answer_result.debug_info,
        "debug_info": answer_result.debug_info,
        "context_text": answer_result.context_text,
        "is_fallback": answer_result.is_fallback,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def render_learning_center(pipeline: RAGTeacherPipeline) -> None:
    """Renders the Learning Center with Lecture Summary, Key Concepts, Explain, and Quiz."""
    st.subheader("🎓 Course Learning Center")
    st.caption("Structured, syllabus-grounded educational tools generated directly from indexed course lectures.")

    scope_options = get_lecture_scope_options(pipeline)
    current_learning_idx = 0
    if st.session_state.learning_scope in scope_options:
        current_learning_idx = scope_options.index(st.session_state.learning_scope)

    col_lscope, col_num_q = st.columns([3, 1])
    with col_lscope:
        selected_lscope = st.selectbox(
            "Lecture Scope for Learning Activity",
            options=scope_options,
            index=current_learning_idx,
            key="learning_scope_select",
            help="Choose 'All Lectures' or scope learning activities to a specific lecture.",
        )
        if selected_lscope != st.session_state.learning_scope:
            st.session_state.learning_scope = selected_lscope
            st.session_state.learning_result = None
            st.session_state.quiz_result = None
            st.session_state.quiz_current_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_feedback = {}
            st.session_state.quiz_complete = False

    with col_num_q:
        st.session_state.quiz_num_questions = st.selectbox(
            "Quiz Questions",
            options=[5, 10],
            index=0 if st.session_state.quiz_num_questions == 5 else 1,
            help="Number of multiple-choice questions to generate.",
        )

    # Activity action buttons
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📚 Lecture Summary", use_container_width=True, help="Generate a comprehensive structured overview of the lecture."):
            st.session_state.learning_mode = "summary"
            st.session_state.quiz_result = None
            with st.spinner("Generating grounded lecture summary..."):
                st.session_state.learning_result = pipeline.generate_summary(
                    lecture_scope=st.session_state.learning_scope,
                    top_k=int(st.session_state.top_k),
                    similarity_threshold=st.session_state.threshold,
                )
            st.rerun()

    with c2:
        if st.button("🎯 Key Concepts", use_container_width=True, help="Extract foundational concepts, definitions, and principles."):
            st.session_state.learning_mode = "concepts"
            st.session_state.quiz_result = None
            with st.spinner("Extracting syllabus-grounded key concepts..."):
                st.session_state.learning_result = pipeline.generate_key_concepts(
                    lecture_scope=st.session_state.learning_scope,
                    top_k=int(st.session_state.top_k),
                    similarity_threshold=st.session_state.threshold,
                )
            st.rerun()

    with c3:
        if st.button("🎓 Explain Lecture", use_container_width=True, help="Receive a student-friendly teacher explanation of this lecture."):
            st.session_state.learning_mode = "explain"
            st.session_state.quiz_result = None
            with st.spinner("Synthesizing pedagogical lecture explanation..."):
                st.session_state.learning_result = pipeline.explain_lecture(
                    lecture_scope=st.session_state.learning_scope,
                    top_k=int(st.session_state.top_k),
                    similarity_threshold=st.session_state.threshold,
                )
            st.rerun()

    with c4:
        if st.button("📝 Generate Quiz", use_container_width=True, help="Generate an interactive multiple choice quiz to test your comprehension."):
            st.session_state.learning_mode = "quiz"
            st.session_state.learning_result = None
            st.session_state.quiz_current_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_answers = {}
            st.session_state.quiz_feedback = {}
            st.session_state.quiz_complete = False
            with st.spinner(f"Generating {st.session_state.quiz_num_questions}-question grounded quiz..."):
                st.session_state.quiz_result = pipeline.generate_quiz(
                    lecture_scope=st.session_state.learning_scope,
                    num_questions=int(st.session_state.quiz_num_questions),
                    top_k=int(st.session_state.top_k),
                    similarity_threshold=st.session_state.threshold,
                )
            st.rerun()

    st.markdown("---")

    # Render Summary / Concepts / Explain
    if st.session_state.learning_mode in ("summary", "concepts", "explain"):
        res: LearningResult | None = st.session_state.learning_result
        if res is not None:
            st.markdown(f"### {res.title}")
            st.markdown(res.content)

            # Sources Used section (Section 18)
            if res.sources:
                with st.expander(f"📚 Sources Used ({len(res.sources)} references)", expanded=False):
                    for idx, src in enumerate(res.sources, start=1):
                        lec_title = src.get("lecture_title") or src.get("source_filename") or f"Lecture {idx}"
                        start_ts = src.get("timestamp_start", "00:00")
                        end_ts = src.get("timestamp_end", "00:00")
                        score = float(src.get("similarity_score", 0.0))
                        st.markdown(f"**{idx}. {lec_title}**")
                        st.markdown(f"- ⏱️ `{start_ts} – {end_ts}` | Similarity: `{score:.4f}`")
                        excerpt = (src.get("text") or "").strip()
                        if excerpt:
                            st.caption(f"*\"{excerpt[:200]}...\"*")
                        if idx < len(res.sources):
                            st.markdown("---")
        else:
            st.info("Select an activity above to generate learning materials.")

    # Render Quiz / MCQ Practice Mode (Sections 9, 10, 11, 20)
    elif st.session_state.learning_mode == "quiz":
        quiz_res: QuizResult | None = st.session_state.quiz_result
        if quiz_res is None:
            st.info("Click '📝 Generate Quiz' above to generate practice questions.")
            return

        if quiz_res.is_fallback:
            if quiz_res.error_message:
                st.warning(f"⚠️ {quiz_res.error_message}")
            else:
                st.info("No questions available for this lecture scope.")
            return

        questions = quiz_res.questions
        total_q = len(questions)
        if total_q == 0:
            st.warning("⚠️ No quiz questions could be generated from the selected lecture material.")
            return

        curr_idx = st.session_state.quiz_current_index

        if not st.session_state.quiz_complete:
            # Active question UI
            st.progress((curr_idx) / total_q)
            st.markdown(f"#### Question {curr_idx + 1} of {total_q}")

            q = questions[curr_idx]
            st.markdown(f"**{q.question}**")

            # Option selection
            ans_key = f"quiz_q_{curr_idx}"
            selected_option = st.radio(
                "Select your answer:",
                options=q.options,
                key=ans_key,
                index=None,
            )

            col_sub, col_skip = st.columns([2, 1])
            is_answered = curr_idx in st.session_state.quiz_answers

            with col_sub:
                if not is_answered:
                    if st.button("Submit Answer", type="primary", use_container_width=True):
                        if not selected_option:
                            st.warning("Please select an answer before submitting.")
                        else:
                            st.session_state.quiz_answers[curr_idx] = selected_option
                            is_correct = (selected_option.strip() == q.correct_answer.strip())
                            if is_correct:
                                st.session_state.quiz_score += 1
                                fb_text = f"🎉 **Correct!** {q.explanation}"
                            else:
                                fb_text = (
                                    f"❌ **Not quite.** According to the lecture, the correct answer is: "
                                    f"**{q.correct_answer}**\n\n💡 {q.explanation}"
                                )
                            st.session_state.quiz_feedback[curr_idx] = {
                                "is_correct": is_correct,
                                "feedback": fb_text,
                                "source": q.source,
                            }
                            st.rerun()
                else:
                    fb = st.session_state.quiz_feedback.get(curr_idx, {})
                    if fb.get("is_correct"):
                        st.success(fb.get("feedback", ""))
                    else:
                        st.error(fb.get("feedback", ""))

                    # Show source attribution
                    src_data = fb.get("source", {})
                    if src_data:
                        lec = src_data.get("lecture", "Course Lecture")
                        ts_start = src_data.get("timestamp_start", "")
                        ts_end = src_data.get("timestamp_end", "")
                        time_str = f"⏱️ `{ts_start} – {ts_end}`" if ts_start else ""
                        st.markdown(f"📚 **Source:** {lec} {time_str}")

                    # Next question or finish
                    btn_label = "Next Question ➡️" if (curr_idx + 1 < total_q) else "View Quiz Results 🏁"
                    if st.button(btn_label, use_container_width=True):
                        if curr_idx + 1 < total_q:
                            st.session_state.quiz_current_index += 1
                        else:
                            st.session_state.quiz_complete = True
                        st.rerun()

        else:
            # Quiz complete screen (Section 11)
            score = st.session_state.quiz_score
            pct = int((score / total_q) * 100) if total_q > 0 else 0

            st.balloons()
            st.markdown("### 🏆 Quiz Complete!")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Final Score", f"{score} / {total_q}")
            with col_m2:
                st.metric("Percentage", f"{pct}%")

            st.markdown("#### 📋 Question Breakdown")
            for idx, q_item in enumerate(questions):
                ans = st.session_state.quiz_answers.get(idx, "No answer")
                fb_item = st.session_state.quiz_feedback.get(idx, {})
                correct = fb_item.get("is_correct", False)
                icon = "✅ Correct" if correct else "❌ Incorrect"
                with st.expander(f"Question {idx + 1}: {icon}", expanded=not correct):
                    st.markdown(f"**Question:** {q_item.question}")
                    st.markdown(f"- **Your Answer:** `{ans}`")
                    st.markdown(f"- **Correct Answer:** `{q_item.correct_answer}`")
                    st.markdown(f"- **Explanation:** {q_item.explanation}")
                    if q_item.source:
                        s_lec = q_item.source.get("lecture", "Lecture")
                        s_st = q_item.source.get("timestamp_start", "")
                        s_en = q_item.source.get("timestamp_end", "")
                        st.markdown(f"- 📚 **Source:** `{s_lec}` ⏱️ `{s_st} – {s_en}`")

            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                if st.button("🔄 Restart Quiz", use_container_width=True, help="Retake this same quiz from the beginning."):
                    st.session_state.quiz_current_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_feedback = {}
                    st.session_state.quiz_complete = False
                    st.rerun()

            with col_res2:
                if st.button("➕ Start New Quiz", use_container_width=True, help="Generate a fresh set of questions from the lecture."):
                    st.session_state.quiz_current_index = 0
                    st.session_state.quiz_score = 0
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_feedback = {}
                    st.session_state.quiz_complete = False
                    with st.spinner("Generating fresh quiz..."):
                        st.session_state.quiz_result = pipeline.generate_quiz(
                            lecture_scope=st.session_state.learning_scope,
                            num_questions=int(st.session_state.quiz_num_questions),
                            top_k=int(st.session_state.top_k),
                            similarity_threshold=st.session_state.threshold,
                        )
                    st.rerun()


def render_evaluation_dashboard(pipeline: RAGTeacherPipeline) -> None:
    """Renders the RAG Quality Evaluation and Benchmarking dashboard."""
    import json
    st.subheader("📊 RAG Evaluation & Quality Benchmarking")
    st.caption(
        "Objective, reproducible evaluation framework measuring live retrieval quality, "
        "relevance gate abstention, and citation accuracy on real course lectures."
    )

    results_dir = RESULTS_DIR
    latest_json = results_dir / "latest_evaluation.json"

    # Load from disk if not yet in session state
    if st.session_state.evaluation_summary is None and latest_json.is_file():
        try:
            with open(latest_json, "r", encoding="utf-8") as f:
                st.session_state.evaluation_summary = json.load(f)
        except Exception:
            pass

    col_btn, col_opt = st.columns([2, 2])
    with col_btn:
        run_btn = st.button("🚀 Run Benchmark Evaluation", type="primary", use_container_width=True)
    with col_opt:
        fast_retrieval_only = st.checkbox(
            "Fast Evaluation (Skip generation, test retrieval & gate)",
            value=False,
            help="Evaluate retrieval metrics and relevance gate without running answer synthesis.",
        )

    if run_btn:
        benchmark_file = EVALUATION_DATASET_PATH
        if not benchmark_file.is_file():
            st.error(f"Evaluation benchmark dataset not found at `{benchmark_file}`.")
            return

        corpus_stats = pipeline.get_corpus_stats()
        if corpus_stats.get("total_chunks", 0) == 0:
            st.warning("⚠️ Evaluation notice: Course Knowledge Base has 0 indexed chunks. Please upload/index course lectures before running evaluation.")
            return

        try:
            with st.spinner("Executing reproducible RAG benchmark evaluation..."):
                questions = load_benchmark_dataset(benchmark_file)
                evaluator = RAGEvaluator(
                    pipeline=pipeline,
                    relevance_threshold=float(st.session_state.threshold or DEFAULT_RELEVANCE_THRESHOLD),
                    top_k=int(st.session_state.top_k),
                )
                summary: EvaluationSummary = evaluator.evaluate_dataset(
                    questions=questions,
                    dataset_path=str(benchmark_file),
                    evaluate_answers=not fast_retrieval_only,
                )
                save_evaluation_results(summary, output_dir=results_dir)
                st.session_state.evaluation_summary = summary.to_dict()
                st.success("✅ Benchmark evaluation completed and saved!")
                st.rerun()
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
            return

    data = st.session_state.evaluation_summary
    if not data:
        st.info("No evaluation results available yet. Click '🚀 Run Benchmark Evaluation' above to benchmark your RAG pipeline!")
        return

    rm = data.get("retrieval_metrics", {})
    rel = data.get("relevance_metrics", {})
    sa = data.get("score_analysis", {})
    cm = data.get("citation_metrics", {})
    am = data.get("answer_metrics", {})
    failures = data.get("failures", [])
    q_results = data.get("question_results", [])

    st.markdown(f"**Dataset:** `{data.get('dataset_path', 'Course Benchmark')}` | **Evaluated at:** `{data.get('timestamp', 'N/A')}`")

    # High-level Metrics Row 1
    st.markdown("### 🎯 Retrieval Performance")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Total Questions", data.get("total_questions", 0))
    with m2:
        st.metric("Recall@1", f"{rm.get('recall_at_1', 0.0):.1%}")
    with m3:
        st.metric("Recall@3", f"{rm.get('recall_at_3', 0.0):.1%}")
    with m4:
        st.metric("Recall@5", f"{rm.get('recall_at_5', 0.0):.1%}")
    with m5:
        st.metric("Hit@5", f"{rm.get('hit_at_5', 0.0):.1%}")
    with m6:
        st.metric("MRR", f"{rm.get('mrr', 0.0):.4f}")

    # High-level Metrics Row 2
    st.markdown("### 🛡️ Relevance Gate & Citation Quality")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        st.metric("Decision Accuracy", f"{rel.get('decision_accuracy', 0.0):.1%}")
    with r2:
        st.metric("False Acceptance (FAR)", f"{rel.get('false_acceptance_rate', 0.0):.1%}")
    with r3:
        st.metric("False Rejection (FRR)", f"{rel.get('false_rejection_rate', 0.0):.1%}")
    with r4:
        st.metric("Lecture Accuracy", f"{rm.get('lecture_retrieval_accuracy', 0.0):.1%}")
    with r5:
        st.metric("Citation Validity", f"{cm.get('citation_validity_rate', 0.0):.1%}")

    # Similarity Score Analysis
    with st.expander("📈 Cosine Similarity Score Distribution Analysis", expanded=False):
        c_sa1, c_sa2, c_sa3, c_sa4, c_sa5 = st.columns(5)
        with c_sa1:
            st.metric("Avg Score (Relevant)", f"{sa.get('avg_top_score_relevant', 0.0) or 0.0:.4f}")
        with c_sa2:
            st.metric("Avg Score (Rejected)", f"{sa.get('avg_top_score_rejected', 0.0) or 0.0:.4f}")
        with c_sa3:
            st.metric("Min Relevant Score", f"{sa.get('min_relevant_top_score', 0.0) or 0.0:.4f}")
        with c_sa4:
            st.metric("Max Irrelevant Score", f"{sa.get('max_irrelevant_top_score', 0.0) or 0.0:.4f}")
        with c_sa5:
            st.metric("Gate Threshold", f"{data.get('relevance_threshold', 0.32):.4f}")

    # Failure Analysis
    st.markdown("### 🔍 Failure Analysis")
    if failures:
        st.warning(f"⚠️ {len(failures)} question{'s' if len(failures) > 1 else ''} did not meet 100% benchmark expectations:")
        for idx, f in enumerate(failures, start=1):
            with st.expander(f"❌ {idx}. [{f.get('question_id')}] {f.get('question')}", expanded=True):
                st.markdown(f"- **Category:** `{f.get('question_type')}`")
                st.markdown(f"- **Top Similarity Score:** `{f.get('top_score', 0.0):.4f}`")
                st.markdown("**Reasons:**")
                for r in f.get("reasons", []):
                    st.markdown(f"  - ⚠️ {r}")
    else:
        st.success("🎉 All benchmark questions passed 100% with zero failures!")

    # Question Breakdown Table with Interactive Filters
    st.markdown("### 📋 Per-Question Benchmark Results")
    types = ["All"] + sorted(list(set(q.get("question_type") for q in q_results)))
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        selected_type = st.selectbox("Filter by Question Type", options=types, index=0)
    with f_col2:
        outcome_filter = st.selectbox("Filter by Outcome", options=["All Questions", "Passed Only", "Failed Only"], index=0)

    filtered_q = q_results
    if selected_type != "All":
        filtered_q = [q for q in filtered_q if q.get("question_type") == selected_type]
    if outcome_filter == "Passed Only":
        filtered_q = [q for q in filtered_q if not q.get("failure_reasons")]
    elif outcome_filter == "Failed Only":
        filtered_q = [q for q in filtered_q if q.get("failure_reasons")]

    if filtered_q:
        table_rows = []
        for q in filtered_q:
            passed = len(q.get("failure_reasons", [])) == 0
            status_icon = "✅ Pass" if passed else "❌ Fail"
            exp_dec = "Answer" if q.get("expected_decision") else "Abstain"
            act_dec = "Accepted" if q.get("actual_decision") else "Abstained"
            table_rows.append({
                "ID": q.get("question_id"),
                "Question": q.get("question"),
                "Type": q.get("question_type"),
                "Expected": exp_dec,
                "Actual": act_dec,
                "Top Score": f"{q.get('top_score', 0.0):.4f}",
                "Hit@5": "✅" if q.get("hit_at_5") else ("N/A" if not q.get("expected_decision") else "❌"),
                "MRR": f"{q.get('reciprocal_rank', 0.0):.3f}" if q.get("expected_decision") else "N/A",
                "Status": status_icon,
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No questions match the selected filters.")


def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    render_header()

    pipeline = get_rag_pipeline(
        api_key=st.session_state.api_key,
        model_name=st.session_state.model_name,
    )

    render_sidebar(pipeline)
    render_course_knowledge_base_section(pipeline)

    tab_chat, tab_learning, tab_eval = st.tabs([
        "💬 Student Chat Assistant",
        "🎓 Learning Center",
        "📊 RAG Evaluation",
    ])
    with tab_chat:
        render_chat_interface(pipeline)
    with tab_learning:
        render_learning_center(pipeline)
    with tab_eval:
        render_evaluation_dashboard(pipeline)


if __name__ == "__main__":
    main()
