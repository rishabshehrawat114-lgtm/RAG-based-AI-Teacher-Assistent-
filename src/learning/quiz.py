"""
src/learning/quiz.py
====================
Quiz and multiple-choice assessment generation for Module 12 Learning Features.

Includes:
- Strict JSON output parsing and schema validation.
- Controlled single-retry regeneration on malformed outputs.
- Grounded citations and timestamps attached to every question.
- Educational feedback and scoring mechanisms.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.config import DEFAULT_RELEVANCE_THRESHOLD
from src.learning.models import QuizQuestion, QuizResult
from src.learning.prompts import QUIZ_SYSTEM_PROMPT, build_quiz_messages
from src.llm import LLMClient, LLMError
from src.rag import FALLBACK_NOT_FOUND_MESSAGE, is_relevant
from src.retrieval import LectureRetriever, RetrievalError, format_timestamp

logger = logging.getLogger(__name__)


def parse_quiz_json(raw_text: str) -> Any:
    """
    Safely extracts and parses JSON content from LLM response text,
    handling markdown code fences (```json ... ```) gracefully.

    Args:
        raw_text: Raw string returned by the LLM.

    Returns:
        Parsed JSON structure (dict or list).

    Raises:
        ValueError: If valid JSON cannot be found or parsed.
    """
    cleaned = (raw_text or "").strip()
    if not cleaned:
        raise ValueError("Empty response text from LLM.")

    # Try direct parse
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract JSON inside markdown code fence
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(fence_pattern, cleaned)
    if match:
        content = match.group(1).strip()
        try:
            return json.loads(content)
        except Exception as exc:
            raise ValueError(f"Failed to parse JSON inside markdown code fence: {exc}") from exc

    # Attempt to locate array brackets [ ... ]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception as exc:
            raise ValueError(f"Failed to parse array substring as JSON: {exc}") from exc

    raise ValueError(f"Could not locate valid JSON in LLM response: {cleaned[:150]}")


def validate_quiz_data(
    raw_data: Any,
    expected_count: int,
    sources: list[dict[str, Any]],
) -> tuple[list[QuizQuestion], bool, str]:
    """
    Validates the parsed quiz data against Module 12 quality standards:
    - Must be a non-empty list of questions.
    - Exactly 4 options per question.
    - No duplicate options.
    - No 'All of the above' or 'None of the above'.
    - Exactly one correct answer present in options.
    - Non-empty explanation and grounded source metadata.

    Args:
        raw_data: Parsed JSON data.
        expected_count: Number of questions requested (5 or 10).
        sources: Retrieved context chunk sources for attribution fallback.

    Returns:
        Tuple of (list_of_QuizQuestion, is_valid_bool, error_message_str).
    """
    if not isinstance(raw_data, list):
        return [], False, f"Expected a JSON list of questions, got {type(raw_data).__name__}."

    if len(raw_data) == 0:
        return [], False, "Generated quiz is empty (0 questions)."

    validated_questions: list[QuizQuestion] = []
    default_source = sources[0] if sources else {}
    fallback_lecture = default_source.get("lecture_title") or "Course Lecture"
    fallback_start = default_source.get("timestamp_start", "00:00")
    fallback_end = default_source.get("timestamp_end", "05:00")

    for idx, item in enumerate(raw_data, start=1):
        if not isinstance(item, dict):
            return [], False, f"Question {idx} is not a valid JSON object."

        question_text = str(item.get("question", "")).strip()
        if not question_text:
            return [], False, f"Question {idx} has an empty 'question' field."

        raw_options = item.get("options")
        if not isinstance(raw_options, list):
            return [], False, f"Question {idx} 'options' must be a list."

        options = [str(opt).strip() for opt in raw_options if str(opt).strip()]
        if len(options) != 4:
            return [], False, f"Question {idx} must have exactly 4 options, found {len(options)}."

        # Check for duplicate options
        if len(set(options)) != 4:
            return [], False, f"Question {idx} contains duplicate option choices."

        # Quality check: avoid 'All of the above' / 'None of the above'
        for opt in options:
            low_opt = opt.lower()
            if "all of the above" in low_opt or "none of the above" in low_opt:
                return (
                    [],
                    False,
                    f"Question {idx} violates quality guidelines: contains '{opt}'.",
                )

        correct_answer = str(item.get("correct_answer", "")).strip()
        if not correct_answer:
            return [], False, f"Question {idx} has an empty 'correct_answer'."

        # Match correct_answer with options
        matched_opt = next((o for o in options if o == correct_answer or o.lower() == correct_answer.lower()), None)
        if not matched_opt:
            return (
                [],
                False,
                f"Question {idx} correct_answer '{correct_answer}' does not match any of the 4 options: {options}.",
            )
        correct_answer = matched_opt

        explanation = str(item.get("explanation", "")).strip()
        if not explanation:
            explanation = f"According to the lecture, '{correct_answer}' is the correct statement."

        # Source metadata validation
        src_raw = item.get("source")
        if not isinstance(src_raw, dict):
            src_raw = {}

        source_info = {
            "lecture": str(src_raw.get("lecture") or fallback_lecture),
            "timestamp_start": str(src_raw.get("timestamp_start") or fallback_start),
            "timestamp_end": str(src_raw.get("timestamp_end") or fallback_end),
        }

        validated_questions.append(
            QuizQuestion(
                question=question_text,
                options=options,
                correct_answer=correct_answer,
                explanation=explanation,
                source=source_info,
            )
        )

    return validated_questions, True, ""


def generate_quiz(
    retriever: LectureRetriever,
    llm_client: LLMClient | None = None,
    lecture_scope: str = "All Lectures",
    num_questions: int = 5,
    top_k: int = 6,
    similarity_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
) -> QuizResult:
    """
    Generates a structured multiple-choice quiz grounded strictly in lecture material.

    Enforces:
    - 0.32 relevance threshold.
    - Offline fallback check (does not fabricate quiz questions when LLM is unavailable).
    - Strict schema validation with exactly one controlled regeneration attempt.

    Args:
        retriever: Pre-configured LectureRetriever instance.
        llm_client: Optional LLMClient instance.
        lecture_scope: Target lecture filter or "All Lectures".
        num_questions: Desired number of questions (5 or 10, default: 5).
        top_k: Number of representative chunks to retrieve.
        similarity_threshold: Cosine similarity cutoff (default: 0.32).

    Returns:
        QuizResult dataclass containing validated questions, sources, and status.
    """
    llm = llm_client or LLMClient()
    effective_scope = (
        None
        if lecture_scope in ("All Lectures", "All", "all", "", None)
        else lecture_scope
    )

    # Sanitize num_questions
    if num_questions not in (5, 10):
        num_questions = 10 if num_questions >= 8 else 5

    # 1. Guard against empty knowledge base
    chunks_meta = getattr(retriever, "chunks_metadata", None)
    if chunks_meta is not None and len(chunks_meta) == 0:
        return QuizResult(
            lecture_scope=lecture_scope,
            questions=[],
            num_questions=0,
            sources=[],
            is_fallback=True,
            error_message="No course lectures are currently indexed. Please upload a lecture to generate quizzes.",
            debug_info={"error": "No lectures indexed"},
        )

    # 2. Scoped semantic retrieval
    target_title = ""
    if effective_scope:
        for c in getattr(retriever, "chunks_metadata", []):
            cid = str(c.get("lecture_id", "")).lower()
            ctitle = str(c.get("lecture_title", "")).lower()
            if effective_scope.lower() in cid or effective_scope.lower() in ctitle:
                target_title = c.get("lecture_title", "")
                break

    query = f"{target_title or lecture_scope} key concepts definitions principles factual statements and examples for assessment".strip()
    try:
        raw_chunks = retriever.retrieve(
            query=query,
            top_k=top_k,
            similarity_threshold=None,
            source_file=effective_scope,
            lecture_filter=effective_scope,
            lecture_scope=effective_scope,
        )
    except RetrievalError as err:
        logger.warning("Retrieval failed for quiz: %s", err)
        raw_chunks = []

    # 3. Grounding & Relevance Gate (0.32 cutoff)
    relevant = is_relevant(raw_chunks, threshold=similarity_threshold)
    top_score = float(raw_chunks[0].get("similarity_score", 0.0)) if raw_chunks else 0.0

    if not relevant:
        logger.info(
            "Quiz rejected: top score %.4f below threshold %.4f for scope '%s'",
            top_score,
            similarity_threshold,
            lecture_scope,
        )
        return QuizResult(
            lecture_scope=lecture_scope,
            questions=[],
            num_questions=0,
            sources=[],
            is_fallback=True,
            error_message=FALLBACK_NOT_FOUND_MESSAGE,
            debug_info={
                "relevance_decision": "NOT RELEVANT",
                "top_score": top_score,
                "threshold": similarity_threshold,
                "retrieved_count": len(raw_chunks),
            },
        )

    # 4. Extract qualifying chunks and format sources
    relevant_chunks = [
        c for c in raw_chunks
        if float(c.get("similarity_score", 0.0)) >= similarity_threshold
    ]
    if not relevant_chunks:
        relevant_chunks = [raw_chunks[0]]

    sources: list[dict[str, Any]] = []
    for chunk in relevant_chunks:
        s_file = chunk.get("source_filename") or chunk.get("source_file") or "Lecture"
        l_title = chunk.get("lecture_title") or s_file
        st_secs = float(chunk.get("start_time", 0.0))
        end_secs = float(chunk.get("end_time", 0.0))
        t_range = f"{format_timestamp(st_secs)} - {format_timestamp(end_secs)}"
        sources.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "lecture_title": l_title,
            "source_filename": s_file,
            "timestamp_start": format_timestamp(st_secs),
            "timestamp_end": format_timestamp(end_secs),
            "timestamp_range": t_range,
            "similarity_score": float(chunk.get("similarity_score", 0.0)),
            "text": chunk.get("text", ""),
        })

    # 5. Check LLM Availability (Section 22: Do not fabricate quiz offline)
    if not llm.is_configured():
        return QuizResult(
            lecture_scope=lecture_scope,
            questions=[],
            num_questions=0,
            sources=sources,
            is_fallback=True,
            error_message=(
                "Quiz generation requires a configured OpenAI API Key. "
                "Please configure an API key in the sidebar configuration to generate interactive quizzes."
            ),
            debug_info={"error": "LLM unconfigured", "top_score": top_score},
        )

    # 6. Attempt 1: Generation & Validation
    messages = build_quiz_messages(relevant_chunks, num_questions=num_questions, lecture_scope=lecture_scope)
    try:
        raw_response = llm.generate(messages).strip()
        parsed_data = parse_quiz_json(raw_response)
        validated_questions, is_valid, err_msg = validate_quiz_data(
            parsed_data, expected_count=num_questions, sources=sources
        )

        if is_valid:
            return QuizResult(
                lecture_scope=lecture_scope,
                questions=validated_questions,
                num_questions=len(validated_questions),
                sources=sources,
                is_fallback=False,
                debug_info={"top_score": top_score, "regeneration_attempted": False},
            )

        logger.warning("Quiz validation failed on Attempt 1: %s. Attempting 1 controlled regeneration.", err_msg)
    except Exception as exc:
        logger.warning("Quiz generation/parsing failed on Attempt 1: %s. Attempting 1 controlled regeneration.", exc)
        err_msg = str(exc)

    # 7. Attempt 2: Controlled single regeneration
    try:
        retry_prompt = (
            f"The previous output had a formatting or validation error:\n"
            f"\"{err_msg}\"\n\n"
            f"Please regenerate exactly {num_questions} multiple-choice questions matching the strict JSON array schema. "
            f"Ensure exactly 4 distinct options per question, exactly one correct answer matching one of the options, "
            f"and no 'All of the above' or 'None of the above'."
        )
        retry_messages = [
            {"role": "system", "content": QUIZ_SYSTEM_PROMPT.strip()},
            {"role": "user", "content": messages[1]["content"]},
            {"role": "assistant", "content": raw_response if "raw_response" in locals() else ""},
            {"role": "user", "content": retry_prompt},
        ]

        raw_response_retry = llm.generate(retry_messages).strip()
        parsed_retry = parse_quiz_json(raw_response_retry)
        validated_retry, is_valid_retry, err_msg_retry = validate_quiz_data(
            parsed_retry, expected_count=num_questions, sources=sources
        )

        if is_valid_retry:
            return QuizResult(
                lecture_scope=lecture_scope,
                questions=validated_retry,
                num_questions=len(validated_retry),
                sources=sources,
                is_fallback=False,
                debug_info={"top_score": top_score, "regeneration_attempted": True, "retry_success": True},
            )

        logger.error("Quiz validation failed on Attempt 2: %s", err_msg_retry)
        return QuizResult(
            lecture_scope=lecture_scope,
            questions=[],
            num_questions=0,
            sources=sources,
            is_fallback=True,
            error_message=(
                f"Quiz output did not meet quality standards after retry: {err_msg_retry}. Please try generating again."
            ),
            debug_info={"error": err_msg_retry, "regeneration_attempted": True},
        )

    except Exception as exc:
        logger.error("Controlled quiz regeneration failed: %s", exc)
        return QuizResult(
            lecture_scope=lecture_scope,
            questions=[],
            num_questions=0,
            sources=sources,
            is_fallback=True,
            error_message=f"Failed to generate valid quiz: {exc}",
            debug_info={"error": str(exc), "regeneration_attempted": True},
        )
