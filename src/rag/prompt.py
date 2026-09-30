"""
src/rag/prompt.py
=================
Prompt engineering and context formatting templates for the RAG AI Teacher Assistant.

Enforces strict grounding on retrieved lecture excerpts, prevents hallucination,
provides explicit fallback responses for missing topics, and manages conversational history.
"""

from __future__ import annotations

import re
from typing import Any

from src.config import DEFAULT_CONVERSATION_TURNS
from src.retrieval import format_timestamp

FALLBACK_NOT_FOUND_MESSAGE: str = (
    "I couldn't find enough information about this topic in the provided course material."
)

RESPONSE_STYLES: tuple[str, ...] = (
    "Explain Clearly",
    "Short Answer",
    "Detailed Explanation",
)
DEFAULT_RESPONSE_STYLE: str = "Explain Clearly"

RESPONSE_STYLE_PROMPTS: dict[str, str] = {
    "Short Answer": (
        "RESPONSE STYLE: Short Answer\n"
        "- Target length: approximately 30–70 words (usually 1 short paragraph).\n"
        "- Be extremely concise and answer the student's question directly in 1–3 clear sentences.\n"
        "- Provide minimal extra explanation while remaining strictly grounded in the course evidence.\n"
        "- Optionally include one tiny example (e.g., `3 + 2 → 5`) if it immediately clarifies the concept.\n"
        "- Do NOT reproduce surrounding lecture text or classroom discussion.\n"
        "- Do NOT include timestamps or 'Review timestamp...' notes in the answer; source citations are displayed separately in the dedicated UI.\n"
    ),
    "Explain Clearly": (
        "RESPONSE STYLE: Explain Clearly (Default)\n"
        "- Target length: approximately 60–120 words (1–3 short paragraphs or bullet points).\n"
        "- Explain concepts clearly in an encouraging, student-friendly tone using simple terminology.\n"
        "- Provide clear explanation without copying unnecessary lecture transcript.\n"
        "- For explanatory questions, structure your answer using clear sections when helpful:\n"
        "  ### Explanation\n"
        "  Direct, clear explanation (2-3 sentences) based strictly on the lecture.\n"
        "  ### Example\n"
        "  A single compact example (1-2 lines) if the lecture excerpts contain one.\n"
        "  ### Key Takeaway\n"
        "  One short takeaway sentence explaining the concept.\n"
        "- Note: For simple factual or definition questions, provide a concise direct answer rather than forcing all sections.\n"
        "- Ignore surrounding lecture discussion, tangents, and classroom conversation.\n"
        "- Do NOT include timestamps or 'Review timestamp...' notes in the answer; source citations are displayed separately in the dedicated UI.\n"
    ),
    "Detailed Explanation": (
        "RESPONSE STYLE: Detailed Explanation\n"
        "- Target length: approximately 120–220 words (use only when explicitly chosen).\n"
        "- Provide a comprehensive, structured explanation covering all relevant aspects mentioned in the lecture excerpts.\n"
        "- Break down complex points step-by-step with structured sections (### Explanation, ### Example, ### Key Takeaway).\n"
        "- Even in detailed mode, answer the question directly and do NOT dump or quote the raw transcript.\n"
        "- Do NOT include timestamps or 'Review timestamp...' notes in the answer; source citations are displayed separately in the dedicated UI.\n"
    ),
}

SYSTEM_TEACHER_PROMPT: str = """You are an expert, encouraging, and concise AI Teacher Assistant for educational courses and lecture video content.

Your task is to answer the student's question directly, faithfully, and concisely based strictly on the provided lecture transcript excerpts.

CRITICAL INSTRUCTIONS:
1. Context as Evidence, Not Text to Reproduce: You are a teaching assistant. Answer the student's question directly using the provided course context as evidence. Do not copy, reproduce, or summarize the entire context. Extract only the information needed to answer the question.
2. Answer Only What Was Asked: Answer the question that was asked, not everything mentioned in the retrieved lecture transcript. Source context may contain unrelated surrounding lecture material. Ignore information that is not necessary for the question.
3. No Transcript Dumping: Never reproduce the transcript. Do NOT summarize the entire retrieved chunk unless the student explicitly asks for a summary. Do NOT include unrelated examples from the lecture. Do NOT include classroom conversation, speaker mannerisms, or spoken filler ("right?", "okay?", "let's try this"). Do NOT repeat the question.
4. Conciseness & Structure: Keep your answer focused on exactly what was asked. Prefer 1–3 short paragraphs or concise bullet points. Give an example only when it helps explain the answer. Avoid unnecessary background information.
5. Missing Information: If the provided excerpts do not contain enough information to answer the question, or if no relevant context is found, you MUST state:
   "I couldn't find enough information about this topic in the provided course material."
   Do NOT pretend that unsupported information came from the course, and never invent information outside the retrieved course material.
6. Conversation Continuity: When the student asks follow-up questions (such as "Can you give an example?" or "Explain it further"), use previous conversation turns to understand what "it" or the topic refers to, while keeping all answers grounded in the provided lecture context.
7. Clean Answers Without Citation Duplication: Do NOT include timestamp ranges or notes like 'Review the timestamp interval...' in the answer text. The source video and timestamp interval are automatically rendered separately by the UI citation card.
"""

FOLLOW_UP_PHRASES: tuple[str, ...] = (
    "give me an example",
    "give an example",
    "show an example",
    "can you give an example",
    "can you show an example",
    "can you explain",
    "explain more",
    "explain further",
    "tell me more",
    "what about",
    "how about",
    "why is that",
    "why does it",
    "how does it work",
    "how does that work",
    "what does that mean",
    "what do you mean",
    "can you clarify",
    "more details",
    "another example",
    "and why",
    "and how",
    "and what",
)

FOLLOW_UP_PRONOUNS: set[str] = {
    "it", "this", "that", "these", "those", "they", "them", "its", "their", "such"
}


def is_follow_up_query(query: str) -> bool:
    """
    Determines if a student query is a contextual follow-up to a previous discussion.

    Checks for short length combined with follow-up phrases, pronouns (it, that, this),
    or elliptical question patterns.
    """
    cleaned = (query or "").strip().lower()
    if not cleaned:
        return False

    for phrase in FOLLOW_UP_PHRASES:
        if phrase in cleaned:
            return True

    words = re.findall(r"\b\w+\b", cleaned)
    if not words:
        return False

    # Short questions (<= 7 words) containing pronouns or demonstratives
    if len(words) <= 7:
        if any(w in FOLLOW_UP_PRONOUNS for w in words):
            return True
        if words[0] in ("why", "how", "and", "or", "so", "what", "where"):
            return True

    return False


def extract_example_topic(query: str) -> str | None:
    """
    Extracts the subject topic from example-seeking queries.
    (e.g., 'Give me an example of an expression' -> 'an expression')
    """
    cleaned = (query or "").strip().rstrip(".?!")
    m = re.search(
        r"^(?:can\s+you\s+)?(?:give|show)(?:\s+me)?\s+(?:an?\s+)?example\s+of\s+(.+)$",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().lower()
    m2 = re.search(r"^(?:an?\s+)?example\s+of\s+(.+)$", cleaned, re.IGNORECASE)
    if m2:
        return m2.group(1).strip().lower()
    return None


def build_retrieval_query(
    query: str,
    conversation_history: list[dict[str, Any]] | None = None,
    current_scope: str | None = None,
    max_prev_words: int = 15,
) -> tuple[str, bool]:
    """
    Constructs a lightweight contextual retrieval query for follow-up questions
    and controlled example queries on foundational course concepts.

    If the current query is an elliptical or contextual follow-up, attaches key context
    from the most recent user turn without bloating the query. If the user recently
    switched lecture scope, preserves isolation to prevent stale context bleeding.

    Args:
        query: The current student question.
        conversation_history: List of previous conversation turn dictionaries.
        current_scope: The currently selected lecture scope.
        max_prev_words: Maximum words to borrow from previous user question.

    Returns:
        Tuple of (effective_retrieval_query, is_follow_up_flag).
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return "", False

    # 1. Contextual follow-up query with conversation history
    if conversation_history and is_follow_up_query(cleaned):
        last_user_msg: dict[str, Any] | None = None
        for msg in reversed(conversation_history):
            if msg.get("role") == "user" and (msg.get("content") or "").strip():
                last_user_msg = msg
                break

        if last_user_msg:
            # Prevent cross-lecture context bleeding if the scope explicitly changed
            prev_scope = last_user_msg.get("lecture_scope")
            if not (
                prev_scope
                and current_scope
                and prev_scope not in ("All Lectures", "All", "")
                and current_scope not in ("All Lectures", "All", "")
                and prev_scope.strip().lower() != current_scope.strip().lower()
            ):
                prev_content = (last_user_msg.get("content") or "").strip()
                if prev_content:
                    prev_words = prev_content.split()
                    if len(prev_words) > max_prev_words:
                        prev_core = " ".join(prev_words[:max_prev_words]).rstrip(".,!?;:")
                    else:
                        prev_core = prev_content.rstrip(".,!?;:")
                    return f"{prev_core}. {cleaned}", True

    # 2. Standalone example request on core course concepts (Controlled query expansion)
    topic = extract_example_topic(cleaned)
    if topic:
        if "express" in topic:
            return "example of a Python expression such as 3 + 2", True
        elif "variable" in topic:
            return "example of a Python variable assignment such as x = 10", True
        elif "int" in topic or "float" in topic:
            return "example of Python int and float numbers", True

    return cleaned, False


def format_rag_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunk dictionaries into a structured text context block
    for inclusion in the RAG prompt.

    Args:
        retrieved_chunks: List of chunk dictionaries returned by LectureRetriever.

    Returns:
        Structured context string with clear excerpt numbers, source filenames,
        formatted MM:SS timestamps, and chunk texts.
    """
    if not retrieved_chunks:
        return "[No matching lecture excerpts were found in the indexed course material.]"

    context_parts: list[str] = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        source_file = chunk.get("source_file") or chunk.get("source_transcript") or "Lecture"
        start_secs = float(chunk.get("start_time", 0.0))
        end_secs = float(chunk.get("end_time", 0.0))
        start_str = format_timestamp(start_secs)
        end_str = format_timestamp(end_secs)
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        score = chunk.get("similarity_score")
        score_str = f" (Similarity: {score:.3f})" if score is not None else ""

        text = chunk.get("text", "").strip()

        lecture_title = chunk.get("lecture_title") or chunk.get("source_filename") or source_file
        lecture_line = f"Lecture: {lecture_title}\n" if lecture_title else ""

        part = (
            f"--- EXCERPT {idx} ---\n"
            f"{lecture_line}"
            f"Source Video: {source_file}\n"
            f"Timestamp: {start_str} - {end_str}\n"
            f"Chunk ID: {chunk_id}{score_str}\n"
            f"Content:\n{text}\n"
        )
        context_parts.append(part)

    return "\n".join(context_parts)


def build_rag_messages(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    conversation_history: list[dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    max_history_turns: int = DEFAULT_CONVERSATION_TURNS,
    response_style: str = DEFAULT_RESPONSE_STYLE,
) -> list[dict[str, str]]:
    """
    Assembles the complete list of OpenAI chat messages including system instructions,
    controlled recent conversation turns, retrieved lecture context, student question,
    and pedagogical response style guidance.

    Args:
        question: Current question string asked by the student.
        retrieved_chunks: Top-K retrieved chunk dictionaries.
        conversation_history: Optional list of past chat messages ({'role': 'user'|'assistant', 'content': str}).
        system_prompt: Optional custom system prompt (defaults to SYSTEM_TEACHER_PROMPT).
        max_history_turns: Number of recent turns (user/assistant pairs) to include for continuity.
        response_style: Selected pedagogical response style ("Explain Clearly", "Short Answer", "Detailed Explanation").

    Returns:
        List of message dicts formatted for OpenAI Chat Completion API.
    """
    base_sys = (system_prompt or SYSTEM_TEACHER_PROMPT).strip()
    style_guidance = RESPONSE_STYLE_PROMPTS.get(response_style, "")
    full_system = f"{base_sys}\n\n{style_guidance}".strip() if style_guidance else base_sys

    messages: list[dict[str, str]] = [
        {"role": "system", "content": full_system}
    ]

    # Include recent conversation turns for context continuity without bloating the prompt
    if conversation_history:
        recent_history = conversation_history[-(max_history_turns * 2):]
        for msg in recent_history:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Format the retrieved lecture context
    context_text = format_rag_context(retrieved_chunks)

    # Current user prompt containing grounded context + student question
    user_content = (
        f"LECTURE CONTEXT:\n"
        f"{context_text}\n\n"
        f"STUDENT QUESTION:\n"
        f"{question.strip()}\n\n"
        f"INSTRUCTIONS FOR ANSWERING:\n"
        f"- You are a teaching assistant. Answer the student's question directly using the provided course context as evidence.\n"
        f"- Do not copy, reproduce, or summarize the entire context. Extract only the information needed to answer the question.\n"
        f"- Answer the question that was asked, not everything mentioned in the retrieved lecture transcript.\n"
        f"- Source context may contain unrelated surrounding lecture material. Ignore information that is not necessary for the question.\n"
        f"- Follow the requested style ({response_style}): keep it focused, student-friendly, and concise. Do NOT dump the transcript.\n"
        f"- If the information is not present, respond with:\n"
        f"\"{FALLBACK_NOT_FOUND_MESSAGE}\"\n\n"
        f"REFERENCE EXAMPLES FOR CONCISENESS:\n"
        f"Example 1:\n"
        f"Question: \"What does Python do when it evaluates an expression?\"\n"
        f"Good Answer: \"When Python evaluates an expression, it calculates it and produces a single resulting value. Python stores the result, not the original expression.\n\nExample: 3 + 2 -> 5.\"\n"
        f"Bad Answer: A multi-paragraph reproduction of the transcript discussing type(), floats, parentheses, operators, and classroom dialogue.\n\n"
        f"Example 2:\n"
        f"Question: \"What is a variable in Python?\"\n"
        f"Good Answer: \"A variable is a name that refers to a value stored in memory.\n\nExample:\nx = 10\nHere, x refers to the value 10.\"\n"
        f"Bad Answer: A long transcript dumping every remark made in the lecture about variables.\n\n"
        f"Example 3:\n"
        f"Question: \"What is the difference between int and float?\"\n"
        f"Good Answer: \"`int` represents whole numbers, while `float` represents numbers with decimal values.\n\nExample:\n10 -> int\n10.5 -> float\"\n"
        f"Bad Answer: A lengthy transcript reproduction containing unrelated lecture exercises."
    )

    messages.append({"role": "user", "content": user_content})
    return messages


FILLER_PATTERNS: tuple[str, ...] = (
    r"^(?:and|so|right\?|okay\?|yeah|well|basically|you know|i mean|like|so basically|remember,?\s*)+\s*",
    r"\b(?:right\?|okay\?)\s*$",
)


def clean_sentence(s: str) -> str:
    """Cleans conversational speech filler and spoken artifacts from a sentence."""
    s = s.strip()
    for pat in FILLER_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
    # Normalize common spoken lead-ins to sound professional and direct
    s = re.sub(r"^(?:What it does is it|What it does is)\s*", "Python ", s, flags=re.IGNORECASE)
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def extract_content_keywords(query: str) -> set[str]:
    """Extracts informative query terms and common variations."""
    stop_words = {
        "what", "does", "when", "it", "an", "a", "the", "in", "is", "do", "to",
        "of", "and", "for", "are", "can", "you", "give", "me", "how", "why",
        "between", "python", "this", "that", "with", "tell", "about"
    }
    words = [w.lower() for w in re.findall(r"\b\w+\b", query) if w.lower() not in stop_words]
    expanded = set(words)
    for w in words:
        if w.startswith("evaluat"):
            expanded.update(["evaluate", "evaluates", "evaluating", "evaluation"])
        elif w.startswith("express"):
            expanded.update(["expression", "expressions"])
        elif w.startswith("variab"):
            expanded.update(["variable", "variables"])
        elif w in ("int", "integer"):
            expanded.update(["int", "integer", "integers"])
        elif w in ("float", "floats"):
            expanded.update(["float", "floats", "decimal", "decimals"])
    return expanded


def synthesize_grounded_answer(
    question: str,
    source_text: str,
    lecture_title: str,
    timestamp_range: str,
    response_style: str = DEFAULT_RESPONSE_STYLE,
) -> str:
    """
    Synthesizes a short, direct, student-friendly answer grounded in the retrieved excerpt.
    Used for offline execution and fallback modes to ensure answers remain concise
    and never dump raw lecture transcripts.
    """
    raw_sentences = [
        s.strip() for s in re.split(r"(?<=[.!?])\s+", source_text)
        if len(s.strip()) > 8
    ]
    if not raw_sentences:
        raw_sentences = [source_text.strip()]

    # If already very short (e.g. synthetic test passage), keep it concise
    if len(raw_sentences) <= 2:
        clean_text = " ".join(clean_sentence(s) for s in raw_sentences)
        if response_style == "Short Answer":
            return clean_text
        elif response_style == "Detailed Explanation":
            return (
                f"### Explanation\n{clean_text}\n\n"
                f"### Key Takeaway\nThis foundational concept is covered in {lecture_title}."
            )
        else:
            return (
                f"### Explanation\n{clean_text}\n\n"
                f"### Key Takeaway\nThis foundational concept is introduced and demonstrated in {lecture_title}."
            )

    q_keywords = extract_content_keywords(question)
    bad_dialogue = [
        "let's have you", "type in", "off the top", "console", "what are the values",
        "where is the operator", "when you're inserting", "try these", "i don't know",
        "is easy to operate", "sorry. yeah.", "what's inside the parenthesis",
        "big idea slide", "decided to insert these", "was there another question",
        "why do we want to give names", "ok, so why do we", "just uses 10",
        "whatever it is", "here's some examples", "try these in our console"
    ]

    scored: list[tuple[int, int, str]] = []
    examples: list[str] = []

    for idx, s in enumerate(raw_sentences):
        cleaned = clean_sentence(s)
        s_lower = cleaned.lower()

        # Questions asked by students or teacher during lecture should not be used as answers
        if s.endswith("?") or cleaned.endswith("?"):
            continue

        if any(b in s_lower for b in bad_dialogue):
            continue

        score = 0
        for kw in q_keywords:
            if kw in s_lower:
                score += 15

        bonus_phrases = [
            "evaluates it to", "single value", "one single value", "stores the result",
            "stores the value", "not store the expression", "never stores the expression",
            "reduces to", "refers to", "binds this value", "represents", "is defined as",
            "the difference between", "points to that", "cannot be subdivided",
            "numbers with decimals", "basic primitive objects", "whole numbers",
            "assignment operator", "equal sign binds", "variable name"
        ]
        for bp in bonus_phrases:
            if bp in s_lower:
                score += 25

        if any(ex_cue in s_lower for ex_cue in ["like 5", "like 3 plus 2", "3 plus 2", "x =", "3.14", "5.0", "pi ="]):
            if not s.endswith("?") and not any(b in s_lower for b in bad_dialogue):
                examples.append(cleaned)

        if score > 0:
            scored.append((score, idx, cleaned))

    scored.sort(key=lambda x: -x[0])
    selected_sentences = (
        [s for _, _, s in scored[:4]]
        if scored
        else [clean_sentence(s) for s in raw_sentences if not s.endswith("?")][:2]
    )
    if not selected_sentences:
        selected_sentences = [clean_sentence(raw_sentences[0])]

    chosen_example = None
    if examples:
        for ex in examples:
            if 3 <= len(ex.split()) <= 20 and not ex.endswith("?"):
                chosen_example = ex
                break

    if response_style == "Short Answer":
        body = " ".join(selected_sentences[:2])
        if chosen_example and len(body.split()) < 45:
            body += f"\n\nExample: {chosen_example}"
        return body

    elif response_style == "Detailed Explanation":
        exp = " ".join(selected_sentences[:4])
        ex_sec = f"\n\n### Example\n{chosen_example}" if chosen_example else ""
        return (
            f"### Explanation\n{exp}{ex_sec}\n\n"
            f"### Key Takeaway\nIn {lecture_title}, this foundational concept directly determines how programs evaluate expressions and store values."
        )

    else:  # Explain Clearly
        exp = " ".join(selected_sentences[:2])
        ex_sec = f"\n\n### Example\n{chosen_example}" if chosen_example else ""
        takeaway = f"In {lecture_title}, this concept provides the foundation for evaluating computations and managing state in Python programs."
        return (
            f"### Explanation\n{exp}{ex_sec}\n\n"
            f"### Key Takeaway\n{takeaway}"
        )
