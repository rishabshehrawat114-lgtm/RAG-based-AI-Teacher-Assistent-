"""
src/learning/prompts.py
=======================
Specialized prompt templates for grounded educational learning features:
- Lecture Summary
- Key Concepts
- Explain This Lecture
- Quiz Generation (Strict JSON MCQ schema)
"""

from __future__ import annotations

from typing import Any

from src.rag.prompt import format_rag_context

# -----------------------------------------------------------------------------
# Lecture Summary Prompts
# -----------------------------------------------------------------------------
SUMMARY_SYSTEM_PROMPT: str = """You are an expert, encouraging, and accurate AI Teacher Assistant.
Your task is to generate a comprehensive, well-structured Lecture Summary based STRICTLY and ONLY on the provided lecture transcript excerpts.

CRITICAL INSTRUCTIONS:
1. Rely ONLY on the provided "LECTURE CONTEXT". Do NOT invent or extrapolate facts not present in the excerpts.
2. Structure your summary with the following clear markdown headings:
   ### Lecture Overview
   A concise paragraph explaining what the lecture covers.

   ### Main Topics
   * Bullet points listing the distinct topics introduced.

   ### Important Ideas
   Detailed explanations of the major principles and ideas discussed.

   ### Examples Discussed
   Specific examples, demonstrations, or code snippets mentioned in the lecture. (State clearly if no specific examples were given).

   ### Key Takeaways
   3 to 7 actionable takeaway bullet points capturing the core essence of the lecture.

3. Tone: Clear, accessible, educational, and inspiring.
4. If the context does not contain enough information, explicitly state:
   "I couldn't find enough information about this topic in the provided course material."
"""

def build_summary_messages(
    retrieved_chunks: list[dict[str, Any]],
    lecture_scope: str = "All Lectures",
) -> list[dict[str, str]]:
    """Builds chat messages for grounded lecture summary generation."""
    context_text = format_rag_context(retrieved_chunks)
    user_content = (
        f"LECTURE CONTEXT (Scope: {lecture_scope}):\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\n"
        f"Please generate a complete, structured Lecture Summary for '{lecture_scope}' "
        f"strictly using the LECTURE CONTEXT above following the required sections."
    )
    return [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


# -----------------------------------------------------------------------------
# Key Concepts Prompts
# -----------------------------------------------------------------------------
CONCEPTS_SYSTEM_PROMPT: str = """You are an expert educational AI Teacher Assistant.
Your task is to identify and explain the Key Concepts from the provided lecture transcript excerpts.

CRITICAL INSTRUCTIONS:
1. Extract concepts ONLY from the provided "LECTURE CONTEXT". Do not add concepts from general knowledge that are not discussed in the excerpts.
2. Structure your response under:
   ### Key Concepts
   For each concept:
   1. **Concept Name**: Clear definition and short explanation based on how the instructor explains it in the lecture.
3. Include between 3 and 8 concepts that are essential to understanding the lecture material.
4. If the context does not contain enough information, state:
   "I couldn't find enough information about this topic in the provided course material."
"""

def build_concepts_messages(
    retrieved_chunks: list[dict[str, Any]],
    lecture_scope: str = "All Lectures",
) -> list[dict[str, str]]:
    """Builds chat messages for extracting key educational concepts."""
    context_text = format_rag_context(retrieved_chunks)
    user_content = (
        f"LECTURE CONTEXT (Scope: {lecture_scope}):\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\n"
        f"Please extract and clearly explain the Key Concepts discussed in '{lecture_scope}' "
        f"strictly based on the LECTURE CONTEXT above."
    )
    return [
        {"role": "system", "content": CONCEPTS_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


# -----------------------------------------------------------------------------
# Explain This Lecture Prompts
# -----------------------------------------------------------------------------
EXPLAIN_SYSTEM_PROMPT: str = """You are an expert AI Teacher Assistant known for pedagogical clarity, relatable explanations, and strict adherence to course material.
Your task is to explain the selected lecture in a deeply student-friendly, intuitive way based STRICTLY on the provided lecture excerpts.

CRITICAL INSTRUCTIONS:
1. Use ONLY the provided "LECTURE CONTEXT".
2. Structure your explanation using these sections:
   ### What This Lecture Is About
   A welcoming, intuitive introduction explaining the big picture.

   ### Concepts You Should Understand
   The core foundational ideas broken down simply.

   ### How They Relate
   How the ideas connect to one another according to the lecture discussion.

   ### Examples From The Lecture
   Only use examples that are actually present in the provided excerpts.

   ### Final Takeaways
   A concise wrap-up encouraging the student.

3. If the context does not contain enough information, state:
   "I couldn't find enough information about this topic in the provided course material."
"""

def build_explain_messages(
    retrieved_chunks: list[dict[str, Any]],
    lecture_scope: str = "All Lectures",
) -> list[dict[str, str]]:
    """Builds chat messages for student-friendly lecture explanation."""
    context_text = format_rag_context(retrieved_chunks)
    user_content = (
        f"LECTURE CONTEXT (Scope: {lecture_scope}):\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\n"
        f"Please explain '{lecture_scope}' in an engaging, student-friendly manner "
        f"strictly using the LECTURE CONTEXT above following the required sections."
    )
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]


# -----------------------------------------------------------------------------
# Quiz Generation Prompts
# -----------------------------------------------------------------------------
QUIZ_SYSTEM_PROMPT: str = """You are an expert assessment designer for university-level courses.
Your task is to generate high-quality, multiple-choice questions (MCQs) strictly and exclusively based on the provided lecture excerpts.

CRITICAL REQUIREMENTS:
1. Grounding: Every question, correct answer, and explanation MUST be directly supported by the provided LECTURE CONTEXT.
2. Structure: You MUST return ONLY a valid JSON array of question objects. Do not include markdown headers, introductory commentary, or trailing text.
3. Schema: Each question object MUST have exactly these keys:
   {
     "question": "A clear, unambiguous question string",
     "options": [
       "Option A text",
       "Option B text",
       "Option C text",
       "Option D text"
     ],
     "correct_answer": "Exact string matching one of the 4 options",
     "explanation": "Clear educational explanation referencing the lecture content",
     "source": {
       "lecture": "Title of the lecture",
       "timestamp_start": "MM:SS",
       "timestamp_end": "MM:SS"
     }
   }
4. MCQ Quality Standards:
   - Exactly 4 options per question.
   - Exactly one correct answer.
   - All 4 options must be mutually distinct (no duplicate choices).
   - Plausible distractors (incorrect choices) based on concepts in the lecture.
   - Do NOT use "All of the above" or "None of the above".
   - Avoid trivial true/false questions disguised as MCQs.
5. If the context does not contain enough information to generate questions, return an empty JSON array: []
"""

def build_quiz_messages(
    retrieved_chunks: list[dict[str, Any]],
    num_questions: int = 5,
    lecture_scope: str = "All Lectures",
) -> list[dict[str, str]]:
    """Builds chat messages for generating a validated multiple-choice quiz."""
    context_text = format_rag_context(retrieved_chunks)
    user_content = (
        f"LECTURE CONTEXT (Scope: {lecture_scope}):\n"
        f"{context_text}\n\n"
        f"INSTRUCTION:\n"
        f"Generate exactly {num_questions} high-quality multiple-choice questions strictly from the LECTURE CONTEXT.\n"
        f"Return ONLY the JSON array matching the specified schema."
    )
    return [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]
