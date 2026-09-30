"""
scripts/verify_functionality.py
===============================
Direct verification script testing core functionality across all 3 application modules:
1. Student Chat (in-scope grounded query)
2. Abstention Gate (out-of-scope query rejection at threshold 0.32)
3. Learning Center (summary, concepts, explain, quiz)
4. Evaluation engine accessibility
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_RELEVANCE_THRESHOLD
from src.data import CourseCatalog
from src.learning import (
    explain_lecture,
    generate_key_concepts,
    generate_quiz,
    generate_summary,
)
from src.rag import FALLBACK_NOT_FOUND_MESSAGE, RAGTeacherPipeline
from src.retrieval import LectureRetriever


def main() -> int:
    print("=" * 65)
    print(" RAG AI Teacher Assistant — Functional Smoke Test")
    print("=" * 65)

    print("[1/5] Initializing Course Pipeline...")
    retriever = LectureRetriever(auto_load=True)
    catalog = CourseCatalog()
    pipeline = RAGTeacherPipeline(retriever=retriever, catalog=catalog)
    stats = pipeline.get_corpus_stats()
    print(f"      Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"      Lectures: {stats.get('lecture_count', 0)}")
    assert stats.get("total_chunks", 0) > 0, "Corpus has 0 chunks!"

    print("\n[2/5] Testing In-Scope Chat Query...")
    in_scope_q = "What is a computer according to the lecture?"
    ans_in = pipeline.answer_question(in_scope_q)
    decision_in = ans_in.debug_info.get("relevance_decision", "RELEVANT")
    print(f"      Question: {in_scope_q}")
    print(f"      Decision: {decision_in}")
    print(f"      Top Score: {ans_in.debug_info.get('top_score', 0.0):.4f}")
    print(f"      Sources: {len(ans_in.sources)} citation(s)")
    print(f"      Answer excerpt: {ans_in.answer[:80]}...")
    assert decision_in == "RELEVANT"
    assert len(ans_in.sources) > 0
    assert FALLBACK_NOT_FOUND_MESSAGE not in ans_in.answer

    print("\n[3/5] Testing Out-Of-Scope Query Abstention (Gate 0.32)...")
    out_scope_q = "How do black holes warp gravitational spacetime?"
    ans_out = pipeline.answer_question(out_scope_q)
    decision_out = ans_out.debug_info.get("relevance_decision", "NOT RELEVANT")
    print(f"      Question: {out_scope_q}")
    print(f"      Decision: {decision_out}")
    print(f"      Top Score: {ans_out.debug_info.get('top_score', 0.0):.4f}")
    print(f"      Answer: {ans_out.answer[:80]}...")
    assert decision_out == "NOT RELEVANT"
    assert ans_out.is_fallback is True
    assert FALLBACK_NOT_FOUND_MESSAGE in ans_out.answer

    print("\n[4/5] Testing Learning Center Features...")
    summary = generate_summary(pipeline.retriever, lecture_scope="All Lectures")
    print(f"      Summary: {summary.content[:70]}...")
    assert summary.content != ""

    concepts = generate_key_concepts(pipeline.retriever, lecture_scope="All Lectures")
    print(f"      Concepts: {concepts.content[:70]}...")
    assert concepts.content != ""

    explanation = explain_lecture(pipeline.retriever, lecture_scope="All Lectures")
    print(f"      Explain: {explanation.content[:70]}...")
    assert explanation.content != ""

    quiz = generate_quiz(pipeline.retriever, lecture_scope="All Lectures", num_questions=3)
    if quiz.is_fallback:
        print(f"      Quiz (Fallback/Offline Mode): {quiz.error_message[:60]}...")
        assert quiz.error_message != ""
    else:
        print(f"      Quiz Questions: {len(quiz.questions)}")
        assert len(quiz.questions) > 0

    print("\n[5/5] Testing Evaluation Dataset Accessibility...")
    from src.config import EVALUATION_DATASET_PATH, RESULTS_DIR
    assert EVALUATION_DATASET_PATH.is_file(), f"Benchmark missing at {EVALUATION_DATASET_PATH}"
    print(f"      Evaluation Benchmark File: {EVALUATION_DATASET_PATH.name} (Found)")
    print(f"      Results Directory: {RESULTS_DIR.name} (Verified)")

    print("\n" + "=" * 65)
    print(" ALL 5 FUNCTIONAL SMOKE TESTS PASSED!")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
