"""
scripts/diagnostic_10_questions.py
==================================
Diagnostic test evaluating 10 different question types against the indexed lecture corpus.
Compares behavior under the strict 0.25 similarity threshold vs. adaptive retrieval.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag import FALLBACK_NOT_FOUND_MESSAGE, RAGTeacherPipeline

TEST_QUESTIONS = [
    {
        "category": "Direct factual question",
        "question": "What field is machine learning stated to be a subset of?",
        "expected_in_lecture": True,
    },
    {
        "category": "Conceptual question",
        "question": "How is the relationship between AI and machine learning explained?",
        "expected_in_lecture": True,
    },
    {
        "category": "Paraphrased question",
        "question": "What subject matter will the class be focusing on today?",
        "expected_in_lecture": True,
    },
    {
        "category": "Broad question",
        "question": "Can you summarize what the teacher discusses?",
        "expected_in_lecture": True,
    },
    {
        "category": "Beginning of lecture",
        "question": "What topics are introduced at the beginning of the lecture?",
        "expected_in_lecture": True,
    },
    {
        "category": "Middle of lecture",
        "question": "What concept is explained following the opening announcement?",
        "expected_in_lecture": True,
    },
    {
        "category": "End of lecture",
        "question": "How does the speaker conclude their introductory explanation?",
        "expected_in_lecture": True,
    },
    {
        "category": "Follow-up question",
        "question": "Does it mention whether it is part of computer science or AI?",
        "expected_in_lecture": True,
        "history": [
            {"role": "user", "content": "Tell me about machine learning."},
            {"role": "assistant", "content": "Machine learning is discussed in the lecture."},
        ],
    },
    {
        "category": "Definitely exists",
        "question": "Will the class be studying machine learning today?",
        "expected_in_lecture": True,
    },
    {
        "category": "Definitely does NOT exist",
        "question": "What is the capital city of France?",
        "expected_in_lecture": False,
    },
]


def run_diagnostic() -> None:
    pipeline = RAGTeacherPipeline()
    retriever = pipeline.retriever

    print("=" * 80)
    print("MODULE 9 DIAGNOSTIC REPORT: 10 QUESTION RETRIEVAL EVALUATION")
    print("Corpus Chunks:", len(retriever.chunks_metadata))
    if retriever.chunks_metadata:
        print("Indexed Text:", f"\"{retriever.chunks_metadata[0]['text']}\"")
    print("=" * 80)

    results_table = []

    for idx, item in enumerate(TEST_QUESTIONS, start=1):
        q = item["question"]
        cat = item["category"]
        hist = item.get("history")

        # 1. Raw similarity without threshold
        raw_results = retriever.retrieve(q, top_k=1, similarity_threshold=None)
        if raw_results:
            top_chunk = raw_results[0]
            chunk_id = top_chunk["chunk_id"]
            sim_score = top_chunk["similarity_score"]
            source = top_chunk["source_file"]
            time_range = f"{top_chunk['start_time']}s - {top_chunk['end_time']}s"
        else:
            chunk_id = "None"
            sim_score = 0.0
            source = "None"
            time_range = "None"

        # 2. Test with 0.25 threshold
        results_with_025 = retriever.retrieve(q, top_k=1, similarity_threshold=0.25)
        rejected_by_025 = len(results_with_025) == 0

        # 3. Test end-to-end RAG answer with current threshold (0.25)
        ans_strict = pipeline.answer_question(q, conversation_history=hist, similarity_threshold=0.25)
        fallback_strict = ans_strict.is_fallback or (FALLBACK_NOT_FOUND_MESSAGE in ans_strict.answer)

        # 4. Test end-to-end RAG answer without 0.25 threshold
        ans_adaptive = pipeline.answer_question(q, conversation_history=hist, similarity_threshold=None)
        fallback_adaptive = ans_adaptive.is_fallback or (FALLBACK_NOT_FOUND_MESSAGE in ans_adaptive.answer)

        results_table.append({
            "idx": idx,
            "category": cat,
            "question": q,
            "chunk_id": chunk_id,
            "sim_score": sim_score,
            "source_time": f"{source} ({time_range})",
            "rejected_by_025": rejected_by_025,
            "fallback_with_025": fallback_strict,
            "fallback_without_threshold": fallback_adaptive,
            "expected_in_lecture": item["expected_in_lecture"],
        })

    # Print Detailed Per-Question Report
    for r in results_table:
        print(f"\n--- Question #{r['idx']} [{r['category']}] ---")
        print(f"Question: \"{r['question']}\"")
        print(f"Top retrieved chunk ID: {r['chunk_id']}")
        print(f"Similarity score: {r['sim_score']:.4f}")
        print(f"Retrieved source/timestamp: {r['source_time']}")
        print(f"Rejected by 0.25 threshold: {'YES (REJECTED)' if r['rejected_by_025'] else 'NO (PASSED)'}")
        print(f"Answer status with 0.25 threshold: {'FALLBACK (Information Not Found)' if r['fallback_with_025'] else 'GROUNDED ANSWER'}")
        print(f"Answer status without threshold: {'FALLBACK (Information Not Found)' if r['fallback_without_threshold'] else 'GROUNDED ANSWER'}")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON TABLE:")
    print(f"{'#':<3} | {'Score':<7} | {'0.25 Thresh?':<13} | {'Status (Thresh=0.25)':<25} | {'Category'}")
    print("-" * 80)
    for r in results_table:
        rej = "REJECTED" if r['rejected_by_025'] else "PASSED"
        status = "FALLBACK" if r['fallback_with_025'] else "ANSWERED"
        print(f"{r['idx']:<3} | {r['sim_score']:<7.4f} | {rej:<13} | {status:<25} | {r['category']}")
    print("=" * 80)


if __name__ == "__main__":
    run_diagnostic()
