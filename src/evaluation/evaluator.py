"""
src/evaluation/evaluator.py
===========================
RAG Evaluation Engine: executes reproducible benchmark evaluation across the live RAG pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Sequence

from src.config import DEFAULT_RELEVANCE_THRESHOLD, DEFAULT_TOP_K
from src.evaluation.citation_metrics import evaluate_answer_citations
from src.evaluation.models import (
    BenchmarkQuestion,
    EvaluationSummary,
    QuestionEvaluationResult,
)
from src.evaluation.relevance_metrics import (
    calculate_relevance_metrics,
    calculate_similarity_score_analysis,
)
from src.evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_lecture_accuracy,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from src.rag import (
    FALLBACK_NOT_FOUND_MESSAGE,
    RAGAnswer,
    RAGTeacherPipeline,
    build_retrieval_query,
    is_follow_up_query,
    is_relevant,
)
from src.retrieval import LectureRetriever, RetrievalError

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluator engine for benchmarking retrieval, relevance gate, citations, and grounded answers.
    """

    def __init__(
        self,
        pipeline: RAGTeacherPipeline | None = None,
        retriever: LectureRetriever | None = None,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """
        Initializes the RAGEvaluator.

        Args:
            pipeline: Optional RAGTeacherPipeline instance. If None, initialized from retriever.
            retriever: Optional LectureRetriever instance.
            relevance_threshold: Relevance threshold cutoff (default: 0.32).
            top_k: Default number of chunks to retrieve (default: 5).
        """
        if pipeline is not None:
            self.pipeline = pipeline
            self.retriever = pipeline.retriever
        elif retriever is not None:
            self.retriever = retriever
            self.pipeline = RAGTeacherPipeline(retriever=self.retriever)
        else:
            self.pipeline = RAGTeacherPipeline()
            self.retriever = self.pipeline.retriever

        self.relevance_threshold = float(relevance_threshold)
        self.top_k = int(top_k)

    def _get_known_lectures(self) -> set[str]:
        """Collects all known lecture IDs and titles from catalog and retriever metadata."""
        known: set[str] = set()
        if hasattr(self.pipeline, "catalog") and self.pipeline.catalog:
            catalog_lectures = self.pipeline.catalog.get_all_lectures()
            for info in catalog_lectures:
                lid = str(info.get("lecture_id", "")).strip().lower()
                if lid:
                    known.add(lid)
                ltitle = str(info.get("lecture_title", "")).strip().lower()
                if ltitle:
                    known.add(ltitle)
                src = str(info.get("source_filename", "")).strip().lower()
                if src:
                    known.add(src)

        for c in getattr(self.retriever, "chunks_metadata", []):
            cid = str(c.get("lecture_id", "")).strip().lower()
            if cid:
                known.add(cid)
            ctitle = str(c.get("lecture_title", "")).strip().lower()
            if ctitle:
                known.add(ctitle)
            sfile = str(c.get("source_filename") or c.get("source_file") or "").strip().lower()
            if sfile:
                known.add(sfile)

        return known

    def evaluate_question(
        self,
        question: BenchmarkQuestion,
        evaluate_answer_generation: bool = True,
    ) -> QuestionEvaluationResult:
        """
        Evaluates a single benchmark question through the RAG pipeline.

        Args:
            question: BenchmarkQuestion instance.
            evaluate_answer_generation: If True, invokes pipeline.answer_question() to test citations.

        Returns:
            QuestionEvaluationResult detailing performance on this question.
        """
        failures: list[str] = []

        # 1. Resolve effective query (handling conversation history)
        effective_query = question.question
        if question.conversation_history:
            eff_q, _ = build_retrieval_query(
                question.question,
                question.conversation_history,
                current_scope=question.lecture_scope,
            )
            effective_query = eff_q

        # 2. Scope resolution
        scope_filter = (
            None
            if question.lecture_scope in ("All Lectures", "All", "all", "")
            else question.lecture_scope
        )

        # 3. Retrieve chunks using existing retriever
        try:
            raw_chunks = self.retriever.retrieve(
                query=effective_query,
                top_k=self.top_k,
                similarity_threshold=None,  # keep all top_k to analyze scores
                lecture_scope=scope_filter,
                lecture_filter=scope_filter,
            )
        except RetrievalError as exc:
            logger.warning("Retrieval failed for question %s: %s", question.id, exc)
            raw_chunks = []

        retrieved_chunk_ids = [str(c.get("chunk_id", "")) for c in raw_chunks if c.get("chunk_id")]
        retrieved_lecture_ids = [
            str(c.get("lecture_id") or Path(str(c.get("source_filename") or c.get("source_file") or "")).stem)
            for c in raw_chunks
        ]
        top_k_scores = [float(c.get("similarity_score", 0.0)) for c in raw_chunks]
        top_score = top_k_scores[0] if top_k_scores else 0.0

        # 4. Relevance gate evaluation
        actual_decision = is_relevant(raw_chunks, threshold=self.relevance_threshold)
        expected_decision = question.should_answer
        decision_match = (actual_decision == expected_decision)

        if not decision_match:
            if expected_decision and not actual_decision:
                failures.append(
                    f"False Rejection: question is relevant, but top score {top_score:.4f} < {self.relevance_threshold:.4f} threshold."
                )
            elif not expected_decision and actual_decision:
                failures.append(
                    f"False Acceptance: question is out-of-scope, but top score {top_score:.4f} >= {self.relevance_threshold:.4f} threshold."
                )

        # 5. Retrieval ranking metrics
        recall_at_1 = calculate_recall_at_k(retrieved_chunk_ids, question.expected_relevant_chunk_ids, k=1)
        recall_at_3 = calculate_recall_at_k(retrieved_chunk_ids, question.expected_relevant_chunk_ids, k=3)
        recall_at_5 = calculate_recall_at_k(retrieved_chunk_ids, question.expected_relevant_chunk_ids, k=5)

        hit_at_1 = calculate_hit_at_k(retrieved_chunk_ids, question.expected_relevant_chunk_ids, k=1)
        hit_at_3 = calculate_hit_at_k(retrieved_chunk_ids, question.expected_relevant_chunk_ids, k=3)
        hit_at_5 = calculate_hit_at_k(retrieved_chunk_ids, question.expected_relevant_chunk_ids, k=5)

        reciprocal_rank = calculate_reciprocal_rank(retrieved_chunk_ids, question.expected_relevant_chunk_ids)
        lecture_match = calculate_lecture_accuracy(retrieved_lecture_ids, question.expected_lecture_id)

        if question.should_answer and question.expected_relevant_chunk_ids:
            if not hit_at_5:
                failures.append(
                    f"Retrieval Miss: expected chunks {question.expected_relevant_chunk_ids} not found in top-{self.top_k} results."
                )
            if lecture_match is False:
                failures.append(
                    f"Lecture Mismatch: expected lecture '{question.expected_lecture_id}', top retrieved was '{retrieved_lecture_ids[0] if retrieved_lecture_ids else 'None'}'."
                )

        # 6. End-to-end answer generation and citation verification
        answer_available = False
        is_abstention = False
        citations_evaluated = 0
        valid_citations = 0
        citation_valid: bool | None = None
        deterministic_ok = True

        if evaluate_answer_generation:
            try:
                rag_ans: RAGAnswer = self.pipeline.answer_question(
                    question=question.question,
                    conversation_history=question.conversation_history,
                    top_k=self.top_k,
                    similarity_threshold=self.relevance_threshold,
                    lecture_scope=scope_filter,
                    source_filter=scope_filter,
                )
                answer_available = bool(rag_ans.answer and rag_ans.answer.strip())
                is_abstention = rag_ans.is_fallback

                # Deterministic check: did system abstain when expected?
                if not expected_decision:
                    if not is_abstention or FALLBACK_NOT_FOUND_MESSAGE not in rag_ans.answer:
                        deterministic_ok = False
                        failures.append("Answer check failed: system did not output standard out-of-scope abstention.")
                else:
                    if is_abstention:
                        deterministic_ok = False
                    elif not answer_available:
                        deterministic_ok = False
                        failures.append("Answer check failed: accepted question produced empty answer text.")

                # Citation check
                known_lectures = self._get_known_lectures()
                ret_set = set(retrieved_chunk_ids)
                total_c, valid_c, all_c_valid, cit_errors = evaluate_answer_citations(
                    citations=rag_ans.sources,
                    known_lectures=known_lectures,
                    retrieved_chunk_ids=ret_set,
                    is_abstention=is_abstention,
                )
                citations_evaluated = total_c
                valid_citations = valid_c
                citation_valid = all_c_valid
                if cit_errors:
                    failures.extend(cit_errors)
                    deterministic_ok = False

            except Exception as exc:
                logger.error("Answer generation failed during evaluation of %s: %s", question.id, exc)
                deterministic_ok = False
                failures.append(f"Answer execution error: {exc}")

        return QuestionEvaluationResult(
            question_id=question.id,
            question=question.question,
            question_type=question.question_type,
            expected_decision=expected_decision,
            actual_decision=actual_decision,
            decision_match=decision_match,
            top_score=top_score,
            top_k_scores=top_k_scores,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_lecture_ids=retrieved_lecture_ids,
            expected_chunk_ids=question.expected_relevant_chunk_ids,
            expected_lecture_id=question.expected_lecture_id,
            recall_at_1=recall_at_1,
            recall_at_3=recall_at_3,
            recall_at_5=recall_at_5,
            hit_at_1=hit_at_1,
            hit_at_3=hit_at_3,
            hit_at_5=hit_at_5,
            reciprocal_rank=reciprocal_rank,
            lecture_match=lecture_match,
            citations_evaluated=citations_evaluated,
            valid_citations=valid_citations,
            citation_valid=citation_valid,
            answer_available=answer_available,
            is_abstention=is_abstention,
            deterministic_checks_passed=(deterministic_ok and len(failures) == 0),
            failure_reasons=failures,
            debug_info={
                "effective_query": effective_query,
                "scope_filter": scope_filter,
                "threshold": self.relevance_threshold,
            },
        )

    def evaluate_dataset(
        self,
        questions: Sequence[BenchmarkQuestion],
        dataset_path: str = "",
        evaluate_answers: bool = True,
    ) -> EvaluationSummary:
        """
        Runs the full evaluation benchmark across all questions in the dataset.

        Args:
            questions: List of BenchmarkQuestion instances.
            dataset_path: Source path of benchmark dataset for reporting.
            evaluate_answers: Whether to run end-to-end answer synthesis and citation checks.

        Returns:
            Aggregated EvaluationSummary with detailed metrics.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        total_q = len(questions)

        if total_q == 0:
            return EvaluationSummary(
                timestamp=timestamp,
                dataset_path=dataset_path,
                relevance_threshold=self.relevance_threshold,
                total_questions=0,
                questions_by_type={},
                retrieval_metrics={
                    "recall_at_1": 0.0,
                    "recall_at_3": 0.0,
                    "recall_at_5": 0.0,
                    "hit_at_1": 0.0,
                    "hit_at_3": 0.0,
                    "hit_at_5": 0.0,
                    "mrr": 0.0,
                    "lecture_retrieval_accuracy": 0.0,
                },
                relevance_metrics=calculate_relevance_metrics([]),
                score_analysis=calculate_similarity_score_analysis([]),
                citation_metrics={
                    "total_citations_evaluated": 0,
                    "valid_citations": 0,
                    "citation_validity_rate": 0.0,
                },
                answer_metrics={
                    "deterministic_pass_rate": 0.0,
                    "llm_judge_status": "unavailable",
                },
                question_results=[],
                failures=[],
            )

        question_results: list[QuestionEvaluationResult] = []
        questions_by_type: dict[str, int] = {}
        failures_list: list[dict[str, Any]] = []

        for q in questions:
            q_type = q.question_type
            questions_by_type[q_type] = questions_by_type.get(q_type, 0) + 1

            res = self.evaluate_question(q, evaluate_answer_generation=evaluate_answers)
            question_results.append(res)

            if res.failure_reasons:
                failures_list.append({
                    "question_id": res.question_id,
                    "question": res.question,
                    "question_type": res.question_type,
                    "top_score": res.top_score,
                    "reasons": res.failure_reasons,
                })

        # Calculate average retrieval metrics over questions that expected answers
        relevant_results = [r for r in question_results if r.expected_decision and r.expected_chunk_ids]
        n_rel = len(relevant_results)

        avg_recall_1 = float(sum(r.recall_at_1 for r in relevant_results) / n_rel) if n_rel > 0 else 0.0
        avg_recall_3 = float(sum(r.recall_at_3 for r in relevant_results) / n_rel) if n_rel > 0 else 0.0
        avg_recall_5 = float(sum(r.recall_at_5 for r in relevant_results) / n_rel) if n_rel > 0 else 0.0

        avg_hit_1 = float(sum(1.0 for r in relevant_results if r.hit_at_1) / n_rel) if n_rel > 0 else 0.0
        avg_hit_3 = float(sum(1.0 for r in relevant_results if r.hit_at_3) / n_rel) if n_rel > 0 else 0.0
        avg_hit_5 = float(sum(1.0 for r in relevant_results if r.hit_at_5) / n_rel) if n_rel > 0 else 0.0

        avg_mrr = float(sum(r.reciprocal_rank for r in relevant_results) / n_rel) if n_rel > 0 else 0.0

        lecture_evals = [r for r in question_results if r.lecture_match is not None]
        n_lec = len(lecture_evals)
        lec_acc = float(sum(1.0 for r in lecture_evals if r.lecture_match) / n_lec) if n_lec > 0 else 0.0

        retrieval_summary = {
            "recall_at_1": avg_recall_1,
            "recall_at_3": avg_recall_3,
            "recall_at_5": avg_recall_5,
            "hit_at_1": avg_hit_1,
            "hit_at_3": avg_hit_3,
            "hit_at_5": avg_hit_5,
            "mrr": avg_mrr,
            "lecture_retrieval_accuracy": lec_acc,
        }

        # Relevance gate metrics
        relevance_summary = calculate_relevance_metrics(question_results)
        score_analysis = calculate_similarity_score_analysis(question_results)

        # Citation metrics
        total_cits = sum(r.citations_evaluated for r in question_results)
        valid_cits = sum(r.valid_citations for r in question_results)
        cit_rate = float(valid_cits / total_cits) if total_cits > 0 else (1.0 if any(r.actual_decision for r in question_results) else 0.0)

        citation_summary = {
            "total_citations_evaluated": total_cits,
            "valid_citations": valid_cits,
            "citation_validity_rate": cit_rate,
        }

        # Answer metrics
        det_passed = sum(1 for r in question_results if r.deterministic_checks_passed)
        det_rate = float(det_passed / total_q) if total_q > 0 else 0.0

        answer_summary = {
            "deterministic_pass_rate": det_rate,
            "deterministic_passed_count": det_passed,
            "total_questions": total_q,
            "llm_judge_status": "unavailable",  # Explicitly marked as unavailable (not fabricated)
        }

        return EvaluationSummary(
            timestamp=timestamp,
            dataset_path=dataset_path,
            relevance_threshold=self.relevance_threshold,
            total_questions=total_q,
            questions_by_type=questions_by_type,
            retrieval_metrics=retrieval_summary,
            relevance_metrics=relevance_summary,
            score_analysis=score_analysis,
            citation_metrics=citation_summary,
            answer_metrics=answer_summary,
            question_results=question_results,
            failures=failures_list,
        )
