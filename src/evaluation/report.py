"""
src/evaluation/report.py
========================
Report generation and serialization helpers for RAG benchmark evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.models import EvaluationSummary


def format_terminal_report(summary: EvaluationSummary) -> str:
    """
    Formats the evaluation summary into a clean, human-readable terminal report.

    Args:
        summary: Aggregated EvaluationSummary object.

    Returns:
        Formatted multi-line report string.
    """
    rm = summary.retrieval_metrics
    rel = summary.relevance_metrics
    sa = summary.score_analysis
    cm = summary.citation_metrics
    am = summary.answer_metrics

    avg_rel = f"{sa['avg_top_score_relevant']:.4f}" if sa.get("avg_top_score_relevant") is not None else "N/A"
    avg_rej = f"{sa['avg_top_score_rejected']:.4f}" if sa.get("avg_top_score_rejected") is not None else "N/A"
    min_rel = f"{sa['min_relevant_top_score']:.4f}" if sa.get("min_relevant_top_score") is not None else "N/A"
    max_irr = f"{sa['max_irrelevant_top_score']:.4f}" if sa.get("max_irrelevant_top_score") is not None else "N/A"

    lines = [
        "==================================================",
        "          RAG QUALITY EVALUATION REPORT          ",
        "==================================================",
        f"Timestamp:             {summary.timestamp}",
        f"Dataset:               {summary.dataset_path or 'Standard Benchmark'}",
        f"Questions Evaluated:   {summary.total_questions}",
        f"Relevance Threshold:   {summary.relevance_threshold:.4f}",
        "--------------------------------------------------",
        "RETRIEVAL QUALITY",
        f"  Recall@1:            {rm.get('recall_at_1', 0.0):.2%}",
        f"  Recall@3:            {rm.get('recall_at_3', 0.0):.2%}",
        f"  Recall@5:            {rm.get('recall_at_5', 0.0):.2%}",
        f"  Hit@1:               {rm.get('hit_at_1', 0.0):.2%}",
        f"  Hit@3:               {rm.get('hit_at_3', 0.0):.2%}",
        f"  Hit@5:               {rm.get('hit_at_5', 0.0):.2%}",
        f"  MRR (Mean Rec. Rank):{rm.get('mrr', 0.0):.4f}",
        f"  Lecture Accuracy:    {rm.get('lecture_retrieval_accuracy', 0.0):.2%}",
        "--------------------------------------------------",
        "RELEVANCE / ABSTENTION GATE",
        f"  Decision Accuracy:   {rel.get('decision_accuracy', 0.0):.2%}",
        f"  False Acceptance:    {rel.get('false_acceptance_rate', 0.0):.2%} (FAR)",
        f"  False Rejection:     {rel.get('false_rejection_rate', 0.0):.2%} (FRR)",
        f"  Acceptance Rate:     {rel.get('acceptance_rate', 0.0):.2%}",
        f"  Relevant (TP / FN):  {rel.get('true_positives', 0)} / {rel.get('false_negatives', 0)}",
        f"  Out-of-Scope (TN/FP):{rel.get('true_negatives', 0)} / {rel.get('false_positives', 0)}",
        "--------------------------------------------------",
        "SIMILARITY SCORE DISTRIBUTION",
        f"  Avg Top Score (Relevant):    {avg_rel}",
        f"  Avg Top Score (Rejected):    {avg_rej}",
        f"  Min Relevant Top Score:      {min_rel}",
        f"  Max Irrelevant Top Score:    {max_irr}",
        "--------------------------------------------------",
        "CITATIONS & PROVENANCE",
        f"  Citations Evaluated: {cm.get('total_citations_evaluated', 0)}",
        f"  Valid Citations:     {cm.get('valid_citations', 0)}",
        f"  Validity Rate:       {cm.get('citation_validity_rate', 0.0):.2%}",
        "--------------------------------------------------",
        "ANSWER EVALUATION",
        f"  Deterministic Check: {am.get('deterministic_pass_rate', 0.0):.2%} ({am.get('deterministic_passed_count', 0)}/{am.get('total_questions', 0)})",
        f"  LLM Judge:           {am.get('llm_judge_status', 'unavailable')} (offline deterministic safe)",
        "--------------------------------------------------",
        f"TOTAL FAILURES:        {len(summary.failures)}",
        "==================================================",
    ]

    if summary.failures:
        lines.append("FAILURE BREAKDOWN:")
        for idx, f in enumerate(summary.failures, start=1):
            reasons = "; ".join(f.get("reasons", []))
            lines.append(f"  {idx}. [{f.get('question_id')}] '{f.get('question')}' (Score: {f.get('top_score', 0.0):.4f})")
            lines.append(f"     -> {reasons}")
        lines.append("==================================================")

    return "\n".join(lines)


def generate_markdown_report(summary: EvaluationSummary) -> str:
    """
    Generates a comprehensive GitHub-flavored Markdown report.

    Args:
        summary: Aggregated EvaluationSummary object.

    Returns:
        Markdown document string.
    """
    rm = summary.retrieval_metrics
    rel = summary.relevance_metrics
    sa = summary.score_analysis
    cm = summary.citation_metrics
    am = summary.answer_metrics

    avg_rel = f"{sa['avg_top_score_relevant']:.4f}" if sa.get("avg_top_score_relevant") is not None else "N/A"
    avg_rej = f"{sa['avg_top_score_rejected']:.4f}" if sa.get("avg_top_score_rejected") is not None else "N/A"
    min_rel = f"{sa['min_relevant_top_score']:.4f}" if sa.get("min_relevant_top_score") is not None else "N/A"
    max_irr = f"{sa['max_irrelevant_top_score']:.4f}" if sa.get("max_irrelevant_top_score") is not None else "N/A"

    md = [
        "# 📊 RAG Quality & Retrieval Benchmark Report",
        "",
        f"- **Date & Time:** `{summary.timestamp}`",
        f"- **Dataset:** `{summary.dataset_path or 'Standard Course Benchmark'}`",
        f"- **Total Questions Evaluated:** `{summary.total_questions}`",
        f"- **Relevance Threshold:** `{summary.relevance_threshold:.4f}`",
        "",
        "## 1. High-Level Summary",
        "",
        "| Metric | Result | Description |",
        "| :--- | :--- | :--- |",
        f"| **Recall@1** | `{rm.get('recall_at_1', 0.0):.2%}` | Ground truth chunk in top 1 retrieved |",
        f"| **Recall@3** | `{rm.get('recall_at_3', 0.0):.2%}` | Ground truth chunk in top 3 retrieved |",
        f"| **Recall@5** | `{rm.get('recall_at_5', 0.0):.2%}` | Ground truth chunk in top 5 retrieved |",
        f"| **Hit@5** | `{rm.get('hit_at_5', 0.0):.2%}` | At least one relevant chunk in top 5 |",
        f"| **MRR** | `{rm.get('mrr', 0.0):.4f}` | Mean Reciprocal Rank (ranking quality) |",
        f"| **Lecture Accuracy** | `{rm.get('lecture_retrieval_accuracy', 0.0):.2%}` | Top chunk matches expected lecture ID |",
        f"| **Decision Accuracy** | `{rel.get('decision_accuracy', 0.0):.2%}` | Correct accept / reject decisions |",
        f"| **False Acceptance Rate** | `{rel.get('false_acceptance_rate', 0.0):.2%}` | Out-of-scope queries falsely accepted |",
        f"| **False Rejection Rate** | `{rel.get('false_rejection_rate', 0.0):.2%}` | Relevant queries falsely rejected |",
        f"| **Citation Validity Rate** | `{cm.get('citation_validity_rate', 0.0):.2%}` | Citations verified against indexed data |",
        f"| **Deterministic Check** | `{am.get('deterministic_pass_rate', 0.0):.2%}` | Answers passing all deterministic constraints |",
        "",
        "## 2. Similarity Score Distribution",
        "",
        f"- **Average Top Score (Relevant):** `{avg_rel}`",
        f"- **Average Top Score (Rejected):** `{avg_rej}`",
        f"- **Minimum Relevant Top Score:** `{min_rel}`",
        f"- **Maximum Irrelevant Top Score:** `{max_irr}`",
        "",
        "## 3. Question Categories Evaluated",
        "",
        "| Category | Count |",
        "| :--- | :--- |",
    ]

    for cat, count in summary.questions_by_type.items():
        md.append(f"| `{cat}` | {count} |")

    md.extend([
        "",
        "## 4. Failure Analysis",
        "",
    ])

    if summary.failures:
        md.append("| Question ID | Question | Top Score | Failure Reasons |")
        md.append("| :--- | :--- | :--- | :--- |")
        for f in summary.failures:
            qid = f.get("question_id")
            q_txt = f.get("question")
            sc = f.get("top_score", 0.0)
            reasons = "<br>".join(f.get("reasons", []))
            md.append(f"| `{qid}` | {q_txt} | `{sc:.4f}` | {reasons} |")
    else:
        md.append("✅ **Zero failures detected across all benchmark queries!**")

    md.append("")
    return "\n".join(md)


def save_evaluation_results(
    summary: EvaluationSummary,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """
    Saves latest evaluation results as JSON and Markdown reports.

    Args:
        summary: Aggregated EvaluationSummary object.
        output_dir: Target directory (e.g. data/evaluation/results).

    Returns:
        Tuple of (json_path, markdown_path).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "latest_evaluation.json"
    md_path = out / "latest_evaluation_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown_report(summary))

    return json_path, md_path
