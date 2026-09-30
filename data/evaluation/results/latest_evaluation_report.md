# 📊 RAG Quality & Retrieval Benchmark Report

- **Date & Time:** `2026-09-30T08:41:52.642918+00:00`
- **Dataset:** `C:\Users\risha\.gemini\antigravity-ide\scratch\rag-ai-teacher\data\evaluation\rag_benchmark.json`
- **Total Questions Evaluated:** `20`
- **Relevance Threshold:** `0.3200`

## 1. High-Level Summary

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Recall@1** | `53.57%` | Ground truth chunk in top 1 retrieved |
| **Recall@3** | `82.14%` | Ground truth chunk in top 3 retrieved |
| **Recall@5** | `89.29%` | Ground truth chunk in top 5 retrieved |
| **Hit@5** | `100.00%` | At least one relevant chunk in top 5 |
| **MRR** | `0.7679` | Mean Reciprocal Rank (ranking quality) |
| **Lecture Accuracy** | `100.00%` | Top chunk matches expected lecture ID |
| **Decision Accuracy** | `95.00%` | Correct accept / reject decisions |
| **False Acceptance Rate** | `0.00%` | Out-of-scope queries falsely accepted |
| **False Rejection Rate** | `7.14%` | Relevant queries falsely rejected |
| **Citation Validity Rate** | `100.00%` | Citations verified against indexed data |
| **Deterministic Check** | `95.00%` | Answers passing all deterministic constraints |

## 2. Similarity Score Distribution

- **Average Top Score (Relevant):** `0.5370`
- **Average Top Score (Rejected):** `0.1759`
- **Minimum Relevant Top Score:** `0.2970`
- **Maximum Irrelevant Top Score:** `0.2443`

## 3. Question Categories Evaluated

| Category | Count |
| :--- | :--- |
| `direct_lecture` | 5 |
| `concept` | 4 |
| `cross_lecture` | 1 |
| `follow_up` | 2 |
| `lecture_specific` | 2 |
| `out_of_scope` | 5 |
| `ambiguous` | 1 |

## 4. Failure Analysis

| Question ID | Question | Top Score | Failure Reasons |
| :--- | :--- | :--- | :--- |
| `eval_012` | What department has she been lecturing in? | `0.2970` | False Rejection: question is relevant, but top score 0.2970 < 0.3200 threshold. |
