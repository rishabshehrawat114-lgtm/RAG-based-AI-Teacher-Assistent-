# RAG Evaluation Framework & Benchmark Results

## Evaluation Methodology

The **RAG-Based AI Teaching Assistant** incorporates a systematic, reproducible quality evaluation framework (`src/evaluation/`) to quantitatively benchmark retrieval performance, relevance gate behavior, and answer grounding.

Evaluation is performed using the ground-truth benchmark dataset located at:
```text
data/evaluation/rag_benchmark.json
```
The benchmark comprises **20 carefully curated questions** spanning in-scope lecture concepts, cross-lecture inquiries, and deliberately out-of-scope/unsupported queries to test the relevance abstention gate.

---

## Benchmark Results Summary

- **Total Questions Evaluated:** 20
- **Relevance Cutoff Threshold:** `0.3200`
- **Top-K Retrieved Passages:** 5

### 1. Retrieval Performance Metrics

| Metric | Measured Score | Target | Description |
| :--- | :---: | :---: | :--- |
| **Recall@1** | **53.57%** | > 50% | Proportion of relevant chunks retrieved at rank 1 |
| **Recall@3** | **82.14%** | > 75% | Proportion of relevant chunks retrieved within top 3 |
| **Recall@5** | **89.29%** | > 80% | Proportion of relevant chunks retrieved within top 5 |
| **Hit@1** | **64.29%** | > 60% | Queries with at least one relevant passage at rank 1 |
| **Hit@3** | **92.86%** | > 85% | Queries with at least one relevant passage within top 3 |
| **Hit@5** | **100.00%** | 100% | Queries with at least one relevant passage within top 5 |
| **MRR** | **0.7679** | > 0.70 | Mean Reciprocal Rank across relevant queries |
| **Lecture Accuracy** | **100.00%** | 100% | Proportion of queries identifying the correct lecture |

### 2. Relevance Gate & Abstention Safety

| Metric | Measured Score | Target | Description |
| :--- | :---: | :---: | :--- |
| **Decision Accuracy** | **95.00%** | > 90% | Overall accuracy of binary relevance decisions |
| **False Acceptance (FAR)** | **0.00%** | < 5% | Out-of-scope questions incorrectly accepted |
| **False Rejection (FRR)** | **7.14%** | < 10% | Valid in-scope questions incorrectly rejected |
| **Acceptance Rate** | **65.00%** | N/A | Percentage of queries passing the gate (13/20) |
| **Relevant Queries (TP / FN)**| **13 / 1** | N/A | In-scope questions accepted vs. falsely rejected |
| **Out-of-Scope (TN / FP)** | **6 / 0** | N/A | Irrelevant questions rejected vs. falsely accepted |

### 3. Cosine Similarity Score Distribution

The cosine similarity distributions demonstrate strong separation between relevant and irrelevant queries:
- **Average Top Score (Relevant Queries):** `0.5370`
- **Average Top Score (Rejected / Out-of-Scope):** `0.1759`
- **Minimum Relevant Top Score:** `0.2970` (`eval_012`)
- **Maximum Irrelevant Top Score:** `0.2443`

### 4. Attribution & Answer Correctness

| Metric | Measured Score | Count | Description |
| :--- | :---: | :---: | :--- |
| **Citation Validity Rate** | **100.00%** | 49 / 49 | Cited chunk IDs correctly exist in the course corpus |
| **Deterministic Answer Check** | **95.00%** | 19 / 20 | Keyword/regex verification against expected answers |
| **LLM Judge** | *Unavailable* | — | Skipped during offline/deterministic benchmark runs |

---

## Detailed Failure Analysis: `eval_012`

Out of 20 benchmark questions, the evaluation framework identified exactly **1 failure**:

### Failure Case: `eval_012`
- **Question:** *"What department has she been lecturing in?"*
- **Ground Truth Status:** Relevant (in-scope)
- **Top Retrieved Similarity Score:** `0.2970`
- **Active Threshold:** `0.3200`
- **Relevance Gate Decision:** `NOT RELEVANT` (Abstained)
- **Failure Classification:** **False Rejection**

### Why Did This Happen?
1. **Vocabulary Divergence:** The student’s phrasing (*"What department has she been lecturing in?"*) is abstract and concise. In the spoken lecture, the instructor introduces herself conversationally using natural, informal classroom speech (*"I've been teaching computer science over in the engineering faculty for about seven years..."*).
2. **Dense Bi-Encoder Attenuation:** Because `all-MiniLM-L6-v2` encodes the query and the conversational transcript independently without cross-attention, the semantic similarity score peaked at `0.2970`, falling just short of the `0.3200` threshold.

### Why We Do NOT Lower the Threshold
Lowering the threshold to `0.2900` would flip `eval_012` from a False Rejection to a True Positive, boosting the benchmark score to 100%. **However, doing so would be bad engineering practice (overfitting to the test set):**
- The maximum similarity observed on an out-of-scope query was `0.2443`. Lowering the threshold to ~0.2900 cuts the safety margin in half, significantly increasing the probability of False Acceptances (hallucinations on unrelated queries) in real-world use.
- In educational RAG, a **False Rejection is safe** (the student is politely told the material wasn't found), whereas a **False Acceptance is hazardous** (the model invents an answer or attaches an unrelated lecture quote).
- Keeping `0.3200` preserves zero False Acceptances (`FAR = 0.00%`) while documenting `eval_012` as a known retrieval edge case to be addressed via future techniques like cross-encoder reranking or query expansion.

---

## Benchmark Limitations & Future Work

1. **Dataset Scale:** The benchmark contains 20 curated questions. While effective for regression testing and pipeline verification, institutional deployments should expand this to 100+ questions across diverse lecture styles.
2. **Deterministic Evaluation:** Ground-truth answer validation currently uses deterministic token matching. While fast and reproducible offline, complementing it with LLM-as-a-judge (e.g., G-Eval or Ragas) would provide richer semantic nuance when API keys are available.
3. **Retrieval Enhancements:** Addressing edge cases like `eval_012` without altering the threshold requires architectural retrieval improvements:
   - **Cross-Encoder Reranking:** Re-evaluating top-20 candidates with joint cross-attention.
   - **HyDE (Hypothetical Document Embeddings):** Expanding student queries into hypothetical lecture explanations before vector search.
