# Retrieval leaderboard

66 questions, 333 chunks, k=10. Embeddings: `BAAI/bge-small-en-v1.5`. Reranker: `BAAI/bge-reranker-v2-m3`. Ranked by MRR, then recall@10, then hit@3. 95% bootstrap confidence intervals (1000 resamples, seed 42). mrr_vs_top is each row's MRR minus the top row's, bootstrapped over per-question differences; "(tie)" marks an interval that contains 0.

| rank | pattern | hit@3 | hit@10 | mrr | recall@10 | mrr_vs_top | p50_ms |
|---|---|---|---|---|---|---|---|
| 1 | 03+04_hybrid_rerank | 0.879 [0.788, 0.955] | 0.939 [0.879, 0.985] | 0.802 [0.721, 0.878] | 0.909 [0.841, 0.962] | -- | 1642.0 |
| 2 | 04_bm25_rerank | 0.848 [0.758, 0.939] | 0.894 [0.818, 0.955] | 0.780 [0.684, 0.864] | 0.856 [0.773, 0.932] | -0.023 [-0.058, +0.000] (tie) | 1608.9 |
| 3 | 02_bm25 | 0.742 [0.636, 0.848] | 0.848 [0.758, 0.924] | 0.669 [0.568, 0.769] | 0.826 [0.735, 0.909] | -0.133 [-0.215, -0.055] | 1.3 |
| 4 | 03_hybrid_rrf | 0.652 [0.530, 0.773] | 0.955 [0.894, 1.000] | 0.591 [0.499, 0.686] | 0.924 [0.871, 0.977] | -0.211 [-0.299, -0.126] | 11.3 |
| 5 | 01_naive_dense | 0.621 [0.500, 0.727] | 0.833 [0.742, 0.924] | 0.531 [0.435, 0.625] | 0.788 [0.697, 0.871] | -0.271 [-0.376, -0.169] | 8.6 |

## By question category

MRR / hit@10 for each question category; the header gives the number of questions.

| pattern | keyword (24) | paraphrase (24) | multi_hop (9) | filter (9) |
|---|---|---|---|---|
| 03+04_hybrid_rerank | 0.946 / 1.000 | 0.747 / 0.917 | 0.630 / 0.889 | 0.741 / 0.889 |
| 04_bm25_rerank | 0.946 / 1.000 | 0.698 / 0.833 | 0.611 / 0.778 | 0.722 / 0.889 |
| 02_bm25 | 0.852 / 1.000 | 0.607 / 0.750 | 0.389 / 0.667 | 0.625 / 0.889 |
| 03_hybrid_rrf | 0.764 / 1.000 | 0.479 / 0.958 | 0.403 / 0.889 | 0.620 / 0.889 |
| 01_naive_dense | 0.693 / 0.958 | 0.440 / 0.792 | 0.485 / 0.889 | 0.389 / 0.556 |
