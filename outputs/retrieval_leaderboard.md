# Retrieval leaderboard

66 questions, 333 chunks, k=10. Embeddings: `BAAI/bge-small-en-v1.5`. Reranker: `BAAI/bge-reranker-v2-m3`. Ranked by MRR, then recall@10, then hit@3. 95% bootstrap confidence intervals (1000 resamples, seed 42). mrr_vs_top is each row's MRR minus the top row's, bootstrapped over per-question differences; "(tie)" marks an interval that contains 0.

| rank | pattern | hit@3 | hit@10 | mrr | recall@10 | mrr_vs_top | p50_ms |
|---|---|---|---|---|---|---|---|
| 1 | 03+04_hybrid_rerank | 0.894 [0.818, 0.955] | 0.955 [0.894, 1.000] | 0.807 [0.726, 0.879] | 0.924 [0.864, 0.970] | -- | 1693.9 |
| 2 | 04_bm25_rerank | 0.864 [0.773, 0.939] | 0.894 [0.818, 0.955] | 0.785 [0.690, 0.867] | 0.856 [0.780, 0.932] | -0.022 [-0.060, +0.000] (tie) | 1655.3 |
| 3 | 02_bm25 | 0.727 [0.621, 0.833] | 0.848 [0.758, 0.924] | 0.673 [0.570, 0.769] | 0.826 [0.735, 0.909] | -0.134 [-0.218, -0.053] | 2.1 |
| 4 | 03_hybrid_rrf | 0.697 [0.576, 0.803] | 0.955 [0.894, 1.000] | 0.605 [0.515, 0.699] | 0.924 [0.864, 0.977] | -0.202 [-0.285, -0.125] | 12.4 |
| 5 | 01_naive_dense | 0.652 [0.530, 0.773] | 0.894 [0.818, 0.955] | 0.554 [0.459, 0.651] | 0.848 [0.773, 0.917] | -0.253 [-0.361, -0.153] | 11.0 |

## By question category

MRR / hit@10 for each question category; the header gives the number of questions.

| pattern | keyword (24) | paraphrase (24) | multi_hop (9) | filter (9) |
|---|---|---|---|---|
| 03+04_hybrid_rerank | 0.946 / 1.000 | 0.747 / 0.917 | 0.630 / 0.889 | 0.775 / 1.000 |
| 04_bm25_rerank | 0.946 / 1.000 | 0.698 / 0.833 | 0.611 / 0.778 | 0.759 / 0.889 |
| 02_bm25 | 0.852 / 1.000 | 0.607 / 0.750 | 0.389 / 0.667 | 0.657 / 0.889 |
| 03_hybrid_rrf | 0.764 / 1.000 | 0.479 / 0.958 | 0.403 / 0.889 | 0.722 / 0.889 |
| 01_naive_dense | 0.693 / 0.958 | 0.440 / 0.792 | 0.485 / 0.889 | 0.559 / 1.000 |
