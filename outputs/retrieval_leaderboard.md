# Retrieval leaderboard

66 questions, 333 chunks, k=10. Embeddings: `BAAI/bge-small-en-v1.5`. Reranker: `BAAI/bge-reranker-v2-m3`. Ranked by MRR, then recall@10, then hit@3. 95% bootstrap confidence intervals (1000 resamples, seed 42). mrr_vs_top is each row's MRR minus the top row's, bootstrapped over per-question differences; "(tie)" marks an interval that contains 0.

| rank | pattern | hit@3 | hit@10 | mrr | recall@10 | mrr_vs_top | p50_ms |
|---|---|---|---|---|---|---|---|
| 1 | 03+04_hybrid_rerank | 0.864 [0.773, 0.939] | 0.924 [0.864, 0.985] | 0.780 [0.696, 0.860] | 0.902 [0.833, 0.962] | -- | 1617.1 |
| 2 | 04_bm25_rerank | 0.833 [0.742, 0.924] | 0.879 [0.803, 0.955] | 0.757 [0.663, 0.845] | 0.856 [0.773, 0.932] | -0.023 [-0.058, +0.000] (tie) | 1598.3 |
| 3 | 02_bm25 | 0.742 [0.636, 0.848] | 0.833 [0.742, 0.924] | 0.674 [0.571, 0.777] | 0.818 [0.727, 0.909] | -0.106 [-0.183, -0.026] | 1.1 |
| 4 | 03_hybrid_rrf | 0.636 [0.515, 0.742] | 0.939 [0.879, 0.985] | 0.589 [0.496, 0.684] | 0.924 [0.856, 0.977] | -0.191 [-0.277, -0.114] | 11.1 |
| 5 | 01_naive_dense | 0.621 [0.500, 0.727] | 0.818 [0.712, 0.909] | 0.540 [0.439, 0.635] | 0.780 [0.682, 0.871] | -0.240 [-0.341, -0.136] | 8.8 |

## By question category

MRR / hit@10 for each question category; the header gives the number of questions.

| pattern | keyword (24) | paraphrase (24) | multi_hop (9) | filter (9) |
|---|---|---|---|---|
| 03+04_hybrid_rerank | 0.946 / 1.000 | 0.747 / 0.917 | 0.463 / 0.778 | 0.741 / 0.889 |
| 04_bm25_rerank | 0.946 / 1.000 | 0.698 / 0.833 | 0.444 / 0.667 | 0.722 / 0.889 |
| 02_bm25 | 0.852 / 1.000 | 0.607 / 0.750 | 0.426 / 0.556 | 0.625 / 0.889 |
| 03_hybrid_rrf | 0.764 / 1.000 | 0.479 / 0.958 | 0.383 / 0.778 | 0.620 / 0.889 |
| 01_naive_dense | 0.693 / 0.958 | 0.440 / 0.792 | 0.546 / 0.778 | 0.389 / 0.556 |
