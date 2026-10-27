# Retrieval leaderboard

66 questions, 333 chunks, k=10. Embeddings: `BAAI/bge-small-en-v1.5`. Reranker: `BAAI/bge-reranker-v2-m3`. Ranked by MRR, then recall@10, then hit@3. 95% bootstrap confidence intervals (1000 resamples, seed 42). mrr_vs_top is each row's MRR minus the top row's, bootstrapped over per-question differences; "(tie)" marks an interval that contains 0.

| rank | pattern | hit@3 | hit@10 | mrr | recall@10 | mrr_vs_top | p50_ms |
|---|---|---|---|---|---|---|---|
| 1 | 03+04_hybrid_rerank | 0.864 [0.773, 0.939] | 0.924 [0.864, 0.985] | 0.802 [0.718, 0.881] | 0.917 [0.848, 0.977] | -- | 1695.7 |
| 2 | 04_bm25_rerank | 0.833 [0.742, 0.924] | 0.879 [0.803, 0.955] | 0.780 [0.687, 0.869] | 0.864 [0.780, 0.939] | -0.023 [-0.058, +0.000] (tie) | 1648.4 |
| 3 | 02_bm25 | 0.788 [0.682, 0.879] | 0.864 [0.773, 0.939] | 0.725 [0.622, 0.826] | 0.841 [0.750, 0.917] | -0.077 [-0.153, -0.008] | 1.6 |
| 4 | 03_hybrid_rrf | 0.758 [0.652, 0.864] | 0.939 [0.879, 0.985] | 0.674 [0.580, 0.764] | 0.932 [0.871, 0.985] | -0.128 [-0.204, -0.054] | 15.5 |
| 5 | 01_naive_dense | 0.712 [0.591, 0.818] | 0.833 [0.742, 0.909] | 0.609 [0.509, 0.703] | 0.818 [0.727, 0.902] | -0.194 [-0.279, -0.100] | 12.5 |

## By question category

MRR / hit@10 for each question category; the header gives the number of questions.

| pattern | keyword (24) | paraphrase (24) | multi_hop (9) | filter (9) |
|---|---|---|---|---|
| 03+04_hybrid_rerank | 0.967 / 1.000 | 0.788 / 0.917 | 0.463 / 0.778 | 0.741 / 0.889 |
| 04_bm25_rerank | 0.967 / 1.000 | 0.740 / 0.833 | 0.444 / 0.667 | 0.722 / 0.889 |
| 02_bm25 | 0.911 / 1.000 | 0.682 / 0.792 | 0.444 / 0.667 | 0.625 / 0.889 |
| 03_hybrid_rrf | 0.837 / 1.000 | 0.599 / 0.958 | 0.494 / 0.778 | 0.624 / 0.889 |
| 01_naive_dense | 0.737 / 0.958 | 0.578 / 0.833 | 0.565 / 0.778 | 0.393 / 0.556 |
