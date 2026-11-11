# CS / AI textbooks: pipeline summary

Collection `cs_ai` (profile `cs_ai`), written 2026-11-11T05:11:23+00:00 by scripts/run_collection_pipeline.py from the tables in this folder.

## Corpus

Pages kept: 502

| chunker | chunks | mean tokens |
|---|---|---|
| fixed_512 | 492 | 510 |
| structural | 1123 | 224 |

## Evaluation set

60 answerable + 20 unanswerable questions (model qwen2.5:3b, automatically filtered).

By type: definition=16, factual=14, multi_section=9, numeric=8, procedural=13, unanswerable=20

By split: dev 20+4, test 40+16

cross_reference questions: only 5 resolved cross-reference pairs between eligible passages (< 30); cross_reference questions are not built

Note: Unanswerable questions are realistic computer science / AI questions on topics the selected chapters do not cover (other tools, chapters that were not ingested, recent model facts, company practices). The generator LLM may know answers to some of them from pretraining; this is intended: the items test whether the system abstains when the documents lack the answer, not whether an answer exists.

## Retrieval (dev grid)

| config | R@5 | MRR@10 | nDCG@10 | cov@1024 | p50 ms |
|---|---|---|---|---|---|
| structural|dense|bge-small | 0.825 | 0.765 | 0.522 | 0.825 | 28.6 |
| fixed_512|dense|bge-small | 0.900 | 0.790 | 0.562 | 0.800 | 28.7 |
| fixed_512|dense|bge-base | 0.875 | 0.821 | 0.576 | 0.800 | 85.8 |
| fixed_512|hybrid_rrf|bge-base | 0.875 | 0.773 | 0.552 | 0.800 | 86.7 |
| fixed_512|dense|bge-base|rerank-minilm | 0.925 | 0.771 | 0.548 | 0.775 | 2597.1 |
| structural|dense|bge-small|rerank-minilm | 0.875 | 0.823 | 0.558 | 0.775 | 1935.2 |
| structural|dense|bge-base | 0.825 | 0.718 | 0.494 | 0.775 | 85.6 |
| fixed_512|dense|bge-small|rerank-minilm | 0.925 | 0.762 | 0.551 | 0.750 | 2557.1 |
| structural|bm25|rerank-minilm | 0.825 | 0.760 | 0.483 | 0.725 | 1803.7 |
| structural|dense|bge-base|rerank-minilm | 0.825 | 0.805 | 0.545 | 0.725 | 2008.4 |
| structural|hybrid_rrf|bge-base | 0.875 | 0.689 | 0.476 | 0.725 | 87.8 |
| structural|hybrid_rrf|bge-base|rerank-minilm | 0.825 | 0.803 | 0.540 | 0.725 | 1927.2 |
| fixed_512|bm25|rerank-minilm | 0.875 | 0.735 | 0.532 | 0.700 | 2519.2 |
| fixed_512|hybrid_rrf|bge-base|rerank-minilm | 0.925 | 0.750 | 0.544 | 0.700 | 2627.1 |
| structural|bm25 | 0.750 | 0.648 | 0.434 | 0.700 | 1.3 |
| fixed_512|bm25 | 0.850 | 0.681 | 0.477 | 0.650 | 0.6 |

Selected for test (maximise DEV coverage@1024 (token-budget coverage); ties: higher DEV recall@5, then lower median total latency, then config_id; best overall + best per chunker; baselines always included):

- `structural|dense|bge-small`: best_overall, best_structural
- `fixed_512|dense|bge-small`: best_fixed_512
- `fixed_512|dense|bge-base`: baseline_naive_rag
- `fixed_512|bm25`: baseline_bm25

## Retrieval (test)

| config | R@5 | MRR@10 | cov@1024 |
|---|---|---|---|
| structural|dense|bge-small | 0.775 | 0.630 | 0.713 |
| fixed_512|dense|bge-small | 0.775 | 0.663 | 0.650 |
| fixed_512|dense|bge-base | 0.800 | 0.670 | 0.637 |
| fixed_512|bm25 | 0.775 | 0.642 | 0.625 |

Paired bootstrap vs the naive baseline:

| config | metric | diff | 95% CI |
|---|---|---|---|
| structural|dense|bge-small | recall@5 | -0.025 | [-0.150, 0.100] |
| structural|dense|bge-small | coverage@1024 | 0.075 | [-0.038, 0.200] |
| fixed_512|dense|bge-small | recall@5 | -0.025 | [-0.150, 0.087] |
| fixed_512|dense|bge-small | coverage@1024 | 0.013 | [-0.100, 0.138] |
| fixed_512|bm25 | recall@5 | -0.025 | [-0.163, 0.087] |
| fixed_512|bm25 | coverage@1024 | -0.013 | [-0.150, 0.125] |

## Generation (dev)

| system | accuracy | faithful | correct (ans.) | cit. prec. | correct refusal | false abst. | p50 ms |
|---|---|---|---|---|---|---|---|
| G2 | 0.458 | 0.342 | 0.350 | 0.553 | 1.000 | 0.050 | 31246 |
| G3 | 0.417 | 0.429 | 0.300 | 0.643 | 1.000 | 0.300 | 31246 |
| G4 | 0.417 | 0.333 | 0.300 | 0.625 | 1.000 | 0.400 | 49380 |

## Generation (test)

| system | accuracy | faithful | correct (ans.) | cit. prec. | correct refusal | false abst. | p50 ms |
|---|---|---|---|---|---|---|---|
| G1 | 0.500 | 0.011 | 0.300 | 0.500 | 1.000 | 0.250 | 31648 |
| G2 | 0.518 | 0.312 | 0.325 | 0.619 | 1.000 | 0.200 | 31090 |
| G3 | 0.446 | 0.455 | 0.225 | 0.659 | 1.000 | 0.450 | 31090 |
| G4 | 0.429 | 0.444 | 0.200 | 0.657 | 1.000 | 0.550 | 39516 |

Paired bootstrap vs G1:

| system | metric | scope | diff | 95% CI |
|---|---|---|---|---|
| G2 | overall_accuracy | all questions | 0.018 | [-0.089, 0.125] |
| G2 | faithfulness | questions answered by both | 0.284 | [0.123, 0.457] |
| G3 | overall_accuracy | all questions | -0.054 | [-0.161, 0.054] |
| G3 | faithfulness | questions answered by both | 0.426 | [0.204, 0.667] |
| G4 | overall_accuracy | all questions | -0.071 | [-0.179, 0.036] |
| G4 | faithfulness | questions answered by both | 0.444 | [0.200, 0.689] |

Systems not run for this collection:

- G5: retrieval-score threshold: underperformed on consumer_law and the cs_ai dev split is small, so a tuned threshold would be unreliable
- G6: fixed_256 hybrid + reranking with guards and critic: the slowest system; fixed_256 is not built for cs_ai

## Pipeline steps

| step | time | peak RSS MB | completed (UTC) |
|---|---|---|---|
| ingest | 13s | 633 | 2026-11-11T01:03:04+00:00 |
| chunk | 13s | 392 | 2026-11-11T01:03:17+00:00 |
| build_index | 16m37s | 1,245 | 2026-11-11T01:19:54+00:00 |
| build_auto_eval | 2h01m | 644 | 2026-11-11T03:21:27+00:00 |
| validate_eval | 1s | 30 | 2026-11-11T03:21:28+00:00 |
| retrieval_dev_grid | 4m12s | 1,189 | 2026-11-11T03:25:40+00:00 |
| select_configs | 1s | 75 | 2026-11-11T03:25:41+00:00 |
| eval_retrieval_test | 24s | 922 | 2026-11-11T03:26:06+00:00 |
| generation_dev | 25m18s | 1,807 | 2026-11-11T03:51:24+00:00 |
| derive_dev | 3s | 104 | 2026-11-11T03:51:27+00:00 |
| generation_test | 1h19m | 1,803 | 2026-11-11T05:11:22+00:00 |

