# LLMRouterBench preparation result

Primary dataset: 8,706 complete prompts × 8 models = 69,648 outcomes. No router has been trained; no provider or embedding API calls were made.

## Selected-model coverage before complete-case filtering

| model_id | expected_pairs | present_pairs | eligible_pairs | labeled_pairs | priced_pairs | valid_token_pairs | outcome_coverage | coverage | missing_pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| claude-sonnet-4 | 8760 | 8760 | 8759 | 8760 | 8759 | 8759 | 1.000000 | 0.999886 | 1 |
| deepseek-r1-0528 | 8760 | 8760 | 8760 | 8760 | 8760 | 8760 | 1.000000 | 1.000000 | 0 |
| deepseek-v3-0324 | 8760 | 8760 | 8760 | 8760 | 8760 | 8760 | 1.000000 | 1.000000 | 0 |
| gemini-2.5-flash | 8760 | 8760 | 8752 | 8760 | 8752 | 8752 | 1.000000 | 0.999087 | 8 |
| gpt-5 | 8760 | 8760 | 8744 | 8744 | 8744 | 8744 | 1.000000 | 0.998174 | 16 |
| gpt-5-chat | 8760 | 8760 | 8760 | 8760 | 8760 | 8760 | 1.000000 | 1.000000 | 0 |
| intern-s1 | 8760 | 8760 | 8760 | 8760 | 8760 | 8760 | 1.000000 | 1.000000 | 0 |
| qwen3-235b-a22b-2507 | 8760 | 8760 | 8731 | 8731 | 8731 | 8731 | 1.000000 | 0.996689 | 29 |

## Task retention and splits

| dataset | before_filter | complete_prompts | standard_train | standard_validation | standard_test | ood_train | ood_validation | ood_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aime | 60 | 56 | 39 | 9 | 8 | 48 | 8 | 0 |
| gpqa | 198 | 198 | 138 | 30 | 30 | 168 | 30 | 0 |
| livecodebench | 1055 | 1055 | 739 | 158 | 158 | 0 | 0 | 1055 |
| livemathbench | 121 | 117 | 82 | 18 | 17 | 99 | 18 | 0 |
| mmlupro | 3000 | 2979 | 2085 | 447 | 447 | 2532 | 447 | 0 |
| simpleqa | 4326 | 4301 | 3011 | 645 | 645 | 3656 | 645 | 0 |

## Baseline references

### standard

Always-cheapest fixed model: `qwen3-235b-a22b-2507` (training mean cost). Best-single: `gpt-5` (validation success only). Cheapest fixed model by test mean cost, descriptive only: `qwen3-235b-a22b-2507`.

| policy | success_rate | mean_cost_usd |
| --- | --- | --- |
| always_cheapest | 0.652874 | 0.000436 |
| best_single | 0.691188 | 0.019637 |
| hindsight_cheapest_cost | 0.640613 | 0.000281 |
| oracle_cheapest_success_with_fallback | 0.866667 | 0.003070 |

Ex-post oracle success headroom: 17.548 percentage points; oracle cost reduction relative to best-single: 84.364%. The oracle and hindsight-cheapest policies use unobservable outcomes/costs and are not achievable routing claims. On unsolved prompts, oracle cost includes the training-cheapest fallback call; conditional cheapest-success cost remains null.

### ood

Always-cheapest fixed model: `qwen3-235b-a22b-2507` (training mean cost). Best-single: `gpt-5` (validation success only). Cheapest fixed model by test mean cost, descriptive only: `qwen3-235b-a22b-2507`.

| policy | success_rate | mean_cost_usd |
| --- | --- | --- |
| always_cheapest | 0.641706 | 0.000990 |
| best_single | 0.864455 | 0.052053 |
| hindsight_cheapest_cost | 0.626540 | 0.000679 |
| oracle_cheapest_success_with_fallback | 0.911848 | 0.013240 |

Ex-post oracle success headroom: 4.739 percentage points; oracle cost reduction relative to best-single: 74.565%. The oracle and hindsight-cheapest policies use unobservable outcomes/costs and are not achievable routing claims. On unsolved prompts, oracle cost includes the training-cheapest fallback call; conditional cheapest-success cost remains null.

## Static model summaries on standard test

| model_id | outcomes | accuracy | mean_raw_score | mean_cost_usd | median_cost_usd | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| claude-sonnet-4 | 1305 | 0.448276 | 0.448276 | 0.005802 | 0.003300 | 1.000000 |
| deepseek-r1-0528 | 1305 | 0.555556 | 0.555556 | 0.008372 | 0.002261 | 1.000000 |
| deepseek-v3-0324 | 1305 | 0.501916 | 0.501916 | 0.000500 | 0.000237 | 1.000000 |
| gemini-2.5-flash | 1305 | 0.514176 | 0.514176 | 0.002889 | 0.000272 | 1.000000 |
| gpt-5 | 1305 | 0.691188 | 0.691188 | 0.019637 | 0.010372 | 1.000000 |
| gpt-5-chat | 1305 | 0.590805 | 0.590805 | 0.002715 | 0.001517 | 1.000000 |
| intern-s1 | 1305 | 0.437548 | 0.437548 | 0.002133 | 0.000796 | 1.000000 |
| qwen3-235b-a22b-2507 | 1305 | 0.652874 | 0.652874 | 0.000436 | 0.000125 | 1.000000 |

Full task-level statistics, both evaluation regimes, and per-prompt baseline/oracle decisions are in Parquet. See `llmrouterbench_processed_summary.json` for exact counts and `ml_experiment_spec.md` for definitions.

## Missingness and leakage

54 prompts excluded out of 8760; 54 unavailable model outcomes. Raw scores and resource anomalies remain in audit artifacts. Availability masks are explicit; absent labels are not zero-filled. No complete-case imputation.

Splits use only prompt text, source dataset, duplicate groups, and seed 3407. All selected outcomes for a request share its split. Normalized original questions and rendered requests form duplicate groups across datasets. OOD holds out all LiveCodeBench; overlapping non-held-out prompts would be purged. Features contain exactly the allowlisted prompt-derived counts; dataset/task labels are separate metadata. Best-single selection rejects non-validation rows. Cost estimates use training rows only. Complete-case filtering may bias evaluation toward callable, accounted-for requests; it is not a reliability benchmark.

## Prompt-visible features

`character_length`, `word_count`, `token_estimate`, `line_count`, `code_fence_count`, `contains_code`, `contains_math`, `answer_choice_count`, `digit_count`, `question_mark_count`, `punctuation_count`, `comma_count`, `period_count`, `colon_count`, `semicolon_count`, `bracket_count`.
