# Hard-pilot routing decision

**Decision: defer ML router training.** This artifact does not provide a completed five-model comparison. The recovered snapshot contains 24/75 graded pairs (32%) and zero fully graded five-model prompts. The available GPT/Claude overlap shows disagreement, but its accuracy oracle does not beat Claude. It is insufficient evidence of a learnable routing advantage.

**Artifact validation and offline recovery**

The requested CSV was read directly and analyzed with the existing `pilot_analysis.py`, using the fixed 15-prompt × 5-model plan, not an inferred plan. Its rows matched all 75 unique planned pairs. It was changing during this task: the first read had 23 grades, 5 provider failures, 46 skips, 1 parsing failure, and $0.3146834 recorded spend. A later read showed an additional Qwen failure and overwrote the offline regrade. No provider calls were made by this analysis.

To avoid further concurrent writes, the later CSV was frozen at `data/results/hard_pilot_analysis_snapshot.csv`. All tables below use that fixed snapshot after offline recovery. These counts are not a claim about the eventual state of the changing source file.

The existing regrade command initially left Claude’s `mmlu_pro:12015` raw response `**F**` unparsed. A two-line change to the existing multiple-choice parser accepts a standalone bold option. Running `generate_dataset.py --regrade-results --output data/results/hard_pilot_analysis_snapshot.csv` then recovered `F`, which matches reference `F`. No raw response, spend, latency, token, request, generation, or timestamp field changed; only that row’s parsing/grading fields changed. All 82 existing/regression tests passed offline. No new evaluation infrastructure was added.

Snapshot SHA-256: `a58e79eb58519bb4a170142a4c1b60bbbc799fb671d40eb512488bfb52becf85`.

| Metric | Snapshot result |
| --- | --- |
| Expected pairs | 75 |
| Recorded pairs | 75 |
| Successfully graded pairs | 24 |
| Provider failures (pair outcomes) | 6 |
| Parsing failures | 0 |
| Grading failures | 0 |
| Skipped pairs | 45 |
| Unrecorded pairs | 0 |
| Spend-limit stops | 0 |
| Total recorded spend | $0.3449374 |
| Historical provider attempts / responses / failed attempts | 31 / 24 / 7 |

Recorded spend sums `estimated_cost_usd`, including failed-attempt reserves. It is not an independently verified provider bill. In this snapshot $0.2535674 belongs to returned responses and $0.0913700 to failed rows/reserves. Three legacy invalid-response failures record zero cost; this analysis does not infer their actual billing. Policy diagnostics use response cost and exclude failure reserves. All successful rows used in the two-model comparison have zero retries, permitting the documented legacy response-cost fallback.

| Model | Expected / recorded | Graded | Provider failures | Skips | Recorded spend |
| --- | --- | --- | --- | --- | --- |
| GPT | 15 / 15 | 10 | 1 | 4 | $0.0883320 |
| Claude | 15 / 15 | 10 | 1 | 4 | $0.0767750 |
| Grok | 15 / 15 | 2 | 1 | 12 | $0.1135040 |
| DeepSeek | 15 / 15 | 2 | 1 | 12 | $0.0039204 |
| Qwen | 15 / 15 | 0 | 2 | 13 | $0.0624060 |

| Benchmark | Expected / recorded | Graded | Provider failures | Skips | Recorded spend |
| --- | --- | --- | --- | --- | --- |
| mmlu_pro | 25 / 25 | 14 | 4 | 7 | $0.2817354 |
| math_500 | 25 / 25 | 10 | 0 | 15 | $0.0632020 |
| livecodebench | 25 / 25 | 0 | 2 | 23 | $0.0000000 |

Model × benchmark completeness: each cell has five expected and five recorded rows. G = graded, F = provider failure, S = skipped. Parsing/grading failures and unrecorded rows are zero in every cell.

| Benchmark | GPT | Claude | Grok | DeepSeek | Qwen |
| --- | --- | --- | --- | --- | --- |
| mmlu_pro | 5G / 0F / 0S | 5G / 0F / 0S | 2G / 1F / 2S | 2G / 1F / 2S | 0G / 2F / 3S |
| math_500 | 5G / 0F / 0S | 5G / 0F / 0S | 0G / 0F / 5S | 0G / 0F / 5S | 0G / 0F / 5S |
| livecodebench | 0G / 1F / 4S | 0G / 1F / 4S | 0G / 0F / 5S | 0G / 0F / 5S | 0G / 0F / 5S |

**Why 51 pairs remain incomplete**

| Prompt | Model | Saved failure | Recorded cost |
| --- | --- | --- | --- |
| mmlu_pro:1346 | Qwen | invalid_provider_response: Provider response did not contain non-empty text output | $0.0321520 |
| mmlu_pro:5022 | Qwen | invalid_provider_response: Provider response did not contain non-empty text output | $0.0302540 |
| mmlu_pro:12015 | Grok | network_error: The read operation timed out | $0.0289640 |
| mmlu_pro:12015 | DeepSeek | invalid_provider_response: Provider response did not contain non-empty text output | $0.0000000 |
| livecodebench:arc196_d | GPT | invalid_provider_response: Provider response did not contain non-empty text output | $0.0000000 |
| livecodebench:arc196_d | Claude | invalid_provider_response: Provider response did not contain non-empty text output | $0.0000000 |

All six provider-failure rows have empty saved raw responses, so none can be recovered offline. The other 45 rows are `skipped_model` with `model_disabled_after_provider_failure` and “No call made after an earlier provider/model failure in this run.” They have no saved raw responses either. GPT/Claude each skipped the four code prompts following `arc196_d`; Grok/DeepSeek each skipped the final two MMLU-Pro prompts and all ten math/code prompts; Qwen skipped the final three MMLU-Pro prompts and all ten math/code prompts. These are historical recorded outcomes; this report does not assume current runner behavior would reproduce that blocking.

**Prompt correctness vectors**

1 = graded correct; 0 = graded incorrect; F = provider failure; S = skipped. F/S are unknown correctness, never zero. Prompt IDs below are exact IDs from the fixed plan.

| Prompt | GPT | Claude | Grok | DeepSeek | Qwen |
| --- | --- | --- | --- | --- | --- |
| mmlu_pro:1346 | 1 | 1 | 0 | 1 | F |
| mmlu_pro:5022 | 0 | 0 | 0 | 0 | F |
| mmlu_pro:12015 | 0 | 1 | F | F | S |
| mmlu_pro:6502 | 0 | 0 | S | S | S |
| mmlu_pro:10442 | 1 | 1 | S | S | S |
| math_500:test/intermediate_algebra/960.json | 0 | 1 | S | S | S |
| math_500:test/number_theory/769.json | 1 | 1 | S | S | S |
| math_500:test/counting_and_probability/870.json | 0 | 1 | S | S | S |
| math_500:test/geometry/965.json | 1 | 1 | S | S | S |
| math_500:test/precalculus/986.json | 0 | 1 | S | S | S |
| livecodebench:arc196_d | F | F | S | S | S |
| livecodebench:arc196_c | S | S | S | S | S |
| livecodebench:arc196_b | S | S | S | S | S |
| livecodebench:arc196_a | S | S | S | S | S |
| livecodebench:abc400_g | S | S | S | S | S |

There are **zero eligible prompts** for the full five-model agreement calculation. On that empty common subset, all-agree, all-correct, all-wrong, and disagree counts are each 0; the disagreement rate is undefined (0/0). These are not counts for the full 15-prompt plan: full-plan all-agree/all-correct/all-wrong/disagree totals remain unknown.

Partial vectors nevertheless prove disagreement when both a 1 and a 0 are already observed. Five prompts satisfy this: `mmlu_pro:1346`, `mmlu_pro:12015`, and math `intermediate_algebra/960.json`, `counting_and_probability/870.json`, `precalculus/986.json`. Thus full-plan disagreement is **at least 5/15 = 33.3%**; the remaining ten prompts are unresolved. No prompt has proven five-model agreement, all-correctness, or all-wrongness.

| Benchmark | Fully graded five-model prompts | Confirmed disagreement lower bound | Unresolved agreement |
| --- | --- | --- | --- |
| mmlu_pro | 0 | 2/5 | 3 |
| math_500 | 0 | 3/5 | 2 |
| livecodebench | 0 | 0/5 | 5 |

Pairwise disagreements use each pair’s own jointly graded prompts, not the empty five-model subset. Different denominators prevent ranking pairs directly. N/A means no overlap, not zero disagreement.

| Pair | All | MMLU-Pro | MATH-500 | LiveCodeBench |
| --- | --- | --- | --- | --- |
| GPT / Claude | 4/10 (40%) | 1/5 (20%) | 3/5 (60%) | N/A (n=0) |
| GPT / Grok | 1/2 (50%) | 1/2 (50%) | N/A (n=0) | N/A (n=0) |
| GPT / DeepSeek | 0/2 (0%) | 0/2 (0%) | N/A (n=0) | N/A (n=0) |
| GPT / Qwen | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| Claude / Grok | 1/2 (50%) | 1/2 (50%) | N/A (n=0) | N/A (n=0) |
| Claude / DeepSeek | 0/2 (0%) | 0/2 (0%) | N/A (n=0) | N/A (n=0) |
| Claude / Qwen | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| Grok / DeepSeek | 1/2 (50%) | 1/2 (50%) | N/A (n=0) | N/A (n=0) |
| Grok / Qwen | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |
| DeepSeek / Qwen | N/A (n=0) | N/A (n=0) | N/A (n=0) | N/A (n=0) |

**Requested five-model routing baselines**

All requested deterministic policies have n=0 on the common fully graded five-model subset. Their accuracy, average/total comparison cost, and latency cannot be estimated. An empty cost sum of $0 would not be a usable policy cost.

| Always model | n | Accuracy | Average cost | Total cost | Average latency |
| --- | --- | --- | --- | --- | --- |
| GPT | 0 | N/A | N/A | N/A | N/A |
| Claude | 0 | N/A | N/A | N/A | N/A |
| Grok | 0 | N/A | N/A | N/A | N/A |
| DeepSeek | 0 | N/A | N/A | N/A | N/A |
| Qwen | 0 | N/A | N/A | N/A | N/A |

Best single model, cheapest model, best accuracy/cost static model, five-model oracle accuracy, cheapest-correct oracle cost, oracle accuracy improvement, and oracle cost ratios relative to best/cheapest are all **unavailable**. Benchmark/task-type and prompt-length heuristic comparisons on that same five-model subset are also unavailable. Selecting different rows for different policies would not answer the requested comparison.

**Supplementary GPT/Claude diagnostic: ten shared graded prompts**

This separate, explicitly reduced comparison includes all five MMLU-Pro and all five MATH-500 prompts. It excludes LiveCodeBench and does not stand in for a five-model benchmark. Costs and latencies are observed single-run measurements, not predicted serving performance.

| Policy | Accuracy | Average cost/prompt | Total cost | Average latency |
| --- | --- | --- | --- | --- |
| GPT | 4/10 (40%) | $0.0088332 | $0.0883320 | 9.026 s |
| Claude | 8/10 (80%) | $0.0076775 | $0.0767750 | 3.439 s |
| Math → Claude; otherwise GPT | 7/10 (70%) | $0.0108630 | $0.1086300 | 9.076 s |
| Prompt > 2,000 characters → Claude; otherwise GPT | 5/10 (50%) | $0.0048566 | $0.0485660 | 3.228 s |

Claude is the best by accuracy, cheapest by observed mean response cost, and best accuracy/cost static model within this reduced set. Defining accuracy/cost as accuracy divided by mean dollars per prompt gives Claude 104.20 versus GPT 45.28 expected correct answers per dollar. This does not identify the best or cheapest among all five models.

The two heuristic rules use only task type or the character count of the saved prompt (including its answer-format instructions). The 2,000-character threshold is a single illustrative cutoff, not a threshold search. Both rules were examined after seeing this pilot, so their results are exploratory and have no held-out validation. Neither beats Claude’s accuracy. The length rule achieves 50% accuracy for $0.0485660, losing 30 percentage points for a 36.7% cost reduction. The task-type rule is both less accurate and more expensive than always-Claude.

GPT and Claude agree on 6/10 prompts: four both correct and two both wrong. They disagree on 4/10 (40%): 1/5 MMLU-Pro (20%) and 3/5 MATH-500 (60%). Every disagreement favors Claude. Claude’s benchmark accuracy is 3/5 MMLU-Pro and 5/5 MATH-500; GPT’s is 2/5 in each. This indicates a larger observed Claude advantage on math, but no benchmark where GPT has an accuracy advantage. Length is confounded with benchmark: all five math prompts are 95–335 characters, while MMLU-Pro prompts are 1,722–3,788. There is no code evidence, and two-prompt Grok/DeepSeek overlaps cannot establish specialization.

The two-model oracle solves 8/10 = 80%, **0 percentage points above Claude**. Cheapest-correct cost on those eight solved prompts is $0.0545990; no correct model exists for `mmlu_pro:5022` or `mmlu_pro:6502`. The solved-only cost must not be compared directly to ten-prompt policy totals as though failures cost nothing.

For a complete ten-prompt oracle cost accounting, choose the cheapest observed correct response when one exists and fall back to always-Claude on the two unsolved prompts. Cost becomes $0.0649990 ($0.0064999/prompt), or 84.662% of both the best and cheapest static model (Claude), a 15.3% reduction at the same 80% accuracy. This is an ex-post bound using correctness and realized response cost, not a prompt-visible deployable policy. Cost savings have not been demonstrated at equal accuracy by the two tested heuristics.

**Interpretation and next decision**

Do not train the first ML router from this artifact. Partial observations establish some disagreement, but disagreement alone does not establish complementary strengths: Claude wins every GPT/Claude disagreement, and the reduced oracle has zero accuracy lift. The existing analyzer’s broad `evidence_of_routing_signal` flag can become true merely because one cheaper model beats a more expensive one; that is compatible with static dominance and is not a sufficient ML gate.

The unresolved requirement is evaluation coverage, not additional infrastructure. Once the concurrently changing run is finished, inspect its artifact again and recover saved answers offline. Obtaining grades for the remaining empty-response failures/skips would require additional provider work, which was not initiated here. Even a completed 15-prompt pilot would be descriptive evidence for whether to collect training data, rather than enough data to establish generalization of an ML router. No embeddings, training, provider calls, or additional rule searches were performed.

Reproduction: run `python3 pilot_analysis.py data/results/hard_pilot_analysis_snapshot.csv` for the existing accounting output (`reports/hard_pilot_existing_analysis.json`). Extended arithmetic, full vectors, pairwise denominators, selected rows for both heuristic policies, and the regrade field audit are saved in `reports/hard_pilot_routing_analysis.json`. The original changing CSV is retained separately.
