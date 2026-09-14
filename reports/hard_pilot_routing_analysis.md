# Final hard-pilot routing analysis

**Decision: defer ML training.** The run is no longer active, but its final artifact has only 24/75 successfully graded pairs and no fully graded five-model prompts. Stopped does not mean successfully completed.

**Final artifact and provenance**

Source final CSV: `data/results/hard_pilot_results.csv`. Primary analysis input: [`reports/snapshots/hard_pilot_final.csv`](snapshots/hard_pilot_final.csv). Frozen at 2026-09-14T01:57:53.090980+00:00, from working tree based on commit `ee278cb63abc4921affd2d2d7d04b194309cb596`. The input CSV is included in the repository; no `data/results/` files or credentials are needed for reproduction.

Final SHA-256: `a58e79eb58519bb4a170142a4c1b60bbbc799fb671d40eb512488bfb52becf85`. File size: 155,902 bytes; row count: 75.

Stability evidence: process inspection found no Python or hard-pilot writer. Identical size, SHA-256, row count and modification time were observed at 2026-09-14T01:56:31.601521+00:00 and 2026-09-14T01:57:14.728262+00:00. No original exit log/status was available, so successful exit is not asserted. Before regrade the source was 155,973 bytes, SHA-256 `a807bea092612d340cda0ac6415d19a55588a88341141c5c1ed10a2ef3bf8359`.

The earlier `data/results/hard_pilot_analysis_snapshot.csv` is superseded. The final recovered bytes happen to have the same hash as that intermediate snapshot; this final snapshot was copied directly from the stopped source after recovery, not reconstructed from the earlier snapshot. No new successful responses were added between those states.

Secret inspection found no API keys, authorization headers, credential values, environment contents, or private-key material. Pattern scans, private exact comparisons against local credential values, and schema/text inspection found no credentials. No sanitization or field removal was necessary. Normal provider request IDs remain.

**One offline recovery and complete change audit**

Ran once against the stable source: `python3 generate_dataset.py --regrade-results --output data/results/hard_pilot_results.csv`. The existing standalone bold-option parser remains unchanged and narrow. Claude’s saved `**F**` on `mmlu_pro:12015` becomes parsed `F`, matching reference `F`. Raw responses and all provider telemetry, costs, latencies, tokens, timestamps, request IDs and generation settings were verified unchanged.

| Prompt / model | Field | Before | After |
| --- | --- | --- | --- |
| mmlu_pro:12015 / claude-opus-5 | correct | '' | 'True' |
| mmlu_pro:12015 / claude-opus-5 | error_message | 'Could not extract one unambiguous multiple-choice option' | '' |
| mmlu_pro:12015 / claude-opus-5 | error_type | 'parsing_failure' | '' |
| mmlu_pro:12015 / claude-opus-5 | parsed_answer | '' | 'F' |
| mmlu_pro:12015 / claude-opus-5 | score | '' | '1.0' |
| mmlu_pro:12015 / claude-opus-5 | status | 'parsing_failure' | 'success' |

**Completeness**

| Metric | Stable source before recovery | Final recovered snapshot |
| --- | --- | --- |
| expected_pairs | 75 | 75 |
| recorded_pairs | 75 | 75 |
| successfully_graded_pairs | 23 | 24 |
| provider_failures | 6 | 6 |
| parsing_failures | 1 | 0 |
| grading_failures | 0 | 0 |
| skipped_pairs | 45 | 45 |
| unrecorded_pairs | 0 | 0 |
| spend_limit_stops | 0 | 0 |
| attempted_provider_calls | 31 | 31 |
| successful_provider_responses | 24 | 24 |
| failed_provider_attempts | 7 | 7 |
| total_recorded_cost_usd | 0.3449374 | 0.3449374 |

Recorded spend includes failed-attempt reserves; it is not independently verified provider billing. Legacy empty-response rows with zero recorded cost do not prove zero actual billing. Attempts/responses include recorded or legacy-inferred history. Failures and skips never become incorrect answers.

| Group | Expected | Recorded | Graded | Provider fail | Parse fail | Grade fail | Skipped | Missing | Spend stops | Attempts | Responses | Spend USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Claude | 15 | 15 | 10 | 1 | 0 | 0 | 4 | 0 | 0 | 11 | 10 | 0.076775 |
| DeepSeek | 15 | 15 | 2 | 1 | 0 | 0 | 12 | 0 | 0 | 3 | 2 | 0.0039204 |
| GPT | 15 | 15 | 10 | 1 | 0 | 0 | 4 | 0 | 0 | 11 | 10 | 0.088332 |
| Qwen | 15 | 15 | 0 | 2 | 0 | 0 | 13 | 0 | 0 | 3 | 0 | 0.062406 |
| Grok | 15 | 15 | 2 | 1 | 0 | 0 | 12 | 0 | 0 | 3 | 2 | 0.113504 |

| Group | Expected | Recorded | Graded | Provider fail | Parse fail | Grade fail | Skipped | Missing | Spend stops | Attempts | Responses | Spend USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mmlu_pro | 25 | 25 | 14 | 4 | 0 | 0 | 7 | 0 | 0 | 19 | 14 | 0.2817354 |
| math_500 | 25 | 25 | 10 | 0 | 0 | 0 | 15 | 0 | 0 | 10 | 10 | 0.063202 |
| livecodebench | 25 | 25 | 0 | 2 | 0 | 0 | 23 | 0 | 0 | 2 | 0 | 0 |

| Group | Expected | Recorded | Graded | Provider fail | Parse fail | Grade fail | Skipped | Missing | Spend stops | Attempts | Responses | Spend USD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| livecodebench / Claude | 5 | 5 | 0 | 1 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 0 |
| livecodebench / DeepSeek | 5 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| livecodebench / GPT | 5 | 5 | 0 | 1 | 0 | 0 | 4 | 0 | 0 | 1 | 0 | 0 |
| livecodebench / Qwen | 5 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| livecodebench / Grok | 5 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| math_500 / Claude | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0.04175 |
| math_500 / DeepSeek | 5 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| math_500 / GPT | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0.021452 |
| math_500 / Qwen | 5 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| math_500 / Grok | 5 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| mmlu_pro / Claude | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0.035025 |
| mmlu_pro / DeepSeek | 5 | 5 | 2 | 1 | 0 | 0 | 2 | 0 | 0 | 3 | 2 | 0.0039204 |
| mmlu_pro / GPT | 5 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 5 | 0.06688 |
| mmlu_pro / Qwen | 5 | 5 | 0 | 2 | 0 | 0 | 3 | 0 | 0 | 3 | 0 | 0.062406 |
| mmlu_pro / Grok | 5 | 5 | 2 | 1 | 0 | 0 | 2 | 0 | 0 | 3 | 2 | 0.113504 |

**Five-model routing signal**

Correctness vectors use 1/0 only for successfully graded answers; F = provider failure and S = skipped.

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

| Subset | Eligible prompts | All correct | All wrong | All agree | Disagree | Disagreement rate |
| --- | --- | --- | --- | --- | --- | --- |
| All five models | 0 | 0 | 0 | 0 | 0 | N/A |
| mmlu_pro | 0 | 0 | 0 | 0 | 0 | N/A |
| math_500 | 0 | 0 | 0 | 0 | 0 | N/A |
| livecodebench | 0 | 0 | 0 | 0 | 0 | N/A |

The zero counts above describe an empty eligible subset, not zero disagreement across the 15-prompt plan. Full-plan totals are unknown. Partial vectors prove disagreement on at least 5/15 prompts (33.3%): two MMLU-Pro and three MATH-500; code has no grades.

Pairwise rows use each pair’s own jointly graded overlap. Denominators differ; N/A is not agreement.

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

**Static policies and oracle on the five-model common subset**

| Policy | n | Accuracy | Total cost | Average cost | Average latency ms |
| --- | --- | --- | --- | --- | --- |
| GPT | 0 | N/A | N/A | N/A | N/A |
| Claude | 0 | N/A | N/A | N/A | N/A |
| Grok | 0 | N/A | N/A | N/A | N/A |
| DeepSeek | 0 | N/A | N/A | N/A | N/A |
| Qwen | 0 | N/A | N/A | N/A | N/A |

Best single model, cheapest model, best static accuracy/cost model, five-model oracle accuracy, oracle lift, cheapest-correct cost and oracle cost relative to best static are all N/A: the common subset is empty. No reduced comparison substitutes for these missing five-model results.

**Secondary diagnostic: GPT/Claude only, ten shared graded prompts**

| Model | n | Accuracy | Total cost | Average cost | Average latency ms |
| --- | --- | --- | --- | --- | --- |
| GPT | 10 | 0.4 | $0.0883320 | $0.0088332 | 9026.2653418 |
| Claude | 10 | 0.8 | $0.0767750 | $0.0076775 | 3438.9818251 |

These ten prompts comprise five MMLU-Pro and five MATH-500; neither model has a code grade. They agree on six (four both correct, two both wrong) and disagree on four (40%). MMLU-Pro disagreement is 1/5; MATH-500 is 3/5. Every disagreement favors Claude. Claude is more accurate, cheaper in observed mean cost, and faster on this subset; it also has the best static accuracy/cost ratio. That is static dominance, not complementary accuracy strengths.

The reduced oracle is 80%, with 0% accuracy lift over Claude. Cheapest-correct cost is $0.0545990 for eight solved prompts only. The two unsolved prompts are `mmlu_pro:5022` and `mmlu_pro:6502`. Charging a Claude fallback on those gives $0.0649990 across all ten, 84.7% of Claude’s total cost (15.3% savings at equal accuracy). This uses ex-post correctness and realized costs; it does not demonstrate a prompt-visible policy or generalization.

**Why the matrix is incomplete and what resume would do**

All 51 incomplete pairs are listed below. None has a saved response, so none is recoverable offline; each would need another provider call. Five empty-text provider failures are classified as “other”, not permanent model failures or incorrect answers: the CSV does not establish a permanent cause. One network timeout is transient. All 45 historical skips followed an earlier provider/model failure. Parsing failures, grading failures, permanent failures and spend-limit stops are zero after recovery.

| Prompt | Model | Classification | Saved reason / error |
| --- | --- | --- | --- |
| mmlu_pro:1346 | Qwen | other | invalid_provider_response: Provider response did not contain non-empty text output |
| mmlu_pro:5022 | Qwen | other | invalid_provider_response: Provider response did not contain non-empty text output |
| mmlu_pro:12015 | Grok | transient provider failure | network_error: The read operation timed out |
| mmlu_pro:12015 | DeepSeek | other | invalid_provider_response: Provider response did not contain non-empty text output |
| mmlu_pro:12015 | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| mmlu_pro:6502 | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| mmlu_pro:6502 | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| mmlu_pro:6502 | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| mmlu_pro:10442 | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| mmlu_pro:10442 | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| mmlu_pro:10442 | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/intermediate_algebra/960.json | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/intermediate_algebra/960.json | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/intermediate_algebra/960.json | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/number_theory/769.json | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/number_theory/769.json | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/number_theory/769.json | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/counting_and_probability/870.json | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/counting_and_probability/870.json | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/counting_and_probability/870.json | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/geometry/965.json | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/geometry/965.json | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/geometry/965.json | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/precalculus/986.json | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/precalculus/986.json | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| math_500:test/precalculus/986.json | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_d | GPT | other | invalid_provider_response: Provider response did not contain non-empty text output |
| livecodebench:arc196_d | Claude | other | invalid_provider_response: Provider response did not contain non-empty text output |
| livecodebench:arc196_d | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_d | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_d | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_c | GPT | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_c | Claude | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_c | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_c | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_c | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_b | GPT | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_b | Claude | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_b | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_b | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_b | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_a | GPT | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_a | Claude | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_a | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_a | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:arc196_a | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:abc400_g | GPT | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:abc400_g | Claude | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:abc400_g | Grok | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:abc400_g | DeepSeek | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |
| livecodebench:abc400_g | Qwen | skipped historical row | model_disabled_after_provider_failure: No call made after an earlier provider/model failure in this run. |

Current safety logic identifies 51 eligible pending pairs and no persistent invalid-model/configuration blocks. The 24 paid responses remain terminal, including incorrect responses. Qwen’s non-retryable empty-response errors are not retried within that request, but their pairs are eligible on resume. Up to 153 attempts could be needed at two retries per pair. A $2 total cap includes the existing $0.3449374, leaving $1.6550626; guards may stop early and completion is not guaranteed.

Preflight only (no provider calls; not executed here):

```bash
python3 generate_dataset.py --hard-pilot --resume --max-spend-usd 2.00 --output data/results/hard_pilot_results.csv
```

After explicit later authorization to spend, the exact resume command is:

```bash
python3 generate_dataset.py --hard-pilot --resume --max-spend-usd 2.00 --output data/results/hard_pilot_results.csv --confirm
```

From a fresh clone, first restore a working resume CSV without overwriting any existing live results: `mkdir -p data/results` then `cp -n reports/snapshots/hard_pilot_final.csv data/results/hard_pilot_results.csv`. Never pass the frozen report snapshot as the live output.

**Decision and reproducibility**

Evidence that routing could help: partial disagreement exists and the reduced cost oracle leaves modest theoretical savings, but five-model complementary strengths cannot be assessed; the available accuracy evidence supports choosing Claude statically. Evidence that a learned router could generalize: none. Fifteen prompts cannot establish generalization, and no held-out learned policy was evaluated. Defer ML; resolve evaluation coverage before revisiting the decision. The existing analyzer’s broad `evidence_of_routing_signal` flag also counts static cheaper-model wins and is not used as the ML decision gate.

Test verification: {"checks": ["standalone bold parser accepts only an unambiguous valid option", "audit reconstructs original CSV with matching SHA-256", "offline regrade changes only parsing/grading fields", "analysis and all reports reproduce byte-for-byte without data/ or credentials", "report references tracked snapshot and verifies SHA-256", "tampered snapshot fails hash verification"], "offline_test_command": "python3 -m unittest discover -s tests -v", "status": "passed", "tests_passed": 84}.

From the repository root of a fresh clone, using Python 3.9+ (standard library is sufficient for these commands):

```bash
python3 pilot_analysis.py reports/snapshots/hard_pilot_final.csv
python3 reports/reproduce_hard_pilot.py
python3 -m unittest discover -s tests -v
```

The report command verifies the snapshot SHA and regenerates all three analysis files deterministically. It reads the tracked snapshot, provenance and fixed benchmark plan; it never reads the moving source, loads credentials, calls providers, or rewrites the snapshot. The provenance audit reconstructs the before-regrade CSV for verification. Keep this snapshot immutable; future paid outcomes belong to a separately versioned artifact.
