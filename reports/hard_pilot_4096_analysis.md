# Completed 4096-token hard pilot

**Decision: unsuitable for training or evaluating a learned router.** Systematic budget exhaustion removes gradeable answers by provider and task. This is nonrandom missingness, which can bias complete-case model comparisons toward easy-to-finish prompts. Five common prompts cannot establish routing generalization.

Immutable input: [`reports/snapshots/hard_pilot_4096_final.csv`](snapshots/hard_pilot_4096_final.csv), frozen at 2026-09-14T17:31:34.782432+00:00 from commit `1b7622000cf703708a15ffa0e155a018397646c1`. SHA-256 `f80c25a369eae3ef7ebb25713729c3327c20d6de5d6c4375a2912e86d04b94e7` matches the source `data/results/hard_pilot_results.csv` byte for byte. The original was not modified or regraded. [Provenance](snapshots/hard_pilot_4096_final.provenance.json) pins request adapters, formatting, grading, benchmark inputs, registry IDs/prices/settings, and ledger totals. Freeze and reproduction block network/provider access. No provider calls occurred.

## Final accounting and completeness

| Metric | Value |
| --- | --- |
| Rows / expected pairs | 75 / 75 |
| Attempted provider calls (recorded/inferred history) | 83 |
| Successful normalized provider responses | 45 |
| Successfully graded pairs | 45 |
| Correct / graded | 32 / 45 |
| Accuracy on graded pairs | 71.11% |
| Provider failures, excluding spend stops | 21 |
| Parsing / grading failures | 0 / 0 |
| Skipped / spend stops / unrecorded | 8 / 1 / 0 |
| Output-limit-exhausted final pairs | 20 |
| Fully graded five-model prompts | 5 |
| Recorded spend USD | 1.9491623 |

The existing response counter excludes 20 saved HTTP-200 exhaustion outcomes; those are provider-returned responses but failed normalized completions. The 83 attempted calls include earlier attempts/resumes, with some history inferred from legacy fields. Exhausted/unknown billed failures carry maximum-cost reserves. Recorded spend is a ledger estimate, not a reconciled invoice; see the historical billing limitations in [xAI cost safety](xai_cost_safety.md). All remaining failure/skipped states remain missing grades, never incorrect answers.

| Group | Graded / expected | Correct / graded | Provider fail | Parse / grade fail | Skipped | Spend stop | Spend USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Claude | 10 / 15 | 8 / 10 | 4 | 0 / 0 | 1 | 0 | 0.6472850 |
| DeepSeek | 7 / 15 | 6 / 7 | 7 | 0 / 0 | 1 | 0 | 0.0678333 |
| GPT-5.6 Sol | 12 / 15 | 6 / 12 | 2 | 0 / 0 | 0 | 1 | 0.4873280 |
| Qwen | 7 / 15 | 6 / 7 | 7 | 0 / 0 | 1 | 0 | 0.3958980 |
| Grok | 9 / 15 | 6 / 9 | 1 | 0 / 0 | 5 | 0 | 0.3508180 |

| Group | Graded / expected | Correct / graded | Provider fail | Parse / grade fail | Skipped | Spend stop | Spend USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mmlu_pro | 20 / 25 | 10 / 20 | 5 | 0 / 0 | 0 | 0 | 0.5375859 |
| math_500 | 23 / 25 | 20 / 23 | 2 | 0 / 0 | 0 | 0 | 0.2765056 |
| livecodebench | 2 / 25 | 2 / 2 | 14 | 0 / 0 | 8 | 1 | 1.1350708 |

Per-model accuracies above use different successful subsets. They do not rank performance over the full 15-prompt plan. All cells are recorded; completeness measures grades. Grok's existing results are retained, but Grok was excluded from the latest paid resume and is excluded from treatment.

## Five-model common subset

**Only 5 fully graded prompts exist.** Correctness vectors (1 correct, 0 incorrect):

| Prompt | Claude | DeepSeek | GPT-5.6 Sol | Qwen | Grok |
| --- | --- | --- | --- | --- | --- |
| math_500:test/counting_and_probability/870.json | 1 | 1 | 0 | 1 | 1 |
| math_500:test/number_theory/769.json | 1 | 1 | 1 | 1 | 1 |
| math_500:test/precalculus/986.json | 1 | 1 | 0 | 1 | 1 |
| mmlu_pro:10442 | 1 | 1 | 1 | 1 | 1 |
| mmlu_pro:1346 | 1 | 1 | 1 | 1 | 0 |

| Static model | Correct / n | Accuracy | Response cost USD |
| --- | --- | --- | --- |
| Claude | 5 / 5 | 100% | 0.020495 |
| DeepSeek | 5 / 5 | 100% | 0.0029718 |
| GPT-5.6 Sol | 3 / 5 | 60% | 0.010580 |
| Qwen | 5 / 5 | 100% | 0.045974 |
| Grok | 4 / 5 | 80% | 0.108568 |

Five-model oracle: 100%; cheapest-correct oracle cost: **$0.0029718** across all five (no unsolved prompts). Best static accuracy is 100%, tied among Claude, DeepSeek, and Qwen; the deterministic provider/ID tie-break selects Claude ($0.0204950). DeepSeek is also 100% and is the cheapest static ($0.0029718); the oracle saves nothing and adds no accuracy relative to static DeepSeek. Costs use response costs, excluding preceding failure reserves. 3/5 prompts show correctness disagreement.

**No prompt-visible routing baseline can be meaningfully evaluated here.** A benchmark rule could be computed on these same five selected prompts, but there is no held-out split, no common code prompt, and no oracle improvement over static DeepSeek. The generic analysis flag for cost-order inversions is descriptive and does not establish actionable routing signal. Freeze this as a feasibility experiment; do not train ML.

## Pairwise common-prompt comparisons

Each comparison uses its own jointly graded overlap; denominators differ and selected subsets remain biased. Wins count correctness disagreements, not answer-text differences.

| Pair A / B | n | A accuracy | B accuracy | A-only / B-only correct | Both correct / wrong |
| --- | --- | --- | --- | --- | --- |
| Claude / DeepSeek | 7 | 85.7% | 85.7% | 0 / 0 | 6 / 1 |
| Claude / GPT-5.6 Sol | 10 | 80.0% | 40.0% | 4 / 0 | 4 / 2 |
| Claude / Qwen | 7 | 85.7% | 85.7% | 0 / 0 | 6 / 1 |
| Claude / Grok | 9 | 77.8% | 66.7% | 1 / 0 | 6 / 2 |
| DeepSeek / GPT-5.6 Sol | 7 | 85.7% | 57.1% | 2 / 0 | 4 / 1 |
| DeepSeek / Qwen | 5 | 100.0% | 100.0% | 0 / 0 | 5 / 0 |
| DeepSeek / Grok | 7 | 85.7% | 71.4% | 1 / 0 | 5 / 1 |
| GPT-5.6 Sol / Qwen | 7 | 42.9% | 85.7% | 0 / 3 | 3 / 1 |
| GPT-5.6 Sol / Grok | 9 | 44.4% | 66.7% | 1 / 3 | 3 / 2 |
| Qwen / Grok | 7 | 85.7% | 71.4% | 1 / 0 | 5 / 1 |

Correctness disagreement matrix: disagree / jointly graded n. The JSON also includes each pair's prompt IDs, per-benchmark overlaps, wins, and rates.

| Model | Claude | DeepSeek | GPT-5.6 Sol | Qwen | Grok |
| --- | --- | --- | --- | --- | --- |
| Claude | 0 / 10 | 0 / 7 | 4 / 10 | 0 / 7 | 1 / 9 |
| DeepSeek | 0 / 7 | 0 / 7 | 2 / 7 | 0 / 5 | 1 / 7 |
| GPT-5.6 Sol | 4 / 10 | 2 / 7 | 0 / 12 | 3 / 7 | 4 / 9 |
| Qwen | 0 / 7 | 0 / 5 | 3 / 7 | 0 / 7 | 1 / 7 |
| Grok | 1 / 9 | 1 / 7 | 4 / 9 | 1 / 7 | 0 / 9 |

## Output exhaustion

| Model | Exhausted |
| --- | --- |
| Claude | 4 |
| DeepSeek | 7 |
| GPT-5.6 Sol | 2 |
| Qwen | 7 |
| Grok | 0 |

| Benchmark | Task type | Exhausted |
| --- | --- | --- |
| livecodebench | code | 14 |
| mmlu_pro | multiple_choice | 4 |
| math_500 | math | 2 |

All final exhausted pairs are listed below. Counts/tags only; no reasoning contents. Code accounts for 14/20 (70%); DeepSeek and Qwen contribute 14/20. Claude's four exhausted pairs and GPT's two are all code. DeepSeek also exhausts two multiple-choice prompts and one math prompt; Qwen exhausts two multiple-choice and one math prompt. This is concentrated in LiveCodeBench, with provider-specific missingness elsewhere. The missing `abc400_g` cells and Grok skips are separate causes, not inferred exhaustion.

| Prompt | Model | Benchmark / task | Requested | Output | Reasoning | Stop | Visible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| livecodebench:arc196_a | Claude | livecodebench / code | 4096 | 4096 | 4096 | max_tokens | False |
| livecodebench:arc196_a | DeepSeek | livecodebench / code | 4096 | 4096 | 4096 | length | False |
| livecodebench:arc196_a | Qwen | livecodebench / code | 4096 | 4096 | 4096 | length | False |
| livecodebench:arc196_b | Claude | livecodebench / code | 4096 | 4096 | 4096 | max_tokens | False |
| livecodebench:arc196_b | DeepSeek | livecodebench / code | 4096 | 4096 | 4096 | length | False |
| livecodebench:arc196_b | Qwen | livecodebench / code | 4096 | 4096 | 4096 | length | False |
| livecodebench:arc196_c | Claude | livecodebench / code | 4096 | 4096 | 4096 | max_tokens | False |
| livecodebench:arc196_c | DeepSeek | livecodebench / code | 4096 | 4096 | 4096 | length | False |
| livecodebench:arc196_c | GPT-5.6 Sol | livecodebench / code | 4096 | 4096 | 4096 | incomplete / max_output_tokens | False |
| livecodebench:arc196_c | Qwen | livecodebench / code | 4096 | 4096 | 3812 | length | False |
| livecodebench:arc196_d | Claude | livecodebench / code | 4096 | 4096 | 4096 | max_tokens | False |
| livecodebench:arc196_d | DeepSeek | livecodebench / code | 4096 | 4096 | 4096 | length | False |
| livecodebench:arc196_d | GPT-5.6 Sol | livecodebench / code | 4096 | 4096 | 4096 | incomplete / max_output_tokens | False |
| livecodebench:arc196_d | Qwen | livecodebench / code | 4096 | 4096 | 4116 | length | False |
| math_500:test/geometry/965.json | Qwen | math_500 / math | 4096 | 4096 | 4096 | length | False |
| math_500:test/intermediate_algebra/960.json | DeepSeek | math_500 / math | 4096 | 4096 | 4096 | length | False |
| mmlu_pro:12015 | DeepSeek | mmlu_pro / multiple_choice | 4096 | 4096 | 4096 | length | False |
| mmlu_pro:12015 | Qwen | mmlu_pro / multiple_choice | 4096 | 4096 | 2161 | length | False |
| mmlu_pro:5022 | Qwen | mmlu_pro / multiple_choice | 4096 | 4096 | 4096 | length | False |
| mmlu_pro:6502 | DeepSeek | mmlu_pro / multiple_choice | 4096 | 4096 | 4096 | length | False |

All 20 report output_tokens=4096 and no visible text. Reasoning diagnostics are provider telemetry: Qwen reports 2161, 3812, and 4116 on three pairs; 4116 exceeds its inclusive output count and is internally inconsistent. Do not add reasoning counts again to output usage or infer hidden contents. Budget-exhausted runs only establish a lower bound on required completion budget; they do not identify the amount needed to finish.

The pilot uniformly records a requested 4096 cap, but the two Qwen successes above that cap mean a uniformly enforced 4096 billed-output treatment is not established. Preserve those successful observations as recorded; this additional contract inconsistency also limits cross-provider budget comparisons.

## Treatment recommendation and strict cost preflight

Recommend **8192** as the smallest useful feasibility treatment, not a promise of sufficiency. It doubles the observed exhausted ceiling; GPT already finishes two other code tasks within 4096, and Qwen has completed two answers with reported usage 5559/6298, below 8192. Neither these different prompts nor right-censored failures prove 8192 sufficient for hard code. 16384 doubles the incremental capped output cost without an observed completion threshold requiring it. Consider a separately preregistered 16384 experiment only after inspecting 8192 recovery; never auto-escalate or retry.

OpenAI documents that reasoning can consume the entire response budget before visible output; its broad starting guidance reserves 25,000 tokens, so neither candidate is universally guaranteed. This study deliberately tests a smaller budget at unchanged effort. [OpenAI reasoning guidance](https://developers.openai.com/api/docs/guides/reasoning#allocating-space-for-reasoning). Claude's hard request cap includes thinking and answer text, and high effort can exhaust it. [Claude steering and cost](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost). DeepSeek exposes thinking separately and documents the completion token ceiling. [DeepSeek thinking](https://api-docs.deepseek.com/guides/thinking_mode/), [completion reference](https://api-docs.deepseek.com/api/create-chat-completion/). OpenRouter bills reasoning as output even when excluded from the returned text. [OpenRouter reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens). Public docs were inspected without provider API calls.

Select exactly the 20 explicitly exhausted final pairs (GPT 2, Claude 4, DeepSeek 7, Qwen 7), preserving prompt bytes, model IDs, temperature/default, reasoning settings, request adapters, benchmark inputs and graders. Only max output tokens changes. One client provider attempt per pair, zero retries for any error, no Grok, fresh output and exclusive execution sidecar. Existing outputs cannot be overwritten or resumed. OpenRouter upstream choice/failover remains its existing behavior; one client HTTP attempt cannot prove one upstream backend attempt, and unpinned upstream variation remains a scientific limitation.

| Budget | Calls / maximum client attempts | Registry-contract maximum USD | Verified worst case USD |
| --- | --- | --- | --- |
| 8192 | 20 / 20 | 1.656349 | N/A |
| 16384 | 20 / 20 | 3.216105 | N/A |

| Model | Calls | 8192 contract USD | 16384 contract USD |
| --- | --- | --- | --- |
| Claude | 4 | 0.864400 | 1.683600 |
| DeepSeek | 7 | 0.0727881 | 0.1416009 |
| GPT-5.6 Sol | 2 | 0.346468 | 0.674148 |
| Qwen | 7 | 0.372692 | 0.716756 |

Formula per actual selected pair: `(UTF-8 prompt byte length × registry input price + budget × registry output price) / 1,000,000`, summed with Decimal and rounded upward to six decimal places. UTF-8 bytes conservatively bound user-text tokenization; existing prices are uncached standard/peak rates. No reasoning double-count, no retries, no xAI. Costs are incremental to the frozen ledger. Registry-contract totals assume the documented inclusive output cap applies to billed usage and existing input-overhead assumptions.

**Strict preflight is currently blocked for both candidates.** Qwen's new saved telemetry contradicts the output ceiling:

| Pair | Requested | Output | Reasoning |
| --- | --- | --- | --- |
| mmlu_pro:1346 | 4096 | 5559 | 5555 |
| math_500:test/intermediate_algebra/960.json | 4096 | 6298 | 6290 |

OpenRouter documents a generation ceiling and bills reasoning as output; the current data do not establish whether backend enforcement, request mapping, or usage interpretation caused this discrepancy. [OpenRouter parameters](https://openrouter.ai/docs/api_reference/parameters). A historical maximum, an arbitrary multiplier, the model/context limit, or a local USD cap cannot repair an unverified billing bound. Thus $1.656349/$3.216105 are conservative **contract-based estimates**, not verified worst-case spend. No finite verified maximum for the complete 20-call treatment can be certified from the available data. Reconcile exact request forwarding and billed inclusive usage using existing offline request/billing exports or provider confirmation before enabling paid execution. Resolve bound assumptions and pin the resulting evidence in code/provenance; there is deliberately no override flag. The analysis is complete despite this execution gate.

## Preregistered success criteria

- At least 16/20 previously exhausted pairs (80%) return visible, parseable, successfully graded answers, counting correct and incorrect grades as successful recovery. Keep visible-answer rate and gradeability rate separate; report parsing/grading failures explicitly.
- At least 70% gradeability recovery within each provider (GPT 2/2, Claude 3/4, DeepSeek 5/7, Qwen 5/7). Report residual exhaustion, network failures, and any budget-usage inconsistency by provider and benchmark; small counts cannot prove absence of clustering.
- At least ten complete four-provider prompts in a separate read-only recovery view, including at least two code prompts. Report five-model completeness separately; Grok is still deferred. This is feasibility coverage, not enough training data.
- Only justify scaling to a larger stratified evaluation after recovery/coverage criteria and strict cost checks pass and no unresolved cap anomaly remains. Establish held-out splits and a uniform-budget collection policy before learned-router evaluation. Failed criteria trigger diagnosis and a newly preregistered treatment, not automatic spend or retries.

Even perfect recovery yields at most 9 five-model and 14 four-provider complete prompts: untouched Grok failures/skips and non-exhaustion stops limit coverage. Recovery views must label reused 4096 outcomes and new higher-budget outcomes separately. A selectively repaired matrix is not a uniform 8192 evaluation and must not be silently merged into either experiment. A larger dataset needs an outcome-independent budget policy.

## Reproduction and exact commands

Offline reproduction from a fresh checkout requires no ignored `data/`, credentials, or network:

```bash
python3 reports/reproduce_4096_pilot.py --output-dir /tmp/hard-pilot-4096-report
```

Exact 8192 preflight (currently exits blocked, writes no treatment file):

```bash
python3 hard_pilot_treatment.py --max-output-tokens 8192 --max-spend-usd 1.66 --output data/results/hard_pilot_exhaustion_8192.csv
```

Exact paid command, **not executed and currently blocked before dispatch**:

```bash
python3 hard_pilot_treatment.py --max-output-tokens 8192 --max-spend-usd 1.66 --output data/results/hard_pilot_exhaustion_8192.csv --confirm
```

The 16384 alternative uses `--max-output-tokens 16384 --max-spend-usd 3.22 --output data/results/hard_pilot_exhaustion_16384.csv`; it is also blocked. No paid treatment has run. The test suite exercises only mocked paid paths and blocks real HTTP/socket access.
