# xAI cost accounting and hard-pilot spend safety

Offline audit, 2026-09-14. No provider API requests were made. Public official
xAI documentation was read; provider tests use mocks. No generation settings,
model IDs, timeouts, prompts, benchmarks, or registry entries changed.

## Official semantics and the historical discrepancy

xAI says `usage.cost_in_usd_ticks` is the actual charge for one request after
discounts, including caching reductions, token charges, and server-side tool
charges. There are 10,000,000,000 ticks per USD. For example, 37,756,000 ticks
is exactly $0.0037756 before conversion to the existing float response field.
[Official cost tracking](https://docs.x.ai/developers/cost-tracking).

The full Responses reference, obtained from the official page using
`Accept: text/markdown`, defines `max_output_tokens` as the generation ceiling
and explicitly says: “This includes both output and reasoning tokens.”
It labels `usage.output_tokens` as output usage and
`usage.output_tokens_details.reasoning_tokens` as generated reasoning usage.
It also documents `usage.context_details.output_tokens` as completion plus
reasoning in the latest context. These fields describe different telemetry;
the request ceiling is shared. Separate reasoning reporting does not establish
an exception allowing output usage above that shared ceiling.
[Official Responses reference](https://docs.x.ai/developers/rest-api-reference/inference/responses).

Reasoning is billed at the completion token rate.
[Official usage and pricing](https://docs.x.ai/developers/advanced-api-usage/prompt-caching/usage-and-pricing).
Grok 4.6 supports high reasoning effort, lists no inherent text output limit,
and lists a context window. Neither that context size nor a default used when
a request limit is absent establishes a separate enforced billing ceiling for
this experiment.
[Official Grok 4.6 model documentation](https://docs.x.ai/developers/grok-4-6).

Historical Grok rows record output usage of 10,610 (`mmlu_pro:1346`), 10,015
(intermediate algebra), and 5,627 (geometry), with a recorded request setting
of 4,096. They have no stored cost ticks, reasoning counts, echoed response
limit, or raw usage payload. Their recorded setting does not prove the exact
historical wire request or provider enforcement. The cause cannot be determined
offline. Official semantics do not justify explaining these rows as a legitimate
reasoning-token exception. Provider enforcement versus historical request or
usage interpretation remains unresolved; no specific root cause is asserted.

## Implementation and operational policy

Successful xAI normalization converts valid nonnegative integer ticks with
`Decimal`, using float only at the `ProviderResponse` boundary. The response's
`provider_reported_cost_usd` feeds the existing authoritative-cost branch in
`SpendTracker.record_call()` and the CSV's response and cumulative cost fields.
Zero billed cost is authoritative. Missing or invalid ticks are flagged, and
no provider-reported cost is invented; the previous local estimate remains an
estimate. Invalid cost payload contents never enter safe diagnostics.

Diagnostics retain integer ticks, reasoning/output/input counts, total/context
usage, echoed response limit, and server-side tool count when available. They
retain no reasoning text, summaries, or encrypted reasoning. An otherwise
successful response above its requested output limit remains successful and
gets `output_tokens_exceed_requested_limit=true`. Paid raw output is checkpointed
and remains terminal regardless of parsing or grading outcome.

The documented token contract would support the existing formula, **if its
applicability and enforcement for this experiment were established**:

```text
(conservative input tokens * input price + 4096 * output price) / 1,000,000
```

The contradictory historical evidence prevents treating that formula as a
verified monetary bound here. This is an operational unresolved-bound policy
(C); it does **not** claim xAI lacks a documented shared token limit.
`max_call_cost_usd()` now raises `SpendPreflightError` for xAI. Thus aggregate
preflight, per-attempt runtime checks, and failed-attempt reservations cannot
present an invented xAI maximum. Cost ticks arrive after dispatch and cannot
prevent an overshoot. No multiplier, context-size assumption, or unenforced
local authorization amount substitutes for a bound.

`--exclude-xai` is the smallest pilot mechanism: it defers Grok dispatch while
keeping the full five-model matrix and its entire recorded spend. A full paid
resume is blocked while any Grok call is pending. Resuming Grok requires a
separate decision after reconciliation; merely authorizing a numeric amount
locally would not enforce a provider billing limit.

## Working results and retry states

The working CSV remains unchanged, SHA-256:
`8a4e61cae01c8310da4f792f3ef6912e0aa9c047cba506bbdd071d8f738a8179`.
It contains 75 rows: 35 successes, 8 provider errors, and 32 historical skips.

| Model | Successfully graded | Pending | Recorded spend USD |
| --- | ---: | ---: | ---: |
| GPT-5.6 Sol | 10 | 5 | 0.1809000 |
| Claude Opus 5 | 10 | 5 | 0.1924850 |
| Grok 4.6 | 9 | 6 | 0.3508180 |
| DeepSeek V4.1 Flash | 6 | 9 | 0.0269847 |
| Qwen3.8 2.4T-A95B | 0 | 15 | 0.0624060 |
| Total | 35 | 40 | 0.8135937 |

There are no saved matching model/configuration blocks or explicit
`output_limit_exhausted` pair blocks in this CSV. The six historical blank
response failures lack evidence of exhaustion and remain eligible to retry.
Future explicit exhaustion blocks remain limited to the same pair and matching
configuration. Transient network errors and old skipped rows are retryable.
All paid terminal rows remain terminal. Qwen's old affordability rejection does
not persist across runs: billing is rechecked. It is eligible after the reported
recharge; account usability was not tested against OpenRouter.

Excluding Grok leaves 34 dispatch-eligible pairs and at most 102 attempts with
two retries. Their maximum additional estimated spend is $4.486052, rounded up;
with no retries it is $1.4953506. Completion under the recorded $2 cap is
therefore not guaranteed. Each bounded-provider attempt still passes the
existing runtime guard and reserves unknown failed-dispatch cost.

Existing failed/interrupted reservations remain part of the cumulative ledger
even for excluded Grok. No historical paid cost was changed. The earlier
[interruption audit](provider_integration_diagnosis.md) identifies one Grok
dispatch missing from the ledger entirely. Its cost remains unknown: the
CSV contains no billed amount for that dispatch. Preserving recorded reservations
cannot account for an unrecorded dispatch, and the old hypothetical
$0.0299000 is not a verified bound. The $0.8135937 ledger is not a reconciled
invoice total.

## Recommendation and exact next command

Temporarily exclude Grok. The exact safe next step is preflight, without
authorization to dispatch:

```bash
python3 generate_dataset.py --hard-pilot --resume --exclude-xai \
  --max-spend-usd 2.00 \
  --output data/results/hard_pilot_results.csv
```

This command was **not executed** against the working artifact. It stops with
the instruction to pass `--confirm`, before generation or any result write.
The all-five-model command with `--confirm` is not recommended.

Before a paid four-provider resume, reconcile the missing Grok dispatch and
historical spend using existing billing exports. If verified cumulative prior
spend exceeds the recorded ledger, subtract that difference from the CLI cap:
`effective_ledger_cap = 2.00 - max(0, verified_prior_spend - 0.8135937)`.
Retain that adjustment on every later resume, recomputing against the ledger
as necessary; a confirmed command with an unadjusted $2 ledger cap does not
prove an actual $2 cumulative billing cap. Stop if the adjusted cap leaves no
room. This workflow preserves historical paid rows and fabricates no charges.
Once reconciled, use `--exclude-xai` and `--confirm` with the adjusted cap to
authorize the other providers. A paid resume cannot yet be certified against
actual cumulative billing from the available evidence.

## Validation and changed files

Baseline: all 100 existing tests passed. Eleven deterministic tests were added
for exact ticks conversion, authoritative/zero billed cost, diagnostic privacy,
invalid/missing cost metadata, closed xAI preflight/runtime/reservation paths,
full-resume rejection, historical-row preservation, excluded-spend retention,
successful response checkpointing above the token limit, and Qwen retry after
a previous run's billing rejection. Existing mode/interrupt tests were updated
to exercise the explicit exclusion and bounded-provider reservation policy.

Full suite: **111 tests passed**, with provider network access blocked.
`git diff --check` passed. The working CSV and frozen snapshot were not modified.

Files changed: `provider_clients/normalization.py`, `spend_control.py`,
`generate_dataset.py`, `tests/test_provider_diagnostics.py`,
`tests/test_spend_control.py`, `tests/test_execution.py`, `README.md`, and this
report.

Suggested commit title: `fix: account for xAI billed cost and block unbounded Grok spend`
