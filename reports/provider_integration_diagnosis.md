Offline diagnosis of the interrupted hard-pilot resume
====================================================

The paid pilot was not resumed. No provider API calls were made. Official public
documentation was consulted, and HTTP responses were mocked in all adapter tests.
This is a working-state diagnosis, not a new frozen experiment artifact.

The working CSV remains byte-for-byte unchanged with SHA-256
`8a4e61cae01c8310da4f792f3ef6912e0aa9c047cba506bbdd071d8f738a8179`.
The frozen snapshot remains
`a58e79eb58519bb4a170142a4c1b60bbbc799fb671d40eb512488bfb52becf85`.
Every original field of all 24 frozen successful rows matches the working CSV.
All 35 current successful responses were preserved. No model IDs, model set,
prompts, reasoning effort, output limits, or timeout values were changed.

Current results
---------------

There are **35/75 graded pairs (46.67%)**, **40 pending**, and **11 additional
successful grades** compared with the frozen 24/75 snapshot. Of those 11 new
grades, 10 are correct and one is incorrect. Overall, 23 of 35 graded answers
are correct. “Successfully graded” includes an answer graded incorrect.
The pending rows comprise 8 provider errors and 32 skipped rows.

| Model | Graded | Pending | Recorded spend |
| --- | ---: | ---: | ---: |
| GPT-5.6 Sol | 10 | 5 | $0.1809000 |
| Claude Opus 5 | 10 | 5 | $0.1924850 |
| Grok 4.6 | 9 | 6 | $0.3508180 |
| DeepSeek V4.1 Flash | 6 | 9 | $0.0269847 |
| Qwen3.8 2.4T-A95B | 0 | 15 | $0.0624060 |
| Total | 35 | 40 | **$0.8135937** |

The ledger increased by **$0.4686563** from $0.3449374. It combines priced
responses and conservative failed-attempt reserves; it is not an invoice.
It records 52 historical provider attempts versus 31 in the snapshot, excluding
the user-reported uncheckpointed Grok dispatch. Thus at least one additional
possibly billed attempt is missing from the ledger. Its actual cost cannot be
recovered from this CSV. No historical cost or telemetry was backfilled.

The interrupted request is consistent with Grok on `livecodebench:arc196_d`, the
next pair after the final saved Claude failure in the sequential runner. This is
an inference from runner order and the user's report, not a recorded request ID.
The existing reservation formula for that pair gives $0.0299000, which would
yield a ledger of $0.8434937. That is only a hypothetical accounting adjustment,
not a verified bill or maximum: the xAI bound problem below remains unresolved.

| Prompt | Newly graded models | Correctness |
| --- | --- | --- |
| `mmlu_pro:6502` | Grok | Incorrect |
| `mmlu_pro:10442` | Grok, DeepSeek | Both correct |
| `math_500:test/intermediate_algebra/960.json` | Grok | Correct |
| `math_500:test/number_theory/769.json` | Grok, DeepSeek | Both correct |
| `math_500:test/counting_and_probability/870.json` | Grok, DeepSeek | Both correct |
| `math_500:test/geometry/965.json` | Grok | Correct |
| `math_500:test/precalculus/986.json` | Grok, DeepSeek | Both correct |

No additional OpenAI, Anthropic, or Qwen grades were obtained during this resume.

Root causes and limits of the evidence
--------------------------------------

The confirmed adapter defect was requiring visible text before retaining stop
reasons and usage. An empty answer consequently became
`invalid_provider_response`, with zero placeholder token counts and no response
ID. Those zeros do not establish that the provider generated zero tokens. The
original responses were discarded, so historical diagnosis is necessarily
uncertain.

| Pattern | Current evidence and likely explanation |
| --- | --- |
| OpenAI blank on `livecodebench:arc196_d` | Plausible output-budget exhaustion; not proven. Responses can end `incomplete` with `incomplete_details.reason=max_output_tokens` before producing visible text, while still billing reasoning tokens. [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning#allocating-space-for-reasoning). |
| Anthropic blank on `livecodebench:arc196_d` | Plausible exhaustion of the shared thinking/output budget; not proven. Claude documents missing text when thinking consumes `max_tokens`. [Claude thinking troubleshooting](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting). |
| DeepSeek blanks on `mmlu_pro:12015`, `mmlu_pro:6502`, `math_500:test/intermediate_algebra/960.json`, and `math_500:test/geometry/965.json` | Plausible reasoning-only generation reaching the ceiling; not proven. Reasoning is separate from final content, which can be null. `finish_reason=length` supplies the exhaustion signal; reasoning presence alone does not. [Thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/) and [Chat Completions reference](https://api-docs.deepseek.com/api/create-chat-completion/). |
| Grok on `mmlu_pro:12015` | Confirmed read timeout, not a blank response. Three attempts in this resume took approximately 1,803 seconds including backoff. Its cumulative count of four includes one earlier attempt. Server-side cause is unknown. |
| Qwen on `mmlu_pro:1346` | Confirmed affordability rejection: requested 4,096 tokens, could afford 2,736. Non-transient billing/configuration failure. Its cumulative three attempts include two historical attempts; this resume made one. [OpenRouter errors](https://openrouter.ai/docs/api_reference/errors-and-debugging). |
| User-interrupted Grok request | Confirmed user interruption; completion and billing unknown. No saved row proves its final provider outcome. |

None of the six historical blank rows proves a genuinely malformed JSON body,
normal empty completion, or refusal. They remain ambiguous integration outcomes.
The fixtures establish how future responses will be distinguished; they are not
reconstructions of the missing historical responses.

The 4,096 ceiling appears plausibly insufficient for some DeepSeek prompts and
the OpenAI/Claude code prompt, but this dataset cannot confirm it. OpenAI's
general experimentation guidance suggests reserving at least 25,000 tokens for
reasoning and output; that is context for a later study, not a setting applied
to this pilot. [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning#allocating-space-for-reasoning).

An additional xAI accounting concern is directly visible: Grok returned 10,610
output tokens on `mmlu_pro:1346` in the older run, and 10,015 and 5,627 on the
intermediate-algebra and geometry prompts during the latest resume, all with
`max_output_tokens=4096` recorded. Its reported response costs exceed the
existing per-call preflight bounds for those pairs. xAI exposes separate
reasoning usage metadata, which the adapter now preserves, but its presence
does not explain these historical discrepancies. [xAI reasoning](https://docs.x.ai/developers/model-capabilities/text/reasoning)
and [Responses reference](https://docs.x.ai/developers/rest-api-reference/inference/responses).
The CSV cannot distinguish a provider limit-enforcement problem from different
limit/usage semantics. Therefore, fixing lost interruption accounting does not
establish that the existing 4,096-based reservation is a true xAI billing bound.

Implementation
--------------

`call_model_with_retries()` catches `KeyboardInterrupt` only around the active
call. It invokes the failed-attempt callback with `interrupted_inflight`, reserves
the same cost as an unknown failed dispatch, preserves attempt counts, and
checkpoints through the existing atomic CSV writer before re-raising the
interrupt. The row has null score/correctness, pair scope, and retryable metadata.
An explicit later resume retains its cost and attempts. Interruptions before
dispatch or during retry backoff do not create phantom dispatch reservations.

All adapters now use shared response normalization. Metadata includes status,
finish/stop/incomplete reasons, input/output/reasoning token counts when provided,
item/content types, reasoning/refusal presence, response/request IDs, and an
indicator when reported output usage exceeds the requested ceiling. Transport
failures retain phase, exception type, timeout flag, elapsed time, configured
timeout, and HTTP status/request ID when available. xAI remains high effort with
the existing 600-second timeout.

The optional `provider_diagnostics` CSV column contains a JSON history of safe
metadata per attempt, retained across retries and resumes. It contains no
thinking, reasoning content, summaries, encrypted reasoning, or raw refusal text.
Malformed JSON bodies are no longer copied into error messages. Historical CSVs
without the new field still load and frozen reports still reproduce.

Empty-output classifications now distinguish `output_limit_exhausted`,
`empty_completion`, `reasoning_without_answer`, `refusal`,
`incomplete_response`, `failed_provider_response`, and genuinely malformed
`invalid_provider_response`. Network failures remain `network_error`.
New response outcomes reserve conservative failed-attempt costs even though
they are non-transient. Their reported usage is retained as diagnostic evidence.
Budget exhaustion stops retries immediately and persists a block only for the
same pair/configuration, including CLI preflight. It does not disable another
prompt or provider. A different configuration releases that pair block; existing
successes remain terminal. OpenRouter affordability errors remain non-retryable
and block the model for the current run; an explicit future run may recheck
billing after credits change.

Files changed
-------------

- `provider_clients/base.py`: diagnostics, transport errors, interruption metadata, billing classification.
- `provider_clients/normalization.py`: new shared response normalization.
- `provider_clients/openai.py`, `anthropic.py`, `deepseek.py`, `xai.py`, `openrouter.py`: use normalization; request payloads unchanged.
- `providers.py`: interruption callback and re-raise.
- `generate_dataset.py`: conservative accounting, diagnostic history, persistent pair/configuration exhaustion blocks.
- `dataset_schema.py`: optional diagnostics column and stable legacy required fields.
- `tests/test_execution.py`: use explicit legacy schema fields.
- `tests/test_provider_diagnostics.py`: 16 new deterministic regression tests.
- `tests/fixtures/provider_responses.json`: seven deterministic response fixtures.
- `reports/inspect_interrupted_resume.py`: read-only comparison command.
- `reports/provider_integration_diagnosis.md`: this diagnosis.

Validation and next step
------------------------

`python3 -m unittest discover -s tests -v` passed **100/100 tests** (84 existing,
16 added). Tests cover all requested provider fixtures, genuine malformed output,
refusal, normal empty completion, reasoning without final output, timeout
diagnostics, Qwen's one-dispatch rejection, interruption during HTTP read,
cumulative reservations, cap enforcement after resume, successful later explicit
resume, pre-dispatch/backoff interruption, diagnostic privacy, and configuration
blocks. Existing frozen-artifact hash and report reproduction tests also pass.
`git diff --check` passes.

Keep this pilot's configuration unchanged and paused. First reconcile the missing
historical Grok charge and investigate its limit/usage semantics using available
billing exports or documentation; do not assume the current xAI bound is safe.
Any later paid diagnostic run requires explicit authorization. For that later
stage, retain 4,096 and the current effort initially to collect the missing
metadata on a small selection of failed pairs, with no automatic retries for
exhaustion. If exhaustion is confirmed, design a separate experiment with a
larger shared output budget (for example, a 16,384-token treatment), unchanged
effort/prompts/IDs, and a recalculated spend limit. Do not merge that treatment
into the current matrix or claim equivalence with the 4,096 configuration.

The exact safe next command is read-only, uses no credentials, writes no results,
and makes no provider calls:

```sh
python3 reports/inspect_interrupted_resume.py
```
