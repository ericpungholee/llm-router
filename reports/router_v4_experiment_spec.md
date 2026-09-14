# Router v4: fixed-reference routing with source-domain validation

Frozen before v4 fitting or policy selection. This is a separate, bounded
follow-up to v3, authorized by the user's request to continue and publish the
completed work. It does not modify processed data, canonical splits, provider
infrastructure, v1/v2/v3 results, or the candidate model pool. No API evaluation,
downloads, embeddings, new representations, or external data are used.

## Question and interpretation

V3 revealed that selecting the quality reference from non-code validation can
select Qwen, leaving a router that is trivially equal to its weak reference on
unseen code. Freeze the deployment-quality reference to the benchmark's
`gpt-5` before this experiment. It is not GPT-5.6 Sol. Test whether a comparative
TF-IDF router, selected using held-out source domains as well as ordinary
validation, reduces recorded inference cost relative to this fixed reference.

All benchmark outcomes have been inspected in earlier work. This campaign is
exploratory and cannot supply independent confirmation. Source-domain validation
is additional evidence for selection, not a guarantee under arbitrary shift.
Do not change the reference, search grid, fit parameters, selection conditions,
or declared primary after inspecting v4 results.

## Fixed fits and source-domain validation

Use the exact canonical standard 6,094/1,307/1,305 and OOD 6,503/1,148/1,055
partitions. Run all standard work first, then fit the OOD system independently.
Use prompt-only TF-IDF with the original parameters and eight L2/liblinear
logistic regressions, C=1, seed 3407, no class weighting. C=1 is fixed from v2/v3;
there is no hyperparameter or representation search in v4.

Within each regime:

1. Fit a final vocabulary and eight heads on canonical training only. Do not
   refit on train+validation. Estimate mean model costs on training only.
2. For each dataset represented in that training set, fit a fresh vocabulary
   and eight heads on all other training datasets. Predict the excluded
   dataset's training prompts with this auxiliary model and route using costs
   estimated from its own fitting rows. These held-out predictions create a
   source-domain validation block. Every training prompt occurs exactly once
   as an auxiliary held-out prediction. Groups cannot cross training/held-out
   datasets. No canonical validation/test text enters these auxiliary fits.
3. Use the final model for ordinary canonical validation predictions. These
   predictions form a second validation block, disjoint from source holdouts.

Standard uses six source-domain auxiliary fits; OOD uses five. OOD final and
auxiliary fits/validation contain zero LiveCodeBench prompts. No OOD policy
decision may use standard source-code holdouts or any OOD code outcomes.
There are 13 distinct fits total, plus one complete deterministic replay of
each final fit. Preserve auxiliary models, memberships, predictions and costs.

## Fixed candidates and selection

Five comparative margins: {0, 0.02, 0.05, 0.10, 0.20}, plus static GPT-5 fallback.
For each prompt, choose the training-cheapest model strictly cheaper than GPT-5
whose predicted success is at least GPT-5's prediction plus the margin. Ties:
higher prediction, then stable model ID. If none qualify, choose GPT-5.
No novelty gate, domain-name routing input, or absolute-probability threshold.

For each validation block, compute router-minus-GPT-5 quality for every
candidate, both micro and equal-dataset macro, and separately for every dataset.
Use 2,000 paired whole-group bootstrap replicates within dataset strata with
seed 3407. A replicate uses common resamples for all candidates within a block;
the two blocks have separate draws from the same RNG stream. Calculate the
95th percentile of the maximum studentized centered error across all candidates,
both blocks, and every reported micro/macro/dataset comparison. Lower bounds
are observed differences minus this common critical value times bootstrap SE.
Zero-variance comparisons retain empirical zero width, not a population safety
guarantee. Shared auxiliary training sets create dependence not captured by
this conditional bootstrap; report that limitation.

The primary is the lowest realized-cost ordinary-validation candidate whose
simultaneous lower bounds are >=-0.005 in **every** comparison in both blocks.
Tie breaks: higher ordinary-validation micro quality, then stable policy ID.
Static GPT-5 is always feasible. No requirement forces the selector to save cost.

Save two secondary controls before test evaluation:

- Ordinary-validation-only selector: same candidate grid, same per-dataset
  lower-bound condition, maximum error computed only over that validation block.
  This isolates the effect of source-domain validation.
- Fixed v3 margin with the GPT-5 reference: 0.02 standard, 0.05 OOD. This isolates
  reference stability from the new margin selection. Neither control can replace
  the primary after seeing test outcomes.

Save selected policy, controls, costs, all bounds and hashes before requesting
test outcomes. Save label-free test predictions/actions before evaluation.

## Evaluation and criteria

For the primary, both controls, every fixed candidate, every static model,
always-cheapest, validation-selected best-single (secondary reference), and the
existing hindsight oracle, report quality, cost, cost savings, descriptive
Pareto position, and dataset-macro/per-dataset results. Oracle means cheapest
realized-cost successful model, or charged training-cheapest when unsolved.
Report recovered and remaining oracle gap relative to fixed GPT-5.

For each frozen primary/control, use the original 2,000-replicate paired,
dataset-stratified whole-group test bootstrap, with seed 3407 and no retraining.
Report micro/macro quality-difference and cost-savings 95% intervals. Add
per-dataset primary quality/cost intervals, selection distributions,
dataset-by-model counts, calibration/diagnostic data, and composition checks
excluding SimpleQA and excluding SimpleQA plus MMLU-Pro where applicable.

Retain the four meaningful-routing criteria: savings >=10%, micro and macro
quality 95% lower bounds >=-0.005, and all dataset quality point differences
>=-0.005. Also display the stricter test per-dataset interval condition, without
silently replacing the declared four criteria. Passing quality while falling
back with zero savings is deferral, not learned routing value. Report standard
and OOD independently, including failure. Compare standard and OOD on their
shared 158 code test prompts as a paired, exploratory comparison; also disclose
that their complete test populations and training domains differ.

## Artifacts, tests, publication and stop

Use `artifacts/router_v4/` for source snapshots, prior-report/source hashes,
package versions, split manifests, fitted models, training cost estimates,
label-free predictions/actions, separate evaluation outcomes, validation grids,
selected policies, all bootstrap samples, calibration, and artifact hashes.
Keep large reproducible artifacts ignored by git; commit compact reports,
publication figures, code and tests. Provide offline inference for the frozen
primary, explicitly binding it to the fixed GPT-5 reference.

Test target/source isolation, training-only vocabularies and costs, independent
auxiliary/regime fits, no code access in OOD selection, stable candidate order,
fixed-reference semantics, validation-only selection, fallback feasibility,
bootstrap strata/groups and joint maximum computation, exact model replay,
prediction/outcome alignment, and offline inference reload. Run the full suite.

Stop after this fixed campaign and its checks regardless of outcome. Do not
sweep seeds, add models/features/margins, or reuse test outcomes to produce a
better primary. Publish the completed work to the repository's existing GitHub
origin as the user requested. Any next scientific claim still needs evidence
that these reused benchmark tests cannot supply.
