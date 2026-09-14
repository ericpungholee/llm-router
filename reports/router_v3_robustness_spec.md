# Router v3: frozen TF-IDF cross-fitting robustness check

This protocol is written before any v3 fits. It checks the already-selected v2
comparative TF-IDF rule on the existing 8,706-prompt benchmark. It does not select
a replacement router, acquire data, change canonical splits, modify providers,
call APIs, or fit additional representations. V1 and v2 remain unchanged.

## Interpretation and fixed hypotheses

The benchmark outcomes have already been inspected. New partitions cannot make
these data independent confirmation. This is a descriptive robustness check of
a fixed algorithm, with held-out predictions for every prompt and broader math
and code coverage than the original standard test. It cannot remove adaptation
to this benchmark. External confirmation requires genuinely unseen outcomes.

Freeze C=1, seed 3407, the original TF-IDF/logistic parameters and eight-model
order. Do not tune C again. Freeze the v2 selected comparative margins: 0.02 for
standard and 0.05 for OOD, with no novelty gate. Choose the cheapest model whose
training mean cost is strictly below the validation-selected best-single and
whose predicted success exceeds or equals its prediction plus the fixed margin.
Use the existing probability and model-ID tie breaks; otherwise use best-single.
There is no policy grid, validation acceptance filter, or test-driven fallback.

## Fixed partitions and fitting

Sort prompt IDs. Create five dataset-stratified, whole-leakage-group outer folds
using StratifiedGroupKFold(shuffle=True, random_state=3407). Each outer fold is
held out exactly once. On the other four folds, create five analogous inner
folds with the same seed and take inner fold zero as validation. This produces
approximately 64% fitting / 16% validation / 20% evaluation, with exact counts
saved. These are separate experimental assignments, never edits to processed
standard/ood split columns. The reduced training size is part of interpretation.

Fit each standard vocabulary and eight heads from scratch on its fitting rows.
Estimate mean model costs only from those rows. Select best-single using only
that fold's validation quality, ties by training cost and stable ID. Freeze all
choices before loading the fold's evaluation labels/costs. Save probabilities
and actions before outcome evaluation. No train+validation refit.

After all standard folds finish, fit five OOD systems from scratch. For fold k,
use the same train/validation assignments with every LiveCodeBench row removed;
evaluate only the code rows in its outer fold. Every code prompt therefore has
one standard and one independent OOD-model prediction. OOD models never see
any code text or labels in fitting or validation. Retaining the paired outer
partitions also withholds non-code outer rows, so OOD training sets are smaller
than v2's 6,503 prompts. This check estimates the fixed algorithm under these
training sizes; it is not a replay of the canonical v2 system.

## Comparators and evaluation

Report pooled out-of-fold micro and equal-dataset macro quality/cost, each fold,
each of six datasets, model selection distributions, classifier diagnostics,
and calibration. Comparators are fold-validation-selected best-single,
training-cheapest, each of eight static models, and the existing hindsight
oracle (cheapest realized-cost successful model; training-cheapest if unsolved).
The varying fold-best reference is explicitly not one globally fitted model.

Retain one secondary privileged domain-static reference: select the best model
within each fold's validation dataset, ties by training mean cost and stable ID;
unseen datasets use global validation best-single. This requires the dataset
name and cannot replace the prompt-only router. Report its quality/cost and a
paired comparison; no selection among these references is allowed.

Use 2,000 seed-3407 paired whole-group percentile bootstrap replicates within
dataset strata for router-minus-best quality and relative cost savings, both
micro and macro; also report per-dataset intervals and the matched code
standard/OOD comparison. Save samples. These intervals condition on all fitted
fold models and frozen actions. Overlapping training sets create dependencies
between out-of-fold predictions, and the bootstrap does not capture refitting
or model-selection uncertainty. Label intervals descriptive, not independent
generalization guarantees. Report fold ranges without treating folds as
independent replicates. Do not retrain inside the bootstrap.

Preserve the v2 descriptive checks: pooled savings >=10%, micro and macro
quality CI lower bounds >=-0.005, and every dataset point delta >=-0.005. Report
each separately, plus how many folds meet the quality point margin and savings
threshold. Do not weaken a failed condition. Report metrics excluding SimpleQA,
and excluding both SimpleQA and MMLU-Pro, to expose composition dependence.
Show the primary's descriptive Pareto position against the fixed comparators;
never promote an alternative using evaluation results. Oracle-gap recovery
may be negative and is undefined if the denominator is zero.

## Reproducibility and stop

Save source/data hashes before fitting, immutable partition assignments, each
fold's model, vocabulary hash, costs, baseline choices, label-free predictions
and decisions, separate evaluation outcomes, bootstrap samples and summaries,
package versions, and a compact report. Replay the complete first fold fit in
each regime and require identical parameters and predictions. Refuse overwrite
of a completed or partially started campaign. Run all existing tests plus checks
for group separation, exactly-once coverage, train-only inputs/costs, independent
OOD fits, validation-only baselines, saved outcome alignment, and replay.

Stop after these ten fits, fixed replays, evaluation and tests, regardless of
whether the descriptive checks pass. Do not sweep seeds, margins, features,
representations, or partitions to improve a reported result.
