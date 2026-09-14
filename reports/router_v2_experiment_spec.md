# Router v2: frozen local continuation protocol

Written before fitting v2 or evaluating its test outcomes. The user authorized
continuing after the failed v1 baseline. The processed dataset, eight candidates,
split assignments, labels, costs, seed 3407, and v1 results remain unchanged.
This is a new experiment, not a revision of the v1 outcome or its frozen spec.

## Research questions and interpretation

1. Does a fixed local semantic representation improve success prediction and
   routing over TF-IDF and the 16 handcrafted features?
2. Can comparative routing, with validation uncertainty and domain safeguards,
   reduce inference cost without sacrificing micro or macro quality?
3. Does a training-derived novelty gate help the independent OOD system defer
   unfamiliar prompts to its validation-selected best-single model?

Existing standard/OOD tests have already been inspected in v1. Every v2 test
result is therefore **exploratory**, even with a frozen new protocol. No new
test result selects features, C, a policy, a gate threshold, or a checkpoint.
Repeatedly searching these tests until one passes would not be confirmation.
No new dataset acquisition, preprocessing changes, provider changes, or paid
API calls are authorized by this protocol.

## Fixed candidate representations and fits

Four families, with no additional family search in this campaign:

- TF-IDF, exactly the v1 configuration.
- The 16 existing prompt features, exactly the v1 configuration.
- Frozen `BAAI/bge-small-en-v1.5` embeddings.
- The same frozen embeddings concatenated with the 16 prompt features.

Encoder revision: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
The [official model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
specifies 384 dimensions, 512-token context, CLS pooling, and L2 normalization.
Use raw prompt text without a retrieval instruction. To retain long prompt
content, tokenize without truncation, partition into consecutive nonoverlapping
chunks of at most 510 content tokens, add CLS/SEP, encode each chunk, normalize
CLS vectors, average with weights equal to content-token counts, and L2 normalize
the prompt vector. Empty text uses one special-token-only chunk. No encoder
fine-tuning. Use float32 CPU inference, eval/inference mode, fixed batch size 16,
four CPU threads, no remote code, and safetensors weights. An immutable encoder
and response-free feature cache may be shared across regimes; learned scalers,
heads, cost estimates, validation selection, and novelty reference sets may not.

For both dense families, use training-only StandardScaler (mean and variance),
refitted inside each CV fold. Eight independent logistic regressions, shared C
from {0.1, 1, 10}, mean per-model/fold log loss, five dataset-stratified group folds,
liblinear/L2/max_iter=2000/no class weights/seed 3407, as in v1. Smaller C wins
ties. No train+validation refit or calibration fit. If numerical convergence
fails, report it as an implementation issue rather than silently expanding C.

Fit all four standard families and freeze standard choices first. Fit OOD heads
from scratch using zero LiveCodeBench train/validation prompts. Neither regime's
learned parameters can initialize the other.

## Fixed policy candidates

For each family, retain the v1 threshold grid (21) and utility grid (8).
Add five comparative policies, with margins {0, 0.02, 0.05, 0.10, 0.20}:

Choose the cheapest model whose training mean cost is strictly below the
best-single cost and whose predicted success is at least best-single predicted
success plus the margin. Break qualifying ties by higher predicted success,
then stable model ID. If none qualify, choose best-single. The best-single model
is selected using validation success, ties by training mean cost and model ID.

For every candidate, evaluate four gates: none, or the 1st/5th/10th percentile of
training leave-one-group-out maximum cosine similarity to training embeddings.
At validation/inference, maximum cosine similarity below the fixed cutoff causes
fallback to best-single. Cutoffs and references use current-regime training
prompts only. Dataset names never enter routing features or gate decisions.

Total: 4 × 34 × 4 = 544 candidate IDs, plus static best-single fallback. Identical
action vectors can be deduplicated computationally with their aliases preserved.
Gate thresholds, policy grids, or candidate representations must not expand after
viewing this campaign's validation or test outcomes.

## Selection and uncertainty

Save two comparisons:

1. **Representation control:** for each family, use the original v1 minimum-cost
   micro-quality validation selector over the original ungated 29 policies. This
   isolates representation changes; it is secondary and cannot override primary.
2. **Primary conservative selector:** choose the cheapest validation policy with
   both micro and dataset-macro quality lower bounds at least best-single minus
   0.005, and a point quality difference at least −0.005 on every validation
   dataset. Ties: greater micro quality, then stable candidate ID. Best-single
   itself is always feasible. No minimum savings is imposed on feasibility;
   a static fallback with zero savings is reported as no routing benefit.

Selection bounds use 2,000 paired, dataset-stratified whole-group bootstrap
replicates with seed 3407. Use a common resample for all policy/model comparisons.
Compute bootstrap standard errors and the 95th percentile of the maximum
studentized centered error across all 544 candidates, both micro/macro metrics,
all eight possible best-single anchor identities (for comparative policies and
gate fallbacks), and all eight static comparators; lower bounds are observed difference minus
that common critical value times the bootstrap standard error. This accounts
approximately for searching policies/families and selecting best-single on the
same validation set. Including every possible anchor is necessary because the
anchor also defines comparative actions and gated fallback identities. Only the
544 policies anchored at the validation-selected best-single are eligible for
deployment. Zero-variance comparisons have zero centered error; they
remain empirical, not distribution-free guarantees. Report the critical value,
effective candidate count, group counts, and all feasibility columns. These are
approximate simultaneous bootstrap bounds, not a finite-sample safety proof.

Persist all selected policies and hashes before reading new test outcomes. Report
the frozen primary, all representation controls, baselines, and the full frozen
grid on test; never promote a better test point. Use the v1 2,000-replicate paired
percentile test bootstrap for the primary difference and savings; also report
macro intervals and per-dataset results. Test intervals condition on fixed fits
and selection; prior test reuse still makes the result exploratory.

## A meaningful result and stopping rule

Call a regime's primary result promising only when its test cost savings are at
least 10%, its micro AND macro quality 95% CI lower bounds are at least −0.005,
and every dataset's point quality difference is at least −0.005. This stricter
domain guard can fail even if micro noninferiority passes. Report all conditions
individually; do not relax a failed condition after inspection. On OOD, a fallback
that preserves quality without saving cost is useful deferral behavior but does
not demonstrate learned routing value on unseen code.

Complete this campaign regardless of outcome. Preserve all failed candidates.
If validation selects only fallback or this fixed campaign fails, any further
experiment needs a new written hypothesis and a new bounded validation-only
search; repeated test peeking cannot turn failure into evidence. Final reports
must distinguish promising exploratory results from external confirmation.

## Artifacts and checks

Use `artifacts/router_v2/` for immutable encoder/cache, regime-specific models,
predictions, CV results, train-derived novelty thresholds, validation candidates,
simultaneous-bootstrap metadata/samples, frozen selections, test decisions,
outcomes, diagnostics, calibration, frontier, paired-bootstrap samples, and
hash/version manifests. Keep the v1 outputs unchanged. Save a source snapshot
with the experiment so subsequent reporting edits do not break provenance.

Tests must cover response-free encoder inputs, complete long-prompt chunking,
cache hash and prompt-ID alignment, offline-only inference, independent learned
fits, training-only scaling/novelty estimates, group exclusion in nearest-neighbor
references, comparative-policy semantics and ties, gate fallback, simultaneous
selection bounds including all static comparators, always-feasible fallback,
no test access before freezing, and deterministic predictions/replay. Run the
entire existing offline suite as well.
