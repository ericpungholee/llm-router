# LLMRouterBench data audit

Audit performed on 2026-09-14 **before implementing preprocessing**, by streaming every JSON record in the official archive. This report distinguishes archive inventory from the final experimental subset. Machine-readable inventories and the processed summary accompany the reproducible pipeline.

## Official source and observed scale

- [Official source](https://github.com/ynulihao/LLMRouterBench/tree/c77cb0506949d8f959e97967d2fefca0e8ff1b05), commit `c77cb0506949d8f959e97967d2fefca0e8ff1b05` (2026-04-06).
- [Official Hugging Face release](https://huggingface.co/datasets/NPULH/LLMRouterBench/tree/0e5af1b84bf73437a01a1849c0f1d2468baa93fc), revision `0e5af1b84bf73437a01a1849c0f1d2468baa93fc`.
- `bench-release.tar.gz`: 1,283,503,080 compressed bytes; 6,987,596,078 JSON payload bytes; 700 result files; **548,059 outcome records**, 27 dataset labels, 40 exact model identifiers.
- SHA-256: `b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1` (matches Hugging Face LFS object hash).
- **27,203** distinct `(dataset, source split, record index)` keys; **25,193** distinct `(dataset, exact prompt text)` keys; **24,443** globally distinct prompt strings. These differ because subsets overlap, category exports duplicate ArenaHard, and some source indices repeat prompt text. The archive's 548,059 records are not 548,059 independent prompts, nor all successful generations.
- Hugging Face publishes an archive, with no dataset card or working tabular viewer. Download the pinned file directly; do not invoke `datasets.load_dataset` or deserialize pickles. The current source supports additional evaluators that are absent from this archive.

## Full dataset inventory

Counts below include the demo and overlapping exports; they must not be summed as unique independent prompts.

| dataset | split | outcomes | source_indices | prompt_texts | models | null_scores | zero_costs | score_values |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aime | hybrid | 2280 | 60 | 60 | 38 | 0 | 552 | 0.0, 1.0 |
| arc-agi | v1 | 6800 | 400 | 400 | 17 | 0 | 296 | 0.0, 1.0 |
| arcc | test | 23440 | 1172 | 1172 | 20 | 0 | 10548 | 0.0, 1.0 |
| arenahard | test | 26250 | 750 | 751 | 35 | 0 | 15093 | 0.0, 0.5, 1.0 |
| arenahard_coding | test | 8349 | 253 | 253 | 33 | 0 | 5095 | 0.0, 0.5, 1.0 |
| arenahard_creative_writing | test | 8250 | 250 | 250 | 33 | 0 | 5006 | 0.0, 0.5, 1.0 |
| arenahard_math | test | 8151 | 247 | 247 | 33 | 0 | 4987 | 0.0, 0.5, 1.0 |
| bbh | test | 21600 | 1080 | 1080 | 20 | 0 | 9720 | 0.0, 1.0 |
| emorynlp | test | 13940 | 697 | 696 | 20 | 0 | 6273 | 0.0, 1.0 |
| finqa | test | 22940 | 1147 | 1138 | 20 | 0 | 10323 | 0.0, 1.0 |
| gpqa | test | 7524 | 198 | 198 | 38 | 0 | 1782 | 0.0, 1.0 |
| hle | subset_500 | 7500 | 500 | 500 | 15 | 0 | 93 | 0.0, 1.0 |
| hle | test | 32370 | 2158 | 2159 | 15 | 0 | 1495 | 0.0, 1.0 |
| humaneval | test | 3608 | 164 | 164 | 22 | 0 | 1476 | 0.0, 1.0 |
| kandk | test | 14000 | 700 | 700 | 20 | 0 | 6300 | 0.0, 1.0 |
| korbench | test | 25000 | 1250 | 1248 | 20 | 0 | 11250 | 0.0, 1.0 |
| livecodebench | test | 40090 | 1055 | 1055 | 38 | 0 | 9850 | 0.0, 1.0 |
| livemathbench | test | 4235 | 121 | 121 | 35 | 0 | 1131 | 0.0, 1.0 |
| math500 | test | 10000 | 500 | 500 | 20 | 0 | 4500 | 0.0, 1.0 |
| mathbench | test | 3000 | 150 | 150 | 20 | 0 | 1350 | 0.0, 1.0 |
| mbpp | test | 19480 | 974 | 974 | 20 | 1 | 8766 | 0.0, 1.0 |
| medqa | test | 25460 | 1273 | 1273 | 20 | 0 | 11457 | 0.0, 1.0 |
| meld | test | 24640 | 1232 | 1231 | 20 | 0 | 11088 | 0.0, 1.0 |
| mmlupro | test_1000 | 36036 | 1001 | 1000 | 36 | 0 | 9071 | 0.0, 1.0 |
| mmlupro | test_3000 | 45000 | 3000 | 3000 | 15 | 0 | 242 | 0.0, 1.0 |
| simpleqa | subset_500 | 7050 | 500 | 500 | 15 | 0 | 0 | 0.0, 1.0 |
| simpleqa | test | 64890 | 4326 | 4326 | 15 | 4326 | 4380 | 0.0, 1.0 |
| swe-bench | verified | 7500 | 500 | 500 | 15 | 0 | 101 | 0.0, 1.0 |
| tau2 | test | 3336 | 278 | 278 | 12 | 0 | 0 | 0.0, 1.0 |
| winogrande | valid | 25340 | 1267 | 1267 | 20 | 0 | 11403 | 0.0, 1.0 |

## Observed schema

Most files are `bench-release/<dataset>/<split>/<model>/<timestamped-name>.json`. The 133 ArenaHard/category files without a split directory instead use `bench-release/<dataset>/<model>/<filename>.json`; the header supplies the split. File-level identity is authoritative. One excluded SWE-bench path names `qwen3-235b-a22b-thinking` but its header says `qwen3-235b-a22b-thinking-2507`; the loader records this exact path exception, without applying a general model alias. All other paths must agree with headers. `index` is local to the dataset/split and can be zero- or one-based; it is not a global prompt ID. No aliasing or lowercasing of model identifiers is safe.

File fields (presence across 700 files):

| field | files |
| --- | --- |
| completion_tokens | 700 |
| cost | 700 |
| counts | 700 |
| data_fingerprint | 488 |
| dataset_name | 700 |
| demo | 635 |
| model_name | 700 |
| performance | 700 |
| prompt_tokens | 700 |
| records | 700 |
| split | 700 |
| time_taken | 700 |

Record fields (presence across 548,059 rows):

| field | rows |
| --- | --- |
| completion_tokens | 548059 |
| cost | 548059 |
| extra_fields | 11668 |
| ground_truth | 548059 |
| index | 548059 |
| instance_id | 7500 |
| origin_query | 548059 |
| prediction | 548059 |
| prompt | 548059 |
| prompt_tokens | 548059 |
| raw_output | 548059 |
| score | 548059 |

`origin_query` and `prompt` are strings; `prediction`, `ground_truth`, and `raw_output` are outcome/grading data, never features. `score` is numeric or null. `extra_fields.actual_model` occurs in 11,666 records and identifies the model chosen by the preexisting `openrouter` router; it is response-derived and excluded. `instance_id` appears in SWE-bench only. No consistent per-record latency, model context-window limit, or authoritative collection-code revision is available. Source `data_fingerprint` is present for only 488 files and does not replace content hashes.

## Scoring semantics

The [collector](https://github.com/ynulihao/LLMRouterBench/blob/c77cb0506949d8f959e97967d2fefca0e8ff1b05/data_collector/runner.py) preserves evaluator booleans as 0/1, numeric rewards as floats, and explicit unavailable grades as null. Observed non-null scores are 0/1 for every dataset except ArenaHard and its three category exports, which contain 0/0.5/1.

- AIME, LiveMathBench, MATH500, MATHBench: answer extraction and mathematical equivalence grading.
- GPQA, MMLU-Pro, ARC-C, MedQA, BBH, KORBench, Knights & Knaves, Winogrande, FinQA, EmoryNLP, MELD: task-specific answer/label correctness; binary in this release. The exact evaluator templates differ, including few-shot prompts.
- LiveCodeBench, HumanEval, MBPP: correctness of the single stored completion under benchmark execution tests (single-sample pass, not a new pass@k estimate).
- [SimpleQA](https://github.com/ynulihao/LLMRouterBench/blob/c77cb0506949d8f959e97967d2fefca0e8ff1b05/evaluation/SimpleQA/simpleqa.py): judge category A=correct, B=incorrect, C=not attempted; `score=1` only for A. Its parser takes the first A/B/C match and defaults to C; these are benchmark judge labels, not independently reverified facts.
- [HLE](https://github.com/ynulihao/LLMRouterBench/blob/c77cb0506949d8f959e97967d2fefca0e8ff1b05/evaluation/HLE/hle.py): judge `correct: yes/no`; parser defaults to no on missing/invalid grading text. Judge parse failures cannot reliably be separated from wrong answers in the release.
- [ArenaHard](https://github.com/ynulihao/LLMRouterBench/blob/c77cb0506949d8f959e97967d2fefca0e8ff1b05/evaluation/ArenaHard/arenahard.py): two order-swapped comparisons with a category baseline; points map to loss=0, tie=0.5, win=1. Invalid judgment rounds map to 0. This is a relative preference reward, not absolute correctness. No binarization is introduced; raw scores are preserved in audit data and these tasks are excluded from initial binary training.
- SWE-bench and tau2: imported external agent results with binary resolved/success outcomes. Integration code is not a complete frozen reconstruction of the external runs. Costs cover a workflow, not necessarily a single completion. tau2's stored prompt contains user-simulator goals/contingencies and a scenario ID, so its text cannot be assumed agent-visible at routing time.
- ARC-AGI: binary in this archive, but a different structured/visual task interface; excluded from the initial text-routing experiment.

## Missingness, duplicate identity, and cost quirks

- No duplicate `(dataset, split, model, index)` keys and no duplicate run files per `(dataset, split, model)`.
- **25,820** duplicate `(dataset, model, exact prompt)` records across/within exported subsets. Category exports add further cross-dataset duplication. Never combine all source splits indiscriminately.
- One demo file: SimpleQA/subset_500/GPT-5-chat, 50 rows. Ignore demos and overlapping subset_500/test_1000 exports in the initial experiment.
- **4,327 null scores**: 4,326 SimpleQA/test records for `intern-s1-new`, plus one MBPP record for `internlm3-8b-instruct`. Do not replace these with zero.
- **13 negative completion-token counts**: 5 GPT-5 HLE, 5 GPT-5 SimpleQA, 3 openrouter HLE. Quarantine as invalid resources. No negative cost or input-token values, no fractional tokens, and no null cost fields were observed.
- **163,628 zero-cost records**, **6,416 explicit generation-failure strings**, and **3,083 empty raw-output fields** occur across the archive. Zero costs are not necessarily free inference: many small-model runs have no prices. Some flagged failure strings and empty completions carry `score=0` and zero resources. Others contain an actual answer but lack cost accounting. Preserve raw scores and raw resources; explicit failed requests and empty outputs without positive completion usage get null success. Empty final text with positive generation usage keeps the benchmark score (e.g. reasoning without a final answer under the generation budget). Unavailable costs/tokens prevent inclusion in the complete matrix regardless of grade.
- HLE GPT-5 has 507/2,158 zero-cost rows, including 132 successful scores; reconstructing these from invented pricing would hide a material accounting defect. Exclude HLE from the initial task subset.
- Qwen thinking-2507 has large gaps (e.g. 117/1,055 LiveCodeBench rows with zero cost); Gemini Pro has 95/1,055 such LiveCodeBench gaps. Prefer strong candidates with better coverage for the first pool.
- Two source index/text conflicts: ArenaHard index 103 (Claude removes a Unicode line separator), HLE index 1494 (GPT-5 text variant). No selected-task index/text conflicts. Exact content identity prevents silent merging.
- 39 ArenaHard category files retain the parent dataset's total cost in their header. Use per-record costs, never allocate aggregate header costs to prompts.
- The [generator](https://github.com/ynulihao/LLMRouterBench/blob/c77cb0506949d8f959e97967d2fefca0e8ff1b05/generators/generator.py) prefers provider-reported usage cost, otherwise configured token prices. The release does not mark which branch produced each cost. Preserve positive per-record cost as `cost_usd` with provenance `benchmark_record`; no price reconstruction is needed or performed. Zero costs are marked unavailable for this initial cost-aware experiment. This is a conservative project policy, not a claim that all zero-cost answers failed.

## Candidate identifiers observed

- `DeepHermes-3-Llama-3-8B-Preview`
- `DeepSeek-R1-0528-Qwen3-8B`
- `DeepSeek-R1-Distill-Qwen-7B`
- `Fin-R1`
- `GLM-Z1-9B-0414`
- `Intern-S1-mini`
- `Llama-3.1-8B-Instruct`
- `Llama-3.1-8B-UltraMedical`
- `Llama-3.1-Nemotron-Nano-8B-v1`
- `MiMo-7B-RL-0530`
- `MiniCPM4.1-8B`
- `NVIDIA-Nemotron-Nano-9B-v2`
- `OpenThinker3-7B`
- `Qwen2.5-Coder-7B-Instruct`
- `Qwen3-8B`
- `claude-opus-4.1`
- `claude-sonnet-4`
- `cogito-v1-preview-llama-8B`
- `deepseek-r1-0528`
- `deepseek-v3-0324`
- `deepseek-v3.1-terminus`
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- `gemma-2-9b-it`
- `glm-4-9b-chat`
- `glm-4.6`
- `glm-4.6-3888`
- `gpt-4.1`
- `gpt-5`
- `gpt-5-chat`
- `granite-3.3-8b-instruct`
- `intern-s1`
- `intern-s1-new`
- `internlm3-8b-instruct`
- `kimi-k2-0905`
- `openrouter`
- `qwen3-235b-a22b-2507`
- `qwen3-235b-a22b-no-thinking`
- `qwen3-235b-a22b-thinking`
- `qwen3-235b-a22b-thinking-2507`

## Initial selection decision

Use six disjoint source selections: AIME/hybrid (math), LiveMathBench/test (math), GPQA/test (scientific reasoning), LiveCodeBench/test (code), MMLU-Pro/test_3000 (knowledge), SimpleQA/test (knowledge). They contain 8,760 prompt vectors before filtering. Keep eight diverse, well-covered cost-bearing models: Qwen3-235B-2507, Intern-S1, DeepSeek-V3-0324, DeepSeek-R1-0528, Gemini-2.5-Flash, GPT-5-chat, GPT-5, Claude-Sonnet-4. This gives open-weight families, closed models, general/chat and reasoning behavior, cheap and expensive candidates. Pool choices prioritize resource coverage and family/behavior diversity; test scores are not used to select a winning baseline or optimize the pool.

Only 54/8,760 prompts (0.6164%) lose at least one valid resource record in this pool, leaving **8,706 complete prompts** in the preliminary audit. No label imputation or masked training is necessary. Exact final counts, additional failure checks, split sizes, model-level coverage, baseline summaries, and leakage validation are recorded by the preparation pipeline in `reports/llmrouterbench_processed_summary.json` and `reports/llmrouterbench_preparation.md`.

The entire 1,055-prompt LiveCodeBench domain will be the OOD test. No other selected task contains code-generation outcomes. Standard evaluation remains stratified across all six datasets. Dataset/task names are metadata, excluded from default numeric/text features; an explicitly privileged task-label ablation is specified separately.
