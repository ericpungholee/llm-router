# Third-party data and models

The repository owner has not selected a license for the original project code.

The learned-router experiments use the public
[LLMRouterBench release](https://huggingface.co/datasets/NPULH/LLMRouterBench)
and its [upstream repository](https://github.com/ynulihao/LLMRouterBench).
Exact revisions, URLs, SHA-256 hashes, and the audit date are recorded in
[the source configuration](configs/llmrouterbench_source.json).

Included tasks originate from AIME, LiveMathBench, GPQA, LiveCodeBench, MMLU-Pro,
and SimpleQA. Raw releases and processed matrices are not distributed in this
Git repository. Upstream data, benchmark questions, model outputs, and model
weights retain their respective terms; a license for this repository's original
code does not relicense those materials.

The small historical pilot samples in `benchmarks/samples/` retain their source
identities and revisions in [the benchmark manifest](benchmarks/manifest.json).
The recorded responses in `reports/snapshots/` are historical experimental
outputs with accompanying provenance. They are separate from the primary
LLMRouterBench training matrix.

The optional V2 encoder is
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), pinned in
[routing_ml/embeddings.py](routing_ml/embeddings.py). Its weights are downloaded
separately and are not included here. Python dependencies retain their upstream
licenses.
