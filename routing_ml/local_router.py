"""Use a frozen router on new prompt text; no labels, dataset IDs or provider calls."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from routing_data.features import FEATURE_COLUMNS, extract_prompt_features
from routing_ml.conservative import Candidate, candidate_actions
from routing_ml.embeddings import encode_prompts, file_hash
from routing_ml.training import MODEL_IDS, probabilities


class LocalRouter:
    def __init__(self, artifact_dir):
        self.directory = Path(artifact_dir)
        hashes = json.loads((self.directory / "artifact_hashes.json").read_text())
        self._verify("frozen_selection.json", hashes)
        self.frozen = json.loads((self.directory / "frozen_selection.json").read_text())
        self.candidate = Candidate(**self.frozen["primary"])
        self.costs = np.asarray([self.frozen["mean_training_cost_usd"][m] for m in MODEL_IDS])
        reference = self.frozen["reference"] if "reference" in self.frozen else self.frozen["best_single"]
        self.baseline_index = MODEL_IDS.index(reference["model_id"])
        self.model = self.gate = None
        if self.candidate.family != "static":
            name = self.candidate.family + "/models.joblib"
            self._verify(name, hashes)
            self.model = joblib.load(self.directory / name)
        if self.candidate.gate_quantile:
            self._verify("novelty_gate.joblib", hashes)
            self.gate = joblib.load(self.directory / "novelty_gate.joblib")
            if self.gate.regime != self.model.regime:
                raise ValueError("Novelty gate/model regime mismatch")

    def _verify(self, name, hashes):
        if file_hash(self.directory / name) != hashes[name]:
            raise ValueError(f"Frozen router artifact hash mismatch: {name}")

    def predict(self, prompts):
        if not {"prompt_id", "prompt"}.issubset(prompts.columns) or prompts.prompt_id.duplicated().any():
            raise ValueError("Unique prompt_id and prompt columns are required")
        prompts = prompts[["prompt_id", "prompt"]].copy()
        if not prompts.prompt.map(lambda x: isinstance(x, str)).all():
            raise ValueError("Prompt text must be strings")
        if prompts.empty:
            return []
        if self.candidate.family == "static":
            return [dict(prompt_id=pid, selected_model_id=MODEL_IDS[self.baseline_index],
                         estimated_cost_usd=float(self.costs[self.baseline_index]), probabilities=None,
                         policy_id=self.candidate.candidate_id, novelty_fallback=False)
                    for pid in prompts.prompt_id]
        family = self.candidate.family
        embedding = None
        if family.startswith("embedding") or self.gate is not None:
            embedding, _, _ = encode_prompts(prompts, self.directory.parent / "encoder", progress=None)
        if family == "tfidf":
            x = self.model.preprocessor.transform(prompts.prompt.tolist())
        else:
            features = [extract_prompt_features(text) for text in prompts.prompt] if family in {"handcrafted", "embedding_handcrafted"} else None
            numeric = np.asarray([[row[column] for column in FEATURE_COLUMNS] for row in features], dtype=float) if features is not None else None
            raw = (numeric if family == "handcrafted" else np.column_stack([embedding, numeric])
                   if family == "embedding_handcrafted" else embedding)
            x = self.model.preprocessor.transform(np.asarray(raw, dtype=float))
        p = pd.DataFrame(probabilities(self.model.predictors, x), index=prompts.prompt_id, columns=MODEL_IDS)
        similarity = self.gate.score(embedding) if self.gate is not None else np.ones(len(prompts))
        action = candidate_actions({family: p}, self.costs, self.baseline_index, similarity, self.gate,
                                   [self.candidate])[:, 0]
        return [dict(prompt_id=pid, selected_model_id=MODEL_IDS[j], estimated_cost_usd=float(self.costs[j]),
                     probabilities={m: float(p.iloc[i, k]) for k, m in enumerate(MODEL_IDS)},
                     policy_id=self.candidate.candidate_id,
                     novelty_fallback=bool(self.gate is not None and similarity[i] < self.gate.cutoffs[self.candidate.gate_quantile]))
                for i, (pid, j) in enumerate(zip(prompts.prompt_id, action))]
