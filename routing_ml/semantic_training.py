"""Train-only logistic heads for the two fixed dense semantic feature families."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from routing_data.features import FEATURE_COLUMNS
from routing_ml.embeddings import DIMENSIONS
from routing_ml.training import (
    C_GRID,
    LOGREG_PARAMS,
    MODEL_IDS,
    SEED,
    clipped_log_loss,
    digest,
    fit_predictors,
    probabilities,
)

FAMILIES = ("tfidf", "handcrafted", "embedding", "embedding_handcrafted")


def dense_inputs(part, embeddings, family):
    if family not in {"embedding", "embedding_handcrafted"}:
        raise ValueError("Unknown dense family")
    if not embeddings.index.equals(part.prompts.index) or embeddings.shape[1] != DIMENSIONS:
        raise ValueError("Embedding IDs/dimensions must align exactly")
    values = embeddings.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite embeddings")
    if family == "embedding_handcrafted":
        values = np.column_stack([values, part.features[FEATURE_COLUMNS].to_numpy(dtype=float)])
    return values


@dataclass
class SemanticModel:
    regime: str
    kind: str
    selected_c: float
    preprocessor: object
    predictors: list
    metadata: dict

    def predict(self, part, embeddings):
        if part.regime != self.regime:
            raise ValueError("Cannot reuse learned model across regimes")
        x = self.preprocessor.transform(dense_inputs(part, embeddings, self.kind))
        return pd.DataFrame(
            probabilities(self.predictors, x), index=part.prompts.index, columns=MODEL_IDS
        )


def train_semantic(train, embeddings, family="embedding", progress=print):
    train.require("train")
    raw = dense_inputs(train, embeddings, family)
    y = train.y.to_numpy(dtype=float)
    groups = train.prompts.leakage_group.to_numpy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    records, assignments, preprocessing_audit = [], [], []
    with threadpool_limits(limits=1):
        for fold, (fit_idx, score_idx) in enumerate(
            folds.split(raw, train.prompts.dataset, groups)
        ):
            if set(groups[fit_idx]) & set(groups[score_idx]):
                raise ValueError("CV group leakage")
            scaler = StandardScaler()
            fit_x = scaler.fit_transform(raw[fit_idx])
            score_x = scaler.transform(raw[score_idx])
            preprocessing_audit.append(
                dict(
                    fold=fold,
                    mean=scaler.mean_.tolist(),
                    scale=scaler.scale_.tolist(),
                    training_prompt_ids_hash=digest(train.prompts.index[fit_idx].tolist()),
                )
            )
            assignments.extend(
                dict(
                    prompt_id=train.prompts.index[i],
                    fold=fold,
                    leakage_group=groups[i],
                    dataset=train.prompts.dataset.iloc[i],
                )
                for i in score_idx
            )
            for c in C_GRID:
                models, fallbacks = fit_predictors(fit_x, y[fit_idx], c)
                predicted = probabilities(models, score_x)
                for j, model_id in enumerate(MODEL_IDS):
                    records.append(
                        dict(
                            fold=fold,
                            C=c,
                            model_id=model_id,
                            n_score=len(score_idx),
                            log_loss=clipped_log_loss(y[score_idx, j], predicted[:, j]),
                            constant_fallback=any(r["model_id"] == model_id for r in fallbacks),
                        )
                    )
            if progress:
                progress(f"{train.regime}/{family}: CV fold {fold + 1}/5 complete", flush=True)
        cv = pd.DataFrame(records)
        scores = cv.groupby("C").log_loss.mean().to_dict()
        c = min(scores, key=lambda k: (scores[k], k))
        scaler = StandardScaler()
        x = scaler.fit_transform(raw)
        models, fallbacks = fit_predictors(x, y, c)
    columns = [f"embedding_{i:03d}" for i in range(DIMENSIONS)]
    if family == "embedding_handcrafted":
        columns += FEATURE_COLUMNS
    metadata = dict(
        seed=SEED,
        selected_C=c,
        family=family,
        candidate_model_order=list(MODEL_IDS),
        training_prompt_ids_hash=digest(train.prompts.index.tolist()),
        training_prompt_count=len(train.prompts),
        training_dataset_counts=train.prompts.dataset.value_counts().sort_index().to_dict(),
        cv_mean_per_model_log_loss=scores,
        cv_stratification="dataset",
        cv_preprocessing=preprocessing_audit,
        feature_columns=columns,
        logistic_parameters=LOGREG_PARAMS,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        final_constant_fallbacks=fallbacks,
        final_n_iter=[int(m.n_iter_.max()) if hasattr(m, "n_iter_") else 0 for m in models],
        fit_split="train",
        refit_train_validation=False,
        calibration="raw probabilities",
    )
    return (
        SemanticModel(train.regime, family, c, scaler, models, metadata),
        cv,
        pd.DataFrame(assignments).sort_values("prompt_id"),
    )


def predict_family(model, part, embeddings):
    if model.kind in {"tfidf", "handcrafted"}:
        return model.predict(part)
    return model.predict(part, embeddings.loc[part.prompts.index])
