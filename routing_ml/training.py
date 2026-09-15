"""Train-only preprocessing and eight independent binary success predictors."""

import hashlib
import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from routing_data.features import FEATURE_COLUMNS
from routing_data.loading import load_split

SEED = 3407
MODEL_IDS = (
    "qwen3-235b-a22b-2507",
    "intern-s1",
    "deepseek-v3-0324",
    "deepseek-r1-0528",
    "gemini-2.5-flash",
    "gpt-5-chat",
    "gpt-5",
    "claude-sonnet-4",
)
TFIDF_PARAMS = dict(
    ngram_range=(1, 2), lowercase=True, min_df=2, max_features=50000, sublinear_tf=True, norm="l2"
)
LOGREG_PARAMS = dict(
    penalty="l2", solver="liblinear", max_iter=2000, random_state=SEED, class_weight=None, tol=1e-4
)
C_GRID = (0.1, 1.0, 10.0)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class Split:
    regime: str
    name: str
    prompts: pd.DataFrame
    features: pd.DataFrame
    y: pd.DataFrame
    costs: pd.DataFrame

    def __post_init__(self):
        if self.regime not in {"standard", "ood"} or self.name not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError("Unknown regime/split")
        ids = self.prompts.index
        if ids.has_duplicates or any(
            not ids.equals(x.index) for x in (self.features, self.y, self.costs)
        ):
            raise ValueError("Prompt ID alignment failure")
        if tuple(self.y.columns) != MODEL_IDS or tuple(self.costs.columns) != MODEL_IDS:
            raise ValueError("Candidate model order differs from frozen pool")
        if set(self.prompts[self.regime + "_split"]) != {self.name}:
            raise ValueError("Split metadata mismatch")
        if (
            self.regime == "ood"
            and self.name != "test"
            and self.prompts.dataset.eq("livecodebench").any()
        ):
            raise ValueError("OOD training/validation contains LiveCodeBench")
        if not np.isin(self.y.to_numpy(dtype=float), [0, 1]).all():
            raise ValueError("Nonbinary or missing labels")
        c = self.costs.to_numpy(dtype=float)
        if not np.isfinite(c).all() or (c <= 0).any():
            raise ValueError("Invalid costs")

    def require(self, name):
        if self.name != name:
            raise ValueError(f"This operation accepts {name} only")


def read_split(directory, regime, name):
    return Split(regime, name, *load_split(directory, regime, name))


def inputs(part, kind):
    # Explicit projections prevent IDs, domains, labels, costs and outputs entering X.
    if kind == "tfidf":
        return part.prompts["prompt"].tolist()
    if kind == "handcrafted":
        return part.features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    raise ValueError("Unknown feature kind")


def preprocessing(kind):
    return TfidfVectorizer(**TFIDF_PARAMS) if kind == "tfidf" else StandardScaler()


def clipped_log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


@dataclass
class ConstantPredictor:
    prevalence: float

    def predict_proba(self, x):
        return np.tile([1 - self.prevalence, self.prevalence], (x.shape[0], 1))


def fit_predictors(x, y, c):
    predictors, fallbacks = [], []
    for j, model_id in enumerate(MODEL_IDS):
        labels = y[:, j]
        if len(np.unique(labels)) == 1:
            predictor = ConstantPredictor(float(labels.mean()))
            fallbacks.append(dict(model_id=model_id, prevalence=predictor.prevalence))
        else:
            predictor = LogisticRegression(C=c, **LOGREG_PARAMS)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                predictor.fit(x, labels)
        predictors.append(predictor)
    return predictors, fallbacks


def probabilities(predictors, x):
    columns = []
    for model in predictors:
        # Some NumPy/macOS BLAS builds emit spurious floating-point flags in
        # finite dense matmul. Keep sklearn's probabilities, but only consume
        # such a warning after an independent, non-BLAS calculation agrees.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            p = model.predict_proba(x)[:, 1]
        if not np.isfinite(p).all():
            raise FloatingPointError("Nonfinite classifier probabilities")
        if isinstance(x, np.ndarray) and hasattr(model, "coef_"):
            reference = expit(
                np.einsum("ij,j->i", x, model.coef_[0], optimize=False) + model.intercept_[0]
            )
            np.testing.assert_allclose(p, reference, rtol=1e-13, atol=1e-15)
        elif caught:
            raise FloatingPointError(str(caught[0].message))
        columns.append(p)
    return np.column_stack(columns)


def preprocessing_hash(preprocessor, kind):
    if kind == "tfidf":
        return digest(
            sorted((word, int(index)) for word, index in preprocessor.vocabulary_.items())
        )
    return digest(
        dict(
            columns=FEATURE_COLUMNS,
            mean=preprocessor.mean_.tolist(),
            scale=preprocessor.scale_.tolist(),
        )
    )


@dataclass
class RouterModel:
    regime: str
    kind: str
    selected_c: float
    preprocessor: object
    predictors: list
    metadata: dict

    def predict(self, part):
        if part.regime != self.regime:
            raise ValueError("Cannot reuse model across standard/OOD regimes")
        x = self.preprocessor.transform(inputs(part, self.kind))
        return pd.DataFrame(
            probabilities(self.predictors, x), index=part.prompts.index, columns=MODEL_IDS
        )


def train_router(train, kind="tfidf", progress=None):
    """No validation/test argument exists. Each fold fits preprocessing on its train rows."""
    train.require("train")
    raw = inputs(train, kind)
    y = train.y.to_numpy(dtype=float)
    ids = train.prompts.index.to_numpy()
    groups = train.prompts.leakage_group.to_numpy()
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    records, fold_metadata, assignments = [], [], []
    with threadpool_limits(limits=1):
        # Eight binary targets have no single multiclass stratum: use frozen dataset strata.
        for fold, (fit_idx, score_idx) in enumerate(
            folds.split(ids, train.prompts.dataset, groups)
        ):
            assert not set(groups[fit_idx]) & set(groups[score_idx])
            prep = preprocessing(kind)
            fit_raw = [raw[i] for i in fit_idx] if kind == "tfidf" else raw[fit_idx]
            score_raw = [raw[i] for i in score_idx] if kind == "tfidf" else raw[score_idx]
            x_fit, x_score = prep.fit_transform(fit_raw), prep.transform(score_raw)
            fold_metadata.append(
                dict(
                    fold=fold,
                    fit_prompt_ids_hash=digest(ids[fit_idx].tolist()),
                    score_prompt_ids_hash=digest(ids[score_idx].tolist()),
                    preprocessing_hash=preprocessing_hash(prep, kind),
                    n_fit=len(fit_idx),
                    n_score=len(score_idx),
                )
            )
            assignments.extend(
                dict(
                    prompt_id=ids[i],
                    fold=fold,
                    leakage_group=groups[i],
                    dataset=train.prompts.dataset.iloc[i],
                )
                for i in score_idx
            )
            for c in C_GRID:
                predictors, fallbacks = fit_predictors(x_fit, y[fit_idx], c)
                pred = probabilities(predictors, x_score)
                for j, model_id in enumerate(MODEL_IDS):
                    records.append(
                        dict(
                            fold=fold,
                            C=c,
                            model_id=model_id,
                            n_score=len(score_idx),
                            log_loss=clipped_log_loss(y[score_idx, j], pred[:, j]),
                            constant_fallback=any(f["model_id"] == model_id for f in fallbacks),
                        )
                    )
            if progress:
                progress(f"{train.regime}/{kind}: CV fold {fold + 1}/5 complete")
        cv = pd.DataFrame(records)
        # Equal mean across the five fold scores and eight candidate predictors.
        scores = cv.groupby("C").log_loss.mean().to_dict()
        c = min(scores, key=lambda c: (scores[c], c))
        prep = preprocessing(kind)
        x = prep.fit_transform(raw)
        predictors, fallbacks = fit_predictors(x, y, c)
    metadata = dict(
        seed=SEED,
        candidate_model_order=list(MODEL_IDS),
        selected_C=c,
        cv_mean_per_model_log_loss=scores,
        cv_folds=fold_metadata,
        cv_stratification="dataset",
        cv_aggregation="equal mean over folds and candidate models",
        training_prompt_ids_hash=digest(ids.tolist()),
        training_prompt_count=len(ids),
        training_dataset_counts=train.prompts.dataset.value_counts().sort_index().to_dict(),
        preprocessing_hash=preprocessing_hash(prep, kind),
        vocabulary_hash=preprocessing_hash(prep, kind) if kind == "tfidf" else None,
        vocabulary_size=len(prep.vocabulary_) if kind == "tfidf" else None,
        tfidf_parameters=TFIDF_PARAMS if kind == "tfidf" else None,
        feature_columns=["prompt"] if kind == "tfidf" else FEATURE_COLUMNS,
        logistic_parameters=LOGREG_PARAMS,
        final_constant_fallbacks=fallbacks,
        final_n_iter=[int(p.n_iter_.max()) if hasattr(p, "n_iter_") else 0 for p in predictors],
        fit_split="train",
        refit_train_validation=False,
        calibration="raw probabilities",
    )
    return (
        RouterModel(train.regime, kind, c, prep, predictors, metadata),
        cv,
        pd.DataFrame(assignments).sort_values("prompt_id"),
    )
