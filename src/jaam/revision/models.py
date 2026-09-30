from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FoldEncoder:
    """Training-fold-only encoding for LightGBM native categorical handling."""

    def __init__(self, features: Iterable[str], categorical: Iterable[str]):
        self.features = list(features)
        self.categorical = [col for col in self.features if col in set(categorical)]
        self.numeric = [
            col for col in self.features if col not in set(self.categorical)
        ]
        self.categories_: dict[str, list[Any]] = {}

    def fit(self, frame: pd.DataFrame) -> "FoldEncoder":
        self.categories_ = {}
        for column in self.categorical:
            values = frame[column].dropna().unique().tolist()
            try:
                values = sorted(values)
            except TypeError:
                values = sorted(values, key=lambda value: str(value))
            self.categories_[column] = values
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.categories_ and self.categorical:
            raise RuntimeError("FoldEncoder must be fitted before transform")
        result = pd.DataFrame(index=frame.index)
        for column in self.features:
            if column in self.categories_:
                dtype = pd.CategoricalDtype(
                    categories=self.categories_[column],
                    ordered=False,
                )
                result[column] = frame[column].astype(dtype)
            else:
                result[column] = pd.to_numeric(frame[column], errors="coerce").astype(
                    "float32"
                )
        return result

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(frame).transform(frame)


@dataclass
class ConstantProbabilityModel:
    probability: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        p = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack((1.0 - p, p))


@dataclass
class FittedTabularModel:
    encoder: FoldEncoder
    estimator: BaseEstimator | ConstantProbabilityModel
    kind: str

    def encoded(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(frame)

    def predict_probability(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = self.encoded(frame)
        return np.asarray(self.estimator.predict_proba(encoded)[:, 1], dtype=float)

    def predict_effect(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = self.encoded(frame)
        return np.asarray(self.estimator.predict(encoded), dtype=float)


def fit_probability_model(
    frame: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    features: list[str],
    categorical: set[str],
    params: Mapping[str, Any],
    seed: int,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> FittedTabularModel:
    encoder = FoldEncoder(features, categorical)
    encoded = encoder.fit_transform(frame)
    target = np.asarray(y, dtype=int)
    unique = np.unique(target)
    if len(unique) == 1:
        estimator: BaseEstimator | ConstantProbabilityModel = ConstantProbabilityModel(
            float(unique[0])
        )
    else:
        model_params = dict(params)
        model_params["random_state"] = int(seed)
        estimator = LGBMClassifier(**model_params)
        estimator.fit(
            encoded,
            target,
            sample_weight=None if sample_weight is None else np.asarray(sample_weight),
            categorical_feature=encoder.categorical,
        )
    return FittedTabularModel(encoder=encoder, estimator=estimator, kind="probability")


def fit_effect_model(
    frame: pd.DataFrame,
    pseudo_outcome: pd.Series | np.ndarray,
    *,
    features: list[str],
    categorical: set[str],
    params: Mapping[str, Any],
    seed: int,
    sample_weight: np.ndarray | pd.Series,
) -> FittedTabularModel:
    encoder = FoldEncoder(features, categorical)
    encoded = encoder.fit_transform(frame)
    model_params = dict(params)
    model_params["random_state"] = int(seed)
    estimator = LGBMRegressor(**model_params)
    estimator.fit(
        encoded,
        np.asarray(pseudo_outcome, dtype=float),
        sample_weight=np.asarray(sample_weight, dtype=float),
        categorical_feature=encoder.categorical,
    )
    return FittedTabularModel(encoder=encoder, estimator=estimator, kind="effect")


def make_logistic_propensity_pipeline(
    *,
    numeric: list[str],
    categorical: list[str],
    c_value: float,
    max_iter: int,
    seed: int,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "impute",
                SimpleImputer(strategy="most_frequent", add_indicator=True),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    transform = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=0.3,
    )
    logistic = LogisticRegression(
        C=float(c_value),
        penalty="l2",
        solver="lbfgs",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    return Pipeline(steps=[("preprocess", transform), ("model", logistic)])
