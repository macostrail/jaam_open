from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .estimators import effective_sample_size


def probability_metrics(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probability, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid].astype(int)
    p = np.clip(p[valid], 1e-12, 1.0 - 1e-12)
    result: dict[str, float | int] = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()) if len(y) else float("nan"),
        "prediction_min": float(p.min()) if len(p) else float("nan"),
        "prediction_max": float(p.max()) if len(p) else float("nan"),
    }
    if len(y) == 0:
        for key in [
            "auroc",
            "auprc",
            "brier",
            "log_loss",
            "calibration_intercept",
            "calibration_slope",
        ]:
            result[key] = float("nan")
        return result

    result["brier"] = float(brier_score_loss(y, p))
    result["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    if len(np.unique(y)) > 1:
        result["auroc"] = float(roc_auc_score(y, p))
        result["auprc"] = float(average_precision_score(y, p))
        logit_p = np.log(p / (1.0 - p))
        try:
            design = sm.add_constant(logit_p, has_constant="add")
            fit = sm.GLM(y, design, family=sm.families.Binomial()).fit()
            result["calibration_intercept"] = float(fit.params[0])
            result["calibration_slope"] = float(fit.params[1])
        except Exception:
            result["calibration_intercept"] = float("nan")
            result["calibration_slope"] = float("nan")
    else:
        result["auroc"] = float("nan")
        result["auprc"] = float("nan")
        result["calibration_intercept"] = float("nan")
        result["calibration_slope"] = float("nan")
    return result


def calibration_table(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "outcome": np.asarray(y_true, dtype=float),
            "probability": np.asarray(probability, dtype=float),
        }
    ).dropna()
    if work.empty:
        return pd.DataFrame()
    unique = work["probability"].nunique()
    q = min(bins, unique)
    if q < 2:
        work["bin"] = 1
    else:
        work["bin"] = (
            pd.qcut(work["probability"], q=q, duplicates="drop", labels=False) + 1
        )
    return work.groupby("bin", as_index=False).agg(
        n=("outcome", "size"),
        events=("outcome", "sum"),
        observed=("outcome", "mean"),
        predicted=("probability", "mean"),
        prediction_min=("probability", "min"),
        prediction_max=("probability", "max"),
    )


def propensity_distribution(
    treatment: np.ndarray | pd.Series,
    propensity: np.ndarray | pd.Series,
) -> pd.DataFrame:
    a = np.asarray(treatment, dtype=int)
    e = np.asarray(propensity, dtype=float)
    rows = []
    quantiles = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    for arm in [0, 1]:
        values = e[a == arm]
        q_values = np.quantile(values, quantiles)
        row: dict[str, float | int] = {
            "treatment": arm,
            "n": int(len(values)),
        }
        for q, value in zip(quantiles, q_values):
            label = (
                "min"
                if q == 0
                else "max" if q == 1 else f"p{100*q:g}".replace(".", "_")
            )
            row[label] = float(value)
        for threshold in [0.01, 0.025, 0.05]:
            label = str(threshold).replace(".", "_")
            row[f"n_below_{label}"] = int((values < threshold).sum())
            row[f"pct_below_{label}"] = float(100 * (values < threshold).mean())
        for threshold in [0.95, 0.975, 0.99]:
            label = str(threshold).replace(".", "_")
            row[f"n_above_{label}"] = int((values > threshold).sum())
            row[f"pct_above_{label}"] = float(100 * (values > threshold).mean())
        rows.append(row)
    treated = e[a == 1]
    untreated = e[a == 0]
    common_low = max(float(treated.min()), float(untreated.min()))
    common_high = min(float(treated.max()), float(untreated.max()))
    for row in rows:
        row["common_support_low"] = common_low
        row["common_support_high"] = common_high
        arm = int(row["treatment"])
        values = e[a == arm]
        row["n_outside_observed_common_support"] = int(
            ((values < common_low) | (values > common_high)).sum()
        )
    return pd.DataFrame(rows)


def weight_distribution(
    treatment: np.ndarray | pd.Series,
    propensity: np.ndarray | pd.Series,
) -> pd.DataFrame:
    a = np.asarray(treatment, dtype=int)
    e = np.asarray(propensity, dtype=float)
    rows = []
    for arm in [0, 1]:
        mask = a == arm
        ipw = 1.0 / e[mask] if arm == 1 else 1.0 / (1.0 - e[mask])
        overlap = 1.0 - e[mask] if arm == 1 else e[mask]
        for name, values in [("ipw_component", ipw), ("overlap_weight", overlap)]:
            row: dict[str, float | int | str] = {
                "treatment": arm,
                "weight": name,
                "n": int(len(values)),
                "ess": effective_sample_size(values),
                "sum": float(values.sum()),
                "max_share": float(values.max() / values.sum()),
            }
            for q, label in [
                (0.0, "min"),
                (0.01, "p1"),
                (0.05, "p5"),
                (0.25, "p25"),
                (0.5, "p50"),
                (0.75, "p75"),
                (0.95, "p95"),
                (0.99, "p99"),
                (1.0, "max"),
            ]:
                row[label] = float(np.quantile(values, q))
            row["n_gt_10"] = int((values > 10).sum())
            row["pct_gt_10"] = float(100 * (values > 10).mean())
            row["n_gt_20"] = int((values > 20).sum())
            row["pct_gt_20"] = float(100 * (values > 20).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not valid.any() or weights[valid].sum() <= 0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _reference_scale(x1: np.ndarray, x0: np.ndarray) -> float:
    v1 = np.nanvar(x1, ddof=1) if np.isfinite(x1).sum() > 1 else 0.0
    v0 = np.nanvar(x0, ddof=1) if np.isfinite(x0).sum() > 1 else 0.0
    return float(np.sqrt((v1 + v0) / 2.0))


def covariate_balance(
    frame: pd.DataFrame,
    treatment: np.ndarray | pd.Series,
    *,
    features: Iterable[str],
    categorical: set[str],
    weights: np.ndarray | pd.Series | None = None,
    method: str,
) -> pd.DataFrame:
    a = np.asarray(treatment, dtype=int)
    w = np.ones(len(frame), dtype=float) if weights is None else np.asarray(weights)
    rows: list[dict[str, float | str | int]] = []

    def add_indicator(
        variable: str,
        level: str,
        indicator: np.ndarray,
        kind: str,
    ) -> None:
        x = indicator.astype(float)
        x1 = x[a == 1]
        x0 = x[a == 0]
        reference = _reference_scale(x1, x0)
        mean1 = _weighted_mean(x1, w[a == 1])
        mean0 = _weighted_mean(x0, w[a == 0])
        smd = (mean1 - mean0) / reference if reference > 0 else 0.0
        rows.append(
            {
                "method": method,
                "variable": variable,
                "component": level,
                "kind": kind,
                "mean_treated": mean1,
                "mean_untreated": mean0,
                "smd": smd,
                "abs_smd": abs(smd),
            }
        )

    for variable in features:
        series = frame[variable]
        if variable in categorical:
            normalized = series.astype("object").where(series.notna(), "__MISSING__")
            levels = sorted(normalized.unique().tolist(), key=lambda value: str(value))
            for level in levels:
                add_indicator(
                    variable,
                    f"level={level}",
                    (normalized == level).to_numpy(),
                    "categorical",
                )
        else:
            values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
            x1 = values[a == 1]
            x0 = values[a == 0]
            reference = _reference_scale(x1, x0)
            mean1 = _weighted_mean(x1, w[a == 1])
            mean0 = _weighted_mean(x0, w[a == 0])
            smd = (mean1 - mean0) / reference if reference > 0 else 0.0
            rows.append(
                {
                    "method": method,
                    "variable": variable,
                    "component": "value",
                    "kind": "numeric",
                    "mean_treated": mean1,
                    "mean_untreated": mean0,
                    "smd": smd,
                    "abs_smd": abs(smd),
                }
            )
            if np.isnan(values).any():
                add_indicator(
                    variable,
                    "missing",
                    np.isnan(values),
                    "missing_indicator",
                )
    return pd.DataFrame(rows)


def summarize_balance(balance: pd.DataFrame) -> pd.DataFrame:
    if balance.empty:
        return pd.DataFrame()
    return balance.groupby("method", as_index=False).agg(
        components=("abs_smd", "size"),
        max_abs_smd=("abs_smd", "max"),
        median_abs_smd=("abs_smd", "median"),
        components_above_0_1=("abs_smd", lambda x: int((x > 0.1).sum())),
        variables_above_0_1=(
            "variable",
            lambda x: int(
                balance.loc[x.index].groupby("variable")["abs_smd"].max().gt(0.1).sum()
            ),
        ),
    )
