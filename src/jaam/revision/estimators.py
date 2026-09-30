from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def clip_probability(values: np.ndarray, epsilon: float) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), epsilon, 1.0 - epsilon)


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    finite = np.isfinite(values) & (values >= 0)
    values = values[finite]
    denominator = np.square(values).sum()
    if denominator <= 0:
        return float("nan")
    return float(np.square(values.sum()) / denominator)


def augmented_overlap_scores(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> pd.DataFrame:
    a = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    e = np.asarray(propensity, dtype=float)
    m0 = np.asarray(mu0, dtype=float)
    m1 = np.asarray(mu1, dtype=float)
    h = e * (1.0 - e)
    psi1 = h * m1 + a * (1.0 - e) * (y - m1)
    psi0 = h * m0 + (1.0 - a) * e * (y - m0)
    return pd.DataFrame({"h": h, "psi1": psi1, "psi0": psi0})


def augmented_ipow_scores(
    treatment: np.ndarray,
    outcome: np.ndarray,
    observed: np.ndarray,
    propensity: np.ndarray,
    observation_probability: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> pd.DataFrame:
    a = np.asarray(treatment, dtype=float)
    y = np.nan_to_num(np.asarray(outcome, dtype=float), nan=0.0)
    r = np.asarray(observed, dtype=float)
    e = np.asarray(propensity, dtype=float)
    pi = np.asarray(observation_probability, dtype=float)
    m0 = np.asarray(mu0, dtype=float)
    m1 = np.asarray(mu1, dtype=float)
    h = e * (1.0 - e)
    psi1 = h * m1 + r * a * (1.0 - e) * (y - m1) / pi
    psi0 = h * m0 + r * (1.0 - a) * e * (y - m0) / pi
    return pd.DataFrame({"h": h, "psi1": psi1, "psi0": psi0})


def aipw_scores(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> pd.DataFrame:
    a = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    e = np.asarray(propensity, dtype=float)
    m0 = np.asarray(mu0, dtype=float)
    m1 = np.asarray(mu1, dtype=float)
    psi1 = m1 + a * (y - m1) / e
    psi0 = m0 + (1.0 - a) * (y - m0) / (1.0 - e)
    return pd.DataFrame({"h": np.ones(len(a), dtype=float), "psi1": psi1, "psi0": psi0})


def dr_pseudo_outcome(
    treatment: np.ndarray,
    outcome: np.ndarray,
    propensity: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
) -> np.ndarray:
    a = np.asarray(treatment, dtype=float)
    y = np.asarray(outcome, dtype=float)
    e = np.asarray(propensity, dtype=float)
    m0 = np.asarray(mu0, dtype=float)
    m1 = np.asarray(mu1, dtype=float)
    return m1 - m0 + a * (y - m1) / e - (1.0 - a) * (y - m0) / (1.0 - e)


def weighted_quantile(
    values: np.ndarray,
    quantiles: Iterable[float],
    sample_weight: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    quantiles = np.asarray(list(quantiles), dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise ValueError("No positive finite weights are available")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(quantiles, positions, values)


def assign_quantile_group(values: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    boundaries = np.unique(np.asarray(boundaries, dtype=float))
    return (
        np.searchsorted(boundaries, np.asarray(values, dtype=float), side="right") + 1
    )


@dataclass
class ClusterEstimate:
    risk_treated: float
    risk_untreated: float
    risk_difference: float
    risk_ratio: float
    se_treated: float
    se_untreated: float
    se_difference: float
    se_log_ratio: float
    ci_treated_low: float
    ci_treated_high: float
    ci_untreated_low: float
    ci_untreated_high: float
    ci_difference_low: float
    ci_difference_high: float
    ci_ratio_low: float
    ci_ratio_high: float
    clusters: int
    n: int
    sum_h: float

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def estimate_clustered_ratio(
    scores: pd.DataFrame,
    clusters: np.ndarray | pd.Series,
    *,
    confidence_level: float = 0.95,
) -> ClusterEstimate:
    required = {"h", "psi1", "psi0"}
    if not required.issubset(scores.columns):
        raise ValueError(f"scores must contain {sorted(required)}")
    work = scores.loc[:, ["h", "psi1", "psi0"]].copy()
    work["cluster"] = np.asarray(clusters)
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if work.empty:
        raise ValueError("No finite estimating scores are available")

    sum_h = float(work["h"].sum())
    if sum_h <= 0:
        raise ValueError("The tilting-weight denominator is not positive")
    theta1 = float(work["psi1"].sum() / sum_h)
    theta0 = float(work["psi0"].sum() / sum_h)

    work["u1"] = work["psi1"] - theta1 * work["h"]
    work["u0"] = work["psi0"] - theta0 * work["h"]
    cluster_u = work.groupby("cluster", sort=False)[["u1", "u0"]].sum()
    cluster_count = len(cluster_u)
    if cluster_count < 2:
        raise ValueError("At least two clusters are required for clustered inference")
    correction = cluster_count / (cluster_count - 1.0)
    contributions = cluster_u.to_numpy(dtype=float) / sum_h
    covariance = correction * contributions.T @ contributions

    se1 = float(np.sqrt(max(covariance[0, 0], 0.0)))
    se0 = float(np.sqrt(max(covariance[1, 1], 0.0)))
    contrast = np.array([1.0, -1.0])
    var_diff = float(contrast @ covariance @ contrast)
    se_diff = float(np.sqrt(max(var_diff, 0.0)))

    alpha = 1.0 - confidence_level
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=cluster_count - 1))
    diff = theta1 - theta0

    if theta1 > 0 and theta0 > 0:
        log_gradient = np.array([1.0 / theta1, -1.0 / theta0])
        var_log_ratio = float(log_gradient @ covariance @ log_gradient)
        se_log_ratio = float(np.sqrt(max(var_log_ratio, 0.0)))
        ratio = theta1 / theta0
        ratio_low = float(np.exp(np.log(ratio) - critical * se_log_ratio))
        ratio_high = float(np.exp(np.log(ratio) + critical * se_log_ratio))
    else:
        ratio = float("nan")
        se_log_ratio = float("nan")
        ratio_low = float("nan")
        ratio_high = float("nan")

    return ClusterEstimate(
        risk_treated=theta1,
        risk_untreated=theta0,
        risk_difference=diff,
        risk_ratio=ratio,
        se_treated=se1,
        se_untreated=se0,
        se_difference=se_diff,
        se_log_ratio=se_log_ratio,
        ci_treated_low=theta1 - critical * se1,
        ci_treated_high=theta1 + critical * se1,
        ci_untreated_low=theta0 - critical * se0,
        ci_untreated_high=theta0 + critical * se0,
        ci_difference_low=diff - critical * se_diff,
        ci_difference_high=diff + critical * se_diff,
        ci_ratio_low=ratio_low,
        ci_ratio_high=ratio_high,
        clusters=cluster_count,
        n=len(work),
        sum_h=sum_h,
    )


def estimate_clustered_two_ratios(
    numerator1: np.ndarray | pd.Series,
    denominator1: np.ndarray | pd.Series,
    numerator0: np.ndarray | pd.Series,
    denominator0: np.ndarray | pd.Series,
    clusters: np.ndarray | pd.Series,
    *,
    confidence_level: float = 0.95,
) -> ClusterEstimate:
    """Clustered inference when the two marginal risks have distinct denominators."""

    work = (
        pd.DataFrame(
            {
                "num1": np.asarray(numerator1, dtype=float),
                "den1": np.asarray(denominator1, dtype=float),
                "num0": np.asarray(numerator0, dtype=float),
                "den0": np.asarray(denominator0, dtype=float),
                "cluster": np.asarray(clusters),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    d1 = float(work["den1"].sum())
    d0 = float(work["den0"].sum())
    if d1 <= 0 or d0 <= 0:
        raise ValueError("Both treatment-specific denominators must be positive")
    theta1 = float(work["num1"].sum() / d1)
    theta0 = float(work["num0"].sum() / d0)
    work["u1"] = (work["num1"] - theta1 * work["den1"]) / d1
    work["u0"] = (work["num0"] - theta0 * work["den0"]) / d0
    cluster_u = work.groupby("cluster", sort=False)[["u1", "u0"]].sum()
    cluster_count = len(cluster_u)
    if cluster_count < 2:
        raise ValueError("At least two clusters are required")
    correction = cluster_count / (cluster_count - 1.0)
    contributions = cluster_u.to_numpy(dtype=float)
    covariance = correction * contributions.T @ contributions

    se1 = float(np.sqrt(max(covariance[0, 0], 0.0)))
    se0 = float(np.sqrt(max(covariance[1, 1], 0.0)))
    contrast = np.array([1.0, -1.0])
    se_diff = float(np.sqrt(max(float(contrast @ covariance @ contrast), 0.0)))
    alpha = 1.0 - confidence_level
    critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=cluster_count - 1))
    diff = theta1 - theta0
    if theta1 > 0 and theta0 > 0:
        gradient = np.array([1.0 / theta1, -1.0 / theta0])
        se_log_ratio = float(np.sqrt(max(float(gradient @ covariance @ gradient), 0.0)))
        ratio = theta1 / theta0
        ratio_low = float(np.exp(np.log(ratio) - critical * se_log_ratio))
        ratio_high = float(np.exp(np.log(ratio) + critical * se_log_ratio))
    else:
        ratio = se_log_ratio = ratio_low = ratio_high = float("nan")
    return ClusterEstimate(
        risk_treated=theta1,
        risk_untreated=theta0,
        risk_difference=diff,
        risk_ratio=ratio,
        se_treated=se1,
        se_untreated=se0,
        se_difference=se_diff,
        se_log_ratio=se_log_ratio,
        ci_treated_low=theta1 - critical * se1,
        ci_treated_high=theta1 + critical * se1,
        ci_untreated_low=theta0 - critical * se0,
        ci_untreated_high=theta0 + critical * se0,
        ci_difference_low=diff - critical * se_diff,
        ci_difference_high=diff + critical * se_diff,
        ci_ratio_low=ratio_low,
        ci_ratio_high=ratio_high,
        clusters=cluster_count,
        n=len(work),
        sum_h=float("nan"),
    )


def add_nnt_fields(result: dict[str, float | int]) -> dict[str, float | int | str]:
    diff = float(result["risk_difference"])
    low = float(result["ci_difference_low"])
    high = float(result["ci_difference_high"])
    enriched: dict[str, float | int | str] = dict(result)
    if not np.isfinite(diff) or abs(diff) < 1e-12 or low <= 0 <= high:
        enriched["nnt_type"] = "not_interpretable"
        enriched["nnt"] = float("nan")
    elif diff > 0:
        enriched["nnt_type"] = "NNT"
        enriched["nnt"] = 1.0 / diff
    else:
        enriched["nnt_type"] = "NNH"
        enriched["nnt"] = 1.0 / abs(diff)
    return enriched


def cluster_multiplier_intervals(
    scores: pd.DataFrame,
    clusters: np.ndarray | pd.Series,
    *,
    repetitions: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, float | int]:
    """Score-level Gaussian cluster multiplier intervals.

    Nuisance fits and any externally learned group boundaries remain fixed.  This
    is the influence-function multiplier bootstrap used for GATES and contrasts;
    it is not a computationally prohibitive full model-refitting bootstrap.
    """

    work = scores.loc[:, ["h", "psi1", "psi0"]].copy()
    work["cluster"] = np.asarray(clusters)
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    grouped = work.groupby("cluster", sort=False)[["h", "psi1", "psi0"]].sum()
    cluster_count = len(grouped)
    if cluster_count < 2:
        raise ValueError("At least two clusters are required")
    totals = grouped.sum()
    sum_h = float(totals["h"])
    theta1 = float(totals["psi1"] / sum_h)
    theta0 = float(totals["psi0"] / sum_h)
    u1 = grouped["psi1"].to_numpy() - theta1 * grouped["h"].to_numpy()
    u0 = grouped["psi0"].to_numpy() - theta0 * grouped["h"].to_numpy()

    rng = np.random.default_rng(seed)
    scale = np.sqrt(cluster_count / (cluster_count - 1.0))
    draws1 = np.empty(repetitions, dtype=float)
    draws0 = np.empty(repetitions, dtype=float)
    chunk_size = min(1000, repetitions)
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        multiplier = rng.normal(size=(stop - start, cluster_count))
        draws1[start:stop] = theta1 + scale * (multiplier @ u1) / sum_h
        draws0[start:stop] = theta0 + scale * (multiplier @ u0) / sum_h

    alpha = 1.0 - confidence_level
    lower_q, upper_q = alpha / 2.0, 1.0 - alpha / 2.0
    diff = draws1 - draws0
    ratio = np.divide(
        draws1,
        draws0,
        out=np.full_like(draws1, np.nan),
        where=(draws1 > 0) & (draws0 > 0),
    )
    return {
        "multiplier_repetitions": int(repetitions),
        "multiplier_clusters": int(cluster_count),
        "multiplier_ci_treated_low": float(np.nanquantile(draws1, lower_q)),
        "multiplier_ci_treated_high": float(np.nanquantile(draws1, upper_q)),
        "multiplier_ci_untreated_low": float(np.nanquantile(draws0, lower_q)),
        "multiplier_ci_untreated_high": float(np.nanquantile(draws0, upper_q)),
        "multiplier_ci_difference_low": float(np.nanquantile(diff, lower_q)),
        "multiplier_ci_difference_high": float(np.nanquantile(diff, upper_q)),
        "multiplier_ci_ratio_low": float(np.nanquantile(ratio, lower_q)),
        "multiplier_ci_ratio_high": float(np.nanquantile(ratio, upper_q)),
    }
