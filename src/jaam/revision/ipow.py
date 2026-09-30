from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .core import _seed
from .data import RevisionData, make_balanced_folds, prepare_revision_data
from .diagnostics import probability_metrics
from .estimators import (
    add_nnt_fields,
    assign_quantile_group,
    augmented_ipow_scores,
    augmented_overlap_scores,
    cluster_multiplier_intervals,
    effective_sample_size,
    estimate_clustered_ratio,
)
from .models import fit_probability_model

LOGGER = logging.getLogger(__name__)


def _prepare_observation_features(
    data: RevisionData,
    *,
    include_30d_auxiliary: bool,
) -> tuple[pd.DataFrame, list[str], set[str]]:
    frame = data.frame.copy()
    frame["__observation_treatment"] = frame["_treatment"]
    frame["__observation_facility"] = frame["_facility"]
    frame["__observation_year"] = frame["_year"]
    features = [
        *data.main_features,
        "__observation_treatment",
        "__observation_facility",
        "__observation_year",
    ]
    categorical = set(data.categorical_features)
    categorical.update({"__observation_treatment", "__observation_facility"})
    if include_30d_auxiliary:
        frame["__aux_survival_30d"] = frame["_outcome_survival_30d"]
        frame["__aux_neurologic_30d"] = frame["_outcome_neurologic_30d"]
        features.extend(["__aux_survival_30d", "__aux_neurologic_30d"])
        categorical.update(["__aux_survival_30d", "__aux_neurologic_30d"])
    return frame, features, categorical


def _crossfit_full_propensity(
    data: RevisionData,
    folds: np.ndarray,
) -> np.ndarray:
    frame = data.frame
    config = data.config
    prediction = np.full(len(frame), np.nan)
    epsilon = float(config["probability_epsilon"])
    for fold in sorted(np.unique(folds)):
        train = frame.loc[folds != fold]
        heldout = frame.loc[folds == fold]
        model = fit_probability_model(
            train,
            train["_treatment"],
            features=data.main_features,
            categorical=data.categorical_features,
            params=config["lightgbm_probability"],
            seed=_seed(int(config["seed"]), 600, int(fold)),
        )
        prediction[folds == fold] = np.clip(
            model.predict_probability(heldout),
            epsilon,
            1.0 - epsilon,
        )
    if np.isnan(prediction).any():
        raise RuntimeError("Full-cohort propensity cross-fitting failed")
    return prediction


def _crossfit_ipow_variant(
    data: RevisionData,
    folds: np.ndarray,
    *,
    include_30d_auxiliary: bool,
    seed_offset: int,
) -> pd.DataFrame:
    frame, observation_features, observation_categorical = (
        _prepare_observation_features(data, include_30d_auxiliary=include_30d_auxiliary)
    )
    config = data.config
    central = data.central_outcome
    outcome_column = f"_outcome_{central}"
    observed_column = "__outcome_observed"
    frame[observed_column] = frame[outcome_column].notna().astype("int8")
    epsilon = float(config["probability_epsilon"])
    outer_splits = int(config["folds"]["outer"])
    inner_splits = int(config["folds"]["inner"])

    result = pd.DataFrame(index=frame.index)
    result["observation_probability"] = np.nan
    result["mu0"] = np.nan
    result["mu1"] = np.nan
    for outer in range(outer_splits):
        train = frame.loc[folds != outer].copy()
        heldout = frame.loc[folds == outer].copy()
        LOGGER.info(
            "IPOW %s auxiliary, outer fold %s/%s",
            "with" if include_30d_auxiliary else "without",
            outer + 1,
            outer_splits,
        )

        observation_model = fit_probability_model(
            train,
            train[observed_column],
            features=observation_features,
            categorical=observation_categorical,
            params=config["lightgbm_probability"],
            seed=_seed(
                int(config["seed"]) + seed_offset,
                610,
                outer,
            ),
        )
        result.loc[heldout.index, "observation_probability"] = np.clip(
            observation_model.predict_probability(heldout),
            epsilon,
            1.0 - epsilon,
        )

        inner_folds = make_balanced_folds(
            train,
            n_splits=inner_splits,
            seed=_seed(
                int(config["seed"]) + seed_offset,
                620,
                outer,
            ),
            outcome_column=outcome_column,
        )
        inner_pi = np.full(len(train), np.nan)
        for inner in range(inner_splits):
            inner_train = train.iloc[inner_folds != inner]
            inner_heldout = train.iloc[inner_folds == inner]
            heldout_positions = np.flatnonzero(inner_folds == inner)
            model = fit_probability_model(
                inner_train,
                inner_train[observed_column],
                features=observation_features,
                categorical=observation_categorical,
                params=config["lightgbm_probability"],
                seed=_seed(
                    int(config["seed"]) + seed_offset,
                    630,
                    outer,
                    inner,
                ),
            )
            inner_pi[heldout_positions] = np.clip(
                model.predict_probability(inner_heldout),
                epsilon,
                1.0 - epsilon,
            )
        if np.isnan(inner_pi).any():
            raise RuntimeError(
                f"Inner observation cross-fitting failed in fold {outer}"
            )

        for arm in [0, 1]:
            fit_mask = (train["_treatment"].to_numpy() == arm) & train[
                outcome_column
            ].notna().to_numpy()
            observation_weight = 1.0 / inner_pi[fit_mask]
            outcome_model = fit_probability_model(
                train.iloc[np.flatnonzero(fit_mask)],
                train.loc[fit_mask, outcome_column],
                features=data.main_features,
                categorical=data.categorical_features,
                params=config["lightgbm_probability"],
                seed=_seed(
                    int(config["seed"]) + seed_offset,
                    640,
                    outer,
                    arm,
                ),
                sample_weight=observation_weight,
            )
            result.loc[heldout.index, f"mu{arm}"] = np.clip(
                outcome_model.predict_probability(heldout),
                epsilon,
                1.0 - epsilon,
            )
    if result.isna().any().any():
        raise RuntimeError("IPOW cross-fitting left missing predictions")
    return result


def _apply_main_effect_score_to_all(
    data: RevisionData,
    main_prediction: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    result = main_prediction[["_row_id", "fold", "ite", "ite_group"]].copy()
    boundaries = pd.read_csv(output / "ite_fold_boundaries.csv")
    boundaries = boundaries[boundaries["prediction"] == "unclipped"]
    for fold in sorted(result["fold"].unique()):
        mask = result["fold"] == fold
        missing_score = mask & result["ite"].isna()
        if missing_score.any():
            with (
                output / "models" / "main_effect" / f"effect_fold_{fold}.pickle"
            ).open("rb") as fp:
                model = pickle.load(fp)
            result.loc[missing_score, "ite"] = model.predict_effect(
                data.frame.loc[missing_score]
            )
        fold_boundaries = (
            boundaries.loc[boundaries["fold"] == fold]
            .sort_values("quantile")["boundary"]
            .to_numpy()
        )
        result.loc[mask, "ite_group"] = assign_quantile_group(
            result.loc[mask, "ite"], fold_boundaries
        )
    if result[["ite", "ite_group"]].isna().any().any():
        raise RuntimeError("Effect score extrapolation did not cover the full cohort")
    result["ite_group"] = result["ite_group"].astype(int)
    return result


def _score_effect_contribution(
    scores: pd.DataFrame,
    facilities: np.ndarray,
) -> tuple[float, pd.Series]:
    effect = scores["psi1"].to_numpy() - scores["psi0"].to_numpy()
    h = scores["h"].to_numpy()
    theta = float(effect.sum() / h.sum())
    contribution = (
        pd.Series(
            (effect - theta * h) / h.sum(),
            index=facilities,
        )
        .groupby(level=0)
        .sum()
    )
    return theta, contribution


def _difference_interval(
    first_scores: pd.DataFrame,
    first_facilities: np.ndarray,
    second_scores: pd.DataFrame,
    second_facilities: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, float | int]:
    theta1, contribution1 = _score_effect_contribution(first_scores, first_facilities)
    theta0, contribution0 = _score_effect_contribution(second_scores, second_facilities)
    union = contribution1.index.union(contribution0.index)
    contributions = contribution1.reindex(union, fill_value=0) - contribution0.reindex(
        union, fill_value=0
    )
    clusters = len(union)
    correction = clusters / (clusters - 1.0)
    se = float(np.sqrt(correction * np.square(contributions).sum()))
    critical = float(stats.t.ppf(0.5 + confidence_level / 2.0, df=clusters - 1))
    difference = theta1 - theta0
    return {
        "first_effect": theta1,
        "second_effect": theta0,
        "difference": difference,
        "se": se,
        "ci_low": difference - critical * se,
        "ci_high": difference + critical * se,
        "clusters": clusters,
    }


def _observation_diagnostics(
    data: RevisionData,
    propensity: np.ndarray,
    variant: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.frame
    central = data.central_outcome
    observed = frame[f"_outcome_{central}"].notna().to_numpy()
    pi = variant["observation_probability"].to_numpy()
    rows: list[dict[str, Any]] = []
    metrics = probability_metrics(observed.astype(int), pi)
    metrics.update({"variant": label, "stratum": "all"})
    rows.append(metrics)
    for arm in [0, 1]:
        mask = frame["_treatment"].to_numpy() == arm
        arm_metrics = probability_metrics(observed[mask].astype(int), pi[mask])
        arm_metrics.update({"variant": label, "stratum": f"treatment_{arm}"})
        rows.append(arm_metrics)

    distribution_rows = []
    for arm in [0, 1]:
        for status in [0, 1]:
            mask = (frame["_treatment"].to_numpy() == arm) & (
                observed.astype(int) == status
            )
            values = pi[mask]
            factor = 1.0 / values
            base_ow = 1.0 - propensity[mask] if arm == 1 else propensity[mask]
            combined = base_ow * factor
            distribution_rows.append(
                {
                    "variant": label,
                    "treatment": arm,
                    "outcome_observed": status,
                    "n": int(mask.sum()),
                    "pi_min": float(values.min()),
                    "pi_p1": float(np.quantile(values, 0.01)),
                    "pi_p5": float(np.quantile(values, 0.05)),
                    "pi_p50": float(np.quantile(values, 0.5)),
                    "pi_p95": float(np.quantile(values, 0.95)),
                    "pi_p99": float(np.quantile(values, 0.99)),
                    "pi_max": float(values.max()),
                    "factor_p50": float(np.quantile(factor, 0.5)),
                    "factor_p95": float(np.quantile(factor, 0.95)),
                    "factor_p99": float(np.quantile(factor, 0.99)),
                    "factor_max": float(factor.max()),
                    "combined_ess": effective_sample_size(combined),
                    "combined_max": float(combined.max()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(distribution_rows)


def _ipow_estimates_and_gates(
    data: RevisionData,
    main_prediction: pd.DataFrame,
    propensity: np.ndarray,
    effect_scores: pd.DataFrame,
    variant_prediction: pd.DataFrame,
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = data.frame
    config = data.config
    central = data.central_outcome
    outcome_column = f"_outcome_{central}"
    y = frame[outcome_column].to_numpy()
    observed = np.isfinite(y)
    a = frame["_treatment"].to_numpy()
    pi = variant_prediction["observation_probability"].to_numpy()
    mu0 = variant_prediction["mu0"].to_numpy()
    mu1 = variant_prediction["mu1"].to_numpy()
    cluster = frame["_facility"].to_numpy()
    confidence = float(config["inference"]["confidence_level"])

    factor = 1.0 / pi
    lower, upper = np.quantile(factor[observed], [0.01, 0.99])
    specifications = [
        ("canonical", pi, float("nan"), float("nan"), 0, 0),
        (
            "observation_factor_truncated_p1_p99",
            1.0 / np.clip(factor, lower, upper),
            lower,
            upper,
            int(((factor < lower) | (factor > upper)).sum()),
            int((observed & ((factor < lower) | (factor > upper))).sum()),
        ),
    ]
    estimate_rows = []
    gate_rows = []
    score_tables: dict[str, pd.DataFrame] = {}
    for (
        specification,
        effective_pi,
        low,
        high,
        truncated_n,
        truncated_observed_n,
    ) in specifications:
        scores = augmented_ipow_scores(
            a,
            y,
            observed,
            propensity,
            effective_pi,
            mu0,
            mu1,
        )
        score_tables[specification] = scores
        estimate = add_nnt_fields(
            estimate_clustered_ratio(
                scores, cluster, confidence_level=confidence
            ).as_dict()
        )
        actual_ow = np.where(a == 1, 1.0 - propensity, propensity)
        combined = actual_ow[observed] / effective_pi[observed]
        estimate.update(
            {
                "variant": label,
                "specification": specification,
                "outcome": central,
                "eligible_n": int(len(frame)),
                "observed_n": int(observed.sum()),
                "missing_n": int((~observed).sum()),
                "events": int(np.nansum(y)),
                "observation_factor_lower_cut": low,
                "observation_factor_upper_cut": high,
                "truncated_n_all_predictions": truncated_n,
                "truncated_n_observed": truncated_observed_n,
                "combined_ess_observed": effective_sample_size(combined),
                "combined_ess_treated": effective_sample_size(
                    combined[a[observed] == 1]
                ),
                "combined_ess_untreated": effective_sample_size(
                    combined[a[observed] == 0]
                ),
                "combined_max": float(combined.max()),
            }
        )
        estimate_rows.append(estimate)

        gates_frame = pd.concat(
            [
                effect_scores.reset_index(drop=True),
                scores.reset_index(drop=True),
                frame[["_facility", "_treatment", outcome_column]].reset_index(
                    drop=True
                ),
            ],
            axis=1,
        )
        for group in sorted(gates_frame["ite_group"].unique()):
            mask = gates_frame["ite_group"] == group
            subset = gates_frame.loc[mask]
            subset_scores = subset[["h", "psi1", "psi0"]]
            group_estimate = add_nnt_fields(
                estimate_clustered_ratio(
                    subset_scores,
                    subset["_facility"],
                    confidence_level=confidence,
                ).as_dict()
            )
            group_estimate.update(
                cluster_multiplier_intervals(
                    subset_scores,
                    subset["_facility"],
                    repetitions=int(
                        config["inference"]["cluster_multiplier_repetitions"]
                    ),
                    seed=_seed(
                        int(config["inference"]["cluster_multiplier_seed"]),
                        700,
                        int(group),
                        1 if specification == "canonical" else 2,
                        1 if label == "with_30d_auxiliary" else 0,
                    ),
                    confidence_level=confidence,
                )
            )
            group_estimate.update(
                {
                    "variant": label,
                    "specification": specification,
                    "group": int(group),
                    "n": int(mask.sum()),
                    "observed_n": int(subset[outcome_column].notna().sum()),
                    "treated_n": int(subset["_treatment"].sum()),
                    "events": int(subset[outcome_column].sum(skipna=True)),
                    "mean_predicted_ite": float(
                        np.average(subset["ite"], weights=subset["h"])
                    ),
                }
            )
            gate_rows.append(group_estimate)

    complete_case = main_prediction[outcome_column].notna()
    cc_work = main_prediction.loc[complete_case]
    cc_scores = augmented_overlap_scores(
        cc_work["_treatment"],
        cc_work[outcome_column],
        cc_work[f"propensity_{central}"],
        cc_work[f"mu0_{central}"],
        cc_work[f"mu1_{central}"],
    )
    differences = []
    for specification, scores in score_tables.items():
        comparison = _difference_interval(
            scores,
            cluster,
            cc_scores,
            cc_work["_facility"].to_numpy(),
            confidence_level=confidence,
        )
        comparison.update(
            {
                "variant": label,
                "specification": specification,
                "comparison": "IPOW minus complete-case ATO",
            }
        )
        differences.append(comparison)
    return (
        pd.DataFrame(estimate_rows),
        pd.DataFrame(gate_rows),
        pd.DataFrame(differences),
        score_tables["canonical"],
    )


def run_ipow_analysis(
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    data = prepare_revision_data(config_path)
    output = Path(data.config["output_dir"])
    marker = output / "ipow_complete.json"
    if marker.exists() and not force:
        LOGGER.info("IPOW analysis is already complete")
        return output
    main_prediction = pd.read_parquet(output / "main_predictions.parquet")
    folds = main_prediction["fold"].to_numpy(dtype=int)

    LOGGER.info("Cross-fitting full-cohort treatment propensity for IPOW")
    propensity = _crossfit_full_propensity(data, folds)
    effect_scores = _apply_main_effect_score_to_all(data, main_prediction, output)
    all_estimates = []
    all_gates = []
    all_differences = []
    all_metrics = []
    all_distributions = []
    prediction_output = pd.DataFrame(
        {
            "_row_id": data.frame["_row_id"],
            "fold": folds,
            "propensity": propensity,
            "ite": effect_scores["ite"],
            "ite_group": effect_scores["ite_group"],
        }
    )
    for label, include_auxiliary, offset in [
        ("with_30d_auxiliary", True, 50000),
        ("without_30d_auxiliary", False, 60000),
    ]:
        variant = _crossfit_ipow_variant(
            data,
            folds,
            include_30d_auxiliary=include_auxiliary,
            seed_offset=offset,
        )
        metrics, distribution = _observation_diagnostics(
            data, propensity, variant, label=label
        )
        estimates, gates, differences, canonical_scores = _ipow_estimates_and_gates(
            data,
            main_prediction,
            propensity,
            effect_scores,
            variant,
            label=label,
        )
        all_estimates.append(estimates)
        all_gates.append(gates)
        all_differences.append(differences)
        all_metrics.append(metrics)
        all_distributions.append(distribution)
        prediction_output[f"pi_{label}"] = variant["observation_probability"]
        prediction_output[f"mu0_{label}"] = variant["mu0"]
        prediction_output[f"mu1_{label}"] = variant["mu1"]

    prediction_output.to_parquet(output / "ipow_predictions.parquet", index=False)
    pd.concat(all_estimates, ignore_index=True).to_csv(
        output / "ipow_estimates.csv", index=False
    )
    pd.concat(all_gates, ignore_index=True).to_csv(
        output / "ipow_gates.csv", index=False
    )
    pd.concat(all_differences, ignore_index=True).to_csv(
        output / "ipow_complete_case_differences.csv", index=False
    )
    pd.concat(all_metrics, ignore_index=True).to_csv(
        output / "ipow_observation_model_metrics.csv", index=False
    )
    pd.concat(all_distributions, ignore_index=True).to_csv(
        output / "ipow_observation_diagnostics.csv", index=False
    )
    with marker.open("w", encoding="utf-8") as fp:
        json.dump({"status": "complete"}, fp, indent=2)
    LOGGER.info("IPOW analysis complete")
    return output
