from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .data import (
    RevisionData,
    cohort_audit,
    make_balanced_folds,
    prepare_revision_data,
)
from .diagnostics import (
    calibration_table,
    covariate_balance,
    probability_metrics,
    propensity_distribution,
    summarize_balance,
    weight_distribution,
)
from .estimators import (
    add_nnt_fields,
    assign_quantile_group,
    augmented_overlap_scores,
    clip_probability,
    cluster_multiplier_intervals,
    dr_pseudo_outcome,
    effective_sample_size,
    estimate_clustered_ratio,
    weighted_quantile,
)
from .models import fit_effect_model, fit_probability_model

LOGGER = logging.getLogger(__name__)


def _seed(base: int, *parts: int) -> int:
    value = int(base)
    for part in parts:
        value = (value * 1009 + int(part) * 9176 + 37) % (2**31 - 1)
    return value


def _distribution_row(
    values: np.ndarray,
    *,
    name: str,
    fold: int | str,
) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    row: dict[str, float | int | str] = {
        "quantity": name,
        "fold": fold,
        "n": int(len(values)),
    }
    for quantile, label in [
        (0.0, "min"),
        (0.01, "p1"),
        (0.05, "p5"),
        (0.5, "p50"),
        (0.95, "p95"),
        (0.99, "p99"),
        (1.0, "max"),
    ]:
        row[label] = (
            float(np.quantile(values, quantile)) if len(values) else float("nan")
        )
    row["n_below_minus_1"] = int((values < -1).sum())
    row["pct_below_minus_1"] = float(100 * (values < -1).mean()) if len(values) else 0
    row["n_above_1"] = int((values > 1).sum())
    row["pct_above_1"] = float(100 * (values > 1).mean()) if len(values) else 0
    return row


def _fit_outer_nuisance(
    data: RevisionData,
    folds: np.ndarray,
    *,
    features: list[str],
) -> pd.DataFrame:
    frame = data.frame
    config = data.config
    epsilon = float(config["probability_epsilon"])
    base_seed = int(config["seed"])
    outer_splits = int(config["folds"]["outer"])
    probability_params = config["lightgbm_probability"]

    prediction = pd.DataFrame(index=frame.index)
    prediction["_row_id"] = frame["_row_id"]
    prediction["fold"] = folds
    prediction["_facility"] = frame["_facility"]
    prediction["_year"] = frame["_year"]
    prediction["_treatment"] = frame["_treatment"]
    for key in data.outcomes:
        prediction[f"_outcome_{key}"] = frame[f"_outcome_{key}"]
        prediction[f"mu0_{key}"] = np.nan
        prediction[f"mu1_{key}"] = np.nan
        prediction[f"propensity_{key}"] = np.nan

    for outer in range(outer_splits):
        LOGGER.info(
            "Outer nuisance fold %s/%s",
            outer + 1,
            outer_splits,
        )
        for outcome_index, (key, spec) in enumerate(data.outcomes.items()):
            outcome_column = f"_outcome_{key}"
            train = frame.loc[(folds != outer) & frame[outcome_column].notna()]
            heldout = frame.loc[(folds == outer) & frame[outcome_column].notna()]
            ps_model = fit_probability_model(
                train,
                train["_treatment"],
                features=features,
                categorical=data.categorical_features,
                params=probability_params,
                seed=_seed(base_seed, 10, outer, outcome_index),
            )
            prediction.loc[heldout.index, f"propensity_{key}"] = clip_probability(
                ps_model.predict_probability(heldout), epsilon
            )
            for arm in [0, 1]:
                fit_mask = train["_treatment"] == arm
                model = fit_probability_model(
                    train.loc[fit_mask],
                    train.loc[fit_mask, outcome_column],
                    features=features,
                    categorical=data.categorical_features,
                    params=probability_params,
                    seed=_seed(base_seed, 20, outer, outcome_index, arm),
                )
                prediction.loc[heldout.index, f"mu{arm}_{key}"] = clip_probability(
                    model.predict_probability(heldout), epsilon
                )
    for key in data.outcomes:
        observed = prediction[f"_outcome_{key}"].notna()
        required = [f"propensity_{key}", f"mu0_{key}", f"mu1_{key}"]
        if prediction.loc[observed, required].isna().any().any():
            raise RuntimeError(f"Outer outcome cross-fitting failed for {key}")
    prediction["propensity"] = prediction[f"propensity_{data.central_outcome}"]
    return prediction


def _nested_central_effect(
    data: RevisionData,
    folds: np.ndarray,
    prediction: pd.DataFrame,
    *,
    features: list[str],
    model_directory: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = data.frame
    config = data.config
    central = data.central_outcome
    outcome_column = f"_outcome_{central}"
    outer_splits = int(config["folds"]["outer"])
    inner_splits = int(config["folds"]["inner"])
    base_seed = int(config["seed"])
    epsilon = float(config["probability_epsilon"])
    probability_params = config["lightgbm_probability"]
    effect_params = config["lightgbm_effect"]
    clip_low, clip_high = [float(x) for x in config["sensitivity"]["ite_clip"]]

    prediction["ite"] = np.nan
    prediction["ite_clipped"] = np.nan
    prediction["ite_group"] = pd.Series(pd.NA, index=prediction.index, dtype="Int64")
    prediction["ite_group_clipped"] = pd.Series(
        pd.NA, index=prediction.index, dtype="Int64"
    )
    boundary_rows: list[dict[str, float | int | str]] = []
    pseudo_rows: list[dict[str, float | int | str]] = []
    shap_rows: list[dict[str, float | int | str]] = []
    model_directory.mkdir(parents=True, exist_ok=True)

    for outer in range(outer_splits):
        train = frame.loc[(folds != outer) & frame[outcome_column].notna()].copy()
        heldout = frame.loc[(folds == outer) & frame[outcome_column].notna()].copy()
        inner_folds = make_balanced_folds(
            train,
            n_splits=inner_splits,
            seed=_seed(base_seed, 100, outer),
            outcome_column=outcome_column,
        )
        inner_e = np.full(len(train), np.nan)
        inner_mu0 = np.full(len(train), np.nan)
        inner_mu1 = np.full(len(train), np.nan)

        LOGGER.info(
            "Nested central-effect fold %s/%s: building inner nuisance predictions",
            outer + 1,
            outer_splits,
        )
        for inner in range(inner_splits):
            inner_train = train.iloc[inner_folds != inner]
            inner_heldout = train.iloc[inner_folds == inner]
            heldout_positions = np.flatnonzero(inner_folds == inner)
            ps_model = fit_probability_model(
                inner_train,
                inner_train["_treatment"],
                features=features,
                categorical=data.categorical_features,
                params=probability_params,
                seed=_seed(base_seed, 110, outer, inner),
            )
            inner_e[heldout_positions] = clip_probability(
                ps_model.predict_probability(inner_heldout), epsilon
            )
            for arm in [0, 1]:
                fit_mask = (inner_train["_treatment"] == arm) & inner_train[
                    outcome_column
                ].notna()
                outcome_model = fit_probability_model(
                    inner_train.loc[fit_mask],
                    inner_train.loc[fit_mask, outcome_column],
                    features=features,
                    categorical=data.categorical_features,
                    params=probability_params,
                    seed=_seed(base_seed, 120, outer, inner, arm),
                )
                target = inner_mu1 if arm == 1 else inner_mu0
                target[heldout_positions] = clip_probability(
                    outcome_model.predict_probability(inner_heldout), epsilon
                )

        if (
            np.isnan(inner_e).any()
            or np.isnan(inner_mu0).any()
            or np.isnan(inner_mu1).any()
        ):
            raise RuntimeError(f"Inner cross-fitting failed in outer fold {outer}")
        pseudo = dr_pseudo_outcome(
            train["_treatment"],
            train[outcome_column],
            inner_e,
            inner_mu0,
            inner_mu1,
        )
        h_inner = inner_e * (1.0 - inner_e)
        pseudo_rows.append(
            _distribution_row(
                pseudo,
                name="dr_pseudo_outcome",
                fold=outer,
            )
        )

        final_effect = fit_effect_model(
            train,
            pseudo,
            features=features,
            categorical=data.categorical_features,
            params=effect_params,
            seed=_seed(base_seed, 130, outer),
            sample_weight=h_inner,
        )
        heldout_score = final_effect.predict_effect(heldout)
        prediction.loc[heldout.index, "ite"] = heldout_score
        prediction.loc[heldout.index, "ite_clipped"] = np.clip(
            heldout_score, clip_low, clip_high
        )

        # Cross-fitted effect scores inside the outer training set are used only
        # to learn group boundaries.  Each inner heldout patient is excluded from
        # the effect model that generates its boundary-training score.
        train_score_oof = np.full(len(train), np.nan)
        for inner in range(inner_splits):
            effect_train_mask = inner_folds != inner
            effect_heldout_positions = np.flatnonzero(inner_folds == inner)
            effect_model = fit_effect_model(
                train.iloc[np.flatnonzero(effect_train_mask)],
                pseudo[effect_train_mask],
                features=features,
                categorical=data.categorical_features,
                params=effect_params,
                seed=_seed(base_seed, 140, outer, inner),
                sample_weight=h_inner[effect_train_mask],
            )
            train_score_oof[effect_heldout_positions] = effect_model.predict_effect(
                train.iloc[effect_heldout_positions]
            )
        if np.isnan(train_score_oof).any():
            raise RuntimeError(f"Inner effect cross-fitting failed in fold {outer}")

        for kind, boundary_score, heldout_values, target_column in [
            ("unclipped", train_score_oof, heldout_score, "ite_group"),
            (
                "clipped",
                np.clip(train_score_oof, clip_low, clip_high),
                np.clip(heldout_score, clip_low, clip_high),
                "ite_group_clipped",
            ),
        ]:
            boundaries = weighted_quantile(
                boundary_score,
                [0.2, 0.4, 0.6, 0.8],
                h_inner,
            )
            for quantile, boundary in zip([0.2, 0.4, 0.6, 0.8], boundaries):
                boundary_rows.append(
                    {
                        "fold": outer,
                        "prediction": kind,
                        "quantile": quantile,
                        "boundary": float(boundary),
                    }
                )
            prediction.loc[heldout.index, target_column] = assign_quantile_group(
                heldout_values, boundaries
            )

        # LightGBM pred_contrib gives exact TreeSHAP values for every raw feature.
        encoded_heldout = final_effect.encoded(heldout)
        contributions = final_effect.estimator.booster_.predict(
            encoded_heldout, pred_contrib=True
        )
        fold_h = prediction.loc[heldout.index, "propensity"].to_numpy()
        fold_h = fold_h * (1.0 - fold_h)
        for feature_index, feature in enumerate(features):
            importance = np.average(
                np.abs(contributions[:, feature_index]),
                weights=fold_h,
            )
            shap_rows.append(
                {
                    "fold": outer,
                    "feature": feature,
                    "weighted_mean_abs_shap": float(importance),
                }
            )

        with (model_directory / f"effect_fold_{outer}.pickle").open("wb") as fp:
            pickle.dump(final_effect, fp)

    central_observed = prediction[f"_outcome_{central}"].notna()
    if prediction.loc[central_observed, ["ite", "ite_clipped"]].isna().any().any():
        raise RuntimeError("Nested effect fitting left missing held-out predictions")
    return (
        prediction,
        pd.DataFrame(boundary_rows),
        pd.DataFrame(pseudo_rows),
        pd.DataFrame(shap_rows),
    )


def _main_estimates(data: RevisionData, prediction: pd.DataFrame) -> pd.DataFrame:
    rows = []
    confidence = float(data.config["inference"]["confidence_level"])
    for key, spec in data.outcomes.items():
        outcome_column = f"_outcome_{key}"
        observed = prediction[outcome_column].notna()
        subset = prediction.loc[observed]
        scores = augmented_overlap_scores(
            subset["_treatment"],
            subset[outcome_column],
            subset[f"propensity_{key}"],
            subset[f"mu0_{key}"],
            subset[f"mu1_{key}"],
        )
        estimate = add_nnt_fields(
            estimate_clustered_ratio(
                scores,
                subset["_facility"],
                confidence_level=confidence,
            ).as_dict()
        )
        estimate.update(
            {
                "outcome": key,
                "outcome_label": spec.label,
                "method": "cross-fitted augmented overlap weighting",
                "estimand": "ATO",
                "eligible_n": int(len(prediction)),
                "observed_n": int(observed.sum()),
                "missing_n": int((~observed).sum()),
                "treated_n": int(subset["_treatment"].sum()),
                "untreated_n": int((1 - subset["_treatment"]).sum()),
                "events": int(subset[outcome_column].sum()),
            }
        )
        rows.append(estimate)
    return pd.DataFrame(rows)


def _model_diagnostics(
    data: RevisionData,
    prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for key, spec in data.outcomes.items():
        outcome_column = f"_outcome_{key}"
        observed = prediction[outcome_column].notna()
        ps_metrics = probability_metrics(
            prediction.loc[observed, "_treatment"],
            prediction.loc[observed, f"propensity_{key}"],
        )
        ps_metrics.update(
            {
                "model": "propensity",
                "outcome": key,
                "outcome_label": spec.label,
                "arm": "all",
            }
        )
        metric_rows.append(ps_metrics)
        for arm in [0, 1]:
            mask = (prediction["_treatment"] == arm) & prediction[
                outcome_column
            ].notna()
            metrics = probability_metrics(
                prediction.loc[mask, outcome_column],
                prediction.loc[mask, f"mu{arm}_{key}"],
            )
            metrics.update(
                {
                    "model": f"mu{arm}",
                    "outcome": key,
                    "outcome_label": spec.label,
                    "arm": arm,
                }
            )
            metric_rows.append(metrics)

    central = data.central_outcome
    calibration_rows = []
    for arm in [0, 1]:
        outcome_column = f"_outcome_{central}"
        mask = (prediction["_treatment"] == arm) & prediction[outcome_column].notna()
        table = calibration_table(
            prediction.loc[mask, outcome_column],
            prediction.loc[mask, f"mu{arm}_{central}"],
        )
        table["outcome"] = central
        table["arm"] = arm
        calibration_rows.append(table)
    return pd.DataFrame(metric_rows), pd.concat(calibration_rows, ignore_index=True)


def _design_diagnostics(
    data: RevisionData,
    prediction: pd.DataFrame,
    *,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ps_rows = []
    weight_rows = []
    balance_rows = []
    balance_summary_rows = []
    frame = data.frame
    for key, spec in data.outcomes.items():
        outcome_column = f"_outcome_{key}"
        observed = prediction[outcome_column].notna().to_numpy()
        a = prediction.loc[observed, "_treatment"].to_numpy()
        e = prediction.loc[observed, f"propensity_{key}"].to_numpy()
        ps = propensity_distribution(a, e)
        ps["outcome"] = key
        ps["outcome_label"] = spec.label
        ps_rows.append(ps)
        weights = weight_distribution(a, e)
        weights["outcome"] = key
        weights["outcome_label"] = spec.label
        weight_rows.append(weights)

        observed_frame = frame.loc[observed]
        unweighted = covariate_balance(
            observed_frame,
            a,
            features=features,
            categorical=data.categorical_features,
            method="unweighted",
        )
        overlap_weights = np.where(a == 1, 1.0 - e, e)
        overlap = covariate_balance(
            observed_frame,
            a,
            features=features,
            categorical=data.categorical_features,
            weights=overlap_weights,
            method="overlap_weighted",
        )
        combined = pd.concat([unweighted, overlap], ignore_index=True)
        combined["outcome"] = key
        combined["outcome_label"] = spec.label
        balance_rows.append(combined)
        summary = summarize_balance(combined)
        summary["outcome"] = key
        summary["outcome_label"] = spec.label
        balance_summary_rows.append(summary)
    return (
        pd.concat(ps_rows, ignore_index=True),
        pd.concat(weight_rows, ignore_index=True),
        pd.concat(balance_rows, ignore_index=True),
        pd.concat(balance_summary_rows, ignore_index=True),
    )


def _gates(
    data: RevisionData,
    prediction: pd.DataFrame,
    *,
    group_column: str,
    prediction_column: str,
    label: str,
) -> pd.DataFrame:
    central = data.central_outcome
    outcome_column = f"_outcome_{central}"
    observed = prediction[outcome_column].notna()
    work = prediction.loc[observed].copy()
    scores = augmented_overlap_scores(
        work["_treatment"],
        work[outcome_column],
        work["propensity"],
        work[f"mu0_{central}"],
        work[f"mu1_{central}"],
    )
    work = pd.concat([work.reset_index(drop=True), scores], axis=1)
    inference = data.config["inference"]
    rows = []
    for group in sorted(work[group_column].unique()):
        mask = work[group_column] == group
        subset = work.loc[mask]
        subset_scores = subset[["h", "psi1", "psi0"]]
        estimate = add_nnt_fields(
            estimate_clustered_ratio(
                subset_scores,
                subset["_facility"],
                confidence_level=float(inference["confidence_level"]),
            ).as_dict()
        )
        multiplier = cluster_multiplier_intervals(
            subset_scores,
            subset["_facility"],
            repetitions=int(inference["cluster_multiplier_repetitions"]),
            seed=_seed(
                int(inference["cluster_multiplier_seed"]),
                int(group),
                1 if label == "clipped" else 0,
            ),
            confidence_level=float(inference["confidence_level"]),
        )
        a = subset["_treatment"].to_numpy()
        e = subset["propensity"].to_numpy()
        ow = np.where(a == 1, 1.0 - e, e)
        estimate.update(multiplier)
        estimate.update(
            {
                "prediction": label,
                "group": int(group),
                "n": int(len(subset)),
                "treated_n": int(a.sum()),
                "untreated_n": int((1 - a).sum()),
                "events": int(subset[outcome_column].sum()),
                "treated_events": int(
                    subset.loc[subset["_treatment"] == 1, outcome_column].sum()
                ),
                "untreated_events": int(
                    subset.loc[subset["_treatment"] == 0, outcome_column].sum()
                ),
                "overlap_weight_sum": float(ow.sum()),
                "ess_overall": effective_sample_size(ow),
                "ess_treated": effective_sample_size(ow[a == 1]),
                "ess_untreated": effective_sample_size(ow[a == 0]),
                "mean_predicted_ite": float(
                    np.average(subset[prediction_column], weights=subset["h"])
                ),
            }
        )
        rows.append(estimate)
    return pd.DataFrame(rows)


def _ite_sensitivity(
    data: RevisionData,
    prediction: pd.DataFrame,
    pseudo_diagnostics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    central_column = f"_outcome_{data.central_outcome}"
    observed = prediction[central_column].notna()
    diagnostics = [pseudo_diagnostics]
    for fold in sorted(prediction["fold"].unique()):
        diagnostics.append(
            pd.DataFrame(
                [
                    _distribution_row(
                        prediction.loc[observed & (prediction["fold"] == fold), "ite"],
                        name="heldout_predicted_ite",
                        fold=int(fold),
                    )
                ]
            )
        )
    diagnostics.append(
        pd.DataFrame(
            [
                _distribution_row(
                    prediction.loc[observed, "ite"],
                    name="heldout_predicted_ite",
                    fold="all_observed",
                ),
                _distribution_row(
                    prediction["ite"],
                    name="heldout_predicted_ite",
                    fold="all_score_available",
                ),
            ]
        )
    )
    diagnostics_table = pd.concat(diagnostics, ignore_index=True)

    rank_correlation = stats.spearmanr(
        prediction.loc[observed, "ite"],
        prediction.loc[observed, "ite_clipped"],
    )
    moved = (
        prediction.loc[observed, "ite_group"]
        != prediction.loc[observed, "ite_group_clipped"]
    )
    comparison = pd.DataFrame(
        [
            {
                "n": int(observed.sum()),
                "spearman_rho": float(rank_correlation.statistic),
                "spearman_p": float(rank_correlation.pvalue),
                "quintile_moved_n": int(moved.sum()),
                "quintile_moved_pct": float(100 * moved.mean()),
            }
        ]
    )

    threshold_rows = []
    thresholds = data.config["sensitivity"]["ite_thresholds"]
    work = prediction.loc[observed]
    for mode, score_column in [("unclipped", "ite"), ("clipped", "ite_clipped")]:
        for threshold in thresholds:
            for arm, relation, mask in [
                (
                    1,
                    "treated_with_score_at_or_below_threshold",
                    (work["_treatment"] == 1) & (work[score_column] <= threshold),
                ),
                (
                    0,
                    "untreated_with_score_above_threshold",
                    (work["_treatment"] == 0) & (work[score_column] > threshold),
                ),
            ]:
                denominator_mask = work["_treatment"] == arm
                denominator = int(denominator_mask.sum())
                numerator = int(mask.sum())
                e = work.loc[denominator_mask, "propensity"].to_numpy()
                arm_weights = 1.0 - e if arm == 1 else e
                selected = mask.loc[denominator_mask].to_numpy()
                weighted_pct = (
                    100 * arm_weights[selected].sum() / arm_weights.sum()
                    if arm_weights.sum() > 0
                    else float("nan")
                )
                threshold_rows.append(
                    {
                        "prediction": mode,
                        "threshold": float(threshold),
                        "treatment": arm,
                        "description": relation,
                        "numerator": numerator,
                        "denominator": denominator,
                        "raw_percent": 100 * numerator / denominator,
                        "overlap_weighted_percent": weighted_pct,
                    }
                )
    return diagnostics_table, comparison, pd.DataFrame(threshold_rows)


def _shap_stability(shap_fold: pd.DataFrame) -> pd.DataFrame:
    work = shap_fold.copy()
    work["rank"] = work.groupby("fold")["weighted_mean_abs_shap"].rank(
        method="min", ascending=False
    )
    work["top_10"] = work["rank"] <= 10
    summary = (
        work.groupby("feature", as_index=False)
        .agg(
            median_weighted_mean_abs_shap=("weighted_mean_abs_shap", "median"),
            min_weighted_mean_abs_shap=("weighted_mean_abs_shap", "min"),
            max_weighted_mean_abs_shap=("weighted_mean_abs_shap", "max"),
            median_rank=("rank", "median"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            top_10_folds=("top_10", "sum"),
        )
        .sort_values(
            ["median_weighted_mean_abs_shap", "top_10_folds"],
            ascending=[False, False],
        )
    )
    return summary


def run_main_analysis(
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    data = prepare_revision_data(config_path)
    output = Path(data.config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "main_predictions.parquet"
    complete_marker = output / "main_complete.json"
    if complete_marker.exists() and predictions_path.exists() and not force:
        LOGGER.info("Main analysis is already complete at %s", output)
        return output

    audit = cohort_audit(data)
    with (output / "cohort_audit.json").open("w", encoding="utf-8") as fp:
        json.dump(audit, fp, ensure_ascii=False, indent=2)
    with Path(config_path).open(encoding="utf-8") as source:
        (output / "analysis_spec.yaml").write_text(source.read(), encoding="utf-8")

    central_column = f"_outcome_{data.central_outcome}"
    folds = make_balanced_folds(
        data.frame,
        n_splits=int(data.config["folds"]["outer"]),
        seed=int(data.config["seed"]),
        outcome_column=central_column,
    )
    prediction = _fit_outer_nuisance(data, folds, features=data.main_features)
    prediction, boundaries, pseudo, shap_fold = _nested_central_effect(
        data,
        folds,
        prediction,
        features=data.main_features,
        model_directory=output / "models" / "main_effect",
    )

    main_estimates = _main_estimates(data, prediction)
    model_metrics, calibration = _model_diagnostics(data, prediction)
    ps_distribution, weight_diagnostics, balance, balance_summary = _design_diagnostics(
        data, prediction, features=data.main_features
    )
    gates = pd.concat(
        [
            _gates(
                data,
                prediction,
                group_column="ite_group",
                prediction_column="ite",
                label="unclipped",
            ),
            _gates(
                data,
                prediction,
                group_column="ite_group_clipped",
                prediction_column="ite_clipped",
                label="clipped",
            ),
        ],
        ignore_index=True,
    )
    ite_diagnostics, ite_comparison, thresholds = _ite_sensitivity(
        data, prediction, pseudo
    )
    shap_summary = _shap_stability(shap_fold)

    prediction.to_parquet(predictions_path, index=False)
    tables = {
        "main_estimates.csv": main_estimates,
        "model_diagnostics.csv": model_metrics,
        "central_outcome_calibration.csv": calibration,
        "propensity_distribution.csv": ps_distribution,
        "weight_diagnostics.csv": weight_diagnostics,
        "covariate_balance.csv": balance,
        "balance_summary.csv": balance_summary,
        "ite_fold_boundaries.csv": boundaries,
        "ite_diagnostics.csv": ite_diagnostics,
        "ite_clipping_comparison.csv": ite_comparison,
        "gates.csv": gates,
        "threshold_scenarios.csv": thresholds,
        "shap_fold_importance.csv": shap_fold,
        "shap_stability.csv": shap_summary,
    }
    for filename, table in tables.items():
        table.to_csv(output / filename, index=False)

    marker = {
        "status": "complete",
        "rows": int(len(prediction)),
        "central_outcome": data.central_outcome,
        "main_features": len(data.main_features),
    }
    with complete_marker.open("w", encoding="utf-8") as fp:
        json.dump(marker, fp, ensure_ascii=False, indent=2)
    LOGGER.info("Main analysis complete: %s", output)
    return output
