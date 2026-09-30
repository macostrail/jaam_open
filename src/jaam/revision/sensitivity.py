from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    _design_diagnostics,
    _fit_outer_nuisance,
    _gates,
    _main_estimates,
    _model_diagnostics,
    _nested_central_effect,
    _seed,
)
from .data import RevisionData, make_balanced_folds, prepare_revision_data
from .diagnostics import (
    covariate_balance,
    probability_metrics,
    propensity_distribution,
    summarize_balance,
    weight_distribution,
)
from .estimators import (
    add_nnt_fields,
    aipw_scores,
    augmented_overlap_scores,
    clip_probability,
    effective_sample_size,
    estimate_clustered_ratio,
    estimate_clustered_two_ratios,
)
from .models import make_logistic_propensity_pipeline

LOGGER = logging.getLogger(__name__)


def _local_data(
    data: RevisionData,
    *,
    frame: pd.DataFrame,
    outcome_keys: list[str],
) -> RevisionData:
    local = copy.deepcopy(data)
    local.frame = frame.copy().reset_index(drop=True)
    local.outcomes = {key: data.outcomes[key] for key in outcome_keys}
    if local.central_outcome not in local.outcomes:
        local.central_outcome = outcome_keys[0]
    return local


def _population_run(
    data: RevisionData,
    *,
    frame: pd.DataFrame,
    outcome_keys: list[str],
    features: list[str],
    output: Path,
    label: str,
    include_ite: bool,
    seed_offset: int,
) -> dict[str, pd.DataFrame]:
    local = _local_data(data, frame=frame, outcome_keys=outcome_keys)
    local.config["seed"] = int(data.config["seed"]) + seed_offset
    outcome_column = f"_outcome_{local.central_outcome}"
    folds = make_balanced_folds(
        local.frame,
        n_splits=int(local.config["folds"]["outer"]),
        seed=int(local.config["seed"]),
        outcome_column=outcome_column,
    )
    prediction = _fit_outer_nuisance(local, folds, features=features)
    results: dict[str, pd.DataFrame] = {}
    if include_ite:
        prediction, boundaries, pseudo, shap = _nested_central_effect(
            local,
            folds,
            prediction,
            features=features,
            model_directory=output / "models" / label,
        )
        results["boundaries"] = boundaries
        results["pseudo"] = pseudo
        results["shap"] = shap
        results["gates"] = _gates(
            local,
            prediction,
            group_column="ite_group",
            prediction_column="ite",
            label="unclipped",
        )
    results["predictions"] = prediction
    results["estimates"] = _main_estimates(local, prediction)
    metrics, calibration = _model_diagnostics(local, prediction)
    results["metrics"] = metrics
    results["calibration"] = calibration
    ps, weights, balance, balance_summary = _design_diagnostics(
        local, prediction, features=features
    )
    results["ps"] = ps
    results["weights"] = weights
    results["balance"] = balance
    results["balance_summary"] = balance_summary
    return results


def _crossfit_logistic_propensity(
    data: RevisionData,
    main_prediction: pd.DataFrame,
    *,
    features: list[str],
) -> pd.DataFrame:
    frame = data.frame
    config = data.config
    folds = main_prediction["fold"].to_numpy(dtype=int)
    result = pd.DataFrame({"_row_id": frame["_row_id"]})
    numeric = [col for col in features if col in data.numeric_features]
    categorical = [col for col in features if col in data.categorical_features]
    logistic_cfg = config["logistic_propensity"]
    epsilon = float(config["probability_epsilon"])

    for outcome_index, key in enumerate(data.outcomes):
        outcome_column = f"_outcome_{key}"
        result[f"propensity_{key}"] = np.nan
        for fold in sorted(np.unique(folds)):
            train_mask = (folds != fold) & frame[outcome_column].notna().to_numpy()
            heldout_mask = (folds == fold) & frame[outcome_column].notna().to_numpy()
            model = make_logistic_propensity_pipeline(
                numeric=numeric,
                categorical=categorical,
                c_value=float(logistic_cfg["C"]),
                max_iter=int(logistic_cfg["max_iter"]),
                seed=_seed(int(config["seed"]), 500, outcome_index, int(fold)),
            )
            model.fit(
                frame.loc[train_mask, features],
                frame.loc[train_mask, "_treatment"],
            )
            result.loc[heldout_mask, f"propensity_{key}"] = clip_probability(
                model.predict_proba(frame.loc[heldout_mask, features])[:, 1],
                epsilon,
            )
        observed = frame[outcome_column].notna()
        if result.loc[observed, f"propensity_{key}"].isna().any():
            raise RuntimeError(f"Logistic PS cross-fitting failed for {key}")
    return result


def _ps_method_sensitivities(
    data: RevisionData,
    main_prediction: pd.DataFrame,
    logistic_prediction: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    estimate_rows: list[dict[str, Any]] = []
    ps_rows: list[pd.DataFrame] = []
    balance_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    confidence = float(data.config["inference"]["confidence_level"])

    main_estimates = pd.read_csv(Path(data.config["output_dir"]) / "main_estimates.csv")
    for key, spec in data.outcomes.items():
        outcome_column = f"_outcome_{key}"
        observed = main_prediction[outcome_column].notna()
        work = main_prediction.loc[observed].copy()
        frame = data.frame.loc[observed]
        a = work["_treatment"].to_numpy(dtype=int)
        y = work[outcome_column].to_numpy(dtype=float)
        e_main = work[f"propensity_{key}"].to_numpy(dtype=float)
        mu0 = work[f"mu0_{key}"].to_numpy(dtype=float)
        mu1 = work[f"mu1_{key}"].to_numpy(dtype=float)
        cluster = work["_facility"].to_numpy()

        primary = main_estimates.loc[main_estimates["outcome"] == key].iloc[0].to_dict()
        primary.update(
            {
                "method_family": "overlap_weighting",
                "method_detail": "LightGBM PS, augmented OW",
                "target_population": "overlap population",
                "excluded_n": 0,
            }
        )
        estimate_rows.append(primary)

        logistic_e = logistic_prediction.loc[observed, f"propensity_{key}"].to_numpy(
            dtype=float
        )
        logistic_scores = augmented_overlap_scores(a, y, logistic_e, mu0, mu1)
        logistic_estimate = add_nnt_fields(
            estimate_clustered_ratio(
                logistic_scores, cluster, confidence_level=confidence
            ).as_dict()
        )
        logistic_estimate.update(
            {
                "outcome": key,
                "outcome_label": spec.label,
                "method_family": "overlap_weighting",
                "method_detail": "regularized logistic PS, augmented OW",
                "target_population": "alternative overlap population",
                "observed_n": int(len(work)),
                "excluded_n": 0,
                "treated_n": int(a.sum()),
                "untreated_n": int((1 - a).sum()),
                "events": int(y.sum()),
            }
        )
        estimate_rows.append(logistic_estimate)

        metrics = probability_metrics(a, logistic_e)
        metrics.update(
            {
                "outcome": key,
                "outcome_label": spec.label,
                "ps_model": "regularized logistic",
            }
        )
        metric_rows.append(metrics)
        ps = propensity_distribution(a, logistic_e)
        ps["outcome"] = key
        ps["outcome_label"] = spec.label
        ps["ps_model"] = "regularized logistic"
        ps_rows.append(ps)
        logistic_ow = np.where(a == 1, 1.0 - logistic_e, logistic_e)
        balance = covariate_balance(
            frame,
            a,
            features=data.main_features,
            categorical=data.categorical_features,
            weights=logistic_ow,
            method="regularized_logistic_overlap",
        )
        balance["outcome"] = key
        balance["outcome_label"] = spec.label
        balance_rows.append(balance)

        for lower, upper in data.config["sensitivity"]["propensity_trimming"]:
            keep = (e_main >= float(lower)) & (e_main <= float(upper))
            trimmed_scores = aipw_scores(
                a[keep], y[keep], e_main[keep], mu0[keep], mu1[keep]
            )
            estimate = add_nnt_fields(
                estimate_clustered_ratio(
                    trimmed_scores,
                    cluster[keep],
                    confidence_level=confidence,
                ).as_dict()
            )
            ipw = np.where(
                a[keep] == 1,
                1.0 / e_main[keep],
                1.0 / (1.0 - e_main[keep]),
            )
            trimmed_prevalence = float(a[keep].mean())
            trimmed_stabilized = np.where(
                a[keep] == 1,
                trimmed_prevalence / e_main[keep],
                (1.0 - trimmed_prevalence) / (1.0 - e_main[keep]),
            )
            estimate.update(
                {
                    "outcome": key,
                    "outcome_label": spec.label,
                    "method_family": "PS-trimmed AIPW",
                    "method_detail": f"PS in [{lower}, {upper}]",
                    "target_population": "PS-trimmed population",
                    "observed_n": int(len(work)),
                    "analysis_n": int(keep.sum()),
                    "excluded_n": int((~keep).sum()),
                    "excluded_treated_n": int(((~keep) & (a == 1)).sum()),
                    "excluded_untreated_n": int(((~keep) & (a == 0)).sum()),
                    "treated_n": int(a[keep].sum()),
                    "untreated_n": int((1 - a[keep]).sum()),
                    "events": int(y[keep].sum()),
                    "ess_treated": effective_sample_size(ipw[a[keep] == 1]),
                    "ess_untreated": effective_sample_size(ipw[a[keep] == 0]),
                    "max_weight": float(ipw.max()),
                }
            )
            estimate_rows.append(estimate)
            trimmed_balance = covariate_balance(
                frame.loc[keep],
                a[keep],
                features=data.main_features,
                categorical=data.categorical_features,
                weights=trimmed_stabilized,
                method=f"PS_trim_{lower}_{upper}",
            )
            trimmed_balance["outcome"] = key
            trimmed_balance["outcome_label"] = spec.label
            balance_rows.append(trimmed_balance)

        prevalence = float(a.mean())
        stabilized = np.where(
            a == 1,
            prevalence / e_main,
            (1.0 - prevalence) / (1.0 - e_main),
        )
        truncation_specs: list[tuple[str, float | None, float | None]] = [
            ("untruncated", None, None)
        ]
        truncation_specs.extend(
            [
                (f"percentile_{low}_{high}", float(low), float(high))
                for low, high in data.config["sensitivity"][
                    "iptw_truncation_percentiles"
                ]
            ]
        )
        for detail, low, high in truncation_specs:
            if low is None:
                weights = stabilized.copy()
                low_cut = high_cut = float("nan")
            else:
                low_cut, high_cut = np.quantile(stabilized, [low, high])
                weights = np.clip(stabilized, low_cut, high_cut)
            numerator1 = a * weights * y
            denominator1 = a * weights
            numerator0 = (1 - a) * weights * y
            denominator0 = (1 - a) * weights
            estimate = add_nnt_fields(
                estimate_clustered_two_ratios(
                    numerator1,
                    denominator1,
                    numerator0,
                    denominator0,
                    cluster,
                    confidence_level=confidence,
                ).as_dict()
            )
            estimate.update(
                {
                    "outcome": key,
                    "outcome_label": spec.label,
                    "method_family": "stabilized IPTW",
                    "method_detail": detail,
                    "target_population": "weighted full complete-case population",
                    "observed_n": int(len(work)),
                    "analysis_n": int(len(work)),
                    "excluded_n": 0,
                    "treated_n": int(a.sum()),
                    "untreated_n": int((1 - a).sum()),
                    "events": int(y.sum()),
                    "lower_weight_cut": low_cut,
                    "upper_weight_cut": high_cut,
                    "ess_treated": effective_sample_size(weights[a == 1]),
                    "ess_untreated": effective_sample_size(weights[a == 0]),
                    "max_weight": float(weights.max()),
                }
            )
            estimate_rows.append(estimate)
            iptw_balance = covariate_balance(
                frame,
                a,
                features=data.main_features,
                categorical=data.categorical_features,
                weights=weights,
                method=f"IPTW_{detail}",
            )
            iptw_balance["outcome"] = key
            iptw_balance["outcome_label"] = spec.label
            balance_rows.append(iptw_balance)

    all_balance = pd.concat(balance_rows, ignore_index=True)
    return (
        pd.DataFrame(estimate_rows),
        pd.concat(ps_rows, ignore_index=True),
        all_balance,
        summarize_balance(
            all_balance.assign(
                method=all_balance["method"] + "|" + all_balance["outcome"]
            )
        ),
        pd.DataFrame(metric_rows),
    )


def _logistic_population_reestimate(
    data: RevisionData,
    prediction: pd.DataFrame,
    *,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logistic = _crossfit_logistic_propensity(data, prediction, features=features)
    estimate_rows = []
    balance_rows = []
    confidence = float(data.config["inference"]["confidence_level"])
    for key, spec in data.outcomes.items():
        outcome_column = f"_outcome_{key}"
        observed = prediction[outcome_column].notna()
        work = prediction.loc[observed]
        e = logistic.loc[observed, f"propensity_{key}"].to_numpy()
        a = work["_treatment"].to_numpy(dtype=int)
        scores = augmented_overlap_scores(
            a,
            work[outcome_column],
            e,
            work[f"mu0_{key}"],
            work[f"mu1_{key}"],
        )
        estimate = add_nnt_fields(
            estimate_clustered_ratio(
                scores,
                work["_facility"],
                confidence_level=confidence,
            ).as_dict()
        )
        estimate.update(
            {
                "outcome": key,
                "outcome_label": spec.label,
                "method": "regularized logistic PS, augmented OW",
                "observed_n": int(observed.sum()),
                "treated_n": int(a.sum()),
                "untreated_n": int((1 - a).sum()),
                "events": int(work[outcome_column].sum()),
            }
        )
        estimate_rows.append(estimate)
        ow = np.where(a == 1, 1.0 - e, e)
        balance = covariate_balance(
            data.frame.loc[observed],
            a,
            features=features,
            categorical=data.categorical_features,
            weights=ow,
            method=f"regularized_logistic_overlap|{key}",
        )
        balance_rows.append(balance)
    all_balance = pd.concat(balance_rows, ignore_index=True)
    return (
        pd.DataFrame(estimate_rows),
        all_balance,
        summarize_balance(all_balance),
    )


def _subgroup_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    rosc = frame["病院収容後の心拍再開"].isin([1, 2])
    no_rosc = frame["病院収容後の心拍再開"] == 3
    vf = frame["波形種別"] == 1
    shockable = frame["波形種別"].isin([1, 2])
    pea = frame["波形種別"] == 3
    asystole = frame["波形種別"] == 4
    st_elevation = frame["心拍再開後のＳＴ上昇の有無"] == 1
    no_st_elevation = frame["心拍再開後のＳＴ上昇の有無"] == 2
    return {
        "all": pd.Series(True, index=frame.index),
        "no_ROSC": no_rosc,
        "ROSC": rosc,
        "ROSC_shockable": rosc & shockable,
        "ROSC_ST_elevation": rosc & st_elevation,
        "ROSC_no_ST_elevation": rosc & no_st_elevation,
        "ROSC_VF": rosc & vf,
        "ROSC_PEA": rosc & pea,
        "ROSC_asystole": rosc & asystole,
        "ROSC_VF_no_ST_elevation": rosc & vf & no_st_elevation,
    }


def _effect_contribution(
    scores: pd.DataFrame,
    facilities: pd.Series,
) -> tuple[float, pd.Series]:
    effect = scores["psi1"].to_numpy() - scores["psi0"].to_numpy()
    h = scores["h"].to_numpy()
    theta = float(effect.sum() / h.sum())
    contribution = (
        pd.Series(
            (effect - theta * h) / h.sum(),
            index=np.asarray(facilities),
        )
        .groupby(level=0)
        .sum()
    )
    return theta, contribution


def _interaction_contrast(
    prediction: pd.DataFrame,
    *,
    outcome_key: str,
    mask1: pd.Series,
    mask0: pd.Series,
    confidence_level: float,
) -> dict[str, float | int]:
    outcome_column = f"_outcome_{outcome_key}"
    available = prediction[outcome_column].notna()

    def group(mask: pd.Series) -> tuple[float, pd.Series]:
        subset = prediction.loc[available & mask]
        scores = augmented_overlap_scores(
            subset["_treatment"],
            subset[outcome_column],
            subset[f"propensity_{outcome_key}"],
            subset[f"mu0_{outcome_key}"],
            subset[f"mu1_{outcome_key}"],
        )
        return _effect_contribution(scores, subset["_facility"])

    theta1, contribution1 = group(mask1)
    theta0, contribution0 = group(mask0)
    union = contribution1.index.union(contribution0.index)
    contrast_contribution = contribution1.reindex(
        union, fill_value=0
    ) - contribution0.reindex(union, fill_value=0)
    clusters = len(union)
    correction = clusters / (clusters - 1.0)
    se = float(np.sqrt(correction * np.square(contrast_contribution).sum()))
    critical = float(
        stats.t.ppf(
            0.5 + confidence_level / 2.0,
            df=clusters - 1,
        )
    )
    contrast = theta1 - theta0
    return {
        "effect_group1": theta1,
        "effect_group0": theta0,
        "difference_in_effects": contrast,
        "se_difference": se,
        "ci_low": contrast - critical * se,
        "ci_high": contrast + critical * se,
        "p_value": (
            float(2 * stats.t.sf(abs(contrast / se), df=clusters - 1))
            if se > 0
            else float("nan")
        ),
        "clusters": clusters,
    }


def _subgroup_analysis(
    data: RevisionData,
    prediction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    masks = _subgroup_masks(data.frame)
    estimate_rows: list[dict[str, Any]] = []
    ps_rows: list[pd.DataFrame] = []
    weight_rows: list[pd.DataFrame] = []
    balance_rows: list[pd.DataFrame] = []
    confidence = float(data.config["inference"]["confidence_level"])
    for key, spec in data.outcomes.items():
        outcome_column = f"_outcome_{key}"
        for name, mask in masks.items():
            available = mask & prediction[outcome_column].notna()
            subset = prediction.loc[available]
            if len(subset) == 0 or subset["_treatment"].nunique() < 2:
                continue
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
            a = subset["_treatment"].to_numpy(dtype=int)
            e = subset[f"propensity_{key}"].to_numpy(dtype=float)
            ow = np.where(a == 1, 1.0 - e, e)
            estimate.update(
                {
                    "outcome": key,
                    "outcome_label": spec.label,
                    "subgroup": name,
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
                    "ess_overall": effective_sample_size(ow),
                    "ess_treated": effective_sample_size(ow[a == 1]),
                    "ess_untreated": effective_sample_size(ow[a == 0]),
                }
            )
            estimate_rows.append(estimate)

            if key == data.central_outcome:
                ps = propensity_distribution(a, e)
                ps["subgroup"] = name
                ps_rows.append(ps)
                weights = weight_distribution(a, e)
                weights["subgroup"] = name
                weight_rows.append(weights)
                balance = covariate_balance(
                    data.frame.loc[available],
                    a,
                    features=data.main_features,
                    categorical=data.categorical_features,
                    weights=ow,
                    method=f"overlap_weighted|{name}",
                )
                balance["subgroup"] = name
                balance_rows.append(balance)
    balance_all = pd.concat(balance_rows, ignore_index=True)
    return (
        pd.DataFrame(estimate_rows),
        pd.concat(ps_rows, ignore_index=True),
        pd.concat(weight_rows, ignore_index=True),
        summarize_balance(balance_all),
    )


def _interaction_analysis(
    data: RevisionData,
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    masks = _subgroup_masks(data.frame)
    pairs = [
        ("ROSC_vs_no_ROSC", masks["ROSC"], masks["no_ROSC"]),
        (
            "ROSC_VF_vs_ROSC_PEA",
            masks["ROSC_VF"],
            masks["ROSC_PEA"],
        ),
        (
            "ROSC_VF_vs_ROSC_asystole",
            masks["ROSC_VF"],
            masks["ROSC_asystole"],
        ),
        (
            "ROSC_ST_elevation_vs_no_ST_elevation",
            masks["ROSC_ST_elevation"],
            masks["ROSC_no_ST_elevation"],
        ),
    ]
    rows = []
    for key, spec in data.outcomes.items():
        for label, mask1, mask0 in pairs:
            result = _interaction_contrast(
                prediction,
                outcome_key=key,
                mask1=mask1,
                mask0=mask0,
                confidence_level=float(data.config["inference"]["confidence_level"]),
            )
            result.update(
                {
                    "outcome": key,
                    "outcome_label": spec.label,
                    "contrast": label,
                }
            )
            rows.append(result)
    return pd.DataFrame(rows)


def _no_rosc_pathway_description(data: RevisionData) -> pd.DataFrame:
    frame = data.frame
    no_rosc = frame["病院収容後の心拍再開"] == 3
    variables = [
        "体外循環治療(ECLS/ECMO/PCPS)施行",
        "PCI施行",
        "心拍再開後のＳＴ上昇の有無",
        "波形種別",
    ]
    rows = []
    for arm in [0, 1]:
        subset = frame.loc[no_rosc & (frame["_treatment"] == arm)]
        for variable in variables:
            normalized = (
                subset[variable]
                .astype("object")
                .where(subset[variable].notna(), "__MISSING__")
            )
            for level, count in normalized.value_counts(dropna=False).items():
                rows.append(
                    {
                        "population": "no_ROSC",
                        "treatment": arm,
                        "variable": variable,
                        "level": str(level),
                        "count": int(count),
                        "denominator": int(len(subset)),
                        "percent": float(100 * count / len(subset)),
                    }
                )
    return pd.DataFrame(rows)


def run_sensitivity_analysis(
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    data = prepare_revision_data(config_path)
    output = Path(data.config["output_dir"])
    marker = output / "sensitivity_complete.json"
    if marker.exists() and not force:
        LOGGER.info("Sensitivity analysis is already complete")
        return output
    main_prediction = pd.read_parquet(output / "main_predictions.parquet")
    if not np.array_equal(
        main_prediction["_row_id"].to_numpy(),
        data.frame["_row_id"].to_numpy(),
    ):
        raise RuntimeError("Main predictions do not align with the prepared cohort")

    LOGGER.info("Cross-fitting alternative logistic propensity scores")
    logistic_prediction = _crossfit_logistic_propensity(
        data, main_prediction, features=data.main_features
    )
    logistic_prediction.to_parquet(
        output / "logistic_propensity_predictions.parquet", index=False
    )
    (
        method_estimates,
        logistic_ps,
        method_balance,
        method_balance_summary,
        logistic_metrics,
    ) = _ps_method_sensitivities(data, main_prediction, logistic_prediction)

    central = data.central_outcome
    LOGGER.info("Running restricted early-baseline central-outcome analysis")
    early = _population_run(
        data,
        frame=data.frame,
        outcome_keys=[central],
        features=data.early_features,
        output=output,
        label="early_baseline",
        include_ite=False,
        seed_offset=10000,
    )

    LOGGER.info("Running facility-identifier nuisance-model sensitivity")
    facility_frame = data.frame.copy()
    facility_feature = "__facility_identifier"
    facility_frame[facility_feature] = facility_frame["_facility"]
    facility_data = copy.deepcopy(data)
    facility_data.categorical_features.add(facility_feature)
    facility_features = [*data.main_features, facility_feature]
    facility = _population_run(
        facility_data,
        frame=facility_frame,
        outcome_keys=[central],
        features=facility_features,
        output=output,
        label="facility_identifier",
        include_ite=False,
        seed_offset=20000,
    )

    rosc = data.frame["病院収容後の心拍再開"].isin([1, 2])
    shockable = data.frame["波形種別"].isin([1, 2])
    LOGGER.info("Running ROSC-restricted re-estimation")
    rosc_run = _population_run(
        data,
        frame=data.frame.loc[rosc],
        outcome_keys=list(data.outcomes),
        features=data.main_features,
        output=output,
        label="rosc",
        include_ite=True,
        seed_offset=30000,
    )
    LOGGER.info("Running ROSC-plus-shockable re-estimation")
    shockable_run = _population_run(
        data,
        frame=data.frame.loc[rosc & shockable],
        outcome_keys=list(data.outcomes),
        features=data.main_features,
        output=output,
        label="rosc_shockable",
        include_ite=True,
        seed_offset=40000,
    )
    rosc_local = _local_data(
        data,
        frame=data.frame.loc[rosc],
        outcome_keys=list(data.outcomes),
    )
    shockable_local = _local_data(
        data,
        frame=data.frame.loc[rosc & shockable],
        outcome_keys=list(data.outcomes),
    )
    rosc_logistic = _logistic_population_reestimate(
        rosc_local,
        rosc_run["predictions"],
        features=rosc_local.main_features,
    )
    shockable_logistic = _logistic_population_reestimate(
        shockable_local,
        shockable_run["predictions"],
        features=shockable_local.main_features,
    )

    subgroup, subgroup_ps, subgroup_weights, subgroup_balance = _subgroup_analysis(
        data, main_prediction
    )
    interactions = _interaction_analysis(data, main_prediction)
    pathway = _no_rosc_pathway_description(data)

    method_estimates.to_csv(output / "sensitivity_estimates.csv", index=False)
    logistic_ps.to_csv(output / "logistic_ps_distribution.csv", index=False)
    logistic_metrics.to_csv(output / "logistic_ps_metrics.csv", index=False)
    method_balance.to_csv(output / "sensitivity_balance.csv", index=False)
    method_balance_summary.to_csv(
        output / "sensitivity_balance_summary.csv", index=False
    )
    early["estimates"].assign(analysis="early_baseline").to_csv(
        output / "early_baseline_estimates.csv", index=False
    )
    early["balance_summary"].to_csv(
        output / "early_baseline_balance_summary.csv", index=False
    )
    early["ps"].to_csv(output / "early_baseline_ps.csv", index=False)
    facility["estimates"].assign(analysis="facility_identifier").to_csv(
        output / "facility_identifier_estimates.csv", index=False
    )
    facility["balance_summary"].to_csv(
        output / "facility_identifier_balance_summary.csv", index=False
    )
    facility["ps"].to_csv(output / "facility_identifier_ps.csv", index=False)

    for label, result in [("rosc", rosc_run), ("rosc_shockable", shockable_run)]:
        result["predictions"].to_parquet(
            output / f"{label}_predictions.parquet", index=False
        )
        for key in [
            "estimates",
            "metrics",
            "ps",
            "weights",
            "balance_summary",
            "gates",
            "boundaries",
            "pseudo",
            "shap",
        ]:
            if key in result:
                result[key].to_csv(output / f"{label}_{key}.csv", index=False)
    for label, result in [
        ("rosc", rosc_logistic),
        ("rosc_shockable", shockable_logistic),
    ]:
        estimates, balance, balance_summary = result
        estimates.to_csv(output / f"{label}_logistic_estimates.csv", index=False)
        balance.to_csv(output / f"{label}_logistic_balance.csv", index=False)
        balance_summary.to_csv(
            output / f"{label}_logistic_balance_summary.csv", index=False
        )

    subgroup.to_csv(output / "clinical_subgroup_estimates.csv", index=False)
    subgroup_ps.to_csv(output / "clinical_subgroup_ps.csv", index=False)
    subgroup_weights.to_csv(output / "clinical_subgroup_weights.csv", index=False)
    subgroup_balance.to_csv(
        output / "clinical_subgroup_balance_summary.csv", index=False
    )
    interactions.to_csv(output / "clinical_subgroup_interactions.csv", index=False)
    pathway.to_csv(output / "no_rosc_pathway_description.csv", index=False)

    with marker.open("w", encoding="utf-8") as fp:
        json.dump({"status": "complete"}, fp, indent=2)
    LOGGER.info("Sensitivity analyses complete")
    return output
