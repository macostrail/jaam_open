from __future__ import annotations

import copy
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (
    _design_diagnostics,
    _fit_outer_nuisance,
    _gates,
    _main_estimates,
    _nested_central_effect,
    _seed,
)
from .data import make_balanced_folds, prepare_revision_data
from .estimators import (
    add_nnt_fields,
    assign_quantile_group,
    augmented_overlap_scores,
    dr_pseudo_outcome,
    estimate_clustered_ratio,
    weighted_quantile,
)
from .ipow import (
    _crossfit_full_propensity,
    _crossfit_ipow_variant,
    _ipow_estimates_and_gates,
    _observation_diagnostics,
)
from .models import fit_effect_model
from .diagnostics import covariate_balance, summarize_balance
from .sensitivity import _crossfit_logistic_propensity

LOGGER = logging.getLogger(__name__)


def _period_data(data, frame, *, seed_offset: int):
    local = copy.deepcopy(data)
    local.frame = frame.copy().reset_index(drop=True)
    central = data.central_outcome
    local.outcomes = {central: data.outcomes[central]}
    local.config["seed"] = int(data.config["seed"]) + seed_offset
    return local


def run_temporal_validation(
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    data = prepare_revision_data(config_path)
    output = Path(data.config["output_dir"])
    marker = output / "temporal_complete.json"
    if marker.exists() and not force:
        LOGGER.info("Temporal validation is already complete")
        return output

    temporal_cfg = data.config["temporal_validation"]
    development_years = set(int(x) for x in temporal_cfg["development_years"])
    validation_year = int(temporal_cfg["validation_year"])
    development_frame = data.frame.loc[data.frame["_year"].isin(development_years)]
    validation_frame = data.frame.loc[data.frame["_year"] == validation_year]
    central = data.central_outcome
    outcome_column = f"_outcome_{central}"

    development = _period_data(data, development_frame, seed_offset=70000)
    validation = _period_data(data, validation_frame, seed_offset=80000)
    LOGGER.info(
        "Temporal split: development=%s, validation=%s",
        len(development.frame),
        len(validation.frame),
    )

    development_folds = make_balanced_folds(
        development.frame,
        n_splits=int(development.config["folds"]["outer"]),
        seed=int(development.config["seed"]),
        outcome_column=outcome_column,
    )
    development_prediction = _fit_outer_nuisance(
        development,
        development_folds,
        features=development.main_features,
    )
    (
        development_prediction,
        development_fold_boundaries,
        development_pseudo_diagnostics,
        development_shap,
    ) = _nested_central_effect(
        development,
        development_folds,
        development_prediction,
        features=development.main_features,
        model_directory=output / "models" / "temporal_development_crossfit",
    )
    development_gates = _gates(
        development,
        development_prediction,
        group_column="ite_group",
        prediction_column="ite",
        label="development_cross_fitted",
    )
    development_estimate = _main_estimates(development, development_prediction).assign(
        period="development_2014_2018"
    )

    # Fit the transport model on every observed development-period outcome, using
    # cross-fitted nuisance predictions to construct its DR target.
    development_observed = development_prediction[outcome_column].notna()
    dev_work = development_prediction.loc[development_observed]
    dev_e = dev_work[f"propensity_{central}"].to_numpy()
    dev_h = dev_e * (1.0 - dev_e)
    dev_pseudo = dr_pseudo_outcome(
        dev_work["_treatment"],
        dev_work[outcome_column],
        dev_e,
        dev_work[f"mu0_{central}"],
        dev_work[f"mu1_{central}"],
    )
    final_effect = fit_effect_model(
        development.frame.loc[development_observed],
        dev_pseudo,
        features=development.main_features,
        categorical=development.categorical_features,
        params=development.config["lightgbm_effect"],
        seed=_seed(int(development.config["seed"]), 710),
        sample_weight=dev_h,
    )
    development_apparent_score = final_effect.predict_effect(
        development.frame.loc[development_observed]
    )
    fixed_boundaries = weighted_quantile(
        development_apparent_score,
        [0.2, 0.4, 0.6, 0.8],
        dev_h,
    )
    fixed_boundary_table = pd.DataFrame(
        {
            "quantile": [0.2, 0.4, 0.6, 0.8],
            "boundary": fixed_boundaries,
            "source": "full development model applied to development complete cases",
        }
    )
    with (output / "models" / "temporal_development_final_effect.pickle").open(
        "wb"
    ) as fp:
        pickle.dump(final_effect, fp)

    validation_score = final_effect.predict_effect(validation.frame)
    validation_group = assign_quantile_group(validation_score, fixed_boundaries)
    validation_folds = make_balanced_folds(
        validation.frame,
        n_splits=int(validation.config["folds"]["outer"]),
        seed=int(validation.config["seed"]),
        outcome_column=outcome_column,
    )
    validation_prediction = _fit_outer_nuisance(
        validation,
        validation_folds,
        features=validation.main_features,
    )
    validation_prediction["ite"] = validation_score
    validation_prediction["ite_group"] = validation_group
    validation_gates = _gates(
        validation,
        validation_prediction,
        group_column="ite_group",
        prediction_column="ite",
        label="validation_fixed_development_model",
    )
    validation_estimate = _main_estimates(validation, validation_prediction).assign(
        period=f"validation_{validation_year}"
    )
    val_ps, val_weights, val_balance, val_balance_summary = _design_diagnostics(
        validation,
        validation_prediction,
        features=validation.main_features,
    )

    # The validation period is much smaller, so a prespecified regularized
    # logistic PS is evaluated alongside LightGBM using the same outcome models
    # and fixed development-derived groups.
    validation_logistic = _crossfit_logistic_propensity(
        validation,
        validation_prediction,
        features=validation.main_features,
    )
    validation_logistic_prediction = validation_prediction.copy()
    validation_logistic_prediction[f"propensity_{central}"] = validation_logistic[
        f"propensity_{central}"
    ]
    validation_logistic_prediction["propensity"] = validation_logistic[
        f"propensity_{central}"
    ]
    observed_validation = validation_logistic_prediction[outcome_column].notna()
    val_log_work = validation_logistic_prediction.loc[observed_validation]
    val_log_scores = augmented_overlap_scores(
        val_log_work["_treatment"],
        val_log_work[outcome_column],
        val_log_work["propensity"],
        val_log_work[f"mu0_{central}"],
        val_log_work[f"mu1_{central}"],
    )
    validation_logistic_estimate = add_nnt_fields(
        estimate_clustered_ratio(
            val_log_scores,
            val_log_work["_facility"],
            confidence_level=float(validation.config["inference"]["confidence_level"]),
        ).as_dict()
    )
    validation_logistic_estimate.update(
        {
            "period": f"validation_{validation_year}",
            "method": "regularized logistic PS, augmented OW",
            "observed_n": int(observed_validation.sum()),
            "treated_n": int(val_log_work["_treatment"].sum()),
            "untreated_n": int((1 - val_log_work["_treatment"]).sum()),
            "events": int(val_log_work[outcome_column].sum()),
        }
    )
    validation_logistic_gates = _gates(
        validation,
        validation_logistic_prediction,
        group_column="ite_group",
        prediction_column="ite",
        label="validation_fixed_development_model_logistic_ps",
    )
    val_log_e = val_log_work["propensity"].to_numpy()
    val_log_a = val_log_work["_treatment"].to_numpy(dtype=int)
    val_log_ow = np.where(val_log_a == 1, 1.0 - val_log_e, val_log_e)
    validation_logistic_balance = covariate_balance(
        validation.frame.loc[observed_validation],
        val_log_a,
        features=validation.main_features,
        categorical=validation.categorical_features,
        weights=val_log_ow,
        method="regularized_logistic_overlap",
    )
    validation_logistic_balance_summary = summarize_balance(validation_logistic_balance)

    # Missing-outcome sensitivity within the 2019 validation period.
    validation_full_propensity = _crossfit_full_propensity(validation, validation_folds)
    validation_ipow_prediction = _crossfit_ipow_variant(
        validation,
        validation_folds,
        include_30d_auxiliary=True,
        seed_offset=81000,
    )
    validation_effect_scores = pd.DataFrame(
        {
            "_row_id": validation.frame["_row_id"],
            "fold": validation_folds,
            "ite": validation_score,
            "ite_group": validation_group,
        }
    )
    (
        validation_ipow_estimates,
        validation_ipow_gates,
        validation_ipow_differences,
        _,
    ) = _ipow_estimates_and_gates(
        validation,
        validation_prediction,
        validation_full_propensity,
        validation_effect_scores,
        validation_ipow_prediction,
        label=f"validation_{validation_year}_with_30d_auxiliary",
    )
    validation_ipow_metrics, validation_ipow_diagnostics = _observation_diagnostics(
        validation,
        validation_full_propensity,
        validation_ipow_prediction,
        label=f"validation_{validation_year}_with_30d_auxiliary",
    )

    development_prediction.to_parquet(
        output / "temporal_development_predictions.parquet", index=False
    )
    validation_prediction.to_parquet(
        output / "temporal_validation_predictions.parquet", index=False
    )
    pd.concat([development_estimate, validation_estimate], ignore_index=True).to_csv(
        output / "temporal_overall_estimates.csv", index=False
    )
    development_gates.assign(period="development_2014_2018").to_csv(
        output / "temporal_development_gates.csv", index=False
    )
    validation_gates.assign(period=f"validation_{validation_year}").to_csv(
        output / "temporal_validation_gates.csv", index=False
    )
    pd.DataFrame([validation_logistic_estimate]).to_csv(
        output / "temporal_validation_logistic_estimate.csv", index=False
    )
    validation_logistic_gates.to_csv(
        output / "temporal_validation_logistic_gates.csv", index=False
    )
    validation_logistic_balance_summary.to_csv(
        output / "temporal_validation_logistic_balance_summary.csv",
        index=False,
    )
    fixed_boundary_table.to_csv(output / "temporal_fixed_boundaries.csv", index=False)
    development_fold_boundaries.to_csv(
        output / "temporal_development_fold_boundaries.csv", index=False
    )
    development_pseudo_diagnostics.to_csv(
        output / "temporal_development_pseudo_diagnostics.csv", index=False
    )
    development_shap.to_csv(output / "temporal_development_shap.csv", index=False)
    val_ps.to_csv(output / "temporal_validation_ps.csv", index=False)
    val_weights.to_csv(output / "temporal_validation_weights.csv", index=False)
    val_balance_summary.to_csv(
        output / "temporal_validation_balance_summary.csv", index=False
    )
    validation_ipow_estimates.to_csv(
        output / "temporal_validation_ipow_estimates.csv", index=False
    )
    validation_ipow_gates.to_csv(
        output / "temporal_validation_ipow_gates.csv", index=False
    )
    validation_ipow_differences.to_csv(
        output / "temporal_validation_ipow_differences.csv", index=False
    )
    validation_ipow_metrics.to_csv(
        output / "temporal_validation_ipow_metrics.csv", index=False
    )
    validation_ipow_diagnostics.to_csv(
        output / "temporal_validation_ipow_diagnostics.csv", index=False
    )
    with marker.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "status": "complete",
                "development_n": int(len(development.frame)),
                "validation_n": int(len(validation.frame)),
                "validation_year": validation_year,
            },
            fp,
            indent=2,
        )
    LOGGER.info("Temporal validation complete")
    return output
