from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import miceforest as mf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .data import prepare_revision_data

LOGGER = logging.getLogger(__name__)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _prepare_mi_frame(data, main_prediction):
    config = data.config["multiple_imputation"]
    central = data.central_outcome
    outcome_column = f"_outcome_{central}"
    mask = (data.frame["_treatment"] == 1) & data.frame[outcome_column].notna()
    source = data.frame.loc[mask].copy().reset_index(drop=True)
    prediction = main_prediction.loc[mask].reset_index(drop=True)
    selected = _unique(
        [
            *config["model3_predictors"],
            *config["auxiliary_predictors"],
        ]
    )
    missing_columns = sorted(set(selected).difference(source.columns))
    if missing_columns:
        raise ValueError(f"MI predictors are absent: {missing_columns}")

    frame = source[selected].copy()
    frame["__mi_facility"] = source["_facility"].to_numpy()
    frame["__mi_year"] = source["_year"].to_numpy()
    frame["__outcome"] = source[outcome_column].astype(int).to_numpy()
    frame["__ite"] = prediction["ite"].to_numpy(dtype=float)

    categorical = set(data.categorical_features).intersection(frame.columns)
    categorical.add("__mi_facility")
    for column in categorical:
        frame[column] = frame[column].astype("category")
    for column in frame.columns:
        if column not in categorical:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

    dropped = []
    for column in list(frame.columns):
        if column in {"__outcome", "__ite"}:
            continue
        if frame[column].notna().sum() == 0:
            dropped.append(
                {
                    "variable": column,
                    "reason": "all missing in CAG-treated complete-outcome subset",
                }
            )
            frame.drop(columns=column, inplace=True)
        elif frame[column].nunique(dropna=True) <= 1:
            dropped.append(
                {
                    "variable": column,
                    "reason": "constant in CAG-treated complete-outcome subset",
                }
            )
            frame.drop(columns=column, inplace=True)

    required_model3 = set(config["model3_predictors"])
    missing_model3 = sorted(required_model3.difference(frame.columns))
    if missing_model3:
        raise ValueError(
            "Model 3 predictors were removed as all-missing or constant: "
            f"{missing_model3}"
        )
    return frame, source, prediction, categorical, pd.DataFrame(dropped)


def _fixed_design_metadata(frame: pd.DataFrame, config: dict[str, Any]):
    categorical_model3 = [
        column
        for column in config["model3_predictors"]
        if str(frame[column].dtype) == "category"
    ]
    categories = {
        column: frame[column].cat.categories.tolist() for column in categorical_model3
    }
    numeric_model3 = [
        column
        for column in config["model3_predictors"]
        if column not in categorical_model3
    ]
    standardizers = {}
    for column in numeric_model3:
        observed = pd.to_numeric(frame[column], errors="coerce")
        mean = float(observed.mean())
        sd = float(observed.std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        standardizers[column] = (mean, sd)
    return categorical_model3, categories, numeric_model3, standardizers


def _categorical_dummies(
    frame: pd.DataFrame,
    column: str,
    categories: list[Any],
) -> pd.DataFrame:
    values = pd.Series(
        pd.Categorical(frame[column], categories=categories),
        index=frame.index,
    )
    return pd.get_dummies(
        values,
        prefix=column,
        prefix_sep="=",
        drop_first=True,
        dtype=float,
    )


def _build_designs(
    completed: pd.DataFrame,
    *,
    config: dict[str, Any],
    categorical_model3: list[str],
    categories: dict[str, list[Any]],
    numeric_model3: list[str],
    standardizers: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    outcome = pd.to_numeric(completed["__outcome"], errors="raise").to_numpy(dtype=int)
    ite_scale = float(config["ite_scale_percentage_points"]) / 100.0
    base = pd.DataFrame(
        {
            "ite_per_5pp": pd.to_numeric(completed["__ite"], errors="raise").to_numpy()
            / ite_scale
        },
        index=completed.index,
    )

    age_column = "年齢(JAAM)"
    sex_column = "性別"
    model2 = base.copy()
    model2["age_per_10y"] = (
        pd.to_numeric(completed[age_column], errors="raise").to_numpy() / 10.0
    )
    sex_dummies = _categorical_dummies(completed, sex_column, categories[sex_column])
    model2 = pd.concat([model2, sex_dummies], axis=1)

    model3 = base.copy()
    for column in config["model3_predictors"]:
        if column in categorical_model3:
            model3 = pd.concat(
                [
                    model3,
                    _categorical_dummies(completed, column, categories[column]),
                ],
                axis=1,
            )
        else:
            values = pd.to_numeric(completed[column], errors="raise").to_numpy()
            if column == age_column:
                model3["age_per_10y"] = values / 10.0
            else:
                mean, sd = standardizers[column]
                model3[f"{column}_per_SD"] = (values - mean) / sd

    designs = {
        "Model 1: ITE": sm.add_constant(base, has_constant="add"),
        "Model 2: ITE + age + sex": sm.add_constant(model2, has_constant="add"),
        "Model 3: prespecified prognostic covariates": sm.add_constant(
            model3, has_constant="add"
        ),
    }
    for name, design in designs.items():
        if design.isna().any().any():
            raise RuntimeError(f"{name} design still contains missing values")
    return outcome, designs


def _fit_models(
    kernel: mf.ImputationKernel,
    *,
    datasets: int,
    config: dict[str, Any],
    categorical_model3: list[str],
    categories: dict[str, list[Any]],
    numeric_model3: list[str],
    standardizers: dict[str, tuple[float, float]],
    original_missing: dict[str, np.ndarray],
) -> tuple[
    dict[str, list[pd.Series]],
    dict[str, list[pd.DataFrame]],
    pd.DataFrame,
]:
    coefficients: dict[str, list[pd.Series]] = {}
    covariances: dict[str, list[pd.DataFrame]] = {}
    distribution_rows = []
    for dataset_index in range(datasets):
        completed = kernel.complete_data(dataset=dataset_index).copy()
        outcome, designs = _build_designs(
            completed,
            config=config,
            categorical_model3=categorical_model3,
            categories=categories,
            numeric_model3=numeric_model3,
            standardizers=standardizers,
        )
        for model_name, design in designs.items():
            fit = sm.GLM(
                outcome,
                design,
                family=sm.families.Binomial(),
            ).fit()
            coefficients.setdefault(model_name, []).append(fit.params)
            covariances.setdefault(model_name, []).append(fit.cov_params())

        for variable in config["model3_predictors"]:
            missing = original_missing[variable]
            if not missing.any():
                continue
            values = completed.loc[missing, variable]
            if variable in categorical_model3:
                counts = values.value_counts(normalize=True, dropna=False)
                for level, proportion in counts.items():
                    distribution_rows.append(
                        {
                            "dataset": dataset_index,
                            "variable": variable,
                            "summary": f"imputed_proportion_level={level}",
                            "value": float(proportion),
                        }
                    )
            else:
                numeric = pd.to_numeric(values, errors="coerce")
                for summary, value in [
                    ("imputed_mean", numeric.mean()),
                    ("imputed_sd", numeric.std(ddof=1)),
                    ("imputed_p05", numeric.quantile(0.05)),
                    ("imputed_p50", numeric.quantile(0.50)),
                    ("imputed_p95", numeric.quantile(0.95)),
                ]:
                    distribution_rows.append(
                        {
                            "dataset": dataset_index,
                            "variable": variable,
                            "summary": summary,
                            "value": float(value),
                        }
                    )
    return coefficients, covariances, pd.DataFrame(distribution_rows)


def _rubin_pool(
    coefficients: dict[str, list[pd.Series]],
    covariances: dict[str, list[pd.DataFrame]],
) -> pd.DataFrame:
    rows = []
    for model_name, model_coefficients in coefficients.items():
        terms = model_coefficients[0].index
        q = np.stack(
            [
                series.reindex(terms).to_numpy(dtype=float)
                for series in model_coefficients
            ]
        )
        u = np.stack(
            [
                covariance.reindex(index=terms, columns=terms).to_numpy(dtype=float)
                for covariance in covariances[model_name]
            ]
        )
        m = q.shape[0]
        q_bar = q.mean(axis=0)
        u_bar_matrix = u.mean(axis=0)
        between_matrix = np.cov(q, rowvar=False, ddof=1)
        if between_matrix.ndim == 0:
            between_matrix = np.asarray([[between_matrix]])
        total_matrix = u_bar_matrix + (1.0 + 1.0 / m) * between_matrix

        for index, term in enumerate(terms):
            within = float(u_bar_matrix[index, index])
            between = float(between_matrix[index, index])
            total = float(total_matrix[index, index])
            se = float(np.sqrt(max(total, 0.0)))
            if between > 0:
                relative_increase = (1.0 + 1.0 / m) * between / within
                degrees_freedom = float((m - 1) * (1.0 + 1.0 / relative_increase) ** 2)
            else:
                relative_increase = 0.0
                degrees_freedom = float("inf")
            critical = (
                float(stats.t.ppf(0.975, degrees_freedom))
                if np.isfinite(degrees_freedom)
                else float(stats.norm.ppf(0.975))
            )
            statistic = q_bar[index] / se if se > 0 else float("nan")
            p_value = (
                float(2 * stats.t.sf(abs(statistic), degrees_freedom))
                if np.isfinite(degrees_freedom)
                else float(2 * stats.norm.sf(abs(statistic)))
            )
            lower = q_bar[index] - critical * se
            upper = q_bar[index] + critical * se
            mcse = float(np.sqrt(max(between, 0.0) / m))
            rows.append(
                {
                    "model": model_name,
                    "term": term,
                    "imputations": m,
                    "coefficient": float(q_bar[index]),
                    "standard_error": se,
                    "degrees_freedom": degrees_freedom,
                    "ci_low": lower,
                    "ci_high": upper,
                    "odds_ratio": float(np.exp(q_bar[index])),
                    "or_ci_low": float(np.exp(lower)),
                    "or_ci_high": float(np.exp(upper)),
                    "p_value": p_value,
                    "within_variance": within,
                    "between_variance": between,
                    "total_variance": total,
                    "relative_increase_variance": relative_increase,
                    "fraction_missing_information": (
                        (1.0 + 1.0 / m) * between / total if total > 0 else float("nan")
                    ),
                    "monte_carlo_error": mcse,
                    "mcse_to_standard_error": mcse / se if se > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        rows.append(
            {
                "variable": column,
                "n": int(len(frame)),
                "missing_n": int(frame[column].isna().sum()),
                "missing_percent": float(100 * frame[column].isna().mean()),
                "dtype": str(frame[column].dtype),
                "observed_unique": int(frame[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missing_percent", "variable"], ascending=[False, True]
    )


def _run_kernel(
    frame: pd.DataFrame,
    *,
    datasets: int,
    iterations: int,
    seed: int,
    save_all_iterations_data: bool = False,
) -> mf.ImputationKernel:
    mean_match_strategy = {
        column: "fast"
        for column in frame.columns
        if str(frame[column].dtype) == "category"
    }
    kernel = mf.ImputationKernel(
        data=frame,
        num_datasets=datasets,
        mean_match_candidates=5,
        mean_match_strategy=mean_match_strategy,
        save_all_iterations_data=save_all_iterations_data,
        copy_data=True,
        random_state=seed,
    )
    kernel.mice(
        iterations=iterations,
        verbose=False,
        num_iterations=20,
        max_depth=6,
        num_leaves=31,
        min_data_in_leaf=20,
        bagging_fraction=0.8,
        feature_fraction_bynode=0.8,
        bagging_freq=1,
        num_threads=8,
        verbosity=-1,
    )
    return kernel


def run_mi_convergence_diagnostic(
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    data = prepare_revision_data(config_path)
    output = Path(data.config["output_dir"])
    target = output / "mi"
    marker = target / "convergence_complete.json"
    if marker.exists() and not force:
        LOGGER.info("MI convergence diagnostic is already complete")
        return output
    main_prediction = pd.read_parquet(output / "main_predictions.parquet")
    mi_frame, _, _, _, _ = _prepare_mi_frame(data, main_prediction)
    config = data.config["multiple_imputation"]
    datasets = int(config["convergence_diagnostic_datasets"])
    iterations = int(config["iterations"])
    LOGGER.info(
        "MI convergence diagnostic: datasets=%s, iterations=%s",
        datasets,
        iterations,
    )
    kernel = _run_kernel(
        mi_frame,
        datasets=datasets,
        iterations=iterations,
        seed=int(config["seed"]),
        save_all_iterations_data=True,
    )
    rows = []
    for variable, values in kernel.imputation_values.items():
        if str(mi_frame[variable].dtype) == "category":
            continue
        for iteration, dataset in values.columns:
            if int(iteration) == 0:
                continue
            numeric = pd.to_numeric(values[(iteration, dataset)], errors="coerce")
            rows.append(
                {
                    "variable": variable,
                    "dataset": int(dataset),
                    "iteration": int(iteration),
                    "imputed_mean": float(numeric.mean()),
                    "imputed_sd": float(numeric.std(ddof=1)),
                    "imputed_n": int(numeric.notna().sum()),
                }
            )
    convergence = pd.DataFrame(rows).sort_values(["variable", "dataset", "iteration"])
    summaries = []
    for variable, group in convergence.groupby("variable"):
        final_iteration = int(group["iteration"].max())
        final = group[group["iteration"] == final_iteration]
        recent = group[group["iteration"] > final_iteration - 5]
        by_dataset = recent.groupby("dataset")["imputed_mean"]
        changes = by_dataset.max() - by_dataset.min()
        summaries.append(
            {
                "variable": variable,
                "missing_n": int(mi_frame[variable].isna().sum()),
                "missing_percent": float(100 * mi_frame[variable].isna().mean()),
                "datasets": datasets,
                "iterations": iterations,
                "final_mean_across_datasets": float(final["imputed_mean"].mean()),
                "final_sd_across_dataset_means": float(
                    final["imputed_mean"].std(ddof=1)
                ),
                "max_within_dataset_mean_range_last_5_iterations": float(changes.max()),
            }
        )
    target.mkdir(parents=True, exist_ok=True)
    convergence.to_csv(target / "convergence_diagnostics.csv", index=False)
    pd.DataFrame(summaries).sort_values("missing_percent", ascending=False).to_csv(
        target / "convergence_summary.csv", index=False
    )
    with marker.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "status": "complete",
                "datasets": datasets,
                "iterations": iterations,
            },
            fp,
            indent=2,
        )
    LOGGER.info("MI convergence diagnostic complete")
    return output


def run_multiple_imputation(
    config_path: str | Path,
    *,
    force: bool = False,
    smoke: bool = False,
) -> Path:
    data = prepare_revision_data(config_path)
    output = Path(data.config["output_dir"])
    marker = output / "mi_complete.json"
    if marker.exists() and not force and not smoke:
        LOGGER.info("Multiple-imputation analysis is already complete")
        return output
    main_prediction = pd.read_parquet(output / "main_predictions.parquet")
    (
        mi_frame,
        source,
        prediction,
        categorical,
        dropped,
    ) = _prepare_mi_frame(data, main_prediction)
    config = data.config["multiple_imputation"]
    datasets = 2 if smoke else int(config["datasets"])
    iterations = 2 if smoke else int(config["iterations"])
    LOGGER.info(
        "MI analysis: n=%s, variables=%s, datasets=%s, iterations=%s",
        len(mi_frame),
        len(mi_frame.columns),
        datasets,
        iterations,
    )

    (
        categorical_model3,
        categories,
        numeric_model3,
        standardizers,
    ) = _fixed_design_metadata(mi_frame, config)
    original_missing = {
        variable: mi_frame[variable].isna().to_numpy()
        for variable in config["model3_predictors"]
    }
    kernel = _run_kernel(
        mi_frame,
        datasets=datasets,
        iterations=iterations,
        seed=int(config["seed"]),
    )
    coefficients, covariances, distributions = _fit_models(
        kernel,
        datasets=datasets,
        config=config,
        categorical_model3=categorical_model3,
        categories=categories,
        numeric_model3=numeric_model3,
        standardizers=standardizers,
        original_missing=original_missing,
    )
    pooled = _rubin_pool(coefficients, covariances)
    ite_rows = pooled[pooled["term"] == "ite_per_5pp"]
    stable = bool((ite_rows["mcse_to_standard_error"] <= 0.10).all())
    if not smoke and not stable and datasets < int(config["max_datasets_if_unstable"]):
        datasets = int(config["max_datasets_if_unstable"])
        LOGGER.warning(
            "ITE Monte Carlo error exceeded 10%% of its SE; rerunning with %s datasets",
            datasets,
        )
        kernel = _run_kernel(
            mi_frame,
            datasets=datasets,
            iterations=iterations,
            seed=int(config["seed"]),
        )
        coefficients, covariances, distributions = _fit_models(
            kernel,
            datasets=datasets,
            config=config,
            categorical_model3=categorical_model3,
            categories=categories,
            numeric_model3=numeric_model3,
            standardizers=standardizers,
            original_missing=original_missing,
        )
        pooled = _rubin_pool(coefficients, covariances)
        ite_rows = pooled[pooled["term"] == "ite_per_5pp"]
        stable = bool((ite_rows["mcse_to_standard_error"] <= 0.10).all())

    target = output / ("mi_smoke" if smoke else "mi")
    target.mkdir(parents=True, exist_ok=True)
    _missingness_table(mi_frame).to_csv(target / "missingness.csv", index=False)
    dropped.to_csv(target / "dropped_variables.csv", index=False)
    pooled.to_csv(target / "rubin_pooled_logistic_regression.csv", index=False)
    distributions.to_csv(target / "imputed_distribution_diagnostics.csv", index=False)
    pd.DataFrame(
        [
            {
                "n": int(len(mi_frame)),
                "events": int(mi_frame["__outcome"].sum()),
                "imputations": datasets,
                "iterations": iterations,
                "variables": int(len(mi_frame.columns)),
                "variables_with_missing": int(mi_frame.isna().any().sum()),
                "ite_scale_percentage_points": float(
                    config["ite_scale_percentage_points"]
                ),
                "ite_mcse_stable": stable,
                "miceforest_version": mf.__version__,
            }
        ]
    ).to_csv(target / "analysis_summary.csv", index=False)
    if not smoke:
        with marker.open("w", encoding="utf-8") as fp:
            json.dump(
                {
                    "status": "complete",
                    "imputations": datasets,
                    "iterations": iterations,
                    "ite_mcse_stable": stable,
                },
                fp,
                indent=2,
            )
    LOGGER.info("Multiple-imputation analysis complete")
    return output
