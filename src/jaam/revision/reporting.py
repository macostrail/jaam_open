from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from jaam.revision.descriptive import generate_descriptive_artifacts

LOGGER = logging.getLogger(__name__)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

CENTRAL_OUTCOME = "neurologic_90d"
BLUE = "#1f77b4"
ORANGE = "#d95f02"
GREEN = "#2ca02c"
GREY = "#666666"


def _read_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open(encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)
    return raw["analysis"]


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    fig.savefig(
        path_without_suffix.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        path_without_suffix.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def _interval(low: float, high: float, digits: int = 1) -> str:
    return f"{100 * low:.{digits}f}% to {100 * high:.{digits}f}%"


def _main_figure(output: Path, figures: Path) -> None:
    estimates = pd.read_csv(output / "main_estimates.csv")
    estimates["label"] = pd.Categorical(
        estimates["outcome_label"],
        categories=[
            "30-day survival",
            "30-day CPC 1–2",
            "90-day survival",
            "90-day CPC 1–2",
        ],
        ordered=True,
    )
    estimates = estimates.sort_values("label").reset_index(drop=True)
    y = np.arange(len(estimates))
    rd = 100 * estimates["risk_difference"].to_numpy()
    low = 100 * estimates["ci_difference_low"].to_numpy()
    high = 100 * estimates["ci_difference_high"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(
        rd,
        y,
        xerr=np.vstack([rd - low, high - rd]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=3,
        markersize=6,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, estimates["label"].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel("Risk difference, percentage points (95% CI)")
    ax.set_title("Augmented overlap-weighted effects in the overlap population")
    margin = max(2.0, 0.16 * float(high.max() - low.min()))
    ax.set_xlim(float(min(0, low.min()) - margin), float(high.max() + 9.0))
    for index, row in estimates.iterrows():
        nnt = row["nnt"]
        label = (
            f"{100 * row['risk_difference']:.1f} "
            f"({100 * row['ci_difference_low']:.1f} to "
            f"{100 * row['ci_difference_high']:.1f}); "
            f"NNT {nnt:.1f}"
        )
        ax.text(
            float(high[index]) + 0.5,
            index,
            label,
            va="center",
            fontsize=8.5,
        )
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures / "main_outcome_effects")


def _propensity_figure(output: Path, figures: Path) -> None:
    prediction = pd.read_parquet(output / "main_predictions.parquet")
    observed = prediction["_outcome_neurologic_90d"].notna()
    work = prediction.loc[observed, ["_treatment", "propensity_neurologic_90d"]]
    treated = work.loc[work["_treatment"] == 1, "propensity_neurologic_90d"].to_numpy()
    untreated = work.loc[
        work["_treatment"] == 0, "propensity_neurologic_90d"
    ].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bins = np.linspace(0, 1, 51)
    source_rows: list[dict[str, float | int | str]] = []
    for treatment_label, values in [("No CAG", untreated), ("CAG", treated)]:
        density, edges = np.histogram(values, bins=bins, density=True)
        for left, right, value in zip(edges[:-1], edges[1:], density):
            source_rows.append(
                {
                    "panel": "propensity",
                    "treatment": treatment_label,
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "density": float(value),
                    "n": int(len(values)),
                }
            )
    axes[0].hist(
        untreated,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=BLUE,
        label=f"No CAG (n={len(untreated):,})",
    )
    axes[0].hist(
        treated,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=ORANGE,
        label=f"CAG (n={len(treated):,})",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Cross-fitted propensity score")
    axes[0].set_ylabel("Density (log scale)")
    axes[0].set_title("Full propensity-score range")
    axes[0].legend(frameon=False, fontsize=8)

    logit_treated = np.log(treated / (1 - treated))
    logit_untreated = np.log(untreated / (1 - untreated))
    lower = float(np.quantile(np.concatenate([logit_treated, logit_untreated]), 0.001))
    upper = float(np.quantile(np.concatenate([logit_treated, logit_untreated]), 0.999))
    logit_bins = np.linspace(lower, upper, 51)
    for treatment_label, values in [
        ("No CAG", logit_untreated),
        ("CAG", logit_treated),
    ]:
        density, edges = np.histogram(values, bins=logit_bins, density=True)
        for left, right, value in zip(edges[:-1], edges[1:], density):
            source_rows.append(
                {
                    "panel": "logit propensity",
                    "treatment": treatment_label,
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "density": float(value),
                    "n": int(len(values)),
                }
            )
    pd.DataFrame(source_rows).to_csv(
        output / "propensity_histogram_90d_cpc.csv", index=False
    )
    axes[1].hist(
        logit_untreated,
        bins=logit_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=BLUE,
        label="No CAG",
    )
    axes[1].hist(
        logit_treated,
        bins=logit_bins,
        density=True,
        histtype="step",
        linewidth=1.8,
        color=ORANGE,
        label="CAG",
    )
    axes[1].set_xlabel("Logit propensity score")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Central 99.8% of logit scores")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
    fig.suptitle("Propensity-score overlap: 90-day CPC 1–2")
    fig.tight_layout()
    _save_figure(fig, figures / "propensity_overlap_90d_cpc")


def _love_plot(output: Path, figures: Path) -> None:
    balance = pd.read_csv(output / "covariate_balance.csv")
    central = balance[balance["outcome"] == CENTRAL_OUTCOME].copy()
    by_variable = (
        central.groupby(["variable", "method"], as_index=False)["abs_smd"]
        .max()
        .pivot(index="variable", columns="method", values="abs_smd")
        .fillna(0)
    )
    selected = (
        by_variable.sort_values("unweighted", ascending=False)
        .head(30)
        .sort_values("unweighted", ascending=True)
    )
    y = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(8.2, 8.5))
    ax.scatter(
        selected["unweighted"],
        y,
        color=GREY,
        s=25,
        label="Unweighted",
        zorder=3,
    )
    ax.scatter(
        selected["overlap_weighted"],
        y,
        color=BLUE,
        s=25,
        label="Overlap weighted",
        zorder=3,
    )
    for index, (_, row) in enumerate(selected.iterrows()):
        ax.plot(
            [row["overlap_weighted"], row["unweighted"]],
            [index, index],
            color="#cccccc",
            linewidth=0.8,
            zorder=1,
        )
    ax.axvline(0.1, color=ORANGE, linestyle="--", linewidth=1.0)
    ax.set_yticks(y, selected.index)
    ax.set_xlabel("Maximum absolute standardized mean difference")
    ax.set_title("Covariate balance: 90-day CPC 1–2")
    ax.legend(frameon=False)
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures / "love_plot_90d_cpc")


def _gates_figure(output: Path, figures: Path) -> None:
    main = pd.read_csv(output / "gates.csv")
    main = main[main["prediction"] == "unclipped"].sort_values("group")
    temporal = pd.read_csv(output / "temporal_validation_gates.csv").sort_values(
        "group"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, data, title, color in [
        (axes[0], main, "Full-period cross-fitted GATES", BLUE),
        (
            axes[1],
            temporal,
            "2019 within-registry assessment\n(fixed 2014–2018 effect model)",
            GREEN,
        ),
    ]:
        x = data["group"].to_numpy()
        effect = 100 * data["risk_difference"].to_numpy()
        low = 100 * data["multiplier_ci_difference_low"].to_numpy()
        high = 100 * data["multiplier_ci_difference_high"].to_numpy()
        ax.errorbar(
            x,
            effect,
            yerr=np.vstack([effect - low, high - effect]),
            fmt="o-",
            color=color,
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xlabel("Predicted-effect group (1=lowest, 5=highest)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="#eeeeee", linewidth=0.7)
    axes[0].set_ylabel(
        "Augmented overlap-weighted risk difference\n(percentage points; multiplier 95% CI)"
    )
    fig.suptitle("Group-level treatment-effect estimates")
    fig.tight_layout()
    _save_figure(fig, figures / "gates_and_temporal_validation")


def _subgroup_figure(output: Path, figures: Path) -> None:
    estimates = pd.read_csv(output / "clinical_subgroup_estimates.csv")
    balance = pd.read_csv(output / "clinical_subgroup_balance_summary.csv")
    central = estimates[estimates["outcome"] == CENTRAL_OUTCOME].copy()
    balance["subgroup"] = balance["method"].str.split("|", regex=False).str[-1]
    central = central.merge(
        balance[["subgroup", "max_abs_smd"]],
        how="left",
        on="subgroup",
    )
    order = [
        "all",
        "ROSC",
        "no_ROSC",
        "ROSC_shockable",
        "ROSC_ST_elevation",
        "ROSC_no_ST_elevation",
        "ROSC_VF",
        "ROSC_PEA",
        "ROSC_asystole",
        "ROSC_VF_no_ST_elevation",
    ]
    central["subgroup"] = pd.Categorical(
        central["subgroup"], categories=order, ordered=True
    )
    central = central.sort_values("subgroup").reset_index(drop=True)
    labels = {
        "all": "All eligible patients",
        "ROSC": "ROSC",
        "no_ROSC": "No ROSC",
        "ROSC_shockable": "ROSC + shockable rhythm",
        "ROSC_ST_elevation": "ROSC + ST elevation",
        "ROSC_no_ST_elevation": "ROSC + no ST elevation",
        "ROSC_VF": "ROSC + VF",
        "ROSC_PEA": "ROSC + PEA",
        "ROSC_asystole": "ROSC + asystole",
        "ROSC_VF_no_ST_elevation": "ROSC + VF + no ST elevation",
    }
    display = [
        f"{labels.get(str(row.subgroup), str(row.subgroup))}"
        f"  (CAG {int(row.treated_n):,}; no CAG {int(row.untreated_n):,})"
        for row in central.itertuples()
    ]
    y = np.arange(len(central))
    effect = 100 * central["risk_difference"].to_numpy()
    low = 100 * central["ci_difference_low"].to_numpy()
    high = 100 * central["ci_difference_high"].to_numpy()
    max_smd = central["max_abs_smd"].fillna(1).to_numpy()
    colors = np.select(
        [
            max_smd <= 0.1,
            max_smd <= 0.2,
            max_smd <= 0.3,
        ],
        [
            BLUE,
            "#d4a72c",
            ORANGE,
        ],
        default="#b2182b",
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.1))
    for index in range(len(central)):
        ax.errorbar(
            effect[index],
            y[index],
            xerr=np.asarray(
                [
                    [effect[index] - low[index]],
                    [high[index] - effect[index]],
                ]
            ),
            fmt="o",
            color=colors[index],
            ecolor=colors[index],
            capsize=3,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, display)
    ax.invert_yaxis()
    ax.set_xlabel("Risk difference for 90-day CPC 1–2, percentage points (95% CI)")
    ax.set_title("Exploratory clinical subgroup estimates")
    legend_handles = [
        Line2D([0], [0], marker="o", color=color, linewidth=0, label=label)
        for color, label in [
            (BLUE, "≤0.10"),
            ("#d4a72c", "0.10–0.20"),
            (ORANGE, "0.20–0.30"),
            ("#b2182b", ">0.30"),
        ]
    ]
    ax.legend(
        handles=legend_handles,
        title="Maximum post-weighting |SMD|",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=4,
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures / "clinical_subgroups_90d_cpc")


def _shap_figure(output: Path, figures: Path) -> None:
    stability = pd.read_csv(output / "shap_stability.csv").head(10).copy()
    stability = stability.sort_values("median_weighted_mean_abs_shap", ascending=True)
    y = np.arange(len(stability))
    median = stability["median_weighted_mean_abs_shap"].to_numpy()
    low = stability["min_weighted_mean_abs_shap"].to_numpy()
    high = stability["max_weighted_mean_abs_shap"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    ax.errorbar(
        median,
        y,
        xerr=np.vstack([median - low, high - median]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=3,
    )
    ax.set_yticks(y, stability["feature"])
    ax.set_xlabel(
        "Overlap-weighted mean absolute SHAP value\n(median and range across 5 held-out folds)"
    )
    ax.set_title("Exploratory effect-model feature importance")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, figures / "shap_fold_stability")


def _calibration_figure(output: Path, figures: Path) -> None:
    calibration = pd.read_csv(output / "central_outcome_calibration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.1), sharex=True, sharey=True)
    for arm, ax, title, color in [
        (0, axes[0], "Outcome model under no CAG", BLUE),
        (1, axes[1], "Outcome model under CAG", ORANGE),
    ]:
        data = calibration[calibration["arm"] == arm]
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--")
        ax.scatter(
            data["predicted"],
            data["observed"],
            s=np.maximum(15, np.sqrt(data["n"]) * 2),
            color=color,
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Mean held-out predicted probability")
        ax.grid(color="#eeeeee", linewidth=0.7)
    axes[0].set_ylabel("Observed proportion")
    fig.suptitle("Nuisance outcome-model calibration: 90-day CPC 1–2")
    fig.tight_layout()
    _save_figure(fig, figures / "outcome_model_calibration_90d_cpc")


def _mi_convergence_figure(output: Path, figures: Path) -> None:
    convergence = pd.read_csv(output / "mi" / "convergence_diagnostics.csv")
    summary = pd.read_csv(output / "mi" / "convergence_summary.csv")
    selected = summary.head(6)["variable"].tolist()
    ncols = 2
    nrows = int(np.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.0, 3.0 * nrows),
        squeeze=False,
    )
    for ax, variable in zip(axes.ravel(), selected):
        data = convergence[convergence["variable"] == variable]
        for _, group in data.groupby("dataset"):
            ax.plot(
                group["iteration"],
                group["imputed_mean"],
                linewidth=1.0,
                alpha=0.8,
            )
        missing_percent = summary.loc[
            summary["variable"] == variable, "missing_percent"
        ].iloc[0]
        ax.set_title(f"{variable}\nmissing {missing_percent:.1f}%", fontsize=9)
        ax.set_xlabel("MICE iteration")
        ax.set_ylabel("Mean imputed value")
        ax.grid(color="#eeeeee", linewidth=0.7)
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")
    fig.suptitle("Multiple-imputation convergence diagnostic (5 datasets)")
    fig.tight_layout()
    _save_figure(fig, figures / "mi_mean_convergence")


def _write_tables(output: Path, tables: Path) -> None:
    main = pd.read_csv(output / "main_estimates.csv")
    main_columns = [
        "outcome_label",
        "eligible_n",
        "observed_n",
        "missing_n",
        "treated_n",
        "untreated_n",
        "events",
        "risk_treated",
        "ci_treated_low",
        "ci_treated_high",
        "risk_untreated",
        "ci_untreated_low",
        "ci_untreated_high",
        "risk_difference",
        "ci_difference_low",
        "ci_difference_high",
        "risk_ratio",
        "ci_ratio_low",
        "ci_ratio_high",
        "nnt_type",
        "nnt",
    ]
    main[main_columns].to_csv(tables / "main_effects.csv", index=False)

    gates = pd.read_csv(output / "gates.csv")
    gates[gates["prediction"] == "unclipped"].to_csv(
        tables / "gates_90d_cpc.csv", index=False
    )

    central_propensity = pd.read_csv(output / "propensity_distribution.csv")
    central_propensity[central_propensity["outcome"] == CENTRAL_OUTCOME].to_csv(
        tables / "propensity_overlap_90d_cpc.csv", index=False
    )
    central_weights = pd.read_csv(output / "weight_diagnostics.csv")
    central_weights[central_weights["outcome"] == CENTRAL_OUTCOME].to_csv(
        tables / "weight_diagnostics_90d_cpc.csv", index=False
    )

    sensitivity = pd.read_csv(output / "sensitivity_estimates.csv")
    sensitivity[sensitivity["outcome"] == CENTRAL_OUTCOME].to_csv(
        tables / "sensitivity_effects_90d_cpc.csv", index=False
    )

    temporal = pd.read_csv(output / "temporal_overall_estimates.csv")
    temporal.to_csv(tables / "temporal_overall_effects.csv", index=False)
    pd.read_csv(output / "temporal_validation_gates.csv").to_csv(
        tables / "temporal_validation_gates.csv", index=False
    )

    subgroups = pd.read_csv(output / "clinical_subgroup_estimates.csv")
    subgroups[subgroups["outcome"] == CENTRAL_OUTCOME].to_csv(
        tables / "clinical_subgroups_90d_cpc.csv", index=False
    )
    pd.read_csv(output / "clinical_subgroup_interactions.csv").query(
        "outcome == @CENTRAL_OUTCOME"
    ).to_csv(tables / "clinical_subgroup_interactions_90d_cpc.csv", index=False)

    mi_file = output / "mi" / "rubin_pooled_logistic_regression.csv"
    if mi_file.exists():
        mi = pd.read_csv(mi_file)
        mi[mi["term"] == "ite_per_5pp"].to_csv(
            tables / "mi_ite_association.csv", index=False
        )
    no_rosc_file = output / "no_rosc_pathway_description.csv"
    if no_rosc_file.exists():
        pd.read_csv(no_rosc_file).to_csv(
            tables / "no_rosc_pathway_description.csv", index=False
        )


def _audit(output: Path) -> dict[str, Any]:
    main = pd.read_csv(output / "main_estimates.csv")
    central = main[main["outcome"] == CENTRAL_OUTCOME].iloc[0]
    balance = pd.read_csv(output / "balance_summary.csv")
    central_balance = balance[
        (balance["outcome"] == CENTRAL_OUTCOME)
        & (balance["method"] == "overlap_weighted")
    ].iloc[0]
    gates = pd.read_csv(output / "gates.csv")
    gates = gates[gates["prediction"] == "unclipped"].sort_values("group")
    clipping = pd.read_csv(output / "ite_clipping_comparison.csv")
    temporal = pd.read_csv(output / "temporal_overall_estimates.csv")
    validation = temporal[temporal["period"] == "validation_2019"].iloc[0]
    ipow = pd.read_csv(output / "ipow_estimates.csv")
    canonical_ipow = ipow[
        (ipow["variant"] == "with_30d_auxiliary")
        & (ipow["specification"] == "canonical")
    ].iloc[0]
    mi_summary_file = output / "mi" / "analysis_summary.csv"
    checks = {
        "main_has_four_outcomes": len(main) == 4,
        "central_observed_n": int(central["observed_n"]) == 36283,
        "central_treated_n": int(central["treated_n"]) == 2962,
        "central_untreated_n": int(central["untreated_n"]) == 33321,
        "central_effect_finite": bool(np.isfinite(central["risk_difference"])),
        "central_ci_ordered": bool(
            central["ci_difference_low"]
            < central["risk_difference"]
            < central["ci_difference_high"]
        ),
        "central_balance_near_threshold": bool(central_balance["max_abs_smd"] <= 0.105),
        "five_gates": len(gates) == 5,
        "gates_rank_gradient": bool(
            gates.iloc[-1]["risk_difference"] > gates.iloc[0]["risk_difference"]
        ),
        "ite_clipping_no_group_moves": bool(
            int(clipping["quintile_moved_n"].max()) == 0
        ),
        "temporal_validation_direction_positive": bool(
            validation["risk_difference"] > 0
        ),
        "ipow_direction_positive": bool(canonical_ipow["risk_difference"] > 0),
        "mi_complete": mi_summary_file.exists(),
        "table1_generated": (output / "tables" / "table1_by_cag.csv").exists(),
        "covariate_missingness_generated": (
            output / "tables" / "variable_level_missingness.csv"
        ).exists(),
        "observed_missing_comparison_generated": (
            output / "tables" / "observed_vs_missing_90d_cpc.csv"
        ).exists(),
        "covariate_timing_table_generated": (
            output / "tables" / "covariate_timing_and_model_use.csv"
        ).exists(),
        "cohort_flow_reconciled": bool(
            pd.read_csv(output / "tables" / "cohort_exclusions.csv").iloc[-1][
                "remaining_n"
            ]
            == 49156
        ),
        "reviewer_figures_generated": all(
            (output / "figures" / name).exists()
            for name in [
                "causal_framework.png",
                "cohort_outcome_flow.png",
                "nested_cross_fitting_workflow.png",
            ]
        ),
    }
    if mi_summary_file.exists():
        mi_summary = pd.read_csv(mi_summary_file).iloc[0]
        checks["mi_mcse_stable"] = bool(mi_summary["ite_mcse_stable"])
    return {
        "status": "pass" if all(checks.values()) else "review",
        "checks": checks,
    }


def _write_markdown_summary(output: Path, audit: dict[str, Any]) -> None:
    main = pd.read_csv(output / "main_estimates.csv")
    central = main[main["outcome"] == CENTRAL_OUTCOME].iloc[0]
    balance = pd.read_csv(output / "balance_summary.csv")
    central_balance = balance[
        (balance["outcome"] == CENTRAL_OUTCOME)
        & (balance["method"] == "overlap_weighted")
    ].iloc[0]
    gates = pd.read_csv(output / "gates.csv")
    gates = gates[gates["prediction"] == "unclipped"].sort_values("group")
    ipow = pd.read_csv(output / "ipow_estimates.csv")
    ipow = ipow[
        (ipow["variant"] == "with_30d_auxiliary")
        & (ipow["specification"] == "canonical")
    ].iloc[0]
    early = pd.read_csv(output / "early_baseline_estimates.csv")
    early = early[early["outcome"] == CENTRAL_OUTCOME].iloc[0]
    temporal = pd.read_csv(output / "temporal_overall_estimates.csv")
    temporal_validation = temporal[temporal["period"] == "validation_2019"].iloc[0]
    shockable = pd.read_csv(output / "rosc_shockable_estimates.csv")
    shockable = shockable[shockable["outcome"] == CENTRAL_OUTCOME].iloc[0]

    lines = [
        "# JAAM-OHCA revision analysis summary",
        "",
        "All effects below are risk differences in the estimand-specific "
        "population and are based on patient-level cross-fitted nuisance "
        "predictions with facility-clustered inference.",
        "",
        "## Main augmented overlap-weighted analysis",
        "",
        "| Outcome | Observed / eligible | CAG risk | No-CAG risk | RD (95% CI) | RR (95% CI) | NNT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main.itertuples():
        lines.append(
            f"| {row.outcome_label} | {int(row.observed_n):,} / "
            f"{int(row.eligible_n):,} | {_percent(row.risk_treated)} | "
            f"{_percent(row.risk_untreated)} | "
            f"{_percent(row.risk_difference)} "
            f"({_interval(row.ci_difference_low, row.ci_difference_high)}) | "
            f"{row.risk_ratio:.2f} ({row.ci_ratio_low:.2f}–"
            f"{row.ci_ratio_high:.2f}) | {row.nnt:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Primary outcome diagnostics",
            "",
            f"- 90-day CPC 1–2: n={int(central.observed_n):,}; "
            f"CAG={int(central.treated_n):,}; no CAG={int(central.untreated_n):,}; "
            f"events={int(central.events):,}.",
            f"- Maximum post-overlap-weighting absolute SMD: "
            f"{central_balance.max_abs_smd:.3f}; components above 0.10: "
            f"{int(central_balance.components_above_0_1)}.",
            f"- GATES group 1 RD: {_percent(gates.iloc[0].risk_difference)} "
            f"({_interval(gates.iloc[0].multiplier_ci_difference_low, gates.iloc[0].multiplier_ci_difference_high)}); "
            f"group 5 RD: {_percent(gates.iloc[-1].risk_difference)} "
            f"({_interval(gates.iloc[-1].multiplier_ci_difference_low, gates.iloc[-1].multiplier_ci_difference_high)}).",
            "",
            "## Sensitivity analyses",
            "",
            f"- 90-day outcome IPOW: RD {_percent(ipow.risk_difference)} "
            f"({_interval(ipow.ci_difference_low, ipow.ci_difference_high)}).",
            f"- Early-baseline-only covariates: RD "
            f"{_percent(early.risk_difference)} "
            f"({_interval(early.ci_difference_low, early.ci_difference_high)}).",
            f"- ROSC + shockable rhythm: RD "
            f"{_percent(shockable.risk_difference)} "
            f"({_interval(shockable.ci_difference_low, shockable.ci_difference_high)}).",
            f"- 2019 within-registry temporal assessment: RD "
            f"{_percent(temporal_validation.risk_difference)} "
            f"({_interval(temporal_validation.ci_difference_low, temporal_validation.ci_difference_high)}).",
            "",
            "## Automated audit",
            "",
            f"Status: **{audit['status']}**",
            "",
        ]
    )
    for name, passed in audit["checks"].items():
        lines.append(f"- {'PASS' if passed else 'REVIEW'}: `{name}`")
    lines.append("")
    (output / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_reporting(
    config_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    config = _read_config(config_path)
    output = Path(config["output_dir"])
    marker = output / "report_complete.json"
    if marker.exists() and not force:
        LOGGER.info("Revision report is already complete")
        return output

    required = [
        "main_estimates.csv",
        "main_predictions.parquet",
        "gates.csv",
        "balance_summary.csv",
        "sensitivity_estimates.csv",
        "ipow_estimates.csv",
        "temporal_overall_estimates.csv",
        "clinical_subgroup_estimates.csv",
        "shap_stability.csv",
        "mi/analysis_summary.csv",
        "mi/convergence_diagnostics.csv",
        "mi/convergence_summary.csv",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot build the revision report; missing outputs: " + ", ".join(missing)
        )

    figures = output / "figures"
    tables = output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    _main_figure(output, figures)
    _propensity_figure(output, figures)
    _love_plot(output, figures)
    _gates_figure(output, figures)
    _subgroup_figure(output, figures)
    _shap_figure(output, figures)
    _calibration_figure(output, figures)
    _mi_convergence_figure(output, figures)
    _write_tables(output, tables)
    descriptive = generate_descriptive_artifacts(
        config_path,
        output,
        figures,
        tables,
    )
    audit = _audit(output)
    with (output / "analysis_audit.json").open("w", encoding="utf-8") as fp:
        json.dump(audit, fp, indent=2)
    _write_markdown_summary(output, audit)
    with marker.open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "status": "complete",
                "audit_status": audit["status"],
                "figures": 8 + descriptive["additional_figures"],
                "eligible_n": descriptive["eligible_n"],
                "main_features": descriptive["main_feature_n"],
            },
            fp,
            indent=2,
        )
    LOGGER.info("Revision analysis report complete: %s", output)
    return output
