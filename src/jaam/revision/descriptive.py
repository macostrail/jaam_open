from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

from jaam.data import load_dataset_csv
from jaam.revision.data import RevisionData, prepare_revision_data
from jaam.variables import Variables

BLUE = "#1f77b4"
ORANGE = "#d95f02"
GREEN = "#2ca02c"
GREY = "#666666"
LIGHT_BLUE = "#e8f1f8"
LIGHT_ORANGE = "#fbeee8"
LIGHT_GREEN = "#eaf5ea"


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


def _table_specs() -> list[dict[str, Any]]:
    return [
        {
            "section": "Facility characteristics",
            "label": "Hospital beds",
            "source": "病床数",
            "kind": "numeric",
            "digits": 0,
        },
        {
            "section": "Facility characteristics",
            "label": "ICU beds",
            "source": "ICU病床数",
            "kind": "numeric",
            "digits": 0,
        },
        {
            "section": "Facility characteristics",
            "label": "OHCA cases transported in the preceding year",
            "source": "直近1年間の搬送心停止症例数",
            "kind": "numeric",
            "digits": 0,
        },
        {
            "section": "Facility characteristics",
            "label": "Facility capable of extracorporeal resuscitation",
            "source": "院外心停止に対する体外循環治療の可否",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Patient characteristics",
            "label": "Age, years",
            "source": "年齢(JAAM)",
            "kind": "numeric",
            "digits": 1,
        },
        {
            "section": "Patient characteristics",
            "label": "Male sex",
            "source": "性別",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Patient characteristics",
            "label": "Registry year 2019",
            "source": "_year",
            "kind": "binary",
            "positive": [2019],
        },
        {
            "section": "Arrest characteristics",
            "label": "Witnessed arrest",
            "source": "目撃",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Arrest characteristics",
            "label": "Bystander CPR",
            "source": "あり／なし",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Arrest characteristics",
            "label": "Initially shockable rhythm",
            "source": "波形種別",
            "kind": "binary",
            "positive": [1, 2],
        },
        {
            "section": "Arrest characteristics",
            "label": "Initially nonshockable rhythm",
            "source": "波形種別",
            "kind": "binary",
            "positive": [3, 4],
        },
        {
            "section": "Arrest characteristics",
            "label": "ROSC before hospital arrival",
            "source": "病院収容後の心拍再開",
            "kind": "binary",
            "positive": [2],
        },
        {
            "section": "Prehospital interventions",
            "label": "Dispatcher CPR instruction",
            "source": "口頭指示あり",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Prehospital interventions",
            "label": "Bystander AED use",
            "source": "バイスタンダーによる除細動",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Prehospital interventions",
            "label": "Prehospital defibrillation",
            "source": "除細動",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Prehospital interventions",
            "label": "Prehospital drug administration",
            "source": "薬剤投与",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Prehospital interventions",
            "label": "Advanced airway device used",
            "source": "特定行為器具使用",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Prehospital intervals",
            "label": "Witness-to-hospital-arrival time, min",
            "source": "目撃から病院収容までの時間",
            "kind": "numeric",
            "digits": 1,
        },
        {
            "section": "Early in-hospital information",
            "label": "ROSC before or after hospital arrival",
            "source": "病院収容後の心拍再開",
            "kind": "binary",
            "positive": [1, 2],
        },
        {
            "section": "Early in-hospital information",
            "label": "Shockable first in-hospital ECG",
            "source": "__early_inhospital_ecg",
            "kind": "binary",
            "positive": [1, 2],
        },
        {
            "section": "Early in-hospital information",
            "label": "ST-segment elevation after ROSC",
            "source": "心拍再開後のＳＴ上昇の有無",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Early in-hospital information",
            "label": "First recorded GCS total",
            "source": "病院収容後の意識状態EVM",
            "kind": "numeric",
            "digits": 0,
        },
        {
            "section": "Early in-hospital information",
            "label": "First pre-ROSC blood pH",
            "source": "pH(ROSC前初回採血)",
            "kind": "numeric",
            "digits": 2,
        },
        {
            "section": "Early in-hospital information",
            "label": "First pre-ROSC lactate, registry-standardized units",
            "source": "LacROSC前初回採血の単位統一後",
            "kind": "numeric",
            "digits": 1,
        },
        {
            "section": "Early in-hospital information",
            "label": "Total epinephrine dose, registry units",
            "source": "アドレナリン総投与量(CRITICAL)",
            "kind": "numeric",
            "digits": 1,
        },
    ]


def _numeric_distribution_diagnostics(
    frame: pd.DataFrame,
    specs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in specs:
        if spec["kind"] != "numeric" or spec["source"] in seen:
            continue
        seen.add(spec["source"])
        values = pd.to_numeric(frame[spec["source"]], errors="coerce").dropna()
        skewness = float(values.skew()) if len(values) > 2 else np.nan
        qq_correlation = (
            float(stats.probplot(values.to_numpy(), dist="norm", fit=True)[1][2])
            if len(values) > 2
            else np.nan
        )
        approximately_symmetric = bool(
            np.isfinite(skewness)
            and np.isfinite(qq_correlation)
            and abs(skewness) <= 1.0
            and qq_correlation >= 0.95
        )
        rows.append(
            {
                "variable": spec["label"],
                "source_variable": spec["source"],
                "observed_n": int(values.size),
                "missing_n": int(frame[spec["source"]].isna().sum()),
                "skewness": skewness,
                "qq_correlation": qq_correlation,
                "summary_type": (
                    "mean_sd" if approximately_symmetric else "median_q1_q3"
                ),
                "classification_rule": (
                    "mean_sd if |skewness| <= 1 and QQ correlation >= 0.95; "
                    "otherwise median_q1_q3"
                ),
            }
        )
    return pd.DataFrame(rows)


def _numeric_summary(
    values: pd.Series,
    *,
    summary_type: str,
    digits: int,
) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "NA"
    if summary_type == "mean_sd":
        return f"{numeric.mean():.{digits}f} ({numeric.std():.{digits}f})"
    q1, median, q3 = numeric.quantile([0.25, 0.5, 0.75])
    return f"{median:.{digits}f} " f"({q1:.{digits}f}–{q3:.{digits}f})"


def _binary_summary(values: pd.Series, positive: list[Any]) -> str:
    observed = values.notna()
    denominator = int(observed.sum())
    if denominator == 0:
        return "NA"
    events = int(values.loc[observed].isin(positive).sum())
    return f"{events:,} ({100 * events / denominator:.1f}%)"


def _numeric_smd(
    values: pd.Series,
    group: pd.Series,
) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    first = numeric[group].dropna()
    second = numeric[~group].dropna()
    if first.empty or second.empty:
        return np.nan
    denominator = np.sqrt((first.var(ddof=1) + second.var(ddof=1)) / 2)
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return float((first.mean() - second.mean()) / denominator)


def _binary_smd(
    values: pd.Series,
    group: pd.Series,
    positive: list[Any],
) -> float:
    first = values[group].dropna().isin(positive).astype(float)
    second = values[~group].dropna().isin(positive).astype(float)
    if first.empty or second.empty:
        return np.nan
    p_first = float(first.mean())
    p_second = float(second.mean())
    denominator = np.sqrt((p_first * (1 - p_first) + p_second * (1 - p_second)) / 2)
    if denominator == 0:
        return np.nan
    return float((p_first - p_second) / denominator)


def _group_characteristics(
    frame: pd.DataFrame,
    specs: list[dict[str, Any]],
    group: pd.Series,
    *,
    group_true_label: str,
    group_false_label: str,
    numeric_formats: dict[str, str],
) -> pd.DataFrame:
    group = group.astype(bool)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        values = frame[spec["source"]]
        true_values = values[group]
        false_values = values[~group]
        if spec["kind"] == "numeric":
            summary_type = numeric_formats[spec["source"]]

            def summary(
                current: pd.Series,
                selected_type: str = summary_type,
                digits: int = spec["digits"],
            ) -> str:
                return _numeric_summary(
                    current,
                    summary_type=selected_type,
                    digits=digits,
                )

            smd = _numeric_smd(values, group)
        else:
            positive = spec["positive"]

            def summary(
                current: pd.Series,
                selected_positive: list[Any] = positive,
            ) -> str:
                return _binary_summary(current, selected_positive)

            smd = _binary_smd(values, group, positive)
            summary_type = "n_percent"
        rows.append(
            {
                "section": spec["section"],
                "variable": spec["label"],
                "source_variable": spec["source"],
                "summary_type": summary_type,
                "overall": summary(values),
                group_true_label: summary(true_values),
                group_false_label: summary(false_values),
                "smd_true_minus_false": smd,
                "abs_smd": abs(smd) if np.isfinite(smd) else np.nan,
                "overall_missing_n": int(values.isna().sum()),
                "overall_missing_percent": float(100 * values.isna().mean()),
                f"{group_true_label}_missing_n": int(true_values.isna().sum()),
                f"{group_true_label}_missing_percent": float(
                    100 * true_values.isna().mean()
                ),
                f"{group_false_label}_missing_n": int(false_values.isna().sum()),
                f"{group_false_label}_missing_percent": float(
                    100 * false_values.isna().mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_characteristic_tables(
    data: RevisionData,
    tables: Path,
) -> None:
    frame = data.frame
    specs = _table_specs()
    distributions = _numeric_distribution_diagnostics(frame, specs)
    distributions.to_csv(tables / "table1_distribution_diagnostics.csv", index=False)
    numeric_formats = dict(
        zip(distributions["source_variable"], distributions["summary_type"])
    )

    table1 = _group_characteristics(
        frame,
        specs,
        frame["_treatment"] == 1,
        group_true_label="CAG",
        group_false_label="no_CAG",
        numeric_formats=numeric_formats,
    )
    table1.to_csv(tables / "table1_by_cag.csv", index=False)

    observation_specs = [
        *specs,
        {
            "section": "Treatment and auxiliary outcomes",
            "label": "CAG-based strategy received",
            "source": "_treatment",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Treatment and auxiliary outcomes",
            "label": "30-day survival",
            "source": "_outcome_survival_30d",
            "kind": "binary",
            "positive": [1],
        },
        {
            "section": "Treatment and auxiliary outcomes",
            "label": "30-day CPC 1–2",
            "source": "_outcome_neurologic_30d",
            "kind": "binary",
            "positive": [1],
        },
    ]
    observed = frame["_outcome_neurologic_90d"].notna()
    observed_table = _group_characteristics(
        frame,
        observation_specs,
        observed,
        group_true_label="observed_90d_CPC",
        group_false_label="missing_90d_CPC",
        numeric_formats=numeric_formats,
    )
    observed_table.to_csv(tables / "observed_vs_missing_90d_cpc.csv", index=False)


def _measurement_setting(
    feature: str,
    *,
    facility_features: set[str],
    phase1: set[str],
) -> str:
    if feature in facility_features:
        return "facility baseline"
    if feature in phase1:
        return "prehospital or hospital-presentation baseline"
    return "early in-hospital assessment"


def _write_covariate_tables(
    data: RevisionData,
    tables: Path,
) -> None:
    frame = data.frame
    variables = Variables()
    phase1 = set(variables.phase1_vars)
    phase2 = set(variables.phase2_vars)
    early_source = data.config["features"]["early_ecg_source"]
    engineered = data.config["features"]["engineered_early_ecg"]
    first_patient_feature = variables.phase1_vars.index("性別")
    facility_features = set(variables.phase1_vars[:first_patient_feature])

    missing_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for feature in data.main_features:
        source = early_source if feature == engineered else feature
        if feature == engineered:
            phase = "phase 2"
        elif feature in phase1:
            phase = "phase 1"
        elif feature in phase2:
            phase = "phase 2"
        else:
            raise ValueError(f"Unclassified main feature: {feature}")
        data_type = "categorical" if feature in data.categorical_features else "numeric"
        series = frame[feature]
        treated = frame["_treatment"] == 1
        missing_rows.append(
            {
                "analysis_variable": feature,
                "source_variable": source,
                "phase": phase,
                "data_type": data_type,
                "overall_n": int(len(frame)),
                "overall_missing_n": int(series.isna().sum()),
                "overall_missing_percent": float(100 * series.isna().mean()),
                "CAG_n": int(treated.sum()),
                "CAG_missing_n": int(series[treated].isna().sum()),
                "CAG_missing_percent": float(100 * series[treated].isna().mean()),
                "no_CAG_n": int((~treated).sum()),
                "no_CAG_missing_n": int(series[~treated].isna().sum()),
                "no_CAG_missing_percent": float(100 * series[~treated].isna().mean()),
            }
        )
        setting = _measurement_setting(
            feature,
            facility_features=facility_features,
            phase1=phase1,
        )
        if phase == "phase 1":
            rationale = (
                "Available before or at hospital presentation; selected "
                "without reference to treatment or outcome."
            )
        else:
            rationale = (
                "Early in-hospital information considered available before "
                "CAG in the main specification; exact patient-level ordering "
                "relative to CAG cannot be verified."
            )
        encoding = (
            "Registry category passed to LightGBM with native missing-value "
            "handling."
            if data_type == "categorical"
            else "Continuous registry value; no primary-analysis imputation."
        )
        if feature == engineered:
            encoding = (
                f"Single categorical copy of {early_source}; redundant source "
                "columns were not entered separately."
            )
        timing_rows.append(
            {
                "analysis_variable": feature,
                "source_variable": source,
                "phase": phase,
                "measurement_setting": setting,
                "data_type": data_type,
                "encoding": encoding,
                "overall_missing_n": int(series.isna().sum()),
                "overall_missing_percent": float(100 * series.isna().mean()),
                "treatment_model": "yes",
                "arm_specific_outcome_models": "yes",
                "effect_model": "yes",
                "early_baseline_sensitivity": (
                    "yes" if feature in data.early_features else "no"
                ),
                "role_or_exclusion_rationale": rationale,
            }
        )

    for key, spec in data.outcomes.items():
        series = frame[f"_outcome_{key}"]
        treated = frame["_treatment"] == 1
        missing_rows.append(
            {
                "analysis_variable": f"_outcome_{key}",
                "source_variable": spec.source,
                "phase": "outcome",
                "data_type": "binary",
                "overall_n": int(len(frame)),
                "overall_missing_n": int(series.isna().sum()),
                "overall_missing_percent": float(100 * series.isna().mean()),
                "CAG_n": int(treated.sum()),
                "CAG_missing_n": int(series[treated].isna().sum()),
                "CAG_missing_percent": float(100 * series[treated].isna().mean()),
                "no_CAG_n": int((~treated).sum()),
                "no_CAG_missing_n": int(series[~treated].isna().sum()),
                "no_CAG_missing_percent": float(100 * series[~treated].isna().mean()),
            }
        )

    for feature in variables.phase3_vars:
        if feature not in frame.columns:
            continue
        series = frame[feature]
        if feature == data.config["treatment"]["source"]:
            role = (
                "Operational exposure defining the CAG-based strategy; not "
                "entered as a covariate."
            )
        else:
            role = (
                "Excluded from all adjustment models because it may be a "
                "strategy component, mediator, or post-baseline "
                "co-intervention."
            )
        timing_rows.append(
            {
                "analysis_variable": feature,
                "source_variable": feature,
                "phase": "phase 3",
                "measurement_setting": "in-hospital pathway/post-baseline",
                "data_type": (
                    "numeric"
                    if pd.api.types.is_numeric_dtype(series)
                    and series.nunique(dropna=True) > 10
                    else "categorical"
                ),
                "encoding": "Not entered as an adjustment covariate.",
                "overall_missing_n": int(series.isna().sum()),
                "overall_missing_percent": float(100 * series.isna().mean()),
                "treatment_model": "no",
                "arm_specific_outcome_models": "no",
                "effect_model": "no",
                "early_baseline_sensitivity": "no",
                "role_or_exclusion_rationale": role,
            }
        )

    pd.DataFrame(missing_rows).to_csv(
        tables / "variable_level_missingness.csv", index=False
    )
    pd.DataFrame(timing_rows).to_csv(
        tables / "covariate_timing_and_model_use.csv", index=False
    )


def _write_target_trial_table(tables: Path) -> None:
    rows = [
        {
            "component": "Eligibility",
            "ideal_target_trial": (
                "Adults with nontraumatic OHCA eligible for either strategy "
                "at hospital presentation."
            ),
            "observational_emulation": (
                "JAAM-OHCA registry, 2014–2019; age >=18 years; in-hospital "
                "physician resuscitation; specified external causes excluded; "
                "one facility excluded during data reconciliation; recorded "
                "CAG status required."
            ),
            "limitation": (
                "Clinical eligibility for both strategies and detailed "
                "comorbidities were not recorded."
            ),
        },
        {
            "component": "Time zero",
            "ideal_target_trial": (
                "Hospital presentation, before the invasive-strategy decision."
            ),
            "observational_emulation": (
                "Hospital presentation used as the conceptual assignment time."
            ),
            "limitation": (
                "Exact CAG decision and procedure times were unavailable, so "
                "immortal-time and temporal-order bias cannot be eliminated."
            ),
        },
        {
            "component": "Strategies",
            "ideal_target_trial": (
                "CAG-based invasive management strategy versus no-CAG strategy."
            ),
            "observational_emulation": (
                "Recorded receipt versus nonreceipt of CAG operationalized "
                "entry into the respective observed pathway."
            ),
            "limitation": (
                "This is not the isolated effect of diagnostic CAG, PCI, ECMO, "
                "temperature control, or other pathway components."
            ),
        },
        {
            "component": "Assignment",
            "ideal_target_trial": "Random allocation at time zero.",
            "observational_emulation": (
                "Clinician-selected observed strategy; measured baseline "
                "differences adjusted using cross-fitted nuisance models and "
                "augmented overlap weighting."
            ),
            "limitation": (
                "Conditional exchangeability remains unverifiable; residual "
                "and unmeasured confounding are expected."
            ),
        },
        {
            "component": "Follow-up",
            "ideal_target_trial": "From time zero through 30 and 90 days.",
            "observational_emulation": "Registry outcomes at 30 and 90 days.",
            "limitation": "90-day outcome observation was incomplete.",
        },
        {
            "component": "Outcomes",
            "ideal_target_trial": (
                "Survival and favorable neurological status (CPC 1–2) at 30 "
                "and 90 days."
            ),
            "observational_emulation": (
                "The same four registry-defined binary outcomes; 90-day CPC "
                "1–2 was the primary outcome."
            ),
            "limitation": (
                "Complete-case primary analysis supplemented by IPOW under a "
                "missing-at-random assumption."
            ),
        },
        {
            "component": "Causal contrast / estimand",
            "ideal_target_trial": (
                "Marginal contrast appropriate to patients for whom either "
                "strategy is clinically plausible."
            ),
            "observational_emulation": (
                "Average treatment effect in the overlap population (ATO), "
                "with marginal risks, RD, RR, and grouped GATES."
            ),
            "limitation": (
                "The estimand is not the full-cohort ATE and should not be "
                "extrapolated to propensity-score extremes."
            ),
        },
        {
            "component": "Analysis",
            "ideal_target_trial": (
                "Intention-to-treat analysis with valid randomization-based "
                "inference."
            ),
            "observational_emulation": (
                "Patient-level nested 5-fold cross-fitting, augmented overlap "
                "weighting, and facility-clustered multiplier inference."
            ),
            "limitation": (
                "Cross-fitting limits overfitting bias but does not create "
                "randomization or independent external validation."
            ),
        },
    ]
    pd.DataFrame(rows).to_csv(tables / "target_trial_specification.csv", index=False)


def _cohort_flow_data(
    data: RevisionData,
    output: Path,
    tables: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = data.config
    raw = load_dataset_csv(config["dataset_csv_path"])
    facility = config["identifiers"]["facility"]
    treatment = config["treatment"]["source"]
    steps: list[tuple[str, pd.Series]] = [
        ("Facility excluded during data reconciliation", raw[facility] == 61),
        (
            "No in-hospital physician resuscitation",
            raw["適格基準4 (院内医師処置有無)"] != 1,
        ),
        (
            "Specified noncardiac external cause (codes 1–3)",
            raw["非心原性外因内訳(JAAM)"].isin([1, 2, 3]),
        ),
        (
            "Age <18 years or age missing",
            ~(raw["年齢(JAAM)"] >= 18),
        ),
        ("CAG status missing", raw[treatment].isna()),
    ]
    keep = pd.Series(True, index=raw.index)
    exclusion_rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "reason": "JAAM-OHCA registry records, 2014–2019",
            "excluded_at_step": 0,
            "remaining_n": int(len(raw)),
        }
    ]
    for index, (reason, mask) in enumerate(steps, start=1):
        excluded = int((keep & mask).sum())
        keep &= ~mask
        exclusion_rows.append(
            {
                "step": index,
                "reason": reason,
                "excluded_at_step": excluded,
                "remaining_n": int(keep.sum()),
            }
        )
    exclusions = pd.DataFrame(exclusion_rows)
    if int(exclusions.iloc[-1]["remaining_n"]) != len(data.frame):
        raise RuntimeError("Reporting cohort flow does not match analysis cohort")
    exclusions.to_csv(tables / "cohort_exclusions.csv", index=False)

    estimates = pd.read_csv(output / "main_estimates.csv")
    outcome_flow = estimates[
        [
            "outcome",
            "outcome_label",
            "eligible_n",
            "observed_n",
            "missing_n",
            "treated_n",
            "untreated_n",
            "events",
        ]
    ].copy()
    outcome_flow["treated_missing_n"] = (
        int(data.frame["_treatment"].sum()) - outcome_flow["treated_n"]
    )
    outcome_flow["untreated_missing_n"] = (
        int((data.frame["_treatment"] == 0).sum()) - outcome_flow["untreated_n"]
    )
    outcome_flow.to_csv(tables / "outcome_flow.csv", index=False)
    return exclusions, outcome_flow


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = "#333333",
    fontsize: float = 9,
    linewidth: float = 1.1,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#444444",
    linestyle: str = "-",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linewidth": 1.2,
            "linestyle": linestyle,
        },
    )


def _causal_framework_figure(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(
        ax,
        0.03,
        0.64,
        0.18,
        0.18,
        "Baseline information X\n(facility, prehospital,\nearly in-hospital)",
        facecolor=LIGHT_BLUE,
    )
    _box(
        ax,
        0.29,
        0.64,
        0.18,
        0.18,
        "Observed strategy A\nat conceptual time zero",
        facecolor=LIGHT_ORANGE,
    )
    _box(
        ax,
        0.57,
        0.76,
        0.18,
        0.14,
        "A = 1\nCAG-based strategy",
        facecolor=LIGHT_ORANGE,
    )
    _box(
        ax,
        0.57,
        0.53,
        0.18,
        0.14,
        "A = 0\nNo-CAG strategy",
        facecolor=LIGHT_BLUE,
    )
    _box(
        ax,
        0.81,
        0.76,
        0.16,
        0.14,
        "Potential outcome\nY(1)",
        facecolor=LIGHT_GREEN,
    )
    _box(
        ax,
        0.81,
        0.53,
        0.16,
        0.14,
        "Potential outcome\nY(0)",
        facecolor=LIGHT_GREEN,
    )
    _arrow(ax, (0.21, 0.73), (0.29, 0.73))
    _arrow(ax, (0.47, 0.73), (0.57, 0.83))
    _arrow(ax, (0.47, 0.73), (0.57, 0.60))
    _arrow(ax, (0.75, 0.83), (0.81, 0.83))
    _arrow(ax, (0.75, 0.60), (0.81, 0.60))

    ax.text(
        0.89,
        0.45,
        "For each patient, only Y(A) is observed.\n"
        "Y(1−A) is unobserved and must be modeled under the stated "
        "assumptions.",
        ha="center",
        va="top",
        fontsize=9,
    )
    _box(
        ax,
        0.15,
        0.26,
        0.70,
        0.12,
        "Held-out DR effect-ranking score: M_DR(X)\n"
        "Primary estimand: ATO in the overlap population  |  "
        "Heterogeneity: grouped GATES",
        facecolor="#f5f5f5",
        fontsize=9,
    )
    _box(
        ax,
        0.08,
        0.05,
        0.84,
        0.12,
        "Interpretive boundary: recorded CAG operationalizes entry into an "
        "invasive management pathway.\nExact CAG timing, strategy versions, "
        "and unmeasured clinical selection remain unresolved.",
        facecolor="#fff5e6",
        edgecolor=ORANGE,
        fontsize=9,
    )
    ax.set_title(
        "Causal framework for the CAG-based versus no-CAG strategy contrast",
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    _save_figure(fig, figures / "causal_framework")


def _cohort_flow_figure(
    data: RevisionData,
    exclusions: pd.DataFrame,
    outcomes: pd.DataFrame,
    figures: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    raw_n = int(exclusions.iloc[0]["remaining_n"])
    eligible_n = int(exclusions.iloc[-1]["remaining_n"])
    treated_n = int(data.frame["_treatment"].sum())
    untreated_n = int((data.frame["_treatment"] == 0).sum())
    exclusion_text = "Sequential exclusions\n" + "\n".join(
        f"• {row.reason}: {int(row.excluded_at_step):,}"
        for row in exclusions.iloc[1:].itertuples()
    )
    _box(
        ax,
        0.27,
        0.86,
        0.46,
        0.10,
        f"JAAM-OHCA registry, 2014–2019\nN = {raw_n:,}",
        facecolor=LIGHT_BLUE,
        fontsize=11,
    )
    _box(
        ax,
        0.66,
        0.64,
        0.31,
        0.18,
        exclusion_text,
        facecolor="#fff5e6",
        edgecolor=ORANGE,
        fontsize=8.5,
    )
    _box(
        ax,
        0.27,
        0.66,
        0.30,
        0.10,
        f"Eligible analysis cohort\nN = {eligible_n:,}",
        facecolor=LIGHT_GREEN,
        fontsize=11,
    )
    _arrow(ax, (0.50, 0.86), (0.42, 0.76))
    _arrow(ax, (0.68, 0.86), (0.75, 0.82))

    _box(
        ax,
        0.08,
        0.48,
        0.34,
        0.10,
        f"CAG-based strategy\nn = {treated_n:,}",
        facecolor=LIGHT_ORANGE,
        fontsize=10,
    )
    _box(
        ax,
        0.58,
        0.48,
        0.34,
        0.10,
        f"No-CAG strategy\nn = {untreated_n:,}",
        facecolor=LIGHT_BLUE,
        fontsize=10,
    )
    _arrow(ax, (0.40, 0.66), (0.25, 0.58))
    _arrow(ax, (0.44, 0.66), (0.75, 0.58))

    columns = [
        "Outcome",
        "Observed",
        "Missing",
        "CAG observed",
        "No-CAG observed",
        "Events",
    ]
    cell_text = [
        [
            row.outcome_label,
            f"{int(row.observed_n):,}",
            f"{int(row.missing_n):,}",
            f"{int(row.treated_n):,}",
            f"{int(row.untreated_n):,}",
            f"{int(row.events):,}",
        ]
        for row in outcomes.itertuples()
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.04, 0.12, 0.92, 0.26],
        colWidths=[0.23, 0.13, 0.13, 0.15, 0.17, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#888888")
    ax.text(
        0.5,
        0.06,
        "Every patient with an observed outcome received one held-out "
        "prediction through 5-fold cross-fitting; cross-fitting caused no "
        "additional sample exclusion.",
        ha="center",
        va="center",
        fontsize=9,
    )
    ax.set_title("Cohort and outcome-specific analysis flow", fontsize=13)
    fig.tight_layout()
    _save_figure(fig, figures / "cohort_outcome_flow")


def _cross_fitting_figure(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    boxes = [
        (
            0.015,
            "1. Outer split",
            "Create 5 patient-level\nfolds dispersed across\nfacilities.",
            LIGHT_BLUE,
        ),
        (
            0.210,
            "2. Outer training",
            "Exclude fold k.\nFit treatment and\narm-specific outcome\nmodels.",
            LIGHT_ORANGE,
        ),
        (
            0.405,
            "3. Nested effect training",
            "Within outer training,\ncreate inner held-out DR\npseudo-outcomes "
            "and fit\nthe effect model.",
            LIGHT_ORANGE,
        ),
        (
            0.600,
            "4. Untouched fold k",
            "After all models are fixed,\npredict propensity, potential\n"
            "outcomes, DR score, and\neffect rank in fold k.",
            LIGHT_GREEN,
        ),
        (
            0.795,
            "5. Pool folds",
            "Pool one held-out record\nper patient. Estimate ATO,\n"
            "diagnostics, and GATES with\nfacility-clustered inference.",
            LIGHT_BLUE,
        ),
    ]
    width = 0.18
    for x, title, body, color in boxes:
        _box(
            ax,
            x,
            0.50,
            width,
            0.30,
            f"{title}\n\n{body}",
            facecolor=color,
            fontsize=7.5,
        )
    for index in range(len(boxes) - 1):
        _arrow(
            ax,
            (boxes[index][0] + width, 0.65),
            (boxes[index + 1][0], 0.65),
        )
    _box(
        ax,
        0.12,
        0.16,
        0.76,
        0.15,
        "Cross-fitting makes each reported prediction out of sample for that "
        "patient.\nIt is not hyperparameter cross-validation, randomization, "
        "or independent external validation.",
        facecolor="#f5f5f5",
        fontsize=9.5,
    )
    ax.set_title(
        "Nested cross-fitting workflow for causal nuisance and effect models",
        fontsize=13,
        pad=12,
    )
    fig.tight_layout()
    _save_figure(fig, figures / "nested_cross_fitting_workflow")


def generate_descriptive_artifacts(
    config_path: str | Path,
    output: Path,
    figures: Path,
    tables: Path,
) -> dict[str, int]:
    data = prepare_revision_data(config_path)
    _write_characteristic_tables(data, tables)
    _write_covariate_tables(data, tables)
    _write_target_trial_table(tables)
    exclusions, outcomes = _cohort_flow_data(data, output, tables)
    _causal_framework_figure(figures)
    _cohort_flow_figure(data, exclusions, outcomes, figures)
    _cross_fitting_figure(figures)
    return {
        "eligible_n": int(len(data.frame)),
        "main_feature_n": int(len(data.main_features)),
        "outcome_n": int(len(data.outcomes)),
        "additional_figures": 3,
    }
