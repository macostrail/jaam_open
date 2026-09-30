from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from jaam.data import load_dataset_csv
from jaam.variables import Variables


@dataclass(frozen=True)
class OutcomeSpec:
    key: str
    source: str
    label: str


@dataclass
class RevisionData:
    frame: pd.DataFrame
    outcomes: dict[str, OutcomeSpec]
    central_outcome: str
    main_features: list[str]
    early_features: list[str]
    categorical_features: set[str]
    numeric_features: set[str]
    config: dict[str, Any]


def load_revision_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    if "analysis" not in config:
        raise ValueError("Revision config must contain an 'analysis' section")
    return config["analysis"]


def _assert_binary(series: pd.Series, name: str) -> None:
    values = set(series.dropna().unique().tolist())
    if not values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"{name} is not binary after cohort preparation: {values}")


def _unique_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def prepare_revision_data(config_path: str | Path) -> RevisionData:
    config = load_revision_config(config_path)
    source = config["dataset_csv_path"]
    df = load_dataset_csv(source)
    variables = Variables()

    treatment_cfg = config["treatment"]
    treatment_source = treatment_cfg["source"]
    facility_source = config["identifiers"]["facility"]
    year_source = config["identifiers"]["canonical_year"]

    eligibility = (
        (df[facility_source] != 61)
        & (df["適格基準4 (院内医師処置有無)"] == 1)
        & (~df["非心原性外因内訳(JAAM)"].isin([1, 2, 3]))
        & (df["年齢(JAAM)"] >= 18)
        & df[treatment_source].notna()
    )
    cohort = df.loc[eligibility].copy()
    cohort.reset_index(drop=False, names="_source_row", inplace=True)
    cohort["_row_id"] = np.arange(len(cohort), dtype=np.int64)
    cohort["_facility"] = cohort[facility_source].astype("int64")
    cohort["_year"] = pd.to_numeric(cohort[year_source], errors="coerce")
    cohort["_treatment"] = cohort[treatment_source].replace(
        {
            treatment_cfg["treated_code"]: 1,
            treatment_cfg["untreated_code"]: 0,
        }
    )
    _assert_binary(cohort["_treatment"], "_treatment")
    cohort["_treatment"] = cohort["_treatment"].astype("int8")

    if cohort["_year"].isna().any():
        raise ValueError(
            f"{year_source} contains missing values in the analysis cohort"
        )
    cohort["_year"] = cohort["_year"].astype("int16")

    outcome_specs: dict[str, OutcomeSpec] = {}
    for key, item in config["outcomes"].items():
        spec = OutcomeSpec(key=key, source=item["source"], label=item["label"])
        outcome_specs[key] = spec
        _assert_binary(cohort[spec.source], spec.source)
        cohort[f"_outcome_{key}"] = cohort[spec.source].astype("float64")

    feature_cfg = config["features"]
    early_source = feature_cfg["early_ecg_source"]
    engineered_ecg = feature_cfg["engineered_early_ecg"]
    cohort[engineered_ecg] = cohort[early_source]

    redundant_ecg = {
        early_source,
        *feature_cfg["redundant_early_ecg_sources"],
    }
    phase2_collapsed = [
        col for col in variables.phase2_vars if col not in redundant_ecg
    ]
    main_features = _unique_preserving_order(
        [*variables.phase1_vars, *phase2_collapsed, engineered_ecg]
    )
    early_features = _unique_preserving_order([*variables.phase1_vars, engineered_ecg])

    typing = variables.numeric_or_categoric_df
    numeric = set(typing["numerical"].dropna().astype(str))
    categorical = set(typing["categorical"].dropna().astype(str))
    categorical.add(engineered_ecg)
    numeric.discard(engineered_ecg)

    required = {
        *main_features,
        "_facility",
        "_year",
        "_treatment",
        *(f"_outcome_{key}" for key in outcome_specs),
    }
    missing = sorted(required.difference(cohort.columns))
    if missing:
        raise ValueError(f"Required columns are missing from the dataset: {missing}")

    untyped = sorted(set(main_features).difference(numeric | categorical))
    if untyped:
        raise ValueError(f"Features lack numeric/categorical typing: {untyped}")

    # A feature that is entirely absent cannot be learned.  This decision uses no
    # treatment or outcome information and is recorded in the audit outputs.
    main_features = [col for col in main_features if cohort[col].notna().any()]
    early_features = [col for col in early_features if cohort[col].notna().any()]

    return RevisionData(
        frame=cohort,
        outcomes=outcome_specs,
        central_outcome=config["central_outcome"],
        main_features=main_features,
        early_features=early_features,
        categorical_features=categorical,
        numeric_features=numeric,
        config=config,
    )


def make_balanced_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    outcome_column: str,
) -> np.ndarray:
    """Disperse patients from each facility across folds while balancing strata.

    Strata combine treatment with the central outcome state (missing, 0, or 1).
    The greedy assignment is deterministic for a fixed seed and keeps small
    facilities in the analysis instead of requiring every facility/stratum cell
    to contain at least ``n_splits`` observations.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(frame) < n_splits:
        raise ValueError("The analysis subset is smaller than n_splits")

    rng = np.random.default_rng(seed)
    outcome_state = frame[outcome_column].map({0.0: "y0", 1.0: "y1"}).fillna("ym")
    strata = frame["_treatment"].astype(str) + "_" + outcome_state
    facilities = frame["_facility"].astype(str)

    work = pd.DataFrame(
        {
            "position": np.arange(len(frame), dtype=int),
            "facility": facilities.to_numpy(),
            "stratum": strata.to_numpy(),
        }
    )
    groups = []
    for (facility, stratum), group in work.groupby(["facility", "stratum"], sort=True):
        positions = group["position"].to_numpy(copy=True)
        rng.shuffle(positions)
        groups.append((facility, stratum, positions))
    groups.sort(key=lambda item: (-len(item[2]), item[0], item[1]))

    assignment = np.full(len(frame), -1, dtype=np.int8)
    total_count = np.zeros(n_splits, dtype=int)
    stratum_count: dict[str, np.ndarray] = {}
    facility_count: dict[str, np.ndarray] = {}

    for facility, stratum, positions in groups:
        s_count = stratum_count.setdefault(stratum, np.zeros(n_splits, dtype=int))
        f_count = facility_count.setdefault(facility, np.zeros(n_splits, dtype=int))
        for position in positions:
            score = np.column_stack((f_count, s_count, total_count))
            best_score = score.min(axis=0)
            candidates = np.flatnonzero(np.all(score == best_score, axis=1))
            if len(candidates) == 0:
                scalar = (
                    f_count * max(len(frame), 1) ** 2
                    + s_count * max(len(frame), 1)
                    + total_count
                )
                candidates = np.flatnonzero(scalar == scalar.min())
            chosen = int(rng.choice(candidates))
            assignment[position] = chosen
            f_count[chosen] += 1
            s_count[chosen] += 1
            total_count[chosen] += 1

    if (assignment < 0).any():
        raise RuntimeError("Fold assignment did not cover every row")
    if len(set(assignment.tolist())) != n_splits:
        raise RuntimeError("Fold assignment produced an empty fold")
    return assignment


def cohort_audit(data: RevisionData) -> Mapping[str, Any]:
    frame = data.frame
    audit: dict[str, Any] = {
        "n": int(len(frame)),
        "treated": int(frame["_treatment"].sum()),
        "untreated": int((1 - frame["_treatment"]).sum()),
        "facilities": int(frame["_facility"].nunique()),
        "years": {
            str(int(year)): int(count)
            for year, count in frame["_year"].value_counts().sort_index().items()
        },
        "outcomes": {},
        "feature_sets": {
            "main_count": len(data.main_features),
            "early_count": len(data.early_features),
        },
    }
    for key, spec in data.outcomes.items():
        series = frame[f"_outcome_{key}"]
        audit["outcomes"][key] = {
            "label": spec.label,
            "observed": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "events": int(series.sum(skipna=True)),
        }
    return audit
