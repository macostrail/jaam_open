from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_DATASET_CSV,
    JAAM_2021_SHEET,
)


def read_raw_dataset(
    source_path: str | Path,
    *,
    sheet_name: str = JAAM_2021_SHEET,
) -> pd.DataFrame:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset source does not exist: {source}")
    return pd.read_excel(source, sheet_name=sheet_name, engine="openpyxl")


def build_dataset_cache(
    source_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sheet_name: str = JAAM_2021_SHEET,
    overwrite: bool = False,
) -> Path:
    out = Path(output_path) if output_path else DEFAULT_DATASET_CSV
    if out.exists() and not overwrite:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    df = read_raw_dataset(source_path, sheet_name=sheet_name)
    df.to_csv(out, index=False)
    return out


def load_dataset_csv(path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else DEFAULT_DATASET_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset CSV does not exist: {csv_path}. "
            "Run `python scripts/run_analysis.py --stage prepare --raw-data PATH` "
            "first."
        )
    return pd.read_csv(csv_path, low_memory=False)
