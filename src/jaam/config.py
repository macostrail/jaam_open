from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
VARIABLES_DIR = CONFIG_DIR / "variables"
RUN_DIR = PROJECT_ROOT / "runs"
JAAM_2021_SHEET = "データ補正、列移動"

DATASET_CACHE_DIR = RUN_DIR / "datasets"
DEFAULT_DATASET_CSV = DATASET_CACHE_DIR / "jaam_2021_dataset.csv"
