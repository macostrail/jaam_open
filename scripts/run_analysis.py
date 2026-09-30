#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from jaam.config import JAAM_2021_SHEET
from jaam.data import build_dataset_cache
from jaam.revision.core import run_main_analysis
from jaam.revision.ipow import run_ipow_analysis
from jaam.revision.mi import (
    run_mi_convergence_diagnostic,
    run_multiple_imputation,
)
from jaam.revision.reporting import run_reporting
from jaam.revision.sensitivity import run_sensitivity_analysis
from jaam.revision.temporal import run_temporal_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final revised JAAM-OHCA analyses."
    )
    parser.add_argument(
        "--config",
        default="config/analysis.yaml",
        help="Path to the frozen analysis YAML specification.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "main",
            "sensitivity",
            "ipow",
            "temporal",
            "mi",
            "mi-diagnostic",
            "report",
            "all",
            "prepare",
        ],
        default="main",
        help="Analysis stage to run. 'all' runs every analysis stage in order.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the requested stage even if its completion marker exists.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="For the MI stage only, run two datasets and two iterations.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        help="Restricted JAAM-OHCA Excel file (required for --stage prepare).",
    )
    parser.add_argument(
        "--sheet",
        default=JAAM_2021_SHEET,
        help="Worksheet containing the analysis data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.stage == "prepare":
        if args.raw_data is None:
            raise SystemExit("--raw-data is required for --stage prepare")
        path = build_dataset_cache(
            args.raw_data,
            sheet_name=args.sheet,
            overwrite=args.force,
        )
        print(path)
    elif args.stage == "all":
        if args.smoke:
            raise SystemExit("--smoke cannot be combined with --stage all")
        run_main_analysis(args.config, force=args.force)
        run_sensitivity_analysis(args.config, force=args.force)
        run_ipow_analysis(args.config, force=args.force)
        run_temporal_validation(args.config, force=args.force)
        run_multiple_imputation(args.config, force=args.force, smoke=False)
        run_mi_convergence_diagnostic(args.config, force=args.force)
        run_reporting(args.config, force=args.force)
    elif args.stage == "main":
        run_main_analysis(args.config, force=args.force)
    elif args.stage == "sensitivity":
        run_sensitivity_analysis(args.config, force=args.force)
    elif args.stage == "ipow":
        run_ipow_analysis(args.config, force=args.force)
    elif args.stage == "temporal":
        run_temporal_validation(args.config, force=args.force)
    elif args.stage == "mi":
        run_multiple_imputation(
            args.config,
            force=args.force,
            smoke=args.smoke,
        )
    elif args.stage == "mi-diagnostic":
        run_mi_convergence_diagnostic(args.config, force=args.force)
    elif args.stage == "report":
        run_reporting(args.config, force=args.force)


if __name__ == "__main__":
    main()
