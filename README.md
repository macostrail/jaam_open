# JAAM-OHCA CAG analysis

Analysis code for the manuscript "Treatment-effect heterogeneity of a coronary
angiography-based invasive strategy after out-of-hospital cardiac arrest: a
causal machine-learning analysis."

## Setup

Python 3.10 and Poetry are required.

```bash
poetry install
```

## Data

The individual-level JAAM-OHCA Registry data are not included because they
cannot be redistributed by the authors. Information about data access is
available from the
[JAAM-OHCA Registry](https://www.jaamohca-web.com/data-use/).

After obtaining an authorized copy of the registry workbook, prepare the local
dataset:

```bash
poetry run python scripts/run_analysis.py \
  --stage prepare \
  --raw-data /path/to/registry-workbook.xlsx \
  --force
```

## Analysis

Run the complete analysis pipeline:

```bash
poetry run python scripts/run_analysis.py --stage all
```

Individual stages can be selected with `--stage`. Run the following command for
the available options:

```bash
poetry run python scripts/run_analysis.py --help
```
