"""Revised, reviewer-facing causal analysis pipeline.

This package is intentionally separate from the historical train/test workflow in
``jaam.train``.  The separation preserves the submitted analysis while making the
revised estimands, sample splitting, diagnostics, and outputs explicit.
"""

from .data import OutcomeSpec, RevisionData

__all__ = ["OutcomeSpec", "RevisionData"]
