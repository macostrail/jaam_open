"""Reusable code for the JAAM-OHCA CAG treatment-effect analyses."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jaam")
except PackageNotFoundError:  # pragma: no cover - editable tree before install
    __version__ = "0.0.0"
