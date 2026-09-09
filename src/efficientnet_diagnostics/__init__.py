"""Exact linear-head channel attribution and coarse spatial evidence."""

from .analysis import diagnose
from .report import save_report
from .intervention import validate_regions

__all__ = ["diagnose", "save_report", "validate_regions"]
