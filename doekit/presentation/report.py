"""Presentation façade: narrative → render → write."""

from __future__ import annotations

from .report_impl import report_html, report_summary, run_report_arg

__all__ = ["report_html", "report_summary", "run_report_arg", "report"]

report = report_html
