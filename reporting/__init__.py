"""Reporting helpers: text reports and charts.

``plots`` is intentionally not imported here so that matplotlib stays an
optional dependency: import it explicitly
(``from surface_pricer.reporting.plots import plot_fit_result``).
"""

from .fit_report import format_fit_report
from .quote_report import format_quote, quote_to_dict

__all__ = ["format_fit_report", "format_quote", "quote_to_dict"]
