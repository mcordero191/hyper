"""Dataset subpackage for hyperMLT.

This checkpoint intentionally starts minimal. Behavior will be migrated here
incrementally from legacy `hyper`.
"""
from .smr import PreparedMeteorWindow, build_window_summary, load_first_window, load_window_for_file

__all__ = [
    "PreparedMeteorWindow",
    "build_window_summary",
    "load_first_window",
    "load_window_for_file",
]
