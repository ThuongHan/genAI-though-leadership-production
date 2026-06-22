"""
Output-path helpers — date/time stamping + finding the latest scan.

Stamp format: DD-MM-HHMM  (e.g. a run on 30 May at 14:30 -> "30-05-1430").
So scanner/filter outputs never overwrite a previous run (unless re-run within
the same minute).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def stamp(dt: datetime | None = None) -> str:
    """Return the current timestamp as DD-MM-HHMM."""
    return (dt or datetime.now()).strftime("%d-%m-%H%M")


def timestamped_path(base_path: str | Path, dt: datetime | None = None) -> str:
    """Insert the DD-MM-HHMM stamp before the extension.

    'data/scans/scanner_output.json' -> 'data/scans/scanner_output_30-05-1430.json'
    """
    p = Path(base_path)
    return str(p.with_name(f"{p.stem}_{stamp(dt)}{p.suffix}"))


def latest_scan(
    directory: str | Path = "data/scans",
    pattern: str = "scanner_output*.json",
) -> str | None:
    """Return the most recently modified scan file matching `pattern`, or None.

    Lets the filters auto-pick the newest scan without typing the dated name.
    """
    files = sorted(
        Path(directory).glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(files[0]) if files else None