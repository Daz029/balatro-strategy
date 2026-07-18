"""Lookup helper for the ``V_curve(ante, dollars)`` artifact produced by
``scripts/extract_v_curve.py`` (offline money-sweep of the frozen s0 shop
critic -- CLAUDE.md "Money/dollar handling", ``docs/post-regen-training-plan.md``
section 3 "Terminal $ term").

Deliberately minimal: this is wave-0 scope. The only planned consumer so
far is the wave-2 hand-env terminal-$ hook (``1 + V_curve(ante,
dollars_after_cashout)`` on a clear), which doesn't exist yet -- this module
just needs to be a stable, simple interface for that wiring to land against
later, not a rich API today.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class VCurve:
    """In-memory lookup over a loaded ``V_curve`` artifact.

    ``value(ante, dollars)`` clamps ``dollars`` into the swept range and
    rounds to the nearest integer cell (the artifact sweeps whole dollars),
    then falls back to the nearest ANTE present in the artifact if the exact
    ante has no cells at all, and to the nearest DOLLAR cell that actually
    has data if the rounded/clamped dollar value itself is missing (e.g. a
    cell dropped for zero samples).
    """

    def __init__(
        self,
        cells: dict[int, dict[int, float]],
        dollar_min: int,
        dollar_max: int,
    ) -> None:
        if not cells:
            raise ValueError("V_curve artifact has no cells")
        self._cells = cells
        self.dollar_min = dollar_min
        self.dollar_max = dollar_max
        self._antes = sorted(cells)

    def value(self, ante: int, dollars: float) -> float:
        """Looked-up mean value for ``(ante, dollars)``, clamped/fallback."""
        nearest_ante = min(self._antes, key=lambda a: abs(a - ante))
        dollar_map = self._cells[nearest_ante]

        clamped = min(max(dollars, self.dollar_min), self.dollar_max)
        rounded = int(round(clamped))
        if rounded in dollar_map:
            return dollar_map[rounded]

        # The exact rounded dollar has no cell (sparse/dropped within the
        # swept range) -- fall back to the nearest dollar that does.
        nearest_dollar = min(dollar_map, key=lambda d: abs(d - rounded))
        return dollar_map[nearest_dollar]

    @property
    def antes(self) -> list[int]:
        return list(self._antes)


def load_v_curve(path: str | Path) -> VCurve:
    """Load a ``V_curve`` artifact written by ``extract_v_curve.py``."""
    with open(path, encoding="utf-8") as fh:
        artifact: dict[str, Any] = json.load(fh)

    meta = artifact["metadata"]
    cells: dict[int, dict[int, float]] = {}
    for ante_str, dollar_map in artifact["cells"].items():
        cells[int(ante_str)] = {int(d): float(cell["mean"]) for d, cell in dollar_map.items()}

    return VCurve(
        cells,
        dollar_min=int(meta["dollar_min"]),
        dollar_max=int(meta["dollar_max"]),
    )
