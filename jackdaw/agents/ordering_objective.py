"""Clear-gated lexicographic objective for copy-joker placement.

The clear gate uses the engine's exact predicate fields: ``game_state["chips"]``
and ``game_state["blind"].chips``.  A clearing play ends the round, so surplus
chips have no value and dollars are the first tie-break; a non-clearing play
still banks chips toward clearing with the remaining hands, and failing the
blind ends the run (making banked dollars worthless), so score is the safe
first tie-break and dollars are last.

The first tuple element identifies the arm.  A tie there means both candidates
are in the same arm, making the mixed tuple positions sound.  The factory must
be called for each hand-turn decision because it snapshots ``banked`` and
``need`` at build time.  Known limitation: the objective is build-blind to
deliberate money-farming lines (the non-clearing arm counts dollars only as a
tie-break); the named upgrade path is a learned copy-target pick at the
in-blind merge, outside this objective.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jackdaw.engine.scoring import ScoreResult

__all__ = ["make_clear_gated_money_objective"]


def make_clear_gated_money_objective(
    game_state,
) -> Callable[[ScoreResult], tuple[float, float, float]]:
    """Build a clear-gated score/dollars lexicographic objective."""
    banked = game_state.get("chips", 0)
    need = getattr(game_state.get("blind"), "chips", 0) or 0

    def objective(result: ScoreResult) -> tuple[float, float, float]:
        if banked + result.total >= need:
            return (1.0, float(result.dollars_earned), float(result.total))
        return (0.0, float(result.total), float(result.dollars_earned))

    return objective
