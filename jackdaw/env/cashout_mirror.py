"""Cash-out mirror — predict post-cash-out dollars without mutating state.

``HandPlayGymEnv``'s episode ends the instant a blind is cleared
(``gs["phase"] == GamePhase.ROUND_EVAL``), which is BEFORE the engine's real
``CashOut`` action runs (deck shuffle, tag firing, shop population). The h1
terminal reward hook (CLAUDE.md "h1 objective & training" — Terminal $ term;
``docs/post-regen-training-plan.md`` section 3) needs
``dollars_after_cashout`` at exactly that point, to look up
``V_curve(ante, dollars_after_cashout)`` — without permanently advancing the
live episode state.

Design rule (CLAUDE.md — "solver/env divergence is the discard-cap bug
class"): re-deriving the interest/reward formulas here would create a
second implementation that can silently drift from the engine's real one.
Instead this module clones the state (the same RNG-exact pickle round-trip
``ShopRunAdapter.snapshot_state``/``restore_state`` uses) and drives the
CLONE through the engine's own ``CashOut`` handler, then discards it and
keeps only the resulting dollar total. Whatever the engine's actual
money-vs-interest ordering is, this mirror reproduces it exactly, because
it IS that computation, not a re-implementation of it.
"""

from __future__ import annotations

import pickle
from typing import Any

from jackdaw.engine.actions import CashOut, GamePhase
from jackdaw.engine.game import step


def dollars_after_cashout(gs: dict[str, Any]) -> int:
    """Return the dollar total the engine would produce after cash-out.

    ``gs`` must be at ``GamePhase.ROUND_EVAL`` — the state right after a
    blind is cleared and ``_round_won`` has computed ``gs["round_earnings"]``,
    but before ``CashOut`` has been applied. This is exactly the state a
    winning ``HandPlayGymEnv`` episode terminates on.

    ``gs`` is never mutated: a pickle round-trip clone (RNG state included)
    absorbs the real ``CashOut`` handler's side effects — deck shuffle, any
    "eval"/"shop_start" tag context firing (Investment, D6), shop
    population — none of which the caller needs; only the clone's final
    ``dollars`` is read back.
    """
    phase = gs.get("phase")
    if phase != GamePhase.ROUND_EVAL:
        raise ValueError(
            f"dollars_after_cashout requires GamePhase.ROUND_EVAL, got {phase!r}"
        )
    clone = pickle.loads(pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL))
    step(clone, CashOut())
    return clone["dollars"]
