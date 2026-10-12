"""Unsupported game modes raise instead of silently diverging (D-scope, 2026-10-12).

Challenges and profiles are covered in ``test_run_init.py``; this file covers
endless (audit D49): the win-ante boss is still beaten, cashed out and shopped
exactly as before, and the FIRST blind decision after the win raises.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.actions import CashOut, GamePhase, NextRound, PlayHand, SelectBlind, SkipBlind
from jackdaw.engine.data.blind_scaling import get_blind_amount
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scope import UnsupportedGameModeError


def _clear_blind(gs: dict[str, Any]) -> None:
    """Select the blind on deck and beat it with one 5-card play at 1 chip."""
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    n = min(5, len(gs["hand"]))
    step(gs, PlayHand(card_indices=tuple(range(n))))
    assert gs["phase"] == GamePhase.ROUND_EVAL, gs["phase"]


def _won_run_in_blind_select() -> dict[str, Any]:
    gs = initialize_run("b_red", 1, "SCOPE_ENDLESS")
    gs["win_ante"] = 1
    for _ in range(3):  # Small, Big, Boss
        _clear_blind(gs)
        if gs.get("won"):
            break
        step(gs, CashOut())
        step(gs, NextRound())
    assert gs["won"] is True
    # Winning still cashes out and shops, as in Lua.
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    step(gs, NextRound())
    assert gs["phase"] == GamePhase.BLIND_SELECT
    return gs


class TestEndlessRaises:
    def test_select_blind_after_win_raises(self) -> None:
        gs = _won_run_in_blind_select()
        with pytest.raises(UnsupportedGameModeError, match="endless"):
            step(gs, SelectBlind())

    def test_skip_blind_after_win_raises(self) -> None:
        gs = _won_run_in_blind_select()
        with pytest.raises(UnsupportedGameModeError, match="endless"):
            step(gs, SkipBlind())

    def test_blind_amount_past_ante_16_raises(self) -> None:
        assert get_blind_amount(16) > 0
        with pytest.raises(UnsupportedGameModeError):
            get_blind_amount(17)
