"""Tests for jackdaw.env.cashout_mirror.dollars_after_cashout.

Verifies the mirror against the REAL engine's cash-out result (never a
re-derived formula — see the module docstring's "solver/env divergence"
rule): drive a run to a cleared blind (``GamePhase.ROUND_EVAL``), call the
mirror, confirm it did not mutate ``gs``, then let the engine actually cash
out and assert the mirror predicted the exact post-cash-out dollar total.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from jackdaw.engine.actions import CashOut, GamePhase, PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.cashout_mirror import dollars_after_cashout

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_gs(seed: str = "CASHOUTMIRROR") -> dict[str, Any]:
    return initialize_run("b_red", 1, seed)


def _beat_blind_at(gs: dict[str, Any], *, dollars: int, boss: bool = False) -> None:
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    gs["dollars"] = dollars
    if boss:
        gs["blind"].boss = True
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    assert gs["phase"] == GamePhase.ROUND_EVAL


def _assert_mirror_matches_real_cashout(gs: dict[str, Any]) -> None:
    """Call the mirror, confirm non-mutation, then compare vs the real path."""
    before = copy.deepcopy(gs)
    predicted = dollars_after_cashout(gs)

    # The mirror must not have mutated the live state at all.
    assert gs["phase"] == before["phase"] == GamePhase.ROUND_EVAL
    assert gs["dollars"] == before["dollars"]
    assert gs.get("shop_cards") == before.get("shop_cards")

    step(gs, CashOut())
    assert gs["dollars"] == predicted


# ---------------------------------------------------------------------------
# Basic cases
# ---------------------------------------------------------------------------


class TestBasicCashout:
    def test_no_interest_threshold_crossed(self):
        gs = _init_gs()
        _beat_blind_at(gs, dollars=0)
        _assert_mirror_matches_real_cashout(gs)

    def test_interest_threshold_crossed(self):
        gs = _init_gs("CASHOUTMIRROR_INTEREST")
        _beat_blind_at(gs, dollars=17)  # comfortably above one $5 bracket
        _assert_mirror_matches_real_cashout(gs)

    def test_negative_dollars(self):
        """Credit Card debt states (h1 stage regen samples these)."""
        gs = _init_gs("CASHOUTMIRROR_NEGATIVE")
        _beat_blind_at(gs, dollars=-5)
        _assert_mirror_matches_real_cashout(gs)


# ---------------------------------------------------------------------------
# Hands-remaining payout
# ---------------------------------------------------------------------------


class TestHandsRemainingPayout:
    def test_unused_hands_bonus_included(self):
        gs = _init_gs("CASHOUTMIRROR_HANDS")
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        gs["dollars"] = 2
        # Win with the very first hand played, leaving the rest unused.
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["current_round"]["hands_left"] > 0
        _assert_mirror_matches_real_cashout(gs)


# ---------------------------------------------------------------------------
# Golden Joker / Rocket present (end-of-round payout jokers)
# ---------------------------------------------------------------------------


class TestPayoutJokersPresent:
    def test_golden_joker(self):
        gs = _init_gs("CASHOUTMIRROR_GOLDEN")
        gs["jokers"].append(create_joker("j_golden"))
        _beat_blind_at(gs, dollars=1)  # crosses $5 only via the joker payout
        _assert_mirror_matches_real_cashout(gs)

    def test_rocket(self):
        gs = _init_gs("CASHOUTMIRROR_ROCKET")
        gs["jokers"].append(create_joker("j_rocket"))
        _beat_blind_at(gs, dollars=8)
        _assert_mirror_matches_real_cashout(gs)

    def test_golden_and_rocket_together(self):
        gs = _init_gs("CASHOUTMIRROR_BOTH")
        gs["jokers"].append(create_joker("j_golden"))
        gs["jokers"].append(create_joker("j_rocket"))
        _beat_blind_at(gs, dollars=3)
        _assert_mirror_matches_real_cashout(gs)


# ---------------------------------------------------------------------------
# Investment tag pending (fires only after a boss blind)
# ---------------------------------------------------------------------------


class TestInvestmentTagPending:
    def test_investment_after_boss(self):
        gs = _init_gs("CASHOUTMIRROR_INVESTMENT")
        gs.setdefault("awarded_tags", []).append(
            {"key": "tag_investment", "result": None, "blind": "Small"}
        )
        _beat_blind_at(gs, dollars=0, boss=True)
        _assert_mirror_matches_real_cashout(gs)

    def test_investment_pending_non_boss_does_not_pay(self):
        gs = _init_gs("CASHOUTMIRROR_INVESTMENT_NONBOSS")
        gs.setdefault("awarded_tags", []).append(
            {"key": "tag_investment", "result": None, "blind": "Small"}
        )
        _beat_blind_at(gs, dollars=4)  # blind.boss stays False
        _assert_mirror_matches_real_cashout(gs)


# ---------------------------------------------------------------------------
# Gold Seal cards held (round-end dollars applied before interest)
# ---------------------------------------------------------------------------


class TestGoldSealHeld:
    def test_gold_seal_card_held_at_round_end(self):
        gs = _init_gs("CASHOUTMIRROR_GOLDSEAL")
        step(gs, SelectBlind())
        gs["hand"][-1].seal = "Gold"  # held card, not among the played 0-4
        gs["blind"].chips = 1
        gs["dollars"] = 3  # +$3 gold seal crosses the $5 bracket
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        _assert_mirror_matches_real_cashout(gs)


# ---------------------------------------------------------------------------
# Rental joker (subtracted before interest)
# ---------------------------------------------------------------------------


class TestRentalJoker:
    def test_rental_joker_deducted(self):
        gs = _init_gs("CASHOUTMIRROR_RENTAL")
        gs["jokers"].append(create_joker("j_joker", rental=True))
        _beat_blind_at(gs, dollars=10)
        _assert_mirror_matches_real_cashout(gs)


# ---------------------------------------------------------------------------
# Guard: wrong phase
# ---------------------------------------------------------------------------


class TestPhaseGuard:
    def test_raises_outside_round_eval(self):
        gs = _init_gs("CASHOUTMIRROR_PHASEGUARD")
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND
        with pytest.raises(ValueError):
            dollars_after_cashout(gs)
