"""Interim env-side bridges to the vanilla engine (Phase 6; CLAUDE.md TODO 4)."""

from __future__ import annotations

from jackdaw.engine.actions import Discard, PlayHand, SelectBlind
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.interim_constraints import with_cerulean_bell


def _cerulean_state() -> dict:
    gs = initialize_run("b_red", 1, "INTERIM_CERULEAN")
    step(gs, SelectBlind())
    gs["hand"][3].ability["forced_selection"] = True
    gs["blind"].chips = 10**9
    return gs


def test_completes_a_play_missing_the_forced_card() -> None:
    gs = _cerulean_state()
    action = with_cerulean_bell(gs, PlayHand(card_indices=(0, 1)))
    assert action == PlayHand(card_indices=(0, 1, 3))
    step(gs, action)  # the vanilla engine accepts it


def test_completes_a_discard_and_respects_the_five_card_cap() -> None:
    gs = _cerulean_state()
    action = with_cerulean_bell(gs, Discard(card_indices=(0, 1, 2, 4, 5)))
    assert action == Discard(card_indices=(0, 1, 2, 4, 3))
    step(gs, action)


def test_no_op_without_a_forced_card() -> None:
    gs = initialize_run("b_red", 1, "INTERIM_PLAIN")
    step(gs, SelectBlind())
    action = PlayHand(card_indices=(0, 1))
    assert with_cerulean_bell(gs, action) == action
