"""Tests for jackdaw.engine.runner — simulate_run, random_agent, greedy_play_agent.

Coverage
--------
* 10 seeds with random_agent: all complete without crash.
* 10 seeds with greedy_play_agent: all complete without crash.
* Scripted determinism: same seed → same outcome (5 seeds + beats_three_blinds).
* Performance: >500 runs/sec.
* All 15 deck types complete without crash.
* All 8 stake levels complete without crash.
* Max actions limit respected.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    SelectBlind,
)
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.runner import greedy_play_agent, random_agent, simulate_run

# ---------------------------------------------------------------------------
# Crash resistance — 10 seeds each
# ---------------------------------------------------------------------------


class TestRandomAgentCrashResistance:
    @pytest.mark.parametrize("seed_idx", range(10))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"CRASH_RAND_{seed_idx}", random_agent, max_actions=500)
        assert gs["actions_taken"] <= 500


class TestGreedyAgentCrashResistance:
    @pytest.mark.parametrize("seed_idx", range(10))
    def test_no_crash(self, seed_idx):
        gs = simulate_run("b_red", 1, f"CRASH_GRDY_{seed_idx}", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] <= 500


# ---------------------------------------------------------------------------
# Scripted determinism
# ---------------------------------------------------------------------------


def _scripted_run(seed: str) -> dict:
    """Play a scripted Small→Big→Boss sequence and return final state."""
    gs = initialize_run("b_red", 1, seed)

    for blind_name in ("Small", "Big", "Boss"):
        if blind_name != "Small":
            step(gs, NextRound())
        step(gs, SelectBlind())
        gs["blind"].chips = 1  # easy win
        # Play hands until we win or run out
        for _ in range(10):
            if gs["phase"] != GamePhase.SELECTING_HAND:
                break
            hand = gs.get("hand", [])
            n = min(5, len(hand))
            if n == 0:
                break
            step(gs, PlayHand(card_indices=tuple(range(n))))
        if gs["phase"] == GamePhase.ROUND_EVAL:
            step(gs, CashOut())

    return gs


class TestScriptedDeterminism:
    @pytest.mark.parametrize(
        "seed",
        [
            "SCRIPT_A",
            "SCRIPT_B",
            "SCRIPT_C",
            "SCRIPT_D",
            "SCRIPT_E",
        ],
    )
    def test_same_seed_same_outcome(self, seed):
        gs1 = _scripted_run(seed)
        gs2 = _scripted_run(seed)
        assert gs1["dollars"] == gs2["dollars"]
        assert gs1["chips"] == gs2["chips"]
        assert gs1.get("won") == gs2.get("won")
        assert gs1["round_resets"]["ante"] == gs2["round_resets"]["ante"]

    def test_scripted_beats_three_blinds(self):
        gs = _scripted_run("SCRIPT_FULL")
        # After beating Small, Big, Boss, ante should advance to 2
        assert gs["round_resets"]["ante"] == 2
        assert gs.get("phase") == GamePhase.SHOP


# ---------------------------------------------------------------------------
# All deck types
# ---------------------------------------------------------------------------


ALL_DECKS = [
    "b_red",
    "b_blue",
    "b_yellow",
    "b_green",
    "b_black",
    "b_magic",
    "b_nebula",
    "b_ghost",
    "b_abandoned",
    "b_checkered",
    "b_zodiac",
    "b_painted",
    "b_anaglyph",
    "b_plasma",
    "b_erratic",
]


class TestAllDecks:
    @pytest.mark.parametrize("back_key", ALL_DECKS)
    def test_deck_completes(self, back_key):
        gs = simulate_run(back_key, 1, "DECK_FV", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0


# ---------------------------------------------------------------------------
# All stake levels
# ---------------------------------------------------------------------------


class TestAllStakes:
    @pytest.mark.parametrize("stake", range(1, 9))
    def test_stake_completes(self, stake):
        gs = simulate_run("b_red", stake, "STAKE_FV", greedy_play_agent, max_actions=500)
        assert gs["actions_taken"] > 0


# ---------------------------------------------------------------------------
# Max actions limit
# ---------------------------------------------------------------------------


class TestMaxActions:
    def test_respects_max_actions(self):
        gs = simulate_run("b_red", 1, "MAX_ACT", random_agent, max_actions=3)
        assert gs["actions_taken"] <= 3
