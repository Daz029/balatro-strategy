"""Tests for ShopRunAdapter — full-run episodes with auto-resolved hands.

The two properties everything downstream depends on:
1. Control only ever returns at a decision phase (SHOP / PACK_OPENING) or
   at episode end — the shop agent never sees a hand/blind-select state.
2. snapshot_state/restore_state round-trips the COMPLETE engine state,
   RNG included: a restored run continues byte-identically. The start-state
   reservoir is built on this.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy
from jackdaw.engine.actions import (
    BuyCard,
    GamePhase,
    NextRound,
    OpenBooster,
    Reroll,
    SkipPack,
)
from jackdaw.env.shop_run_adapter import (
    DECISION_PHASES,
    ShopRunAdapter,
    ShopRunConfig,
)

SEED = "SHOPRUN_TEST"


def _adapter(win_ante: int = 8) -> ShopRunAdapter:
    return ShopRunAdapter(GreedyHandPolicy(), ShopRunConfig(win_ante=win_ante))


def _shop_summary(gs: dict[str, Any]) -> tuple:
    """Comparable fingerprint of the current shop + economy state."""
    return (
        tuple(c.center_key for c in gs.get("shop_cards", [])),
        tuple(v.center_key for v in gs.get("shop_vouchers", [])),
        tuple(b.center_key for b in gs.get("shop_boosters", [])),
        gs.get("dollars"),
        gs.get("round"),
        gs.get("round_resets", {}).get("ante"),
    )


class TestEpisodeFlow:
    def test_reset_lands_on_first_shop(self):
        adapter = _adapter()
        state = adapter.reset("b_red", 1, SEED)
        assert state.phase == GamePhase.SHOP
        assert state.ante == 1
        assert state.round == 1  # one blind cleared

    def test_control_only_at_decision_points(self):
        adapter = _adapter()
        adapter.reset("b_red", 1, SEED)
        for _ in range(64):  # buy nothing, just advance blinds
            if adapter.done:
                break
            state = adapter.step(NextRound())
            assert adapter.done or state.phase in DECISION_PHASES
        assert adapter.done  # a no-purchase run must end within bounds

    def test_horizon_curriculum_win_ante(self):
        # win_ante=1: beating the ante-1 boss ends the episode as a win.
        # Seed pinned to one the greedy baseline clears (it wins ~4/6 of
        # ante-1 runs; SHOPRUN_TEST's Big Blind draws happen to beat it).
        adapter = _adapter(win_ante=1)
        adapter.reset("b_red", 1, "SHOPRUN_B")
        for _ in range(8):
            if adapter.done:
                break
            adapter.step(NextRound())
        assert adapter.done
        assert adapter.won
        # The engine advances the ante on boss defeat before the won flag
        # halts the episode, so a win at win_ante=N leaves ante == N+1.
        assert adapter.raw_state["round_resets"]["ante"] == 2

    def test_round_counts_cleared_blinds(self):
        adapter = _adapter()
        s0 = adapter.reset("b_red", 1, SEED)
        s1 = adapter.step(NextRound())
        if not adapter.done:
            assert s1.round == s0.round + 1  # one more blind cleared

    def test_pack_opening_is_a_decision_point(self):
        adapter = _adapter()
        state = adapter.reset("b_red", 1, SEED)
        gs = adapter.raw_state
        gs["dollars"] = 50  # afford any pack
        boosters = gs.get("shop_boosters", [])
        assert boosters, "shop must offer 2 packs"
        state = adapter.step(OpenBooster(card_index=0))
        assert state.phase == GamePhase.PACK_OPENING
        state = adapter.step(SkipPack())
        assert state.phase == GamePhase.SHOP


class TestSnapshotRestore:
    def test_restore_is_byte_identical_continuation(self):
        adapter = _adapter()
        adapter.reset("b_red", 1, SEED)
        adapter.raw_state["dollars"] = 50
        blob = adapter.snapshot_state()

        # Continuation A: reroll, then advance to the next shop.
        adapter.step(Reroll())
        a_after_reroll = _shop_summary(adapter.raw_state)
        adapter.step(NextRound())
        a_next = _shop_summary(adapter.raw_state)
        a_done = adapter.done

        # Continuation B: restore, repeat the same actions.
        state = adapter.restore_state(blob)
        assert state.phase == GamePhase.SHOP
        adapter.step(Reroll())
        assert _shop_summary(adapter.raw_state) == a_after_reroll
        adapter.step(NextRound())
        assert _shop_summary(adapter.raw_state) == a_next
        assert adapter.done == a_done

    def test_restore_into_fresh_adapter_instance(self):
        adapter = _adapter()
        adapter.reset("b_red", 1, SEED)
        adapter.raw_state["dollars"] = 50
        blob = adapter.snapshot_state()
        adapter.step(Reroll())
        expected = _shop_summary(adapter.raw_state)

        other = _adapter()
        other.restore_state(blob)
        other.step(Reroll())
        assert _shop_summary(other.raw_state) == expected

    def test_snapshot_does_not_alias_live_state(self):
        adapter = _adapter()
        adapter.reset("b_red", 1, SEED)
        blob = adapter.snapshot_state()
        dollars_at_snapshot = adapter.raw_state["dollars"]
        adapter.raw_state["dollars"] = 999
        restored = _adapter()
        state = restored.restore_state(blob)
        assert state.dollars == dollars_at_snapshot


class TestPurchases:
    def test_buy_card_at_shop(self):
        adapter = _adapter()
        adapter.reset("b_red", 1, SEED)
        gs = adapter.raw_state
        gs["dollars"] = 50
        n_before = len(gs["shop_cards"])
        state = adapter.step(BuyCard(shop_index=0))
        # Still our decision point; one fewer card on the shelf.
        assert state.phase in DECISION_PHASES
        assert len(gs["shop_cards"]) == n_before - 1


class TestFailLoud:
    def test_stuck_hand_policy_raises(self):
        calls = {"n": 0}

        def bad_policy(gs):
            # Discards forever without ever playing (never decrements
            # hands_left; discards run out -> engine will reject, but a
            # policy returning the same rejected action would loop).
            from jackdaw.engine.actions import SortHand

            calls["n"] += 1
            return SortHand(mode="rank")  # legal no-op: never advances phase

        adapter = ShopRunAdapter(bad_policy)
        with pytest.raises(RuntimeError, match="auto-advance exceeded"):
            adapter.reset("b_red", 1, SEED)
