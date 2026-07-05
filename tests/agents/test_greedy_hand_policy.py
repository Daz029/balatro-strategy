"""Tests for the scripted greedy hand policy (shop env's default partner)."""

from __future__ import annotations

from typing import Any

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy
from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    PlayHand,
    SelectBlind,
)
from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run


def _gs_with_hand(cards, discards_left=3) -> dict[str, Any]:
    return {
        "hand": cards,
        "jokers": [],
        "current_round": {"discards_left": discards_left, "hands_left": 4},
    }


def _cards(*specs):
    return [create_playing_card(s, r) for s, r in specs]


class TestDecisionRule:
    def test_plays_detected_flush(self):
        hand = _cards(
            (Suit.HEARTS, Rank.ACE),
            (Suit.HEARTS, Rank.KING),
            (Suit.HEARTS, Rank.NINE),
            (Suit.HEARTS, Rank.SIX),
            (Suit.HEARTS, Rank.TWO),
            (Suit.SPADES, Rank.THREE),
            (Suit.CLUBS, Rank.FOUR),
            (Suit.DIAMONDS, Rank.JACK),
        )
        action = GreedyHandPolicy()(_gs_with_hand(hand))
        assert isinstance(action, PlayHand)
        assert action.card_indices == (0, 1, 2, 3, 4)

    def test_discards_chaff_on_weak_hand(self):
        # Best hand is a pair -> spend a discard on the lowest chaff.
        # (Chaff ranks deliberately avoid any 5-window straight, including
        # the ace-low wheel the subset search would otherwise find.)
        hand = _cards(
            (Suit.HEARTS, Rank.ACE),
            (Suit.SPADES, Rank.ACE),
            (Suit.CLUBS, Rank.TEN),
            (Suit.DIAMONDS, Rank.NINE),
            (Suit.HEARTS, Rank.SEVEN),
            (Suit.SPADES, Rank.FIVE),
            (Suit.CLUBS, Rank.FOUR),
            (Suit.DIAMONDS, Rank.TWO),
        )
        action = GreedyHandPolicy()(_gs_with_hand(hand, discards_left=2))
        assert isinstance(action, Discard)
        assert len(action.card_indices) == 5  # capped at the engine limit
        assert 0 not in action.card_indices and 1 not in action.card_indices

    def test_plays_weak_hand_when_no_discards(self):
        hand = _cards(
            (Suit.HEARTS, Rank.ACE),
            (Suit.SPADES, Rank.ACE),
            (Suit.CLUBS, Rank.NINE),
            (Suit.DIAMONDS, Rank.SEVEN),
            (Suit.HEARTS, Rank.FIVE),
        )
        action = GreedyHandPolicy()(_gs_with_hand(hand, discards_left=0))
        assert isinstance(action, PlayHand)
        assert set(action.card_indices) == {0, 1}  # the pair

    def test_deterministic(self):
        hand = _cards(
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.QUEEN),
            (Suit.CLUBS, Rank.NINE),
            (Suit.DIAMONDS, Rank.SEVEN),
            (Suit.HEARTS, Rank.FIVE),
            (Suit.SPADES, Rank.FOUR),
            (Suit.CLUBS, Rank.THREE),
            (Suit.DIAMONDS, Rank.TWO),
        )
        gs = _gs_with_hand(hand)
        first = GreedyHandPolicy()(gs)
        for _ in range(3):
            again = GreedyHandPolicy()(gs)
            assert type(again) is type(first)
            assert again.card_indices == first.card_indices


class TestDrivesRealGame:
    def test_clears_ante_one_small_blind(self):
        """The policy must be able to drive real hand phases end-to-end."""
        gs = initialize_run("b_red", 1, "GREEDY_SMOKE")
        step(gs, SelectBlind())

        policy = GreedyHandPolicy()
        for _ in range(16):  # hands + discards budget, generous
            if gs["phase"] != GamePhase.SELECTING_HAND:
                break
            step(gs, policy(gs))
        else:
            raise AssertionError("policy loop did not terminate")

        # Small blind at ante 1 is trivially clearable by any sane policy
        assert gs["phase"] == GamePhase.ROUND_EVAL
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
