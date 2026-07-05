"""Tests for jackdaw.engine.play_ordering's env-facing entry point.

The detection/covering-set internals are already exercised through
``scripts/hand_solver.py``'s re-imports (test_hand_solver_order_sensitivity,
test_hand_solver_permutation_coverage); this file covers what the RL env
actually calls: ``best_play_order``.
"""

from __future__ import annotations

from itertools import permutations

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.play_ordering import (
    best_play_order,
    candidate_orderings,
    fast_clone_card,
    fast_clone_hand_levels,
    fast_clone_rng,
)
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


def _plain_cards() -> list:
    return [
        create_playing_card(Suit.HEARTS, Rank.TWO),
        create_playing_card(Suit.SPADES, Rank.THREE),
        create_playing_card(Suit.CLUBS, Rank.FOUR),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.HEARTS, Rank.SIX),
    ]


def _face_heavy_cards() -> list:
    return [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.THREE),
        create_playing_card(Suit.CLUBS, Rank.QUEEN),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.HEARTS, Rank.JACK),
    ]


def _score_order(order, jokers, hand_levels, blind, rng) -> float:
    """Score one ordering against cloned state (mirrors env submission)."""
    result = score_hand(
        [fast_clone_card(c) for c in order],
        [],
        [fast_clone_card(j) for j in jokers],
        fast_clone_hand_levels(hand_levels),
        blind,
        fast_clone_rng(rng),
        game_state={},
        blind_chips=300,
    )
    return result.total


class TestBestPlayOrder:
    def test_no_order_sensitivity_returns_input_order(self):
        cards = _plain_cards()
        order = best_play_order(
            cards, [], [], HandLevels(), _small_blind(), PseudoRandom("t1")
        )
        assert order == tuple(cards)

    def test_photograph_finds_brute_force_optimum(self):
        # Photograph doubles mult on the first scored face card -- the
        # brute-force best over all 120 orderings must be matched.
        cards = _face_heavy_cards()
        jokers = [create_joker("j_photograph")]
        hl = HandLevels()
        blind = _small_blind()
        rng = PseudoRandom("t2")

        chosen = best_play_order(cards, [], jokers, hl, blind, rng, blind_chips=300)
        chosen_total = _score_order(chosen, jokers, hl, blind, rng)
        brute_best = max(
            _score_order(order, jokers, hl, blind, rng)
            for order in permutations(cards)
        )
        assert chosen_total == brute_best

    def test_returns_same_card_objects(self):
        # The env maps the returned order back to hand indices by identity;
        # best_play_order must return the ORIGINAL card objects, not clones.
        cards = _face_heavy_cards()
        jokers = [create_joker("j_photograph")]
        order = best_play_order(
            cards, [], jokers, HandLevels(), _small_blind(), PseudoRandom("t3")
        )
        assert sorted(map(id, order)) == sorted(map(id, cards))

    def test_live_state_never_mutated(self):
        cards = _face_heavy_cards()
        jokers = [create_joker("j_photograph")]
        hl = HandLevels()
        rng = PseudoRandom("t4")
        rng_state_before = dict(rng._state)
        abilities_before = [dict(c.ability) for c in cards]

        best_play_order(cards, [], jokers, hl, _small_blind(), rng, blind_chips=300)

        assert rng._state == rng_state_before
        assert [dict(c.ability) for c in cards] == abilities_before
        assert all(hs.played == 0 for hs in hl._hands.values())

    def test_single_card_fast_path(self):
        cards = [create_playing_card(Suit.HEARTS, Rank.ACE)]
        jokers = [create_joker("j_photograph")]
        assert candidate_orderings(cards, jokers) == [tuple(cards)]
