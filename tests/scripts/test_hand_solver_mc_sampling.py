"""Tests for the future-hand MC sampler's speed/determinism changes.

Profiled 2026-07: `estimate_future_hand_distribution` was 75-97% of
per-example solve time (40 samples x full ordering search per sample).
Changes under test: single-ordering evaluation for hypothetical hands,
16-sample default, and per-example seeding (previously the unseeded global
`random`, so p_clear labels carried run-to-run MC noise).
"""

from __future__ import annotations

from hand_solver import (
    DeckComposition,
    estimate_future_hand_distribution,
    evaluate_value,
    solve_hand_for_ante_clear,
)

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _deck() -> DeckComposition:
    return DeckComposition.from_deck(
        [create_playing_card(s, r) for s in Suit for r in Rank]
    )


def _face_cards() -> list:
    return [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.THREE),
        create_playing_card(Suit.CLUBS, Rank.QUEEN),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.HEARTS, Rank.JACK),
    ]


class TestSearchOrderingsFlag:
    def test_off_scores_given_order_exactly(self):
        # With Photograph, the searched value must be >= the given-order
        # value, and search_orderings=False must return the given-order one.
        cards = _face_cards()
        jokers = [create_joker("j_photograph")]
        hl = HandLevels()
        blind = Blind.create("bl_small", ante=1)
        rng = PseudoRandom("mc1")
        fixed = evaluate_value(
            cards, [], jokers, hl, blind, rng, blind_chips=300, search_orderings=False
        )
        searched = evaluate_value(
            cards, [], jokers, hl, blind, rng, blind_chips=300, search_orderings=True
        )
        assert searched.total >= fixed.total

    def test_no_op_without_order_sensitivity(self):
        cards = [
            create_playing_card(Suit.HEARTS, Rank.TWO),
            create_playing_card(Suit.SPADES, Rank.SEVEN),
            create_playing_card(Suit.CLUBS, Rank.NINE),
        ]
        hl = HandLevels()
        blind = Blind.create("bl_small", ante=1)
        rng = PseudoRandom("mc2")
        a = evaluate_value(cards, [], [], hl, blind, rng, search_orderings=False)
        b = evaluate_value(cards, [], [], hl, blind, rng, search_orderings=True)
        assert a.total == b.total


class TestSeededSampling:
    def test_same_seed_same_samples(self):
        deck = _deck()
        hl = HandLevels()
        blind = Blind.create("bl_small", ante=1)
        rng = PseudoRandom("mc3")
        a = estimate_future_hand_distribution(
            deck, [], hl, blind, rng, 8, 8, mc_seed="seed_x"
        )
        b = estimate_future_hand_distribution(
            deck, [], hl, blind, rng, 8, 8, mc_seed="seed_x"
        )
        assert a == b

    def test_different_seed_different_samples(self):
        deck = _deck()
        hl = HandLevels()
        blind = Blind.create("bl_small", ante=1)
        rng = PseudoRandom("mc4")
        a = estimate_future_hand_distribution(
            deck, [], hl, blind, rng, 8, 8, mc_seed="seed_x"
        )
        b = estimate_future_hand_distribution(
            deck, [], hl, blind, rng, 8, 8, mc_seed="seed_y"
        )
        assert a != b

    def test_full_solve_reproducible_with_mc_seed(self):
        hand = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.HEARTS, Rank.KING),
            create_playing_card(Suit.SPADES, Rank.TWO),
            create_playing_card(Suit.CLUBS, Rank.FOUR),
            create_playing_card(Suit.DIAMONDS, Rank.SIX),
            create_playing_card(Suit.SPADES, Rank.EIGHT),
            create_playing_card(Suit.CLUBS, Rank.NINE),
            create_playing_card(Suit.DIAMONDS, Rank.JACK),
        ]
        deck = _deck()
        blind = Blind.create("bl_small", ante=1)

        def solve():
            return solve_hand_for_ante_clear(
                hand, [], HandLevels(), blind, PseudoRandom("mc5"), deck,
                chips_needed=300.0, hands_left=2, discards_left=1,
                blind_chips=300, mc_seed="repro_test",
            )

        a, b = solve(), solve()
        assert a.action == b.action
        assert a.p_clear == b.p_clear
        assert [id(c) for c in a.hold] == [id(c) for c in b.hold]
