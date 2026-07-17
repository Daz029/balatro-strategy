"""Tests for the solver's 5-card discard cap.

The engine (game.py::_handle_discard) and real Balatro cap one discard at 5
selected cards, but template construction can produce 6+ non-matches (e.g.
2 flush cards held in an 8-card hand). Before the cap, the solver labeled
unexecutable 6-8-card discards (8.3% of the first stage-1 dataset) and its
reachability math assumed 6-8 replacement draws where the game allows 5.
Found when BC training first consumed real generated data.
"""

from __future__ import annotations

import pytest
from hand_solver import (
    DISCARD_LIMIT,
    DeckComposition,
    cap_discard,
    rank_templates_cheaply,
    solve_discard_decision,
    solve_hand_for_ante_clear,
)

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _card(suit: Suit, rank: Rank, enhancement: str | None = None) -> Card:
    c = create_playing_card(suit, rank)
    if enhancement:
        c.center_key = enhancement
    return c


def _two_suited_hand() -> list[Card]:
    """8 cards, only 2 hearts -- a flush template discards 6 without the cap."""
    return [
        _card(Suit.HEARTS, Rank.ACE),
        _card(Suit.HEARTS, Rank.KING),
        _card(Suit.SPADES, Rank.TWO),
        _card(Suit.CLUBS, Rank.FOUR),
        _card(Suit.DIAMONDS, Rank.SIX),
        _card(Suit.SPADES, Rank.EIGHT),
        _card(Suit.CLUBS, Rank.NINE),
        _card(Suit.DIAMONDS, Rank.JACK),
    ]


def _full_deck_minus(hand: list[Card]) -> DeckComposition:
    all_cards = [create_playing_card(s, r) for s in Suit for r in Rank]
    held = {(c.base.id, c.base.suit.value) for c in hand}
    remaining = [c for c in all_cards if (c.base.id, c.base.suit.value) not in held]
    return DeckComposition.from_deck(remaining)


class TestCapDiscard:
    def test_no_op_at_or_below_limit(self):
        cards = _two_suited_hand()[:5]
        to_discard, kept = cap_discard(cards)
        assert to_discard == cards
        assert kept == []

    def test_caps_and_partitions(self):
        cards = _two_suited_hand()[:7]
        to_discard, kept = cap_discard(cards)
        assert len(to_discard) == DISCARD_LIMIT
        assert len(kept) == 2
        assert {id(c) for c in to_discard} | {id(c) for c in kept} == {id(c) for c in cards}

    def test_keeps_highest_value_and_enhanced_cards(self):
        low_plain = _card(Suit.SPADES, Rank.TWO)
        enhanced = _card(Suit.CLUBS, Rank.THREE, enhancement="m_steel")
        others = [
            _card(Suit.DIAMONDS, Rank.FOUR),
            _card(Suit.SPADES, Rank.FIVE),
            _card(Suit.CLUBS, Rank.SIX),
            _card(Suit.DIAMONDS, Rank.SEVEN),
            _card(Suit.SPADES, Rank.ACE),
        ]
        to_discard, kept = cap_discard([low_plain, enhanced, *others])
        kept_ids = {id(c) for c in kept}
        assert id(enhanced) in kept_ids  # enhancement beats rank
        assert id(low_plain) not in kept_ids  # lowest plain card goes first
        assert id(others[-1]) in kept_ids  # highest nominal plain card kept


class TestRankTemplatesRespectCap:
    def test_all_candidates_within_limit(self):
        hand = _two_suited_hand()
        deck = _full_deck_minus(hand)
        candidates = rank_templates_cheaply(
            hand, deck, HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom("cap1")
        )
        assert candidates
        for _template, hold, kept, discard, _p, _v, _needed in candidates:
            assert len(discard) <= DISCARD_LIMIT
            # hold + kept + discard partitions the full hand
            assert len(hold) + len(kept) + len(discard) == len(hand)

    def test_hold_stays_matches_only(self):
        # The flush-template hold must contain only hearts even when the
        # cap forces non-hearts to be kept in hand (they go in `kept`,
        # never `hold` -- still_needed/eval math depends on it).
        hand = _two_suited_hand()
        deck = _full_deck_minus(hand)
        candidates = rank_templates_cheaply(
            hand, deck, HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom("cap2")
        )
        flush = [c for c in candidates if c[0].name == "flush_Hearts"]
        if not flush:
            pytest.skip("flush template not in top-k for this hand")
        _t, hold, kept, discard, _p, _v, still_needed = flush[0]
        assert all(c.base.suit == Suit.HEARTS for c in hold)
        assert len(hold) == 2
        assert still_needed == 3
        assert len(discard) == DISCARD_LIMIT
        assert len(kept) == 1


class TestSolversRespectCap:
    def test_solve_discard_decision_capped(self):
        hand = _two_suited_hand()
        deck = _full_deck_minus(hand)
        choice = solve_discard_decision(
            hand,
            [],
            HandLevels(),
            Blind.create("bl_small", ante=1),
            PseudoRandom("cap3"),
            deck,
            discards_left=3,
        )
        if choice.action == "discard":
            assert 1 <= len(choice.discard) <= DISCARD_LIMIT
            assert len(choice.hold) + len(choice.discard) == len(hand)

    @pytest.mark.slow
    def test_real_regression_seed_labels_executable(self):
        """stage1_no_jokers_00000000 produced a 6-card discard label in the
        first generated dataset; the full pipeline must now emit <= 5."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from generate_hand_demos import generate_one_example, stage_presets

        preset = stage_presets()["stage1_no_jokers"]
        example = generate_one_example("stage1_no_jokers_00000000", preset.config)
        assert int((example.card_indices >= 0).sum()) <= DISCARD_LIMIT


def _shortcut_full_hand_straight() -> list[Card]:
    """8 distinct-rank cards a Shortcut-widened straight window swallows
    WHOLE, so the straight template's complement (the discard) is EMPTY.

    Reproduces the board behind stage2 relabel seed 00002997
    (14S 12C 10C 8H 7H 6S 4S 3S). Under Shortcut the gap-tolerant window
    matches every rank present, `construct_hold` holds all 8, and the
    derived discard is []. Without the lower-bound clamp the solver emits a
    "discard nothing" action -- the mirror of the 6-8 card discard this
    module's cap fixes.
    """
    return [
        _card(Suit.SPADES, Rank.ACE),
        _card(Suit.CLUBS, Rank.QUEEN),
        _card(Suit.CLUBS, Rank.TEN),
        _card(Suit.HEARTS, Rank.EIGHT),
        _card(Suit.HEARTS, Rank.SEVEN),
        _card(Suit.SPADES, Rank.SIX),
        _card(Suit.SPADES, Rank.FOUR),
        _card(Suit.SPADES, Rank.THREE),
    ]


class TestDiscardLowerBound:
    """A legal discard is 1-5 cards. `cap_discard` clamps the upper bound;
    these pin the lower one (>=1), the counterpart bug fixed at K3."""

    def test_no_empty_discard_candidate_under_shortcut(self):
        hand = _shortcut_full_hand_straight()
        deck = _full_deck_minus(hand)
        candidates = rank_templates_cheaply(
            hand,
            deck,
            HandLevels(),
            Blind.create("bl_small", ante=1),
            PseudoRandom("lb1"),
            shortcut=True,
            top_k=8,
            jokers=[create_joker("j_shortcut")],
        )
        # discard is field 3 of each candidate tuple
        for cand in candidates:
            assert len(cand[3]) >= 1, "a discard candidate must discard >=1 card"

    def test_solve_discard_decision_never_empty_under_shortcut(self):
        hand = _shortcut_full_hand_straight()
        deck = _full_deck_minus(hand)
        choice = solve_discard_decision(
            hand,
            [create_joker("j_shortcut")],
            HandLevels(),
            Blind.create("bl_small", ante=1),
            PseudoRandom("lb2"),
            deck,
            discards_left=3,
            shortcut=True,
        )
        if choice.action == "discard":
            assert 1 <= len(choice.discard) <= DISCARD_LIMIT

    @pytest.mark.slow
    def test_ante_clear_label_is_executable_under_shortcut(self):
        """End to end: a made-shortcut-straight full hand must never label a
        zero-card discard. The completed hand scores high under Shortcut
        (post the score_hand jokers fix), which is exactly what floated the
        empty phantom to the top of the shortlist and won it the strict-`>`
        tie -- so this only bites with detection working."""
        hand = _shortcut_full_hand_straight()
        deck = _full_deck_minus(hand)
        blind = Blind.create("bl_small", ante=1)
        choice = solve_hand_for_ante_clear(
            hand,
            [create_joker("j_shortcut")],
            HandLevels(),
            blind,
            PseudoRandom("lb3"),
            deck,
            chips_needed=getattr(blind, "chips", 300),
            hands_left=1,
            discards_left=3,
            shortcut=True,
            mc_seed="lb3",
        )
        if choice.action == "discard":
            assert 1 <= len(choice.discard) <= DISCARD_LIMIT
