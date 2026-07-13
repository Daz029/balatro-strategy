"""Tests for the hand-agent-only flush/straight potential features
(``observation.hand_potential_features`` / ``encode_hand_potential``),
the h1 schema bump's B2 slice 1.

The features must MIRROR the engine's detection semantics (Four Fingers
lowers both flush and straight thresholds to 4; Shortcut tolerates one
missing rank between present ranks; Stone cards join neither; ace is high
or low, no wrap) — the randomized cross-check at the bottom pins the
window model against ``get_flush``/``get_straight`` directly.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_eval import get_flush, get_straight
from jackdaw.env.observation import (
    D_GLOBAL,
    D_HAND_CARD,
    D_HAND_CARD_POTENTIAL,
    D_HAND_GLOBAL,
    D_HAND_POTENTIAL,
    D_PLAYING_CARD,
    _straight_windows,
    encode_hand_potential,
    hand_potential_features,
)

# GC extension layout indices (see hand_potential_features docstring)
_GC_SUIT = slice(0, 4)
_GC_RANK = slice(4, 17)
_GC_MAX_SUIT = 17
_GC_BEST_WINDOW = 18
_GC_FOUR_FINGERS = 19
_GC_SHORTCUT = 20

# _SUIT_IDX order in observation.py
_SUIT_POS = {Suit.HEARTS: 0, Suit.DIAMONDS: 1, Suit.CLUBS: 2, Suit.SPADES: 3}


def _cards(*specs: tuple[Suit, Rank]) -> list:
    return [create_playing_card(s, r) for s, r in specs]


class TestDimensions:
    def test_constants(self):
        assert D_HAND_CARD_POTENTIAL == 3
        assert D_HAND_CARD == D_PLAYING_CARD + 3 == 18
        assert D_HAND_POTENTIAL == 21
        assert D_HAND_GLOBAL == D_GLOBAL + 21 == 256

    def test_shapes(self):
        hand = _cards((Suit.HEARTS, Rank.TWO), (Suit.SPADES, Rank.KING))
        per_card, gc_ext = hand_potential_features(hand)
        assert per_card.shape == (2, D_HAND_CARD_POTENTIAL)
        assert gc_ext.shape == (D_HAND_POTENTIAL,)
        assert per_card.dtype == np.float32
        assert gc_ext.dtype == np.float32

    def test_empty_hand(self):
        per_card, gc_ext = hand_potential_features([], four_fingers=True, shortcut=True)
        assert per_card.shape == (0, D_HAND_CARD_POTENTIAL)
        # Flag bits describe the joker modifiers, not the hand
        assert gc_ext[_GC_FOUR_FINGERS] == 1.0
        assert gc_ext[_GC_SHORTCUT] == 1.0
        assert not gc_ext[:19].any()


class TestFlushFeatures:
    def test_four_flush_draw(self):
        hand = _cards(
            (Suit.HEARTS, Rank.TWO),
            (Suit.HEARTS, Rank.FIVE),
            (Suit.HEARTS, Rank.NINE),
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.THREE),
            (Suit.CLUBS, Rank.JACK),
            (Suit.DIAMONDS, Rank.SEVEN),
            (Suit.SPADES, Rank.QUEEN),
        )
        per_card, gc_ext = hand_potential_features(hand)
        # Each heart sees 4 hearts / 5 needed
        for i in range(4):
            assert per_card[i, 0] == pytest.approx(4 / 5)
        # The spades see 2/5, club and diamond 1/5
        assert per_card[4, 0] == pytest.approx(2 / 5)
        assert per_card[5, 0] == pytest.approx(1 / 5)
        assert per_card[6, 0] == pytest.approx(1 / 5)
        assert gc_ext[_GC_MAX_SUIT] == pytest.approx(4 / 5)
        # Per-suit composition: 4H 2S 1C 1D over 8 cards
        assert gc_ext[_GC_SUIT].tolist() == pytest.approx([4 / 8, 1 / 8, 1 / 8, 2 / 8])

    def test_complete_flush_saturates(self):
        hand = [create_playing_card(Suit.CLUBS, r) for r in (
            Rank.TWO, Rank.FOUR, Rank.NINE, Rank.JACK, Rank.KING, Rank.ACE,
        )]
        per_card, gc_ext = hand_potential_features(hand)
        # 6 clubs / 5 needed — deliberately uncapped (spare flush cards are real info)
        assert gc_ext[_GC_MAX_SUIT] == pytest.approx(6 / 5)
        assert per_card[0, 0] == pytest.approx(6 / 5)

    def test_four_fingers_lowers_flush_threshold(self):
        # Engine-confirmed: get_flush threshold is 4 under Four Fingers
        hand = _cards(
            (Suit.HEARTS, Rank.TWO),
            (Suit.HEARTS, Rank.FIVE),
            (Suit.HEARTS, Rank.NINE),
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.THREE),
        )
        per_card, gc_ext = hand_potential_features(hand, four_fingers=True)
        assert per_card[0, 0] == pytest.approx(1.0)  # 4/4 — playable flush
        assert gc_ext[_GC_MAX_SUIT] == pytest.approx(1.0)
        assert gc_ext[_GC_FOUR_FINGERS] == 1.0


class TestRankFeatures:
    def test_rank_counts(self):
        hand = _cards(
            (Suit.HEARTS, Rank.SEVEN),
            (Suit.SPADES, Rank.SEVEN),
            (Suit.CLUBS, Rank.SEVEN),
            (Suit.DIAMONDS, Rank.KING),
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.TWO),
        )
        per_card, gc_ext = hand_potential_features(hand)
        for i in range(3):
            assert per_card[i, 1] == pytest.approx(3 / 4)  # trips
        assert per_card[3, 1] == pytest.approx(2 / 4)  # pair
        assert per_card[5, 1] == pytest.approx(1 / 4)  # singleton
        # GC per-rank block, _RANK_IDX order: 2 at [0], 7 at [5], K at [11]
        rank_block = gc_ext[_GC_RANK]
        assert rank_block[5] == pytest.approx(3 / 4)
        assert rank_block[11] == pytest.approx(2 / 4)
        assert rank_block[0] == pytest.approx(1 / 4)
        assert rank_block.sum() == pytest.approx(6 / 4)


class TestStraightWindows:
    def test_plain_window_table(self):
        windows = _straight_windows(False, False)
        assert len(windows) == 10  # wheel (1-5) through 10-J-Q-K-A
        assert (1, 2, 3, 4, 5) in windows
        assert (10, 11, 12, 13, 14) in windows
        # No wrap-around window exists
        assert all(w == tuple(range(w[0], w[0] + 5)) for w in windows)

    def test_four_fingers_window_table(self):
        windows = _straight_windows(True, False)
        assert len(windows) == 11
        assert all(len(w) == 4 for w in windows)

    def test_shortcut_windows_allow_single_gaps(self):
        windows = _straight_windows(False, True)
        assert (2, 4, 6, 8, 10) in windows  # max-gap straight
        assert (1, 2, 4, 5, 6) in windows  # wheel with one gap
        assert all(
            all(b - a in (1, 2) for a, b in zip(w, w[1:])) for w in windows
        )
        # No two-rank gap
        assert (2, 5, 6, 7, 8) not in windows

    def test_open_ended_straight_draw(self):
        hand = _cards(
            (Suit.HEARTS, Rank.FIVE),
            (Suit.SPADES, Rank.SIX),
            (Suit.CLUBS, Rank.SEVEN),
            (Suit.DIAMONDS, Rank.EIGHT),
            (Suit.HEARTS, Rank.KING),
            (Suit.SPADES, Rank.KING),
        )
        per_card, gc_ext = hand_potential_features(hand)
        assert gc_ext[_GC_BEST_WINDOW] == pytest.approx(4 / 5)
        for i in range(4):
            assert per_card[i, 2] == pytest.approx(4 / 5)
        # Kings sit in windows (9..13)/(10..14) which hold only K: 1/5
        assert per_card[4, 2] == pytest.approx(1 / 5)

    def test_wheel_draw_ace_checks_low_windows(self):
        hand = _cards(
            (Suit.HEARTS, Rank.ACE),
            (Suit.SPADES, Rank.TWO),
            (Suit.CLUBS, Rank.THREE),
            (Suit.DIAMONDS, Rank.FOUR),
            (Suit.HEARTS, Rank.NINE),
        )
        per_card, gc_ext = hand_potential_features(hand)
        assert gc_ext[_GC_BEST_WINDOW] == pytest.approx(4 / 5)
        # The ace's own best window is the wheel (1,2,3,4,5) at 4/5 — its
        # high windows (10..14) hold only the ace itself (1/5)
        assert per_card[0, 2] == pytest.approx(4 / 5)

    def test_no_wraparound(self):
        hand = _cards(
            (Suit.HEARTS, Rank.QUEEN),
            (Suit.SPADES, Rank.KING),
            (Suit.CLUBS, Rank.ACE),
            (Suit.DIAMONDS, Rank.TWO),
            (Suit.HEARTS, Rank.THREE),
        )
        _, gc_ext = hand_potential_features(hand)
        # Best is Q-K-A (or A-2-3 via the wheel): 3/5, never 5/5
        assert gc_ext[_GC_BEST_WINDOW] == pytest.approx(3 / 5)

    def test_duplicate_ranks_do_not_inflate_occupancy(self):
        hand = _cards(
            (Suit.HEARTS, Rank.FIVE),
            (Suit.SPADES, Rank.FIVE),
            (Suit.CLUBS, Rank.SIX),
            (Suit.DIAMONDS, Rank.SEVEN),
        )
        _, gc_ext = hand_potential_features(hand)
        # A straight needs ONE card per rank: 5,6,7 = 3 distinct ranks
        assert gc_ext[_GC_BEST_WINDOW] == pytest.approx(3 / 5)

    def test_four_fingers_straight(self):
        hand = _cards(
            (Suit.HEARTS, Rank.FIVE),
            (Suit.SPADES, Rank.SIX),
            (Suit.CLUBS, Rank.SEVEN),
            (Suit.DIAMONDS, Rank.EIGHT),
            (Suit.HEARTS, Rank.KING),
        )
        per_card, gc_ext = hand_potential_features(hand, four_fingers=True)
        assert gc_ext[_GC_BEST_WINDOW] == pytest.approx(1.0)  # 4-card straight complete
        assert per_card[0, 2] == pytest.approx(1.0)

    def test_shortcut_gap_straight(self):
        hand = _cards(
            (Suit.HEARTS, Rank.TWO),
            (Suit.SPADES, Rank.FOUR),
            (Suit.CLUBS, Rank.SIX),
            (Suit.DIAMONDS, Rank.EIGHT),
            (Suit.HEARTS, Rank.TEN),
        )
        _, gc_plain = hand_potential_features(hand)
        _, gc_shortcut = hand_potential_features(hand, shortcut=True)
        assert gc_plain[_GC_BEST_WINDOW] < 1.0
        assert gc_shortcut[_GC_BEST_WINDOW] == pytest.approx(1.0)
        assert gc_shortcut[_GC_SHORTCUT] == 1.0


class TestExclusions:
    def test_stone_card_excluded(self):
        stone = create_playing_card(Suit.HEARTS, Rank.SIX, "m_stone")
        hand = _cards(
            (Suit.HEARTS, Rank.FIVE),
            (Suit.HEARTS, Rank.SEVEN),
            (Suit.HEARTS, Rank.EIGHT),
        ) + [stone]
        per_card, gc_ext = hand_potential_features(hand)
        # Stone joins neither flushes nor straights in the engine
        assert per_card[3].tolist() == [0.0, 0.0, 0.0]
        assert gc_ext[_GC_MAX_SUIT] == pytest.approx(3 / 5)
        # 5,7,8 → best window (4..8)/(5..9) holds 3 distinct ranks
        assert gc_ext[_GC_BEST_WINDOW] == pytest.approx(3 / 5)
        # Suit composition still normalizes by the FULL hand size (4)
        assert gc_ext[_GC_SUIT][0] == pytest.approx(3 / 4)

    def test_face_down_card_excluded(self):
        hidden = create_playing_card(Suit.HEARTS, Rank.SIX)
        hidden.facing = "back"
        hand = _cards((Suit.HEARTS, Rank.FIVE), (Suit.HEARTS, Rank.SEVEN)) + [hidden]
        per_card, gc_ext = hand_potential_features(hand)
        assert per_card[2].tolist() == [0.0, 0.0, 0.0]
        assert gc_ext[_GC_MAX_SUIT] == pytest.approx(2 / 5)


class TestEncodeHandPotential:
    def test_flags_from_live_jokers(self):
        hand = _cards(
            (Suit.HEARTS, Rank.TWO),
            (Suit.HEARTS, Rank.FIVE),
            (Suit.HEARTS, Rank.NINE),
            (Suit.HEARTS, Rank.KING),
        )
        gs = {"hand": hand, "jokers": [create_joker("j_four_fingers")]}
        per_card, gc_ext = encode_hand_potential(gs)
        assert gc_ext[_GC_FOUR_FINGERS] == 1.0
        assert gc_ext[_GC_MAX_SUIT] == pytest.approx(1.0)
        assert per_card.shape == (4, D_HAND_CARD_POTENTIAL)

    def test_debuffed_modifier_joker_ignored(self):
        # find_joker semantics: a debuffed Four Fingers does NOT enable
        # 4-card flushes — the features must match
        ff = create_joker("j_four_fingers")
        ff.debuff = True
        gs = {"hand": _cards((Suit.HEARTS, Rank.TWO)), "jokers": [ff]}
        _, gc_ext = encode_hand_potential(gs)
        assert gc_ext[_GC_FOUR_FINGERS] == 0.0

    def test_missing_keys(self):
        per_card, gc_ext = encode_hand_potential({})
        assert per_card.shape == (0, D_HAND_CARD_POTENTIAL)
        assert not gc_ext.any()


class TestEngineMirror:
    """The window model must agree with the engine's own detectors on
    5-card hands (the only size where the engine detectors fire):
    best-window occupancy hits 1.0 exactly when ``get_straight`` detects,
    and max-suit-count crosses the threshold exactly when ``get_flush``
    detects (plain suits — no Wild/Smeared in these fixtures)."""

    _FLAG_COMBOS = [(False, False), (True, False), (False, True), (True, True)]

    def _random_hands(self) -> list[list]:
        rng = random.Random(1234)
        deck = [create_playing_card(s, r) for s in Suit for r in Rank]
        hands = [rng.sample(deck, 5) for _ in range(300)]
        # Random 5-card draws rarely make straights — add constructed
        # positives so both directions of the equivalence are exercised
        hands += [
            _cards(  # plain straight
                (Suit.HEARTS, Rank.FIVE), (Suit.SPADES, Rank.SIX),
                (Suit.CLUBS, Rank.SEVEN), (Suit.DIAMONDS, Rank.EIGHT),
                (Suit.HEARTS, Rank.NINE),
            ),
            _cards(  # wheel
                (Suit.HEARTS, Rank.ACE), (Suit.SPADES, Rank.TWO),
                (Suit.CLUBS, Rank.THREE), (Suit.DIAMONDS, Rank.FOUR),
                (Suit.HEARTS, Rank.FIVE),
            ),
            _cards(  # broadway
                (Suit.HEARTS, Rank.TEN), (Suit.SPADES, Rank.JACK),
                (Suit.CLUBS, Rank.QUEEN), (Suit.DIAMONDS, Rank.KING),
                (Suit.HEARTS, Rank.ACE),
            ),
            _cards(  # shortcut-only gap straight
                (Suit.HEARTS, Rank.TWO), (Suit.SPADES, Rank.FOUR),
                (Suit.CLUBS, Rank.SIX), (Suit.DIAMONDS, Rank.EIGHT),
                (Suit.HEARTS, Rank.TEN),
            ),
            _cards(  # four-fingers-only 4-run
                (Suit.HEARTS, Rank.FIVE), (Suit.SPADES, Rank.SIX),
                (Suit.CLUBS, Rank.SEVEN), (Suit.DIAMONDS, Rank.EIGHT),
                (Suit.HEARTS, Rank.KING),
            ),
            _cards(  # ace-low shortcut: A,2,4,5 (+ junk)
                (Suit.HEARTS, Rank.ACE), (Suit.SPADES, Rank.TWO),
                (Suit.CLUBS, Rank.FOUR), (Suit.DIAMONDS, Rank.FIVE),
                (Suit.HEARTS, Rank.NINE),
            ),
            [create_playing_card(Suit.CLUBS, r) for r in (  # flush, no straight
                Rank.TWO, Rank.FIVE, Rank.NINE, Rank.JACK, Rank.KING,
            )],
        ]
        return hands

    def test_straight_equivalence(self):
        for hand in self._random_hands():
            for ff, sc in self._FLAG_COMBOS:
                _, gc_ext = hand_potential_features(hand, four_fingers=ff, shortcut=sc)
                engine_says = bool(get_straight(hand, four_fingers=ff, shortcut=sc))
                features_say = gc_ext[_GC_BEST_WINDOW] == pytest.approx(1.0)
                assert engine_says == features_say, (
                    f"straight mismatch (ff={ff}, sc={sc}): engine={engine_says}, "
                    f"occupancy={gc_ext[_GC_BEST_WINDOW]}, "
                    f"hand={[(c.base.suit.value, c.base.rank.value) for c in hand]}"
                )

    def test_flush_equivalence(self):
        for hand in self._random_hands():
            for ff in (False, True):
                _, gc_ext = hand_potential_features(hand, four_fingers=ff)
                engine_says = bool(get_flush(hand, four_fingers=ff))
                features_say = gc_ext[_GC_MAX_SUIT] >= 1.0 - 1e-6
                assert engine_says == features_say, (
                    f"flush mismatch (ff={ff}): engine={engine_says}, "
                    f"max_suit={gc_ext[_GC_MAX_SUIT]}, "
                    f"hand={[(c.base.suit.value, c.base.rank.value) for c in hand]}"
                )
