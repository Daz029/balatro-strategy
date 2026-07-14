"""Tests for the big-hand play prescreen (B5).

`best_immediate_play` is C(n,5) exact evaluations per call -- fine at n=8
(56), prohibitive at the width-40 obs cap. Above PRESCREEN_HAND_LIMIT it
evaluates only the top-k template-derived candidates from
`prescreen_play_candidates`, selected family-diverse (handoff pitfall #13:
naive top-k returns k variants of one dominant line). These tests pin:

  - the n<=8 path is byte-identical brute force (prescreen never consulted);
  - the prescreen finds the same best play as full brute force on
    constructed big hands where the answer is unambiguous;
  - family diversity (a dominant flush cannot crowd out a rank line);
  - prefix stability in top_k (the validation harness scores k-cuts from
    one call and slices prefixes);
  - repeated calls do not corrupt shared state (the stage-4 shared-blind
    bug class).
"""

from __future__ import annotations

import itertools

import hand_solver
from hand_solver import (
    PRESCREEN_HAND_LIMIT,
    best_immediate_play,
    evaluate_value,
    prescreen_play_candidates,
)

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _card(suit: Suit, rank: Rank, enhancement: str | None = None) -> Card:
    c = create_playing_card(suit, rank)
    if enhancement:
        c.center_key = enhancement
    return c


def _fixtures() -> tuple[HandLevels, Blind, PseudoRandom]:
    return HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom("prescreen")


def _brute_force_best(hand: list[Card], jokers: list[Card]) -> float:
    """The pre-prescreen behavior: exact evaluation of every subset."""
    hand_levels, blind, rng = _fixtures()
    best = None
    for size in range(1, min(5, len(hand)) + 1):
        for combo in itertools.combinations(hand, size):
            combo_ids = {id(c) for c in combo}
            held = [c for c in hand if id(c) not in combo_ids]
            result = evaluate_value(list(combo), held, jokers, hand_levels, blind, rng)
            if best is None or result.total > best.total:
                best = result
    assert best is not None
    return best.total


def _flush_plus_junk_hand() -> list[Card]:
    """10 cards: a complete heart flush plus low offsuit junk."""
    return [
        _card(Suit.HEARTS, Rank.ACE),
        _card(Suit.HEARTS, Rank.KING),
        _card(Suit.HEARTS, Rank.NINE),
        _card(Suit.HEARTS, Rank.SEVEN),
        _card(Suit.HEARTS, Rank.FOUR),
        _card(Suit.SPADES, Rank.TWO),
        _card(Suit.CLUBS, Rank.THREE),
        _card(Suit.DIAMONDS, Rank.FIVE),
        _card(Suit.SPADES, Rank.SIX),
        _card(Suit.CLUBS, Rank.EIGHT),
    ]


def _quads_and_flush_hand() -> list[Card]:
    """11 cards: four Kings AND five non-King hearts -- two strong families."""
    return [
        _card(Suit.HEARTS, Rank.KING),
        _card(Suit.SPADES, Rank.KING),
        _card(Suit.CLUBS, Rank.KING),
        _card(Suit.DIAMONDS, Rank.KING),
        _card(Suit.HEARTS, Rank.QUEEN),
        _card(Suit.HEARTS, Rank.TEN),
        _card(Suit.HEARTS, Rank.EIGHT),
        _card(Suit.HEARTS, Rank.SIX),
        _card(Suit.HEARTS, Rank.THREE),
        _card(Suit.SPADES, Rank.FOUR),
        _card(Suit.CLUBS, Rank.SEVEN),
    ]


class TestPathSelection:
    def test_at_limit_prescreen_never_consulted(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("prescreen must not run at n <= PRESCREEN_HAND_LIMIT")

        monkeypatch.setattr(hand_solver, "prescreen_play_candidates", boom)
        hand = _flush_plus_junk_hand()[:PRESCREEN_HAND_LIMIT]
        hand_levels, blind, rng = _fixtures()
        subset, result = best_immediate_play(hand, [], hand_levels, blind, rng)
        assert 1 <= len(subset) <= 5
        assert result.total == _brute_force_best(hand, [])

    def test_above_limit_prescreen_used(self, monkeypatch):
        calls: list[int] = []
        real = hand_solver.prescreen_play_candidates

        def spy(hand, *args, **kwargs):
            calls.append(len(hand))
            return real(hand, *args, **kwargs)

        monkeypatch.setattr(hand_solver, "prescreen_play_candidates", spy)
        hand = _flush_plus_junk_hand()
        hand_levels, blind, rng = _fixtures()
        best_immediate_play(hand, [], hand_levels, blind, rng)
        assert calls == [len(hand)]


class TestPrescreenQuality:
    def test_complete_flush_matches_brute_force(self):
        hand = _flush_plus_junk_hand()
        hand_levels, blind, rng = _fixtures()
        subset, result = best_immediate_play(hand, [], hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, [])
        assert {c.base.suit for c in subset} == {Suit.HEARTS.value}

    def test_quads_hand_matches_brute_force(self):
        hand = _quads_and_flush_hand()
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, [], hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, [])

    def test_candidates_capped_and_nonempty(self):
        hand = _quads_and_flush_hand()
        hand_levels, blind, rng = _fixtures()
        for k in (1, 3, 8):
            candidates = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=k)
            assert 1 <= len(candidates) <= k
            for cards in candidates:
                assert 1 <= len(cards) <= 5


class TestFamilyDiversity:
    def test_flush_cannot_crowd_out_rank_line(self):
        hand = _quads_and_flush_hand()
        hand_levels, blind, rng = _fixtures()
        candidates = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=6)
        king_sets = [
            cards
            for cards in candidates
            if sum(1 for c in cards if c.base.id == 13) == 4
        ]
        heart_flushes = [
            cards
            for cards in candidates
            if len(cards) == 5 and all(c.base.suit == Suit.HEARTS.value for c in cards)
        ]
        assert king_sets, "the quad-Kings line must survive family-diverse selection"
        assert heart_flushes, "the heart flush line must survive family-diverse selection"

    def test_prefix_stable_in_top_k(self):
        hand = _quads_and_flush_hand()
        hand_levels, blind, rng = _fixtures()
        full = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=8)
        for j in (1, 3, 5):
            prefix = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=j)
            assert [[id(c) for c in cards] for cards in prefix] == [
                [id(c) for c in cards] for cards in full[:j]
            ]


class TestJokerAwareRanking:
    def _two_flush_hand(self) -> list:
        """11 cards: a strong spade flush and a weak diamond flush.
        Jokerless, spades win on chips; Greedy Joker (+3 mult per scored
        diamond) makes the diamond flush the true best play."""
        return [
            _card(Suit.SPADES, Rank.ACE),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.SPADES, Rank.QUEEN),
            _card(Suit.SPADES, Rank.JACK),
            _card(Suit.SPADES, Rank.NINE),
            _card(Suit.DIAMONDS, Rank.TEN),
            _card(Suit.DIAMONDS, Rank.EIGHT),
            _card(Suit.DIAMONDS, Rank.SIX),
            _card(Suit.DIAMONDS, Rank.FOUR),
            _card(Suit.DIAMONDS, Rank.TWO),
            _card(Suit.CLUBS, Rank.THREE),
        ]

    def test_ranking_sees_jokers(self):
        from jackdaw.engine.card_factory import create_joker

        hand = self._two_flush_hand()
        hand_levels, blind, rng = _fixtures()
        greedy = [create_joker("j_greedy_joker")]

        jokerless = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=1)
        with_greedy = prescreen_play_candidates(
            hand, greedy, hand_levels, blind, rng, top_k=1
        )
        assert {c.base.suit for c in jokerless[0]} == {Suit.SPADES.value}
        assert {c.base.suit for c in with_greedy[0]} == {Suit.DIAMONDS.value}

    def test_prescreened_play_matches_brute_force_with_joker(self):
        from jackdaw.engine.card_factory import create_joker

        hand = self._two_flush_hand()
        hand_levels, blind, rng = _fixtures()
        greedy = [create_joker("j_greedy_joker")]
        _, result = best_immediate_play(hand, greedy, hand_levels, blind, rng)

        best = None
        for size in range(1, 6):
            for combo in itertools.combinations(hand, size):
                combo_ids = {id(c) for c in combo}
                held = [c for c in hand if id(c) not in combo_ids]
                r = evaluate_value(list(combo), held, greedy, hand_levels, blind, rng)
                if best is None or r.total > best.total:
                    best = r
        assert result.total == best.total


class TestPairPin:
    def _weak_pair_flashy_board(self) -> list:
        """11 cards: a lone low pair drowned by a complete flush and a
        straight draw -- cheap ranking alone would drop the pair."""
        return [
            _card(Suit.SPADES, Rank.TWO),
            _card(Suit.CLUBS, Rank.TWO),
            _card(Suit.HEARTS, Rank.ACE),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.HEARTS, Rank.QUEEN),
            _card(Suit.HEARTS, Rank.JACK),
            _card(Suit.HEARTS, Rank.NINE),
            _card(Suit.DIAMONDS, Rank.EIGHT),
            _card(Suit.SPADES, Rank.SEVEN),
            _card(Suit.CLUBS, Rank.SIX),
            _card(Suit.DIAMONDS, Rank.FIVE),
        ]

    def test_pair_line_pinned_into_every_k2_cut(self):
        hand = self._weak_pair_flashy_board()
        hand_levels, blind, rng = _fixtures()
        for k in (2, 3, 5):
            candidates = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=k)
            pair_sets = [
                cards
                for cards in candidates[:2]
                if sum(1 for c in cards if c.base.id == 2) == 2
            ]
            assert pair_sets, f"pair of 2s must be pinned at index <= 1 (k={k})"

    def test_no_pin_without_a_rank_line(self):
        # No two cards share a rank -- nothing to pin, no crash.
        hand = [
            _card(s, r)
            for s, r in zip(
                [Suit.HEARTS, Suit.SPADES, Suit.CLUBS, Suit.DIAMONDS] * 3,
                [
                    Rank.TWO, Rank.FOUR, Rank.SIX, Rank.EIGHT, Rank.TEN,
                    Rank.QUEEN, Rank.ACE, Rank.THREE, Rank.FIVE,
                ],
            )
        ]
        hand_levels, blind, rng = _fixtures()
        candidates = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=4)
        assert candidates


class TestStateSafety:
    def test_repeated_calls_identical_on_history_boss(self):
        # The Eye mutates per-hand-type history on every score call; the
        # prescreen's cheap scoring must clone (stage-4 shared-blind class).
        hand = _quads_and_flush_hand()
        hand_levels = HandLevels()
        blind = Blind.create("bl_eye", ante=2)
        rng = PseudoRandom("eye-prescreen")
        first = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=8)
        second = prescreen_play_candidates(hand, [], hand_levels, blind, rng, top_k=8)
        assert [[id(c) for c in cards] for cards in first] == [
            [id(c) for c in cards] for cards in second
        ]
