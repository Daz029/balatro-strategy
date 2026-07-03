"""Coverage tests for `evaluate_value`'s permutation-search capping logic.

`MAX_PERMUTATIONS = 24` used to slice `itertools.permutations(...)[:24]` for
a 5-card hand. Since `itertools.permutations` groups its output by first
element and `4! == 24`, that slice was every permutation sharing the
*original* first card and nothing else -- zero exploration of alternative
first-scored cards, which is exactly the property Photograph/Hanging Chad
care about. `_first_last_covering_permutations` replaces it with a
deterministic set that covers every (first, last) ordered pair exactly once
in `n * (n - 1)` permutations; `_count_order_sensitive_sources` decides when
that's insufficient (two or more interior-order-sensitive contributors) and
falls back to full enumeration.
"""

from __future__ import annotations

import itertools

import pytest
from hand_solver import (
    _count_order_sensitive_sources,
    _first_last_covering_permutations,
    evaluate_value,
)

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand


def _five_plain_cards() -> list:
    return [
        create_playing_card(Suit.HEARTS, Rank.TWO),
        create_playing_card(Suit.SPADES, Rank.THREE),
        create_playing_card(Suit.CLUBS, Rank.FOUR),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.HEARTS, Rank.SIX),
    ]


def test_covering_set_hits_every_first_last_pair_exactly_once() -> None:
    cards = _five_plain_cards()
    perms = _first_last_covering_permutations(cards)

    assert len(perms) == 20
    orderings_by_identity = {tuple(id(c) for c in p) for p in perms}
    assert len(orderings_by_identity) == 20, "covering set should contain no duplicate orderings"

    pairs = {(id(p[0]), id(p[-1])) for p in perms}
    expected_pairs = {(id(a), id(b)) for a in cards for b in cards if a is not b}
    assert pairs == expected_pairs


def test_count_order_sensitive_sources_ignores_identity_only_jokers() -> None:
    cards = _five_plain_cards()
    jokers = [create_joker("j_hanging_chad")]
    assert _count_order_sensitive_sources(cards, jokers) == 0


def test_count_order_sensitive_sources_counts_card_level_xmult() -> None:
    cards = _five_plain_cards()
    cards[0] = create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_glass")
    assert _count_order_sensitive_sources(cards, []) == 1

    cards[1] = create_playing_card(Suit.SPADES, Rank.ACE, enhancement="m_glass")
    assert _count_order_sensitive_sources(cards, []) == 2


def test_count_order_sensitive_sources_counts_xmult_joker_once() -> None:
    cards = _five_plain_cards()
    jokers = [create_joker("j_photograph")]
    assert _count_order_sensitive_sources(cards, jokers) == 1


def _brute_force_best_total(played, held, jokers, hand_levels, blind, rng) -> float:
    import copy as _copy

    best = None
    for order in itertools.permutations(played):
        result = score_hand(
            [_copy.deepcopy(c) for c in order],
            [_copy.deepcopy(c) for c in held],
            [_copy.deepcopy(j) for j in jokers],
            _copy.deepcopy(hand_levels),
            blind,
            _copy.deepcopy(rng),
        )
        if best is None or result.total > best:
            best = result.total
    assert best is not None
    return best


@pytest.mark.parametrize("seed", ["COVER_1", "COVER_2", "COVER_3"])
def test_single_card_level_xmult_source_matches_brute_force(seed: str) -> None:
    """One Glass-enhanced card among five: `evaluate_value` (via the 20-
    permutation covering set) must find the same max as trying all 120."""
    played = _five_plain_cards()
    played[2] = create_playing_card(Suit.CLUBS, Rank.KING, enhancement="m_glass")
    hand_levels = HandLevels()
    blind = Blind.create("bl_small", ante=1)
    rng = PseudoRandom(seed)

    assert _count_order_sensitive_sources(played, []) == 1

    result = evaluate_value(list(played), [], [], hand_levels, blind, rng)
    brute = _brute_force_best_total(played, [], [], hand_levels, blind, rng)
    assert result.total == brute


def test_single_identity_joker_matches_brute_force() -> None:
    """Hanging Chad alone (identity-only, first-card-dependent): the
    covering set's exhaustive coverage of "who is first" must find the same
    max as trying all 120 orderings."""
    played = _five_plain_cards()
    jokers = [create_joker("j_hanging_chad")]
    hand_levels = HandLevels()
    blind = Blind.create("bl_small", ante=1)
    rng = PseudoRandom("IDENTITY_COVER")

    assert _count_order_sensitive_sources(played, jokers) == 0

    result = evaluate_value(list(played), [], jokers, hand_levels, blind, rng)
    brute = _brute_force_best_total(played, [], jokers, hand_levels, blind, rng)
    assert result.total == brute


def test_two_card_level_xmult_sources_falls_back_to_full_enumeration() -> None:
    """Two Glass-enhanced cards: interior order between them can matter, so
    the solver must fall back to full enumeration -- verify it still finds
    the true brute-force max (a regression check that the fallback branch
    actually triggers, not just that it would be correct if it did)."""
    played = _five_plain_cards()
    played[1] = create_playing_card(Suit.SPADES, Rank.KING, enhancement="m_glass")
    played[3] = create_playing_card(Suit.DIAMONDS, Rank.QUEEN, enhancement="m_glass")
    hand_levels = HandLevels()
    blind = Blind.create("bl_small", ante=1)
    rng = PseudoRandom("MULTI_SOURCE")

    assert _count_order_sensitive_sources(played, []) == 2

    result = evaluate_value(list(played), [], [], hand_levels, blind, rng)
    brute = _brute_force_best_total(played, [], [], hand_levels, blind, rng)
    assert result.total == brute
