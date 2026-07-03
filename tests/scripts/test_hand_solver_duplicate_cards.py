"""Regression test: `best_immediate_play` must correctly size `held` when
the hand contains duplicate-valued cards (possible with Erratic decks --
`erratic_suits_and_ranks` assigns rank/suit independently per card, so two
physically distinct cards can end up with identical field values).

`held = [c for c in hand if c not in combo]` used value-equality (`Card` is
a plain @dataclass, so `==` compares fields, not identity). Two distinct
card objects with identical field values would BOTH match a single `combo`
membership check, so `held` came out one card short whenever such a
duplicate pair was split between `combo` and `held`. Fixed to filter by
`id()` instead -- this test checks the invariant that would have failed
before the fix: `len(played) + len(held) == len(hand)` for every combo
tried, even with duplicate-valued cards present.
"""

from __future__ import annotations

import hand_solver
from hand_solver import best_immediate_play

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def test_held_size_correct_with_duplicate_valued_cards(monkeypatch) -> None:
    hand = [
        create_playing_card(Suit.HEARTS, Rank.TWO),
        create_playing_card(Suit.HEARTS, Rank.TWO),  # duplicate of the above
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.QUEEN),
        create_playing_card(Suit.DIAMONDS, Rank.JACK),
    ]

    seen_sizes: list[tuple[int, int]] = []
    real_evaluate_value = hand_solver.evaluate_value

    def spy_evaluate_value(played, held, *args, **kwargs):
        seen_sizes.append((len(played), len(held)))
        return real_evaluate_value(played, held, *args, **kwargs)

    monkeypatch.setattr(hand_solver, "evaluate_value", spy_evaluate_value)

    hand_levels = HandLevels()
    blind = Blind.create("bl_small", ante=1)
    rng = PseudoRandom("DUPLICATE_HELD_CHECK")

    best_immediate_play(hand, [], hand_levels, blind, rng)

    assert seen_sizes, "evaluate_value was never called"
    for played_len, held_len in seen_sizes:
        assert played_len + held_len == len(hand), (
            f"played ({played_len}) + held ({held_len}) != hand size "
            f"({len(hand)}) -- held was miscounted for a duplicate-valued hand"
        )
