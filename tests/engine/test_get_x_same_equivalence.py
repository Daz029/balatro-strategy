"""Equivalence pin for the `get_x_same` O(n) rewrite.

`get_x_same` is 18% of solver runtime and its doubled `get_id()` call inside an
O(n^2) loop is the source of ~14.1M `get_id` calls (and, through them, ~30.4M
`dict.get` calls) in a single labeled example. Rewriting it is a pure speedup
with NO intended behaviour change -- so it is pinned against a literal
transcription of the original Lua-faithful implementation, which is kept here
as the reference oracle.

Three behaviours are load-bearing and easy to lose in a rewrite:

1. **Group ordering is by ASCENDING hand index.** The original walks `i`
   DOWNWARD overwriting `vals[card_id]`, so the surviving group is the one
   built from the LOWEST index with that rank -- i.e. `[hand[i_min]]` followed
   by every other match in ascending `j` order, which is simply ascending
   index order.
2. **Output is by DESCENDING rank id (14 -> 1).**
3. **Rank ids outside 1..14 are silently EXCLUDED** -- Stone Cards return -1
   and base-less cards return 0, and the original's `range(14, 0, -1)` output
   loop drops both. Cards are still counted into their group; the group just
   never reaches the output.

Card identity matters (`is`, not `==`): Card is a value-equality dataclass, so
an Erratic deck's duplicate cards must not be conflated (the same bug class as
`best_immediate_play`'s id()-based `held` filter).
"""

from __future__ import annotations

from itertools import product
import random

import pytest

from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_eval import get_x_same


class _IdCard:
    """Minimal identity-bearing card double for exhaustive ID testing."""

    def __init__(self, card_id: int):
        self.card_id = card_id

    def get_id(self) -> int:
        return self.card_id


def reference_get_x_same(num: int, hand: list[Card]) -> list[list[Card]]:
    """Literal transcription of the ORIGINAL implementation (misc_functions.lua
    :592) -- the oracle the optimized version must match exactly."""
    vals: dict[int, list[Card]] = {}
    for i in range(len(hand) - 1, -1, -1):
        curr = [hand[i]]
        card_id = hand[i].get_id()
        for j in range(len(hand)):
            if hand[i].get_id() == hand[j].get_id() and i != j:
                curr.append(hand[j])
        if len(curr) == num:
            vals[card_id] = curr
    ret: list[list[Card]] = []
    for rank_id in range(14, 0, -1):
        if rank_id in vals:
            ret.append(vals[rank_id])
    return ret


def _card(rank: str, suit: str = "Hearts") -> Card:
    return create_playing_card(Suit(suit), Rank(rank))


def _same(a: list[list[Card]], b: list[list[Card]]) -> bool:
    """Identity-based structural equality: same groups, same order, same card
    OBJECTS in the same positions."""
    if len(a) != len(b):
        return False
    return all(
        len(ga) == len(gb) and all(x is y for x, y in zip(ga, gb, strict=True))
        for ga, gb in zip(a, b, strict=True)
    )


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
ALL_IDS = [-1, 0, *range(1, 15)]


class TestEquivalence:
    def test_matches_reference_on_all_short_id_hands(self):
        """Exhaust every ordered hand up to four cards across all IDs."""
        for size in range(5):
            for values in product(ALL_IDS, repeat=size):
                hand = [_IdCard(value) for value in values]
                for num in range(1, 6):
                    assert _same(get_x_same(num, hand), reference_get_x_same(num, hand)), (
                        f"num={num} values={values}"
                    )

    @pytest.mark.parametrize("num", [1, 2, 3, 4, 5])
    def test_matches_reference_on_random_hands(self, num):
        rng = random.Random(1234)
        for _ in range(300):
            size = rng.randint(0, 8)
            hand = [_card(rng.choice(RANKS), rng.choice(SUITS)) for _ in range(size)]
            assert _same(get_x_same(num, hand), reference_get_x_same(num, hand)), (
                f"num={num} hand={[str(c) for c in hand]}"
            )

    @pytest.mark.parametrize("num", [1, 2, 3, 4, 5])
    def test_matches_reference_on_duplicate_heavy_hands(self, num):
        """Erratic-deck shape: many identical rank/suit cards, where value
        equality would conflate genuinely distinct Card objects."""
        rng = random.Random(99)
        for _ in range(300):
            size = rng.randint(0, 8)
            hand = [_card(rng.choice(["King", "Ace"]), rng.choice(["Hearts", "Spades"]))
                    for _ in range(size)]
            assert _same(get_x_same(num, hand), reference_get_x_same(num, hand))

    def test_empty_hand(self):
        assert get_x_same(2, []) == reference_get_x_same(2, []) == []


class TestLoadBearingBehaviours:
    def test_groups_are_in_ascending_hand_index_order(self):
        hand = [_card("King", "Hearts"), _card("2"), _card("King", "Spades")]
        (group,) = get_x_same(2, hand)
        assert group[0] is hand[0]
        assert group[1] is hand[2]

    def test_output_is_descending_rank(self):
        hand = [_card("2"), _card("2", "Spades"), _card("Ace"), _card("Ace", "Spades")]
        groups = get_x_same(2, hand)
        assert [g[0].get_id() for g in groups] == [14, 2]

    def test_exact_count_only(self):
        """A trio must NOT surface as a pair."""
        hand = [_card("King"), _card("King", "Spades"), _card("King", "Clubs")]
        assert get_x_same(2, hand) == []
        assert len(get_x_same(3, hand)) == 1

    def test_stone_cards_are_excluded_from_output(self):
        """Stone returns id -1, which the 14..1 output loop drops."""
        stone_a, stone_b = _card("King"), _card("King", "Spades")
        for c in (stone_a, stone_b):
            c.ability["effect"] = "Stone Card"
        hand = [stone_a, stone_b]
        assert get_x_same(2, hand) == reference_get_x_same(2, hand) == []

    def test_stone_cards_do_not_join_ranked_groups(self):
        """A Stone King is id -1, so it must not pair with a real King."""
        stone = _card("King")
        stone.ability["effect"] = "Stone Card"
        real = _card("King", "Spades")
        hand = [stone, real]
        assert get_x_same(2, hand) == reference_get_x_same(2, hand) == []
        assert _same(get_x_same(1, hand), reference_get_x_same(1, hand))
