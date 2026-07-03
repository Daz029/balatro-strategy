"""Empirical safety net for `hand_solver._needs_permutation_search`.

The predicate is a static, cheap approximation of a dynamic property (does
permuting scored-card order change `score_hand`'s total). This test checks
that approximation against the real engine directly: whenever the predicate
says permutation search is unnecessary, scoring the same cards in several
different orders must actually produce identical totals. Any mismatch here
means the predicate is unsound and must be corrected before it's trusted by
`hand_solver.py`'s "exact solver" design.
"""

from __future__ import annotations

import copy
import random

import pytest
from hand_solver import _needs_permutation_search

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand

# Every joker verified (by direct source read) to fire per scored card with
# a MULTIPLICATIVE (x_mult) contribution, or with a position-identity effect
# -- these should make the predicate return True.
_ORDER_SENSITIVE_POOL = [
    "j_photograph",
    "j_bloodstone",
    "j_ancient",
    "j_triboulet",
    "j_idol",
    "j_hanging_chad",
]

# Jokers verified to fire per scored card with only ADDITIVE (mult/chips)
# contributions, plus a couple of jokers with no per-card scoring effect at
# all (control group) -- combinations drawn only from this pool should be
# safe to skip permutation search over.
_ORDER_SAFE_POOL = [
    "j_greedy_joker",
    "j_lusty_joker",
    "j_wrathful_joker",
    "j_gluttenous_joker",
    "j_onyx_agate",
    "j_fibonacci",
    "j_scholar",
    "j_walkie_talkie",
    "j_even_steven",
    "j_smiley",
    "j_odd_todd",
    "j_scary_face",
    "j_arrowhead",
    "j_joker",  # flat +mult, joker_main only -- not per-card at all
    "j_rough_gem",  # +$ per diamond, not mult
    "j_business",  # +$ per face card, not mult
]

# Enhancements/editions with no per-card xmult and no shared-RNG-stream
# effect -- safe to mix freely in the "should be order-invariant" pool.
# `m_glass` (xmult) and `m_lucky` (RNG-stream order) are excluded here on
# purpose; they're covered by the dedicated positive tests below.
_SAFE_ENHANCEMENTS = [None, "m_mult", "m_bonus", "m_steel"]
_SAFE_EDITIONS: list[dict[str, bool] | None] = [None, {"foil": True}, {"holo": True}]


def _random_safe_card(rng: random.Random) -> object:
    suit = rng.choice(list(Suit))
    rank = rng.choice(list(Rank))
    enhancement = rng.choice(_SAFE_ENHANCEMENTS) or "c_base"
    edition = rng.choice(_SAFE_EDITIONS)
    return create_playing_card(suit, rank, enhancement=enhancement, edition=edition)


def _score(cards: list, jokers: list, seed: str) -> float:
    hand_levels = HandLevels()
    blind = Blind.create("bl_small", ante=1)
    rng = PseudoRandom(seed)
    result = score_hand(
        list(cards),
        [],
        jokers,
        hand_levels,
        blind,
        rng,
    )
    return result.total


@pytest.mark.parametrize("trial", range(60))
def test_no_permutation_needed_implies_order_invariant(trial: int) -> None:
    py_rng = random.Random(f"XCHECK_{trial}")

    n_jokers = py_rng.randint(0, 3)
    joker_keys = py_rng.sample(_ORDER_SAFE_POOL, k=n_jokers)
    jokers = [create_joker(k) for k in joker_keys]

    n_cards = py_rng.randint(2, 5)
    base_cards = [_random_safe_card(py_rng) for _ in range(n_cards)]

    assert not _needs_permutation_search(base_cards, jokers), (
        f"predicate flagged order-sensitivity for a safe-pool-only combo: "
        f"jokers={joker_keys}"
    )

    seed = f"XCHECK_SCORE_{trial}"
    baseline = _score([copy.deepcopy(c) for c in base_cards], jokers, seed)

    for _ in range(3):
        shuffled = base_cards[:]
        py_rng.shuffle(shuffled)
        total = _score([copy.deepcopy(c) for c in shuffled], jokers, seed)
        assert total == baseline, (
            f"order changed total ({total} != {baseline}) despite "
            f"_needs_permutation_search()==False; jokers={joker_keys}"
        )


@pytest.mark.parametrize("joker_key", _ORDER_SENSITIVE_POOL)
def test_order_sensitive_jokers_are_flagged(joker_key: str) -> None:
    py_rng = random.Random(f"POSITIVE_{joker_key}")
    cards = [_random_safe_card(py_rng) for _ in range(py_rng.randint(2, 5))]
    jokers = [create_joker(joker_key)]
    assert _needs_permutation_search(cards, jokers)


def test_glass_card_is_flagged() -> None:
    cards = [create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_glass")]
    assert _needs_permutation_search(cards, [])


def test_polychrome_edition_is_flagged() -> None:
    cards = [
        create_playing_card(Suit.HEARTS, Rank.ACE, edition={"polychrome": True}),
    ]
    assert _needs_permutation_search(cards, [])


def test_pure_additive_scene_is_not_flagged() -> None:
    cards = [
        create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_mult"),
        create_playing_card(Suit.SPADES, Rank.KING, edition={"holo": True}),
    ]
    jokers = [create_joker("j_greedy_joker"), create_joker("j_smiley")]
    assert not _needs_permutation_search(cards, jokers)
