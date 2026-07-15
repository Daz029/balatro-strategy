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
    MAIN_PHASE_XMULT_JOKER_KEYS,
    best_joker_order,
    best_play_order,
    candidate_orderings,
    fast_clone_blind,
    fast_clone_card,
    fast_clone_hand_levels,
    fast_clone_rng,
    joker_multiplies_at_position,
    joker_order_matters,
    sorted_joker_order,
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


# ---------------------------------------------------------------------------
# B3: joker-list auto-ordering
# ---------------------------------------------------------------------------


def _pair_cards() -> list:
    # Kings pair so hand-type-conditional xmult jokers (The Duo) fire.
    return [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.FOUR),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.HEARTS, Rank.SIX),
    ]


def _score_joker_order(
    played, held, joker_order, hand_levels, blind, rng
) -> float:
    """Score one JOKER ordering against fully cloned state."""
    result = score_hand(
        [fast_clone_card(c) for c in played],
        [fast_clone_card(c) for c in held],
        [fast_clone_card(j) for j in joker_order],
        fast_clone_hand_levels(hand_levels),
        fast_clone_blind(blind),
        fast_clone_rng(rng),
        game_state={},
        blind_chips=300,
    )
    return result.total


def _brute_force_joker_best(played, held, jokers, hand_levels, blind, rng) -> float:
    return max(
        _score_joker_order(played, held, list(order), hand_levels, blind, rng)
        for order in permutations(jokers)
    )


def _scan_main_xmult_keys() -> frozenset:
    """Regenerate the main-phase Xmult key set from the handler SOURCE --
    the drift guard for MAIN_PHASE_XMULT_JOKER_KEYS (a new/changed handler
    that starts returning Xmult_mod in joker_main must show up in the
    hand-written set or this test fails)."""
    import inspect
    import re

    import jackdaw.engine.jokers as jokers_mod

    src = inspect.getsource(jokers_mod)
    pairs = re.findall(r'@register\("(j_[a-z0-9_]+)"\)\s*\ndef (_[a-z0-9_]+)\(', src)
    parts = re.split(r"\ndef (_[a-z0-9_]+)\(", src)
    xmult_handlers = {
        parts[i]
        for i in range(1, len(parts), 2)
        if "Xmult_mod" in parts[i + 1] and "joker_main" in parts[i + 1]
    }
    return frozenset(k for k, h in pairs if h in xmult_handlers)


class TestJokerOrderClassification:
    def test_xmult_keys_pinned_to_handler_source(self):
        assert MAIN_PHASE_XMULT_JOKER_KEYS == _scan_main_xmult_keys()

    def test_polychrome_edition_classifies_as_xmult(self):
        j = create_joker("j_joker")  # plain +4 mult joker
        assert not joker_multiplies_at_position(j)
        j.set_edition({"polychrome": True})
        assert joker_multiplies_at_position(j)

    def test_order_matters_gate(self):
        additive = create_joker("j_joker")
        xmult = create_joker("j_duo")
        assert not joker_order_matters([])
        assert not joker_order_matters([xmult])  # single joker
        assert not joker_order_matters([additive, create_joker("j_joker")])
        assert not joker_order_matters([xmult, create_joker("j_duo")])  # all-x
        assert joker_order_matters([additive, xmult])
        assert joker_order_matters([additive, create_joker("j_blueprint")])

    def test_sorted_order_additive_first_stable(self):
        a1 = create_joker("j_joker")
        x1 = create_joker("j_duo")
        a2 = create_joker("j_joker")
        x2 = create_joker("j_tribe")
        order = sorted_joker_order([x1, a1, x2, a2])
        assert [id(j) for j in order] == [id(a1), id(a2), id(x1), id(x2)]


class TestBestJokerOrder:
    def _board(self, joker_keys):
        played = _pair_cards()
        jokers = [create_joker(k) for k in joker_keys]
        hl = HandLevels()
        blind = _small_blind()
        rng = PseudoRandom("tj")
        return played, [], jokers, hl, blind, rng

    def _assert_matches_brute_force(self, joker_keys):
        played, held, jokers, hl, blind, rng = self._board(joker_keys)
        chosen = best_joker_order(jokers, played, held, hl, blind, rng, blind_chips=300)
        chosen_total = _score_joker_order(played, held, chosen, hl, blind, rng)
        brute_best = _brute_force_joker_best(played, held, jokers, hl, blind, rng)
        assert chosen_total == brute_best

    def test_pure_additive_matches_brute_force(self):
        self._assert_matches_brute_force(["j_joker", "j_joker", "j_joker"])

    def test_pure_xmult_matches_brute_force(self):
        self._assert_matches_brute_force(["j_duo", "j_tribe", "j_trio"])

    def test_mixed_matches_brute_force(self):
        self._assert_matches_brute_force(["j_duo", "j_joker", "j_joker", "j_trio"])

    def test_blueprint_matches_brute_force(self):
        self._assert_matches_brute_force(["j_blueprint", "j_joker", "j_duo"])

    def test_brainstorm_matches_brute_force(self):
        self._assert_matches_brute_force(["j_brainstorm", "j_duo", "j_joker"])

    def test_both_copy_jokers_match_brute_force(self):
        self._assert_matches_brute_force(
            ["j_blueprint", "j_brainstorm", "j_joker", "j_duo"]
        )

    def test_per_card_phase_jokers_match_brute_force(self):
        # Greedy Joker adds mult PER SCORED DIAMOND (Phase 7, per-card),
        # interleaving with Duo's Phase-9 xmult -- the closed form is only
        # proven for the independent Phase-9 chain, so this pins the
        # per-card interleaving empirically.
        played = [
            create_playing_card(Suit.DIAMONDS, Rank.KING),
            create_playing_card(Suit.DIAMONDS, Rank.KING),
            create_playing_card(Suit.DIAMONDS, Rank.FOUR),
            create_playing_card(Suit.CLUBS, Rank.FIVE),
            create_playing_card(Suit.HEARTS, Rank.SIX),
        ]
        jokers = [create_joker("j_duo"), create_joker("j_greedy_joker"), create_joker("j_joker")]
        hl = HandLevels()
        blind = _small_blind()
        rng = PseudoRandom("tg")
        chosen = best_joker_order(jokers, played, [], hl, blind, rng, blind_chips=300)
        chosen_total = _score_joker_order(played, [], chosen, hl, blind, rng)
        brute_best = _brute_force_joker_best(played, [], jokers, hl, blind, rng)
        assert chosen_total == brute_best

    def test_context_free_tier_needs_no_evaluation(self):
        # Without a candidate play the closed form must not score anything:
        # the RNG state is untouched even with a copy joker owned.
        jokers = [create_joker("j_blueprint"), create_joker("j_joker"), create_joker("j_duo")]
        rng = PseudoRandom("tc")
        state_before = dict(rng._state)
        order = best_joker_order(jokers)
        assert sorted(map(id, order)) == sorted(map(id, jokers))
        assert rng._state == state_before

    def test_returns_same_objects_and_never_mutates(self):
        played, held, jokers, hl, blind, rng = self._board(
            ["j_blueprint", "j_joker", "j_duo"]
        )
        rng_before = dict(rng._state)
        abilities_before = [dict(j.ability) for j in jokers]
        hands_used_before = dict(blind.hands_used)

        order = best_joker_order(jokers, played, held, hl, blind, rng, blind_chips=300)

        assert sorted(map(id, order)) == sorted(map(id, jokers))
        assert rng._state == rng_before
        assert [dict(j.ability) for j in jokers] == abilities_before
        assert blind.hands_used == hands_used_before
        assert all(hs.played == 0 for hs in hl._hands.values())
