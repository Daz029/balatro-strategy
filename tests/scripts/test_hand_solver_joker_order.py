"""Tests for B3 -- joker auto-ordering on the solver's exact path.

`best_joker_order` itself is brute-force-validated in
tests/engine/test_play_ordering.py; this file pins the SOLVER wiring:

  - `evaluate_value` re-orders the joker list on the exact path (a
    deliberately mis-ordered board scores as well as the brute-force best
    joker order);
  - the MC tier (`search_orderings=False`) keeps the caller's fixed order
    (its documented approximation tier);
  - solver/env consistency: the value `evaluate_value` assigns a subset
    equals what env execution produces for the same subset --
    `action_to_engine_action`'s persistent joker mutation + card ordering,
    scored through the same engine (the discard-cap bug class dies here:
    the solver must never value an ordering the env won't execute);
  - the `objective` hook re-targets copy placement (a dollars-maximizing
    objective picks a different Blueprint neighbor than raw score when a
    Business-Card-class economy joker is on the board).
"""

from __future__ import annotations

from itertools import permutations

from hand_solver import evaluate_value

from jackdaw.agents.hand_action_space import combo_to_action
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.play_ordering import (
    best_joker_order,
    fast_clone_blind,
    fast_clone_card,
    fast_clone_hand_levels,
    fast_clone_rng,
)
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand
from jackdaw.env.action_space import ActionType
from jackdaw.env.hand_play_gym import action_to_engine_action

BLIND_CHIPS = 300


def _pair_board():
    """Kings pair + fillers; jokers deliberately x-mult-FIRST (worst order)."""
    hand = [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.FOUR),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.HEARTS, Rank.SIX),
        create_playing_card(Suit.CLUBS, Rank.EIGHT),
        create_playing_card(Suit.DIAMONDS, Rank.NINE),
        create_playing_card(Suit.SPADES, Rank.TEN),
    ]
    jokers = [create_joker("j_duo"), create_joker("j_joker"), create_joker("j_joker")]
    return hand, jokers, HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom("b3")


def _score_fixed(played, held, joker_order, hl, blind, rng) -> float:
    result = score_hand(
        [fast_clone_card(c) for c in played],
        [fast_clone_card(c) for c in held],
        [fast_clone_card(j) for j in joker_order],
        fast_clone_hand_levels(hl),
        fast_clone_blind(blind),
        fast_clone_rng(rng),
        game_state={},
        blind_chips=BLIND_CHIPS,
    )
    return result.total


class TestEvaluateValueJokerOrdering:
    def test_exact_path_reaches_brute_force_joker_order(self):
        hand, jokers, hl, blind, rng = _pair_board()
        played, held = hand[:2], hand[2:]

        value = evaluate_value(
            played, held, jokers, hl, blind, rng, {}, BLIND_CHIPS
        ).total
        brute_best = max(
            _score_fixed(played, held, list(order), hl, blind, rng)
            for order in permutations(jokers)
        )
        given_order = _score_fixed(played, held, jokers, hl, blind, rng)
        assert value == brute_best
        assert brute_best > given_order  # the mis-ordered board was really worse

    def test_mc_tier_keeps_callers_fixed_order(self):
        hand, jokers, hl, blind, rng = _pair_board()
        played, held = hand[:2], hand[2:]

        value = evaluate_value(
            played, held, jokers, hl, blind, rng, {}, BLIND_CHIPS,
            search_orderings=False,
        ).total
        assert value == _score_fixed(played, held, jokers, hl, blind, rng)

    def test_live_jokers_never_mutated(self):
        hand, jokers, hl, blind, rng = _pair_board()
        ids_before = [id(j) for j in jokers]
        abilities_before = [dict(j.ability) for j in jokers]
        evaluate_value(hand[:2], hand[2:], jokers, hl, blind, rng, {}, BLIND_CHIPS)
        assert [id(j) for j in jokers] == ids_before
        assert [dict(j.ability) for j in jokers] == abilities_before


class TestSolverEnvConsistency:
    def test_label_value_equals_env_execution_value(self):
        # Same subset, two routes: (a) the solver's evaluate_value; (b) the
        # env's action_to_engine_action -- which persistently reorders
        # gs["jokers"] and picks the card order -- then the engine scores
        # what the env would submit. The two totals must agree.
        hand, jokers, hl, blind, rng = _pair_board()
        combo = (0, 1)
        played = [hand[i] for i in combo]
        held = [c for i, c in enumerate(hand) if i not in combo]

        label_value = evaluate_value(
            played, held, jokers, hl, blind, rng, None, BLIND_CHIPS
        ).total

        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "hand": hand,
            "jokers": list(jokers),
            "hand_levels": hl,
            "blind": blind,
            "rng": rng,
            "current_round": {"hands_left": 3, "discards_left": 3},
        }
        # Neutralize blind_chips difference: the test blind's real chips.
        blind.chips = BLIND_CHIPS
        engine_action = action_to_engine_action(
            combo_to_action(ActionType.PlayHand, combo), gs
        )
        env_played = [hand[i] for i in engine_action.card_indices]
        env_value = _score_fixed(env_played, held, gs["jokers"], hl, blind, rng)
        assert env_value == label_value

    def test_env_mutation_is_persistent_and_permutation(self):
        hand, jokers, hl, blind, rng = _pair_board()
        blind.chips = BLIND_CHIPS
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "hand": hand,
            "jokers": list(jokers),
            "hand_levels": hl,
            "blind": blind,
            "rng": rng,
            "current_round": {"hands_left": 3, "discards_left": 3},
        }
        before_ids = sorted(map(id, gs["jokers"]))
        action_to_engine_action(combo_to_action(ActionType.PlayHand, (0, 1)), gs)
        assert sorted(map(id, gs["jokers"])) == before_ids  # same objects
        # additive (j_joker) now precedes x-mult (j_duo)
        keys = [j.center_key for j in gs["jokers"]]
        assert keys == ["j_joker", "j_joker", "j_duo"]

    def test_discard_never_reorders(self):
        hand, jokers, hl, blind, rng = _pair_board()
        gs = {
            "phase": GamePhase.SELECTING_HAND,
            "hand": hand,
            "jokers": list(jokers),
            "hand_levels": hl,
            "blind": blind,
            "rng": rng,
            "current_round": {"hands_left": 3, "discards_left": 3},
        }
        keys_before = [j.center_key for j in gs["jokers"]]
        action_to_engine_action(combo_to_action(ActionType.Discard, (0, 1)), gs)
        assert [j.center_key for j in gs["jokers"]] == keys_before


class TestOrderingObjectiveHook:
    def test_dollar_objective_changes_copy_placement(self):
        # Business Card: scored cards have a chance to pay $2 -- copyable
        # (flows through scoring), and worth nothing to a score-argmax.
        # With a dollars-first objective, Blueprint should prefer the
        # Business Card neighbor; with raw score, the x-mult neighbor.
        played = [
            create_playing_card(Suit.HEARTS, Rank.KING),
            create_playing_card(Suit.SPADES, Rank.KING),
        ]
        jokers = [
            create_joker("j_blueprint"),
            create_joker("j_business"),
            create_joker("j_duo"),
        ]
        hl = HandLevels()
        blind = Blind.create("bl_small", ante=1)
        rng = PseudoRandom("hook")

        score_order = best_joker_order(
            jokers, played, [], hl, blind, rng, blind_chips=BLIND_CHIPS
        )
        dollar_order = best_joker_order(
            jokers, played, [], hl, blind, rng, blind_chips=BLIND_CHIPS,
            objective=lambda result: (result.dollars_earned, result.total),
        )

        def blueprint_right_neighbor(order):
            keys = [j.center_key for j in order]
            i = keys.index("j_blueprint")
            return keys[i + 1] if i + 1 < len(keys) else None

        assert blueprint_right_neighbor(score_order) == "j_duo"
        assert blueprint_right_neighbor(dollar_order) == "j_business"
