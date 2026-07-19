"""Tests for the clear-gated money-aware joker-ordering objective."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from jackdaw.agents.hand_action_space import combo_to_action
from jackdaw.agents.ordering_objective import make_clear_gated_money_objective
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import blueprint_compatible
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
from jackdaw.env.trigger_match import resolve_copy_targets


@dataclass(frozen=True)
class _Result:
    total: int
    dollars_earned: int


def _result(total: int, dollars: int) -> _Result:
    return _Result(total=total, dollars_earned=dollars)


def _pair_board():
    """Kings pair with diamonds; copy targets are score/economy-sensitive."""
    hand = [
        create_playing_card(Suit.DIAMONDS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.KING),
    ]
    rough_gem = create_joker("j_rough_gem")
    jokers = [create_joker("j_blueprint"), rough_gem, create_joker("j_duo")]
    return (
        hand,
        jokers,
        rough_gem,
        HandLevels(),
        Blind.create("bl_small", ante=1),
        PseudoRandom("objective"),
    )


def _resolved_blueprint_target(jokers):
    """Return the target selected by the engine-backed copy resolver."""
    resolutions = resolve_copy_targets({"jokers": jokers})
    blueprint_index = next(i for i, joker in enumerate(jokers) if joker.center_key == "j_blueprint")
    resolution = resolutions[blueprint_index]
    assert resolution.active
    return jokers[resolution.target_index]


def _score(played, jokers, hand_levels, blind, rng):
    return score_hand(
        [fast_clone_card(card) for card in played],
        [],
        [fast_clone_card(joker) for joker in jokers],
        fast_clone_hand_levels(hand_levels),
        fast_clone_blind(blind),
        fast_clone_rng(rng),
        game_state={},
        blind_chips=blind.chips,
    )


def test_clearing_arm_prefers_dollars():
    """Pins dollars-first ordering after the clear gate wins."""
    objective = make_clear_gated_money_objective(
        {"chips": 100, "blind": SimpleNamespace(chips=100)}
    )
    assert objective(_result(1, 9)) > objective(_result(9, 1))


def test_non_clearing_arm_prefers_score():
    """Pins score-first ordering when neither candidate clears."""
    objective = make_clear_gated_money_objective(
        {"chips": 0, "blind": SimpleNamespace(chips=100)}
    )
    assert objective(_result(9, 1)) > objective(_result(1, 9))


def test_clear_boundary_is_inclusive():
    """Pins equality at the blind target to the clearing arm."""
    objective = make_clear_gated_money_objective(
        {"chips": 99, "blind": SimpleNamespace(chips=100)}
    )
    assert objective(_result(1, 1)) == (1.0, 1.0, 1.0)


def test_none_blind_is_clearing_arm():
    """Pins no blind as an immediately clearable state."""
    objective = make_clear_gated_money_objective({"chips": 0, "blind": None})
    assert objective(_result(0, 4)) == (1.0, 4.0, 0.0)


def test_objective_snapshots_gate_state():
    """Pins factory-time capture of banked chips and blind need."""
    game_state = {"chips": 0, "blind": SimpleNamespace(chips=100)}
    objective = make_clear_gated_money_objective(game_state)
    game_state["chips"] = 100
    assert objective(_result(0, 4)) == (0.0, 0.0, 4.0)


def test_engine_ordering_objective_selects_money_copy_target():
    """Pins engine placement argmax across clear and non-clear arms."""
    hand, jokers, rough_gem, hand_levels, blind, rng = _pair_board()
    assert blueprint_compatible(rough_gem)

    secured = {"chips": blind.chips, "blind": blind}
    money_order = best_joker_order(
        jokers,
        hand,
        [],
        hand_levels,
        blind,
        rng,
        game_state=secured,
        blind_chips=blind.chips,
        objective=make_clear_gated_money_objective(secured),
    )
    assert _resolved_blueprint_target(money_order) is rough_gem

    raw_order = best_joker_order(
        jokers,
        hand,
        [],
        hand_levels,
        blind,
        rng,
        blind_chips=blind.chips,
        objective=None,
    )
    non_clear = {"chips": 0, "blind": SimpleNamespace(chips=10_000)}
    non_clear_order = best_joker_order(
        jokers,
        hand,
        [],
        hand_levels,
        blind,
        rng,
        game_state=non_clear,
        blind_chips=blind.chips,
        objective=make_clear_gated_money_objective(non_clear),
    )
    assert _resolved_blueprint_target(raw_order).center_key == "j_duo"
    assert _resolved_blueprint_target(non_clear_order).center_key == "j_duo"
    assert _score(hand, money_order, hand_levels, blind, rng).dollars_earned > _score(
        hand, raw_order, hand_levels, blind, rng
    ).dollars_earned


def test_decoder_passes_objective_to_joker_ordering():
    """Pins decoder plumbing through to persistent money copy placement."""
    hand, jokers, rough_gem, hand_levels, blind, rng = _pair_board()
    assert blueprint_compatible(rough_gem)
    game_state = {
        "phase": GamePhase.SELECTING_HAND,
        "hand": hand,
        "jokers": jokers,
        "hand_levels": hand_levels,
        "blind": blind,
        "rng": rng,
        "chips": blind.chips,
        "current_round": {"hands_left": 3, "discards_left": 3},
    }
    action_to_engine_action(
        combo_to_action(ActionType.PlayHand, (0, 1)),
        game_state,
        ordering_objective=make_clear_gated_money_objective(game_state),
    )
    assert _resolved_blueprint_target(game_state["jokers"]) is rough_gem
