"""Tests for HandPlayAdapter — isolated hand-play episode injection.

Covers:
- reset() lands directly in SELECTING_HAND with sampled ante/hands/discards/
  money/jokers applied, skipping BLIND_SELECT and SHOP entirely
- reset() is deterministic given the same seed
- step() reuses the real engine and can run a hand-play episode to
  completion (win via ROUND_EVAL or loss via GAME_OVER)
- GameAdapter protocol compliance
"""

from __future__ import annotations

import random

from jackdaw.engine.actions import Discard, GamePhase, PlayHand
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.hand_play_adapter import HandPlayAdapter, HandPlayConfig

SEED = "TEST_HAND_PLAY_1"
BACK = "b_red"
STAKE = 1


def _random_agent_step(adapter: HandPlayAdapter) -> None:
    legal = adapter.get_legal_actions()
    assert legal, "No legal actions available"

    hand = adapter.raw_state.get("hand", [])
    action = random.choice(legal)

    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        action = PlayHand(card_indices=indices)

    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        action = Discard(card_indices=indices)

    adapter.step(action)


def test_protocol_compliance() -> None:
    adapter = HandPlayAdapter()
    assert isinstance(adapter, GameAdapter)


def test_reset_lands_in_selecting_hand() -> None:
    adapter = HandPlayAdapter()
    state = adapter.reset(BACK, STAKE, SEED)

    assert state.phase == GamePhase.SELECTING_HAND
    assert adapter.raw_state["phase"] == GamePhase.SELECTING_HAND
    assert state.hands_left >= 1
    assert state.discards_left >= 0
    assert not adapter.done


def test_reset_applies_sampled_ante_hands_discards_money() -> None:
    cfg = HandPlayConfig(
        ante_range=(3, 3),
        hands_range=(2, 2),
        discards_range=(1, 1),
        dollars_range=(20, 20),
        blind_stages=("Small",),
    )
    adapter = HandPlayAdapter(cfg)
    state = adapter.reset(BACK, STAKE, SEED)

    assert state.ante == 3
    assert state.hands_left == 2
    assert state.discards_left == 1
    assert state.dollars == 20
    assert state.blind_on_deck == "Small"


def test_reset_injects_jokers() -> None:
    cfg = HandPlayConfig(
        joker_pool=("j_joker", "j_greedy_joker", "j_lusty_joker"),
        joker_count_range=(2, 2),
    )
    adapter = HandPlayAdapter(cfg)
    adapter.reset(BACK, STAKE, SEED)

    jokers = adapter.raw_state.get("jokers", [])
    assert len(jokers) == 2


def test_reset_is_deterministic_given_same_seed() -> None:
    adapter_a = HandPlayAdapter()
    adapter_b = HandPlayAdapter()

    state_a = adapter_a.reset(BACK, STAKE, SEED)
    state_b = adapter_b.reset(BACK, STAKE, SEED)

    assert state_a == state_b
    hand_a = [(c.base.suit, c.base.rank) for c in adapter_a.raw_state["hand"]]
    hand_b = [(c.base.suit, c.base.rank) for c in adapter_b.raw_state["hand"]]
    assert hand_a == hand_b


def test_no_blind_select_or_shop_phase_reached() -> None:
    adapter = HandPlayAdapter()
    adapter.reset(BACK, STAKE, SEED)

    seen_phases = {adapter.raw_state["phase"]}
    steps = 0
    while not adapter.done and steps < 200:
        _random_agent_step(adapter)
        seen_phases.add(adapter.raw_state["phase"])
        steps += 1

    assert GamePhase.BLIND_SELECT not in seen_phases
    assert GamePhase.SHOP not in seen_phases
    assert adapter.done


def test_runs_to_win_or_loss() -> None:
    adapter = HandPlayAdapter()
    adapter.reset(BACK, STAKE, SEED)

    steps = 0
    while not adapter.done and steps < 200:
        _random_agent_step(adapter)
        steps += 1

    assert adapter.done
    phase = adapter.raw_state["phase"]
    assert phase in (GamePhase.ROUND_EVAL, GamePhase.GAME_OVER)
    assert adapter.won == (phase == GamePhase.ROUND_EVAL)
