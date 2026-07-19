"""Tests for the optional terminal dollar-value term."""

from __future__ import annotations

import numpy as np

import jackdaw.env.hand_play_gym as hand_play_gym
from jackdaw.agents.hand_action_space import combo_to_action
from jackdaw.engine.hand_eval import get_best_hand
from jackdaw.env.action_space import ActionType
from jackdaw.env.cashout_mirror import dollars_after_cashout
from jackdaw.env.hand_play_adapter import HandPlayConfig
from jackdaw.env.hand_play_gym import HandPlayGymEnv


class RecordingVCurve:
    def __init__(self, term: float) -> None:
        self.term = term
        self.calls: list[tuple[int, int]] = []

    def value(self, ante: int, dollars: int) -> float:
        self.calls.append((ante, dollars))
        return self.term


def _winning_config() -> HandPlayConfig:
    return HandPlayConfig(
        ante_range=(1, 1),
        hands_range=(4, 4),
        discards_range=(3, 3),
        blind_stages=("Small",),
    )


def _losing_config() -> HandPlayConfig:
    return HandPlayConfig(
        ante_range=(8, 8),
        hands_range=(1, 1),
        discards_range=(0, 0),
        blind_stages=("Small",),
    )


def _best_hand_action(env: HandPlayGymEnv) -> int | np.ndarray:
    gs = env._adapter.raw_state
    _hand_name, cards, _results = get_best_hand(gs["hand"])
    index_by_identity = {id(card): index for index, card in enumerate(gs["hand"])}
    combo = tuple(sorted(index_by_identity[id(card)] for card in cards))
    if env.action_version == 1:
        return combo_to_action(ActionType.PlayHand, combo)
    return np.asarray([0, *combo, *([40] * (5 - len(combo)))], dtype=np.int64)


def _run_to_terminal(env: HandPlayGymEnv, seed: int) -> tuple[float, bool, bool, dict]:
    env.reset(seed=seed)
    while True:
        _obs, reward, terminated, truncated, info = env.step(_best_hand_action(env))
        if terminated or truncated:
            return reward, terminated, truncated, info


class TestTerminalDollarTerm:
    def test_won_episode_uses_terminal_ante_and_cashout_dollars(self):
        curve = RecordingVCurve(0.375)
        env = HandPlayGymEnv(config=_winning_config(), seed_prefix="DISCOVER", v_curve=curve)

        reward, terminated, truncated, info = _run_to_terminal(env, seed=2)

        terminal_state = env._adapter.raw_state
        expected_dollars = dollars_after_cashout(terminal_state)
        expected_ante = terminal_state["round_resets"]["ante"]
        assert terminated and not truncated
        assert reward == 1.0 + curve.term
        assert curve.calls == [(expected_ante, expected_dollars)]
        assert info["balatro/v_curve_term"] == curve.term
        assert info["balatro/dollars_after_cashout"] == expected_dollars

    def test_won_pointer_episode_gets_the_same_terminal_term(self):
        curve = RecordingVCurve(0.25)
        env = HandPlayGymEnv(
            config=_winning_config(),
            seed_prefix="DISCOVER",
            obs_version=2,
            action_version=2,
            v_curve=curve,
        )

        reward, terminated, truncated, info = _run_to_terminal(env, seed=2)

        assert terminated and not truncated
        assert reward == 1.0 + curve.term
        assert info["balatro/v_curve_term"] == curve.term
        assert len(curve.calls) == 1

    def test_lost_episode_does_not_call_curve_or_mirror(self):
        curve = RecordingVCurve(0.375)
        env = HandPlayGymEnv(config=_losing_config(), seed_prefix="DISCOVER", v_curve=curve)

        reward, terminated, truncated, info = _run_to_terminal(env, seed=0)

        assert terminated and not truncated
        assert reward == 0.0
        assert curve.calls == []
        assert info["balatro/v_curve_term"] == 0.0
        assert "balatro/dollars_after_cashout" not in info

    def test_default_does_not_invoke_mirror(self, monkeypatch):
        def fail_if_called(_gs):
            raise AssertionError("cash-out mirror must stay unused without a V_curve")

        monkeypatch.setattr(hand_play_gym, "dollars_after_cashout", fail_if_called)
        env = HandPlayGymEnv(config=_winning_config(), seed_prefix="DISCOVER")

        reward, terminated, truncated, info = _run_to_terminal(env, seed=2)

        assert terminated and not truncated
        assert reward == 1.0
        assert info["balatro/v_curve_term"] == 0.0
        assert "balatro/dollars_after_cashout" not in info
