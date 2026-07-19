"""Tests for HandPlayGymEnv snapshot starts and reservoir sampling."""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.env.hand_play_adapter import HandPlayConfig
from jackdaw.env.hand_play_gym import HandPlayGymEnv


def _first_legal_action(env: HandPlayGymEnv) -> int:
    legal = np.flatnonzero(env.action_masks())
    assert len(legal) > 0
    return int(legal[0])


def _assert_observations_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def test_restore_continuation_is_rng_exact_in_a_fresh_env() -> None:
    config = HandPlayConfig(hands_range=(4, 4), discards_range=(3, 3))
    source = HandPlayGymEnv(config=config)
    source.reset(options={"episode_seed": "RESTORE_CONTINUATION"})
    for _ in range(2):
        source.step(_first_legal_action(source))
    snapshot_obs = source._build_obs(source._adapter.raw_state)
    blob = source._adapter.snapshot_state()

    expected: list[tuple[int, dict, float, bool, bool]] = []
    while not source._adapter.done:
        action = _first_legal_action(source)
        obs, reward, terminated, truncated, _ = source.step(action)
        expected.append((action, obs, reward, terminated, truncated))
        if terminated or truncated:
            break

    restored = HandPlayGymEnv(config=config)
    initial, info = restored.reset(options={"snapshot": blob})
    assert info["episode_seed"] == "<restored>"
    _assert_observations_equal(initial, snapshot_obs)

    for action, expected_obs, expected_reward, expected_terminated, expected_truncated in expected:
        obs, reward, terminated, truncated, _ = restored.step(action)
        _assert_observations_equal(obs, expected_obs)
        assert reward == expected_reward
        assert terminated == expected_terminated
        assert truncated == expected_truncated


def test_sampler_blob_and_none_follow_reset_contract() -> None:
    source = HandPlayGymEnv(seed_prefix="SOURCE")
    source.reset(options={"episode_seed": "SAMPLER_SOURCE"})
    blob = source._adapter.snapshot_state()

    sampler_calls = 0

    def sampler() -> bytes:
        nonlocal sampler_calls
        sampler_calls += 1
        return blob

    restored = HandPlayGymEnv(start_state_sampler=sampler)
    _, info = restored.reset(options={"snapshot": blob})
    assert info["episode_seed"] == "<restored>"
    assert sampler_calls == 0
    counter = restored._episode_counter
    _, info = restored.reset()
    assert info["episode_seed"] == "<restored>"
    assert sampler_calls == 1
    assert restored._episode_counter == counter

    fresh = HandPlayGymEnv(start_state_sampler=lambda: None, seed_prefix="SAMPLER")
    _, info_0 = fresh.reset(seed=0)
    _, info_1 = fresh.reset()
    assert info_0["episode_seed"] == "SAMPLER_00000000"
    assert info_1["episode_seed"] == "SAMPLER_00000001"


def test_snapshot_restore_requires_selecting_hand_phase() -> None:
    terminal = HandPlayGymEnv(config=HandPlayConfig(hands_range=(1, 1)))
    terminal.reset(options={"episode_seed": "RESTORE_TERMINAL"})
    while not terminal._adapter.done:
        terminal.step(_first_legal_action(terminal))
    terminal_blob = terminal._adapter.snapshot_state()

    with pytest.raises(ValueError, match="SELECTING_HAND"):
        HandPlayGymEnv().reset(options={"snapshot": terminal_blob})

    non_selecting = HandPlayGymEnv()
    non_selecting.reset(options={"episode_seed": "RESTORE_PHASE"})
    non_selecting._adapter.raw_state["phase"] = GamePhase.BLIND_SELECT
    phase_blob = non_selecting._adapter.snapshot_state()

    with pytest.raises(ValueError, match="SELECTING_HAND"):
        HandPlayGymEnv().reset(options={"snapshot": phase_blob})


def test_restored_consumables_are_v2_input_but_v1_padding() -> None:
    source = HandPlayGymEnv(obs_version=2)
    source.reset(options={"episode_seed": "RESTORE_CONSUMABLES"})
    source._adapter.raw_state["consumables"] = [
        create_consumable("c_pluto"),
        create_consumable("c_magician"),
    ]
    source._adapter.raw_state["dollars"] = -7
    source._adapter.raw_state["chips"] = 321
    source._adapter.raw_state["current_round"]["discards_left"] = 1
    blob = source._adapter.snapshot_state()

    v2 = HandPlayGymEnv(obs_version=2)
    obs, _ = v2.reset(options={"snapshot": blob})
    gs = v2._adapter.raw_state
    assert gs["dollars"] == -7
    assert gs["chips"] == 321
    assert gs["current_round"]["discards_left"] == 1
    assert int(obs["consumable_mask"].sum()) == 2
    assert np.all(np.any(obs["consumables"][:2] != 0.0, axis=1))
    v2.step(_first_legal_action(v2))

    v1 = HandPlayGymEnv(obs_version=1)
    obs, _ = v1.reset(options={"snapshot": blob})
    assert not obs["consumable_mask"].any()
    assert not obs["consumables"].any()


def test_restored_negative_joker_wide_build_truncates_only_in_v1() -> None:
    source = HandPlayGymEnv(obs_version=2)
    source.reset(options={"episode_seed": "RESTORE_WIDE_JOKERS"})
    source._adapter.raw_state["jokers"] = [
        create_joker("j_joker", {"negative": True}) for _ in range(6)
    ]
    blob = source._adapter.snapshot_state()

    v2 = HandPlayGymEnv(obs_version=2)
    obs, _ = v2.reset(options={"snapshot": blob})
    assert int(obs["joker_mask"].sum()) == 6

    v1 = HandPlayGymEnv(obs_version=1)
    obs, _ = v1.reset(options={"snapshot": blob})
    assert int(obs["joker_mask"].sum()) == 5
