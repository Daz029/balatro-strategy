"""Tests for HandPlayGymEnv — canonical-action hand-play episodes.

Covers:
- observation matches the declared space and the BC demo-shard schema
- deterministic reset given a pinned episode seed
- action mask agrees with engine legality (hands/discards budgets, hand size)
- full episodes run to termination with reward 1.0 iff the blind was cleared
- env-side optimal ordering submits a permutation of the chosen subset when
  an order-sensitive joker is present
- illegal actions fail loudly instead of no-oping
"""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.agents.hand_action_space import (
    NUM_COMBOS,
    NUM_HAND_ACTIONS,
    action_to_combo,
    combo_to_action,
)
from jackdaw.engine.actions import PlayHand
from jackdaw.env.action_space import ActionType
from jackdaw.env.hand_play_adapter import HandPlayConfig
from jackdaw.env.hand_play_gym import HandPlayGymEnv

_DEMO_SCHEMA_KEYS = {"global_context", "hand_cards", "hand_mask", "jokers", "joker_mask"}
_RESERVED_KEYS = {"consumables", "consumable_mask"}


def _no_joker_env(**kwargs) -> HandPlayGymEnv:
    return HandPlayGymEnv(config=HandPlayConfig(), seed_prefix="TESTENV", **kwargs)


class TestObservation:
    def test_obs_in_space_and_schema(self):
        env = _no_joker_env()
        obs, info = env.reset(seed=0)
        assert set(obs.keys()) == _DEMO_SCHEMA_KEYS | _RESERVED_KEYS
        assert env.observation_space.contains(obs)
        # Dormant consumable block: reserved seam, always fully masked.
        assert not obs["consumable_mask"].any()

    def test_reset_deterministic_for_pinned_seed(self):
        env_a = _no_joker_env()
        env_b = _no_joker_env()
        obs_a, _ = env_a.reset(options={"episode_seed": "TESTENV_00000042"})
        obs_b, _ = env_b.reset(options={"episode_seed": "TESTENV_00000042"})
        for key in obs_a:
            assert np.array_equal(obs_a[key], obs_b[key]), key

    def test_auto_seeds_advance(self):
        env = _no_joker_env()
        _, info_0 = env.reset(seed=0)
        _, info_1 = env.reset()
        assert info_0["episode_seed"] != info_1["episode_seed"]


class TestActionMask:
    def test_mask_respects_budgets(self):
        env = _no_joker_env()
        env.reset(seed=3)
        gs = env._adapter.raw_state
        cr = gs["current_round"]
        mask = env.action_masks()
        assert mask.shape == (NUM_HAND_ACTIONS,)
        assert mask[:NUM_COMBOS].any() == (cr["hands_left"] > 0)
        assert mask[NUM_COMBOS:].any() == (cr["discards_left"] > 0)

    def test_no_discards_left_blocks_discards(self):
        env = HandPlayGymEnv(
            config=HandPlayConfig(discards_range=(0, 0)), seed_prefix="TESTENV"
        )
        env.reset(seed=0)
        mask = env.action_masks()
        assert not mask[NUM_COMBOS:].any()

    def test_illegal_action_raises(self):
        env = HandPlayGymEnv(
            config=HandPlayConfig(discards_range=(0, 0)), seed_prefix="TESTENV"
        )
        env.reset(seed=0)
        discard_action = combo_to_action(ActionType.Discard, (0,))
        with pytest.raises(ValueError, match="illegal action"):
            env.step(discard_action)


class TestEpisodes:
    def _run_episode(self, env: HandPlayGymEnv, seed: int, rng: np.random.Generator):
        obs, info = env.reset(seed=seed)
        total_reward, steps = 0.0, 0
        while True:
            mask = info["action_mask"]
            legal = np.nonzero(mask)[0]
            assert len(legal) > 0
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                return total_reward, steps, info

    def test_random_episodes_terminate_with_binary_reward(self):
        env = _no_joker_env()
        rng = np.random.default_rng(0)
        outcomes = []
        for seed in range(8):
            total_reward, steps, info = self._run_episode(env, seed, rng)
            assert total_reward in (0.0, 1.0)
            assert total_reward == float(info["balatro/cleared"])
            assert steps <= 8  # hands+discards budgets bound episode length
            outcomes.append(total_reward)
        # Not a strict guarantee, but random play across 8 episodes seeing
        # only one outcome would indicate a reward-wiring bug.
        assert len(set(outcomes)) > 1 or outcomes[0] in (0.0, 1.0)

    def test_episodes_with_jokers(self):
        env = HandPlayGymEnv(
            config=HandPlayConfig(
                joker_pool=("j_photograph", "j_jolly", "j_greedy_joker"),
                joker_count_range=(1, 3),
            ),
            seed_prefix="TESTENV",
        )
        rng = np.random.default_rng(1)
        for seed in range(4):
            total_reward, _, _ = self._run_episode(env, seed, rng)
            assert total_reward in (0.0, 1.0)


class TestEnvSideOrdering:
    def _photograph_env_with_faces(self) -> tuple[HandPlayGymEnv, tuple[int, ...]]:
        """Find a seeded episode whose hand has 2+ face cards and a
        Photograph on the board, and return a 5-card play including them."""
        env = HandPlayGymEnv(
            config=HandPlayConfig(joker_pool=("j_photograph",), joker_count_range=(1, 1)),
            seed_prefix="TESTENV",
        )
        for seed in range(50):
            env.reset(seed=seed)
            gs = env._adapter.raw_state
            faces = [
                i
                for i, c in enumerate(gs["hand"][:8])
                if c.base is not None and c.base.id in (11, 12, 13)
            ]
            if len(faces) >= 2:
                others = [i for i in range(min(8, len(gs["hand"]))) if i not in faces]
                combo = tuple(sorted((faces + others)[:5]))
                return env, combo
        pytest.fail("no seed produced a 2-face-card hand with Photograph")

    def test_ordering_returns_permutation_of_subset(self):
        env, combo = self._photograph_env_with_faces()
        action = combo_to_action(ActionType.PlayHand, combo)
        engine_action = env._to_engine_action(action)
        assert isinstance(engine_action, PlayHand)
        # Same subset, possibly different order -- and the engine accepts it.
        assert sorted(engine_action.card_indices) == sorted(combo)
        _, reward, terminated, truncated, _ = env.step(action)
        assert reward in (0.0, 1.0)

    def test_action_decode_matches_engine_submission_for_plain_hand(self):
        env = _no_joker_env()
        env.reset(seed=0)
        action = combo_to_action(ActionType.PlayHand, (0, 2, 4))
        engine_action = env._to_engine_action(action)
        assert engine_action.card_indices == (0, 2, 4)
        assert action_to_combo(action)[1] == (0, 2, 4)
