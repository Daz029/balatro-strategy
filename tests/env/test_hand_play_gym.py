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

import copy

import numpy as np
import pytest
from gymnasium import spaces

from jackdaw.agents.hand_action_space import (
    NUM_COMBOS,
    NUM_HAND_ACTIONS,
    action_to_combo,
    combo_to_action,
)
from jackdaw.engine.actions import PlayHand
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.env.action_space import ActionType
from jackdaw.env.hand_play_adapter import HandPlayConfig
from jackdaw.env.hand_play_gym import (
    MAX_CONSUMABLES_V2,
    MAX_HAND_CARDS_OBS,
    MAX_JOKERS,
    MAX_JOKERS_V2,
    HandPlayGymEnv,
    build_observation,
    build_observation_v2,
    observation_space,
    observation_space_v2,
)
from jackdaw.env.observation import (
    D_GLOBAL,
    D_HAND_CARD,
    D_HAND_GLOBAL,
    center_key_id,
    encode_hand_potential,
)

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


class TestHandOverflow:
    """The Serpent (and +hand-size effects) legitimately grow the hand past
    the action space's 8 positions. The obs block is MAX_HAND_CARDS_OBS=12
    wide and must encode 9-12 cards, truncating beyond -- never raise."""

    def _grow_hand(self, env: HandPlayGymEnv, target: int) -> dict:
        gs = env._adapter.raw_state
        deck = gs["deck"]
        while len(gs["hand"]) < target and deck:
            gs["hand"].append(deck.pop())
        return gs

    def test_oversized_hand_is_encodable(self):
        env = _no_joker_env()
        env.reset(seed=0)
        gs = self._grow_hand(env, MAX_HAND_CARDS_OBS)
        obs = build_observation(gs)
        assert env.observation_space.contains(obs)
        assert int(obs["hand_mask"].sum()) == MAX_HAND_CARDS_OBS

    def test_hand_beyond_obs_width_truncates(self):
        env = _no_joker_env()
        env.reset(seed=0)
        gs = self._grow_hand(env, MAX_HAND_CARDS_OBS + 3)
        assert len(gs["hand"]) > MAX_HAND_CARDS_OBS
        obs = build_observation(gs)
        assert env.observation_space.contains(obs)
        # Truncated, fully-masked hand block -- and rows stay index-aligned
        # with action positions (row i is hand[i]).
        assert int(obs["hand_mask"].sum()) == MAX_HAND_CARDS_OBS

    def test_serpent_overdraw_episode_is_encodable(self):
        """Regression for the stage4 eval blocker: under The Serpent a
        discard of fewer than 3 cards grows the hand past 8 mid-round;
        build_observation used to raise 'entity count 9 exceeds max 8'."""
        env = HandPlayGymEnv(
            config=HandPlayConfig(
                blind_stages=("Boss",), ante_range=(5, 8), discards_range=(2, 3)
            ),
            seed_prefix="TESTENV",
        )
        for seed in range(300):
            obs, info = env.reset(seed=seed)
            if getattr(env._adapter.raw_state["blind"], "name", "") != "The Serpent":
                continue
            grew = False
            for _ in range(2):  # 2nd discard is always Serpent-drawn (+3)
                action = combo_to_action(ActionType.Discard, (0,))
                if not env.action_masks()[action]:
                    break
                obs, _, terminated, truncated, info = env.step(action)
                assert env.observation_space.contains(obs)
                if len(env._adapter.raw_state["hand"]) > 8:
                    grew = True
                if terminated or truncated:
                    break
            if grew:
                return
        pytest.fail("no seed in range produced a growable Serpent hand")


class TestJokerOverflow:
    """Negative-edition jokers don't consume a slot, and slot-expanding
    vouchers raise joker_slots, so a full-run state legitimately holds >5
    physical jokers. The width-5 hand-obs joker block (frozen BC demo schema,
    widened only at the shop merge) truncates such states -- an informativeness
    gap, not a crash. A non-negative overfill stays a loud engine bug. This is
    the s0 blocker: the shop agent buys a Negative joker, an auto-resolved hand
    phase calls build_observation via HandCheckpointPolicy, and it used to raise
    'entity count 6 exceeds max 5'."""

    def _joker_state(self, n: int, joker_slots: int, negatives: int = 0) -> tuple:
        env = HandPlayGymEnv(
            config=HandPlayConfig(joker_pool=("j_jolly",), joker_count_range=(1, 1)),
            seed_prefix="TESTENV",
        )
        env.reset(seed=0)
        gs = env._adapter.raw_state
        base = gs["jokers"][0]
        gs["jokers"] = [copy.deepcopy(base) for _ in range(n)]
        gs["joker_slots"] = joker_slots
        for j in range(negatives):
            gs["jokers"][n - 1 - j].edition = {"negative": True}
        return env, gs

    def test_negative_excess_truncates(self):
        # 6 physical jokers in 5 slots, but 2 are Negative -> legal build.
        env, gs = self._joker_state(n=6, joker_slots=5, negatives=2)
        obs = build_observation(gs)
        assert env.observation_space.contains(obs)
        assert int(obs["joker_mask"].sum()) == MAX_JOKERS  # v1 truncates to 5 (frozen)

    def test_expanded_slots_truncates(self):
        # joker_slots raised to 7 by a voucher -> 6 non-negative jokers legal.
        env, gs = self._joker_state(n=6, joker_slots=7)
        obs = build_observation(gs)
        assert env.observation_space.contains(obs)
        assert int(obs["joker_mask"].sum()) == 5

    def test_nonnegative_overfill_raises(self):
        # 6 base-edition jokers in 5 slots with no negatives -> Riff-raff-class
        # overfill, still loud.
        _env, gs = self._joker_state(n=6, joker_slots=5)
        with pytest.raises(ValueError, match="overfill"):
            build_observation(gs)


class TestJokerOverflowV2:
    """v2 (h1 schema) raises the joker cap to MAX_JOKERS_V2=15 and EXPANDS
    rather than truncates: a legal wide/negative build is encoded in full, so
    the h1 model (and the harvest labeler that shares this discipline) sees
    every real joker instead of dropping negatives. The genuine-overfill raise
    (dual counter: non-negative jokers > joker_slots) is unchanged."""

    def _joker_state(self, n: int, joker_slots: int, negatives: int = 0) -> tuple:
        env = HandPlayGymEnv(
            config=HandPlayConfig(joker_pool=("j_jolly",), joker_count_range=(1, 1)),
            seed_prefix="TESTENV",
            obs_version=2,
        )
        env.reset(seed=0)
        gs = env._adapter.raw_state
        base = gs["jokers"][0]
        gs["jokers"] = [copy.deepcopy(base) for _ in range(n)]
        gs["joker_slots"] = joker_slots
        for j in range(negatives):
            gs["jokers"][n - 1 - j].edition = {"negative": True}
        return env, gs

    def test_negative_excess_expands_not_truncates(self):
        # 6 physical jokers in 5 slots, 2 Negative -> legal, and v2 keeps ALL 6
        # (v1 would truncate to 5). This is the whole point of the wider cap.
        env, gs = self._joker_state(n=6, joker_slots=5, negatives=2)
        obs = build_observation_v2(gs)
        assert env.observation_space.contains(obs)
        assert int(obs["joker_mask"].sum()) == 6
        # Parallel id / copy / trigger blocks stay index-aligned at full width.
        assert obs["jokers"].shape[0] == MAX_JOKERS_V2
        assert obs["joker_ids"].shape == (MAX_JOKERS_V2,)
        assert obs["trigger_match"].shape[1] == MAX_JOKERS_V2
        assert int((obs["joker_ids"] != 0).sum()) == 6

    def test_expanded_slots_expands(self):
        # joker_slots raised to 7 by a voucher -> 6 non-negative jokers legal,
        # all encoded.
        env, gs = self._joker_state(n=6, joker_slots=7)
        obs = build_observation_v2(gs)
        assert env.observation_space.contains(obs)
        assert int(obs["joker_mask"].sum()) == 6

    def test_beyond_cap_truncates_tail(self):
        # 17 physical jokers (12 Negative in 5 slots) exceeds the 15 cap ->
        # truncates lowest-slot-first as the pure safety valve, never raises.
        env, gs = self._joker_state(n=17, joker_slots=5, negatives=12)
        obs = build_observation_v2(gs)
        assert env.observation_space.contains(obs)
        assert int(obs["joker_mask"].sum()) == MAX_JOKERS_V2

    def test_nonnegative_overfill_still_raises(self):
        # 6 base-edition jokers in 5 slots, no negatives -> genuine engine bug.
        _env, gs = self._joker_state(n=6, joker_slots=5)
        with pytest.raises(ValueError, match="overfill"):
            build_observation_v2(gs)


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


class TestObservationV2:
    """Schema v2 (h1 bump, B2 slice 4): a versioned seam next to v1, never a
    replacement -- v1 is the frozen h0.5 checkpoint obs and must stay the
    default (see docs/pre-regen-handoff.md, slice-4 sequencing flag)."""

    def test_default_env_stays_v1(self):
        env = _no_joker_env()
        obs, _ = env.reset(seed=0)
        assert set(obs.keys()) == _DEMO_SCHEMA_KEYS | _RESERVED_KEYS
        assert env.observation_space == observation_space()
        assert obs["global_context"].shape == (D_GLOBAL,)

    def test_unknown_obs_version_raises(self):
        with pytest.raises(ValueError, match="obs_version"):
            _no_joker_env(obs_version=3)

    def test_v2_obs_in_space(self):
        env = _no_joker_env(obs_version=2)
        obs, _ = env.reset(seed=0)
        assert env.observation_space == observation_space_v2()
        assert env.observation_space.contains(obs)
        assert obs["global_context"].shape == (D_HAND_GLOBAL,)
        assert obs["hand_cards"].shape == (MAX_HAND_CARDS_OBS, D_HAND_CARD)
        assert obs["trigger_match"].shape == (MAX_HAND_CARDS_OBS, MAX_JOKERS_V2, 2)
        assert obs["consumables"].shape[0] == MAX_CONSUMABLES_V2
        # HandPlayAdapter injects no consumables at this stage.
        assert not obs["consumable_mask"].any()

    def test_v2_episode_steps_in_space(self):
        env = _no_joker_env(obs_version=2)
        obs, info = env.reset(seed=1)
        rng = np.random.default_rng(0)
        for _ in range(8):
            legal = np.nonzero(info["action_mask"])[0]
            obs, _, terminated, truncated, info = env.step(int(rng.choice(legal)))
            assert env.observation_space.contains(obs)
            if terminated or truncated:
                break

    def test_v2_potential_features_wired(self):
        env = _no_joker_env(obs_version=2)
        obs, _ = env.reset(seed=2)
        gs = env._adapter.raw_state
        per_card, gc_ext = encode_hand_potential(gs)
        n = len(gs["hand"])
        assert np.allclose(obs["hand_cards"][:n, 15:], per_card)
        assert np.allclose(obs["global_context"][D_GLOBAL:], gc_ext)
        # The v1 feature prefix is untouched by the append.
        v1_obs = build_observation(gs)
        assert np.allclose(obs["hand_cards"][:n, :15], v1_obs["hand_cards"][:n])
        assert np.allclose(obs["global_context"][:D_GLOBAL], v1_obs["global_context"])

    def test_v2_trigger_match_wired(self):
        env = HandPlayGymEnv(
            config=HandPlayConfig(joker_pool=("j_greedy_joker",), joker_count_range=(1, 1)),
            seed_prefix="TESTENV",
            obs_version=2,
        )
        obs, _ = env.reset(seed=0)
        gs = env._adapter.raw_state
        assert obs["joker_ids"][0] == center_key_id("j_greedy_joker")
        for i, card in enumerate(gs["hand"][:MAX_HAND_CARDS_OBS]):
            expected = card.is_suit("Diamonds")
            assert obs["trigger_match"][i, 0, 0] == float(expected), i

    def test_v2_copy_fields_wired(self):
        env = HandPlayGymEnv(
            config=HandPlayConfig(joker_pool=("j_photograph",), joker_count_range=(1, 1)),
            seed_prefix="TESTENV",
            obs_version=2,
        )
        env.reset(seed=0)
        gs = env._adapter.raw_state
        gs["jokers"].insert(0, create_joker("j_blueprint"))
        obs = build_observation_v2(gs)
        # Blueprint copies its right neighbor (Photograph): active bit set,
        # resolved-target id stored, match column inherited (faces marked).
        assert obs["copy_active"][0] == 1.0
        assert obs["copy_target_ids"][0] == center_key_id("j_photograph")
        assert obs["copy_active"][1] == 0.0  # Photograph itself is not a copy
        faces = [
            i
            for i, c in enumerate(gs["hand"][:MAX_HAND_CARDS_OBS])
            if c.base is not None and c.base.id in (11, 12, 13)
        ]
        for i in faces:
            assert obs["trigger_match"][i, 0, 0] == 1.0
            assert obs["trigger_match"][i, 1, 0] == 1.0

    def test_v2_consumables_encoded(self):
        env = _no_joker_env(obs_version=2)
        env.reset(seed=0)
        gs = env._adapter.raw_state
        gs["consumables"] = [create_consumable("c_magician"), create_consumable("c_pluto")]
        obs = build_observation_v2(gs)
        assert int(obs["consumable_mask"].sum()) == 2
        assert obs["consumables"][0].any() and obs["consumables"][1].any()
        assert not obs["consumables"][2:].any()

    def test_v2_consumable_overflow_truncates(self):
        # Perkeo-style negative copies can exceed any slot count; the block
        # tail-truncates rather than raising (rows are engine slot order).
        env = _no_joker_env(obs_version=2)
        env.reset(seed=0)
        gs = env._adapter.raw_state
        gs["consumables"] = [create_consumable("c_pluto") for _ in range(MAX_CONSUMABLES_V2 + 3)]
        obs = build_observation_v2(gs)
        assert int(obs["consumable_mask"].sum()) == MAX_CONSUMABLES_V2
        assert env.observation_space.contains(obs)


def test_pointer_action_version_requires_v2_observations_and_has_no_env_mask():
    with pytest.raises(ValueError, match="action_version=2 requires obs_version=2"):
        _no_joker_env(action_version=2)
    env = _no_joker_env(obs_version=2, action_version=2)
    _obs, info = env.reset(seed=0)
    assert env.action_space == spaces.MultiDiscrete([2] + [41] * 5)
    assert "action_mask" not in info
    with pytest.raises(AttributeError, match="no env-side action masks"):
        env.action_masks()


def test_pointer_action_version_matches_v1_execution_for_same_seed():
    env_v1 = HandPlayGymEnv(config=HandPlayConfig(), seed_prefix="POINTER_EQ")
    env_v2 = HandPlayGymEnv(
        config=HandPlayConfig(),
        obs_version=2,
        action_version=2,
        seed_prefix="POINTER_EQ",
    )
    env_v1.reset(seed=7)
    env_v2.reset(seed=7)
    v1_result = env_v1.step(combo_to_action(ActionType.PlayHand, (0,)))
    pointer_action = np.array([0, 0, 40, 40, 40, 40], dtype=np.int64)
    v2_result = env_v2.step(pointer_action)
    assert v2_result[1:4] == v1_result[1:4]


@pytest.mark.parametrize(
    "action,match",
    [
        ([0, 1, 0, 40, 40, 40], "strictly ascending"),
        ([0, 40, 40, 40, 40, 40], "at least one card"),
        ([0, 39, 40, 40, 40, 40], "dead hand index"),
    ],
)
def test_pointer_action_version_rejects_invalid_vectors(action, match):
    env = HandPlayGymEnv(
        config=HandPlayConfig(),
        obs_version=2,
        action_version=2,
        seed_prefix="POINTER_BAD",
    )
    env.reset(seed=0)
    with pytest.raises(ValueError, match=match):
        env.step(np.asarray(action, dtype=np.int64))


def test_pointer_action_version_rejects_type_without_budget():
    env = HandPlayGymEnv(
        config=HandPlayConfig(discards_range=(0, 0)),
        seed_prefix="POINTER_BAD_TYPE",
        obs_version=2,
        action_version=2,
    )
    env.reset(seed=0)
    with pytest.raises(ValueError, match="no remaining budget"):
        env.step(np.array([1, 0, 40, 40, 40, 40], dtype=np.int64))
