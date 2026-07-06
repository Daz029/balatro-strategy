"""Tests for ShopGymEnv — the Discrete(686) shop env with pending targeting.

Covers: gym contract + masks, the two-step pending-target state machine
(carrier -> observable pending state -> SelectTarget completion), env-side
pack-row legality gating (the engine applies picks unvalidated), reward
components in info, snapshot/restore round-trip including pending state,
and a full win_ante=1 horizon episode.
"""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.agents.shop_action_space import (
    FAMILY_OFFSETS,
    NUM_COMBOS,
    NUM_TOTAL_ACTIONS,
    SELECT_TARGET_BASE,
    ShopActionFamily,
    shop_action,
    target_combo_for_action,
)
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.env.shop_gym import (
    ShopGymEnv,
    blind_clear_bonus,
    consumable_target_info,
    pack_row_legal,
)
from jackdaw.env.shop_obs import build_shop_observation
from jackdaw.env.shop_run_adapter import ShopRunConfig


def _first_legal(mask: np.ndarray) -> int:
    legal = np.flatnonzero(mask)
    assert legal.size > 0
    return int(legal[0])


def _find_pending_pack_env() -> tuple[ShopGymEnv, int]:
    """Probe seeds until opening a booster yields an Arcana/Spectral pack
    containing a legal, target-needing card. Deterministic engine -> the
    first hit is stable forever; returns (env in pending state, slot).
    """
    for i in range(40):
        # Booster slot 0 is the deterministic first-shop Buffoon pack;
        # Arcana/Spectral packs show up in slot 1.
        for booster_slot in (1, 0):
            env = ShopGymEnv()
            env.reset(options={"episode_seed": f"SHOPGYM_{i:02d}"})
            env._adapter.raw_state["dollars"] = 50
            open_action = shop_action(ShopActionFamily.OpenBooster, booster_slot)
            if not env.action_masks()[open_action]:
                continue
            env.step(open_action)
            gs = env._adapter.raw_state
            if gs.get("pack_type") not in ("Arcana", "Spectral"):
                continue
            for slot, card in enumerate(gs.get("pack_cards", [])):
                _, _, needs = consumable_target_info(card)
                if needs and pack_row_legal(card, gs):
                    pick = shop_action(ShopActionFamily.PickPackCard, slot)
                    assert env.action_masks()[pick]
                    env.step(pick)
                    return env, slot
    raise AssertionError("no target-needing pack card found in 40 probe seeds")


class TestGymContract:
    def test_reset_matches_spaces(self):
        env = ShopGymEnv()
        obs, info = env.reset(options={"episode_seed": "SHOPGYM_CONTRACT"})
        assert env.action_space.n == NUM_TOTAL_ACTIONS
        assert env.observation_space.contains(obs)
        mask = info["action_mask"]
        assert mask.shape == (NUM_TOTAL_ACTIONS,)
        # A fresh shop always allows leaving.
        assert mask[shop_action(ShopActionFamily.NextRound)]
        # The hand block is permanently masked in s0.
        assert not mask[:436].any()

    def test_illegal_action_fails_loud(self):
        env = ShopGymEnv()
        _, info = env.reset(options={"episode_seed": "SHOPGYM_CONTRACT"})
        illegal = int(np.flatnonzero(~info["action_mask"])[0])
        with pytest.raises(ValueError, match="illegal action"):
            env.step(illegal)

    def test_buy_card_spends_dollars(self):
        env = ShopGymEnv()
        env.reset(options={"episode_seed": "SHOPGYM_CONTRACT"})
        gs = env._adapter.raw_state
        gs["dollars"] = 50
        buy_0 = shop_action(ShopActionFamily.BuyCard, 0)
        assert env.action_masks()[buy_0]
        cost = gs["shop_cards"][0].cost
        env.step(buy_0)
        assert gs["dollars"] == 50 - cost

    def test_truncation_counts_as_loss(self):
        env = ShopGymEnv(max_steps=1)
        _, info = env.reset(options={"episode_seed": "SHOPGYM_CONTRACT"})
        # Reroll/buy actions keep the episode in-shop; NextRound could
        # terminate it for real, so pick a non-NextRound legal action.
        mask = info["action_mask"]
        mask[shop_action(ShopActionFamily.NextRound)] = False
        _, reward, terminated, truncated, _ = env.step(_first_legal(mask))
        assert truncated and not terminated
        assert reward == 0.0


class TestRewardComponents:
    def test_bonus_normalization(self):
        assert sum(3 * blind_clear_bonus(a) for a in range(1, 9)) == pytest.approx(1.0)

    def test_next_round_emits_blind_bonus(self):
        env = ShopGymEnv()
        env.reset(options={"episode_seed": "SHOPRUN_B"})
        ante = env._adapter.raw_state["round_resets"]["ante"]
        _, reward, terminated, _, info = env.step(shop_action(ShopActionFamily.NextRound))
        rc = info["reward_components"]
        if terminated and not env._adapter.won:
            assert rc["blinds_cleared"] == 0
        else:
            assert rc["blinds_cleared"] == 1
            assert rc["blind_bonus"] == pytest.approx(blind_clear_bonus(ante))
        assert rc["win"] == reward

    def test_in_shop_actions_emit_zero_components(self):
        env = ShopGymEnv()
        env.reset(options={"episode_seed": "SHOPGYM_CONTRACT"})
        env._adapter.raw_state["dollars"] = 50
        _, reward, _, _, info = env.step(shop_action(ShopActionFamily.Reroll))
        assert reward == 0.0
        assert info["reward_components"]["blinds_cleared"] == 0
        assert info["reward_components"]["blind_bonus"] == 0.0


class TestFullEpisode:
    def test_win_ante_1_horizon(self):
        # SHOPRUN_B is pinned in the adapter tests as a greedy-policy win
        # at win_ante=1 when the shop policy just leaves every shop.
        env = ShopGymEnv(config=ShopRunConfig(win_ante=1))
        env.reset(options={"episode_seed": "SHOPRUN_B"})
        next_round = shop_action(ShopActionFamily.NextRound)

        total_bonus = 0.0
        cleared = 0
        for _ in range(20):
            mask = env.action_masks()
            action = next_round if mask[next_round] else _first_legal(mask)
            _, reward, terminated, truncated, info = env.step(action)
            total_bonus += info["reward_components"]["blind_bonus"]
            cleared += info["reward_components"]["blinds_cleared"]
            if terminated or truncated:
                break
        else:
            raise AssertionError("episode did not terminate")

        assert terminated and info["balatro/won"]
        assert reward == 1.0
        # The Small blind is auto-cleared during reset (no decision before
        # it), so the episode itself sees Big + Boss = 2 clears at ante 1.
        assert cleared == 2
        assert total_bonus == pytest.approx(2 * blind_clear_bonus(1))


class TestPackRowLegality:
    def _gs(self, n_jokers: int, hand: list | None = None) -> dict:
        return {
            "jokers": [create_joker("j_joker") for _ in range(n_jokers)],
            "joker_slots": 5,
            "consumables": [],
            "consumable_slots": 2,
            "hand": hand or [],
        }

    def test_joker_pick_needs_free_slot(self):
        card = create_joker("j_joker")
        assert pack_row_legal(card, self._gs(4))
        assert not pack_row_legal(card, self._gs(5))

    def test_negative_joker_bypasses_slots(self):
        card = create_joker("j_joker")
        card.set_edition({"negative": True})
        assert pack_row_legal(card, self._gs(5))

    def test_targeted_tarot_needs_hand_cards(self):
        magician = create_consumable("c_magician")
        assert not pack_row_legal(magician, self._gs(0))
        assert pack_row_legal(magician, self._gs(0, hand=[create_joker("j_joker")]))

    def test_planet_always_legal(self):
        assert pack_row_legal(create_consumable("c_pluto"), self._gs(5))

    def test_aura_special_case(self):
        assert consumable_target_info(create_consumable("c_aura")) == (1, 1, True)


class TestPendingTargetFlow:
    @pytest.fixture(scope="class")
    def pending_env(self) -> tuple[ShopGymEnv, int]:
        return _find_pending_pack_env()

    def test_carrier_enters_observable_pending_state(self, pending_env):
        env, slot = pending_env
        blob = env.snapshot()
        try:
            gs = env._adapter.raw_state
            obs = build_shop_observation(gs, env._pending)
            # Observable, not mask-only: flag + selected bit on the carrier.
            assert obs["shop_context"][7] == 1.0
            assert obs["pack_items"][slot, 14] == 1.0
            # The dealt pack_hand occupies the hand rows (targetable cards
            # live in the hand rows -- the merge invariant).
            assert obs["hand_mask"].sum() > 0

            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            assert legal.size > 0
            assert (legal >= SELECT_TARGET_BASE).all()
            assert (legal < SELECT_TARGET_BASE + NUM_COMBOS).all()
            # Every legal combo indexes only dealt cards.
            n_hand = len(gs["hand"])
            for a in legal:
                assert max(target_combo_for_action(int(a))) < n_hand
        finally:
            env.reset(options={"snapshot": blob})

    def test_select_target_completes_engine_action(self, pending_env):
        env, _ = pending_env
        blob = env.snapshot()
        try:
            gs = env._adapter.raw_state
            n_before = len(gs["pack_cards"])
            action = _first_legal(env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            assert reward == 0.0 and not terminated and not truncated
            assert env._pending is None
            assert obs["shop_context"][7] == 0.0
            # The engine consumed the pick: card gone (or the pack closed
            # and we are back in the shop).
            gs = env._adapter.raw_state
            assert len(gs.get("pack_cards", [])) < n_before or not gs.get("pack_cards")
        finally:
            env.reset(options={"snapshot": blob})

    def test_snapshot_restore_roundtrip_with_pending(self, pending_env):
        env, _ = pending_env
        blob = env.snapshot()

        mask_before = env.action_masks().copy()
        action = _first_legal(mask_before)
        obs_after_1, *_ = env.step(action)

        # Restore and replay: byte-identical continuation, pending included.
        obs_restored, info = env.reset(options={"snapshot": blob})
        assert info["episode_seed"] == "<restored>"
        assert env._pending is not None
        np.testing.assert_array_equal(env.action_masks(), mask_before)
        obs_after_2, *_ = env.step(action)
        for key in obs_after_1:
            np.testing.assert_array_equal(obs_after_1[key], obs_after_2[key], err_msg=key)

        env.reset(options={"snapshot": blob})  # leave fixture state intact

    def test_pack_mask_excludes_overfill_joker_picks(self):
        # Direct unit check of the mask path: a Buffoon-style joker pick
        # must be masked when joker slots are full (the engine would
        # happily overfill -- Riff-raff-class corruption).
        env = ShopGymEnv()
        env.reset(options={"episode_seed": "SHOPGYM_CONTRACT"})
        gs = env._adapter.raw_state
        gs["phase"] = GamePhase.PACK_OPENING
        gs["pack_cards"] = [create_joker("j_joker")]
        gs["pack_choices_remaining"] = 1
        gs["pack_type"] = "Buffoon"
        gs["jokers"] = [create_joker("j_joker") for _ in range(5)]

        offset = FAMILY_OFFSETS[ShopActionFamily.PickPackCard]
        mask = env.action_masks()
        assert not mask[offset]
        assert mask[shop_action(ShopActionFamily.SkipPack)]

        gs["jokers"].pop()
        assert env.action_masks()[offset]


class TestStartStateSampler:
    def test_sampler_blob_restores(self):
        source = ShopGymEnv()
        source.reset(options={"episode_seed": "SHOPGYM_CONTRACT"})
        blob = source.snapshot()

        env = ShopGymEnv(start_state_sampler=lambda: blob)
        obs, info = env.reset()
        assert info["episode_seed"] == "<restored>"
        src_obs, _ = source.reset(options={"snapshot": blob})
        for key in obs:
            np.testing.assert_array_equal(obs[key], src_obs[key], err_msg=key)

    def test_sampler_none_means_fresh_run(self):
        env = ShopGymEnv(start_state_sampler=lambda: None, seed_prefix="SHOPGYMFRESH")
        _, info = env.reset()
        assert info["episode_seed"].startswith("SHOPGYMFRESH_")
