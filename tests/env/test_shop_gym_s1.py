"""Tests for the s1 seam in ``ShopGymEnv`` (``ShopRunConfig(s1_schema=True)``).

Default (``s1_schema=False``, or omitting ``config`` entirely) must stay
byte-identical to pre-s1 -- covered here explicitly (in addition to the
untouched ``tests/env/test_shop_gym.py`` suite passing unmodified). This
file covers the OPT-IN widening: the Discrete(694) action space, the
non-boss blind-select decision (SkipBlind vs the reused NextRound
"select/proceed" row), boss blind-select staying auto-resolved, and
SellJoker rows 8-14 addressing obs joker rows 8-14 via a >8-joker state.
"""

from __future__ import annotations

import numpy as np

from jackdaw.agents.shop_action_space import (
    NUM_TOTAL_ACTIONS,
    NUM_TOTAL_ACTIONS_S1,
    ShopActionFamily,
    joker_row_for_sell_action,
    shop_action,
)
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card_factory import create_joker
from jackdaw.env.shop_gym import ShopGymEnv
from jackdaw.env.shop_obs import D_SHOP_CONTEXT, build_shop_observation
from jackdaw.env.shop_run_adapter import ShopRunAdapter, ShopRunConfig

SEED = "SHOPGYM_S1_TEST"

SKIP_BLIND = shop_action(ShopActionFamily.SkipBlind)
SELECT_BLIND = shop_action(ShopActionFamily.NextRound)  # reused at BLIND_SELECT


class TestFlagOffByteIdentical:
    def test_default_action_space_is_s0(self):
        env = ShopGymEnv()
        assert env.action_space.n == NUM_TOTAL_ACTIONS

    def test_explicit_false_matches_default(self):
        env_default = ShopGymEnv()
        env_explicit = ShopGymEnv(config=ShopRunConfig(s1_schema=False))
        obs_a, info_a = env_default.reset(options={"episode_seed": SEED})
        obs_b, info_b = env_explicit.reset(options={"episode_seed": SEED})

        assert env_default.action_space.n == env_explicit.action_space.n == NUM_TOTAL_ACTIONS
        for key in obs_a:
            np.testing.assert_array_equal(obs_a[key], obs_b[key], err_msg=key)
        np.testing.assert_array_equal(info_a["action_mask"], info_b["action_mask"])

    def test_default_never_exposes_blind_select(self):
        # s0 auto-resolves BLIND_SELECT unconditionally -- reset() always
        # lands on SHOP (or PACK_OPENING/terminal), never blind-select.
        env = ShopGymEnv()
        env.reset(options={"episode_seed": SEED})
        assert env._adapter.raw_state.get("phase") != GamePhase.BLIND_SELECT


class TestBlindSelectDecision:
    def test_reset_lands_on_non_boss_blind_select(self):
        env = ShopGymEnv(config=ShopRunConfig(s1_schema=True))
        obs, info = env.reset(options={"episode_seed": SEED})
        gs = env._adapter.raw_state
        assert gs.get("phase") == GamePhase.BLIND_SELECT
        assert gs.get("blind_on_deck") == "Small"

        mask = info["action_mask"]
        assert mask.shape == (NUM_TOTAL_ACTIONS_S1,)
        legal = np.flatnonzero(mask)
        assert set(legal.tolist()) == {SKIP_BLIND, SELECT_BLIND}

    def test_skip_awards_the_offered_tag_and_advances_to_big(self):
        env = ShopGymEnv(config=ShopRunConfig(s1_schema=True))
        env.reset(options={"episode_seed": SEED})
        gs = env._adapter.raw_state
        expected_tag = gs["round_resets"]["blind_tags"]["Small"]
        n_tags_before = len(gs.get("awarded_tags", []))

        obs, reward, terminated, truncated, info = env.step(SKIP_BLIND)
        gs = env._adapter.raw_state
        assert not terminated and not truncated
        assert reward == 0.0
        awarded = gs.get("awarded_tags", [])
        assert len(awarded) == n_tags_before + 1
        assert awarded[-1]["key"] == expected_tag
        # Advanced to Big's blind-select decision (still exposed under s1).
        assert gs.get("phase") == GamePhase.BLIND_SELECT
        assert gs.get("blind_on_deck") == "Big"
        legal = np.flatnonzero(info["action_mask"])
        assert set(legal.tolist()) == {SKIP_BLIND, SELECT_BLIND}

    def test_select_proceeds_into_the_blind(self):
        env = ShopGymEnv(config=ShopRunConfig(s1_schema=True))
        env.reset(options={"episode_seed": SEED})
        obs, reward, terminated, truncated, info = env.step(SELECT_BLIND)
        gs = env._adapter.raw_state
        # The greedy hand policy + CashOut auto-resolve the rest; control
        # returns at the next decision point (SHOP/PACK_OPENING) or ends.
        assert terminated or truncated or gs.get("phase") in (
            GamePhase.SHOP,
            GamePhase.PACK_OPENING,
        )

    def test_offered_tag_one_hot_matches_the_masked_decision(self):
        env = ShopGymEnv(config=ShopRunConfig(s1_schema=True))
        obs, info = env.reset(options={"episode_seed": SEED})
        gs = env._adapter.raw_state
        expected_tag = gs["round_resets"]["blind_tags"]["Small"]
        from jackdaw.env.observation import _TAG_IDX

        tag_slice = obs["shop_context"][D_SHOP_CONTEXT:]
        assert tag_slice.sum() == 1.0
        assert tag_slice[_TAG_IDX[expected_tag]] == 1.0

    def test_boss_blind_select_stays_auto_resolved(self):
        # Direct adapter-level check of the real guarantee ShopGymEnv
        # relies on: even with s1_schema=True, a Boss on_deck is never a
        # stopping point (SkipBlind is illegal there -- no real choice).
        # Probing _advance() directly (rather than driving a full,
        # hand-policy-dependent run to a real boss) isolates exactly the
        # on_deck branch under test.
        from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy

        adapter = ShopRunAdapter(GreedyHandPolicy(), ShopRunConfig(s1_schema=True))
        adapter.reset("b_red", 1, SEED)
        gs = adapter.raw_state
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Boss"
        adapter._advance()
        # Boss has no skip option -- the loop must auto-resolve past it,
        # landing on a real decision phase or terminal, never staying at
        # BLIND_SELECT.
        assert adapter.raw_state.get("phase") != GamePhase.BLIND_SELECT or adapter.done

    def test_mask_defensively_hides_skipblind_at_boss(self):
        # Defense-in-depth at the env layer, independent of the adapter
        # guarantee above: if BLIND_SELECT/Boss is ever observed, the mask
        # must not offer SkipBlind (an illegal engine action there).
        env = ShopGymEnv(config=ShopRunConfig(s1_schema=True))
        env.reset(options={"episode_seed": SEED})
        gs = env._adapter.raw_state
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Boss"
        mask = env.action_masks()
        assert not mask[SKIP_BLIND]


class TestSellJokerExt:
    def test_sell_joker_slot_10_sells_obs_joker_row_10(self):
        env = ShopGymEnv(config=ShopRunConfig(s1_schema=True))
        env.reset(options={"episode_seed": SEED})
        gs = env._adapter.raw_state
        gs["phase"] = GamePhase.SHOP
        jokers = [create_joker("j_joker") for _ in range(12)]
        for j in jokers[8:]:
            j.set_edition({"negative": True})  # bypass joker_slots
        gs["jokers"] = jokers
        obs = build_shop_observation(gs, s1_schema=True)
        assert obs["joker_mask"].sum() == 12

        sell_row_10 = shop_action(ShopActionFamily.SellJokerExt, 10 - 8)
        assert joker_row_for_sell_action(sell_row_10) == 10
        mask = env.action_masks()
        assert mask[sell_row_10]

        action = env._resolve_action(sell_row_10)
        assert action.area == "jokers"
        assert action.card_index == 10
        target_joker = jokers[10]
        env._adapter.step(action)
        remaining = env._adapter.raw_state["jokers"]
        assert len(remaining) == 11
        assert target_joker not in remaining

    def test_default_off_mask_never_sets_selljoker_ext_rows(self):
        env = ShopGymEnv()
        env.reset(options={"episode_seed": SEED})
        gs = env._adapter.raw_state
        gs["phase"] = GamePhase.SHOP
        jokers = [create_joker("j_joker") for _ in range(9)]
        jokers[-1].set_edition({"negative": True})
        gs["jokers"] = jokers
        mask = env.action_masks()
        assert mask.shape == (NUM_TOTAL_ACTIONS,)
