"""Tests for the s1 obs-schema seam in ``shop_obs.py``.

Default (``s1_schema=False``) must be byte-identical to the pre-s1 schema
-- covered by the untouched ``tests/agents/test_shop_policy.py`` suite.
This file covers the OPT-IN s1 widening: joker rows 8 -> 15, the
offered-tag one-hot appended to ``shop_context``, the overfill dual-counter
discipline, and -- load-bearing for the s1 Phi-shaping truncation design
(docs/post-regen-training-plan.md section 8) -- the inverse property that
truncating a widened obs reproduces the default-path obs exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.observation import _TAG_IDX, NUM_TAGS
from jackdaw.env.shop_obs import (
    D_SHOP_CONTEXT,
    D_SHOP_CONTEXT_S1,
    MAX_JOKER_ROWS,
    MAX_JOKER_ROWS_S1,
    build_shop_observation,
    observation_space,
)
from jackdaw.env.shop_run_adapter import ShopRunAdapter

SEED = "SHOPOBS_S1_TEST"


@pytest.fixture(scope="module")
def shop_adapter() -> ShopRunAdapter:
    adapter = ShopRunAdapter(GreedyHandPolicy())
    adapter.reset("b_red", 1, SEED)
    adapter.raw_state["dollars"] = 50
    return adapter


def _blind_select_state(on_deck: str = "Small") -> dict:
    gs = initialize_run("b_red", 1, "SHOPOBS_BLINDSELECT")
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = on_deck
    return gs


class TestDefaultUnchanged:
    def test_default_call_matches_explicit_false(self, shop_adapter):
        gs = shop_adapter.raw_state
        default_obs = build_shop_observation(gs)
        explicit_obs = build_shop_observation(gs, s1_schema=False)
        for key in default_obs:
            np.testing.assert_array_equal(default_obs[key], explicit_obs[key], err_msg=key)

    def test_default_space_unchanged(self):
        space = observation_space()
        assert space["jokers"].shape == (MAX_JOKER_ROWS, 15)
        assert space["shop_context"].shape == (D_SHOP_CONTEXT,)


class TestS1JokerWidening:
    def test_widens_to_15_rows(self, shop_adapter):
        space = observation_space(s1_schema=True)
        assert space["jokers"].shape == (MAX_JOKER_ROWS_S1, 15)
        assert space["joker_mask"].shape == (MAX_JOKER_ROWS_S1,)
        assert space["joker_ids"].shape == (MAX_JOKER_ROWS_S1,)

    def test_up_to_15_real_jokers_encoded(self, shop_adapter):
        gs = shop_adapter.raw_state
        blob = shop_adapter.snapshot_state()
        try:
            jokers = [create_joker("j_joker") for _ in range(12)]
            for j in jokers[8:]:
                j.set_edition({"negative": True})  # bypass joker_slots
            gs["jokers"] = jokers
            obs = build_shop_observation(gs, s1_schema=True)
            assert obs["joker_mask"].sum() == 12
            assert obs["jokers"].shape == (MAX_JOKER_ROWS_S1, 15)
        finally:
            shop_adapter.restore_state(blob)

    def test_genuine_overfill_still_raises(self, shop_adapter):
        gs = shop_adapter.raw_state
        blob = shop_adapter.snapshot_state()
        try:
            # 16 non-negative jokers > MAX_JOKER_ROWS_S1 (15) and > joker_slots
            # (5) -- a genuine Riff-raff-class overfill.
            gs["jokers"] = [create_joker("j_joker") for _ in range(16)]
            gs["joker_slots"] = 5
            with pytest.raises(ValueError, match="overfill"):
                build_shop_observation(gs, s1_schema=True)
        finally:
            shop_adapter.restore_state(blob)

    def test_default_path_never_raises_on_the_same_state(self, shop_adapter):
        # The dual-counter discipline is s1-only; the default path keeps
        # its pre-existing silent-clip behavior untouched.
        gs = shop_adapter.raw_state
        blob = shop_adapter.snapshot_state()
        try:
            gs["jokers"] = [create_joker("j_joker") for _ in range(16)]
            gs["joker_slots"] = 5
            obs = build_shop_observation(gs)  # no raise
            assert obs["joker_mask"].sum() == MAX_JOKER_ROWS
        finally:
            shop_adapter.restore_state(blob)


class TestOfferedTagOneHot:
    def test_zero_mid_shop(self, shop_adapter):
        obs = build_shop_observation(shop_adapter.raw_state, s1_schema=True)
        assert not obs["shop_context"][D_SHOP_CONTEXT:].any()

    def test_small_blind_offers_its_tag(self):
        gs = _blind_select_state("Small")
        expected_key = gs["round_resets"]["blind_tags"]["Small"]
        obs = build_shop_observation(gs, s1_schema=True)
        tag_slice = obs["shop_context"][D_SHOP_CONTEXT:]
        assert tag_slice.sum() == 1.0
        assert tag_slice[_TAG_IDX[expected_key]] == 1.0

    def test_big_blind_offers_its_own_tag(self):
        gs = _blind_select_state("Big")
        expected_key = gs["round_resets"]["blind_tags"]["Big"]
        obs = build_shop_observation(gs, s1_schema=True)
        tag_slice = obs["shop_context"][D_SHOP_CONTEXT:]
        assert tag_slice.sum() == 1.0
        assert tag_slice[_TAG_IDX[expected_key]] == 1.0

    def test_boss_blind_offers_nothing(self):
        gs = _blind_select_state("Boss")
        obs = build_shop_observation(gs, s1_schema=True)
        assert not obs["shop_context"][D_SHOP_CONTEXT:].any()

    def test_default_path_has_no_tag_slice(self):
        gs = _blind_select_state("Small")
        obs = build_shop_observation(gs)  # default: no s1_schema
        assert obs["shop_context"].shape == (D_SHOP_CONTEXT,)


class TestInverseProperty:
    """Load-bearing: truncating a widened obs reproduces the default-path
    obs byte-identical, key by key -- this licenses the s1 Phi-shaping
    truncation design (docs/post-regen-training-plan.md section 8)."""

    def test_shop_state_round_trips(self, shop_adapter):
        gs = shop_adapter.raw_state
        default_obs = build_shop_observation(gs)
        widened_obs = build_shop_observation(gs, s1_schema=True)

        for key in default_obs:
            widened_value = widened_obs[key]
            if key in ("jokers", "joker_mask", "joker_ids"):
                truncated = widened_value[:MAX_JOKER_ROWS]
            elif key == "shop_context":
                truncated = widened_value[:D_SHOP_CONTEXT]
            else:
                truncated = widened_value
            np.testing.assert_array_equal(truncated, default_obs[key], err_msg=key)

    def test_blind_select_state_round_trips_with_offered_tag(self):
        # Even when the tag one-hot is non-zero, truncating it away
        # reproduces the (tag-blind) default-path shop_context exactly.
        gs = _blind_select_state("Small")
        default_obs = build_shop_observation(gs)
        widened_obs = build_shop_observation(gs, s1_schema=True)
        assert widened_obs["shop_context"][D_SHOP_CONTEXT:].any()  # tag IS set
        np.testing.assert_array_equal(
            widened_obs["shop_context"][:D_SHOP_CONTEXT], default_obs["shop_context"]
        )

    def test_widened_dims_pinned(self):
        assert D_SHOP_CONTEXT_S1 == D_SHOP_CONTEXT + NUM_TAGS
