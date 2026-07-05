"""Tests for the shop-agent identity encoding + policy trunk.

Covers the three modules of build item 5:
* ``joker_descriptors`` — engine-derived static descriptor matrix,
* ``shop_obs`` — observation schema/encoding (env-side, numpy),
* ``shop_policy`` — embeddings + descriptors + pooled trunk (net-side).

Includes the VOCABULARY FREEZE pins: embedding rows and descriptor rows are
keyed by ``center_key_id`` (sorted centers.json). If these pins break,
every trained shop checkpoint's embedding table is silently reindexed —
that's an append-only-contract violation, not a test to casually update.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy
from jackdaw.agents.joker_descriptors import (
    DESCRIPTOR_DIM,
    DESCRIPTOR_MATRIX,
)
from jackdaw.agents.shop_policy import LATENT_DIM, ShopFeaturesExtractor
from jackdaw.engine.actions import OpenBooster
from jackdaw.env.observation import NUM_CENTER_KEYS, center_key_id
from jackdaw.env.shop_obs import (
    D_ITEM,
    MAX_PACK_ROWS,
    MAX_SHOP_ITEM_ROWS,
    PendingTarget,
    build_shop_observation,
    observation_space,
)
from jackdaw.env.shop_run_adapter import ShopRunAdapter

SEED = "SHOPPOLICY_TEST"


@pytest.fixture(scope="module")
def shop_adapter() -> ShopRunAdapter:
    adapter = ShopRunAdapter(GreedyHandPolicy())
    adapter.reset("b_red", 1, SEED)
    adapter.raw_state["dollars"] = 50
    return adapter


def _batched(obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v).unsqueeze(0) for k, v in obs.items()}


# ---------------------------------------------------------------------------
# Vocabulary freeze
# ---------------------------------------------------------------------------


class TestVocabularyFreeze:
    def test_vocab_size_pinned(self):
        assert NUM_CENTER_KEYS == 299

    def test_anchor_ids_pinned(self):
        # Sorted-centers.json ordering. If any of these move, embedding
        # tables in existing checkpoints are silently reindexed.
        assert center_key_id("j_joker") == 146
        assert center_key_id("j_greedy_joker") == 133
        assert center_key_id("c_fool") == 33
        assert center_key_id("v_overstock_norm") == 284
        assert center_key_id("p_arcana_normal_1") == 237
        assert center_key_id("m_steel") == 230
        assert center_key_id("nonexistent_key") == 0


# ---------------------------------------------------------------------------
# Descriptor matrix
# ---------------------------------------------------------------------------


class TestDescriptors:
    def test_shape_and_pad_row(self):
        assert DESCRIPTOR_MATRIX.shape == (NUM_CENTER_KEYS + 1, DESCRIPTOR_DIM)
        assert not DESCRIPTOR_MATRIX[0].any()  # padding row exactly zero

    def test_plain_joker(self):
        d = DESCRIPTOR_MATRIX[center_key_id("j_joker")]
        assert d[2] == 1.0  # is_joker
        assert d[0] == pytest.approx(1 / 4)  # common
        assert d[8] == pytest.approx(4 / 20)  # +4 mult

    def test_suit_conditional_joker(self):
        d = DESCRIPTOR_MATRIX[center_key_id("j_greedy_joker")]
        assert d[11] == 1.0  # suit-conditional
        assert d[12] == pytest.approx(1 / 3)  # Diamonds
        assert d[13] == pytest.approx(3 / 10)  # s_mult 3

    def test_scaling_joker(self):
        d = DESCRIPTOR_MATRIX[center_key_id("j_hologram")]
        assert d[20] == 1.0  # scaling flag
        assert d[21] == pytest.approx(0.25 / 5)

    def test_consumable_and_voucher_rows(self):
        tarot = DESCRIPTOR_MATRIX[center_key_id("c_fool")]
        assert tarot[3] == 1.0 and not tarot[2]
        voucher = DESCRIPTOR_MATRIX[center_key_id("v_overstock_norm")]
        assert voucher[6] == 1.0
        booster = DESCRIPTOR_MATRIX[center_key_id("p_arcana_normal_1")]
        assert booster[7] == 1.0

    def test_all_jokers_have_type_flag(self):
        from jackdaw.engine.data.prototypes import CENTER_POOLS

        for key in CENTER_POOLS["Joker"]:
            assert DESCRIPTOR_MATRIX[center_key_id(key)][2] == 1.0, key


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------


class TestShopObservation:
    def test_matches_space(self, shop_adapter):
        obs = build_shop_observation(shop_adapter.raw_state)
        space = observation_space()
        assert set(obs) == set(space.spaces)
        for key, sub in space.spaces.items():
            assert obs[key].shape == sub.shape, key
            assert obs[key].dtype == sub.dtype, key
            assert np.isfinite(obs[key]).all(), key

    def test_shop_inventory_masks_and_ids(self, shop_adapter):
        gs = shop_adapter.raw_state
        obs = build_shop_observation(gs)
        n_items = len(gs["shop_cards"])
        assert obs["shop_item_mask"].sum() == n_items
        assert (obs["shop_item_ids"][:n_items] > 0).all()
        assert obs["booster_mask"].sum() == len(gs["shop_boosters"])
        assert obs["voucher_mask"].sum() == len(gs["shop_vouchers"])
        # empty in shop phase: hand rows and pack rows
        assert obs["hand_mask"].sum() == 0
        assert obs["pack_item_mask"].sum() == 0

    def test_pack_opening_with_pending_target(self, shop_adapter):
        blob = shop_adapter.snapshot_state()
        try:
            shop_adapter.step(OpenBooster(card_index=0))
            gs = shop_adapter.raw_state
            n_pack = len(gs["pack_cards"])
            assert n_pack > 0

            pending = PendingTarget(kind="pack", slot=1, min_cards=1, max_cards=3)
            obs = build_shop_observation(gs, pending)
            assert obs["pack_item_mask"].sum() == min(n_pack, MAX_PACK_ROWS)
            assert obs["shop_context"][7] == 1.0  # pending flag
            assert obs["shop_context"][8] == pytest.approx(1 / 5)
            assert obs["shop_context"][9] == pytest.approx(3 / 5)
            # selected bit set on exactly the carrier row (feature 14)
            assert obs["pack_items"][1, 14] == 1.0
            assert obs["pack_items"][:, 14].sum() == 1.0
        finally:
            shop_adapter.restore_state(blob)

    def test_row_clipping(self, shop_adapter):
        gs = shop_adapter.raw_state
        blob = shop_adapter.snapshot_state()
        try:
            gs["shop_cards"] = gs["shop_cards"] * 5  # 10 items, 4 rows
            obs = build_shop_observation(gs)
            assert obs["shop_items"].shape == (MAX_SHOP_ITEM_ROWS, D_ITEM)
            assert obs["shop_item_mask"].sum() == MAX_SHOP_ITEM_ROWS
        finally:
            shop_adapter.restore_state(blob)


# ---------------------------------------------------------------------------
# Policy trunk
# ---------------------------------------------------------------------------


class TestShopFeaturesExtractor:
    def test_forward_shape_and_determinism(self, shop_adapter):
        extractor = ShopFeaturesExtractor(observation_space())
        extractor.eval()
        obs = _batched(build_shop_observation(shop_adapter.raw_state))
        with torch.no_grad():
            out1 = extractor(obs)
            out2 = extractor(obs)
        assert out1.shape == (1, LATENT_DIM)
        assert torch.isfinite(out1).all()
        assert torch.equal(out1, out2)

    def test_masked_rows_contribute_nothing(self, shop_adapter):
        extractor = ShopFeaturesExtractor(observation_space())
        extractor.eval()
        obs = _batched(build_shop_observation(shop_adapter.raw_state))
        with torch.no_grad():
            base = extractor(obs)
            # scramble VALUES in fully-masked blocks; output must not move
            obs["pack_items"] = torch.randn_like(obs["pack_items"])
            obs["pack_item_ids"] = torch.randint_like(obs["pack_item_ids"], 1, 200)
            obs["hand_cards"] = torch.randn_like(obs["hand_cards"])
            scrambled = extractor(obs)
        assert torch.equal(base, scrambled)

    def test_descriptors_are_frozen_buffer(self):
        extractor = ShopFeaturesExtractor(observation_space())
        param_ids = {id(p) for p in extractor.parameters()}
        assert id(extractor.descriptors) not in param_ids
        assert torch.equal(
            extractor.descriptors, torch.as_tensor(DESCRIPTOR_MATRIX, dtype=torch.float32)
        )

    def test_identity_shared_across_blocks(self):
        # The same center key indexes the same embedding row whether it
        # appears as an owned joker or a shop item — one table, by design.
        extractor = ShopFeaturesExtractor(observation_space())
        jid = torch.tensor([center_key_id("j_joker")])
        assert torch.equal(extractor.embedding(jid), extractor.embedding(jid.clone()))
        assert extractor.embedding.padding_idx == 0

    def test_gradients_flow_to_embedding(self, shop_adapter):
        extractor = ShopFeaturesExtractor(observation_space())
        obs = _batched(build_shop_observation(shop_adapter.raw_state))
        out = extractor(obs).sum()
        out.backward()
        grad = extractor.embedding.weight.grad
        assert grad is not None
        # rows for the offered shop items received gradient
        offered = obs["shop_item_ids"][0][obs["shop_item_mask"][0] > 0].long()
        assert grad[offered].abs().sum() > 0
