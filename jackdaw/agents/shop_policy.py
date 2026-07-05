"""Policy network for the shop agent (raw-RL MaskablePPO, no BC stage).

Same structural contract as ``hand_policy``: the whole trunk lives in a
custom SB3 features extractor, PPO is built with ``net_arch=[]``, so the
policy is exactly ``features_extractor + action_net + value_net``. The
action head is FULL canonical width (``NUM_TOTAL_ACTIONS`` = 686) with the
hand block permanently masked in s0 — dead rows now, a literal-index merge
seam later.

Item identity (grilled decision — CLAUDE.md shop-agent design):

* ``nn.Embedding(NUM_CENTER_KEYS + 1, EMBED_DIM, padding_idx=0)`` — ONE
  learned table over the whole ``centers.json`` vocabulary, indexed by the
  ``*_ids`` obs arrays. One table (rather than per-type tables) keeps the
  sharing property trivially true: the same joker in an owned row, a shop
  slot, or a pack card hits the same vector.
* ``joker_descriptors.DESCRIPTOR_MATRIX`` as a non-trainable buffer —
  engine-derived effect facts for cold-start/pool-transfer.

Each identity-bearing entity row is augmented net-side with
``[embedding | descriptor]`` before its encoder; masked pooling then makes
absent entities contribute exact zeros, same as the hand net.

VOCABULARY FREEZE: embedding rows are keyed by ``center_key_id``, which is
built from sorted ``centers.json`` keys. Changing that file reorders ids and
silently corrupts every shop checkpoint — pinned by tests in
``tests/agents/test_shop_policy.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from jackdaw.agents.hand_policy import _entity_mlp, masked_pool
from jackdaw.agents.joker_descriptors import DESCRIPTOR_DIM, DESCRIPTOR_MATRIX
from jackdaw.env.observation import NUM_CENTER_KEYS

LATENT_DIM = 256
EMBED_DIM = 16
_AUG = EMBED_DIM + DESCRIPTOR_DIM  # identity channels appended to each row


class ShopFeaturesExtractor(BaseFeaturesExtractor):
    """Dict-observation trunk producing the shared 256-d latent.

    Consumes the ``shop_obs.build_shop_observation`` schema. The shop-item
    and pack-content blocks share one encoder (identical union row layout —
    an item's value shouldn't depend on which shelf it sits on), but pool
    separately so the trunk still sees them as distinct context.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = LATENT_DIM) -> None:
        super().__init__(observation_space, features_dim)

        self.embedding = nn.Embedding(NUM_CENTER_KEYS + 1, EMBED_DIM, padding_idx=0)
        self.register_buffer("descriptors", torch.as_tensor(DESCRIPTOR_MATRIX, dtype=torch.float32))

        d_global = observation_space["global_context"].shape[0]
        d_ctx = observation_space["shop_context"].shape[0]
        d_card = observation_space["hand_cards"].shape[1]
        d_joker = observation_space["jokers"].shape[1]
        d_cons = observation_space["consumables"].shape[1]
        d_item = observation_space["shop_items"].shape[1]
        d_voucher = observation_space["vouchers"].shape[1]
        d_booster = observation_space["boosters"].shape[1]

        self.hand_encoder = _entity_mlp(d_card, 64)  # playing cards: no identity aug
        self.joker_encoder = _entity_mlp(d_joker + _AUG, 64)
        self.consumable_encoder = _entity_mlp(d_cons + _AUG, 32)
        self.item_encoder = _entity_mlp(d_item + _AUG, 64)  # shared: shop slots + pack
        self.voucher_encoder = _entity_mlp(d_voucher + _AUG, 32)
        self.booster_encoder = _entity_mlp(d_booster + _AUG, 32)
        self.global_encoder = nn.Sequential(
            nn.Linear(d_global, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        concat_dim = (
            256  # global
            + d_ctx  # shop context, raw
            + 2 * 64  # hand pool
            + 2 * 64  # joker pool
            + 2 * 32  # consumable pool
            + 2 * 64  # shop-item pool
            + 2 * 64  # pack pool
            + 2 * 32  # voucher pool
            + 2 * 32  # booster pool
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def _aug(self, rows: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """Append [embedding | descriptor] identity channels to entity rows."""
        idx = ids.long()
        return torch.cat([rows, self.embedding(idx), self.descriptors[idx]], dim=-1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        hand = masked_pool(self.hand_encoder(obs["hand_cards"]), obs["hand_mask"])
        jokers = masked_pool(
            self.joker_encoder(self._aug(obs["jokers"], obs["joker_ids"])),
            obs["joker_mask"],
        )
        cons = masked_pool(
            self.consumable_encoder(self._aug(obs["consumables"], obs["consumable_ids"])),
            obs["consumable_mask"],
        )
        shop = masked_pool(
            self.item_encoder(self._aug(obs["shop_items"], obs["shop_item_ids"])),
            obs["shop_item_mask"],
        )
        pack = masked_pool(
            self.item_encoder(self._aug(obs["pack_items"], obs["pack_item_ids"])),
            obs["pack_item_mask"],
        )
        vouchers = masked_pool(
            self.voucher_encoder(self._aug(obs["vouchers"], obs["voucher_ids"])),
            obs["voucher_mask"],
        )
        boosters = masked_pool(
            self.booster_encoder(self._aug(obs["boosters"], obs["booster_ids"])),
            obs["booster_mask"],
        )
        global_ctx = self.global_encoder(obs["global_context"])
        return self.trunk(
            torch.cat(
                [
                    global_ctx,
                    obs["shop_context"],
                    hand,
                    jokers,
                    cons,
                    shop,
                    pack,
                    vouchers,
                    boosters,
                ],
                dim=-1,
            )
        )
