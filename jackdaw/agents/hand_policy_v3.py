"""v3 hand-play trunk with identity gathers and trigger-match attention.

The card encoder consumes the explicit trigger_match matrix as fixed
cross-attention weights: matched joker identity vectors are summed into
separate scored and held channels for each card. The weights are not learned;
the policy learns how to use the engine-derived candidate matches.

Every identity-bearing joker row uses one fresh hand-net embedding table over
the frozen centers.json vocabulary, concatenated with the corresponding
24-d engine-derived descriptor row. The embedding captures residual identity
information while the descriptor gives cold-start structure. The vocabulary
is an append-only contract: changing its ordering silently reindexes trained
embedding rows and corrupts checkpoints.

This is a plain nn.Module rather than SB3's BaseFeaturesExtractor because the
v3 seam returns both the per-card latents and the pooled trunk latent. The
later pointer head consumes the former, while the flat-head control consumes
the latter; SB3 adapters belong at that integration seam.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces

from jackdaw.agents.hand_policy import _entity_mlp, masked_pool
from jackdaw.agents.joker_descriptors import DESCRIPTOR_DIM, DESCRIPTOR_MATRIX
from jackdaw.env.observation import NUM_CENTER_KEYS

LATENT_DIM = 256
EMBED_DIM = 16
_IDENTITY_DIM = EMBED_DIM + DESCRIPTOR_DIM


class HandPlayFeaturesExtractorV3(nn.Module):
    """v3 hand observation trunk returning card and pooled representations.

    card_latents has one 64-d post-MLP representation per observation card,
    including exact zero rows for masked padding. pooled_latent is the 256-d
    representation used by the flat-head control.
    """

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__()

        self.card_latent_dim = 64
        self.latent_dim = LATENT_DIM
        self.embedding = nn.Embedding(NUM_CENTER_KEYS + 1, EMBED_DIM, padding_idx=0)
        self.register_buffer(
            "descriptors", torch.as_tensor(DESCRIPTOR_MATRIX, dtype=torch.float32)
        )

        d_global = observation_space["global_context"].shape[0]
        d_card = observation_space["hand_cards"].shape[1]
        d_joker = observation_space["jokers"].shape[1]
        d_consumable = observation_space["consumables"].shape[1]

        self.card_encoder = _entity_mlp(d_card + 2 * _IDENTITY_DIM, self.card_latent_dim)
        self.joker_encoder = _entity_mlp(
            d_joker + 2 * _IDENTITY_DIM + 1, self.card_latent_dim
        )
        self.consumable_encoder = _entity_mlp(d_consumable, 32)
        self.global_encoder = nn.Sequential(
            nn.Linear(d_global, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, LATENT_DIM),
            nn.ReLU(),
        )

        concat_dim = (
            LATENT_DIM
            + 2 * self.card_latent_dim
            + 2 * self.card_latent_dim
            + 2 * 32
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, LATENT_DIM),
            nn.ReLU(),
            nn.Linear(LATENT_DIM, self.latent_dim),
            nn.ReLU(),
        )

    def _identity(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather the learned and frozen identity channels for center IDs."""
        ids = ids.long()
        return torch.cat([self.embedding(ids), self.descriptors[ids]], dim=-1)

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (per-card latents, pooled trunk latent)."""
        joker_identity = self._identity(obs["joker_ids"])
        copy_identity = self._identity(obs["copy_target_ids"])
        joker_rows = torch.cat(
            [
                obs["jokers"],
                joker_identity,
                copy_identity,
                obs["copy_active"].unsqueeze(-1),
            ],
            dim=-1,
        )
        joker_latents = self.joker_encoder(joker_rows)

        # The matrix supplies fixed weights; joker identity vectors are the
        # values. Gating columns defensively keeps padded joker edits out of
        # card latents even when an upstream writer does not zero them.
        weights = obs["trigger_match"] * obs["joker_mask"].unsqueeze(1).unsqueeze(-1)
        cross_sums = torch.einsum("bcjk,bjd->bckd", weights, joker_identity)
        card_rows = torch.cat(
            [
                obs["hand_cards"],
                cross_sums[..., 0, :],
                cross_sums[..., 1, :],
            ],
            dim=-1,
        )
        card_latents = self.card_encoder(card_rows)
        card_latents = card_latents * obs["hand_mask"].unsqueeze(-1)

        hand = masked_pool(card_latents, obs["hand_mask"])
        jokers = masked_pool(joker_latents, obs["joker_mask"])
        consumables = masked_pool(
            self.consumable_encoder(obs["consumables"]), obs["consumable_mask"]
        )
        global_ctx = self.global_encoder(obs["global_context"])
        pooled = self.trunk(torch.cat([global_ctx, hand, jokers, consumables], dim=-1))
        return card_latents, pooled


class FlatV3BCModel(nn.Module):
    """The v3 trunk with the legacy flat action and pooled value heads."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__()
        from jackdaw.agents.hand_action_space import NUM_HAND_ACTIONS

        self.features_extractor = HandPlayFeaturesExtractorV3(observation_space)
        self.action_head = nn.Linear(LATENT_DIM, NUM_HAND_ACTIONS)
        self.value_net = nn.Linear(LATENT_DIM, 1)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flat action logits and the pooled p-clear value estimate."""

        _, pooled = self.features_extractor(obs)
        return self.action_head(pooled), self.value_net(pooled).squeeze(-1)
