"""Shared policy network for the hand-play agent (BC and PPO).

The transfer contract (decided in the /grilling session, see CLAUDE.md):
the entire trunk lives in a custom SB3 features extractor, PPO is built
with ``net_arch=[]`` (identity ``mlp_extractor``), so the policy decomposes
into exactly three modules that exist identically in both phases:

    features_extractor  (HandPlayFeaturesExtractor)   -> 256-d latent
    action_net          (Linear 256 -> NUM_HAND_ACTIONS)
    value_net           (Linear 256 -> 1)

``HandPlayBCModel`` is those same three modules for supervised training;
moving weights into ``MaskableActorCriticPolicy`` is a plain
``load_state_dict`` per module (see ``load_bc_weights_into_policy``) -- no
name mapping, no surgery. The value head regresses the solver's ``p_clear``
in BC, and with terminal 1/0 reward and ``gamma=1.0`` in PPO the critic
target is the same quantity, so the warm start is calibrated, not just
initialized.

Architecture: pooled per-entity-type MLP encoders (no attention -- with
<=8+5+2 entities and a global vector already carrying hand-type/synergy
summaries, pooled MLPs are sufficient and far cheaper; revisit inside the
features extractor alone if stage-3 BC accuracy plateaus). Masked pooling
means an absent entity type contributes exactly nothing -- which is what
makes the dormant consumable block in the observation a free forward-compat
seam rather than a false zero signal.

Torch is imported here (and only here within ``jackdaw.agents``): keep this
module out of any import path that must work without the ``train`` extra.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from jackdaw.agents.hand_action_space import NUM_HAND_ACTIONS

LATENT_DIM = 256


def masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean ⊕ masked max over the entity axis.

    x: (B, N, D), mask: (B, N) in {0, 1}. Returns (B, 2D). Fully-masked
    rows (no entities of this type) contribute exact zeros for both pools.
    """
    m = mask.unsqueeze(-1)
    total = (x * m).sum(dim=1)
    count = m.sum(dim=1).clamp(min=1.0)
    mean = total / count
    maxed = x.masked_fill(m == 0, float("-inf")).amax(dim=1)
    maxed = torch.where(torch.isfinite(maxed), maxed, torch.zeros_like(maxed))
    return torch.cat([mean, maxed], dim=-1)


def _entity_mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
    )


class HandPlayFeaturesExtractor(BaseFeaturesExtractor):
    """Dict-observation trunk producing the shared 256-d latent.

    Consumes the ``HandPlayGymEnv`` observation schema (== BC demo schema +
    dormant consumable block). Works as ``features_extractor_class`` for
    ``MaskableActorCriticPolicy`` and as the trunk of ``HandPlayBCModel``.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = LATENT_DIM) -> None:
        super().__init__(observation_space, features_dim)

        d_global = observation_space["global_context"].shape[0]
        d_card = observation_space["hand_cards"].shape[1]
        d_joker = observation_space["jokers"].shape[1]
        d_consumable = observation_space["consumables"].shape[1]

        self.hand_encoder = _entity_mlp(d_card, 64)
        self.joker_encoder = _entity_mlp(d_joker, 64)
        self.consumable_encoder = _entity_mlp(d_consumable, 32)
        self.global_encoder = nn.Sequential(
            nn.Linear(d_global, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        concat_dim = 256 + 2 * 64 + 2 * 64 + 2 * 32  # 576
        self.trunk = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        hand = masked_pool(
            self.hand_encoder(observations["hand_cards"]), observations["hand_mask"]
        )
        jokers = masked_pool(
            self.joker_encoder(observations["jokers"]), observations["joker_mask"]
        )
        consumables = masked_pool(
            self.consumable_encoder(observations["consumables"]),
            observations["consumable_mask"],
        )
        global_ctx = self.global_encoder(observations["global_context"])
        return self.trunk(torch.cat([global_ctx, hand, jokers, consumables], dim=-1))


class HandPlayBCModel(nn.Module):
    """The BC-phase model: same three modules the PPO policy will hold."""

    def __init__(self, observation_space: spaces.Dict) -> None:
        super().__init__()
        self.features_extractor = HandPlayFeaturesExtractor(observation_space)
        self.action_net = nn.Linear(LATENT_DIM, NUM_HAND_ACTIONS)
        self.value_net = nn.Linear(LATENT_DIM, 1)

    def forward(
        self, observations: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_logits, value)."""
        latent = self.features_extractor(observations)
        return self.action_net(latent), self.value_net(latent).squeeze(-1)

    def masked_log_probs(
        self, observations: dict[str, torch.Tensor], action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Log-probabilities with illegal actions masked to -inf, matching
        MaskablePPO's masking semantics exactly."""
        logits, _ = self.forward(observations)
        masked = logits.masked_fill(~action_mask, float("-inf"))
        return torch.log_softmax(masked, dim=-1)


def load_bc_weights_into_policy(policy: Any, bc_model: HandPlayBCModel) -> None:
    """Copy BC weights into a MaskableActorCriticPolicy built with
    ``features_extractor_class=HandPlayFeaturesExtractor`` and
    ``net_arch=[]``.

    Copies the trunk (features extractor), the action head, and the value
    head (critic warm start -- calibrated to P(clear), see module
    docstring). Raises if the architectures don't line up, rather than
    silently part-loading.
    """
    policy.features_extractor.load_state_dict(bc_model.features_extractor.state_dict())
    # With share_features_extractor=True (SB3 default), pi/vf extractors
    # alias policy.features_extractor; load into them too in case sharing
    # was disabled.
    if getattr(policy, "pi_features_extractor", None) is not policy.features_extractor:
        policy.pi_features_extractor.load_state_dict(bc_model.features_extractor.state_dict())
    if getattr(policy, "vf_features_extractor", None) is not policy.features_extractor:
        policy.vf_features_extractor.load_state_dict(bc_model.features_extractor.state_dict())
    policy.action_net.load_state_dict(bc_model.action_net.state_dict())
    policy.value_net.load_state_dict(bc_model.value_net.state_dict())
