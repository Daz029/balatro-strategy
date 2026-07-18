"""SB3 adapter for the v2 autoregressive hand pointer policy.

The actor is deliberately kept as :class:`PointerActionHead`: SB3's
``MultiCategoricalDistribution`` cannot express the prefix-dependent masks or
the stop-padded sequence event.  This small ``BasePolicy`` adapter exposes the
surfaces PPO actually uses while preserving the v3 feature trunk, pointer
distribution, and calibrated value head as ordinary torch modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import preprocess_obs

from jackdaw.agents.hand_pointer_head import (
    CARD_SLOTS,
    MAX_PICKS,
    STOP_INDEX,
    HandPointerBCModel,
    PointerActionHead,
    initial_type_mask,
    pick_step_mask,
)
from jackdaw.agents.hand_policy_v3 import HandPlayFeaturesExtractorV3
from jackdaw.env.hand_play_gym import observation_space_v2

MASK_FUNCTIONS = (initial_type_mask, pick_step_mask)


def _validate_action_space(action_space: spaces.Space) -> None:
    if not isinstance(action_space, spaces.MultiDiscrete) or not np.array_equal(
        action_space.nvec, np.array([2] + [CARD_SLOTS + 1] * MAX_PICKS)
    ):
        raise ValueError(
            "PointerPPOPolicy requires MultiDiscrete([2] + [41] * 5), "
            f"got {action_space!r}"
        )


def _action_vector_from_decode(
    action_types: torch.Tensor, picked: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    actions = torch.full(
        (action_types.shape[0], 1 + MAX_PICKS),
        STOP_INDEX,
        dtype=torch.long,
        device=action_types.device,
    )
    actions[:, 0] = action_types.long()
    for row, indices in enumerate(picked):
        length = min(indices.numel(), MAX_PICKS)
        if length:
            actions[row, 1 : 1 + length] = indices[:length]
    return actions


class PointerPPOPolicy(BasePolicy):
    """Plain torch pointer actor-critic with the SB3 policy adapter surface."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.MultiDiscrete,
        lr_schedule,
        *,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        _validate_action_space(action_space)
        super().__init__(observation_space, action_space, normalize_images=False)
        self.features_extractor = HandPlayFeaturesExtractorV3(observation_space)
        self.pointer_head = PointerActionHead()
        self.value_net = nn.Linear(256, 1)

        optimizer_kwargs = dict(optimizer_kwargs or {})
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_class(
            self.parameters(), lr=float(lr_schedule(1.0)), **optimizer_kwargs
        )

    def _setup_model(self) -> None:
        """The adapter builds its modules in ``__init__``; PPO owns setup."""

    @staticmethod
    def _budgets(obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return HandPointerBCModel._budgets(obs)

    def _features(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        preprocessed = preprocess_obs(obs, self.observation_space, normalize_images=False)
        return self.features_extractor(preprocessed)

    @staticmethod
    def _labels_from_actions(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        if actions.ndim != 2 or actions.shape[1] != 1 + MAX_PICKS:
            raise ValueError(f"actions must have shape (B, 6), got {tuple(actions.shape)}")
        actions = actions.long()
        action_type = actions[:, 0]
        pointers = actions[:, 1:]
        labels = torch.full_like(pointers, -1)
        for row in range(actions.shape[0]):
            stop = torch.where(pointers[row] == STOP_INDEX)[0]
            length = int(stop[0]) if len(stop) else MAX_PICKS
            if len(stop) and bool((pointers[row, length:] != STOP_INDEX).any()):
                raise ValueError("pointer action padding must trail STOP_INDEX")
            if length:
                labels[row, :length] = pointers[row, :length]
        return action_type, labels

    def _step_distributions(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        card_latents, pooled = self._features(obs)
        action_type, card_indices = self._labels_from_actions(actions)
        hands_left, discards_left = self._budgets(obs)
        return self.pointer_head.teacher_forced_step_distributions(
            card_latents,
            pooled,
            obs["hand_mask"],
            hands_left,
            discards_left,
            action_type,
            card_indices,
        )

    def teacher_forced_step_distributions(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the actor's masked distributions along taken prefixes."""

        return self._step_distributions(obs, actions)

    def teacher_forced_step_masks(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(type_mask, pointer_masks, active)`` for parity checks."""

        type_log_probs, pointer_log_probs, active = self._step_distributions(obs, actions)
        return (
            torch.isfinite(type_log_probs),
            torch.isfinite(pointer_log_probs),
            active,
        )

    def _act_components(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        card_latents, pooled = self._features(obs)
        hands_left, discards_left = self._budgets(obs)
        action_types, picked = self.pointer_head.sample(
            card_latents,
            pooled,
            obs["hand_mask"],
            hands_left,
            discards_left,
        )
        actions = _action_vector_from_decode(action_types, picked)
        values, sequence_log_prob, _ = self.evaluate_actions(obs, actions)
        return actions, values.squeeze(-1), sequence_log_prob

    def act(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a valid action vector and return value and sequence log-prob."""

        return self._act_components(obs)

    def evaluate_actions(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate taken sequences and sum entropy over active tokens."""

        type_log_probs, pointer_log_probs, active = self._step_distributions(obs, actions)
        action_type, card_indices = self._labels_from_actions(actions)
        type_target = type_log_probs.gather(1, action_type.unsqueeze(-1)).squeeze(-1)
        pointer_targets = torch.where(
            card_indices >= 0,
            card_indices,
            torch.full_like(card_indices, STOP_INDEX),
        )
        pointer_targets_log_prob = pointer_log_probs.gather(
            2, pointer_targets.unsqueeze(-1)
        ).squeeze(-1)
        per_step_log_prob = torch.cat((type_target.unsqueeze(-1), pointer_targets_log_prob), dim=-1)
        sequence_log_prob = (per_step_log_prob * active).sum(dim=-1)

        type_mask = torch.isfinite(type_log_probs)
        pointer_mask = torch.isfinite(pointer_log_probs)
        type_entropy = -(
            type_log_probs.exp()
            * torch.where(type_mask, type_log_probs, torch.zeros_like(type_log_probs))
        ).sum(dim=-1)
        pointer_entropy = -(
            pointer_log_probs.exp()
            * torch.where(pointer_mask, pointer_log_probs, torch.zeros_like(pointer_log_probs))
        ).sum(dim=-1)
        entropy_sum = type_entropy + (pointer_entropy * active[:, 1:]).sum(dim=-1)

        _, pooled = self._features(obs)
        values = self.value_net(pooled)
        return values, sequence_log_prob, entropy_sum

    def forward(
        self, obs: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if deterministic:
            actions = self.predict_deterministic(obs)
            values, log_prob, _ = self.evaluate_actions(obs, actions)
            return actions, values, log_prob
        actions, values, log_prob = self.act(obs)
        return actions, values.unsqueeze(-1), log_prob

    def predict_values(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        _, pooled = self._features(obs)
        return self.value_net(pooled)

    def predict_deterministic(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the pinned per-step greedy decode as a padded action vector."""

        card_latents, pooled = self._features(obs)
        hands_left, discards_left = self._budgets(obs)
        action_types, picked = self.pointer_head.greedy_decode(
            card_latents,
            pooled,
            obs["hand_mask"],
            hands_left,
            discards_left,
        )
        return _action_vector_from_decode(action_types, picked)

    def _predict(
        self, observation: dict[str, torch.Tensor], deterministic: bool = False
    ) -> torch.Tensor:
        return (
            self.predict_deterministic(observation)
            if deterministic
            else self.act(observation)[0]
        )

    def load_bc_weights(self, checkpoint_path: str | Path) -> None:
        load_bc_weights(self, checkpoint_path)


PointerActorCriticPolicy = PointerPPOPolicy


def load_bc_weights_into_policy(policy: PointerPPOPolicy, bc_model: HandPointerBCModel) -> None:
    """Copy all pointer BC modules, raising rather than partially loading."""

    transfers = (
        (policy.features_extractor, bc_model.features_extractor, "features_extractor"),
        (policy.pointer_head, bc_model.pointer_head, "pointer_head"),
        (policy.value_net, bc_model.value_net, "value_net"),
    )
    for target, source, name in transfers:
        target_state = target.state_dict()
        source_state = source.state_dict()
        if target_state.keys() != source_state.keys() or any(
            target_state[key].shape != source_state[key].shape for key in target_state
        ):
            raise RuntimeError(f"BC {name} architecture does not match pointer PPO policy")
    for target, source, _name in transfers:
        target.load_state_dict(source.state_dict(), strict=True)


def load_bc_weights(
    policy: PointerPPOPolicy,
    checkpoint: str | Path | HandPointerBCModel,
) -> None:
    """Load a complete ``bc_v3_pointer.pt`` into a pointer PPO policy."""

    if isinstance(checkpoint, HandPointerBCModel):
        bc_model = checkpoint
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata.get("head", "pointer") != "pointer":
            raise ValueError("BC checkpoint is not a pointer checkpoint")
        bc_model = HandPointerBCModel(policy.observation_space)
        bc_model.load_state_dict(payload["model_state_dict"])
    load_bc_weights_into_policy(policy, bc_model)


def load_bc_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> HandPointerBCModel:
    """Load the frozen BC pointer reference used by the KL leash."""

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = payload.get("metadata", {})
    if metadata.get("head", "pointer") != "pointer":
        raise ValueError("BC checkpoint is not a pointer checkpoint")
    model = HandPointerBCModel(observation_space_v2()).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    model.requires_grad_(False)
    return model
