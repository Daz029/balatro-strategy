"""Checkpoint-backed hand policy for the shop env's ``hand_policy`` slot.

Wraps a trained hand-agent checkpoint (BC ``.pt`` or MaskablePPO ``.zip``)
as a ``game_state -> engine Action`` callable — the same contract
:class:`~jackdaw.agents.greedy_hand_policy.GreedyHandPolicy` satisfies. This
is the "real partner" the shop agent trains against once h0 exists;
GreedyHandPolicy stays the fixed scripted ablation baseline.

Inference is deterministic (masked argmax), so the induced hand-phase state
distribution is reproducible per seed — exactly the property the bootstrap
loop's self-play rollout harvest depends on.

Observation, action mask, and index->engine-action decoding are REUSED from
``hand_play_gym`` verbatim (``build_observation`` / ``hand_action_mask`` /
``action_to_engine_action``), so the partner feeds the policy byte-identical
inputs to what it was trained on, and the >8-card Serpent over-draw degrades
identically to the gym env: the observation truncates to 12 hand rows and
the Discrete(436) mask spans positions 0-7 (always a legal play).

Torch / sb3-contrib are imported lazily in the constructor so importing this
module (e.g. to reference the class in a factory) stays free of the ``train``
extra until a checkpoint is actually loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.engine.actions import Action
from jackdaw.env.hand_play_gym import (
    action_to_engine_action,
    build_observation,
    hand_action_mask,
    observation_space,
)


class HandCheckpointPolicy:
    """Deterministic hand policy backed by a trained checkpoint.

    Parameters
    ----------
    checkpoint_path:
        A BC checkpoint (``.pt`` with a ``model_state_dict``) or a saved
        MaskablePPO model (``.zip``). The kind is inferred from the suffix.
    device:
        Torch device string (default ``"cpu"`` — the partner runs inline in
        the shop env's step loop, where per-decision CPU inference is
        cheaper than GPU dispatch overhead).
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self._device = device
        path = Path(checkpoint_path)

        if path.suffix == ".zip":
            from sb3_contrib import MaskablePPO

            self._kind = "ppo"
            self._model = MaskablePPO.load(str(path), device=device)
        else:
            from jackdaw.agents.hand_policy import HandPlayBCModel

            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = HandPlayBCModel(observation_space()).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._kind = "bc"
            self._model = model

    def __call__(self, game_state: dict[str, Any]) -> Action:
        mask = hand_action_mask(game_state)
        if not mask.any():
            # The shop adapter only calls the hand policy in SELECTING_HAND
            # with budget remaining; an empty mask means a wiring bug.
            raise ValueError("HandCheckpointPolicy called with no legal hand action")
        obs = build_observation(game_state)
        action = self._infer(obs, mask)
        return action_to_engine_action(action, game_state)

    def _infer(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        torch = self._torch
        if self._kind == "ppo":
            action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)
        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=self._device).unsqueeze(0) for k, v in obs.items()
            }
            mask_t = torch.as_tensor(mask, device=self._device).unsqueeze(0)
            log_probs = self._model.masked_log_probs(batch, mask_t)
        return int(log_probs.argmax(dim=-1).item())
