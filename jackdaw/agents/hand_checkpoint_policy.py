"""Checkpoint-backed hand policy for the shop env's ``hand_policy`` slot.

Wraps a trained hand-agent checkpoint (v1 BC ``.pt``, pointer BC ``.pt``,
MaskablePPO ``.zip``, or pointer PPO ``.zip``) as a
``game_state -> engine Action`` callable. The v1 path uses the frozen
``build_observation`` / ``hand_action_mask`` / ``action_to_engine_action``
Discrete(436) contract. Pointer checkpoints use ``build_observation_v2``,
their pinned greedy per-step decoder, and the shared pointer engine-action
path directly; they never enter the v1 machinery.

Torch / sb3-contrib are imported lazily in the constructor so importing this
module stays free of the training extras until a checkpoint is loaded.
"""

from __future__ import annotations

import base64
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.agents.ordering_objective import make_clear_gated_money_objective
from jackdaw.engine.actions import Action
from jackdaw.env.hand_play_gym import (
    action_to_engine_action,
    build_observation,
    build_observation_v2,
    hand_action_mask,
    observation_space,
    pointer_action_to_engine_action,
)


def _pointer_class_record(data: dict) -> str:
    """Return the serialized policy-class record used for dispatch."""

    record = data.get("policy_class")
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""

    parts = [str(record.get(key, "")) for key in ("__name__", "__qualname__", "__module__")]
    serialized = record.get(":serialized:")
    if isinstance(serialized, str):
        try:
            parts.append(base64.b64decode(serialized).decode("latin1"))
        except Exception:  # noqa: BLE001 -- malformed metadata is handled below
            pass
    return " ".join(parts)


def _zip_policy_kind(policy_path: Path) -> str:
    """Classify an SB3 archive as the pointer or frozen v1 policy kind."""

    try:
        with zipfile.ZipFile(policy_path) as archive:
            data = json.loads(archive.read("data"))
    except (
        OSError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        raise ValueError(f"invalid SB3 checkpoint archive: {policy_path}") from exc

    class_record = _pointer_class_record(data)
    if "PointerPPOPolicy" in class_record:
        return "pointer"
    if "Maskable" in class_record:
        return "v1"
    raise ValueError(f"unrecognized SB3 policy class in {policy_path}: {class_record!r}")


def _load_pointer_ppo_class():
    """Import the pointer PPO loader, adding ``scripts`` when needed."""

    try:
        from train_hand_ppo_b import KLToBCPointerPPO
    except ImportError:
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.append(str(scripts_dir))
        try:
            from train_hand_ppo_b import KLToBCPointerPPO
        except ImportError as second_exc:
            raise ImportError(
                "loading pointer PPO checkpoints requires scripts/train_hand_ppo_b.py "
                "and its dependencies"
            ) from second_exc
        return KLToBCPointerPPO
    return KLToBCPointerPPO


class HandCheckpointPolicy:
    """Deterministic hand policy backed by a content-dispatched checkpoint.

    Parameters
    ----------
    checkpoint_path:
        A v1 or pointer BC checkpoint (``.pt``), or a v1 MaskablePPO or
        pointer PPO checkpoint (``.zip``).
    device:
        Torch device string (default ``"cpu"`` -- the partner runs inline in
        the shop env's step loop).
    money_aware_ordering:
        If true, build a clear-gated money objective fresh for every call and
        pass it to both decoder paths; the factory must snapshot the current
        banked chips for that hand-turn decision. Defaults to false.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        *,
        money_aware_ordering: bool = False,
    ) -> None:
        import torch

        self._torch = torch
        self._device = device
        self._money_aware_ordering = money_aware_ordering
        path = Path(checkpoint_path)

        if path.suffix.lower() == ".zip":
            policy_kind = _zip_policy_kind(path)
            if policy_kind == "pointer":
                ppo_class = _load_pointer_ppo_class()
                self._kind = "pointer_ppo"
                self._model = ppo_class.load(str(path), device=device)
            else:
                from sb3_contrib import MaskablePPO

                self._kind = "ppo"
                self._model = MaskablePPO.load(str(path), device=device)
            return

        if path.suffix.lower() != ".pt":
            raise ValueError(f"unsupported policy checkpoint suffix: {path.suffix!r}")

        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001 -- normalize bad checkpoint errors
            raise ValueError(f"invalid BC checkpoint: {path}") from exc
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError(f"unrecognized BC checkpoint payload: {path}")
        metadata = checkpoint.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"unrecognized BC checkpoint metadata: {path}")
        head = metadata.get("head") if metadata else None
        if head == "pointer":
            from jackdaw.agents.pointer_ppo_policy import load_bc_model

            self._kind = "pointer_bc"
            self._model = load_bc_model(path, device=device)
        elif head is None:
            from jackdaw.agents.hand_policy import HandPlayBCModel

            model = HandPlayBCModel(observation_space()).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self._kind = "bc"
            self._model = model
        else:
            raise ValueError(f"unrecognized BC checkpoint head {head!r}: {path}")

    def __call__(self, game_state: dict[str, Any]) -> Action:
        ordering_objective = (
            make_clear_gated_money_objective(game_state)
            if self._money_aware_ordering
            else None
        )
        if self._kind in {"pointer_bc", "pointer_ppo"}:
            obs = build_observation_v2(game_state)
            action = self._infer_pointer(obs)
            return pointer_action_to_engine_action(
                action, game_state, ordering_objective=ordering_objective
            )

        mask = hand_action_mask(game_state)
        if not mask.any():
            # The shop adapter only calls the hand policy in SELECTING_HAND
            # with budget remaining; an empty mask means a wiring bug.
            raise ValueError("HandCheckpointPolicy called with no legal hand action")
        obs = build_observation(game_state)
        action = self._infer(obs, mask)
        return action_to_engine_action(action, game_state, ordering_objective=ordering_objective)

    def _infer_pointer(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from jackdaw.agents.pointer_ppo_policy import _action_vector_from_decode

        torch = self._torch
        with torch.no_grad():
            if self._kind == "pointer_bc":
                batch = {
                    k: torch.as_tensor(v, device=self._device).unsqueeze(0)
                    for k, v in obs.items()
                }
                action_types, picked = self._model.decode(batch)
                action = _action_vector_from_decode(action_types, picked)
            else:
                policy = self._model.policy
                obs_tensor, _ = policy.obs_to_tensor(obs)
                action = policy.predict_deterministic(obs_tensor)
        return action.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)

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
