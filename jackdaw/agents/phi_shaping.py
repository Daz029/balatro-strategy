"""s1 observation truncation and a frozen s0 critic potential.

The s1 observation schema is a pure append-only widening of s0: joker rows
8--14 are appended after s0's eight rows, and the offered-tag one-hot is
appended after the original shop context.  Therefore the frozen s0 critic
must receive a truncated s1 observation, not a widened copy.  The latter
would be out of distribution because its extra joker rows would contribute
through the shared encoder and masked pooling.

The pinned inverse property is::

    truncate_s1_obs(build_shop_observation(gs, pending, s1_schema=True))
    == build_shop_observation(gs, pending, s1_schema=False)

byte-identically for all states, including builds with more than eight
jokers (both schemas use the same first-eight prefix) and blind-select states
with an offered tag (the appended tag one-hot is dropped).

An s1-width critic is also accepted as a potential and is fed the s1
observation unchanged.  Which critic is the RIGHT potential is a separate
question from which one loads: a potential should be in-distribution for the
horizon being trained, so an a4 run wants a critic that has seen ante-3/4
states, not merely one whose partner matches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jackdaw.agents.shop_action_space import NUM_TOTAL_ACTIONS, NUM_TOTAL_ACTIONS_S1
from jackdaw.env.shop_obs import (
    D_SHOP_CONTEXT,
    D_SHOP_CONTEXT_S1,
    MAX_JOKER_ROWS,
    MAX_JOKER_ROWS_S1,
)

_JOKER_KEYS = ("jokers", "joker_mask", "joker_ids")


def _shape(value: np.ndarray, key: str) -> tuple[int, ...]:
    """Return an array shape, normalizing malformed input to ``ValueError``."""

    if not isinstance(value, np.ndarray):
        raise ValueError(f"observation key {key!r} must contain a numpy array")
    return value.shape


def truncate_s1_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Truncate an s1 shop observation back to the exact s0 schema.

    The function is intentionally strict: passing an s0 observation is an
    error rather than a silent no-op.  It pins
    ``truncate_s1_obs(build_shop_observation(gs, pending, s1_schema=True)) ==
    build_shop_observation(gs, pending, s1_schema=False)`` byte-identically,
    including states with more than eight jokers (both schemas use the same
    first-eight prefix) and offered-tag blind-select states (the appended
    tag one-hot is dropped).
    """

    result = dict(obs)

    for key in _JOKER_KEYS:
        if key not in obs:
            raise ValueError(f"s1 observation is missing required key {key!r}")
        value_shape = _shape(obs[key], key)
        if value_shape[0] != MAX_JOKER_ROWS_S1:
            raise ValueError(
                f"expected s1 {key!r} to have {MAX_JOKER_ROWS_S1} rows, "
                f"got shape {value_shape}"
            )
        result[key] = obs[key][:MAX_JOKER_ROWS]

    if "shop_context" not in obs:
        raise ValueError("s1 observation is missing required key 'shop_context'")
    context_shape = _shape(obs["shop_context"], "shop_context")
    if context_shape[0] != D_SHOP_CONTEXT_S1:
        raise ValueError(
            f"expected s1 'shop_context' to have width {D_SHOP_CONTEXT_S1}, "
            f"got shape {context_shape}"
        )
    result["shop_context"] = obs["shop_context"][:D_SHOP_CONTEXT]

    return result


def _to_s0_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Accept either schema and return an observation in the s0 schema."""

    required = (*_JOKER_KEYS, "shop_context")
    missing = [key for key in required if key not in obs]
    if missing:
        raise ValueError(f"observation is missing required keys: {missing}")

    shapes = {key: _shape(obs[key], key) for key in required}
    is_s1 = all(shapes[key][0] == MAX_JOKER_ROWS_S1 for key in _JOKER_KEYS) and (
        shapes["shop_context"][0] == D_SHOP_CONTEXT_S1
    )
    if is_s1:
        return truncate_s1_obs(obs)

    is_s0 = all(shapes[key][0] == MAX_JOKER_ROWS for key in _JOKER_KEYS) and (
        shapes["shop_context"][0] == D_SHOP_CONTEXT
    )
    if is_s0:
        return obs

    raise ValueError(
        "observation does not match either the s0 or s1 shop schema: "
        f"{shapes}"
    )


class S0CriticPhi:
    """Evaluate a fixed s0 ``MaskablePPO`` critic as a shaping potential."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        import torch
        from sb3_contrib import MaskablePPO

        self._torch = torch
        model = MaskablePPO.load(str(checkpoint_path), device=device)
        policy = model.policy
        action_width = policy.action_net.weight.shape[0]
        if action_width not in (NUM_TOTAL_ACTIONS, NUM_TOTAL_ACTIONS_S1):
            raise ValueError(
                f"checkpoint action_net has {action_width} rows, expected the "
                f"frozen s0 width {NUM_TOTAL_ACTIONS} or the s1 width "
                f"{NUM_TOTAL_ACTIONS_S1}"
            )
        # The critic must be fed the schema it was trained on.  An s0 critic
        # takes the truncated observation; an s1 critic takes it unchanged --
        # truncating for the latter would drop joker rows 8-14 and the offered
        # tag that its own encoder was fitted with.
        self._truncate = action_width == NUM_TOTAL_ACTIONS

        policy.eval()
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        # This policy is inference-only.  Remove the optimizer created by
        # SB3 so the frozen potential cannot accidentally be optimized.
        policy.optimizer = None
        self._policy = policy
        self._device = device

    def __call__(self, obs: dict[str, np.ndarray]) -> float:
        critic_obs = _to_s0_obs(obs) if self._truncate else obs
        obs_tensor, _ = self._policy.obs_to_tensor(critic_obs)
        with self._torch.no_grad():
            value = self._policy.predict_values(obs_tensor)
        return float(value.item())
