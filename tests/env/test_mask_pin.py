"""Pin every env mask across the engine Phase 6 legality pass.

User decision 2026-10-12: the engine becomes fully vanilla and the
environments hold the constraints; env masks are reconstructed LATER
(CLAUDE.md TODO 4). Until then no engine change may alter what an agent sees.

Seeded episodes are driven ONLY by the env masks (a seeded pick among
``action_masks()``), so the trajectory depends on env legality plus engine
dynamics, never on the engine's own legal-action list. Each step hashes the
env's canonical mask, the generic ``get_action_mask`` arrays and
``hand_action_mask``; the per-env digest chain is compared with a fixture
generated on the pre-Phase-6 engine.

Regenerate (ONLY on a commit whose masks are known-good):
    REGEN_MASK_PIN=1 uv run pytest tests/env/test_mask_pin.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.agents.shop_action_space import ShopActionFamily, shop_action
from jackdaw.env.action_space import get_action_mask
from jackdaw.env.hand_play_gym import HandPlayConfig, HandPlayGymEnv, hand_action_mask
from jackdaw.env.shop_gym import ShopGymEnv
from jackdaw.env.shop_run_adapter import ShopRunConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mask_pin.json"
SHOP_SEEDS = [f"MASKPIN_SHOP_{i:02d}" for i in range(16)]
HAND_SEEDS = [f"MASKPIN_HAND_{i:02d}" for i in range(60)]
MAX_STEPS = 400


def _generic_mask_bytes(gs: dict[str, Any]) -> bytes:
    m = get_action_mask(gs)
    parts = [
        m.type_mask.tobytes(),
        m.card_mask.tobytes(),
        repr((m.max_card_select, m.min_card_select)).encode(),
    ]
    for key in sorted(m.entity_masks):
        parts.append(f"{int(key)}:".encode() + m.entity_masks[key].tobytes())
    return b"|".join(parts)


def _step_digest(h: Any, env_mask: np.ndarray, gs: dict[str, Any], action: int) -> None:
    h.update(np.asarray(env_mask, dtype=bool).tobytes())
    h.update(_generic_mask_bytes(gs))
    h.update(hand_action_mask(gs).tobytes())
    h.update(str(action).encode())


def _run(
    env: Any,
    reset_kwargs: dict[str, Any],
    rnd: random.Random,
    *,
    dollar_floor: int = 0,
    leave_row: int | None = None,
) -> tuple[str, int]:
    """Drive one episode from the env's own mask; return (digest, steps).

    ``dollar_floor`` and a rarely-taken ``leave_row`` (NextRound) keep shop
    episodes spending — buys, packs (Spectral included), rerolls, sells —
    instead of leaving the first shop broke.
    """
    h = hashlib.sha256()
    env.reset(**reset_kwargs)
    steps = 0
    for steps in range(1, MAX_STEPS + 1):
        if dollar_floor:
            gs = env._adapter.raw_state
            gs["dollars"] = max(gs.get("dollars", 0), dollar_floor)
        mask = env.action_masks()
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            break
        others = [int(a) for a in legal if int(a) != leave_row]
        if leave_row is not None and others and rnd.random() > 0.08:
            action = others[rnd.randrange(len(others))]
        else:
            action = int(legal[rnd.randrange(legal.size)])
        _step_digest(h, mask, env._adapter.raw_state, action)
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return h.hexdigest(), steps


def _shop_digests() -> dict[str, list]:
    out = {}
    leave = shop_action(ShopActionFamily.NextRound)
    for s1 in (False, True):
        for seed in SHOP_SEEDS:
            env = ShopGymEnv(config=ShopRunConfig(win_ante=4, s1_schema=s1))
            key = f"{seed}{'_S1' if s1 else ''}"
            out[key] = list(
                _run(
                    env,
                    {"options": {"episode_seed": seed}},
                    random.Random(key),
                    dollar_floor=40,
                    leave_row=leave,
                )
            )
    return out


def _hand_digests() -> dict[str, list]:
    out = {}
    for seed in HAND_SEEDS:
        env = HandPlayGymEnv(config=HandPlayConfig(), seed_prefix="MASKPIN")
        out[seed] = list(_run(env, {"options": {"episode_seed": seed}}, random.Random(seed)))
    return out


def _current() -> dict[str, dict[str, list]]:
    return {"shop": _shop_digests(), "hand": _hand_digests()}


def test_env_masks_unchanged() -> None:
    current = _current()
    if os.environ.get("REGEN_MASK_PIN"):
        FIXTURE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
    pinned = json.loads(FIXTURE.read_text())
    assert current == pinned
