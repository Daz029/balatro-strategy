"""Fixed-suite evaluation for hand-play policies (BC or PPO).

The consistent yardstick across BC -> PPO -> later h1 iterations:

  - **Fixed eval suite**: episode seeds ``EVAL_{i:08d}`` under a stage
    preset's config. ``HandPlayAdapter`` is seed-deterministic, so every
    policy forever faces the identical state distribution. The ``EVAL_``
    prefix is reserved -- demo generation and training rollouts must never
    use it.
  - **Metric**: clear rate under deterministic (masked argmax) actions.
  - **Solver ceiling**: ``--solver-ceiling`` computes the exact solver's
    mean ``p_clear`` over the same seeds -- the exact-play reference that
    contextualizes an absolute clear rate ("0.62 vs ceiling 0.71"). It
    costs ~12s/seed, so results are cached to JSON next to --output and
    reused on later runs.

Usage::

    uv run python scripts/eval_hand_policy.py \
        --policy runs/bc/run1/bc_checkpoint.pt --stage stage2_curated \
        --n-episodes 500 --output runs/bc/run1/eval.json

    uv run python scripts/eval_hand_policy.py \
        --policy runs/hand_ppo/run1/best_model/best_model.zip \
        --stage stage2_curated --n-episodes 500 --solver-ceiling
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import HandPlayGymEnv, observation_space  # noqa: E402

EVAL_SEED_PREFIX = "EVAL"


def eval_seeds(n_episodes: int) -> list[str]:
    return [f"{EVAL_SEED_PREFIX}_{i:08d}" for i in range(n_episodes)]


class _BCPolicy:
    """Deterministic masked-argmax wrapper for a BC checkpoint."""

    obs_version = 1
    action_version = 1

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        import torch

        from jackdaw.agents.hand_policy import HandPlayBCModel

        self._torch = torch
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self._model = HandPlayBCModel(observation_space()).to(device)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()
        self._device = device

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        torch = self._torch
        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=self._device).unsqueeze(0) for k, v in obs.items()
            }
            mask_t = torch.as_tensor(mask, device=self._device).unsqueeze(0)
            log_probs = self._model.masked_log_probs(batch, mask_t)
        return int(log_probs.argmax(dim=-1).item())


class _PPOPolicy:
    """Deterministic wrapper for a saved MaskablePPO .zip."""

    obs_version = 1
    action_version = 1

    def __init__(self, model_path: Path, device: str) -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(model_path), device=device)

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


class _PointerBCPolicy:
    """Deterministic wrapper for a pointer BC checkpoint."""

    obs_version = 2
    action_version = 2

    def __init__(self, checkpoint_path: Path, device: str) -> None:
        import torch

        from jackdaw.agents.pointer_ppo_policy import load_bc_model

        self._torch = torch
        self._model = load_bc_model(checkpoint_path, device=device)
        self._device = device

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from jackdaw.agents.pointer_ppo_policy import _action_vector_from_decode

        torch = self._torch
        with torch.no_grad():
            batch = {
                k: torch.as_tensor(v, device=self._device).unsqueeze(0) for k, v in obs.items()
            }
            action_types, picked = self._model.decode(batch)
            action = _action_vector_from_decode(action_types, picked)
        return action.squeeze(0).cpu().numpy().astype(np.int64, copy=False)


class _PointerPPOPolicy:
    """Deterministic wrapper for a saved pointer PPO .zip."""

    obs_version = 2
    action_version = 2

    def __init__(self, model_path: Path, device: str) -> None:
        from train_hand_ppo_b import KLToBCPointerPPO

        self._model = KLToBCPointerPPO.load(str(model_path), device=device)

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        policy = self._model.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        action = policy.predict_deterministic(obs_tensor)
        return action.squeeze(0).detach().cpu().numpy().astype(np.int64, copy=False)


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


def load_policy(policy_path: Path, device: str):
    suffix = policy_path.suffix.lower()
    if suffix == ".zip":
        return (
            _PointerPPOPolicy(policy_path, device)
            if _zip_policy_kind(policy_path) == "pointer"
            else _PPOPolicy(policy_path, device)
        )
    if suffix == ".pt":
        import torch

        try:
            payload = torch.load(policy_path, map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001 -- normalize bad checkpoint errors
            raise ValueError(f"invalid BC checkpoint: {policy_path}") from exc
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"unrecognized BC checkpoint payload: {policy_path}")
        metadata = payload.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"unrecognized BC checkpoint metadata: {policy_path}")
        head = metadata.get("head") if metadata else None
        if head == "pointer":
            return _PointerBCPolicy(policy_path, device)
        if head is None:
            return _BCPolicy(policy_path, device)
        raise ValueError(f"unrecognized BC checkpoint head {head!r}: {policy_path}")
    raise ValueError(f"unsupported policy checkpoint suffix: {policy_path.suffix!r}")


def run_suite(policy, config: HandPlayConfig, n_episodes: int) -> dict:
    env = HandPlayGymEnv(
        config=config,
        obs_version=policy.obs_version,
        action_version=policy.action_version,
    )
    clears: list[bool] = []
    steps_total = 0
    for seed in eval_seeds(n_episodes):
        obs, info = env.reset(options={"episode_seed": seed})
        while True:
            if policy.action_version == 1:
                action = policy.act(obs, info["action_mask"])
            else:
                action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_total += 1
            if terminated or truncated:
                clears.append(bool(info["balatro/cleared"]))
                break
    return {
        "n_episodes": n_episodes,
        "clear_rate": float(np.mean(clears)),
        "mean_steps": steps_total / n_episodes,
    }


def solver_ceiling(
    config_stage: str | None,
    config: HandPlayConfig,
    n_episodes: int,
    cache_path: Path,
) -> dict:
    """Mean solver p_clear over the eval seeds (cached -- ~12s/seed)."""
    cache_key = f"{config_stage or 'default'}_{n_episodes}"
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache_key in cache:
        return cache[cache_key]

    from generate_hand_demos import generate_one_example

    p_clears: list[float] = []
    failures = 0
    t0 = time.time()
    for i, seed in enumerate(eval_seeds(n_episodes)):
        try:
            example = generate_one_example(seed, config)
            p_clears.append(example.p_clear)
        except Exception as exc:  # noqa: BLE001 -- one bad seed shouldn't kill the suite
            failures += 1
            print(f"  solver failed on {seed}: {exc}")
        if (i + 1) % 10 == 0:
            rate = (time.time() - t0) / (i + 1)
            print(f"  ceiling {i + 1}/{n_episodes} ({rate:.1f}s/seed)")

    result = {
        "mean_p_clear": float(np.mean(p_clears)) if p_clears else None,
        "n_solved": len(p_clears),
        "n_failures": failures,
    }
    cache[cache_key] = result
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help=".pt (BC) or .zip (PPO)")
    parser.add_argument("--stage", default=None, help="Stage preset for the eval distribution")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--solver-ceiling", action="store_true")
    parser.add_argument(
        "--ceiling-cache",
        type=Path,
        default=Path("data/eval_solver_ceiling.json"),
        help="Cache file for the (expensive) solver ceiling",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.stage:
        from generate_hand_demos import stage_presets

        config = stage_presets()[args.stage].config
    else:
        config = HandPlayConfig()

    policy = load_policy(args.policy, args.device)
    result = run_suite(policy, config, args.n_episodes)
    result["policy"] = str(args.policy)
    result["stage"] = args.stage

    if args.solver_ceiling:
        ceiling = solver_ceiling(args.stage, config, args.n_episodes, args.ceiling_cache)
        result["solver_ceiling"] = ceiling
        if ceiling["mean_p_clear"]:
            print(
                f"clear_rate={result['clear_rate']:.3f} vs "
                f"solver ceiling ~{ceiling['mean_p_clear']:.3f}"
            )

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
