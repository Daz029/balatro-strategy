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
import json
import sys
import time
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

    def __init__(self, model_path: Path, device: str) -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(model_path), device=device)

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def load_policy(policy_path: Path, device: str):
    if policy_path.suffix == ".zip":
        return _PPOPolicy(policy_path, device)
    return _BCPolicy(policy_path, device)


def run_suite(policy, config: HandPlayConfig, n_episodes: int) -> dict:
    env = HandPlayGymEnv(config=config)
    clears: list[bool] = []
    steps_total = 0
    for seed in eval_seeds(n_episodes):
        obs, info = env.reset(options={"episode_seed": seed})
        while True:
            action = policy.act(obs, info["action_mask"])
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
