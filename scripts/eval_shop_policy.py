"""Fixed-suite evaluation for shop policies.

The consistent yardstick across s0 horizon stages and later bootstrap
iterations:

  - **Fixed eval suite**: episode seeds ``EVAL_{i:08d}``. ShopGymEnv resets
    are seed-deterministic, so every policy forever faces the identical run
    distribution (given the same hand-policy partner). The ``EVAL_`` prefix
    is reserved — training rollouts must never use it.
  - **Metrics**: win rate at the given ``--win-ante`` horizon under
    deterministic (masked argmax) actions, plus mean final ante / rounds
    cleared / decision count — the progress fingerprint when win rate is
    still near zero.
  - **Baseline**: ``--policy nextround`` is the do-nothing shop (leave every
    shop immediately, skip every pack). The gap between a trained policy
    and this baseline isolates shop value from hand-play skill (same
    partner on both sides).

Seeds where the hand policy loses the auto-resolved FIRST blind (before
the shop agent's first decision) are excluded from the rates and reported
as ``n_dead_at_reset`` — no shop decision influenced them.

Usage::

    uv run python scripts/eval_shop_policy.py \
        --policy runs/shop_ppo/stage_a2/best_model/best_model.zip \
        --win-ante 2 --n-episodes 200 --output runs/shop_ppo/stage_a2/eval.json

    uv run python scripts/eval_shop_policy.py --policy nextround --win-ante 2
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jackdaw.agents.shop_action_space import (  # noqa: E402
    ShopActionFamily,
    decode_shop_action,
    shop_action,
    target_combo_for_action,
)
from jackdaw.env.shop_gym import ShopGymEnv  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunConfig  # noqa: E402

EVAL_SEED_PREFIX = "EVAL"

# Generous per-episode decision budget; the env's own max_steps also caps.
_MAX_EPISODE_STEPS = 512


def eval_seeds(n_episodes: int) -> list[str]:
    return [f"{EVAL_SEED_PREFIX}_{i:08d}" for i in range(n_episodes)]


class NextRoundPolicy:
    """Do-nothing shop baseline: leave every shop, skip every pack."""

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        for family in (ShopActionFamily.NextRound, ShopActionFamily.SkipPack):
            action = shop_action(family)
            if mask[action]:
                return action
        return int(np.flatnonzero(mask)[0])  # pending-target etc.: first legal


class PPOPolicy:
    """Deterministic wrapper for a saved MaskablePPO .zip."""

    def __init__(self, model_path: Path, device: str) -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(model_path), device=device)

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def load_policy(policy: str, device: str):
    if policy == "nextround":
        return NextRoundPolicy()
    return PPOPolicy(Path(policy), device)


def run_suite(
    policy,
    win_ante: int,
    n_episodes: int,
    hand_policy=None,
    s1_schema: bool = False,
    dump_decisions: Path | None = None,
) -> dict:
    env = ShopGymEnv(
        config=ShopRunConfig(win_ante=win_ante, s1_schema=s1_schema),
        hand_policy=hand_policy,
    )
    wins: list[bool] = []
    final_antes: list[int] = []
    rounds_cleared: list[int] = []
    steps: list[int] = []
    dead_at_reset = 0

    with contextlib.ExitStack() as stack:
        trace_file = None
        if dump_decisions is not None:
            dump_decisions.parent.mkdir(parents=True, exist_ok=True)
            trace_file = stack.enter_context(dump_decisions.open("w", encoding="utf-8"))

        for seed in eval_seeds(n_episodes):
            try:
                obs, info = env.reset(options={"episode_seed": seed})
            except RuntimeError:
                # Hand policy lost the auto-resolved first blind — no shop
                # decision was ever made; not attributable to this policy.
                dead_at_reset += 1
                continue

            for step_count in range(1, _MAX_EPISODE_STEPS + 1):
                mask = info["action_mask"]
                gs = env._adapter.raw_state
                action = int(policy.act(obs, mask))
                family, slot = decode_shop_action(action)
                action_label = (
                    f"SelectTarget{list(target_combo_for_action(action))}"
                    if family is ShopActionFamily.SelectTarget
                    else f"{family.name}[{slot}]"
                )
                record = {
                    "seed": seed,
                    "step": step_count,
                    "ante": gs.get("round_resets", {}).get("ante", 1),
                    "round": gs.get("round", 0),
                    "dollars": gs.get("dollars", 0),
                    "pending_target": env._pending is not None,
                    "action": int(action),
                    "action_family": family.name,
                    "action_slot": slot,
                    "action_label": action_label,
                    "n_legal": int(mask.sum()),
                    "legal_actions": [int(i) for i in np.nonzero(mask)[0]],
                }
                obs, _, terminated, truncated, info = env.step(action)
                terminal = bool(terminated or truncated)
                record["terminal"] = terminal
                record["won"] = bool(info.get("balatro/won", False)) if terminal else None
                if trace_file is not None:
                    trace_file.write(json.dumps(record) + "\n")
                if terminal:
                    wins.append(bool(info.get("balatro/won", False)))
                    final_antes.append(int(info.get("balatro/ante", 1)))
                    rounds_cleared.append(int(info.get("balatro/round", 0)))
                    steps.append(step_count)
                    break
            else:
                raise AssertionError(f"episode {seed} did not terminate")

    n_played = len(wins)
    return {
        "win_ante": win_ante,
        "n_episodes": n_episodes,
        "n_played": n_played,
        "n_dead_at_reset": dead_at_reset,
        "win_rate": float(np.mean(wins)) if wins else None,
        "mean_final_ante": float(np.mean(final_antes)) if final_antes else None,
        "mean_rounds_cleared": float(np.mean(rounds_cleared)) if rounds_cleared else None,
        "mean_steps": float(np.mean(steps)) if steps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        required=True,
        help='MaskablePPO .zip path, or "nextround" for the do-nothing baseline',
    )
    parser.add_argument("--win-ante", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dump-decisions",
        type=Path,
        default=None,
        help="write a full per-decision JSONL trace to this path (one JSON object "
        "per policy decision). Aggregate metrics are unaffected.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--s1-schema", action="store_true")
    parser.add_argument(
        "--hand-policy",
        type=Path,
        default=None,
        help="hand-partner checkpoint (.pt/.zip); omit for the greedy baseline. "
        "Match this to the partner s0 was TRAINED against.",
    )
    parser.add_argument(
        "--partner-money-ordering",
        action="store_true",
        help="use clear-gated money-aware copy-joker ordering with the hand partner",
    )
    args = parser.parse_args()
    if args.partner_money_ordering and args.hand_policy is None:
        parser.error("--partner-money-ordering requires --hand-policy")

    hand_policy = None
    if args.hand_policy is not None:
        from jackdaw.agents.hand_checkpoint_policy import HandCheckpointPolicy

        hand_policy = HandCheckpointPolicy(
            str(args.hand_policy), money_aware_ordering=args.partner_money_ordering
        )

    policy = load_policy(args.policy, args.device)
    result = run_suite(
        policy,
        args.win_ante,
        args.n_episodes,
        hand_policy=hand_policy,
        s1_schema=args.s1_schema,
        dump_decisions=args.dump_decisions,
    )
    result["policy"] = args.policy
    result["hand_policy"] = str(args.hand_policy) if args.hand_policy is not None else "greedy"
    result["partner_money_ordering"] = bool(args.partner_money_ordering)

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
