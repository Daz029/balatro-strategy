"""Measure how far two hand-play pointer policies diverge on a fixed state set.

Throwaway diagnostic (h1 vs h2): the s1-training-hiccups doc's harvest->h2 step
turns on ONE fork -- did h2's policy actually move from h1, or is
"s1+h2 == s1+h1" just an un-co-adapted-pair artifact (Future-worry #5)?

This settles it directly, no training and no shop in the loop: draw a fixed,
reproducible sample of harvested (induced-distribution, deep-ante) hand states,
and on each one compare both policies' DETERMINISTIC (greedy-decode) action plus
the log-prob each assigns to the other's chosen action.

  - agreement_rate:    fraction of states where the argmax play/discard is the
                       identical (type, card-set). ~1.0 => policy barely moved.
  - type_agreement:    same but only the play-vs-discard choice.
  - mean_jeffrey:      symmetric chosen-action log-prob gap
                       0.5 * ((logpA[aA]-logpB[aA]) + (logpB[aB]-logpA[aB])),
                       >= 0; a graded "how different" that survives argmax ties.
  - value gap:         mean |V_A - V_B| (are the two critics different too?).

Usage (on the machine holding both checkpoints)::

    uv run python scripts/hand_policy_divergence.py \
        --policy-a runs/hand_ppo_b/h1/best_model/best_model.zip \
        --policy-b runs/hand_ppo_b/h2/best_model/best_model.zip \
        --harvest-dir data/harvest_s0 --n 800 --output data/h1_h2_divergence.json

Self-test: pass the SAME checkpoint for -a and -b; agreement must be 1.0 and
every divergence exactly 0.0.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jackdaw.env.hand_play_gym import HandPlayGymEnv  # noqa: E402
from scripts.harvest_restore import restore_state  # noqa: E402


def load_pointer(path: Path, device: str):
    """Load a pointer PPO .zip and return its policy (obs_to_tensor etc.)."""
    from train_hand_ppo_b import KLToBCPointerPPO

    model = KLToBCPointerPPO.load(str(path), device=device)
    model.policy.set_training_mode(False)
    return model.policy


def sample_records(harvest_dir: Path, n: int, seed: int) -> list[tuple[str, str]]:
    """Deterministic fixed sample of (run_seed, record_id) hand records."""
    records: list[tuple[str, str]] = []
    with (harvest_dir / "metadata.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") == "hand" and rec.get("source") in ("det", "sampled"):
                records.append((str(rec["run_seed"]), str(rec["record_id"])))
    if not records:
        raise ValueError(f"no eligible hand records in {harvest_dir}")
    records.sort()  # stable order independent of file iteration
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n]


def iter_snapshots(harvest_dir: Path, records: list[tuple[str, str]]):
    """Yield repaired-gs snapshot blobs, mirroring HarvestSnapshotSampler."""
    shard_cache: dict[str, dict[str, bytes]] = {}
    for run_seed, record_id in records:
        shard = shard_cache.get(run_seed)
        if shard is None:
            with (harvest_dir / "blobs" / f"{run_seed}.pkl").open("rb") as fh:
                shard = pickle.load(fh)
            shard_cache[run_seed] = shard
        gs = restore_state(shard[record_id])
        yield pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL)


def canonical(policy, action_tensor):
    """(action_type, sorted card-set) from a padded action vector."""
    action_type, card_indices = policy._labels_from_actions(action_tensor)
    a_type = int(action_type.item())
    picks = tuple(sorted(int(i) for i in card_indices.reshape(-1).tolist() if i >= 0))
    return a_type, picks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-a", type=Path, required=True)
    p.add_argument("--policy-b", type=Path, required=True)
    p.add_argument("--harvest-dir", type=Path, default=Path("data/harvest_s0"))
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    import torch

    pol_a = load_pointer(args.policy_a, args.device)
    pol_b = load_pointer(args.policy_b, args.device)
    env = HandPlayGymEnv(obs_version=2, action_version=2)

    records = sample_records(args.harvest_dir, args.n, args.seed)

    full_agree = 0
    type_agree = 0
    jeffrey_sum = 0.0
    value_gap_sum = 0.0
    n = 0
    by_type = {0: [0, 0], 1: [0, 0]}  # action_type -> [agree, total]

    with torch.no_grad():
        for blob in iter_snapshots(args.harvest_dir, records):
            obs, _ = env.reset(options={"snapshot": blob})
            ta, _ = pol_a.obs_to_tensor(obs)
            tb, _ = pol_b.obs_to_tensor(obs)

            act_a = pol_a.predict_deterministic(ta)
            act_b = pol_b.predict_deterministic(tb)

            ca = canonical(pol_a, act_a)
            cb = canonical(pol_b, act_b)

            v_a, logpA_aA, _ = pol_a.evaluate_actions(ta, act_a)
            v_b, logpB_bB, _ = pol_b.evaluate_actions(tb, act_b)
            _, logpB_aA, _ = pol_b.evaluate_actions(tb, act_a)
            _, logpA_bB, _ = pol_a.evaluate_actions(ta, act_b)

            jeffrey = 0.5 * (
                (float(logpA_aA) - float(logpB_aA)) + (float(logpB_bB) - float(logpA_bB))
            )
            jeffrey_sum += jeffrey
            value_gap_sum += abs(float(v_a) - float(v_b))

            t_ok = ca[0] == cb[0]
            f_ok = ca == cb
            type_agree += int(t_ok)
            full_agree += int(f_ok)
            by_type[ca[0]][1] += 1
            by_type[ca[0]][0] += int(f_ok)
            n += 1

    result = {
        "policy_a": str(args.policy_a),
        "policy_b": str(args.policy_b),
        "harvest_dir": str(args.harvest_dir),
        "n_states": n,
        "agreement_rate": full_agree / n,
        "type_agreement": type_agree / n,
        "mean_jeffrey_logprob_gap": jeffrey_sum / n,
        "mean_value_gap": value_gap_sum / n,
        "agreement_by_type": {
            "play(0)": (by_type[0][0] / by_type[0][1]) if by_type[0][1] else None,
            "discard(1)": (by_type[1][0] / by_type[1][1]) if by_type[1][1] else None,
            "n_play": by_type[0][1],
            "n_discard": by_type[1][1],
        },
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
