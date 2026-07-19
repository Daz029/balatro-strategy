"""A3 -- discard-bias fingerprint eval (+ flush/straight archetype decomposition).

Decides whether B6 (banked-discard credit in the solver's future-hand
estimator) gets built. The solver's documented label bias: future hands are
valued play-only, so discards BANKED for later turns are credited at zero
-> labels tilt toward spending discards early. The fingerprint is a
CONJUNCTION of two signals (handoff pitfall #15 -- either alone is NOT
enough):

  1. LOCATE  -- ceiling recovery (policy clear rate / solver-expected
     clears, from shard `p_clear` labels -- the free solver-ceiling trick)
     bucketed by starting ``(hands_left, discards_left)``. Symptom =
     recovery in discards>=2 buckets materially below discards==0 buckets
     (>5 recovery points, bootstrap CI excluding zero). Necessary but NOT
     sufficient (also consistent with plain learning weakness).
  2. ATTRIBUTE -- the bias has a known SIGN: it can only push toward
     discarding TOO EAGERLY. Bias survived = the policy's first-decision
     discard rate sits at/above the teacher labels' discard rate in the
     deficit buckets. Directionless error or UNDER-discarding = solver
     exonerated; the remedy is training, not labels. Do NOT build B6.

Greedy (`GreedyHandPolicy`) runs as a control on stage 1 only (it discards
chaff on weak hands and is joker-blind -- not a clean no-discard floor, per
the decision record). The flush/straight/pair archetype decomposition of
recovery rides the same pass; it gates nothing (it calibrates how much
recovery the B2 hand-potential features should buy back).

Teacher-side stats come from the existing demo shards (label-mean p_clear,
label discard frequency) -- same distribution the policy is evaluated on,
zero solver cost. Buckets recover exactly from GC[13]/GC[14] (x10);
archetypes from GC[227] flush_proximity / GC[228] straight_proximity
(>=0.8 = 4-to-a-line), identical encodings on both sides.

Usage (paths point at the MAIN checkout -- runs/ and data/ are gitignored
and live only there)::

    uv run python scripts/fingerprint_discard_bias.py \
        --policy C:/Code/balatro-strategy/runs/hand_ppo/hand_ppo_2000000_steps.zip \
        --data-dir C:/Code/balatro-strategy/data/hand_agent_demos \
        --n-episodes 800 --output data/fingerprint_a3.json
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

from eval_hand_policy import eval_seeds, load_policy  # noqa: E402

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy  # noqa: E402
from jackdaw.agents.hand_action_space import NUM_COMBOS  # noqa: E402
from jackdaw.engine.actions import PlayHand as EnginePlayHand  # noqa: E402
from jackdaw.env.action_space import ActionType  # noqa: E402
from jackdaw.env.hand_play_gym import (  # noqa: E402
    POINTER_MAX_PICKS,
    POINTER_STOP_INDEX,
    HandPlayGymEnv,
)

# v2 appends 21 global-context features to the v1 vector, so these v1 indices
# remain valid for pointer observations.
GC_HANDS_LEFT = 13  # x10 normalization -- round() recovers the integer
GC_DISCARDS_LEFT = 14
GC_FLUSH_PROX = 227
GC_STRAIGHT_PROX = 228
LINE_THRESHOLD = 0.8 - 1e-6  # 4-to-a-line on the /5 proximity features

STAGES = ("stage1_no_jokers", "stage2_curated", "stage3_full", "stage4_boss")


def archetype(flush_prox: float, straight_prox: float) -> str:
    if flush_prox >= LINE_THRESHOLD:
        return "flush"
    if straight_prox >= LINE_THRESHOLD:
        return "straight"
    return "pair"


# ---------------------------------------------------------------------------
# Teacher side (shards)
# ---------------------------------------------------------------------------


def teacher_stats(stage_dir: Path) -> dict:
    """Per-bucket and per-archetype label stats from one stage's shards."""
    buckets: dict[tuple[int, int], dict] = {}
    arch: dict[str, dict] = {a: {"n": 0, "sum_p": 0.0} for a in ("flush", "straight", "pair")}
    shards = sorted(stage_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards under {stage_dir}")
    for shard in shards:
        with np.load(shard, allow_pickle=False) as z:
            gc = z["global_context"]
            action_type = z["action_type"]
            p_clear = z["p_clear"]
        hands = np.rint(gc[:, GC_HANDS_LEFT] * 10).astype(int)
        discards = np.rint(gc[:, GC_DISCARDS_LEFT] * 10).astype(int)
        for i in range(len(p_clear)):
            key = (int(hands[i]), int(discards[i]))
            b = buckets.setdefault(key, {"n": 0, "sum_p": 0.0, "n_discard": 0})
            b["n"] += 1
            b["sum_p"] += float(p_clear[i])
            b["n_discard"] += int(action_type[i] == 1)
            a = arch[archetype(float(gc[i, GC_FLUSH_PROX]), float(gc[i, GC_STRAIGHT_PROX]))]
            a["n"] += 1
            a["sum_p"] += float(p_clear[i])
    return {"buckets": buckets, "archetypes": arch}


# ---------------------------------------------------------------------------
# Policy side (episodes)
# ---------------------------------------------------------------------------


class _GreedyGymPolicy:
    """GreedyHandPolicy adapted to the gym loop.

    The fingerprint uses the h1 stage presets, whose hand-size tail can create
    9-12-card hands. Emit the pointer action format so the control can address
    every live hand position; the frozen v1/436-action space only reaches
    positions 0-7.
    """

    obs_version = 2
    action_version = 2

    def __init__(self, env: HandPlayGymEnv) -> None:
        self._env = env
        self._greedy = GreedyHandPolicy()

    def act(self, obs) -> np.ndarray:
        engine_action = self._greedy(self._env._adapter.raw_state)
        action_type = (
            int(ActionType.PlayHand)
            if isinstance(engine_action, EnginePlayHand)
            else int(ActionType.Discard)
        )
        card_indices = tuple(sorted(int(i) for i in engine_action.card_indices))
        if not 1 <= len(card_indices) <= POINTER_MAX_PICKS:
            raise ValueError(f"greedy policy selected {len(card_indices)} cards")
        tokens = card_indices + (POINTER_STOP_INDEX,) * (
            POINTER_MAX_PICKS - len(card_indices)
        )
        return np.asarray((action_type, *tokens), dtype=np.int64)


def run_episodes(policy_factory, config, n_episodes: int) -> list[dict]:
    """One row per episode: starting bucket, archetype, first-decision
    action kind, cleared. Uses the reserved EVAL_ seed suite."""
    probe_env = HandPlayGymEnv(config=config)
    policy = policy_factory(probe_env)
    obs_version = getattr(policy, "obs_version", 1)
    action_version = getattr(policy, "action_version", 1)
    if action_version == 1:
        env = probe_env
    else:
        env = HandPlayGymEnv(
            config=config,
            obs_version=obs_version,
            action_version=action_version,
        )
        policy = policy_factory(env)
    rows: list[dict] = []
    for seed in eval_seeds(n_episodes):
        obs, info = env.reset(options={"episode_seed": seed})
        gc = obs["global_context"]
        row = {
            "seed": seed,
            "hands_left": int(round(float(gc[GC_HANDS_LEFT]) * 10)),
            "discards_left": int(round(float(gc[GC_DISCARDS_LEFT]) * 10)),
            "archetype": archetype(float(gc[GC_FLUSH_PROX]), float(gc[GC_STRAIGHT_PROX])),
        }
        first = True
        while True:
            if action_version == 1:
                action = policy.act(obs, info["action_mask"])
            else:
                action = policy.act(obs)
            if first:
                if action_version == 1:
                    row["first_discard"] = bool(action >= NUM_COMBOS)
                else:
                    row["first_discard"] = bool(int(action[0]) == int(ActionType.Discard))
                first = False
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                row["cleared"] = bool(info["balatro/cleared"])
                break
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _pool(rows: list[dict], teacher_buckets: dict, pred) -> dict:
    """Pooled recovery over episodes matching `pred`: observed clears /
    solver-expected clears (each episode's expectation = its bucket's
    teacher label-mean p_clear). Teacher means are treated as fixed --
    shard n per bucket is 100-1000x the episode count."""
    sel = [r for r in rows if pred(r)]
    clears = sum(r["cleared"] for r in sel)
    expected = 0.0
    n_disc = 0
    teacher_disc_expected = 0.0
    for r in sel:
        b = teacher_buckets.get((r["hands_left"], r["discards_left"]))
        if b and b["n"] > 0:
            expected += b["sum_p"] / b["n"]
            teacher_disc_expected += b["n_discard"] / b["n"]
        n_disc += int(r.get("first_discard", False))
    n = len(sel)
    return {
        "n": n,
        "clears": clears,
        "expected_clears": expected,
        "recovery": (clears / expected) if expected > 0 else None,
        "policy_discard_rate": (n_disc / n) if n else None,
        "teacher_discard_rate": (teacher_disc_expected / n) if n else None,
    }


def _bootstrap_ci(rows, teacher_buckets, stat_fn, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(rows)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = [rows[i] for i in idx]
        v = stat_fn(sample, teacher_buckets)
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _deficit_stat(rows, teacher_buckets):
    d0 = _pool(rows, teacher_buckets, lambda r: r["discards_left"] == 0)
    d2 = _pool(rows, teacher_buckets, lambda r: r["discards_left"] >= 2)
    if d0["recovery"] is None or d2["recovery"] is None:
        return None
    return (d0["recovery"] - d2["recovery"]) * 100  # recovery points


def _discard_gap_stat(rows, teacher_buckets):
    d2 = _pool(rows, teacher_buckets, lambda r: r["discards_left"] >= 2)
    if d2["policy_discard_rate"] is None or d2["teacher_discard_rate"] is None:
        return None
    return d2["policy_discard_rate"] - d2["teacher_discard_rate"]


def analyze(rows: list[dict], teacher_buckets: dict) -> dict:
    table = {}
    for h in range(1, 5):
        for d in range(0, 4):
            cell = _pool(
                rows, teacher_buckets,
                lambda r, h=h, d=d: r["hands_left"] == h and r["discards_left"] == d,
            )
            if cell["n"]:
                tb = teacher_buckets.get((h, d))
                cell["teacher_mean_p_clear"] = tb["sum_p"] / tb["n"] if tb else None
                table[f"h{h}_d{d}"] = cell

    deficit = _deficit_stat(rows, teacher_buckets)
    deficit_ci = _bootstrap_ci(rows, teacher_buckets, _deficit_stat)
    gap = _discard_gap_stat(rows, teacher_buckets)
    gap_ci = _bootstrap_ci(rows, teacher_buckets, _discard_gap_stat)

    signal1 = (
        deficit is not None and deficit > 5 and deficit_ci[0] is not None and deficit_ci[0] > 0
    )
    # Bias survived only if the policy discards AT/ABOVE the teacher's
    # (known-inflated) rate; a CI entirely below zero = under-discarding =
    # solver exonerated outright.
    signal2 = gap is not None and gap >= 0
    exonerated_by_sign = gap_ci[1] is not None and gap_ci[1] < 0

    return {
        "bucket_table": table,
        "pooled_d0": _pool(rows, teacher_buckets, lambda r: r["discards_left"] == 0),
        "pooled_d2plus": _pool(rows, teacher_buckets, lambda r: r["discards_left"] >= 2),
        "deficit_recovery_points": deficit,
        "deficit_ci95": deficit_ci,
        "discard_rate_gap_d2plus": gap,
        "discard_rate_gap_ci95": gap_ci,
        "signal1_locate": bool(signal1),
        "signal2_attribute_bias_survived": bool(signal2),
        "under_discarding_exoneration": bool(exonerated_by_sign),
        "verdict": "TRIGGERED" if (signal1 and signal2 and not exonerated_by_sign) else "CLEARED",
    }


def archetype_decomposition(rows: list[dict], teacher_buckets: dict) -> dict:
    return {
        a: _pool(rows, teacher_buckets, lambda r, a=a: r["archetype"] == a)
        for a in ("flush", "straight", "pair")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True, help="h0.5 PPO .zip")
    parser.add_argument("--data-dir", type=Path, required=True, help="demo shard root")
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--n-episodes", type=int, default=800, help="per stage")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("data/fingerprint_a3.json"))
    args = parser.parse_args()

    from generate_hand_demos import stage_presets

    presets = stage_presets()
    stages = args.stages.split(",")
    policy = load_policy(args.policy, args.device)

    report: dict = {"policy": str(args.policy), "n_episodes_per_stage": args.n_episodes,
                    "stages": {}}
    all_rows: list[dict] = []
    combined_teacher: dict[tuple[int, int], dict] = {}

    for stage in stages:
        t0 = time.time()
        print(f"[{stage}] teacher stats from shards...")
        teacher = teacher_stats(args.data_dir / stage)
        print(f"[{stage}] rolling out {args.n_episodes} episodes...")
        rows = run_episodes(lambda env: policy, presets[stage].config, args.n_episodes)
        stage_report = analyze(rows, teacher["buckets"])
        stage_report["archetypes"] = {
            "policy": archetype_decomposition(rows, teacher["buckets"]),
            "teacher_mean_p_clear": {
                a: (v["sum_p"] / v["n"] if v["n"] else None)
                for a, v in teacher["archetypes"].items()
            },
        }
        stage_report["wall_seconds"] = round(time.time() - t0, 1)
        report["stages"][stage] = stage_report
        print(f"[{stage}] verdict={stage_report['verdict']} "
              f"deficit={stage_report['deficit_recovery_points']} "
              f"gap={stage_report['discard_rate_gap_d2plus']}")

        # Merge for the combined verdict: episodes keep their own stage's
        # teacher expectations, so buckets merge by summation.
        for r in rows:
            r["_stage"] = stage
        all_rows.extend(rows)
        for key, b in teacher["buckets"].items():
            cb = combined_teacher.setdefault(key, {"n": 0, "sum_p": 0.0, "n_discard": 0})
            for f in ("n", "sum_p", "n_discard"):
                cb[f] += b[f]

        if stage == "stage1_no_jokers":
            print("[stage1] greedy control...")
            greedy_rows = run_episodes(
                lambda env: _GreedyGymPolicy(env), presets[stage].config, args.n_episodes
            )
            report["greedy_control_stage1"] = analyze(greedy_rows, teacher["buckets"])

    # NOTE: the combined pool weights each stage by its episode count and
    # uses summed teacher buckets -- a coarse pooled view; per-stage tables
    # above are the primary read.
    report["combined"] = analyze(all_rows, combined_teacher)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== A3 fingerprint ===")
    for stage in stages:
        s = report["stages"][stage]
        deficit = s["deficit_recovery_points"]
        print(f"{stage}: verdict={s['verdict']} "
              f"deficit={deficit and round(deficit, 1)}pts "
              f"CI={s['deficit_ci95']} discard_gap={s['discard_rate_gap_d2plus']} "
              f"CI={s['discard_rate_gap_ci95']}")
    c = report["combined"]
    print(f"COMBINED: verdict={c['verdict']} "
          f"deficit={c['deficit_recovery_points'] and round(c['deficit_recovery_points'], 1)}pts "
          f"discard_gap={c['discard_rate_gap_d2plus']}")
    print(f"report -> {args.output}")


if __name__ == "__main__":
    main()
