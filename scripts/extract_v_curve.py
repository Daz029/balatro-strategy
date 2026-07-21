"""Wave-0 harness: extract ``V_curve(ante, dollars)`` from the frozen s0 shop
critic (CLAUDE.md "Money/dollar handling"; ``docs/post-regen-training-plan.md``
section 3 "Terminal $ term").

This is an OFFLINE, one-shot extraction: counterfactual money-sweeps of the
harvested shop-state corpus (``data/harvest_s0``) through s0's value head
(``runs/shop_ppo/s0_a4_v4/best_model/best_model.zip`` -- the same MaskablePPO
checkpoint path ``harvest_s0_rollouts.py``'s own usage docstring already
uses for this final s0 stage), averaged per ``(ante, dollars)`` cell. s0's
reward was ``1{run won}``, so cell means are already in P(win) units -- no
extra scale hyperparameter, exactly as the money-handling design record
requires.

THE ONE HARD RULE (verbatim from the plan): "Counterfactuals edit engine
state, never obs vectors." Dollars is derived into >=5 obs feature families
(``shop_obs.py`` voucher affordability + raw ``shop_context`` dollars;
``observation.py`` per-item affordability, GC log-dollars, interest +
spendable-above-interest). Editing the obs vector directly would leave those
families mutually contradictory -- OOD garbage the critic would happily
score. So every sweep step is: restore the harvested engine blob -> mutate
``gs["dollars"]`` -> re-encode the FULL observation via
``jackdaw.env.shop_obs.build_shop_observation`` (the real env's encode path)
-> forward the frozen critic's value head. Nothing here ever touches an
already-encoded array.

Wave-0 scope only: this produces the artifact + gut-checks. Wiring the
result into ``HandPlayGymEnv``'s terminal-$ hook is wave 2 -- this script
does not touch that env.

Usage::

    uv run python scripts/extract_v_curve.py --max-states 50   # quick smoke
    uv run python scripts/extract_v_curve.py                   # full run

Consumers read the artifact via ``jackdaw.agents.v_curve.load_v_curve``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from harvest_restore import restore_state  # noqa: E402

from jackdaw.env.shop_obs import (  # noqa: E402
    D_SHOP_CONTEXT,
    D_SHOP_CONTEXT_S1,
    build_shop_observation,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# NOTE: the top-level `runs/shop_ppo/s0_a4_v4.zip` some docs mention is a
# whole-log-dir transfer archive, not itself a loadable MaskablePPO
# checkpoint -- the actual checkpoint (matching harvest_s0_rollouts.py's own
# --shop-policy usage example) lives one level down, inside the extracted
# run directory.
DEFAULT_CHECKPOINT = "runs/shop_ppo/s0_a4_v4/best_model/best_model.zip"
DEFAULT_HARVEST_DIR = "data/harvest_s0"
DEFAULT_OUT = "data/v_curve.json"
# Sensible sweep range: Credit Card allows -$20 debt (jokers.py comment); the
# harvested per-ante $ marginals (reductions.json) top out around $29 at
# ante 5 -- $60 gives headroom into economy-build territory the corpus
# under-samples rather than clipping the curve exactly where it matters.
DEFAULT_DOLLAR_MIN = -20
DEFAULT_DOLLAR_MAX = 60
DEFAULT_SOURCE_FILTER = "det"  # deterministic corpus = the deployed-distribution anchor
DEFAULT_BATCH_SIZE = 1024
DEFAULT_SEED = 0
DEFAULT_SPARSE_THRESHOLD = 30

# Tolerance on the HARD [0,1] range check -- the critic is a regression, not
# a clamp, so it can overshoot by float noise even in-distribution.
RANGE_EPS = 1e-3

ARTIFACT_VERSION = 1


def git_sha() -> str:
    """Current checkout SHA, or ``"unknown"`` outside a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 -- best-effort stamp, never fatal
        return "unknown"


# ---------------------------------------------------------------------------
# Corpus loading -- metadata is blob-free; blobs restore via harvest_restore
# ---------------------------------------------------------------------------


def load_shop_records(harvest_dir: Path, source_filter: str | None) -> list[dict[str, Any]]:
    """Read ``metadata.jsonl``, keeping only ``kind == "shop"`` rows.

    Metadata-only: never unpickles a blob (mirrors the harvest's own
    blob-free-query rule).
    """
    path = Path(harvest_dir) / "metadata.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("kind") != "shop":
                continue
            if source_filter and row.get("source") != source_filter:
                continue
            rows.append(row)
    return rows


def subsample_records(
    records: list[dict[str, Any]], max_states: int | None, seed: int
) -> list[dict[str, Any]]:
    """Deterministic seeded subsample -- a quick ``--max-states`` pass must
    be reproducible, not a fresh random cut every invocation."""
    if max_states is None or max_states >= len(records):
        return list(records)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=max_states, replace=False)
    idx.sort()
    return [records[i] for i in idx]


def iter_restored_states(
    records: Iterable[dict[str, Any]], blob_dir: Path
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield ``(row, gs)`` for each metadata row, restoring via its shard.

    Records should be pre-sorted by ``run_seed`` (:func:`sorted_for_locality`)
    so each per-run blob shard is opened exactly once -- the same locality
    trick ``generate_hand_demos.py``'s ``BlobStore`` uses. Restoration always
    goes through ``harvest_restore.restore_state`` (never a bare
    ``pickle.loads``) so capture-skew repairs apply.
    """
    current_seed: str | None = None
    shard: dict[str, bytes] = {}
    for row in records:
        run_seed = row["run_seed"]
        if run_seed != current_seed:
            path = Path(blob_dir) / f"{run_seed}.pkl"
            with path.open("rb") as fh:
                shard = pickle.load(fh)
            current_seed = run_seed
        blob = shard[row["record_id"]]
        gs = restore_state(blob)
        yield row, gs


def sorted_for_locality(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort by ``(run_seed, record_id)`` so :func:`iter_restored_states`
    reads each blob shard once."""
    return sorted(records, key=lambda r: (str(r["run_seed"]), str(r["record_id"])))


# ---------------------------------------------------------------------------
# Critic -- the only thing that ever sees an encoded observation
# ---------------------------------------------------------------------------


class Critic(Protocol):
    """``obs_batch -> per-example value``. ``obs_batch`` is a dict of
    STACKED arrays (leading dim = batch); tests substitute a stub here so
    the sweep-loop mechanics can be verified with no checkpoint at all."""

    def values(self, obs_batch: dict[str, np.ndarray]) -> np.ndarray: ...


class CriticForwardError(RuntimeError):
    """A batch-level failure that must abort the sweep, not skip one state."""


class MaskablePPOCritic:
    """Wraps a saved MaskablePPO ``.zip``; forwards batched obs through
    ``predict_values`` ONLY -- the action head is never touched, since the
    V_curve is a pure value-function query."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(checkpoint_path), device=device)
        self._model.policy.set_training_mode(False)
        self.s1_schema = _s1_schema_from_observation_space(self._model.observation_space)

    def values(self, obs_batch: dict[str, np.ndarray]) -> np.ndarray:
        import torch
        from stable_baselines3.common.utils import obs_as_tensor

        tensor_obs = obs_as_tensor(obs_batch, self._model.policy.device)
        with torch.no_grad():
            values = self._model.policy.predict_values(tensor_obs)
        return values.squeeze(-1).cpu().numpy()


def _s1_schema_from_observation_space(observation_space: Any) -> bool:
    """Recover the encoder schema stored with a checkpoint."""
    context_width = int(observation_space["shop_context"].shape[0])
    if context_width == D_SHOP_CONTEXT:
        return False
    if context_width == D_SHOP_CONTEXT_S1:
        return True
    raise ValueError(
        "checkpoint has an unrecognized shop_context width: "
        f"{context_width} (expected {D_SHOP_CONTEXT} or {D_SHOP_CONTEXT_S1})"
    )


def stack_obs(obs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack a list of single-state obs dicts into one batched dict."""
    keys = obs_list[0].keys()
    return {k: np.stack([o[k] for o in obs_list], axis=0) for k in keys}


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


@dataclass
class CellAccumulator:
    """Running per-``(ante, dollar)`` mean + raw-value range stats.

    Streaming so a full 25k-state x ~80-dollar sweep (~2M forwards) never
    needs to hold every raw value in memory -- only one running sum/count
    per cell plus a handful of scalars for the range check.
    """

    sums: dict[tuple[int, int], float] = field(default_factory=dict)
    counts: dict[tuple[int, int], int] = field(default_factory=dict)
    raw_min: float = float("inf")
    raw_max: float = float("-inf")
    n_out_of_range: int = 0
    n_total: int = 0

    def add(self, ante: int, dollar: int, value: float) -> None:
        key = (ante, dollar)
        self.sums[key] = self.sums.get(key, 0.0) + value
        self.counts[key] = self.counts.get(key, 0) + 1
        self.raw_min = min(self.raw_min, value)
        self.raw_max = max(self.raw_max, value)
        self.n_total += 1
        if value < -RANGE_EPS or value > 1.0 + RANGE_EPS:
            self.n_out_of_range += 1

    def add_batch(self, keys: list[tuple[int, int]], values: np.ndarray) -> None:
        for (ante, dollar), v in zip(keys, values, strict=True):
            self.add(ante, dollar, float(v))

    def cells(self) -> dict[int, dict[int, dict[str, Any]]]:
        """Nested ``{ante: {dollar: {"mean": ..., "count": ...}}}``."""
        out: dict[int, dict[int, dict[str, Any]]] = {}
        for (ante, dollar), total in self.sums.items():
            count = self.counts[(ante, dollar)]
            out.setdefault(ante, {})[dollar] = {"mean": total / count, "count": count}
        return out


# ---------------------------------------------------------------------------
# Sweep loop -- state-edit-then-full-reencode, batched
# ---------------------------------------------------------------------------


def run_sweep(
    records: list[dict[str, Any]],
    blob_dir: Path,
    dollar_values: list[int],
    critic: Critic,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_every: int | None = None,
    s1_schema: bool | None = None,
) -> tuple[CellAccumulator, list[dict[str, Any]]]:
    """Sweep every restored state across ``dollar_values``, batched through
    ``critic``. Returns ``(accumulator, failures)`` -- a restore/encode
    failure on one record is logged and skipped, never fatal to the run
    (the repo's standard skip-not-fatal convention).
    """
    acc = CellAccumulator()
    failures: list[dict[str, Any]] = []
    if s1_schema is None:
        # The production critic records the schema from its checkpoint.  Keep
        # the default s0 path for lightweight test critics that do not expose
        # a schema flag.
        s1_schema = bool(getattr(critic, "s1_schema", False))

    buffer_obs: list[dict[str, np.ndarray]] = []
    buffer_keys: list[tuple[int, int]] = []

    def flush() -> None:
        if not buffer_obs:
            return
        try:
            batch = stack_obs(buffer_obs)
            values = critic.values(batch)
            acc.add_batch(buffer_keys, values)
        except Exception as exc:  # noqa: BLE001 -- preserve the original cause
            raise CriticForwardError(
                "critic forward failed; aborting sweep instead of retrying the "
                f"failed batch (size={len(buffer_obs)}, keys={buffer_keys[0]}.."
                f"{buffer_keys[-1]}, s1_schema={s1_schema})"
            ) from exc
        buffer_obs.clear()
        buffer_keys.clear()

    ordered = sorted_for_locality(records)
    n_states_done = 0
    start = time.time()
    for row, gs in iter_restored_states(ordered, blob_dir):
        ante = int(row["ante"])
        try:
            for dollar in dollar_values:
                gs["dollars"] = dollar  # <-- the ONE mutation: engine state, not obs
                obs = build_shop_observation(
                    gs, None, s1_schema=s1_schema
                )  # <-- full re-encode, every time
                buffer_obs.append(obs)
                buffer_keys.append((ante, dollar))
                if len(buffer_obs) >= batch_size:
                    flush()
        except CriticForwardError:
            raise
        except Exception as exc:  # noqa: BLE001 -- one bad record must not kill the run
            failures.append({"record_id": row.get("record_id"), "error": repr(exc)})
        n_states_done += 1
        if progress_every and n_states_done % progress_every == 0:
            elapsed = time.time() - start
            print(
                f"[v_curve] {n_states_done}/{len(ordered)} states "
                f"({elapsed:.1f}s elapsed, {acc.n_total} forwards)"
            )
    flush()
    return acc, failures


# ---------------------------------------------------------------------------
# Gut-checks
# ---------------------------------------------------------------------------


def range_summary(acc: CellAccumulator) -> dict[str, Any]:
    """HARD check: raw values should sit in [0,1] (s0's reward is
    ``1{win}``). Reported, not enforced -- a regression-net critic can
    overshoot by float noise even fully in-distribution."""
    return {
        "min_observed": acc.raw_min if acc.n_total else None,
        "max_observed": acc.raw_max if acc.n_total else None,
        "n_out_of_range": acc.n_out_of_range,
        "n_total": acc.n_total,
        "fraction_out_of_range": (acc.n_out_of_range / acc.n_total) if acc.n_total else 0.0,
        "tolerance": RANGE_EPS,
    }


def monotonicity_violations(
    cells: dict[int, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """HARD check: per ante, mean V should be weakly nondecreasing in
    dollars. Every decreasing adjacent pair is reported (magnitude
    included) -- small decreases are noise, but they must be VISIBLE, not
    silently swallowed."""
    violations: list[dict[str, Any]] = []
    for ante, dollar_map in sorted(cells.items()):
        dollars_sorted = sorted(dollar_map)
        for lo, hi in zip(dollars_sorted, dollars_sorted[1:], strict=False):
            v_lo = dollar_map[lo]["mean"]
            v_hi = dollar_map[hi]["mean"]
            delta = v_hi - v_lo
            if delta < 0.0:
                violations.append(
                    {
                        "ante": ante,
                        "dollar_from": lo,
                        "dollar_to": hi,
                        "value_from": v_lo,
                        "value_to": v_hi,
                        "delta": delta,
                    }
                )
    return violations


def sparse_antes(
    cells: dict[int, dict[int, dict[str, Any]]], threshold: int
) -> list[dict[str, Any]]:
    """Flag whole antes whose sample count is thin (every dollar in the
    sweep shares one count per ante -- one state contributes to every
    swept dollar -- so sparsity is really an ante-level fact, e.g.
    negative-dollar debt states or ante 7's small population)."""
    out: list[dict[str, Any]] = []
    for ante, dollar_map in sorted(cells.items()):
        counts = [cell["count"] for cell in dollar_map.values()]
        if not counts:
            continue
        min_count = min(counts)
        max_count = max(counts)
        if min_count < threshold:
            out.append({"ante": ante, "min_count": min_count, "max_count": max_count})
    return out


def kink_diagnostic(
    cells: dict[int, dict[int, dict[str, Any]]], dollar_min: int, dollar_max: int
) -> dict[int, list[dict[str, Any]]]:
    """SOFT, diagnostic-only: discrete second difference of mean V around
    every $5 boundary, per ante. Per the plan's 2026-07-17 downgrade: a
    sharp kink CONFIRMS the critic learned interest, but its absence is
    ambiguous (future-round interest is smeared by whatever the policy
    does with money before the next cashout) -- so this must never fail
    anything, only be reported."""
    out: dict[int, list[dict[str, Any]]] = {}
    for ante, dollar_map in sorted(cells.items()):
        entries: list[dict[str, Any]] = []
        for dollar in range(dollar_min, dollar_max + 1):
            if dollar % 5 != 0:
                continue
            if (dollar - 1) not in dollar_map or (dollar + 1) not in dollar_map:
                continue
            v_lo = dollar_map[dollar - 1]["mean"]
            v_mid = dollar_map[dollar]["mean"]
            v_hi = dollar_map[dollar + 1]["mean"]
            second_diff = v_hi - 2.0 * v_mid + v_lo
            entries.append({"dollar": dollar, "second_diff": second_diff})
        if entries:
            out[ante] = entries
    return out


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


def build_artifact(
    cells: dict[int, dict[int, dict[str, Any]]],
    acc: CellAccumulator,
    *,
    checkpoint: str,
    harvest_dir: str,
    source_filter: str | None,
    dollar_min: int,
    dollar_max: int,
    n_source_records: int,
    n_states: int,
    max_states: int | None,
    seed: int,
    batch_size: int,
    sparse_threshold: int,
    n_failures: int,
) -> dict[str, Any]:
    metadata = {
        "artifact_version": ARTIFACT_VERSION,
        "checkpoint": checkpoint,
        "git_sha": git_sha(),
        "harvest_dir": harvest_dir,
        "source_filter": source_filter,
        "dollar_min": dollar_min,
        "dollar_max": dollar_max,
        "n_source_records": n_source_records,
        "n_states": n_states,
        "max_states": max_states,
        "seed": seed,
        "batch_size": batch_size,
        "sparse_threshold": sparse_threshold,
        "n_failures": n_failures,
        "created_at": datetime.now(UTC).isoformat(),
    }
    gut_checks = {
        "hard": {
            "range": range_summary(acc),
            "monotonicity_violations": monotonicity_violations(cells),
        },
        "sparse": {
            "threshold": sparse_threshold,
            "flagged_antes": sparse_antes(cells, sparse_threshold),
        },
        "soft_kink_diagnostic": {
            str(ante): entries
            for ante, entries in kink_diagnostic(cells, dollar_min, dollar_max).items()
        },
    }
    gut_checks["hard"]["n_monotonicity_violations"] = len(
        gut_checks["hard"]["monotonicity_violations"]
    )
    return {
        "metadata": metadata,
        "cells": {
            str(ante): {str(dollar): cell for dollar, cell in dollar_map.items()}
            for ante, dollar_map in cells.items()
        },
        "gut_checks": gut_checks,
    }


def print_summary(artifact: dict[str, Any]) -> None:
    meta = artifact["metadata"]
    gc = artifact["gut_checks"]
    print("=" * 72)
    print("V_curve extraction summary")
    print("=" * 72)
    print(f"checkpoint:        {meta['checkpoint']}")
    print(f"git_sha:           {meta['git_sha']}")
    print(f"source_filter:     {meta['source_filter']}")
    print(f"n_states:          {meta['n_states']} (of {meta['n_source_records']} candidates)")
    print(f"dollar sweep:      [{meta['dollar_min']}, {meta['dollar_max']}]")
    print(f"n_failures:        {meta['n_failures']}")
    print("-" * 72)
    rng = gc["hard"]["range"]
    print(
        f"HARD range check:  observed [{rng['min_observed']}, {rng['max_observed']}], "
        f"{rng['n_out_of_range']}/{rng['n_total']} out of [0,1] (tol {rng['tolerance']})"
    )
    n_mono = gc["hard"]["n_monotonicity_violations"]
    print(f"HARD monotonicity: {n_mono} decreasing adjacent-dollar pair(s)")
    for v in gc["hard"]["monotonicity_violations"][:10]:
        print(
            f"    ante={v['ante']:>2} ${v['dollar_from']:>4} -> ${v['dollar_to']:>4} "
            f"delta={v['delta']:+.4f} ({v['value_from']:.4f} -> {v['value_to']:.4f})"
        )
    if n_mono > 10:
        print(f"    ... and {n_mono - 10} more")
    sparse = gc["sparse"]["flagged_antes"]
    print(f"SPARSE (< {gc['sparse']['threshold']} samples): {len(sparse)} ante(s) flagged")
    for s in sparse:
        print(f"    ante={s['ante']:>2} min_count={s['min_count']} max_count={s['max_count']}")
    print("-" * 72)
    print("Sample cells:")
    cells = artifact["cells"]
    for ante in sorted(cells, key=int)[:4]:
        dollar_map = cells[ante]
        sample_dollars = sorted((int(d) for d in dollar_map), key=abs)[:5]
        for d in sorted(sample_dollars):
            cell = dollar_map[str(d)]
            print(f"    ante={ante} $ ={d:>4}  mean={cell['mean']:.4f}  count={cell['count']}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--harvest-dir", type=Path, default=Path(DEFAULT_HARVEST_DIR))
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--dollar-min", type=int, default=DEFAULT_DOLLAR_MIN)
    parser.add_argument("--dollar-max", type=int, default=DEFAULT_DOLLAR_MAX)
    parser.add_argument("--source-filter", default=DEFAULT_SOURCE_FILTER)
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="subsample cap for a quick pass (deterministic, seeded by --seed)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sparse-threshold", type=int, default=DEFAULT_SPARSE_THRESHOLD)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="print a progress line every N states (0 disables)",
    )
    args = parser.parse_args()

    if args.dollar_min > args.dollar_max:
        parser.error("--dollar-min must be <= --dollar-max")

    records = load_shop_records(args.harvest_dir, args.source_filter)
    n_source_records = len(records)
    if n_source_records == 0:
        parser.error(
            f"no kind=='shop' records with source={args.source_filter!r} "
            f"found under {args.harvest_dir}"
        )
    selected = subsample_records(records, args.max_states, args.seed)

    print(
        f"[v_curve] {len(selected)}/{n_source_records} shop states selected "
        f"(source={args.source_filter!r}, max_states={args.max_states})"
    )
    print(f"[v_curve] loading critic from {args.checkpoint}")
    critic = MaskablePPOCritic(args.checkpoint, device=args.device)

    dollar_values = list(range(args.dollar_min, args.dollar_max + 1))
    blob_dir = Path(args.harvest_dir) / "blobs"
    acc, failures = run_sweep(
        selected,
        blob_dir,
        dollar_values,
        critic,
        batch_size=args.batch_size,
        progress_every=args.progress_every or None,
    )
    if failures:
        print(f"[v_curve] WARNING: {len(failures)} record(s) failed to restore/encode:")
        for f in failures[:10]:
            print(f"    {f['record_id']}: {f['error']}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")

    cells = acc.cells()
    artifact = build_artifact(
        cells,
        acc,
        checkpoint=str(args.checkpoint),
        harvest_dir=str(args.harvest_dir),
        source_filter=args.source_filter,
        dollar_min=args.dollar_min,
        dollar_max=args.dollar_max,
        n_source_records=n_source_records,
        n_states=len(selected) - len(failures),
        max_states=args.max_states,
        seed=args.seed,
        batch_size=args.batch_size,
        sparse_threshold=args.sparse_threshold,
        n_failures=len(failures),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"[v_curve] wrote artifact to {args.out}")

    print_summary(artifact)


if __name__ == "__main__":
    main()
