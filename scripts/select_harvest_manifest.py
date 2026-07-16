"""C1 of the pre-regen build plan: choose WHICH harvested records get labeled,
and write that choice down as a reproducible, versioned manifest.

The harvest (A1/A2) banked ~38k hand-turn states under a deliberate "capture
EVERYTHING, thin later" policy. Labeling is the expensive step (~12s/example
through the exact solver), so only ~8k of them can be afforded. This script is
that thinning — and it exists as a SEPARATE artifact-producing step, rather
than as inline subsampling inside the labeler, precisely so the subsampling
parameters stay re-runnable QUERIES instead of irreversible capture-time
commitments. Want a different ante mix? Re-run C1. Never re-harvest.

Reads the metadata table ONLY: no blob is ever unpickled here (the metadata
was designed to carry every stratification field for exactly this reason), so
this script has no engine dependency and runs in a second.

Selection (all knobs are CLI-exposed; the defaults are the locked spec):

* ``kind == "hand"`` only. Shop records live in the same table but feed the
  offline ``V_curve`` money-sweep, not BC.
* **det:sampled = 75:25** (``--det-frac``). Det-heavy by the deployed-
  distribution anchor logic: h1 targets the distribution s0's ARGMAX policy
  actually induces; the sampled pass buys coverage breadth around it.
* **<=3 records per (run, ante)** (``--per-ante-cap``) and **<=8 per run**
  (``--per-run-cap``). This is the "ante-stratified" requirement, and its job
  is WITHIN-RUN CORRELATION CONTROL: consecutive hand-turns of one run share a
  deck, a build and a blind, so 15 turns from one ante-1 run are nowhere near
  15 independent examples.
* When the per-run cap binds, slots are filled ROUND-ROBIN over the antes that
  run reached, DEEPEST ante first. Deepest-first matters because deep antes are
  scarce (the det corpus has 15,461 records at ante 1 and 20 at ante 7): an
  ante-1-first fill would spend a deep run's whole budget before ever reaching
  the states that make it interesting.
* If a source's candidate pool still exceeds its target, it is thinned
  ante-stratified by largest-remainder, which preserves the pool's ante mix
  exactly rather than letting a uniform draw randomly gut a scarce deep bucket.

NOTE ON THE ANTE MARGINAL: the caps above deliberately do NOT flatten the ante
distribution toward uniform. Per CLAUDE.md, deep-ante coverage for h1 is the
retained domain-randomized stages 1-4's job; the harvest's job is REALISM for
the early antes s0 actually reaches. Flattening here would fight that division
of labor (and the supply can't support it regardless -- ante 7 has 20 records).
The caps reshape the marginal from "turns spent at each ante" toward "runs that
reached each ante", which is the honest correlation-controlled view.

Determinism is a hard requirement (``--seed``): the same metadata table and the
same parameters must always emit a byte-identical manifest, so the manifest can
be checked in and a dataset traced back to the exact query that produced it.
Every dict iteration below is over SORTED keys for that reason.

The manifest is consumed by C2 (``generate_hand_demos.py``'s snapshot-fed
front-end), which partitions workers over the manifest ORDER. That order is
therefore shuffled (seeded) rather than left grouped by run/ante, so the deep,
joker-heavy, slow-to-solve states spread evenly across workers instead of
piling into the last one.

Usage::

    uv run python scripts/select_harvest_manifest.py \
        --metadata data/harvest_s0/metadata.jsonl \
        --output manifests/h1_harvested.json \
        --n-total 8000 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MANIFEST_VERSION = 1

# Locked defaults (docs/pre-regen-handoff.md, C1).
DEFAULT_N_TOTAL = 8_000
DEFAULT_DET_FRAC = 0.75
DEFAULT_PER_RUN_CAP = 8
DEFAULT_PER_ANTE_CAP = 3

# Seed prefixes the harvest reserves. EVAL_ is the held-out policy-eval suite:
# a single EVAL_ record reaching the manifest would leak the held-out set into
# h1's training data (pitfall #6), so it is checked, not assumed.
FORBIDDEN_SEED_PREFIX = "EVAL_"

SOURCES = ("det", "sampled")


class SelectionError(Exception):
    """Raised when the corpus cannot honour the requested selection."""


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Read the harvest metadata table (JSONL, one row per captured record)."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hand_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Labelable records only, with the seed-hygiene guard applied.

    Shop records share the table but are V_curve input, not BC input.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != "hand":
            continue
        run_seed = str(row.get("run_seed", ""))
        if run_seed.startswith(FORBIDDEN_SEED_PREFIX):
            raise SelectionError(
                f"record {row.get('record_id')!r} carries a held-out {FORBIDDEN_SEED_PREFIX} "
                "seed; harvesting the eval suite would leak it into training data"
            )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _allocate_within_run(
    by_ante: dict[int, list[dict[str, Any]]], per_run_cap: int
) -> list[dict[str, Any]]:
    """Fill this run's <=``per_run_cap`` slots round-robin over its antes,
    deepest ante first.

    Deepest-first is the whole point: a run that reached ante 5 has plenty of
    ante-1 turns and very few ante-5 ones, and the ante-5 ones are the scarce,
    strategically distinct states. Round-robin (rather than "take all of the
    deepest") keeps the run's early antes represented too -- the harvest's
    stated job is realism at the antes s0 actually reaches, and for most runs
    that IS ante 1-2.
    """
    queues = {ante: list(recs) for ante, recs in by_ante.items()}
    antes = sorted(queues, reverse=True)  # deepest first
    out: list[dict[str, Any]] = []
    while len(out) < per_run_cap and any(queues[a] for a in antes):
        for ante in antes:
            if not queues[ante]:
                continue
            out.append(queues[ante].pop(0))
            if len(out) >= per_run_cap:
                break
    return out


def build_candidate_pool(
    rows: list[dict[str, Any]],
    *,
    per_run_cap: int,
    per_ante_cap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Apply the within-run correlation caps, yielding this source's pool.

    Which records survive a (run, ante) bucket is a seeded random draw rather
    than "the first N turns": early turns of a blind are systematically
    different states (full hands/discards budget) from late ones, so taking a
    prefix would bias the whole stage toward turn 0.
    """
    runs: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        runs[str(row["run_seed"])][int(row["ante"])].append(row)

    pool: list[dict[str, Any]] = []
    for run_seed in sorted(runs):
        capped: dict[int, list[dict[str, Any]]] = {}
        for ante in sorted(runs[run_seed]):
            recs = sorted(runs[run_seed][ante], key=lambda r: int(r["turn_idx"]))
            k = min(len(recs), per_ante_cap)
            chosen = rng.sample(recs, k)
            capped[ante] = sorted(chosen, key=lambda r: int(r["turn_idx"]))
        pool.extend(_allocate_within_run(capped, per_run_cap))
    return pool


def thin_to_target(
    pool: list[dict[str, Any]], target: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Thin ``pool`` down to ``target``, ante-stratified by largest remainder.

    Proportional-per-ante rather than a uniform draw over the pool: both have
    the same expected ante mix, but the stratified version reproduces it
    exactly, so a scarce deep-ante bucket can't be randomly wiped out. A pool
    already at or below target passes through untouched.
    """
    if len(pool) <= target:
        return list(pool)

    by_ante: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_ante[int(row["ante"])].append(row)

    total = len(pool)
    quota: dict[int, int] = {}
    remainder: dict[int, float] = {}
    for ante in sorted(by_ante):
        exact = target * len(by_ante[ante]) / total
        quota[ante] = int(exact)
        remainder[ante] = exact - quota[ante]

    # Largest-remainder: hand out the seats integer division dropped.
    shortfall = target - sum(quota.values())
    for ante in sorted(remainder, key=lambda a: (-remainder[a], a))[:shortfall]:
        quota[ante] += 1

    out: list[dict[str, Any]] = []
    for ante in sorted(by_ante):
        recs = sorted(by_ante[ante], key=lambda r: str(r["record_id"]))
        out.extend(rng.sample(recs, min(quota[ante], len(recs))))
    return out


def select_records(
    rows: list[dict[str, Any]],
    *,
    n_total: int = DEFAULT_N_TOTAL,
    det_frac: float = DEFAULT_DET_FRAC,
    per_run_cap: int = DEFAULT_PER_RUN_CAP,
    per_ante_cap: int = DEFAULT_PER_ANTE_CAP,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Select the manifest's records from the metadata table.

    Per-source targets are honoured as CEILINGS, not quotas: if a source's
    capped pool falls short of its target, the shortfall is NOT backfilled from
    the other source. The 75:25 split is a deliberate deployed-distribution
    anchor, so topping det up with sampled records would silently make the
    stage MORE sampled-heavy than the decision that set the ratio.
    """
    if not 0.0 <= det_frac <= 1.0:
        raise SelectionError(f"--det-frac must be in [0, 1], got {det_frac}")

    hand = hand_records(rows)
    if not hand:
        raise SelectionError("no hand records in metadata; nothing to select")

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hand:
        by_source[str(row.get("source", "?"))].append(row)

    n_det = round(n_total * det_frac)
    targets = {"det": n_det, "sampled": n_total - n_det}

    selected: list[dict[str, Any]] = []
    for source in SOURCES:
        rows_for_source = by_source.get(source, [])
        if not rows_for_source:
            continue
        # One RNG stream per source, seeded from the run seed, so that changing
        # one source's supply can't shift the other's draws.
        rng = random.Random(f"{seed}:{source}")
        pool = build_candidate_pool(
            rows_for_source,
            per_run_cap=per_run_cap,
            per_ante_cap=per_ante_cap,
            rng=rng,
        )
        selected.extend(thin_to_target(pool, targets[source], rng))

    # C2 partitions workers over manifest order; shuffle so slow deep-ante
    # states spread across workers instead of clustering in one range.
    random.Random(f"{seed}:order").shuffle(selected)
    return selected


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def manifest_stats(selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Readout of what the selection actually produced (targets are ceilings,
    so the realized numbers are the ones that matter)."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_source[str(row["source"])].append(row)

    per_source: dict[str, Any] = {}
    for source in sorted(by_source):
        rows = by_source[source]
        per_run = Counter(str(r["run_seed"]) for r in rows)
        per_source[source] = {
            "n_records": len(rows),
            "n_runs": len(per_run),
            "max_records_per_run": max(per_run.values()) if per_run else 0,
            "ante_counts": dict(sorted(Counter(int(r["ante"]) for r in rows).items())),
        }

    total = len(selected)
    return {
        "n_records": total,
        "realized_det_frac": (
            round(len(by_source.get("det", [])) / total, 4) if total else 0.0
        ),
        "ante_counts": dict(sorted(Counter(int(r["ante"]) for r in selected).items())),
        "blind_type_counts": dict(sorted(Counter(str(r["blind_type"]) for r in selected).items())),
        "by_source": per_source,
    }


def build_manifest(
    selected: list[dict[str, Any]],
    *,
    params: dict[str, Any],
    corpus: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the manifest document: header (params + provenance + stats)
    followed by the ordered record list C2 consumes."""
    records = [
        {
            "record_id": str(r["record_id"]),
            # Carried EXPLICITLY rather than parsed back out of record_id: it
            # names the blob shard (blobs/{run_seed}.pkl), and both seed
            # prefixes (HARVEST_ / HARVEST_S_) contain underscores.
            "run_seed": str(r["run_seed"]),
            "source": str(r["source"]),
            "ante": int(r["ante"]),
            # Per-record so C2's engine-skew check (pitfall #7) is honest even
            # if a corpus ever mixes capture sessions.
            "git_sha": str(r.get("git_sha", "unknown")),
        }
        for r in selected
    ]
    return {
        "manifest_version": MANIFEST_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": params,
        "corpus": corpus,
        "stats": manifest_stats(selected),
        "records": records,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write the manifest as valid JSON with ONE COMPACT LINE PER RECORD.

    The header stays indented and readable; records do not, because ~8k
    records at indent=2 is a multi-MB file whose diffs are unreadable. One
    line per record keeps a re-run's change visible as a line diff.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in manifest.items() if k != "records"}
    body = ",\n".join(
        "    " + json.dumps(r, sort_keys=True) for r in manifest["records"]
    )
    text = json.dumps(head, indent=2, sort_keys=False)[:-2]  # drop trailing "\n}"
    text += ',\n  "records": [\n' + body + "\n  ]\n}\n"
    path.write_text(text, encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest back (used by C2 and by the round-trip tests)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/harvest_s0/metadata.jsonl"),
        help="harvest metadata table (JSONL)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifests/h1_harvested.json"),
        help="manifest path. NOTE: data/ is gitignored, so the default lives "
        "outside it -- the manifest is a checked-in versioned artifact.",
    )
    parser.add_argument("--n-total", type=int, default=DEFAULT_N_TOTAL)
    parser.add_argument("--det-frac", type=float, default=DEFAULT_DET_FRAC)
    parser.add_argument("--per-run-cap", type=int, default=DEFAULT_PER_RUN_CAP)
    parser.add_argument("--per-ante-cap", type=int, default=DEFAULT_PER_ANTE_CAP)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_metadata(args.metadata)
    selected = select_records(
        rows,
        n_total=args.n_total,
        det_frac=args.det_frac,
        per_run_cap=args.per_run_cap,
        per_ante_cap=args.per_ante_cap,
        seed=args.seed,
    )

    all_hand = hand_records(rows)
    corpus = {
        # as_posix: this artifact is checked in on Windows and consumed by the
        # regen on the 9600X, so the provenance path must not be OS-flavoured.
        "metadata_path": args.metadata.as_posix(),
        "n_rows_total": len(rows),
        "n_hand_records": len(all_hand),
        "git_shas": dict(
            sorted(Counter(str(r.get("git_sha", "unknown")) for r in all_hand).items())
        ),
    }
    params = {
        "seed": args.seed,
        "n_total": args.n_total,
        "det_frac": args.det_frac,
        "per_run_cap": args.per_run_cap,
        "per_ante_cap": args.per_ante_cap,
    }
    manifest = build_manifest(selected, params=params, corpus=corpus)
    write_manifest(args.output, manifest)

    print(f"[c1] wrote {len(selected)} records -> {args.output}")
    print(json.dumps(manifest["stats"], indent=2))


if __name__ == "__main__":
    main()
