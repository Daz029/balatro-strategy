"""K4 surgery: relabel ONLY the K3-corrupt rows of the stage2 brute corpus.

WHY SURGERY AND NOT A WHOLESALE REGEN. The stage2 corpus was generated
2026-07-16 with `PRESCREEN_HAND_LIMIT = 8`, so every n<=8 label is BRUTE
force -- exact over all 218 subsets. K3 deleted the limit, so a fresh label
is now "exact among prescreened candidates" (0.980 capture). Regenerating a
clean row would therefore DOWNGRADE it. The clean rows are strictly better
than anything we can produce now, and the corpus is complete (4000/4000, zero
failures), so the completeness is worth preserving.

WHAT IS CORRUPT, and why the set is exactly this (measured, not assumed):
  1. OWNS Four Fingers / Shortcut / Smeared (29.0%). `score_hand` passed
     `jokers=None` to `evaluate_hand` (03e288d), so those three were INERT
     in every label; and Bug A (a60dbbf) had Four Fingers REPLACE the 5-card
     straight windows. Bug A also reaches the DISCARD side
     (`rank_templates_cheaply` -> `build_templates`) at every hand size, but
     only under four_fingers -- still inside this set.
  2. HAND WIDTH > 8 (10.1%). Only these took the prescreen path, so only
     these carry the emission-order lottery (caf3394), the seating gap
     (a02e47b), and the pre-K1 kicker padding (b2ec3ff landed after the run).
  UNION = 1445 (36.1%). The other 2555 rows are brute-exact and CLEAN.

WHY BUG C DOES NOT WIDEN THIS (checked, and it is why the earlier
"wholesale regen" call was wrong): Bug C (7ce81e7) broke smeared, WILD and
stone flushes. Wild/stone are card ENHANCEMENTS -- not joker-gated, so no
query could find them -- but stage2 deals a plain `b_red` deck (60 sampled
states: every card `c_base`) and its 21-joker pool contains no Smeared. So
Bug C cannot fire on this stage at all.

ENGINE DRIFT since generation (the C2 capture-skew rule, applied to LABELS
rather than blobs -- a label is not re-scored by current code, so every
label-semantics change since the run dirties it):
  5434b31 B7 depth gate   12:38 +08  BEFORE the 14:23 +08 run start -> in
  b32c6d0 dollar marginals 13:53 +08  BEFORE -> in
  5b9ab27 O(n) get_x_same  22:12 +08  after, but an equivalence-PINNED perf
                                      rewrite -> no label change
  b2ec3ff K1 / 03e288d / a60dbbf / 7ce81e7 / a02e47b / caf3394 / cb9eeb0
                                      after -> all either prescreen-only
                                      (tail set) or FF/Shortcut-gated (set 1)
so the clean 2555 are genuinely clean.

OUTPUT: a NEW directory (never in place -- the pre_discard_cap backup
pattern). Clean rows are copied through verbatim; corrupt seeds are
relabeled with current code and written as additional shards matching the
loader's `worker_*_shard_*.npz` glob.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_hand_demos import (  # noqa: E402
    Example,
    generate_one_example,
    write_shard,
)

# The ONE implementation that rebuilds the exact HandPlayConfig a run used.
# Duplicating it is the rot pattern, and its own docstring is the reason this
# matters here: "any field mismatch silently regenerates a DIFFERENT state for
# the same seed" -- a relabelled row must be the SAME state the seed produced
# originally, or it is not a relabel, it is a different example wearing the
# same seed. (stage2 passed --dollar-marginals, which the preset lacks.)
from validate_prescreen_n8 import config_from_manifest  # noqa: E402

from jackdaw.env.observation import center_key_id  # noqa: E402

INERT_JOKER_IDS = frozenset(
    center_key_id(k) for k in ("j_four_fingers", "j_shortcut", "j_smeared")
)


def row_is_corrupt(joker_ids: np.ndarray, joker_mask: np.ndarray, hand_mask: np.ndarray) -> bool:
    owned = {int(x) for x, m in zip(joker_ids, joker_mask) if m}
    if owned & INERT_JOKER_IDS:
        return True
    return int(hand_mask.sum()) > 8


def _relabel_one(args: tuple[str, object]) -> tuple[str, Example | None, str | None]:
    seed, config = args
    try:
        return seed, generate_one_example(seed, config), None
    except Exception as exc:  # noqa: BLE001 -- logged and counted, never swallowed
        return seed, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--num-workers", type=int, required=True,
        help="ALWAYS explicit (the 2026-07-05 partitioning lesson)",
    )
    ap.add_argument(
        "--shard-size", type=int, default=25,
        help="small shards cap crash-loss (doc section 7)",
    )
    ap.add_argument("--dry-run", action="store_true", help="report the split, write nothing")
    args = ap.parse_args()

    manifest = json.loads((args.shard_dir / "manifest.json").read_text(encoding="utf-8"))
    config = config_from_manifest(manifest)

    shards = sorted(args.shard_dir.glob("worker_*_shard_*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {args.shard_dir}")

    clean_by_shard: dict[Path, np.ndarray] = {}
    corrupt_seeds: list[str] = []
    n_tot = 0
    for path in shards:
        d = np.load(path, allow_pickle=True)
        ji, jm, hm, seeds = d["joker_ids"], d["joker_mask"], d["hand_mask"], d["seed"]
        keep = np.ones(len(seeds), dtype=bool)
        for r in range(len(seeds)):
            n_tot += 1
            if row_is_corrupt(ji[r], jm[r], hm[r]):
                keep[r] = False
                corrupt_seeds.append(str(seeds[r]))
        clean_by_shard[path] = keep

    n_clean = sum(int(k.sum()) for k in clean_by_shard.values())
    print(f"corpus {args.shard_dir}: {n_tot} rows")
    print(f"  CLEAN  (brute-exact, copied verbatim): {n_clean:5d}  ({100*n_clean/n_tot:.1f}%)")
    n_bad = len(corrupt_seeds)
    print(f"  RELABEL (K3-corrupt)                 : {n_bad:5d}  ({100 * n_bad / n_tot:.1f}%)")
    assert n_clean + len(corrupt_seeds) == n_tot
    if len(set(corrupt_seeds)) != len(corrupt_seeds):
        raise SystemExit("duplicate seeds in the corrupt set -- corpus is not unique-seeded")
    if args.dry_run:
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # manifest travels, with the surgery recorded in it
    manifest["k3_relabel"] = {
        "source_dir": str(args.shard_dir),
        "n_clean_brute_exact": n_clean,
        "n_relabelled_prescreened": len(corrupt_seeds),
        "note": (
            "MIXED PROVENANCE BY DESIGN: clean rows are brute-exact (generated "
            "pre-K3 when n<=8 brute-forced); relabelled rows are 'exact among "
            "prescreened candidates' (post-K3). Relabelled = owned "
            "FourFingers/Shortcut/Smeared (inert pre-03e288d) or hand width > 8 "
            "(the only rows that took the prescreen path)."
        ),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # 1. copy clean rows through, shard by shard, preserving names
    for path, keep in clean_by_shard.items():
        d = np.load(path, allow_pickle=True)
        if keep.all():
            shutil.copy2(path, args.out_dir / path.name)
            continue
        if not keep.any():
            continue  # every row corrupt: no shard to write
        np.savez_compressed(
            args.out_dir / path.name,
            **{k: (d[k] if k == "schema_version" else d[k][keep]) for k in d.files},
        )
    print(f"  wrote clean shards -> {args.out_dir}")

    # 2. relabel the corrupt seeds with CURRENT code
    t0 = time.time()
    jobs = [(s, config) for s in corrupt_seeds]
    failures: list[dict] = []
    done: list[Example] = []
    shard_idx = 0
    with mp.Pool(args.num_workers) as pool:
        for i, (seed, ex, err) in enumerate(pool.imap_unordered(_relabel_one, jobs), 1):
            if err is not None:
                failures.append({"seed": seed, "error": err})
            else:
                done.append(ex)
            if len(done) >= args.shard_size:
                write_shard(args.out_dir / f"worker_900_shard_{shard_idx:05d}.npz", done)
                shard_idx += 1
                done = []
            if i % 50 == 0:
                print(f"  relabel {i}/{len(jobs)}  ({time.time()-t0:.0f}s, {len(failures)} failed)")
    if done:
        write_shard(args.out_dir / f"worker_900_shard_{shard_idx:05d}.npz", done)

    if failures:
        (args.out_dir / "worker_900_failures.jsonl").write_text(
            "\n".join(json.dumps(f) for f in failures), encoding="utf-8"
        )
    frac = len(failures) / max(len(jobs), 1)
    print(
        f"  relabelled {len(jobs) - len(failures)}/{len(jobs)} in "
        f"{time.time() - t0:.0f}s ({100 * frac:.1f}% failed)"
    )
    if frac > 0.03:
        raise SystemExit(f"FAILURE RATE {100*frac:.1f}% > 3% -- stop, do not ship a thinned stage")

    # 3. verify: every original seed present exactly once
    seen: list[str] = []
    for path in sorted(args.out_dir.glob("worker_*_shard_*.npz")):
        seen.extend(str(s) for s in np.load(path, allow_pickle=True)["seed"])
    dupes = len(seen) - len(set(seen))
    print(f"VERIFY: {len(seen)} rows, {len(set(seen))} unique, {dupes} duplicates, "
          f"{n_tot - len(seen)} missing vs source")
    if dupes:
        raise SystemExit("duplicate seeds in output")


if __name__ == "__main__":
    main()
