"""Phase 1 of the harvest pass (pre-regen build plan A1/A2): drive full runs of
the shop agent (s0 by default, or an s1 checkpoint with ``--s1-schema``)
paired with the hand agent (h0.5 by default, or an h1 partner with
``--partner-money-ordering``) and capture EVERY hand-turn decision state, plus
shop-state snapshots.

This is deliberately schema-INDEPENDENT: a captured record is a pickle of the
live engine ``gs`` (RNG included — the exact ``ShopRunAdapter.snapshot_state``
mechanism), not an encoded observation. So the corpus can be banked now, in
parallel with the schema-bump / solver work in phase B, and labeled later
(phase 2, ``generate_hand_demos.py`` snapshot front-end) once the schema lands.

Capture policy (grilled — CLAUDE.md "Pre-regeneration build plan"): capture
EVERYTHING, thin later. No subsampling and no per-run caps at capture time; the
~8k ante-stratified selection is a separate seeded manifest step (C1) over the
metadata table. Two passes are banked up front:

* ~1200 deterministic runs (``HARVEST_{i:08d}``): shop argmax — the DEPLOYED
  induced distribution h1 actually targets.
* ~500 sampled runs (``HARVEST_S_{i:08d}``, ``--sample-shop``): shop sampling for
  coverage breadth. The hand partner stays deterministic in BOTH passes —
  it is the partner being targeted, not the thing we vary.

Every record carries a ``source`` tag (``det``/``sampled``) and a ``git_sha``
stamp (engine-version-skew check, loud at phase 2 — pitfall #7).

Interception (A1): the hand phase is auto-resolved inside ``ShopRunAdapter``
at ``self._hand_policy(self._gs)``. Wrapping that callable in
:class:`HarvestingHandPolicy` pickles ``self._gs`` BEFORE delegating, so the
captured state is exactly the decision state the solver will later label.
Shop-state snapshots (for the offline ``V_curve`` money-sweep of the s0 critic)
are captured in the same pass at each SHOP decision point — one extra sink, no
second rollout.

Blobs go in per-run shard files (``blobs/{run_seed}.pkl``, keyed by
``record_id``); metadata goes in ONE flat JSONL table so later stratification
and the free reductions never unpickle a blob (pitfall: metadata queries must
be blob-free). ``record_id = {run_seed}_{turn_idx}`` for hand records (keys
``mc_seed`` and phase-2 partitioning), ``{run_seed}_s{k}`` for shop records.

Free reductions (A2), emitted at end of the run over the DETERMINISTIC corpus
only (pitfall #14 — sampled runs would smear a worse policy's money behaviour
into the money prior): per-ante ``$`` marginals (feed stage 1-4 money regen)
and the hand-size histogram (informs Candidate B's max decode length). Plus a
coverage READOUT (both sources) that tunes the C1 manifest's det:sampled ratio —
it has NO pass/fail threshold (pitfall #18).

Usage::

    uv run python scripts/harvest_s0_rollouts.py \
        --shop-policy runs/shop_ppo/s0_a4_v4/best_model/best_model.zip \
        --hand-policy runs/hand_ppo/hand_ppo_2000000_steps.zip \
        --output-dir data/harvest_s0 \
        --n-det 1200 --n-sampled 500

    # recompute reductions/coverage from an already-banked corpus:
    uv run python scripts/harvest_s0_rollouts.py \
        --output-dir data/harvest_s0 --reductions-only
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from jackdaw.engine.actions import Action, GamePhase  # noqa: E402
from jackdaw.env.shop_gym import ShopGymEnv  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunConfig  # noqa: E402

# Reserved seed prefixes. DISJOINT from ``EVAL_`` — harvesting an eval seed
# would leak the held-out suite into training data (pitfall #6).
DET_PREFIX = "HARVEST"
SAMPLED_PREFIX = "HARVEST_S"

# Whole-run horizon: s0 was trained to win_ante=4, but we drive win_ante=8 so
# runs reach natural death/win rather than halting at an artificial horizon
# (the induced mid-run distribution is what we want, all the way out).
HARVEST_WIN_ANTE = 8

_PICKLE = pickle.HIGHEST_PROTOCOL


# ---------------------------------------------------------------------------
# Metadata extraction (blob-free queries depend on these being complete)
# ---------------------------------------------------------------------------


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
    except Exception:  # noqa: BLE001 — best-effort stamp, never fatal
        return "unknown"


def owned_joker_keys(gs: dict[str, Any]) -> list[str]:
    """Sorted center keys of owned jokers — the build-family fingerprint the
    coverage readout groups on WITHOUT unpickling blobs."""
    return sorted(getattr(j, "center_key", "") for j in gs.get("jokers", []))


def _blind_type(gs: dict[str, Any]) -> str:
    """Which of Small/Big/Boss is the active blind (``"Current"``/``"Select"``)."""
    bs = gs.get("round_resets", {}).get("blind_states", {})
    for name in ("Small", "Big", "Boss"):
        if bs.get(name) in ("Current", "Select"):
            return name
    return ""


def extract_hand_meta(gs: dict[str, Any]) -> dict[str, Any]:
    """Stratification/stat fields for a SELECTING_HAND decision state."""
    rr = gs.get("round_resets", {})
    cr = gs.get("current_round", {})
    btype = _blind_type(gs)
    blind = gs.get("blind")
    boss_key = getattr(blind, "key", "") if (blind and btype == "Boss") else ""
    return {
        "ante": int(rr.get("ante", 1)),
        "blind_type": btype,
        "boss_key": boss_key,
        "hand_size": len(gs.get("hand", [])),
        "dollars": int(gs.get("dollars", 0)),
        "hands_left": int(cr.get("hands_left", 0)),
        "discards_left": int(cr.get("discards_left", 0)),
        "owned_jokers": owned_joker_keys(gs),
    }


def extract_shop_meta(gs: dict[str, Any]) -> dict[str, Any]:
    """Stratification fields for a SHOP decision state (V_curve money sweeps)."""
    rr = gs.get("round_resets", {})
    return {
        "ante": int(rr.get("ante", 1)),
        "dollars": int(gs.get("dollars", 0)),
        "owned_jokers": owned_joker_keys(gs),
    }


# ---------------------------------------------------------------------------
# Sink — per-run blob shards + one flat metadata table
# ---------------------------------------------------------------------------


class HarvestSink:
    """Buffers one run's records, then flushes blobs + metadata atomically.

    A single sink instance spans a whole invocation (both passes append to the
    same ``metadata.jsonl``). Blobs for a run live in one shard file keyed by
    ``record_id``; metadata is one JSONL line per record. Buffer-then-flush per
    run keeps blob shard writes single-writer and append-free.
    """

    def __init__(self, output_dir: str | Path, git_sha_stamp: str, schema_note: str = "") -> None:
        self._dir = Path(output_dir)
        self._blob_dir = self._dir / "blobs"
        self._blob_dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "metadata.jsonl"
        # Append so both passes accumulate; main() clears the dir for a fresh
        # harvest. Line-buffered so a crash mid-run keeps prior runs' rows.
        self._meta_fh = self._meta_path.open("a", encoding="utf-8")
        self._git_sha = git_sha_stamp
        self._schema_note = schema_note

        self._run_seed = ""
        self._source = ""
        self._hand_idx = 0
        self._shop_idx = 0
        self._blobs: dict[str, bytes] = {}
        self._rows: list[dict[str, Any]] = []
        self.n_records = 0

    # -- run lifecycle --------------------------------------------------------

    def begin_run(self, run_seed: str, source: str) -> None:
        self._run_seed = run_seed
        self._source = source
        self._hand_idx = 0
        self._shop_idx = 0
        self._blobs = {}
        self._rows = []

    def end_run(self) -> None:
        """Flush this run's blobs + metadata. No-op if nothing was captured
        (e.g. a seed that died before any hand turn)."""
        if not self._rows:
            return
        blob_path = self._blob_dir / f"{self._run_seed}.pkl"
        with blob_path.open("wb") as fh:
            pickle.dump(self._blobs, fh, protocol=_PICKLE)
        for row in self._rows:
            self._meta_fh.write(json.dumps(row) + "\n")
        self._meta_fh.flush()
        self.n_records += len(self._rows)
        self._blobs = {}
        self._rows = []

    def close(self) -> None:
        self._meta_fh.close()

    def __enter__(self) -> HarvestSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- capture --------------------------------------------------------------

    def capture_hand(self, gs: dict[str, Any]) -> str:
        record_id = f"{self._run_seed}_{self._hand_idx}"
        self._capture(record_id, "hand", self._hand_idx, gs, extract_hand_meta(gs))
        self._hand_idx += 1
        return record_id

    def capture_shop(self, gs: dict[str, Any]) -> str:
        record_id = f"{self._run_seed}_s{self._shop_idx}"
        self._capture(record_id, "shop", self._shop_idx, gs, extract_shop_meta(gs))
        self._shop_idx += 1
        return record_id

    def _capture(
        self,
        record_id: str,
        kind: str,
        turn_idx: int,
        gs: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        self._blobs[record_id] = pickle.dumps(gs, protocol=_PICKLE)
        row = {
            "record_id": record_id,
            "run_seed": self._run_seed,
            "turn_idx": turn_idx,
            "kind": kind,
            "source": self._source,
            "git_sha": self._git_sha,
            "schema_note": self._schema_note,
            **meta,
        }
        self._rows.append(row)


# ---------------------------------------------------------------------------
# Harvesting hand policy — transparent capture wrapper
# ---------------------------------------------------------------------------


class HarvestingHandPolicy:
    """Wraps the inner hand policy; pickles ``gs`` then delegates unchanged.

    Transparency is the contract: the run must play out byte-identically with
    or without the wrapper. So it captures BEFORE calling ``inner`` and returns
    ``inner``'s action verbatim, never touching ``gs``.
    """

    def __init__(
        self,
        inner: Callable[[dict[str, Any]], Action],
        sink: HarvestSink,
    ) -> None:
        self._inner = inner
        self._sink = sink

    def __call__(self, gs: dict[str, Any]) -> Action:
        self._sink.capture_hand(gs)
        return self._inner(gs)


def _maybe_capture_shop(env: ShopGymEnv, sink: HarvestSink) -> None:
    """Capture a clean SHOP decision state (pending-free) for V_curve."""
    gs = env.raw_state
    if env.pending is None and gs.get("phase") == GamePhase.SHOP:
        sink.capture_shop(gs)


# ---------------------------------------------------------------------------
# Rollout driver
# ---------------------------------------------------------------------------


class NextRoundPolicy:
    """Do-nothing shop policy (leave every shop, skip every pack). Used as the
    test double for the rollout driver so the capture path needs no checkpoint."""

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        from jackdaw.agents.shop_action_space import ShopActionFamily, shop_action

        for family in (ShopActionFamily.NextRound, ShopActionFamily.SkipPack):
            action = shop_action(family)
            if mask[action]:
                return action
        return int(np.flatnonzero(mask)[0])


class ShopModelPolicy:
    """MaskablePPO ``.zip`` wrapper with a deterministic/sampling switch."""

    def __init__(self, model_path: str | Path, device: str, deterministic: bool) -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(model_path), device=device)
        self._deterministic = deterministic

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        action, _ = self._model.predict(
            obs, action_masks=mask, deterministic=self._deterministic
        )
        return int(action)


def harvest_runs(
    shop_policy: Any,
    hand_inner: Callable[[dict[str, Any]], Action],
    sink: HarvestSink,
    *,
    n_runs: int,
    seed_prefix: str,
    source: str,
    win_ante: int = HARVEST_WIN_ANTE,
    max_steps: int = 512,
    s1_schema: bool = False,
) -> int:
    """Drive ``n_runs`` full runs, capturing every hand-turn + shop state.

    Returns the number of runs that produced at least one record. The hand
    policy is wrapped once; ``sink.begin_run`` re-scopes ``record_id``s and
    counters per seed.
    """
    harvesting = HarvestingHandPolicy(hand_inner, sink)
    env = ShopGymEnv(
        config=ShopRunConfig(win_ante=win_ante, s1_schema=s1_schema),
        hand_policy=harvesting,
        max_steps=max_steps,
    )
    n_with_records = 0

    for i in range(n_runs):
        seed = f"{seed_prefix}_{i:08d}"
        sink.begin_run(seed, source)
        before = sink.n_records
        try:
            obs, info = env.reset(options={"episode_seed": seed})
        except RuntimeError:
            # Hand policy lost the auto-resolved first blind. Any ante-1 hand
            # turns captured during that resolve are still real, labelable
            # states — flush them; there's simply no shop state for this run.
            sink.end_run()
            if sink.n_records > before:
                n_with_records += 1
            continue

        _maybe_capture_shop(env, sink)
        for _ in range(max_steps):
            action = shop_policy.act(obs, info["action_mask"])
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            _maybe_capture_shop(env, sink)

        sink.end_run()
        if sink.n_records > before:
            n_with_records += 1

    return n_with_records


# ---------------------------------------------------------------------------
# Reductions + coverage readout (free group-bys over the metadata table)
# ---------------------------------------------------------------------------


def load_metadata(output_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(output_dir) / "metadata.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_reductions(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per-ante ``$`` marginals + hand-size histogram, DETERMINISTIC hand
    records only (pitfall #14: the deployed policy's distribution, not a
    smear of the deliberately-worse sampled pass)."""
    dollars_by_ante: dict[int, Counter] = defaultdict(Counter)
    hand_size_hist: Counter = Counter()
    for r in rows:
        if r.get("kind") != "hand" or r.get("source") != "det":
            continue
        dollars_by_ante[int(r["ante"])][int(r["dollars"])] += 1
        hand_size_hist[int(r["hand_size"])] += 1
    return {
        "dollar_marginals_by_ante": {
            str(ante): {str(d): c for d, c in sorted(counter.items())}
            for ante, counter in sorted(dollars_by_ante.items())
        },
        "hand_size_histogram": {str(h): c for h, c in sorted(hand_size_hist.items())},
        "note": "deterministic hand records only; dollars sampled at hand-turn entry",
    }


def _coverage_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hand_rows = [r for r in rows if r.get("kind") == "hand"]
    ante_counts: Counter = Counter(int(r["ante"]) for r in hand_rows)
    distinct_jokers: set[str] = set()
    keysets: set[tuple[str, ...]] = set()
    hand_sizes: list[int] = []
    for r in hand_rows:
        owned = r.get("owned_jokers", [])
        distinct_jokers.update(k for k in owned if k)
        keysets.add(tuple(owned))
        hand_sizes.append(int(r["hand_size"]))
    return {
        "n_hand_records": len(hand_rows),
        "n_shop_records": sum(1 for r in rows if r.get("kind") == "shop"),
        "ante_counts": {str(a): c for a, c in sorted(ante_counts.items())},
        "n_at_ante_ge2": sum(c for a, c in ante_counts.items() if a >= 2),
        "n_at_ante_ge3": sum(c for a, c in ante_counts.items() if a >= 3),
        "distinct_jokers_owned": len(distinct_jokers),
        "distinct_owned_keysets": len(keysets),
        "hand_size_gt8": sum(1 for h in hand_sizes if h > 8),
        "hand_size_gt12": sum(1 for h in hand_sizes if h > 12),
        "max_hand_size": max(hand_sizes) if hand_sizes else 0,
    }


def compute_coverage_readout(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage split by source. Tunes the C1 manifest det:sampled ratio; it
    has NO pass/fail threshold (pitfall #18)."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r.get("source", "?")].append(r)
    return {source: _coverage_for(rs) for source, rs in sorted(by_source.items())}


def emit_reductions(output_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = load_metadata(output_dir)
    reductions = compute_reductions(rows)
    coverage = compute_coverage_readout(rows)
    out = Path(output_dir)
    (out / "reductions.json").write_text(json.dumps(reductions, indent=2), encoding="utf-8")
    (out / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    return reductions, coverage


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--shop-policy",
        type=Path,
        default=None,
        help="MaskablePPO .zip checkpoint (s0, or s1 with --s1-schema). "
        "Required unless --reductions-only.",
    )
    parser.add_argument(
        "--hand-policy",
        type=Path,
        default=None,
        help="hand partner checkpoint (.zip/.pt; h0.5, or h1 with "
        "--partner-money-ordering). Required unless --reductions-only.",
    )
    parser.add_argument("--n-det", type=int, default=1200, help="deterministic runs (HARVEST_)")
    parser.add_argument("--n-sampled", type=int, default=500, help="sampled runs (HARVEST_S_)")
    parser.add_argument("--win-ante", type=int, default=HARVEST_WIN_ANTE)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--schema-note", default="", help="free-text stamp on every record")
    parser.add_argument(
        "--s1-schema",
        action="store_true",
        help="configure the environment for an s1-schema shop checkpoint",
    )
    parser.add_argument(
        "--partner-money-ordering",
        action="store_true",
        help="deploy the hand partner with money-aware ordering (h1)",
    )
    parser.add_argument(
        "--seed-prefix",
        type=str,
        default=DET_PREFIX,
        help="prefix for deterministic seeds; sampled seeds append _S",
    )
    parser.add_argument(
        "--reductions-only",
        action="store_true",
        help="recompute reductions.json/coverage.json from an existing corpus; no rollouts",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.partner_money_ordering and args.hand_policy is None:
        parser.error("--partner-money-ordering requires --hand-policy")

    if args.reductions_only:
        reductions, coverage = emit_reductions(args.output_dir)
        print(json.dumps({"reductions": reductions, "coverage": coverage}, indent=2))
        return

    if args.shop_policy is None or args.hand_policy is None:
        parser.error("--shop-policy and --hand-policy are required unless --reductions-only")

    from jackdaw.agents.hand_checkpoint_policy import HandCheckpointPolicy

    if args.partner_money_ordering:
        hand_inner = HandCheckpointPolicy(
            str(args.hand_policy), money_aware_ordering=True
        )
    else:
        hand_inner = HandCheckpointPolicy(str(args.hand_policy))
    sha = git_sha()

    with HarvestSink(args.output_dir, sha, args.schema_note) as sink:
        if args.n_det > 0:
            det_policy = ShopModelPolicy(args.shop_policy, args.device, deterministic=True)
            n = harvest_runs(
                det_policy,
                hand_inner,
                sink,
                n_runs=args.n_det,
                seed_prefix=args.seed_prefix,
                source="det",
                win_ante=args.win_ante,
                max_steps=args.max_steps,
                s1_schema=args.s1_schema,
            )
            print(f"[harvest] det pass: {n}/{args.n_det} runs produced records")
        if args.n_sampled > 0:
            sampled_policy = ShopModelPolicy(args.shop_policy, args.device, deterministic=False)
            n = harvest_runs(
                sampled_policy,
                hand_inner,
                sink,
                n_runs=args.n_sampled,
                seed_prefix=f"{args.seed_prefix}_S",
                source="sampled",
                win_ante=args.win_ante,
                max_steps=args.max_steps,
                s1_schema=args.s1_schema,
            )
            print(f"[harvest] sampled pass: {n}/{args.n_sampled} runs produced records")
        total = sink.n_records

    reductions, coverage = emit_reductions(args.output_dir)
    print(f"[harvest] banked {total} records to {args.output_dir}")
    print(json.dumps({"coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
