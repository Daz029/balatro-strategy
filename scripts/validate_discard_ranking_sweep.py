"""B7 top_k sweep -- does widening the discard shortlist kill the regressions?

Companion to ``validate_discard_ranking.py``. That harness runs the faithful-MC
paired best-in-box comparison at a SINGLE ``top_k`` (production uses 4) and found
a net-positive but regressive picture (helps and regressions both live at the
rank-k/rank-(k+1) truncation boundary, per the box-frontier analysis). This
script sweeps ``top_k`` to answer the one question that resolves the ship/hold
call:

  - If widening the box drops regressions toward 0 while the helps survive ->
    ship B7 at that discard ``top_k`` (clean win, small extra recursion cost).
  - If widening erases BOTH -> B7 is a wash at the boundary -> keep the simpler
    jokerless ranking (don't ship B7); nothing rides the regen.
  - The largest ``top_k`` is the UNION endpoint: once the box covers every
    template, both arms explore the identical set and B7 is a provable no-op
    (n_disagreements -> 0), which also sanity-checks the whole "all B7 effects
    live at the truncation boundary" framing.

WHY IT IS NEARLY FREE. ``rank_templates_cheaply`` is prefix-stable in ``top_k``
and B7 changes only the cheap value, so the union of every discard action that
can appear at ANY swept ``top_k`` equals the two arms' boxes at the LARGEST k.
We value each unique discard's faithful redraw distribution ONCE (the only
expensive step -- ``n_samples`` x ``best_immediate_play``) and then re-slice
which actions fall in ``old_box`` / ``new_box`` at each k with pure ``max`` over
cached totals, no new Monte Carlo. Cost ~= one single-run at the largest k.

Every MC primitive (``faithful_totals``, ``_value_at``, the ``noise_floor``, the
``hands_left=1``/``discards_left=1`` isolation, the adaptive goal lines) is
imported verbatim from ``validate_discard_ranking`` -- there is no new value
logic here to trust, only per-k box membership.

Usage::

    uv run python scripts/validate_discard_ranking_sweep.py \
        --n-states 200 --n-samples 80 --top-ks 4,6,8,12,64 \
        --output data/discard_ranking_sweep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_solver import (  # noqa: E402
    DeckComposition,
    best_immediate_play,
    rank_templates_cheaply,
)
from jackdaw.engine.play_ordering import best_joker_order  # noqa: E402
from validate_discard_ranking import (  # noqa: E402
    GOAL_QUANTILES,
    SEED_PREFIX,
    VALUE_TOLERANCE,
    _dedup_box,
    _deck_pool,
    _value_at,
    faithful_totals,
    noise_floor,
    sample_state,
)


def _boxes_at(state, deck, common, *, top_k: int) -> tuple[set, set]:
    """The (old, new) physical-discard key sets the solver explores at this
    ``top_k`` -- exactly what ``solve_hand_turn`` iterates (dedup mirrors the
    harmless duplicate-template collapse; the outer max is order-free)."""
    old_box = _dedup_box(
        rank_templates_cheaply(
            state.hand, deck, state.hand_levels, state.blind, state.rng,
            top_k=top_k, joker_aware=False, **common,
        ),
        state.hand,
    )
    new_box = _dedup_box(
        rank_templates_cheaply(
            state.hand, deck, state.hand_levels, state.blind, state.rng,
            top_k=top_k, joker_aware=True, **common,
        ),
        state.hand,
    )
    return {k for k, _ in old_box}, {k for k, _ in new_box}


def _score_state_sweep(state, *, top_ks: list[int], n_samples: int) -> dict[str, Any]:
    started = time.time()
    jokers = best_joker_order(state.jokers)
    deck = DeckComposition.from_deck(state.gs.get("deck", []))
    common = dict(
        four_fingers=state.four_fingers,
        shortcut=state.shortcut,
        jokers=jokers,
        game_state=state.gs,
        blind_chips=state.blind_chips,
    )

    # Per-k box membership (cheap: sort only, no MC). We only ever read cached
    # totals for a k where the boxes DISAGREE (agree-k -> identical label, no
    # comparison), so value only those k's box members. Boxes are nested in k
    # (prefix-stable ranking), so this caps the valued union at the LARGEST
    # disagreeing k -- which collapses toward k=4's size as widening kills
    # disagreements, instead of paying for every template at union width.
    per_k_keys: dict[int, tuple[set, set]] = {}
    union_keys: set = set()
    for k in top_ks:
        old_k, new_k = _boxes_at(state, deck, common, top_k=k)
        per_k_keys[k] = (old_k, new_k)
        if old_k != new_k:
            union_keys |= old_k | new_k

    row: dict[str, Any] = {
        "seed": state.seed,
        "n_jokers": len(jokers),
        "n_valued": len(union_keys),
        "per_k": {},
    }

    # Value each unique discard's faithful redraw distribution ONCE.
    pool = _deck_pool(deck)
    totals_by_key: dict[tuple[int, ...], list[float]] = {}
    for key in union_keys:
        totals = faithful_totals(
            hand=state.hand, discard_key=key, jokers=jokers, hand_levels=state.hand_levels,
            blind=state.blind, rng=state.rng, pool=pool, game_state=state.gs,
            blind_chips=state.blind_chips, n_samples=n_samples, mc_seed=f"{state.seed}:{key}",
        )
        if totals is not None:
            totals_by_key[key] = totals

    play_now_total = float(
        best_immediate_play(
            state.hand, jokers, state.hand_levels, state.blind, state.rng,
            state.gs, state.blind_chips,
        )[1].total
    )
    row["play_now_total"] = play_now_total

    pooled = [t for totals in totals_by_key.values() for t in totals]
    play_dominated = not pooled or max(pooled) <= play_now_total + VALUE_TOLERANCE
    goals: list[float] = []
    if not play_dominated:
        goals = sorted(
            {
                float(g)
                for g in np.quantile(pooled, GOAL_QUANTILES)
                if g > play_now_total + VALUE_TOLERANCE
            }
        )
        play_dominated = not goals

    def best_in(keys: set, goal: float) -> float:
        return max(
            (_value_at(totals_by_key[k], goal) for k in keys if k in totals_by_key),
            default=0.0,
        )

    for k in top_ks:
        old_k, new_k = per_k_keys[k]
        disagree = old_k != new_k
        entry: dict[str, Any] = {"disagree": disagree}
        if not disagree:
            entry["category"] = "agree"
        elif play_dominated:
            entry["category"] = "play_dominated"
        else:
            diffs = [best_in(new_k, g) - best_in(old_k, g) for g in goals]
            entry["category"] = "measured"
            entry["max_help"] = max(diffs)
            entry["min_paired"] = min(diffs)
        row["per_k"][str(k)] = entry

    row["seconds"] = round(time.time() - started, 2)
    return row


def _aggregate_k(rows: list[dict[str, Any]], k: int, threshold: float) -> dict[str, Any]:
    entries = [r["per_k"][str(k)] for r in rows]
    disagreements = [e for e in entries if e["disagree"]]
    measured = [e for e in disagreements if e["category"] == "measured"]
    helped = [e for e in measured if e["max_help"] > threshold]
    regressions = [
        (r["seed"], e)
        for r, e in zip(rows, entries)
        if e["disagree"] and e["category"] == "measured" and e["min_paired"] < -threshold
    ]
    max_helps = [e["max_help"] for e in measured]
    return {
        "top_k": k,
        "n_disagreements": len(disagreements),
        "n_play_dominated": sum(e["category"] == "play_dominated" for e in disagreements),
        "n_measured": len(measured),
        "n_helped": len(helped),
        "n_regressions": len(regressions),
        "regression_seeds": [s for s, _ in regressions],
        "frac_helped": (len(helped) / len(measured)) if measured else None,
        "mean_max_help": float(np.mean(max_helps)) if max_helps else None,
        "worst_paired_diff": float(min(e["min_paired"] for e in measured)) if measured else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="stage3_full")
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--top-ks", default="4,6,8,12,64", help="comma list; largest ~= union")
    parser.add_argument("--n-samples", type=int, default=80)
    parser.add_argument("--floor-states", type=int, default=20)
    parser.add_argument("--floor-seeds", type=int, default=4)
    parser.add_argument("--accept-factor", type=float, default=1.33)
    parser.add_argument("--output", type=Path, default=Path("data/discard_ranking_sweep.json"))
    args = parser.parse_args()
    top_ks = sorted({int(x) for x in args.top_ks.split(",") if x.strip()})
    if not top_ks or top_ks[0] <= 0 or args.n_states <= 0 or args.n_samples <= 0:
        parser.error("bad --top-ks / --n-states / --n-samples")

    from generate_hand_demos import stage_presets

    config = stage_presets()[args.stage].config
    rows: list[dict[str, Any]] = []
    attempt = 0
    max_attempts = args.n_states * 100
    t_start = time.time()
    while len(rows) < args.n_states and attempt < max_attempts:
        seed = f"{SEED_PREFIX}_{attempt:08d}"
        attempt += 1
        state = sample_state(seed, config)
        cr = state.gs["current_round"]
        if int(cr.get("discards_left", 0)) < 1:
            continue
        if state.hands_left <= 0 or state.chips_needed <= 0 or not state.jokers:
            continue
        state.hands_left = 1
        cr["hands_left"] = 1
        state.gs["hands_left"] = 1
        cr["discards_left"] = 1

        row = _score_state_sweep(state, top_ks=top_ks, n_samples=args.n_samples)
        rows.append(row)
        base = row["per_k"][str(top_ks[0])]
        tag = base["category"] if base["disagree"] else "agree"
        print(
            f"[{len(rows)}/{args.n_states}] seed={state.seed} k{top_ks[0]}={tag} "
            f"valued={row['n_valued']} t={row['seconds']:.2f}s",
            flush=True,
        )

    print("computing n<=8 MC-reseed noise floor...", flush=True)
    floor = noise_floor(
        config, n_states=args.floor_states, n_seeds=args.floor_seeds, n_samples=args.n_samples
    )
    floor_val = floor["mean_pairwise_abs_delta"]
    threshold = (args.accept_factor * floor_val) if floor_val is not None else VALUE_TOLERANCE
    by_k = [_aggregate_k(rows, k, threshold) for k in top_ks]

    summary = {
        "stage": args.stage,
        "n_states": len(rows),
        "top_ks": top_ks,
        "n_samples": args.n_samples,
        "noise_floor": floor_val,
        "help_threshold": threshold,
        "accept_factor": args.accept_factor,
        "by_k": by_k,
        "attempts": attempt,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    report = {"summary": summary, "floor_detail": floor, "states": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n  top_k | disagree | measured | helped | regress | mean_help | worst_paired")
    print("  ------+----------+----------+--------+---------+-----------+-------------")
    for s in by_k:
        mh = f"{s['mean_max_help']:.3f}" if s["mean_max_help"] is not None else "  -  "
        wp = f"{s['worst_paired_diff']:.3f}" if s["worst_paired_diff"] is not None else "  -  "
        print(
            f"  {s['top_k']:>5} | {s['n_disagreements']:>8} | {s['n_measured']:>8} | "
            f"{s['n_helped']:>6} | {s['n_regressions']:>7} | {mh:>9} | {wp:>11}"
        )
    print(f"\n  noise_floor={floor_val:.4f} help_threshold={threshold:.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
