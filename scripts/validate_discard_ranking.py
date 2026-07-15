"""Validate B7 (joker/held-aware discard-branch ranking) against a FAITHFUL
value model -- because the solver cannot judge B7 with its own model.

B7 changes only the cheap scorer inside ``rank_templates_cheaply``: it
reorders which template branches survive the ``top_k`` cut. ``solve_hand_turn``
takes an outer ``max`` over that box with no order-dependent early exit
(verified), so **box-set agreement implies a byte-identical label**. B7 can
only change a label on states where the old (jokerless) and new (joker/held-
aware) top-k SETS differ.

WHY THE SOLVER'S OWN VALUE MODEL IS THE WRONG ARBITER (the finding that
reshaped this harness). The solver values a discard branch with a two-point
representative hit/miss recursion, and the "hit" hand is refilled by
``_fill_hand_to_size`` with the HIGHEST-remaining-rank filler. That filler is
optimistic: a still_needed==0 branch (e.g. discard around trip Kings) refills
to a monster and clears EVERY goal line, so its p_clear saturates at 1.0. When
B7 flips the emitted discard from that trips-refill line to a joker-favored
flush line, both read p_clear=1.0 under the solver's model -- it records a
different, genuinely better ACTION but cannot see that it is better. Any
p_clear-regret metric built on the solver's own recursion therefore reads ~0
(and can even read a false REGRESSION, since the optimistic refill pins the
old box at 1.0). The 30-state shortlist-coverage run that reported "new loses
rank-2/3 coverage, regret 0/0" was this exact artifact.

So the ground truth here is a FAITHFUL Monte Carlo estimate: discard the
action's cards, draw REAL replacements from the deck, score the best play,
and take the fraction that clears. Real draws de-saturate the trips-refill
line (random draws rarely make a monster) while the flush completes with its
true probability and the joker mult carries it over the line.

Scheme:

1. **Disagreement filter (cheap, large N).** The box is chosen by
   ``cheap_value``, goal-line-INDEPENDENT, so this is two cheap ``score_hand``
   passes. Agreement states are exactly-zero and never valued.
2. **Paired best-in-box, faithful value.** Report
   ``best_in_new_box - best_in_old_box``; the global-best term cancels, so we
   value only ``old_box | new_box``. Absolute regret vs ALL template branches
   conflates B7 (ranking within the generated set) with GENERATOR coverage
   (B5's job) -- out of scope.
3. **Adaptive goal-line sweep.** Isolation: ``hands_left`` and
   ``discards_left`` are forced to 1, so an action's faithful value is exactly
   ``P(best play after this one discard >= chips_needed)`` -- no downstream
   solver model, no future-hand boundary. The per-draw best-play totals are
   goal-line-INDEPENDENT, so we sample them ONCE per action and threshold at
   every goal line for free. Goal lines are quantiles of the pooled achievable
   totals that sit ABOVE the best play-now total ``P`` (below ``P`` the label
   is not even a discard -> B7 irrelevant); this adapts the band to each
   state's real reachable range instead of a fixed multiple.
4. **MC-reseed noise floor.** Faithful values are Monte Carlo, so a paired
   diff below label-reproducibility noise is meaningless. A separate n<=8
   pass reseeds the best action and measures the mean pairwise |delta|; a
   state counts as a B7 win only if its ``max_help`` clears
   ``accept_factor x floor``.

Metrics:
  - ``disagreement_rate``: how often the boxes differ at all.
  - among disagreements with a non-empty band, per state ``max_help`` (B7
    upside over goal lines) and ``min_paired`` (a NEGATIVE value is a
    REGRESSION -- new dropped a strictly better branch; must stay ~0).
  - ``n_regressions`` (below -floor) is the safety gate; ``frac_helped`` /
    ``mean_max_help`` quantify the upside.

The full-solver existence proofs (constructed states where the EMITTED
discard flips old->new toward an action with strictly higher FAITHFUL value)
live in ``tests/scripts/test_hand_solver_discard_ranking.py`` -- that is the
primary correctness argument; this aggregate is the how-often / does-it-
regress companion.

Usage::

    .venv/Scripts/python.exe scripts/validate_discard_ranking.py \
        --n-states 200 --n-samples 80 --output data/discard_ranking_validation.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_solver import (  # noqa: E402
    RANK_ID,
    DeckComposition,
    best_immediate_play,
    rank_templates_cheaply,
)

from jackdaw.engine.card_factory import create_playing_card  # noqa: E402
from jackdaw.engine.data.enums import Rank  # noqa: E402
from jackdaw.engine.data.enums import Suit as SuitEnum  # noqa: E402
from jackdaw.engine.play_ordering import best_joker_order  # noqa: E402
from validate_prescreen import sample_state  # noqa: E402

SEED_PREFIX = "DISCARD_RANK_VAL"
VALUE_TOLERANCE = 1e-9
# Goal-line positions as quantiles of the pooled achievable-total
# distribution; only those above the best play-now total are swept.
GOAL_QUANTILES = (0.5, 0.65, 0.8, 0.9, 0.97)
_ID_TO_RANK = {v: k for k, v in RANK_ID.items()}


def _action_key(hand: list, discard: Iterable) -> tuple[int, ...]:
    discard_ids = {id(card) for card in discard}
    return tuple(i for i, card in enumerate(hand) if id(card) in discard_ids)


def _dedup_box(candidates: list[tuple], hand: list) -> list[tuple[tuple[int, ...], tuple]]:
    """Collapse a raw top-k template box to unique physical discard actions,
    first-wins. Empty-discard entries are dropped -- they are not discards."""
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[tuple[int, ...], tuple]] = []
    for cand in candidates:
        key = _action_key(hand, cand[3])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((key, cand))
    return out


def _deck_pool(deck: DeckComposition) -> list[tuple[int, str]]:
    pool: list[tuple[int, str]] = []
    for (rid, suit), n in deck.by_rank_suit.items():
        pool.extend([(rid, suit)] * n)
    return pool


def faithful_totals(
    *,
    hand: list,
    discard_key: tuple[int, ...],
    jokers: list,
    hand_levels,
    blind,
    rng,
    pool: list[tuple[int, str]],
    game_state: dict | None,
    blind_chips: int,
    n_samples: int,
    mc_seed: str,
) -> list[float] | None:
    """Best-play totals over ``n_samples`` REAL random redraws of the
    discarded slots (goal-line-independent). ``None`` if the deck can't
    supply the draw. ``best_immediate_play`` deep-copies rng internally, so
    passing the live rng is safe (same contract as
    ``estimate_future_hand_distribution``)."""
    kept = [c for i, c in enumerate(hand) if i not in set(discard_key)]
    n_draw = len(discard_key)
    if len(pool) < n_draw:
        return None
    sampler = random.Random(mc_seed)
    totals: list[float] = []
    for _ in range(n_samples):
        drawn = sampler.sample(pool, n_draw)
        new_hand = kept + [
            create_playing_card(SuitEnum(suit), Rank(_ID_TO_RANK[rid])) for rid, suit in drawn
        ]
        _, result = best_immediate_play(
            new_hand, jokers, hand_levels, blind, rng, game_state, blind_chips
        )
        totals.append(float(result.total))
    return totals


def _value_at(totals: list[float], goal: float) -> float:
    return float(np.mean([t >= goal for t in totals]))


def _score_state(state, *, top_k: int, n_samples: int) -> dict[str, Any]:
    started = time.time()
    hand = state.hand
    jokers = best_joker_order(state.jokers)
    deck = DeckComposition.from_deck(state.gs.get("deck", []))

    common = dict(
        four_fingers=state.four_fingers,
        shortcut=state.shortcut,
        jokers=jokers,
        game_state=state.gs,
        blind_chips=state.blind_chips,
    )
    old_box = _dedup_box(
        rank_templates_cheaply(
            hand, deck, state.hand_levels, state.blind, state.rng,
            top_k=top_k, joker_aware=False, **common,
        ),
        hand,
    )
    new_box = _dedup_box(
        rank_templates_cheaply(
            hand, deck, state.hand_levels, state.blind, state.rng,
            top_k=top_k, joker_aware=True, **common,
        ),
        hand,
    )
    old_keys = {key for key, _ in old_box}
    new_keys = {key for key, _ in new_box}
    disagree = old_keys != new_keys

    row: dict[str, Any] = {
        "seed": state.seed,
        "discards_left": 1,
        "n_jokers": len(jokers),
        "joker_keys": [getattr(j, "center_key", None) for j in jokers],
        "old_box": [list(k) for k in old_keys],
        "new_box": [list(k) for k in new_keys],
        "disagree": disagree,
        "category": "agree",
        "seconds": round(time.time() - started, 2),
    }
    if not disagree:
        return row  # identical box -> identical label -> paired diff == 0

    union_keys = old_keys | new_keys
    pool = _deck_pool(deck)
    totals_by_key: dict[tuple[int, ...], list[float]] = {}
    for key in union_keys:
        totals = faithful_totals(
            hand=hand, discard_key=key, jokers=jokers, hand_levels=state.hand_levels,
            blind=state.blind, rng=state.rng, pool=pool, game_state=state.gs,
            blind_chips=state.blind_chips, n_samples=n_samples,
            mc_seed=f"{state.seed}:{key}",
        )
        if totals is not None:
            totals_by_key[key] = totals

    play_now_total = float(
        best_immediate_play(
            hand, jokers, state.hand_levels, state.blind, state.rng,
            state.gs, state.blind_chips,
        )[1].total
    )
    row["play_now_total"] = play_now_total

    pooled = [t for totals in totals_by_key.values() for t in totals]
    if not pooled or max(pooled) <= play_now_total + VALUE_TOLERANCE:
        # Nothing a discard can reach beats playing now -> B7 cannot move the
        # label on this state.
        row["category"] = "play_dominated"
        return row

    goals = sorted(
        {
            float(g)
            for g in np.quantile(pooled, GOAL_QUANTILES)
            if g > play_now_total + VALUE_TOLERANCE
        }
    )
    if not goals:
        row["category"] = "play_dominated"
        return row

    row["category"] = "measured"
    sweep = []
    for goal in goals:
        best_old = max(
            (_value_at(totals_by_key[k], goal) for k in old_keys if k in totals_by_key),
            default=0.0,
        )
        best_new = max(
            (_value_at(totals_by_key[k], goal) for k in new_keys if k in totals_by_key),
            default=0.0,
        )
        sweep.append(
            {
                "goal": goal,
                "best_in_old": best_old,
                "best_in_new": best_new,
                "paired_diff": best_new - best_old,
            }
        )
    diffs = [pt["paired_diff"] for pt in sweep]
    row["sweep"] = sweep
    row["max_help"] = max(diffs)
    row["min_paired"] = min(diffs)
    row["seconds"] = round(time.time() - started, 2)
    return row


def noise_floor(config, *, n_states: int, n_seeds: int, n_samples: int) -> dict[str, Any]:
    """Faithful-MC reproducibility noise: the SAME state's faithful-best
    discard, valued under different MC seeds at a mid goal line. Mean pairwise
    |delta| over seed pairs, averaged across states."""
    per_state: list[float] = []
    produced = 0
    attempt = 0
    while produced < n_states and attempt < n_states * 60:
        seed = f"{SEED_PREFIX}_FLOOR_{attempt:08d}"
        attempt += 1
        state = sample_state(seed, config)
        cr = state.gs["current_round"]
        if int(cr.get("discards_left", 0)) < 1 or len(state.hand) > 8:
            continue
        if state.hands_left <= 0 or not state.jokers:
            continue
        state.hands_left = 1
        cr["hands_left"] = 1
        state.gs["hands_left"] = 1
        jokers = best_joker_order(state.jokers)
        deck = DeckComposition.from_deck(state.gs.get("deck", []))
        pool = _deck_pool(deck)
        play_now = float(
            best_immediate_play(
                hand := state.hand, jokers, state.hand_levels, state.blind, state.rng,
                state.gs, state.blind_chips,
            )[1].total
        )
        # A representative discard: drop the lowest card, redraw one.
        discard_key = (len(hand) - 1,)
        base = faithful_totals(
            hand=hand, discard_key=discard_key, jokers=jokers,
            hand_levels=state.hand_levels, blind=state.blind, rng=state.rng, pool=pool,
            game_state=state.gs, blind_chips=state.blind_chips, n_samples=n_samples,
            mc_seed=f"{seed}#fs0",
        )
        if base is None or max(base) <= play_now:
            continue
        goal = float(np.quantile(base, 0.8))
        values = [_value_at(base, goal)]
        for j in range(1, n_seeds):
            reseeded = faithful_totals(
                hand=hand, discard_key=discard_key, jokers=jokers,
                hand_levels=state.hand_levels, blind=state.blind, rng=state.rng, pool=pool,
                game_state=state.gs, blind_chips=state.blind_chips, n_samples=n_samples,
                mc_seed=f"{seed}#fs{j}",
            )
            values.append(_value_at(reseeded, goal))
        diffs = [abs(a - b) for a, b in itertools.combinations(values, 2)]
        per_state.append(float(np.mean(diffs)) if diffs else 0.0)
        produced += 1
    return {
        "n_states": produced,
        "n_seeds": n_seeds,
        "mean_pairwise_abs_delta": float(np.mean(per_state)) if per_state else None,
    }


def _aggregate(rows: list[dict[str, Any]], floor: float | None, accept_factor: float) -> dict[str, Any]:
    n = len(rows)
    disagreements = [r for r in rows if r["disagree"]]
    measured = [r for r in disagreements if r["category"] == "measured"]
    threshold = (accept_factor * floor) if floor is not None else VALUE_TOLERANCE
    helped = [r for r in measured if r["max_help"] > threshold]
    regressions = [r for r in measured if r["min_paired"] < -threshold]
    max_helps = [r["max_help"] for r in measured]
    return {
        "n_states": n,
        "n_disagreements": len(disagreements),
        "disagreement_rate": (len(disagreements) / n) if n else None,
        "n_play_dominated": sum(r["category"] == "play_dominated" for r in disagreements),
        "n_measured": len(measured),
        "noise_floor": floor,
        "help_threshold": threshold,
        "n_helped_above_floor": len(helped),
        "n_regressions_below_floor": len(regressions),
        "regression_seeds": [r["seed"] for r in regressions],
        "frac_helped": (len(helped) / len(measured)) if measured else None,
        "mean_max_help": float(np.mean(max_helps)) if max_helps else None,
        "max_max_help": float(np.max(max_helps)) if max_helps else None,
        "worst_paired_diff": (float(min(r["min_paired"] for r in measured)) if measured else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="stage3_full")
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=80, help="MC draws per action")
    parser.add_argument("--floor-states", type=int, default=20)
    parser.add_argument("--floor-seeds", type=int, default=4)
    parser.add_argument("--accept-factor", type=float, default=1.33)
    parser.add_argument("--output", type=Path, default=Path("data/discard_ranking_validation.json"))
    args = parser.parse_args()
    if args.n_states <= 0 or args.top_k <= 0 or args.n_samples <= 0:
        parser.error("--n-states, --top-k, --n-samples must be positive")

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

        # Isolate one discard node: force hands_left=1 (no future-hand MC) and
        # discards_left=1 (the leaf ranking decision, valued faithfully).
        state.hands_left = 1
        cr["hands_left"] = 1
        state.gs["hands_left"] = 1
        cr["discards_left"] = 1

        row = _score_state(state, top_k=args.top_k, n_samples=args.n_samples)
        rows.append(row)
        tag = row["category"]
        extra = ""
        if tag == "measured":
            extra = f" max_help={row['max_help']:.4f} min_paired={row['min_paired']:.4f}"
        print(
            f"[{len(rows)}/{args.n_states}] seed={state.seed} disagree={row['disagree']} "
            f"{tag}{extra} t={row['seconds']:.2f}s",
            flush=True,
        )

    print("computing n<=8 MC-reseed noise floor...", flush=True)
    floor = noise_floor(
        config, n_states=args.floor_states, n_seeds=args.floor_seeds, n_samples=args.n_samples
    )
    summary = _aggregate(rows, floor["mean_pairwise_abs_delta"], args.accept_factor)
    summary.update(
        {
            "stage": args.stage,
            "top_k": args.top_k,
            "n_samples": args.n_samples,
            "accept_factor": args.accept_factor,
            "goal_quantiles": list(GOAL_QUANTILES),
            "attempts": attempt,
            "wall_seconds": round(time.time() - t_start, 1),
        }
    )
    report = {"summary": summary, "floor_detail": floor, "states": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["n_regressions_below_floor"]:
        print(
            f"REGRESSIONS on {summary['n_regressions_below_floor']} state(s) "
            f"{summary['regression_seeds']}: new ranker dropped a strictly better "
            "branch than old kept (beyond noise) -- investigate before trusting B7."
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
