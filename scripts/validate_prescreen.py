"""One-time validation harness for the big-hand play prescreen (B5).

Question it answers: how much p_clear does labeling lose by letting
`best_immediate_play` evaluate only the top-k prescreened candidates on
hands wider than 8, instead of full C(n,5) brute force? The answer picks
PRESCREEN_TOP_K in `hand_solver.py` (record it there and in CLAUDE.md).

Method (decision record: CLAUDE.md "Pre-regeneration build plan", B5):

  - ~``--n-hands`` states dealt FLAT over hand sizes 9-12 via B1's flat
    hand-size tail knob on a stage preset's config (default stage3_full --
    the prescreen must hold up under jokers, not just bare boards).
  - Per state: ONE full brute-force pass exactly-evaluates every subset,
    then every ``--k-cuts`` cut is scored from those same evaluations via
    one `prescreen_play_candidates` call PER k (K2). Since K1, `top_k`
    counts LINES and every surviving line carries its kicker variants, so
    the output is prefix-stable at line granularity but NOT indexable by
    k -- slicing ``[:k]`` from one max-k call would cut mid-line and
    misassign every cut.
  - Metric = REGRET, not disagreement (pitfall #12): p_clear(brute-force
    best) - p_clear(prescreen's choice), BOTH valued by brute force.
    Near-tied plays that "disagree" harmlessly cost ~0 here.
  - Noise floor: the same regret is meaningless below label-reproducibility
    noise, so ``--floor-states`` n<=8 states are re-valued under
    ``--floor-seeds`` different MC seeds; the floor is the mean pairwise
    |delta p_clear| of the SAME best play across seeds. Accept the smallest
    k with mean regret <= ``--accept-factor`` (default 1.33) x floor.

Heavy at n=11-12 with order-sensitive jokers (permutation search inside
every exact evaluation) -- run on the 9600X if local runtime is painful
(pitfall #19). Seeds use the reserved ``PRESCREEN_VAL_`` prefix: never
``EVAL_`` (held-out suite), never ``HARVEST_``.

Usage::

    uv run python scripts/validate_prescreen.py --n-hands 48 \
        --output data/prescreen_validation.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hand_solver  # noqa: E402
from hand_solver import (  # noqa: E402
    DeckComposition,
    estimate_future_hand_distribution,
    evaluate_value,
    prescreen_play_candidates,
    prob_clear_given_future,
)

from jackdaw.engine.hand_eval import get_hand_eval_flags  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayAdapter  # noqa: E402

SEED_PREFIX = "PRESCREEN_VAL"


def _reset_pclear_lcg(tag: str) -> None:
    """`prob_clear_given_future` draws from a module-level LCG that carries
    state across calls; pin it per (state, purpose) so every p_clear here is
    reproducible -- same mechanism as `solve_hand_for_ante_clear`."""
    hand_solver._rand_state[0] = zlib.crc32(tag.encode()) & 0x7FFFFFFF or 12345


class _StateBundle:
    """Everything one sampled decision state needs, pulled once."""

    def __init__(self, seed: str, gs: dict) -> None:
        self.seed = seed
        self.hand = gs["hand"]
        self.jokers = gs["jokers"]
        self.hand_levels = gs["hand_levels"]
        self.blind = gs["blind"]
        self.rng = gs["rng"]
        self.gs = gs
        cr = gs["current_round"]
        self.hands_left = cr.get("hands_left", 0)
        self.blind_chips = getattr(self.blind, "chips", 0) if self.blind else 0
        self.chips_needed = max(0.0, self.blind_chips - gs.get("chips", 0))
        flags = get_hand_eval_flags(self.jokers)
        self.four_fingers = flags["four_fingers"]
        self.shortcut = flags["shortcut"]


def sample_state(seed: str, config) -> _StateBundle:
    adapter = HandPlayAdapter(config)
    adapter.reset("b_red", 1, seed)
    return _StateBundle(seed, adapter.raw_state)


def brute_force_totals(state: _StateBundle) -> dict[frozenset[int], float]:
    """Exact evaluate_value total for EVERY playable subset, keyed by the
    subset's card-identity set. The single expensive pass everything else
    reads from."""
    totals: dict[frozenset[int], float] = {}
    hand = state.hand
    for size in range(1, min(5, len(hand)) + 1):
        for combo in itertools.combinations(hand, size):
            combo_ids = frozenset(id(c) for c in combo)
            held = [c for c in hand if id(c) not in combo_ids]
            result = evaluate_value(
                list(combo),
                held,
                state.jokers,
                state.hand_levels,
                state.blind,
                state.rng,
                state.gs,
                state.blind_chips,
            )
            totals[combo_ids] = float(result.total)
    return totals


def make_pclear_gap(state: _StateBundle, fs_seed: str) -> callable:
    """p_clear-of-a-remaining-gap valuation under one MC seed: shared
    future_samples + memoized `prob_clear_given_future` per distinct gap
    (identical gaps MUST map to identical p_clear, or comparisons drown
    in resample jitter). Gap-keyed rather than total-keyed so the boundary
    stress sweep can re-value the same totals under synthetic blind
    sizes."""
    future_samples = estimate_future_hand_distribution(
        DeckComposition.from_deck(state.gs.get("deck", [])),
        state.jokers,
        state.hand_levels,
        state.blind,
        state.rng,
        len(state.hand),
        game_state=state.gs,
        blind_chips=state.blind_chips,
        mc_seed=fs_seed,
    )
    memo: dict[float, float] = {}

    def p_clear_gap(gap: float) -> float:
        if gap not in memo:
            if gap <= 0:
                memo[gap] = 1.0
            elif state.hands_left - 1 <= 0:
                memo[gap] = 0.0
            else:
                _reset_pclear_lcg(f"{fs_seed}:{gap}")
                memo[gap] = prob_clear_given_future(
                    gap, state.hands_left - 1, future_samples
                )
        return memo[gap]

    return p_clear_gap


def _argmax_first(pairs: list[tuple[frozenset[int], float]]) -> tuple[frozenset[int], float]:
    """First-wins argmax by total -- matches best_immediate_play's strict
    `>` comparison so the harness scores the choice the solver would make."""
    best_key, best_total = pairs[0]
    for key, total in pairs[1:]:
        if total > best_total:
            best_key, best_total = key, total
    return best_key, best_total


# Boundary stress: synthetic blind sizes as fractions of the best play's
# total. The sampled state distribution is mostly saturated (hopeless or
# safe boards where every play has identical p_clear -- observed 45/48 in
# the first run), so sampled-distribution regret alone under-exercises the
# metric; the sweep asks what a prescreen miss WOULD cost if the blind
# landed near the decision boundary. Diagnostic only -- acceptance stays on
# the sampled distribution per the locked B5 spec.
STRESS_FRACTIONS = (0.9, 1.0, 1.1, 1.5)


def score_one_hand(state: _StateBundle, k_cuts: list[int]) -> dict:
    t0 = time.time()
    totals = brute_force_totals(state)
    brute_time = time.time() - t0

    p_clear_gap = make_pclear_gap(state, state.seed)

    def p_clear(total: float) -> float:
        return p_clear_gap(state.chips_needed - total)
    # Brute best in enumeration order (sizes ascending), first-wins on ties
    # -- identical to the pre-prescreen best_immediate_play.
    hand = state.hand
    enum_pairs: list[tuple[frozenset[int], float]] = []
    for size in range(1, min(5, len(hand)) + 1):
        for combo in itertools.combinations(hand, size):
            key = frozenset(id(c) for c in combo)
            enum_pairs.append((key, totals[key]))
    _, best_total = _argmax_first(enum_pairs)
    best_p = p_clear(best_total)

    # One prescreen call per k (K2): top_k counts LINES and variants ride,
    # so a j-line cut must come from its own top_k=j call -- the max-k
    # output is a line-granular prefix but entry j is not line j.
    cut_pairs_by_k: dict[int, list[tuple[frozenset[int], float]]] = {}
    for k in k_cuts:
        candidates = prescreen_play_candidates(
            hand,
            state.jokers,
            state.hand_levels,
            state.blind,
            state.rng,
            four_fingers=state.four_fingers,
            shortcut=state.shortcut,
            top_k=k,
            game_state=state.gs,
            blind_chips=state.blind_chips,
        )
        cut_pairs_by_k[k] = [
            (frozenset(id(c) for c in cards), totals[frozenset(id(c) for c in cards)])
            for cards in candidates
        ]
    cand_pairs = cut_pairs_by_k[max(k_cuts)]

    row: dict = {
        "seed": state.seed,
        "hand_size": len(hand),
        "n_jokers": len(state.jokers),
        "chips_needed": state.chips_needed,
        "hands_left": state.hands_left,
        "n_subsets": len(totals),
        "brute_seconds": round(brute_time, 2),
        "best_total": best_total,
        "best_p_clear": best_p,
        "best_in_candidates": any(t == best_total for _, t in cand_pairs),
        # MC-active = the best play does NOT clear outright, so p_clear is
        # a genuine MC estimate. The noise floor is measured on exactly
        # this population; acceptance compares active-mean regret to it.
        "mc_active": best_total < state.chips_needed,
        "cuts": {},
    }
    for k in k_cuts:
        cut = cut_pairs_by_k[k]
        _, cut_total = _argmax_first(cut)
        cut_p = p_clear(cut_total)
        stress = {}
        for f in STRESS_FRACTIONS:
            synthetic_needed = f * best_total
            stress[str(f)] = p_clear_gap(synthetic_needed - best_total) - p_clear_gap(
                synthetic_needed - cut_total
            )
        row["cuts"][str(k)] = {
            "choice_total": cut_total,
            "choice_p_clear": cut_p,
            "regret": best_p - cut_p,
            "n_candidates": len(cut),
            "stress_regret": stress,
        }
    return row


def noise_floor(config, n_states: int, n_seeds: int) -> dict:
    """Label-reproducibility noise at n<=8: the SAME state's brute-best play,
    p_clear-valued under different MC seeds. Mean pairwise |delta| over
    seed pairs, averaged across states."""
    per_state: list[float] = []
    rows: list[dict] = []
    produced = 0
    attempt = 0
    while produced < n_states and attempt < n_states * 40:
        seed = f"{SEED_PREFIX}_FLOOR_{attempt:08d}"
        attempt += 1
        state = sample_state(seed, config)
        if len(state.hand) > 8 or state.hands_left < 2 or state.chips_needed <= 0:
            continue  # need the p_clear MC boundary actually exercised
        totals = brute_force_totals(state)
        best_total = max(totals.values())
        if state.chips_needed - best_total <= 0:
            continue  # best play clears outright: p_clear == 1.0 under every
            # seed, no MC noise to measure -- would deflate the floor
        values = []
        for j in range(n_seeds):
            p_clear_gap = make_pclear_gap(state, f"{seed}#fs{j}")
            values.append(p_clear_gap(state.chips_needed - best_total))
        diffs = [
            abs(a - b) for a, b in itertools.combinations(values, 2)
        ]
        spread = float(np.mean(diffs)) if diffs else 0.0
        per_state.append(spread)
        rows.append({"seed": seed, "best_total": best_total, "p_clears": values})
        produced += 1
    return {
        "n_states": produced,
        "n_seeds": n_seeds,
        "mean_pairwise_abs_delta": float(np.mean(per_state)) if per_state else None,
        "per_state": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="stage3_full")
    parser.add_argument("--n-hands", type=int, default=48, help="total across sizes 9-12")
    parser.add_argument("--sizes", default="9,10,11,12")
    parser.add_argument("--k-cuts", default="3,5,8")
    parser.add_argument("--floor-states", type=int, default=20)
    parser.add_argument("--floor-seeds", type=int, default=4)
    parser.add_argument("--accept-factor", type=float, default=1.33)
    parser.add_argument("--output", type=Path, default=Path("data/prescreen_validation.json"))
    args = parser.parse_args()

    from generate_hand_demos import stage_presets

    base_config = stage_presets()[args.stage].config
    sizes = [int(s) for s in args.sizes.split(",")]
    k_cuts = sorted(int(k) for k in args.k_cuts.split(","))

    # Force the flat tail on: base hand size is 8 (+/- what injected jokers
    # add), so delta 1-4 spans the 9-12 target band; off-band deals are
    # simply skipped by the quota binning below.
    big_config = replace(
        base_config, hand_size_tail_prob=1.0, hand_size_delta_range=(1, 4)
    )

    quota = {s: args.n_hands // len(sizes) for s in sizes}
    quota[sizes[-1]] += args.n_hands - sum(quota.values())
    rows: list[dict] = []
    attempt = 0
    t_start = time.time()
    while any(q > 0 for q in quota.values()) and attempt < args.n_hands * 60:
        seed = f"{SEED_PREFIX}_{attempt:08d}"
        attempt += 1
        state = sample_state(seed, big_config)
        n = len(state.hand)
        if quota.get(n, 0) <= 0:
            continue
        if state.chips_needed <= 0:
            continue  # play-choice regret is trivially 0 when already cleared
        row = score_one_hand(state, k_cuts)
        quota[n] -= 1
        rows.append(row)
        done = len(rows)
        print(
            f"[{done}/{args.n_hands}] size={n} subsets={row['n_subsets']} "
            f"brute={row['brute_seconds']}s regrets="
            + ", ".join(f"k{k}={row['cuts'][str(k)]['regret']:.4f}" for k in k_cuts)
        )

    print("computing n<=8 MC-reseed noise floor...")
    floor = noise_floor(base_config, args.floor_states, args.floor_seeds)
    floor_value = floor["mean_pairwise_abs_delta"]

    summary: dict = {
        "stage": args.stage,
        "n_hands": len(rows),
        "wall_seconds": round(time.time() - t_start, 1),
        "k_cuts": {},
        "noise_floor": floor_value,
        "accept_factor": args.accept_factor,
        "chosen_k": None,
    }
    active_rows = [r for r in rows if r["mc_active"]]
    summary["n_mc_active"] = len(active_rows)
    for k in k_cuts:
        regrets = [r["cuts"][str(k)]["regret"] for r in rows]
        active_regrets = [r["cuts"][str(k)]["regret"] for r in active_rows]
        summary["k_cuts"][str(k)] = {
            "mean_regret": float(np.mean(regrets)),
            "mean_regret_mc_active": (
                float(np.mean(active_regrets)) if active_regrets else None
            ),
            "max_regret": float(np.max(regrets)),
            "p90_regret": float(np.percentile(regrets, 90)),
            "frac_zero": float(np.mean([r <= 1e-12 for r in regrets])),
            "best_in_cut_rate": float(
                np.mean(
                    [
                        r["cuts"][str(k)]["choice_total"] == r["best_total"]
                        for r in rows
                    ]
                )
            ),
            "stress_regret": {
                str(f): {
                    "mean": float(
                        np.mean([r["cuts"][str(k)]["stress_regret"][str(f)] for r in rows])
                    ),
                    "max": float(
                        np.max([r["cuts"][str(k)]["stress_regret"][str(f)] for r in rows])
                    ),
                }
                for f in STRESS_FRACTIONS
            },
        }
        # Acceptance compares the ACTIVE-mean (the floor's own population);
        # trivially-cleared boards would dilute regret against a floor that
        # deliberately excludes them.
        active_mean = summary["k_cuts"][str(k)]["mean_regret_mc_active"]
        if (
            summary["chosen_k"] is None
            and floor_value is not None
            and active_mean is not None
            and active_mean <= args.accept_factor * floor_value
        ):
            summary["chosen_k"] = k

    report = {"summary": summary, "hands": rows, "floor_detail": floor}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if summary["chosen_k"] is None:
        print(
            "NO k PASSED: raise k / widen families; if still failing, the prescreen "
            "design goes back to review -- do NOT ship a failing k (handoff B5)."
        )
    else:
        print(
            f"chosen k = {summary['chosen_k']} -- set PRESCREEN_TOP_K in "
            "scripts/hand_solver.py and record in CLAUDE.md."
        )


if __name__ == "__main__":
    main()
