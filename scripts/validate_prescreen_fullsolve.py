"""Full-solve arm of the prescreen gate (K2): node-level capture/regret at
TRUE depth, measured inside real solves.

WHY ROOT-ONLY GATING IS REJECTED (CLAUDE.md "Kicker variants +
prescreen-at-n=8"): the root harness (`validate_prescreen_n8.py`) measures
the prescreen where it fires LEAST -- one node of ~488 per solve, and none
of them post-discard. Once `PRESCREEN_HAND_LIMIT` is deleted the prescreen
fires at EVERY `best_immediate_play` call: every recursion node (including
deep inside discard branches, on post-discard/redraw hands -- a different,
more conditioned distribution where per-node errors could compound) and
every MC future-hand sample. Deep misses are also DIRECTIONAL: an
undervalued future hand tilts the label toward playing NOW, compounding the
documented play-only MC bias. So the gate requires node-level capture at
true depth, not just at the root.

MECHANISM: `PrescreenNodeProbe` patches three module attributes of
`hand_solver` (all internal calls resolve through module globals, so
recursive and MC calls route through the wrappers):

  * `best_immediate_play` -- runs the ORIGINAL first and returns its result
    unchanged, so the solve under measurement is byte-identical to an
    uninstrumented one (the non-perturbation contract; `evaluate_value` and
    the prescreen fast-clone everything they touch, and nothing here calls
    `prob_clear_given_future`, so the module LCG is untouched -- pinned by
    test). Then, per node, computes the prescreen box at each k and scores
    capture-by-value / regret against truth. At n <= PRESCREEN_HAND_LIMIT
    the original call IS the brute force, so truth is free; above it, truth
    is an explicit C(n,1..5) enumeration (the original would have
    prescreened -- see `validate_prescreen_n8._brute_argmax`'s rationale).
  * `solve_hand_turn` -- maintains a `discards_left` stack so every node
    knows its true depth.
  * `estimate_future_hand_distribution` -- flags its extent so MC
    future-hand nodes land in the "mc" stratum (they run
    `search_orderings=False`, and box candidates are valued under the SAME
    flag as the node's own call -- comparing tiers would manufacture fake
    regret on order-sensitive boards).

ROOT-ACTION AGREEMENT is a smoke readout ONLY (never the gate): after the
instrumented brute solve, each k re-solves the same state with
`PRESCREEN_HAND_LIMIT` patched to 0 and `PRESCREEN_TOP_K` to k -- the exact
post-K3 production configuration, prescreen firing at every node INCLUDING
the MC sampler -- and compares the root choice and p_clear.

Cost: a stage2 solve is ~200s+ (the recursion), measurement adds ~15 exact
evals + one prescreen ranking per node per k. ~10 solves is tens of minutes
to a couple of hours at stage2 density -- run on the 9600X if painful
(pitfall #19). `--node-sample-prob` thins measurement (never the solve).

Gate run (K3)::

    uv run python scripts/validate_prescreen_fullsolve.py \
        --shard-dir data/stage_2_h1_shards --n-solves 10 \
        --out data/prescreen_fullsolve.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hand_solver  # noqa: E402
from hand_solver import (  # noqa: E402
    PRESCREEN_TOP_K,
    DeckComposition,
    evaluate_value,
    prescreen_play_candidates,
    solve_hand_for_ante_clear,
)
from validate_prescreen_n8 import build_state, config_from_manifest  # noqa: E402

from jackdaw.engine.hand_eval import get_hand_eval_flags  # noqa: E402


class PrescreenNodeProbe:
    """Counterfactual node-level prescreen measurement inside a real solve.

    Use as a context manager around `solve_hand_for_ante_clear`; every
    `best_immediate_play` call becomes a measured node in `self.records`
    while the solve's own outputs stay byte-identical (the wrappers call the
    originals first and return their results unchanged).

    A record::

        {"depth": <discards_left at the node, or "mc">, "is_root": bool,
         "hand_size": int, "search_orderings": bool, "truth_total": float,
         "by_k": {k: {"captured", "captured_by_value", "regret",
                      "rel_regret", "box_size"}}}
    """

    def __init__(
        self, ks: list[int], node_sample_prob: float = 1.0, seed: int = 0
    ) -> None:
        self.ks = ks
        self.node_sample_prob = node_sample_prob
        self.records: list[dict[str, Any]] = []
        self._rand = random.Random(seed)
        self._depth_stack: list[int] = []
        self._mc_depth = 0
        self._orig_bip = None
        self._orig_sht = None
        self._orig_mc = None

    # -- patching ------------------------------------------------------------

    def __enter__(self) -> PrescreenNodeProbe:
        self._orig_bip = hand_solver.best_immediate_play
        self._orig_sht = hand_solver.solve_hand_turn
        self._orig_mc = hand_solver.estimate_future_hand_distribution
        hand_solver.best_immediate_play = self._wrapped_best_immediate_play
        hand_solver.solve_hand_turn = self._wrapped_solve_hand_turn
        hand_solver.estimate_future_hand_distribution = self._wrapped_mc
        return self

    def __exit__(self, *exc: object) -> None:
        hand_solver.best_immediate_play = self._orig_bip
        hand_solver.solve_hand_turn = self._orig_sht
        hand_solver.estimate_future_hand_distribution = self._orig_mc

    # -- wrappers ------------------------------------------------------------

    def _wrapped_solve_hand_turn(self, *args: Any, **kwargs: Any) -> Any:
        # discards_left is positional arg 8 of solve_hand_turn (hand, jokers,
        # hand_levels, blind, rng, deck, chips_needed, hands_left,
        # discards_left, ...) or a kwarg.
        discards_left = kwargs.get("discards_left", args[8] if len(args) > 8 else 0)
        self._depth_stack.append(int(discards_left))
        try:
            return self._orig_sht(*args, **kwargs)
        finally:
            self._depth_stack.pop()

    def _wrapped_mc(self, *args: Any, **kwargs: Any) -> Any:
        self._mc_depth += 1
        try:
            return self._orig_mc(*args, **kwargs)
        finally:
            self._mc_depth -= 1

    def _wrapped_best_immediate_play(
        self,
        hand: list,
        jokers: list,
        hand_levels: Any,
        blind: Any,
        rng: Any,
        game_state: dict | None = None,
        blind_chips: int = 0,
        *,
        search_orderings: bool = True,
        prescreen_top_k: int | None = None,
    ) -> Any:
        subset, result = self._orig_bip(
            hand, jokers, hand_levels, blind, rng, game_state, blind_chips,
            search_orderings=search_orderings, prescreen_top_k=prescreen_top_k,
        )
        if self._rand.random() < self.node_sample_prob:
            self._measure(
                hand, jokers, hand_levels, blind, rng, game_state, blind_chips,
                search_orderings, subset, result,
            )
        return subset, result

    # -- measurement ---------------------------------------------------------

    def _value(
        self,
        cards: list,
        hand: list,
        jokers: list,
        hand_levels: Any,
        blind: Any,
        rng: Any,
        game_state: dict | None,
        blind_chips: int,
        search_orderings: bool,
    ) -> float:
        ids = {id(c) for c in cards}
        held = [c for c in hand if id(c) not in ids]
        return float(
            evaluate_value(
                list(cards), held, jokers, hand_levels, blind, rng,
                game_state, blind_chips, search_orderings=search_orderings,
            ).total
        )

    def _measure(
        self,
        hand: list,
        jokers: list,
        hand_levels: Any,
        blind: Any,
        rng: Any,
        game_state: dict | None,
        blind_chips: int,
        search_orderings: bool,
        subset: list,
        result: Any,
    ) -> None:
        n = len(hand)
        if n <= hand_solver.PRESCREEN_HAND_LIMIT:
            # The original call brute-forced every subset: its own argmax IS
            # the truth, already paid for.
            truth_total = float(result.total)
            truth_ids = frozenset(id(c) for c in subset)
        else:
            # The original call PRESCREENED -- enumerate explicitly, under
            # the node's own search_orderings tier.
            truth_total, truth_ids = None, None
            for size in range(1, min(5, n) + 1):
                for combo in itertools.combinations(hand, size):
                    total = self._value(
                        list(combo), hand, jokers, hand_levels, blind, rng,
                        game_state, blind_chips, search_orderings,
                    )
                    if truth_total is None or total > truth_total:
                        truth_total = total
                        truth_ids = frozenset(id(c) for c in combo)
            assert truth_total is not None and truth_ids is not None

        flags = get_hand_eval_flags(jokers)
        by_k: dict[int, dict[str, Any]] = {}
        for k in self.ks:
            box = prescreen_play_candidates(
                hand, jokers, hand_levels, blind, rng,
                four_fingers=flags["four_fingers"],
                shortcut=flags["shortcut"],
                smeared=flags["smeared"],
                top_k=k,
                game_state=game_state,
                blind_chips=blind_chips,
                eval_flags=flags,
            )
            best_in_box = max(
                (
                    self._value(
                        cards, hand, jokers, hand_levels, blind, rng,
                        game_state, blind_chips, search_orderings,
                    )
                    for cards in box
                ),
                default=0.0,
            )
            regret = max(0.0, truth_total - best_in_box)
            by_k[k] = {
                "captured": truth_ids in (frozenset(id(c) for c in cards) for cards in box),
                "captured_by_value": regret <= 1e-6,
                "regret": regret,
                "rel_regret": regret / truth_total if truth_total > 0 else 0.0,
                "box_size": len(box),
            }

        if self._mc_depth > 0:
            depth: int | str = "mc"
            is_root = False
        else:
            depth = self._depth_stack[-1] if self._depth_stack else "bare"
            is_root = len(self._depth_stack) == 1
        self.records.append(
            {
                "depth": depth,
                "is_root": is_root,
                "hand_size": n,
                "search_orderings": search_orderings,
                "truth_total": truth_total,
                "by_k": by_k,
            }
        )


# -- driver -------------------------------------------------------------------


def _solve(gs: dict[str, Any], mc_seed: str) -> Any:
    """The exact production labeling call (`generate_hand_demos.
    label_and_encode`'s solver invocation) on a built state."""
    blind = gs["blind"]
    cr = gs["current_round"]
    blind_chips = getattr(blind, "chips", 0) if blind else 0
    flags = get_hand_eval_flags(gs["jokers"])
    return solve_hand_for_ante_clear(
        gs["hand"],
        gs["jokers"],
        gs["hand_levels"],
        blind,
        gs["rng"],
        DeckComposition.from_deck(gs.get("deck", [])),
        max(0.0, blind_chips - gs.get("chips", 0)),
        cr.get("hands_left", 0),
        cr.get("discards_left", 0),
        game_state=gs,
        blind_chips=blind_chips,
        four_fingers=flags["four_fingers"],
        shortcut=flags["shortcut"],
        mc_seed=mc_seed,
    )


def _choice_key(choice: Any, hand: list) -> dict[str, Any]:
    by_id = {id(c): i for i, c in enumerate(hand)}
    return {
        "action": choice.action,
        "template": choice.template_name,
        "hold": sorted(by_id[id(c)] for c in choice.hold if id(c) in by_id),
        "discard": sorted(by_id[id(c)] for c in choice.discard if id(c) in by_id),
        "p_clear": float(choice.p_clear),
    }


def run_smoke_solve(gs: dict[str, Any], mc_seed: str, k: int) -> Any:
    """Re-solve with the prescreen forced at EVERY hand size -- the post-K3
    production configuration (limit deleted, box width k). Prescreen fires
    at every recursion node AND inside the MC future-hand sampler, so the
    p_clear delta vs the brute solve is the true end-to-end label shift."""
    old_limit = hand_solver.PRESCREEN_HAND_LIMIT
    old_k = hand_solver.PRESCREEN_TOP_K
    hand_solver.PRESCREEN_HAND_LIMIT = 0
    hand_solver.PRESCREEN_TOP_K = k
    try:
        return _solve(gs, mc_seed)
    finally:
        hand_solver.PRESCREEN_HAND_LIMIT = old_limit
        hand_solver.PRESCREEN_TOP_K = old_k


def _summarize_nodes(records: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_nodes": len(records), "by_k": {}}
    for k in ks:
        rows = [r["by_k"][k] for r in records]
        n = max(1, len(rows))
        capv = sum(r["captured_by_value"] for r in rows)
        out["by_k"][str(k)] = {
            "capture_rate_by_value": round(capv / n, 4),
            "n_missed_by_value": len(rows) - capv,
            "capture_rate_strict": round(sum(r["captured"] for r in rows) / n, 4),
            "mean_regret": round(sum(r["regret"] for r in rows) / n, 3),
            "max_regret": round(max((r["regret"] for r in rows), default=0.0), 2),
            "mean_rel_regret": round(sum(r["rel_regret"] for r in rows) / n, 5),
            "max_rel_regret": round(max((r["rel_regret"] for r in rows), default=0.0), 4),
            "mean_box_size": round(sum(r["box_size"] for r in rows) / n, 2),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=Path("data/stage_2_h1_shards"))
    parser.add_argument(
        "--stage-preset",
        default=None,
        help="config from stage_presets() instead of the shard manifest",
    )
    parser.add_argument("--stage-name", default=None)
    parser.add_argument("--total-examples", type=int, default=4000)
    parser.add_argument("--n-solves", type=int, default=10)
    parser.add_argument(
        "--min-discards", type=int, default=2, help="deep recursions = many measured nodes"
    )
    parser.add_argument(
        "--ks", default=str(PRESCREEN_TOP_K), help="comma list; default = the production k"
    )
    parser.add_argument("--node-sample-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/prescreen_fullsolve.json"))
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    stage_name = args.stage_name or args.stage_preset or "stage2_curated"
    if args.stage_preset:
        from generate_hand_demos import stage_presets

        config = stage_presets()[args.stage_preset].config
    else:
        manifest = json.loads((args.shard_dir / "manifest.json").read_text(encoding="utf-8"))
        config = config_from_manifest(manifest)

    rng = random.Random(args.seed)
    indices = list(range(args.total_examples))
    rng.shuffle(indices)

    states: list[dict[str, Any]] = []
    node_records: list[dict[str, Any]] = []
    for idx in indices:
        if len(states) >= args.n_solves:
            break
        seed = f"{stage_name}_{idx:08d}"
        gs = build_state(seed, config)
        cr = gs["current_round"]
        blind = gs["blind"]
        blind_chips = getattr(blind, "chips", 0) if blind else 0
        if len(gs["hand"]) != 8:
            continue  # the n=8 question; the tail arm lives in the root harness
        if cr.get("discards_left", 0) < args.min_discards:
            continue
        if blind_chips - gs.get("chips", 0) <= 0:
            continue  # already cleared: 1-node solve, nothing to measure

        t0 = time.perf_counter()
        probe = PrescreenNodeProbe(ks, args.node_sample_prob, seed=args.seed)
        with probe:
            brute_choice = _solve(gs, seed)
        solve_seconds = time.perf_counter() - t0

        brute_key = _choice_key(brute_choice, gs["hand"])
        smoke: dict[str, Any] = {}
        for k in ks:
            t1 = time.perf_counter()
            pres_choice = run_smoke_solve(gs, seed, k)
            pres_key = _choice_key(pres_choice, gs["hand"])
            smoke[str(k)] = {
                **pres_key,
                "action_match": pres_key["action"] == brute_key["action"],
                "exact_match": (
                    pres_key["action"] == brute_key["action"]
                    and pres_key["hold"] == brute_key["hold"]
                    and pres_key["discard"] == brute_key["discard"]
                ),
                "p_clear_delta": pres_key["p_clear"] - brute_key["p_clear"],
                "solve_seconds": round(time.perf_counter() - t1, 1),
            }

        node_records.extend(probe.records)
        states.append(
            {
                "seed": seed,
                "discards_left": int(cr.get("discards_left", 0)),
                "hands_left": int(cr.get("hands_left", 0)),
                "n_jokers": len(gs["jokers"]),
                "n_nodes": len(probe.records),
                "brute_solve_seconds": round(solve_seconds, 1),
                "brute_root": brute_key,
                "prescreened_root": smoke,
            }
        )
        print(
            f"[{len(states)}/{args.n_solves}] {seed} d={states[-1]['discards_left']} "
            f"nodes={len(probe.records)} brute={solve_seconds:.0f}s",
            flush=True,
        )

    # Depth-stratified node summary: recursion depths (discards_left at the
    # node), the MC future-hand stratum, and roots alone.
    strata: dict[str, list[dict[str, Any]]] = {}
    for rec in node_records:
        strata.setdefault(f"d{rec['depth']}" if rec["depth"] != "mc" else "mc", []).append(rec)
    root_records = [r for r in node_records if r["is_root"]]

    root_agreement: dict[str, Any] = {}
    for k in ks:
        rows = [s["prescreened_root"][str(k)] for s in states]
        n = max(1, len(rows))
        deltas = [abs(r["p_clear_delta"]) for r in rows]
        root_agreement[str(k)] = {
            "n_states": len(rows),
            "action_match_rate": round(sum(r["action_match"] for r in rows) / n, 3),
            "exact_match_rate": round(sum(r["exact_match"] for r in rows) / n, 3),
            "mean_abs_p_clear_delta": round(sum(deltas) / n, 5),
            "max_abs_p_clear_delta": round(max(deltas, default=0.0), 5),
        }

    report = {
        "params": {
            "ks": ks,
            "seed": args.seed,
            "stage_name": stage_name,
            "n_solves": len(states),
            "min_discards": args.min_discards,
            "node_sample_prob": args.node_sample_prob,
            "prescreen_top_k_default": PRESCREEN_TOP_K,
        },
        "nodes": {
            "note": "counterfactual measurement inside brute solves; capture/regret "
            "on best_immediate_play's own objective at every node's true depth",
            "overall": _summarize_nodes(node_records, ks),
            "roots_only": _summarize_nodes(root_records, ks),
            "by_stratum": {
                name: _summarize_nodes(recs, ks) for name, recs in sorted(strata.items())
            },
        },
        "root_agreement_smoke": {
            "note": "smoke readout ONLY, never the gate: full re-solve with the "
            "prescreen forced everywhere (limit=0), root choice + p_clear vs brute",
            **root_agreement,
        },
        "states": states,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "states"}, indent=2))


if __name__ == "__main__":
    main()
