"""One-time validation for B7 -- joker/held-aware discard-branch ranking.

`rank_templates_cheaply` now scores candidate templates with a joker- and
held-aware `score_hand` (see the B7 spec in docs/pre-regen-handoff.md).
That changes WHICH top-k branches `solve_hand_turn` explores at every hand
size, i.e. existing-label churn -- the discard-cap class of change that
must be validated before any label is generated with it.

Method: sample discard-rich states (discards >= 2 -- the population where
branch selection matters), solve each twice with the SAME `mc_seed`
(identical future-hand samples): once with the new ranking, once with the
old jokerless scorer (`joker_aware=False`, the escape hatch kept for this
harness). The solver's claimed p_clear for its chosen action is an honest
max over the explored branch set under identical valuation machinery, so

    delta = p_clear(new arm) - p_clear(old arm)

is the value of the ranking change. Better information should give
delta >= 0 up to MC noise; accept if mean delta >= -accept_factor x floor,
where the floor is the same-arm MC-reseed spread (re-solve under a second
mc_seed). A materially NEGATIVE mean delta means the new ranking is
steering the recursion into worse branches -- do not ship, investigate.

Also reports per-arm solve times (rank_templates_cheaply runs per
recursion node; score_hand with jokers costs a few x score_hand_base --
the budget is the ~12s/example regen envelope).

Usage::

    uv run python scripts/validate_discard_ranking.py --n-states 30 \
        --output data/discard_ranking_validation.json
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

import hand_solver  # noqa: E402
from hand_solver import DeckComposition, solve_hand_for_ante_clear  # noqa: E402
from validate_prescreen import sample_state  # noqa: E402

SEED_PREFIX = "DISCARD_RANK_VAL"


def solve_once(state, mc_seed: str, *, joker_aware: bool) -> dict:
    """One full solve; the old arm forces the legacy jokerless ranking by
    wrapping rank_templates_cheaply (solve_hand_turn resolves it through
    the module global, so the wrapper is picked up)."""
    cr = state.gs["current_round"]
    deck = DeckComposition.from_deck(state.gs.get("deck", []))

    original = hand_solver.rank_templates_cheaply
    if not joker_aware:
        def forced_old(*args, **kwargs):
            kwargs["joker_aware"] = False
            return original(*args, **kwargs)

        hand_solver.rank_templates_cheaply = forced_old
    try:
        t0 = time.time()
        choice = solve_hand_for_ante_clear(
            state.hand,
            state.jokers,
            state.hand_levels,
            state.blind,
            state.rng,
            deck,
            state.chips_needed,
            cr.get("hands_left", 0),
            cr.get("discards_left", 0),
            game_state=state.gs,
            blind_chips=state.blind_chips,
            four_fingers=state.four_fingers,
            shortcut=state.shortcut,
            mc_seed=mc_seed,
        )
        seconds = time.time() - t0
    finally:
        hand_solver.rank_templates_cheaply = original
    return {
        "action": choice.action,
        "template": choice.template_name,
        "p_clear": choice.p_clear,
        "seconds": round(seconds, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="stage3_full")
    parser.add_argument("--n-states", type=int, default=30)
    parser.add_argument("--accept-factor", type=float, default=1.33)
    parser.add_argument(
        "--output", type=Path, default=Path("data/discard_ranking_validation.json")
    )
    args = parser.parse_args()

    from generate_hand_demos import stage_presets

    config = stage_presets()[args.stage].config

    rows: list[dict] = []
    attempt = 0
    while len(rows) < args.n_states and attempt < args.n_states * 60:
        seed = f"{SEED_PREFIX}_{attempt:08d}"
        attempt += 1
        state = sample_state(seed, config)
        cr = state.gs["current_round"]
        if cr.get("discards_left", 0) < 2 or state.chips_needed <= 0:
            continue
        if not state.jokers:
            continue  # both arms identical without jokers -- no information

        new_arm = solve_once(state, seed, joker_aware=True)
        old_arm = solve_once(state, seed, joker_aware=False)
        floor_arm = solve_once(state, f"{seed}#b", joker_aware=True)

        rows.append(
            {
                "seed": seed,
                "n_jokers": len(state.jokers),
                "discards_left": cr.get("discards_left", 0),
                "hands_left": cr.get("hands_left", 0),
                "new": new_arm,
                "old": old_arm,
                "reseeded_new_p_clear": floor_arm["p_clear"],
                "delta": new_arm["p_clear"] - old_arm["p_clear"],
                "choice_changed": (new_arm["action"], new_arm["template"])
                != (old_arm["action"], old_arm["template"]),
            }
        )
        r = rows[-1]
        print(
            f"[{len(rows)}/{args.n_states}] jokers={r['n_jokers']} "
            f"d={r['discards_left']} delta={r['delta']:+.4f} "
            f"changed={r['choice_changed']} "
            f"t_new={new_arm['seconds']}s t_old={old_arm['seconds']}s"
        )

    deltas = [r["delta"] for r in rows]
    floor_spread = [abs(r["new"]["p_clear"] - r["reseeded_new_p_clear"]) for r in rows]
    floor = float(np.mean(floor_spread)) if floor_spread else None
    mean_delta = float(np.mean(deltas)) if deltas else None
    accepted = (
        mean_delta is not None
        and floor is not None
        and mean_delta >= -args.accept_factor * floor
    )
    summary = {
        "stage": args.stage,
        "n_states": len(rows),
        "mean_delta_new_minus_old": mean_delta,
        "frac_new_better": float(np.mean([d > 1e-12 for d in deltas])) if deltas else None,
        "frac_equal": float(np.mean([abs(d) <= 1e-12 for d in deltas])) if deltas else None,
        "frac_new_worse": float(np.mean([d < -1e-12 for d in deltas])) if deltas else None,
        "choice_change_rate": float(np.mean([r["choice_changed"] for r in rows]))
        if rows
        else None,
        "mc_reseed_floor": floor,
        "accept_factor": args.accept_factor,
        "mean_seconds_new": float(np.mean([r["new"]["seconds"] for r in rows])),
        "mean_seconds_old": float(np.mean([r["old"]["seconds"] for r in rows])),
        "accepted": bool(accepted),
    }
    report = {"summary": summary, "states": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("ACCEPTED" if accepted else "NOT ACCEPTED -- investigate before generating labels")


if __name__ == "__main__":
    main()
