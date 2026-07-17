"""Does the B5 prescreen's candidate box capture the brute-force-optimal play
at hand size 8 -- the size where the prescreen is currently switched OFF?

WHY THIS EXISTS. Profiling a representative stage2 label (205s) showed the cost
is almost entirely `best_immediate_play` brute-forcing every 1-5 card subset at
every recursion node:

    488 nodes x 218 subsets x ~1.65ms score_hand ~= 178s  (74% of runtime)

218 = C(8,1..5). `PRESCREEN_HAND_LIMIT = 8` gates the prescreen to n > 8, but
the harvest's max hand size is exactly 8 and stages 1-4 deal 8 outside B1's 10%
tail -- so the prescreen machinery, already built and validated at n=9-12
(regret 0.0, best-in-cut 0.958), never fires where the labels actually live.
Lowering the limit would cut ~218 candidates to ~k, but it CHANGES LABEL
SEMANTICS for every stage, so it must be measured, not assumed.

METRIC: CAPTURE RATE, not just regret (user call). The B7 session established
that the solver's recursive `p_clear` SATURATES -- `_fill_hand_to_size`'s
optimistic refill pins branches at 1.0, so regret measured through the
recursion reads ~0 and hides real differences. This harness therefore measures
at the SINGLE NODE, on `best_immediate_play`'s own objective (`result.total`),
which does not saturate:

  * capture  -- is the brute-force argmax subset INSIDE the prescreen box?
                (pure set membership; no scoring at all)
  * regret   -- total(brute-force best) - total(best-in-box), in score units.
                Capture implies regret 0; regret says how much a miss costs,
                since a near-tied miss is harmless (pitfall #12).

TWO ARMS, because each covers the other's bias:

  * `shard` -- FREE ORACLE. For a PlayHand label, `solve_hand_turn` takes its
    `hold` straight from `best_immediate_play`'s argmax over all 218, so the
    STORED LABEL *is* the brute-force answer, already paid for by the regen
    run. Costs k+1 evaluations instead of 218. But it is BIASED two ways:
    completed shards are the workers that finished FIRST (i.e. cheap ranges),
    and only play-labeled roots qualify -- states where playing beat every
    discard, plausibly the easy, dominant-line cases. Restricted to n == 8:
    n > 8 labels are ALREADY prescreened and so are not a brute-force oracle.

  * `brute` -- UNBIASED. Samples seeds uniformly across the stage's whole
    index range (including ones the run never reached) and computes the
    argmax itself. This is affordable for a reason worth stating: a node's
    brute force is 218 evals ~= 0.36s REGARDLESS of discards_left. What makes
    a "slow" example slow is the RECURSION (hundreds of nodes), not any one
    node -- and capture is a per-node question. So the expensive states cost
    the same to check here as the cheap ones, and survivorship bias dissolves.

Both arms regenerate states from their seeds (generation is deterministic in
`f"{stage}_{i:08d}"`) using the config recorded in the run's manifest.json --
NOT the preset, which lacks the harvested `dollar_marginals` the run passed.
The shard arm additionally VERIFIES reproduction by re-encoding the state and
comparing against the shard's stored `hand_cards`, so a config mismatch is
caught rather than silently measuring different states.

K2 GENERALIZATIONS (the harness is n-parameterised, per the design doc's §5;
the n==8 filter is a CLI-level choice):

  * Brute-arm truth is an EXPLICIT C(n,1..5) enumeration, never a
    `best_immediate_play` call -- above PRESCREEN_HAND_LIMIT that function
    prescreens, so "truth" would silently become the box under test and the
    9-12 tail would always read regret 0.
  * ``--hand-sizes`` filters the brute arm (the shard arm stays pinned to
    n == 8: n > 8 labels are already prescreened and are not an oracle).
  * ``--force-tail`` deals via B1's flat hand-size tail (prob 1.0, delta
    1-4) for the 9-12 tail gate arm. Skips the shard arm -- the modified
    config no longer regenerates the shards' states.
  * ``--stage-preset`` sources the config from
    `generate_hand_demos.stage_presets()` instead of a shard manifest (the
    stage3/4 copy-joker sample -- no shard run exists for those yet). Skips
    the shard arm.
  * ``--require-jokers`` keeps only brute-arm states owning at least one of
    the given center keys (Blueprint/Brainstorm coverage: uniform stage3/4
    sampling hits a copy joker too rarely to measure).

Gate runs (K3)::

    # n=8, stage2 density, both arms; --n-shard 0 = the full stage2
    # brute-force corpus folded in as the free oracle
    uv run python scripts/validate_prescreen_n8.py \
        --shard-dir data/stage_2_h1_shards --n-shard 0 --n-brute 300

    # 9-12 tail, stage2 density
    uv run python scripts/validate_prescreen_n8.py --force-tail \
        --hand-sizes 9,10,11,12 --n-brute 200 --out data/prescreen_tail.json

    # stage3/4 copy-joker sample
    uv run python scripts/validate_prescreen_n8.py --stage-preset stage3_full \
        --total-examples 20000 --require-jokers j_blueprint,j_brainstorm \
        --n-brute 100 --out data/prescreen_copy_jokers.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hand_solver import (  # noqa: E402
    PRESCREEN_TOP_K,
    evaluate_value,
    prescreen_play_candidates,
)

from jackdaw.engine.hand_eval import get_hand_eval_flags  # noqa: E402
from jackdaw.env.hand_play_adapter import (  # noqa: E402
    HandPlayAdapter,
    HandPlayConfig,
    JokerCountBand,
)
from jackdaw.env.hand_play_gym import encode_hand_state_v2  # noqa: E402

BACK_KEY = "b_red"
STAKE = 1
ACTION_PLAY = 0  # ActionType.PlayHand

_TUPLE_FIELDS = (
    "ante_range",
    "hands_range",
    "discards_range",
    "dollars_range",
    "hand_size_delta_range",
    "boss_history_hands_played_range",
    "joker_count_range",
)


def config_from_manifest(manifest: dict[str, Any]) -> HandPlayConfig:
    """Rebuild the EXACT HandPlayConfig the run used.

    Not the stage preset: the run passed `--dollar-marginals`, which the
    preset does not carry, and any field mismatch silently regenerates a
    DIFFERENT state for the same seed.
    """
    cfg = dict(manifest["hand_play_config"])
    for field in _TUPLE_FIELDS:
        if cfg.get(field) is not None:
            cfg[field] = tuple(cfg[field])
    for field in ("joker_pool", "blind_stages"):
        if cfg.get(field) is not None:
            cfg[field] = tuple(cfg[field])
    if cfg.get("joker_count_bands"):
        cfg["joker_count_bands"] = tuple(
            JokerCountBand(
                count=b["count"],
                weight=b["weight"],
                ante_range=tuple(b["ante_range"]) if b.get("ante_range") else None,
            )
            for b in cfg["joker_count_bands"]
        )
    if cfg.get("dollar_marginals"):
        cfg["dollar_marginals"] = {
            int(ante): {int(d): int(n) for d, n in hist.items()}
            for ante, hist in cfg["dollar_marginals"].items()
        }
    return HandPlayConfig(**cfg)


def build_state(seed: str, config: HandPlayConfig) -> dict[str, Any]:
    adapter = HandPlayAdapter(config)
    adapter.reset(BACK_KEY, STAKE, seed)
    return adapter.raw_state


def _solver_args(gs: dict[str, Any]) -> dict[str, Any]:
    blind = gs["blind"]
    return {
        "hand": gs["hand"],
        "jokers": gs["jokers"],
        "hand_levels": gs["hand_levels"],
        "blind": blind,
        "rng": gs["rng"],
        "game_state": gs,
        "blind_chips": getattr(blind, "chips", 0) if blind else 0,
    }


def _index_set(subset: list, hand: list) -> frozenset[int]:
    by_id = {id(c): i for i, c in enumerate(hand)}
    return frozenset(by_id[id(c)] for c in subset)


_SUIT_SHORT = {"Diamonds": "D", "Hearts": "H", "Spades": "S", "Clubs": "C"}


def describe_card(card: Any) -> str:
    """Compact render carrying every attribute a kicker hypothesis reads:
    rank/suit (Lusty/Greedy/Even Steven), enhancement (Steel/Gold/Glass),
    edition (Polychrome -- K1 correction 1), seal. A Stone card has no
    `base`, hence the guard."""
    base = getattr(card, "base", None)
    if base is None:
        core = "Stone"
    else:
        rank = str(base.rank.value)
        core = (rank if len(rank) <= 2 else rank[0]) + _SUIT_SHORT.get(
            str(base.suit.value), "?"
        )
    tags = []
    effect = (card.ability or {}).get("effect") if hasattr(card, "ability") else None
    if effect and effect not in ("Base",):
        tags.append(str(effect))
    if getattr(card, "edition", None):
        tags.append(str(card.edition))
    if getattr(card, "seal", None):
        tags.append(f"{card.seal} seal")
    if getattr(card, "debuff", False):
        tags.append("DEBUFFED")
    return core + (f"[{'+'.join(tags)}]" if tags else "")


def describe_miss(
    gs: dict[str, Any],
    seed: str,
    rec: dict[str, Any],
    k: int,
) -> dict[str, Any]:
    """Everything needed to CLASSIFY a surviving prescreen miss by hypothesis
    (CLAUDE.md's K3 fail-path "rescan/classify surviving misses"): the board,
    the line truth picked, the line the box settled for, and the delta.

    Jokers are rendered RESOLVED-agnostic here (raw center keys): the copy
    chain matters for gating, but for reading a miss you want to see that a
    Blueprint is present at all.
    """
    hand = gs["hand"]
    flags = get_hand_eval_flags(gs["jokers"])
    by_k = rec["by_k"][k]
    truth_idx = sorted(rec["truth_indices"])
    box_idx = sorted(by_k.get("box_best_indices") or [])
    return {
        "seed": seed,
        "k": k,
        "regret": round(by_k["regret"], 2),
        "rel_regret": round(by_k["rel_regret"], 4),
        "truth_total": round(rec["truth_total"], 2),
        "hand": [describe_card(c) for c in hand],
        "jokers": [j.center_key for j in gs["jokers"]],
        "joker_editions": [str(getattr(j, "edition", None)) for j in gs["jokers"]],
        "four_fingers": bool(flags.get("four_fingers")),
        "shortcut": bool(flags.get("shortcut")),
        "splash": any(j.center_key == "j_splash" for j in gs["jokers"]),
        "truth_indices": truth_idx,
        "truth_cards": [describe_card(hand[i]) for i in truth_idx],
        "box_best_indices": box_idx,
        "box_best_cards": [describe_card(hand[i]) for i in box_idx],
        "discards_left": rec.get("discards_left"),
        "hands_left": int(gs["current_round"].get("hands_left", 0)),
        "chips_needed": max(
            0, (getattr(gs["blind"], "chips", 0) or 0) - gs.get("chips", 0)
        ),
    }


def _value_of(subset: list, args: dict[str, Any]) -> float:
    hand = args["hand"]
    ids = {id(c) for c in subset}
    held = [c for c in hand if id(c) not in ids]
    return float(
        evaluate_value(
            list(subset),
            held,
            args["jokers"],
            args["hand_levels"],
            args["blind"],
            args["rng"],
            args["game_state"],
            args["blind_chips"],
        ).total
    )


def _brute_argmax(args: dict[str, Any]) -> tuple[list, float]:
    """Exact argmax over every 1-5-card subset: sizes ascending, enumeration
    order, first-wins on ties -- byte-identical to `best_immediate_play`'s
    n <= 8 branch. Explicit rather than a `best_immediate_play` call because
    above PRESCREEN_HAND_LIMIT that function PRESCREENS, so "truth" would
    silently become the box under test and the 9-12 tail arm would always
    read regret 0 (K2)."""
    hand = args["hand"]
    best_subset: list | None = None
    best_total: float | None = None
    for size in range(1, min(5, len(hand)) + 1):
        for combo in itertools.combinations(hand, size):
            total = _value_of(list(combo), args)
            if best_total is None or total > best_total:
                best_subset, best_total = list(combo), total
    assert best_subset is not None and best_total is not None
    return best_subset, best_total


def _box_at_k(args: dict[str, Any], k: int) -> list[list]:
    flags = get_hand_eval_flags(args["jokers"])
    return prescreen_play_candidates(
        args["hand"],
        args["jokers"],
        args["hand_levels"],
        args["blind"],
        args["rng"],
        four_fingers=flags["four_fingers"],
        shortcut=flags["shortcut"],
        top_k=k,
        game_state=args["game_state"],
        blind_chips=args["blind_chips"],
    )


def measure_state(
    gs: dict[str, Any],
    ks: list[int],
    truth_subset: list | None,
) -> dict[str, Any]:
    """Capture + regret at every k for one state.

    `truth_subset` = the shard's label (free oracle). None => compute the
    brute-force argmax here (the unbiased arm).
    """
    args = _solver_args(gs)
    hand = args["hand"]

    if truth_subset is None:
        truth_subset, truth_total = _brute_argmax(args)
    else:
        truth_total = _value_of(truth_subset, args)

    truth_ids = _index_set(truth_subset, hand)
    n = len(hand)
    n_subsets = sum(math.comb(n, size) for size in range(1, min(5, n) + 1))
    out: dict[str, Any] = {
        "truth_total": truth_total,
        "truth_indices": sorted(truth_ids),
        "hand_size": n,
        "n_subsets": n_subsets,  # the brute baseline: 218 at n=8, C(n,1..5) above
        "by_k": {},
    }
    for k in ks:
        box = _box_at_k(args, k)
        box_id_sets = [_index_set(c, hand) for c in box]
        # argmax (not just max): a miss is only classifiable if you can see
        # WHICH line the box settled for versus what truth played.
        box_best_cards: list = []
        best_in_box = 0.0
        for cand in box:
            v = _value_of(cand, args)
            if not box_best_cards or v > best_in_box:
                box_best_cards, best_in_box = list(cand), v
        regret = max(0.0, truth_total - best_in_box)
        out["by_k"][k] = {
            # STRICT: the box holds the argmax SET itself. Systematically
            # pessimistic -- the argmax is not unique (a played hand's
            # non-scoring kickers are interchangeable, so many subsets tie at
            # the same total), and the box gets no credit for an
            # equally-optimal twin.
            "captured": truth_ids in box_id_sets,
            # BY VALUE: the box holds SOME subset achieving the max total.
            # This is the one that matters -- an equal-value alternative is a
            # perfect substitute, and the solver only ever reads `.total`
            # (pitfall #12: regret, not disagreement).
            "captured_by_value": regret <= 1e-6,
            "regret": regret,
            "rel_regret": regret / truth_total if truth_total > 0 else 0.0,
            "box_size": len(box),
            "box_best_indices": sorted(_index_set(box_best_cards, hand))
            if box_best_cards
            else [],
        }
    return out


def load_shard_rows(shard_dir: Path) -> list[dict[str, Any]]:
    """Eligible shard examples: PlayHand labels on n == 8 hands ONLY.

    n > 8 labels are already prescreened (the n > PRESCREEN_HAND_LIMIT path),
    so they are not a brute-force oracle. Discard labels don't identify the
    play argmax at all.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(shard_dir.glob("worker_*_shard_*.npz")):
        data = np.load(path)
        for i in range(len(data["seed"])):
            hand_size = int(data["hand_mask"][i].sum())
            if hand_size != 8 or int(data["action_type"][i]) != ACTION_PLAY:
                continue
            picks = [int(x) for x in data["card_indices"][i] if x >= 0]
            rows.append(
                {
                    "seed": str(data["seed"][i]),
                    "picks": picks,
                    "hand_cards": data["hand_cards"][i][:hand_size],
                }
            )
    return rows


def _summarize(records: list[dict[str, Any]], ks: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_states": len(records), "by_k": {}}
    for k in ks:
        caps = [r["by_k"][k]["captured"] for r in records]
        capv = [r["by_k"][k]["captured_by_value"] for r in records]
        regs = [r["by_k"][k]["regret"] for r in records]
        rels = [r["by_k"][k]["rel_regret"] for r in records]
        boxes = [r["by_k"][k]["box_size"] for r in records]
        subsets = [r["n_subsets"] for r in records]
        n = max(1, len(records))
        out["by_k"][str(k)] = {
            "capture_rate_by_value": round(sum(capv) / n, 4),
            "n_missed_by_value": int(len(capv) - sum(capv)),
            "capture_rate_strict": round(sum(caps) / n, 4),
            "n_missed_strict": int(len(caps) - sum(caps)),
            "mean_regret": round(sum(regs) / n, 2),
            "max_regret": round(max(regs, default=0.0), 2),
            "mean_rel_regret": round(sum(rels) / n, 5),
            "max_rel_regret": round(max(rels, default=0.0), 4),
            "mean_box_size": round(sum(boxes) / n, 2),
            # vs the ACTUAL brute count per state (218 at n=8, C(n,1..5) above)
            "speedup_vs_brute": round((sum(subsets) / n) / max(1e-9, sum(boxes) / n), 1),
        }
    return out


def _stratify(records: list[dict[str, Any]], ks: list[int], key: str) -> dict[str, Any]:
    groups: dict[Any, list] = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    return {
        str(g): {
            "n": len(rs),
            **{
                f"capture_by_value_k{k}": round(
                    sum(r["by_k"][k]["captured_by_value"] for r in rs) / len(rs), 3
                )
                for k in ks
            },
            **{
                f"mean_regret_k{k}": round(
                    sum(r["by_k"][k]["regret"] for r in rs) / len(rs), 2
                )
                for k in ks
            },
        }
        for g, rs in sorted(groups.items())
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-dir", type=Path, default=Path("data/stage_2_h1_shards"))
    parser.add_argument("--ks", default="3,5,8,12")
    parser.add_argument("--n-shard", type=int, default=250, help="0 = all eligible")
    parser.add_argument("--n-brute", type=int, default=300)
    parser.add_argument("--total-examples", type=int, default=4000)
    parser.add_argument(
        "--stage-name",
        default=None,
        help="seed prefix; defaults to the preset name or stage2_curated",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/prescreen_n8.json"))
    parser.add_argument(
        "--hand-sizes",
        default="8",
        help="brute-arm hand-size filter, comma list (shard arm stays n==8: "
        "n>8 labels are already prescreened, not an oracle)",
    )
    parser.add_argument(
        "--force-tail",
        action="store_true",
        help="deal via B1's flat hand-size tail (prob 1.0, delta 1-4) for the "
        "9-12 tail arm; skips the shard arm (config no longer matches shards)",
    )
    parser.add_argument(
        "--stage-preset",
        default=None,
        help="config from generate_hand_demos.stage_presets() instead of a "
        "shard manifest (stage3/4 copy-joker sample); skips the shard arm",
    )
    parser.add_argument(
        "--require-jokers",
        default=None,
        help="comma center keys; brute arm keeps only states owning >=1 of them",
    )
    parser.add_argument(
        "--dump-miss-k",
        type=int,
        default=None,
        help="dump the full board of every state MISSED at this k (default: the "
        "production PRESCREEN_TOP_K) into report['misses'] -- the input to "
        "CLAUDE.md's K3 'rescan/classify surviving misses' path",
    )
    args = parser.parse_args()

    ks = [int(k) for k in args.ks.split(",")]
    dump_k = args.dump_miss_k if args.dump_miss_k is not None else PRESCREEN_TOP_K
    if dump_k not in ks:
        # Silently dumping nothing because the k wasn't measured is the
        # quiet-failure class this harness exists to avoid.
        raise SystemExit(f"--dump-miss-k {dump_k} not in --ks {ks}")
    misses: list[dict[str, Any]] = []
    hand_sizes = {int(s) for s in args.hand_sizes.split(",")}
    require_jokers = (
        {k.strip() for k in args.require_jokers.split(",")} if args.require_jokers else None
    )
    stage_name = args.stage_name or args.stage_preset or "stage2_curated"

    if args.stage_preset:
        from generate_hand_demos import stage_presets

        config = stage_presets()[args.stage_preset].config
    else:
        manifest = json.loads((args.shard_dir / "manifest.json").read_text(encoding="utf-8"))
        config = config_from_manifest(manifest)
    if args.force_tail:
        config = replace(config, hand_size_tail_prob=1.0, hand_size_delta_range=(1, 4))
    rng = random.Random(args.seed)

    # ---- arm 1: shard (free oracle, biased) ----
    # Only in default manifest mode: --stage-preset has no shard run to read,
    # and --force-tail's modified config would fail state reproduction anyway.
    shard_records: list[dict[str, Any]] = []
    n_mismatch = 0
    run_shard_arm = not (args.stage_preset or args.force_tail)
    if run_shard_arm:
        rows = load_shard_rows(args.shard_dir)
        if args.n_shard and len(rows) > args.n_shard:
            rows = rng.sample(rows, args.n_shard)
        t0 = time.perf_counter()
        for i, row in enumerate(rows):
            gs = build_state(row["seed"], config)
            hand = gs["hand"]
            if len(hand) != 8:
                n_mismatch += 1
                continue
            # Integrity: the regenerated state must BE the shard's state.
            enc = encode_hand_state_v2(gs)["hand_cards"]
            if not np.allclose(enc, row["hand_cards"], atol=1e-5):
                n_mismatch += 1
                continue
            truth = [hand[j] for j in row["picks"]]
            rec = measure_state(gs, ks, truth)
            rec["discards_left"] = int(gs["current_round"].get("discards_left", 0))
            rec["n_jokers"] = len(gs["jokers"])
            if not rec["by_k"][dump_k]["captured_by_value"]:
                misses.append({"arm": "shard", **describe_miss(gs, row["seed"], rec, dump_k)})
            shard_records.append(rec)
            if (i + 1) % 25 == 0:
                print(
                    f"[shard] {i + 1}/{len(rows)}  ({time.perf_counter() - t0:.0f}s)",
                    flush=True,
                )

    # ---- arm 2: brute force (unbiased; a node is ~0.36s at ANY depth) ----
    # Iterate a seeded shuffle of the whole index range and keep the first
    # n_brute states passing the filters -- with --hand-sizes or
    # --require-jokers active, a fixed pre-sample would silently under-fill.
    indices = list(range(args.total_examples))
    rng.shuffle(indices)
    brute_records: list[dict[str, Any]] = []
    n_target = min(args.n_brute, args.total_examples)
    n_checked = 0
    t1 = time.perf_counter()
    for idx in indices:
        if len(brute_records) >= n_target:
            break
        seed = f"{stage_name}_{idx:08d}"
        gs = build_state(seed, config)
        n_checked += 1
        if len(gs["hand"]) not in hand_sizes:
            continue
        if require_jokers is not None and not (
            {j.center_key for j in gs["jokers"]} & require_jokers
        ):
            continue
        rec = measure_state(gs, ks, None)
        rec["discards_left"] = int(gs["current_round"].get("discards_left", 0))
        rec["n_jokers"] = len(gs["jokers"])
        if not rec["by_k"][dump_k]["captured_by_value"]:
            misses.append({"arm": "brute", **describe_miss(gs, seed, rec, dump_k)})
        brute_records.append(rec)
        if len(brute_records) % 25 == 0:
            print(
                f"[brute] {len(brute_records)}/{n_target} "
                f"(checked {n_checked}, {time.perf_counter() - t1:.0f}s)",
                flush=True,
            )

    report = {
        "params": {
            "ks": ks,
            "seed": args.seed,
            "shard_dir": args.shard_dir.as_posix() if run_shard_arm else None,
            "stage_name": stage_name,
            "stage_preset": args.stage_preset,
            "hand_sizes": sorted(hand_sizes),
            "force_tail": args.force_tail,
            "require_jokers": sorted(require_jokers) if require_jokers else None,
            "n_brute_checked": n_checked,
            "prescreen_top_k_default": PRESCREEN_TOP_K,
            "dump_miss_k": dump_k,
        },
        "misses": sorted(misses, key=lambda m: -m["rel_regret"]),
        "shard_arm": {
            "note": "free oracle (label == brute-force argmax); BIASED: finished-first "
            "workers, play-labeled roots only, n==8 only"
            if run_shard_arm
            else "skipped (--stage-preset / --force-tail)",
            "n_state_mismatch_skipped": n_mismatch,
            **_summarize(shard_records, ks),
            "by_discards_left": _stratify(shard_records, ks, "discards_left"),
            "by_n_jokers": _stratify(shard_records, ks, "n_jokers"),
        },
        "brute_arm": {
            "note": "unbiased: seeds sampled uniformly over the stage range; argmax "
            "enumerated here explicitly (C(n,1..5) evals, ~0.36s at n=8, "
            "independent of discards_left)",
            **_summarize(brute_records, ks),
            "by_discards_left": _stratify(brute_records, ks, "discards_left"),
            "by_n_jokers": _stratify(brute_records, ks, "n_jokers"),
            "by_hand_size": _stratify(brute_records, ks, "hand_size"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"dumped {len(misses)} miss boards at k={dump_k} -> {args.out}")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("params", "misses")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
