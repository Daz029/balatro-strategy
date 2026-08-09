"""Dump every joker's CURRENT descriptor next to its engine truth, for hand review.

The shop policy's generalization channel is ``joker_descriptors.DESCRIPTOR_MATRIX``
— 24 hand-derived numbers per center key. ``docs/embedding-analysis.md`` §5
measured that channel doing almost no work (twin/non-twin synergy gap ratio
0.87), and §5.1 found why: it is both thin and partly WRONG.

This script exists so the repair is done by INSPECTION rather than by another
round of pattern-matching on ``config`` keys — the scaling-flag bug is exactly
what heuristic extraction produces. It emits one reviewable record per joker
containing:

* the engine's own truth (raw ``config``, ``effect`` family label, rarity, cost,
  ``blueprint_compat``, unlock/pool gates),
* the ``trigger_match`` 4-class taxonomy — engine-derived, coverage-enforced at
  import, and NOT currently represented in the descriptor at all,
* what the descriptor CURRENTLY says (nonzero fields, named),
* which other jokers share a bit-identical descriptor row,
* auto-flags for the known defect classes,
* an empty ``review`` block to fill in by hand.

AUTO-FLAGS ARE HINTS, NOT VERDICTS. They are themselves heuristics over the
same config data that produced the bug; a joker with no flags can still be
mis-described (Photograph carries FALSE_SCALING but its missing x2 mult is only
visible by knowing what Photograph does). The flags rank the review queue, they
do not replace it.

Usage::

    uv run python scripts/dump_joker_descriptors.py
    uv run python scripts/dump_joker_descriptors.py --output data/my_review.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw.agents.joker_descriptors import DESCRIPTOR_DIM, DESCRIPTOR_MATRIX
from jackdaw.engine.data.prototypes import CENTER_POOLS, JOKERS
from jackdaw.env.observation import center_key_id
from jackdaw.env.trigger_match import (
    _CLASS1_PREDICATES,
    _CLASS2_PREDICATES,
    _CLASS3_SET_LEVEL,
    _CLASS4_NON_CARD,
)

DEFAULT_OUT = "data/joker_descriptors_review.json"

# Mirrors the frozen layout comment in joker_descriptors.py:34-58. Kept here as
# (name, what it currently reads) so a reviewer can see the DERIVATION, not just
# the value -- most defects are in the derivation.
DESCRIPTOR_LAYOUT: list[tuple[str, str]] = [
    ("rarity", "proto.rarity / 4"),
    ("base_cost", "proto.cost / 10"),
    ("is_joker", "constant 1.0 for jokers"),
    ("is_tarot", "always 0 on jokers"),
    ("is_planet", "always 0 on jokers"),
    ("is_spectral", "always 0 on jokers"),
    ("is_voucher", "always 0 on jokers"),
    ("is_booster", "always 0 on jokers"),
    ("flat_mult", "(cfg.mult + extra.mult) / 20"),
    ("flat_chips", "extra.chips / 100"),
    ("x_mult", "(max(cfg.Xmult, extra.Xmult) - 1) / 3, only if > 1"),
    ("suit_conditional_flag", "1 if extra IS A DICT and extra.suit is a suit name"),
    ("suit_ordinal", "H=0 D=1 C=2 S=3, / 3"),
    ("suit_mult", "extra.s_mult / 10"),
    ("hand_type_conditional_flag", "1 if cfg.type or extra.poker_hand"),
    ("hand_type_mult", "cfg.t_mult / 20"),
    ("hand_type_chips", "cfg.t_chips / 100"),
    ("per_trigger_dollars", "extra.dollars / 10"),
    ("probabilistic_flag", "1 if extra.odds"),
    ("trigger_probability", "1 / extra.odds"),
    ("scaling_flag", "1 if extra is a BARE NUMBER, or any of _SCALING_EXTRA_KEYS"),
    ("scaling_rate", "min(that number, 5) / 5"),
    ("hand_discard_size_mod", "(cfg.h_size + extra.h_size + cfg.d_size) / 3"),
    ("blueprint_compat", "proto.blueprint_compat"),
]
assert len(DESCRIPTOR_LAYOUT) == DESCRIPTOR_DIM

# Effect columns: everything except rarity/cost/type-flags/blueprint_compat.
# All-zero here means the descriptor says nothing about what the joker DOES.
EFFECT_COLUMNS = list(range(8, 23))

# embedding-analysis.md §8: conditionally pooled, so never seen in any obs.
POOL_GATED = frozenset(
    {"j_cavendish", "j_glass", "j_lucky_cat", "j_steel_joker", "j_stone", "j_ticket"}
)


def trigger_class(key: str) -> str:
    """The trigger_match 4-class taxonomy label for a joker key.

    Read from the taxonomy's own tables rather than restated here: a second
    copy of that classification is the drift class the module's import-time
    coverage check exists to prevent.
    """
    if key in _CLASS1_PREDICATES:
        return "class1_per_card_static"
    if key in _CLASS2_PREDICATES:
        return "class2_per_card_state_dependent"
    if key in _CLASS3_SET_LEVEL:
        return "class3_set_level"
    if key in _CLASS4_NON_CARD:
        return "class4_non_card"
    return "UNCLASSIFIED"


def auto_flags(key: str, proto: Any, desc: np.ndarray, dupes: list[str]) -> list[str]:
    cfg = proto.config or {}
    extra = cfg.get("extra")
    flags: list[str] = []

    if key in POOL_GATED:
        flags.append("POOL_GATED_NEVER_OBSERVED")
    if isinstance(extra, (int, float)) and desc[20] > 0:
        flags.append("FALSE_SCALING_SUSPECT")
    if dupes:
        flags.append(f"DUPLICATE_DESCRIPTOR_x{len(dupes) + 1}")
    if not np.any(desc[EFFECT_COLUMNS]):
        flags.append("EFFECT_UNCAPTURED")
    if isinstance(extra, dict) and "suit" in extra and desc[11] == 0:
        flags.append("SUIT_PRESENT_BUT_UNCAPTURED")
    return flags


def build_records() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    keys = list(CENTER_POOLS.get("Joker", []))

    by_row: dict[bytes, list[str]] = defaultdict(list)
    for k in keys:
        by_row[DESCRIPTOR_MATRIX[center_key_id(k)].tobytes()].append(k)
    dup_groups = {
        f"group_{i:02d}": sorted(v) for i, v in enumerate(sorted(by_row.values())) if len(v) > 1
    }
    group_of = {k: gid for gid, members in dup_groups.items() for k in members}

    records = []
    for key in sorted(keys, key=lambda k: JOKERS[k].order):
        proto = JOKERS[key]
        desc = DESCRIPTOR_MATRIX[center_key_id(key)]
        dupes = [k for k in by_row[desc.tobytes()] if k != key]
        nonzero = {
            DESCRIPTOR_LAYOUT[i][0]: round(float(v), 4) for i, v in enumerate(desc) if v != 0
        }
        records.append(
            {
                "key": key,
                "name": proto.name,
                "order": proto.order,
                "rarity": proto.rarity,
                "cost": proto.cost,
                "blueprint_compat": proto.blueprint_compat,
                "engine_config": proto.config,
                "engine_effect_label": proto.effect or None,
                "trigger_class": trigger_class(key),
                "unlock_condition": proto.unlock_condition,
                "enhancement_gate": proto.enhancement_gate,
                "descriptor_nonzero": nonzero,
                "descriptor_raw": [round(float(v), 5) for v in desc],
                "duplicate_group": group_of.get(key),
                "duplicate_with": dupes,
                "auto_flags": auto_flags(key, proto, desc, dupes),
                "review": {"effect_summary": "", "should_encode": [], "notes": ""},
            }
        )
    return records, dup_groups


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUT))
    args = p.parse_args()

    records, dup_groups = build_records()
    flagged = defaultdict(int)
    for r in records:
        for f in r["auto_flags"]:
            flagged[f.split("_x")[0]] += 1

    payload = {
        "_meta": {
            "generated_utc": datetime.now(UTC).isoformat(),
            "n_jokers": len(records),
            "descriptor_dim": DESCRIPTOR_DIM,
            "context": (
                "docs/embedding-analysis.md §5.1 — descriptor channel is thin "
                "and partly wrong"
            ),
        },
        "_how_to_review": {
            "effect_summary": "one line: what the joker ACTUALLY does in game",
            "should_encode": (
                "list the mechanical facts a shop agent needs to judge this joker's "
                "synergy, e.g. ['xmult', 'triggers on face cards', 'retrigger']. "
                "Free text — we derive the column set AFTER seeing what the pool needs, "
                "rather than fitting jokers to the columns that already exist."
            ),
            "notes": "anything the current descriptor gets actively wrong",
            "warning": (
                "auto_flags are heuristics over the same config data that produced the "
                "scaling bug. An unflagged joker can still be mis-described — Photograph "
                "is flagged FALSE_SCALING but its uncaptured x2 mult is only visible to "
                "someone who knows the joker."
            ),
        },
        "_descriptor_layout": [
            {"index": i, "name": n, "derived_from": src}
            for i, (n, src) in enumerate(DESCRIPTOR_LAYOUT)
        ],
        "_known_defects": {
            "false_scaling": (
                "76/150 jokers use a bare-numeric `extra`; the extractor reads any bare "
                "number as a growth rate, so 76 of the 87 scaling flags are false. The "
                "idiom is shared by genuine scalers (Obelisk) and flat effects (Credit "
                "Card, Banner, Mime), so the column cannot discriminate."
            ),
            "dead_columns": "cols 3-7 are always 0 on jokers; col 2 is constant 1",
            "duplicates": f"{sum(len(v) for v in dup_groups.values())}/150 jokers share a "
            f"bit-identical row across {len(dup_groups)} groups",
            "not_represented": (
                "the trigger_match 4-class taxonomy is engine-derived and "
                "coverage-enforced but absent from the descriptor"
            ),
        },
        "_flag_counts": dict(sorted(flagged.items(), key=lambda kv: -kv[1])),
        "_duplicate_groups": dup_groups,
        "jokers": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"wrote {args.output}  ({len(records)} jokers)")
    print(f"duplicate groups: {len(dup_groups)} covering "
          f"{sum(len(v) for v in dup_groups.values())} jokers")
    print("\nflag counts:")
    for f, n in sorted(flagged.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
