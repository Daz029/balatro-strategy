"""Dump every joker's CURRENT descriptor next to its engine truth, for hand review.

The shop policy's generalization channel is ``joker_descriptors.DESCRIPTOR_MATRIX``
— 24 hand-derived numbers per center key. ``docs/embedding-analysis.md`` §5
measured that channel doing almost no work (twin/non-twin synergy gap ratio
0.87), and §5.1 found why: it is both thin and partly WRONG.

This script exists so the repair is done by INSPECTION rather than by another
round of pattern-matching on ``config`` keys — that is exactly what produced
the scaling-flag bug. Emits a CSV review sheet (one row per joker) plus an
optional structured JSON.

WHAT IS DERIVED vs WHAT IS PROPOSED — the columns are deliberately separated:

* ``E_evidence_*`` columns are DERIVED from the joker's registered handler:
  whether it writes back to its own ``card.ability`` (the signature of a joker
  that changes over time), which fields it writes, whether it can return
  ``JokerResult(remove=True)``, and whether it ever assigns a bare constant
  (a reset). These are facts about what the code does.
* ``E_proposed`` is a READING over that evidence — the growth category a human
  inferred. It is NOT derived and is exactly what the review should check.

That split is the whole point. The existing `scaling_flag` is what happens when
a reading gets recorded as if it were a derivation: it reads `config` and
concludes "grows over time", which is wrong for 60 of the 87 jokers it fires on
(cross-checked against handler behaviour — see the module's ``--stats`` output).

The handler derivation is not exhaustive: a joker storing state in ``gs``
rather than ``card.ability`` would be invisible to it, and the three ``on_sell``
jokers (Invisible Joker, Luchador, Diet Cola) act outside the scoring registry
entirely. Those are review items, not derived values.

Usage::

    uv run python scripts/dump_joker_descriptors.py
    uv run python scripts/dump_joker_descriptors.py --json data/review.json
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

import jackdaw.engine.jokers as joker_handlers
from jackdaw.agents.joker_descriptors import DESCRIPTOR_DIM, DESCRIPTOR_MATRIX
from jackdaw.engine.data.prototypes import CENTER_POOLS, JOKERS
from jackdaw.env.observation import center_key_id
from jackdaw.env.trigger_match import (
    _CLASS1_PREDICATES,
    _CLASS2_PREDICATES,
    _CLASS3_SET_LEVEL,
    _CLASS4_NON_CARD,
)

DEFAULT_OUT = "data/joker_descriptors_review.csv"

# Mirrors the frozen layout comment in joker_descriptors.py:34-58, as
# (name, what it currently reads) — most defects are in the DERIVATION, not
# the value, so a reviewer needs to see both.
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

EFFECT_COLUMNS = list(range(8, 23))
SCALING_FLAG_COL = 20

POOL_GATED = frozenset(
    {"j_cavendish", "j_glass", "j_lucky_cat", "j_steel_joker", "j_stone", "j_ticket"}
)

# A joker that changes over time must write back to its OWN ability dict.
# `card` is the joker in every handler signature; other_card / target are the
# playing cards it acts on, so Hiker (which writes perma_bonus to SCORED cards)
# must not match — that is a block-C "modifies_cards" fact, not growth.
_SELF_WRITE = re.compile(r"(?<!other_)(?<!target\.)\bcard\.ability\[[\"']([A-Za-z_]+)[\"']\]\s*=")
_RESET_WRITE = re.compile(r"\bcard\.ability\[[\"'][A-Za-z_]+[\"']\]\s*=\s*[01](\.0)?\s*$")

# A READING over the derived evidence, recorded so review is verification
# rather than authoring. NOT derived — see the module docstring. Anything
# absent is proposed `static`.
PROPOSED_GROWTH: dict[str, str] = {
    **dict.fromkeys(
        [
            "j_caino", "j_castle", "j_ceremonial", "j_constellation", "j_flash",
            "j_glass", "j_hologram", "j_lucky_cat", "j_madness", "j_red_card",
            "j_rocket", "j_runner", "j_trousers", "j_square", "j_vampire", "j_wee",
        ],
        "grows_permanently",
    ),
    **dict.fromkeys(
        ["j_campfire", "j_hit_the_road", "j_obelisk", "j_ride_the_bus"], "grows_with_reset"
    ),
    "j_green_joker": "grows_bidirectionally",
    **dict.fromkeys(
        ["j_ice_cream", "j_popcorn", "j_ramen", "j_turtle_bean"], "decays_then_expires"
    ),
    "j_selzer": "countdown_then_expires",
    **dict.fromkeys(["j_invisible", "j_yorick"], "countdown_then_payoff"),
    "j_egg": "sell_value_growth",
}

# Destruction cause, for the 10 handlers that can return remove=True. Split
# four ways because "self destructs" conflates unrelated behaviours — the
# destruction is the PRICE, not the effect.
PROPOSED_DESTRUCTION: dict[str, str] = {
    **dict.fromkeys(
        ["j_ice_cream", "j_popcorn", "j_ramen", "j_selzer", "j_turtle_bean"],
        "expires_when_depleted",
    ),
    **dict.fromkeys(["j_gros_michel", "j_cavendish"], "random_destruction"),
    **dict.fromkeys(["j_sixth_sense", "j_trading"], "consumed_on_use"),
    "j_mr_bones": "destroyed_on_save",
}

# Act on the SELL path, outside the scoring registry, so the handler
# derivation structurally cannot see them. Review items, not derived values.
KNOWN_ON_SELL = ("j_invisible", "j_luchador", "j_diet_cola")


def _check_proposals() -> None:
    """Import-time guard on the hand-written tables above.

    Modelled on ``trigger_match._check_taxonomy``: a mistyped center key would
    silently fall through to ``static``/``""`` and be indistinguishable from a
    deliberate classification. That is precisely the failure mode this whole
    review exists to remove, so it hard-fails rather than defaults. Caught
    ``j_glass_joker`` (really ``j_glass``) and ``j_ride_bus`` (really
    ``j_ride_the_bus``) on first run.

    The second check pins the PROPOSAL against the DERIVATION: every joker
    proposed non-static must be one whose handler actually writes back to its
    own ability, and vice versa. A divergence means either a typo or a reading
    that the evidence does not support — both worth stopping for.
    """
    pool = set(CENTER_POOLS.get("Joker", []))
    unknown = sorted((set(PROPOSED_GROWTH) | set(PROPOSED_DESTRUCTION) | set(KNOWN_ON_SELL)) - pool)
    if unknown:
        raise RuntimeError(
            f"proposal tables reference {len(unknown)} non-joker center key(s): {unknown} — "
            "a typo here silently reads as 'static', so it is fatal by design"
        )

    derived = {k for k in pool if handler_evidence(k)["self_mutates"] == "yes"}
    proposed = set(PROPOSED_GROWTH)
    if derived != proposed:
        raise RuntimeError(
            "E_proposed disagrees with the handler derivation — "
            f"mutates but proposed static: {sorted(derived - proposed)}; "
            f"proposed non-static but never mutates: {sorted(proposed - derived)}"
        )

    remove_capable = {k for k in pool if handler_evidence(k)["can_remove"] == "yes"}
    if remove_capable != set(PROPOSED_DESTRUCTION):
        raise RuntimeError(
            "E_destruction_proposed disagrees with the handler derivation — "
            f"can remove but unclassified: {sorted(remove_capable - set(PROPOSED_DESTRUCTION))}; "
            f"classified but cannot remove: {sorted(set(PROPOSED_DESTRUCTION) - remove_capable)}"
        )


def trigger_class(key: str) -> str:
    """The trigger_match 4-class label, read from the taxonomy's own tables
    rather than restated — a second copy is the drift its import-time
    coverage check exists to prevent."""
    for members, label in (
        (_CLASS1_PREDICATES, "C1_per_card_static"),
        (_CLASS2_PREDICATES, "C2_per_card_state_dep"),
        (_CLASS3_SET_LEVEL, "C3_set_level"),
        (_CLASS4_NON_CARD, "C4_non_card"),
    ):
        if key in members:
            return label
    return "UNCLASSIFIED"


def handler_evidence(key: str) -> dict[str, Any]:
    """Derived facts about what the joker's registered handler DOES."""
    handler = joker_handlers._REGISTRY.get(key)
    if handler is None:
        return {"self_mutates": "", "fields": "", "can_remove": "", "has_reset": ""}
    src = inspect.getsource(handler)
    fields = sorted(set(_SELF_WRITE.findall(src)))
    has_reset = any(_RESET_WRITE.search(line) for line in src.splitlines())
    return {
        "self_mutates": "yes" if fields else "no",
        "fields": ",".join(fields),
        "can_remove": "yes" if "remove=True" in src else "no",
        "has_reset": "yes" if has_reset else "no",
    }


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
        dupes = [JOKERS[k].name for k in by_row[desc.tobytes()] if k != key]
        nonzero = {
            DESCRIPTOR_LAYOUT[i][0]: round(float(v), 4) for i, v in enumerate(desc) if v != 0
        }
        ev = handler_evidence(key)

        flags = []
        if key in POOL_GATED:
            flags.append("POOL_GATED_NEVER_OBSERVED")
        if isinstance((proto.config or {}).get("extra"), (int, float)) and desc[SCALING_FLAG_COL]:
            flags.append("FALSE_SCALING_SUSPECT")
        if dupes:
            flags.append(f"DUPLICATE_x{len(dupes) + 1}")
        if not np.any(desc[EFFECT_COLUMNS]):
            flags.append("EFFECT_UNCAPTURED")
        if key in KNOWN_ON_SELL:
            flags.append("ON_SELL_OUTSIDE_HANDLER")

        records.append(
            {
                "key": key,
                "name": proto.name,
                "rarity": proto.rarity,
                "cost": proto.cost,
                "trigger_class": trigger_class(key),
                "blueprint_compat": int(proto.blueprint_compat),
                "engine_config": json.dumps(proto.config, sort_keys=True) if proto.config else "",
                "engine_effect_label": proto.effect or "",
                "current_descriptor": "; ".join(f"{k}={v}" for k, v in nonzero.items()),
                "current_says_scaling": "yes" if desc[SCALING_FLAG_COL] else "no",
                "dup_group": group_of.get(key, ""),
                "dup_with": "; ".join(dupes),
                "auto_flags": "; ".join(flags),
                "E_evidence_self_mutates": ev["self_mutates"],
                "E_evidence_fields": ev["fields"],
                "E_evidence_can_remove": ev["can_remove"],
                "E_evidence_has_reset": ev["has_reset"],
                "E_proposed": PROPOSED_GROWTH.get(key, "static"),
                "E_destruction_proposed": PROPOSED_DESTRUCTION.get(key, ""),
                # --- blank review columns below ---
                "E_confirm": "",
                "B_when": "",
                "C_what": "",
                "D_gated_by": "",
                "F_depends_on": "",
                "notes": "",
            }
        )
    return records, dup_groups


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--json", type=Path, default=None, help="also write structured JSON")
    args = p.parse_args()

    _check_proposals()
    records, dup_groups = build_records()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "_meta": {
                        "generated_utc": datetime.now(UTC).isoformat(),
                        "n_jokers": len(records),
                        "context": "docs/embedding-analysis.md §5.1 / §5.4",
                    },
                    "_descriptor_layout": [
                        {"index": i, "name": n, "derived_from": s}
                        for i, (n, s) in enumerate(DESCRIPTOR_LAYOUT)
                    ],
                    "_duplicate_groups": dup_groups,
                    "jokers": records,
                },
                indent=2,
            )
        )

    # Cross-check: the config-derived flag against the handler-derived truth.
    changes = {r["key"] for r in records if r["E_evidence_self_mutates"] == "yes"}
    flagged = {r["key"] for r in records if r["current_says_scaling"] == "yes"}
    print(f"wrote {args.output}  ({len(records)} jokers, {len(records[0])} columns)")
    if args.json:
        print(f"wrote {args.json}")
    print(f"\nduplicate groups: {len(dup_groups)} covering "
          f"{sum(len(v) for v in dup_groups.values())} jokers")
    print("\nscaling_flag vs handler behaviour:")
    print(f"{'':<24}{'handler: changes':>18}{'handler: static':>18}")
    print(f"{'descriptor: scaling':<24}{len(flagged & changes):>18}{len(flagged - changes):>18}")
    print(f"{'descriptor: static':<24}{len(changes - flagged):>18}"
          f"{len(records) - len(flagged | changes):>18}")
    missed = sorted(JOKERS[k].name for k in changes - flagged)
    print(f"\nmissed by the flag: {missed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
