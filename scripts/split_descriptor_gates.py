"""Split D into D1/D2, mirroring the B1/B2 split, and re-run the invariants.

Input is the HAND-EDITED sheet (data/joker_descriptors_v2.csv); output is v3.
Idempotent: run it on a sheet that already has D1/D2 and it re-derives the
check flags without disturbing the routing.

Why the split: B already separates the payout event (B1) from the mutation
event (B2), so a single D column cannot say WHICH of the two a condition
gates. Wee Joker pays its accumulated chips unconditionally and grows only
when a 2 is scored; Hit the Road pays unconditionally and grows per Jack
discarded. Filing "rank" under one D column made both look like conditional
payouts, which is the opposite of how they play.

Routing rule:

* D2 takes the gate whenever the payout is unconditional and the condition
  describes the growth event. That is every gated growth joker except the
  countdown family.
* D1 takes gates on FIRING. For the countdown family the gate is the joker's
  own accumulated counter, which is structurally unlike an external gate --
  it is predictable from E plus the counter rather than from board state --
  so it gets its own ``own_counter`` token.
* Static and queries jokers have no B2, so their gate stays in D1 unchanged.
"""

from __future__ import annotations

import collections
import csv
import sys
from pathlib import Path

SRC = Path("data/joker_descriptors_v2.csv")
OUT = Path("data/joker_descriptors_v3.csv")
LEGEND = Path("data/joker_descriptors_v3_legend.csv")
# The hand-filled original, still the only source for E_destruction_proposed.
REVIEW = Path("data/joker_descriptors_review.csv")

GROWTH_E = {
    "grows_permanently",
    "grows_with_reset",
    "grows_bidirectionally",
    "decays_then_expires",
    "countdown_then_expires",
    "countdown_then_payoff",
    "sell_value_growth",
}

# Growth jokers whose existing gate describes the ALTERATION, not the payout.
GATE_TO_D2 = frozenset({
    "j_ride_the_bus", "j_runner", "j_square", "j_vampire", "j_obelisk",
    "j_lucky_cat", "j_trousers", "j_castle", "j_glass", "j_wee",
    "j_hit_the_road", "j_caino",
})

# Jokers whose FIRING is gated on their own accumulated counter.
OWN_COUNTER_GATE = frozenset({
    "j_invisible", "j_loyalty_card", "j_selzer", "j_yorick",
})

# Backfill for rows whose E became a growth type during hand review, so the
# dispatcher invariant (growth obliges B2) can still be satisfied.
B2_BACKFILL = {"j_loyalty_card": "after_hand_played"}

# Corrections found by cross-checking the sheet against the engine's own
# trigger_class taxonomy, config keys, and scenario text. Each is a
# contradiction with engine data, not a matter of taste.
FIXES: dict[str, dict[str, str]] = {
    # "x3 Mult if hand has Diamond+Club+Heart+Spade" -- no rank condition.
    "j_flower_pot": {"D1_gates_firing": "suit"},
    # "Retrigger held-in-hand effects", and it is C1_per_card_static, so
    # passive wrongly claims it never fires on a card event.
    "j_mime": {"B1_fires_when": "held"},
    # Writes perma_bonus to the SCORED cards; the dump script's docstring
    # calls this out as a block-C modifies_cards fact, not a chips payout.
    "j_hiker": {"C1_effect": "modifies_cards", "B1_fires_when": "scored"},
    # C2_per_card_state_dep with a rank that re-rolls each round -- the
    # Ancient Joker / The Idol shape, both of which carry D_variable.
    "j_mail": {"D_variable": "Y"},
    # odds: 2 in config gates the payout alongside the card condition.
    "j_reserved_parking": {"D1_gates_firing": "rank, probability"},
    "j_bloodstone": {"D1_gates_firing": "suit, probability"},
}
# Gros Michel and Cavendish also carry odds, deliberately WITHOUT a
# probability gate: theirs gates self-destruction, not the payout, so it
# belongs to E_destruction_proposed instead. Confirmed 2026-09-03.

FIELDNAMES = [
    "key", "name", "rarity", "cost", "blueprint_compat",
    "engine_effect_label", "engine_config",
    "E_growth", "E_memory", "E_destruction",
    "B1_fires_when", "B2_altered_when",
    "C1_effect", "C2_effect", "C_rule_subtype",
    "D1_gates_firing", "D2_gates_alteration", "D_variable",
    "F_operand", "F_feeds",
    "check_flags",
    "orig_E_confirm", "orig_B_when", "orig_C_what",
    "orig_D_gated_by", "orig_F_depends_on", "orig_MEMORY",
]

LEGEND_ROWS = [
    ("block", "column", "means", "vocabulary"),
    ("E", "E_growth",
     "Is today's number predictive of tomorrow's? Dispatcher: grows* obliges "
     "B2, queries obliges F.",
     "static | queries | grows_permanently | grows_with_reset | "
     "grows_bidirectionally | decays_then_expires | countdown_then_expires | "
     "countdown_then_payoff | sell_value_growth"),
    ("E", "E_memory",
     "Does evaluating it need past-state info, wherever that state lives?", "Y | N"),
    ("E", "E_destruction",
     "How the joker leaves play. Destruction is the PRICE, not the effect, so "
     "it is its own axis rather than a C entry.",
     "expires_when_depleted | random_destruction | consumed_on_use | "
     "destroyed_on_save"),
    ("B", "B1_fires_when", "The PAYOUT event.",
     "scored | held | before_hand_played | after_hand_played | discard | round_end | "
     "blind_select | shop_enter | shop_buy | shop_reroll | shop_sell | on_sell | passive"),
    ("B", "B2_altered_when",
     "The MUTATION event -- what changes the stored number. Blank unless E is a "
     "growth type.",
     "B1 vocabulary plus boss_defeated | card_destroyed | card_added_to_deck | "
     "consumable_used | pack_skip"),
    ("C", "C1_effect", "Primary resource or rule it moves.",
     "chips | mult | xmult | money | retrigger | hand_levels | creates_cards | "
     "modifies_cards | destroys_cards | copies_joker | sell_value | hand_size_mod | "
     "hand_count_mod | discard_count_mod | changes_rules"),
    ("C", "C2_effect",
     "Secondary effects, so multi-effect jokers stop collapsing into one cell.",
     "same as C1"),
    ("C", "C_rule_subtype", "Keeps changes_rules from becoming a junk drawer.",
     "detection | probability | shop_economy | run_rules"),
    ("D", "D1_gates_firing",
     "The predicate that decides whether B1 PAYS. own_counter means the gate "
     "reads the joker's own accumulated counter.",
     "rank | suit | hand_type | played_size | enhancement | probability | "
     "discard_count | hand_count | money | boss_blind | joker_type | joker_slots | "
     "hand_history | copy_target | own_counter"),
    ("D", "D2_gates_alteration",
     "The predicate that decides whether B2 ALTERS the stored number. Blank "
     "unless B2 is filled.", "same vocabulary as D1"),
    ("D", "D_variable",
     "Condition re-rolls (Ancient Joker, Mail-In Rebate). A modifier on a filled "
     "gate, never a replacement for one.", "Y | blank"),
    ("F", "F_operand",
     "The OPERAND -- external game state actually read. Card-level predicates do "
     "NOT appear here.",
     "money | deck_size | deck_composition | owned_jokers | joker_slots | "
     "hands_remaining | discards_remaining | hands_played_this_round | "
     "consumables_used | skips_taken | held_cards | rng"),
    ("F", "F_feeds",
     "Whether the query drives the gate (a frequency question) or the magnitude "
     "(a scaling one).", "gate | magnitude | magnitude, gate"),
    ("--", "check_flags", "Invariant violations. Empty is clean.", ""),
]


def route_gates(row: dict[str, str]) -> tuple[str, str]:
    """Return (D1, D2) for a row that may or may not already be split."""
    if "D1_gates_firing" in row:
        return row.get("D1_gates_firing", ""), row.get("D2_gates_alteration", "")

    key, gate = row["key"], row.get("D_gate", "")
    if key in OWN_COUNTER_GATE:
        d1 = ", ".join(t for t in ("own_counter", gate) if t)
        return d1, ""
    if key in GATE_TO_D2:
        return "", gate
    return gate, ""


def check(row: dict[str, str], d1: str, d2: str) -> list[str]:
    e, mem = row["E_growth"], row["E_memory"]
    flags: list[str] = []

    if e in GROWTH_E and not row["B2_altered_when"]:
        flags.append("growth_without_B2")
    if e == "queries" and not row["F_operand"]:
        flags.append("queries_without_F")
    if e == "static" and row["F_feeds"].startswith("magnitude"):
        flags.append("static_with_magnitude_query")
    if e in GROWTH_E and mem == "N":
        flags.append("growth_but_memory_N")
    # RULING: E tracks whether the NUMBER changes; whether it FIRES is the
    # gate's business. So a static magnitude may legitimately depend on
    # history, as long as a gate carries that dependence -- Delayed
    # Gratification pays a fixed $2 gated on discards used this round. Only
    # flag a memory claim that nothing explains.
    if e == "static" and mem == "Y" and not (d1 or row["F_operand"]):
        flags.append("memory_Y_with_nothing_reading_it")
    # Corollary of the same ruling: if the query only gates, the number is
    # fixed, so E is static -- Blackboard and Acrobat are the same shape.
    if e == "queries" and row["F_feeds"] == "gate":
        flags.append("queries_but_gate_only")
    if d2 and not row["B2_altered_when"]:
        flags.append("D2_without_B2")
    if e not in GROWTH_E and d2:
        flags.append("D2_on_non_growth")
    return flags


def load_destruction() -> dict[str, str]:
    """Pull E_destruction_proposed (col 18) from the hand-filled original.

    The v2 rebuild dropped this column; the original Excel export is cp1252
    and padded out to the sheet row limit, so only the first 150 data rows
    are real.
    """
    if not REVIEW.exists():
        return {}
    csv.field_size_limit(10**9)
    with REVIEW.open(encoding="cp1252", newline="") as fh:
        rows = list(csv.reader(fh))[1:151]
    return {r[0]: r[18].strip() for r in rows if r[18].strip()}


def main() -> int:
    # Prefer the live sheet once it exists, so a bare re-run re-derives the
    # invariants over hand edits instead of overwriting them from v2.
    default = OUT if OUT.exists() else SRC
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    with src.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    destruction = load_destruction()

    out_rows = []
    for row in rows:
        row = {**row, **FIXES.get(row["key"], {})}
        d1, d2 = route_gates(row)
        new = {k: row.get(k, "") for k in FIELDNAMES}
        if not new["E_destruction"]:
            new["E_destruction"] = destruction.get(row["key"], "")
        if not new["B2_altered_when"]:
            new["B2_altered_when"] = B2_BACKFILL.get(row["key"], "")
            row = {**row, "B2_altered_when": new["B2_altered_when"]}
        new["D1_gates_firing"] = d1
        new["D2_gates_alteration"] = d2
        new["check_flags"] = "; ".join(check(row, d1, d2))
        out_rows.append(new)

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    with LEGEND.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(LEGEND_ROWS)

    flagged = [r for r in out_rows if r["check_flags"]]
    d2_filled = sum(1 for r in out_rows if r["D2_gates_alteration"])
    d1_filled = sum(1 for r in out_rows if r["D1_gates_firing"])
    print(f"wrote {OUT} ({len(out_rows)} rows) and {LEGEND}")
    print(f"D1 filled: {d1_filled}   D2 filled: {d2_filled}")
    print(f"rows with check_flags: {len(flagged)}")
    tally = collections.Counter(
        f for r in flagged for f in r["check_flags"].split("; ")
    )
    for name, count in tally.most_common():
        print(f"  {count:4}  {name}")
    for r in flagged:
        print(f"    {r['key']:20} {r['name']:22} {r['check_flags']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
