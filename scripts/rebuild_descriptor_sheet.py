"""Rebuild the joker descriptor review sheet under the settled block conventions.

Reads the hand-filled Excel export (cp1252, Excel-padded to 1,048,576 rows) and
emits a clean UTF-8 sheet applying the conventions from
docs/embedding-analysis.md 5.4 plus the seams settled in review:

* B splits into B1_fires_when (payout event) and B2_altered_when (mutation
  event) -- Rocket pays at round end but grows on boss defeat.
* E is the dispatcher: E=grows* obliges B2, E=queries obliges F, E=static
  forbids both as magnitude sources.
* D keeps the PREDICATE (shape of the check), F keeps the OPERAND (state read).
  Card-level conditions on the trigger cards stay D-only; external game state
  earns an F entry as well.
* F_feeds tags whether the query drives the gate or the magnitude.
* C splits into C1/C2 so multi-effect jokers stop collapsing into one cell, and
  changes_rules gains a subtype so it is not a junk drawer.

Every original entry is preserved in an orig_* column; nothing is discarded.
"""

from __future__ import annotations

import collections
import csv
import re
import sys
from pathlib import Path

SRC = Path("data/joker_descriptors_review.csv")
OUT = Path("data/joker_descriptors_v2.csv")
LEGEND = Path("data/joker_descriptors_v2_legend.csv")

# --- normalization tables -------------------------------------------------
# Free-text entry produced heavy typo variance ("xmlt", "xmlyt", "cjhanges"),
# so fragments are matched by pattern after lowercasing, not by lookup.

C_MAP: list[tuple[str, str]] = [
    (r"^x\s*m", "xmult"),
    (r"^m(u|l)", "mult"),
    (r"^c(h|i|j)?(i|h)?p", "chips"),
    (r"money|cash|econ", "money"),
    (r"sell value", "sell_value"),
    (r"retrigger", "retrigger"),
    (r"rule", "changes_rules"),
    (r"level", "hand_levels"),
    (r"copy", "copies_joker"),
    (r"(make|add|generate)s? ?(card|taro|tarot|tartos|joker|spectral|tag)", "creates_cards"),
    (r"^(tarot|taro|tartos|spectral|tag)s?$", "creates_cards"),
    (r"alters cards|modifies cards", "modifies_cards"),
    (r"destroys", "destroys_cards"),
    (r"hand size", "hand_size_mod"),
    (r"discard", "discard_count_mod"),
    (r"hands", "hand_count_mod"),
]

D_MAP: list[tuple[str, str]] = [
    (r"probail|probabil|luck", "probability"),
    (r"hand.?type", "hand_type"),
    (r"played size", "played_size"),
    (r"discard(ed)? count", "discard_count"),
    (r"hand count", "hand_count"),
    (r"enha|enah", "enhancement"),
    (r"joker types", "joker_type"),
    (r"history", "hand_history"),
    (r"money", "money"),
    (r"boss", "boss_blind"),
    (r"space", "joker_slots"),
    (r"copy", "copy_target"),
    (r"rank", "rank"),
    (r"suit", "suit"),
]

# Operands naming state OUTSIDE the cards being played. These earn an F entry
# (tagged as feeding the gate); pure card-level predicates stay D-only.
D_EXTERNAL_TO_F = {
    "discard_count": "discards_remaining",
    "hand_count": "hands_remaining",
    "money": "money",
    "boss_blind": "boss_blind",
    "joker_type": "owned_jokers",
    "joker_slots": "joker_slots",
    "hand_history": "hands_played_this_round",
}

F_MAP = {
    "joker slots used": "joker_slots",
    "discard count": "discards_remaining",
    "random, unaltered by  dice": "rng",
    "smallest card": "held_cards",
    "steels in deck": "deck_composition",
    "hands played": "hands_played_this_round",
    "cards left in deck": "deck_size",
    "pervious hands played this round": "hands_played_this_round",
    "nines in deck": "deck_composition",
    "deck size": "deck_size",
    "tarots used": "consumables_used",
    "stones in deck": "deck_composition",
    "joker types": "owned_jokers",
    "skips taken": "skips_taken",
    "unique planets used": "consumables_used",
    "enhacnement in deck count": "deck_composition",
    "money": "money",
}

B_MAP = {
    "before_hand_plated": "before_hand_played",
    "after_hand_scored": "after_hand_played",
    "after_round": "round_end",
    "blind_start": "blind_select",
    "blind": "blind_select",
    "blind/ round_start": "blind_select",
    "entering shop": "shop_enter",
    "shop: buy pack": "shop_buy",
    "shop: reroll": "shop_reroll",
    "shop:reroll": "shop_reroll",
    "shop: sell": "shop_sell",
    "passive/shop": "passive",
    "passive ?": "passive",
    "skip": "pack_skip",
    "copy": "copies_joker",
}

# --- hand-authored: the mutation event, per the B1/B2 split ----------------
# Blank for every static and queries joker by construction.
B2_ALTERED: dict[str, str] = {
    "j_ceremonial": "blind_select",
    "j_ride_the_bus": "after_hand_played",
    "j_egg": "round_end",
    "j_runner": "after_hand_played",
    "j_ice_cream": "after_hand_played",
    "j_constellation": "consumable_used",
    "j_green_joker": "after_hand_played, discard",
    "j_red_card": "pack_skip",
    "j_madness": "blind_select",
    "j_square": "after_hand_played",
    "j_vampire": "scored",
    "j_hologram": "card_added_to_deck",
    "j_rocket": "boss_defeated",
    "j_obelisk": "after_hand_played",
    "j_turtle_bean": "round_end",
    "j_lucky_cat": "scored",
    "j_flash": "shop_reroll",
    "j_popcorn": "round_end",
    "j_trousers": "after_hand_played",
    "j_ramen": "discard",
    "j_selzer": "scored",
    "j_castle": "discard",
    "j_campfire": "shop_sell",
    "j_glass": "card_destroyed",
    "j_wee": "scored",
    "j_hit_the_road": "discard",
    "j_invisible": "round_end",
    "j_caino": "card_destroyed",
    "j_yorick": "discard",
}

RULE_SUBTYPE: dict[str, str] = {
    "j_four_fingers": "detection",
    "j_shortcut": "detection",
    "j_splash": "detection",
    "j_smeared": "detection",
    "j_pareidolia": "detection",
    "j_oops": "probability",
    "j_credit_card": "shop_economy",
    "j_ring_master": "shop_economy",
    "j_chicot": "run_rules",
    "j_luchador": "run_rules",
    "j_mr_bones": "run_rules",
}

# Whether a query drives the GATE (fixed magnitude, the query only decides
# whether it fires) or the MAGNITUDE (the payout scales with the operand).
# Hand-authored: the distinction is not recoverable from config, since both
# shapes store a number. Everything else defaults to magnitude.
QUERY_FEEDS_GATE = frozenset({"j_card_sharp", "j_drivers_license"})

GROWTH_E = {
    "grows_permanently",
    "grows_with_reset",
    "grows_bidirectionally",
    "decays_then_expires",
    "countdown_then_expires",
    "countdown_then_payoff",
    "sell_value_growth",
}


def clean(s: str) -> str:
    """Strip the cp1252 ellipsis artifact left by the Excel round-trip."""
    return s.replace("\x85", "").replace("…", "").strip()


def split_frags(s: str) -> list[str]:
    return [f.strip() for f in re.split(r"[,+/]", s) if f.strip()]


def map_frag(frag: str, table: list[tuple[str, str]]) -> str:
    f = frag.lower().strip(" ?()-")
    for pat, out in table:
        if re.search(pat, f):
            return out
    return ""


def norm_multi(raw: str, table: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    """Return (mapped tokens in order, fragments that did not map)."""
    mapped: list[str] = []
    unmapped: list[str] = []
    for frag in split_frags(raw):
        tok = map_frag(frag, table)
        if tok:
            if tok not in mapped:
                mapped.append(tok)
        else:
            unmapped.append(frag)
    return mapped, unmapped


def norm_b(raw: str) -> str:
    out: list[str] = []
    for frag in split_frags(raw):
        f = frag.lower().strip(" ?")
        f = B_MAP.get(f, f).replace(" ", "_")
        if f and f not in out:
            out.append(f)
    return ", ".join(out)


FIELDNAMES = [
    "key", "name", "rarity", "cost", "blueprint_compat",
    "engine_effect_label", "engine_config",
    "E_growth", "E_memory",
    "B1_fires_when", "B2_altered_when",
    "C1_effect", "C2_effect", "C_rule_subtype",
    "D_gate", "D_variable",
    "F_operand", "F_feeds",
    "check_flags",
    "orig_E_confirm", "orig_B_when", "orig_C_what",
    "orig_D_gated_by", "orig_F_depends_on", "orig_MEMORY",
]

LEGEND_ROWS = [
    ("block", "column", "means", "vocabulary"),
    ("E", "E_growth",
     "Is today's number predictive of tomorrow's? Dispatcher: grows* obliges B2, "
     "queries obliges F.",
     "static | queries | grows_permanently | grows_with_reset | grows_bidirectionally | "
     "decays_then_expires | countdown_then_expires | countdown_then_payoff | sell_value_growth"),
    ("E", "E_memory",
     "Does evaluating it need past-state info, wherever that state lives?", "Y | N"),
    ("B", "B1_fires_when", "The PAYOUT event.",
     "scored | held | before_hand_played | after_hand_played | discard | round_end | "
     "blind_select | shop_enter | shop_buy | shop_reroll | shop_sell | on_sell | passive"),
    ("B", "B2_altered_when",
     "The MUTATION event -- what changes the stored number. Blank unless E is a growth type.",
     "B1 vocabulary plus boss_defeated | card_destroyed | card_added_to_deck | "
     "consumable_used | pack_skip"),
    ("C", "C1_effect", "Primary resource or rule it moves.",
     "chips | mult | xmult | money | retrigger | hand_levels | creates_cards | "
     "modifies_cards | destroys_cards | copies_joker | sell_value | hand_size_mod | "
     "hand_count_mod | discard_count_mod | changes_rules"),
    ("C", "C2_effect",
     "Secondary effects, so multi-effect jokers stop collapsing into one cell.", "same as C1"),
    ("C", "C_rule_subtype", "Keeps changes_rules from becoming a junk drawer.",
     "detection | probability | shop_economy | run_rules"),
    ("D", "D_gate",
     "The PREDICATE -- shape of the check that lets it fire. Card-level conditions live here only.",
     "rank | suit | hand_type | played_size | enhancement | probability | discard_count | "
     "hand_count | money | boss_blind | joker_type | joker_slots | hand_history | copy_target"),
    ("D", "D_variable",
     "Condition re-rolls (Ancient Joker, Mail-In Rebate). A modifier on a filled gate, "
     "never a replacement for one.", "Y | blank"),
    ("F", "F_operand",
     "The OPERAND -- external game state actually read. Card-level predicates do NOT appear here.",
     "money | deck_size | deck_composition | owned_jokers | joker_slots | hands_remaining | "
     "discards_remaining | hands_played_this_round | consumables_used | skips_taken | "
     "held_cards | rng"),
    ("F", "F_feeds",
     "Whether the query drives the gate (a frequency question) or the magnitude (a scaling one).",
     "gate | magnitude | magnitude, gate"),
    ("--", "check_flags", "Invariant violations and unmapped free text. Empty is clean.", ""),
]


def build_row(r: list[str]) -> dict[str, str]:
    key, name = r[0], r[1]
    e_raw, b_raw, c_raw = clean(r[19]), clean(r[20]), clean(r[21])
    d_raw, f_raw, m_raw = clean(r[22]), clean(r[23]), clean(r[25])

    e = "queries" if e_raw.startswith("queries") else e_raw
    e_extra = "grows_permanently" if "grows monotonically" in e_raw else ""

    c_tokens, c_unmapped = norm_multi(c_raw, C_MAP)
    d_tokens, d_unmapped = norm_multi(d_raw, D_MAP)

    f_operands: list[str] = []
    f_feeds = ""
    if f_raw:
        f_operands.append(F_MAP.get(f_raw.lower(), f_raw))
        f_feeds = "gate" if key in QUERY_FEEDS_GATE else "magnitude"
    gate_ops = [D_EXTERNAL_TO_F[t] for t in d_tokens if t in D_EXTERNAL_TO_F]
    for op in gate_ops:
        if op not in f_operands:
            f_operands.append(op)
    if gate_ops and f_feeds != "gate":
        f_feeds = "magnitude, gate" if f_raw else "gate"

    flags: list[str] = []
    if e in GROWTH_E and key not in B2_ALTERED:
        flags.append("growth_without_B2")
    if e == "queries" and not f_operands:
        flags.append("queries_without_F")
    if e == "static" and f_feeds.startswith("magnitude"):
        flags.append("static_with_magnitude_query")
    if e in GROWTH_E and m_raw == "N":
        flags.append("growth_but_memory_N")
    if e == "static" and m_raw == "Y":
        flags.append("static_but_memory_Y")
    if c_unmapped:
        flags.append("C_unmapped:" + "|".join(c_unmapped))
    if d_unmapped:
        flags.append("D_unmapped:" + "|".join(d_unmapped))
    if e_extra:
        flags.append("E_compound:" + e_extra)

    return {
        "key": key, "name": name, "rarity": r[2], "cost": r[3],
        "blueprint_compat": r[5],
        "engine_effect_label": r[7], "engine_config": r[6],
        "E_growth": e, "E_memory": m_raw,
        "B1_fires_when": norm_b(b_raw),
        "B2_altered_when": B2_ALTERED.get(key, ""),
        "C1_effect": c_tokens[0] if c_tokens else "",
        "C2_effect": ", ".join(c_tokens[1:]),
        "C_rule_subtype": RULE_SUBTYPE.get(key, ""),
        "D_gate": ", ".join(d_tokens),
        "D_variable": "Y" if "variab" in d_raw.lower() else "",
        "F_operand": ", ".join(f_operands), "F_feeds": f_feeds,
        "check_flags": "; ".join(flags),
        "orig_E_confirm": e_raw, "orig_B_when": b_raw, "orig_C_what": c_raw,
        "orig_D_gated_by": d_raw, "orig_F_depends_on": f_raw, "orig_MEMORY": m_raw,
    }


def main() -> int:
    csv.field_size_limit(10**9)
    with SRC.open(encoding="cp1252", newline="") as fh:
        rd = list(csv.reader(fh))

    out_rows = [build_row(r) for r in rd[1:151]]

    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    with LEGEND.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(LEGEND_ROWS)

    flagged = [r for r in out_rows if r["check_flags"]]
    print(f"wrote {OUT} ({len(out_rows)} rows) and {LEGEND}")
    print(f"rows with check_flags: {len(flagged)}")
    tally = collections.Counter(
        f.split(":")[0] for r in flagged for f in r["check_flags"].split("; ")
    )
    for name, count in tally.most_common():
        print(f"  {count:4}  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
