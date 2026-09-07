"""Apply the 2026-09-03 review rulings to the descriptor sheet: v3 -> v4.

Input  : data/joker_descriptors_v3.csv
Output : data/joker_descriptors_v4.csv, data/joker_descriptors_v4_legend.csv

Every edit below is a ruling from the review pass, not a matter of taste. The
orig_* audit columns are carried through untouched.

Vocabulary changes (the key moves with the sheet):
  B1  + shop_exit, pack_open, varies
      - shop_buy, shop_enter
  C   + destroys_joker
  D   - discard_count, hand_history  (a bare count is an F operand, not a
        firing predicate; hand_history duplicated first_hand_of_round)
      + first_hand_of_round, first_discard_of_round, played_cards,
        cards_discarded
  E_destruction - consumed_on_use  (the two rows that carried it destroy a
        CARD, which is a C-block fact)
  F   hands_played_this_round -> hands_played_this_blind, so the run/blind
      ladder reads consistently across the D and F blocks
      + hands_played_this_run

Every token that survives has at least one member. A zero-member token is a
trap, not headroom: consumed_on_use had none, and both rows filed under it
were wrong. Re-add one when a joker needs it.
"""

from __future__ import annotations

import csv
from pathlib import Path

SRC = Path("data/joker_descriptors_v3.csv")
OUT = Path("data/joker_descriptors_v4.csv")
LEGEND = Path("data/joker_descriptors_v4_legend.csv")

# key -> {column: new value}. "" clears the cell.
RULINGS: dict[str, dict[str, str]] = {
    # --- vocabulary violations -------------------------------------------
    # Free reroll is a shop-economy rule change, and the event it attaches to
    # is the reroll. "shop" was never a token.
    "j_chaos": {
        "B1_fires_when": "shop_reroll",
        "C1_effect": "changes_rules",
        "C_rule_subtype": "shop_economy",
    },
    # Engine fires on first_hand_drawn; blind_select is the vocabulary's
    # nearest functional equivalent. Card creation alone -- the seal rides on
    # the created card rather than modifying an existing one.
    "j_certificate": {"B1_fires_when": "blind_select", "C2_effect": ""},
    # Reads game.skips (blinds skipped this run). The skip is an ALTERATION,
    # carried by F_operand=skips_taken; it is not a payout event.
    "j_throwback": {"B1_fires_when": "after_hand_played"},
    # Fires whenever the copy target fires -- no fixed event exists.
    "j_blueprint": {"B1_fires_when": "varies"},
    "j_brainstorm": {"B1_fires_when": "varies"},
    # Gated on the boss blind AND on a card predicate that differs per boss,
    # which is exactly what D_variable means.
    "j_matador": {
        "D1_gates_firing": "boss_blind, played_cards",
        "D_variable": "Y",
        "F_operand": "",
        "F_feeds": "",
    },
    # --- wrong cell values -----------------------------------------------
    # Engine does j.sell_cost += increment. No money moves.
    "j_gift": {"C1_effect": "sell_value"},
    # Passive shop pricing rule, not income and not hand levels.
    "j_astronomer": {
        "C1_effect": "changes_rules",
        "C2_effect": "",
        "C_rule_subtype": "shop_economy",
    },
    # Sums every other joker's sell_cost -- a magnitude query over owned jokers.
    "j_swashbuckler": {
        "E_growth": "queries",
        "F_operand": "owned_jokers",
        "F_feeds": "magnitude",
    },
    # Pays under ctx.joker_main. (The target hand re-rolling at round end is
    # already carried by D_variable.)
    "j_todo_list": {"B1_fires_when": "after_hand_played"},
    # Mutation on reroll, payout during scoring. B1 previously held only the
    # mutation, so the row claimed a joker that pays out mid-shop.
    "j_flash": {"B1_fires_when": "after_hand_played", "B2_altered_when": "shop_reroll"},
    # Gate is ability["name"] == "Gold Card".
    "j_ticket": {"D1_gates_firing": "enhancement"},
    # rng.random("business") gates the payout alongside the face-card check.
    "j_business": {"D1_gates_firing": "rank, probability"},
    # rng.random("space") is the whole gate.
    "j_space": {"D1_gates_firing": "probability"},
    # Fires under joker_main; held cards are the OPERAND, not the channel.
    "j_blackboard": {
        "B1_fires_when": "after_hand_played",
        "F_operand": "held_cards",
        "F_feeds": "gate",
    },
    # Returns discards_left * extra: the query sets the magnitude as well as
    # the gate, which makes this the Banner shape -- a query, not a static.
    "j_delayed_grat": {
        "E_growth": "queries",
        "D1_gates_firing": "",
        "F_feeds": "magnitude, gate",
    },
    # remove=True destroys the PLAYED card, not the joker (scoring.py:814).
    # The destruction is already carried by C1=destroys_cards.
    "j_sixth_sense": {
        "E_destruction": "",
        "D1_gates_firing": "first_hand_of_round, played_size, rank",
    },
    # remove=True + extra.destroy destroys the DISCARDED card (game.py:882).
    "j_trading": {
        "E_destruction": "",
        "D1_gates_firing": "first_discard_of_round, played_size",
        "F_operand": "",
        "F_feeds": "",
    },
    # grows_with_reset, so B2 must carry the reset: end_of_round on a boss.
    "j_campfire": {
        "B2_altered_when": "shop_sell, round_end",
        "D2_gates_alteration": "boss_blind",
    },
    "j_hit_the_road": {"B2_altered_when": "discard, round_end"},
    # Counts ROUNDS on an internal counter (invis_rounds at end_of_round).
    # Nothing external is read, and hand_history was never involved.
    "j_invisible": {"D1_gates_firing": "own_counter", "F_operand": "", "F_feeds": ""},
    # hand_levels[name].played -- the whole run, not the current blind.
    "j_supernova": {"F_operand": "hands_played_this_run"},
    # ctx.ending_shop: leaving, not entering.
    "j_perkeo": {"B1_fires_when": "shop_exit"},
    # ctx.open_booster, which is not a purchase.
    "j_hallucination": {"B1_fires_when": "pack_open"},
    # ctx.setting_blind gated on blind.boss.
    "j_chicot": {"B1_fires_when": "blind_select", "D1_gates_firing": "boss_blind"},
    # Sells into a boss blind; the boss check is a real gate on the payout.
    "j_luchador": {"D1_gates_firing": "boss_blind"},
    # --- spurious B1 tokens (handler never reads that context) ------------
    # Supernova and Vampire KEEP before_hand_played: the played-hand counter
    # and the enhancement strip both land ahead of scoring.
    "j_seance": {"B1_fires_when": "after_hand_played"},
    "j_bootstraps": {"B1_fires_when": "after_hand_played"},
    # --- joker destruction now has a home --------------------------------
    "j_ceremonial": {"C2_effect": "destroys_joker"},
    "j_madness": {"C2_effect": "destroys_joker"},
    # --- E reclassification ----------------------------------------------
    # The counter resets but the xMult does not: the gain is permanent, so
    # grows_permanently is the more predictive answer to E's question. The
    # countdown survives as the alteration gate. E is single-valued, so it
    # cannot carry both.
    "j_yorick": {
        "E_growth": "grows_permanently",
        "D1_gates_firing": "",
        "D2_gates_alteration": "cards_discarded, own_counter",
    },
    # Gated on whether this hand type has been seen before this blind -- the
    # first-occurrence predicate, paying on false. The raw per-blind counter it
    # reads stays in F, since a bare count is not a predicate.
    "j_card_sharp": {
        "D1_gates_firing": "first_hand_of_round",
        "F_operand": "hands_played_this_blind",
    },
    # --- first-hand / first-discard gates now exist ----------------------
    "j_dna": {"D1_gates_firing": "first_hand_of_round, played_size"},
    "j_burnt": {"D1_gates_firing": "first_discard_of_round", "F_operand": "", "F_feeds": ""},
    # discard_count leaves the D block; the gate it stood for is the F query.
    "j_mystic_summit": {"D1_gates_firing": ""},
}

# Global token rename inside the F block.
F_RENAME = {"hands_played_this_round": "hands_played_this_blind"}

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

B1_VOCAB = (
    "scored | held | before_hand_played | after_hand_played | discard | round_end | "
    "blind_select | shop_exit | shop_reroll | shop_sell | pack_open | on_sell | "
    "passive | varies"
)
C_VOCAB = (
    "chips | mult | xmult | money | retrigger | hand_levels | creates_cards | "
    "modifies_cards | destroys_cards | destroys_joker | copies_joker | sell_value | "
    "hand_size_mod | hand_count_mod | discard_count_mod | changes_rules"
)
D_VOCAB = (
    "rank | suit | hand_type | played_size | played_cards | enhancement | probability | "
    "first_hand_of_round | first_discard_of_round | cards_discarded | hand_count | "
    "money | boss_blind | joker_type | joker_slots | copy_target | own_counter"
)
E_DESTRUCTION_VOCAB = "expires_when_depleted | random_destruction | destroyed_on_save"
F_VOCAB = (
    "money | deck_size | deck_composition | owned_jokers | joker_slots | "
    "hands_remaining | discards_remaining | hands_played_this_run | "
    "hands_played_this_blind | consumables_used | skips_taken | held_cards | rng"
)

LEGEND_ROWS = [
    ("block", "column", "means", "vocabulary"),
    ("E", "E_growth",
     "Is today's number predictive of tomorrow's? Dispatcher: grows* obliges "
     "B2, queries obliges F. Single-valued: a joker that both counts down and "
     "keeps its gain (Yorick) files under the permanent half, and the countdown "
     "survives as its D2 gate.",
     "static | queries | grows_permanently | grows_with_reset | "
     "grows_bidirectionally | decays_then_expires | countdown_then_expires | "
     "countdown_then_payoff | sell_value_growth"),
    ("E", "E_memory",
     "Does evaluating it need past-state info, wherever that state lives?", "Y | N"),
    ("E", "E_destruction",
     "How the JOKER leaves play. Destruction is the PRICE, not the effect, so it "
     "is its own axis rather than a C entry. A joker that destroys a CARD is not "
     "itself destroyed: that is C=destroys_cards. Being sold is not destruction "
     "either -- every joker can be sold.",
     E_DESTRUCTION_VOCAB),
    ("B", "B1_fires_when",
     "The PAYOUT event. varies = fires whenever the copy target fires.", B1_VOCAB),
    ("B", "B2_altered_when",
     "The MUTATION event -- what changes the stored number. Blank unless E is a "
     "growth type. A grows_with_reset joker must name its RESET event here too.",
     "B1 vocabulary plus boss_defeated | card_destroyed | card_added_to_deck | "
     "consumable_used | pack_skip"),
    ("C", "C1_effect", "Primary resource or rule it moves.", C_VOCAB),
    ("C", "C2_effect",
     "Secondary effects, so multi-effect jokers stop collapsing into one cell.",
     "same as C1"),
    ("C", "C_rule_subtype", "Keeps changes_rules from becoming a junk drawer.",
     "detection | probability | shop_economy | run_rules"),
    ("D", "D1_gates_firing",
     "The predicate that decides whether B1 PAYS. own_counter means the gate reads "
     "the joker's own accumulated counter. played_cards means a predicate over the "
     "played cards whose specifics are not fixed (pair it with D_variable). A bare "
     "COUNT is not a predicate -- it belongs in F.",
     D_VOCAB),
    ("D", "D2_gates_alteration",
     "The predicate that decides whether B2 ALTERS the stored number. Blank "
     "unless B2 is filled.", "same vocabulary as D1"),
    ("D", "D_variable",
     "Condition re-rolls (Ancient Joker, Mail-In Rebate, To Do List) or differs by "
     "boss (Matador). A modifier on a filled gate, never a replacement for one.",
     "Y | blank"),
    ("F", "F_operand",
     "The OPERAND -- external game state actually read. Card-level predicates do "
     "NOT appear here. The hands_played_* ladder is run > blind > turn; 'blind' is "
     "the engine's played_this_round.",
     F_VOCAB),
    ("F", "F_feeds",
     "Whether the query drives the gate (a frequency question) or the magnitude "
     "(a scaling one).", "gate | magnitude | magnitude, gate"),
    ("--", "check_flags", "Invariant violations. Empty is clean.", ""),
]


def main() -> int:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    by_key = {r["key"]: r for r in rows}

    missing = sorted(set(RULINGS) - set(by_key))
    if missing:
        raise SystemExit(f"rulings reference unknown keys: {missing}")

    applied = 0
    for key, edits in RULINGS.items():
        row = by_key[key]
        for col, val in edits.items():
            if col not in FIELDNAMES:
                raise SystemExit(f"{key}: unknown column {col}")
            if row[col] != val:
                applied += 1
            row[col] = val

    renamed = 0
    for row in rows:
        toks = [t.strip() for t in row["F_operand"].split(",") if t.strip()]
        new = [F_RENAME.get(t, t) for t in toks]
        if new != toks:
            renamed += 1
        row["F_operand"] = ", ".join(new)

    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})

    with LEGEND.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(LEGEND_ROWS)

    print(f"rows={len(rows)} cells_changed={applied} F_tokens_renamed={renamed}")
    print(f"wrote {OUT} and {LEGEND}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
