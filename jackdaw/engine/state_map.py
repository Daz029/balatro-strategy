"""Where every Lua game-state value lives in the engine (engine overhaul Phase 1).

Every ``G.GAME.<path>`` the vanilla 1.0.1o source reads or writes (at the depth
``scripts/engine_state_inventory.py`` scans, 2), every field Lua's ``Card`` and
``Blind`` objects carry, and every ``card.ability.<key>`` Lua touches is assigned
to exactly one of:

- :class:`Stored` — held in the game state at a dotted path (dict keys, or
  object attributes for ``Blind`` / ``Card`` / ``HandLevels``).
- :class:`Grabber` — derived on demand from live state by a getter (Phase 2
  writes these; the name here is the planned getter).
- :class:`OutOfScope` — UI, animation, profile/statistics, challenge-only, or
  provably equivalent; the reason says which.

``tests/engine/test_state_map.py`` enforces it: stored paths must resolve on a
fresh run, and (when the Lua source is present) every scanned Lua path must be
mapped here. See ``docs/engine_fixing_plan_2026-10-06.md`` Part 1b.

``lazy=True`` marks a stored key that legitimately does not exist on a fresh
run (Lua creates it lazily too, or only some decks/stakes/phases set it).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stored:
    path: str
    lazy: bool = False
    note: str = ""


@dataclass(frozen=True)
class Grabber:
    name: str
    note: str = ""


@dataclass(frozen=True)
class OutOfScope:
    reason: str


Entry = Stored | Grabber | OutOfScope

_UI = OutOfScope("UI / animation / display text")
_STATS = OutOfScope("statistics / profile / unlock tracking (D59, D44)")
_CHALLENGE = OutOfScope("challenge-only modifier; challenges raise (D-scope, C15)")

# ---------------------------------------------------------------------------
# G.GAME paths (depth 2, as scanned)
# ---------------------------------------------------------------------------

GAME_PATHS: dict[str, Entry] = {
    "PACK_INTERRUPT": _UI,
    "STOP_USE": _UI,
    "bankrupt_at": Stored("bankrupt_at"),
    "banned_keys": Stored("banned_keys"),
    "blind": Stored("blind", note="Blind object; None outside a round"),
    "blind.blind_set": _UI,
    "blind.block_play": OutOfScope("UI: blocks input while the debuff alert animates"),
    "blind.boss": Stored("blind.boss"),
    "blind.children": _UI,
    "blind.chip_text": _UI,
    "blind.chips": Stored("blind.chips"),
    "blind.config": Stored("blind.key", note="config.blind is the prototype, found by key"),
    "blind.disabled": Stored("blind.disabled"),
    "blind.dollars": Stored("blind.dollars"),
    "blind.loc_debuff_lines": _UI,
    "blind.loc_debuff_text": _UI,
    "blind.name": Stored("blind.name"),
    "blind.pos": _UI,
    "blind.states": _UI,
    "blind.triggered": Stored("blind.triggered"),
    "blind_on_deck": Stored("blind_on_deck"),
    "bosses_used": Stored("bosses_used"),
    "cards_played": _STATS,
    "challenge": _CHALLENGE,
    "chips": Stored("chips"),
    "chips_text": _UI,
    "consumeable_buffer": Stored("consumeable_buffer", note="writer: Phase 4 applier"),
    "consumeable_usage": Stored("consumeable_usage", note="writer: set_consumeable_usage (D40)"),
    "consumeable_usage_total": Stored("consumeable_usage_total"),
    "consumeable_usage_total.all": Stored("consumeable_usage_total.all"),
    "consumeable_usage_total.planet": Stored("consumeable_usage_total.planet"),
    "consumeable_usage_total.spectral": Stored("consumeable_usage_total.spectral"),
    "consumeable_usage_total.tarot": Stored("consumeable_usage_total.tarot"),
    "consumeable_usage_total.tarot_planet": Stored("consumeable_usage_total.tarot_planet"),
    "current_boss_streak": _STATS,
    "current_round": Stored("current_round"),
    "current_round.ancient_card": Stored("current_round.ancient_card"),
    "current_round.castle_card": Stored("current_round.castle_card"),
    "current_round.current_hand": OutOfScope("UI: HUD mirror of the running chips/mult"),
    "current_round.discards_left": Stored("current_round.discards_left"),
    "current_round.discards_used": Stored("current_round.discards_used"),
    "current_round.dollars": Stored(
        "round_earnings",
        lazy=True,
        note="Lua's round-eval total paid at cash-out; engine keeps the breakdown",
    ),
    "current_round.dollars_to_be_earned": _UI,
    "current_round.free_rerolls": Stored("current_round.free_rerolls"),
    "current_round.hands_left": Stored("current_round.hands_left"),
    "current_round.hands_played": Stored("current_round.hands_played"),
    "current_round.idol_card": Stored("current_round.idol_card"),
    "current_round.jokers_purchased": Stored("current_round.jokers_purchased"),
    "current_round.mail_card": Stored("current_round.mail_card"),
    "current_round.most_played_poker_hand": Stored("current_round.most_played_poker_hand"),
    "current_round.reroll_cost": Stored("current_round.reroll_cost"),
    "current_round.reroll_cost_increase": Stored("current_round.reroll_cost_increase"),
    "current_round.round_text": _UI,
    "current_round.used_packs": Stored("current_round.used_packs"),
    "current_round.voucher": Stored("current_round.voucher"),
    "discount_percent": Stored("discount_percent"),
    "dollar_buffer": Stored("dollar_buffer", note="writer: Phase 5 money ledger"),
    "dollars": Stored("dollars"),
    "ecto_minus": Stored("ecto_minus", note="writer: Ectoplasm (D34)"),
    "edition_rate": Stored("edition_rate"),
    "facing_blind": Stored("facing_blind", note="game select/evaluate round (Certificate, C02)"),
    "first_shop_buffoon": Stored("first_shop_buffoon", lazy=True),
    "first_used_hand_level": _STATS,
    "hand_usage": _STATS,
    "hands": Stored("hand_levels", note="HandLevels object"),
    "hands_played": Stored("hands_played"),
    "inflation": Stored("inflation"),
    "interest_amount": Stored("interest_amount"),
    "interest_cap": Stored("interest_cap"),
    "joker_buffer": Stored("joker_buffer", note="writer: Phase 4 applier"),
    "joker_rate": Stored("joker_rate"),
    "last_blind": Grabber(
        "last_blind",
        note=(
            "equivalent to gs['blind'] at every read: Lua's eval tags run synchronously "
            "in evaluate_round, before the queued Blind:defeat resets last_blind"
        ),
    ),
    "last_blind.boss": Grabber("last_blind"),
    "last_blind.name": Grabber("last_blind"),
    "last_hand_played": Stored("last_hand_played"),
    "last_tarot_planet": Stored("last_tarot_planet"),
    "max_jokers": _STATS,
    "modifiers.all_eternal": _CHALLENGE,
    "modifiers.booster_ante_scaling": _CHALLENGE,
    "modifiers.chips_dollar_cap": _CHALLENGE,
    "modifiers.debuff_played_cards": _CHALLENGE,
    "modifiers.discard_cost": _CHALLENGE,
    "modifiers.enable_eternals_in_shop": Stored("modifiers.enable_eternals_in_shop", lazy=True),
    "modifiers.enable_perishables_in_shop": Stored(
        "modifiers.enable_perishables_in_shop", lazy=True
    ),
    "modifiers.enable_rentals_in_shop": Stored("modifiers.enable_rentals_in_shop", lazy=True),
    "modifiers.flipped_cards": _CHALLENGE,
    "modifiers.inflation": _CHALLENGE,
    "modifiers.minus_hand_size_per_X_dollar": _CHALLENGE,
    "modifiers.money_per_discard": Stored("modifiers.money_per_discard", lazy=True),
    "modifiers.money_per_hand": Stored("modifiers.money_per_hand", lazy=True),
    "modifiers.no_blind_reward": Stored("modifiers.no_blind_reward", lazy=True),
    "modifiers.no_extra_hand_money": _CHALLENGE,
    "modifiers.no_interest": Stored("modifiers.no_interest", lazy=True),
    "modifiers.scaling": Stored("modifiers.scaling", lazy=True),
    "modifiers.set_eternal_ante": _CHALLENGE,
    "modifiers.set_joker_slots_ante": _CHALLENGE,
    "orbital_choices": Stored("orbital_choices", note="writer: Orbital Tag offer (D54)"),
    "pack_choices": Stored("pack_choices_remaining", lazy=True, note="engine name"),
    "pack_size": OutOfScope(
        "equivalent: the engine sizes the pack from the booster's extra at open; "
        "nothing reads it afterwards"
    ),
    "perishable_rounds": Stored("perishable_rounds"),
    "perscribed_bosses": OutOfScope("debug / challenge boss override"),
    "planet_rate": Stored("planet_rate"),
    "playing_card_rate": Stored("playing_card_rate"),
    "pool_flags": Stored("pool_flags"),
    "pool_flags.gros_michel_extinct": Stored("pool_flags.gros_michel_extinct", lazy=True),
    "previous_round.dollars": _UI,
    "probabilities": Stored("probabilities"),
    "probabilities.normal": Stored("probabilities.normal"),
    "pseudorandom": Stored("rng", note="PseudoRandom keeps Lua's per-key stream state"),
    "pseudorandom.hashed_seed": Stored("rng"),
    "pseudorandom.seed": Stored("rng"),
    "rental_rate": Stored("rental_rate"),
    "round": Stored("round"),
    "round_bonus.discards": Stored("round_bonus.discards"),
    "round_bonus.next_hands": Stored("round_bonus.next_hands"),
    "round_resets": Stored("round_resets"),
    "round_resets.ante": Stored("round_resets.ante"),
    "round_resets.blind": Stored("round_resets.blind", lazy=True),
    "round_resets.blind_ante": Stored("round_resets.blind_ante"),
    "round_resets.blind_choices": Stored("round_resets.blind_choices"),
    "round_resets.blind_states": Stored("round_resets.blind_states"),
    "round_resets.blind_tags": Stored("round_resets.blind_tags"),
    "round_resets.boss_rerolled": Stored("round_resets.boss_rerolled"),
    "round_resets.discards": Stored("round_resets.discards"),
    "round_resets.hands": Stored("round_resets.hands"),
    "round_resets.loc_blind_states": _UI,
    "round_resets.reroll_cost": Stored("round_resets.reroll_cost"),
    "round_resets.temp_handsize": Stored("round_resets.temp_handsize"),
    "round_resets.temp_reroll_cost": Stored("round_resets.temp_reroll_cost"),
    "round_scores": _STATS,
    "round_scores.cards_discarded": _STATS,
    "round_scores.cards_played": _STATS,
    "round_scores.cards_purchased": _STATS,
    "round_scores.new_collection": _STATS,
    "round_scores.times_rerolled": _STATS,
    "seeded": Stored("seeded"),
    "selected_back": Stored("selected_back_key", note="Back object built from the key"),
    "selected_back.effect": Stored("selected_back_key"),
    "selected_back.loc_name": _UI,
    "selected_back.pos": _UI,
    "shop": Stored("shop"),
    "shop.joker_max": Stored("shop.joker_max"),
    "shop_d6ed": Stored("shop_d6ed", note="one D6 Tag per shop visit"),
    "shop_free": Stored("shop_free", note="one Coupon Tag per shop visit"),
    "skips": Stored("skips"),
    "spectral_rate": Stored("spectral_rate"),
    "stake": Stored("stake"),
    "starting_deck_size": Stored("starting_deck_size"),
    "starting_params.ante_scaling": Stored("starting_params.ante_scaling"),
    "starting_params.consumable_slots": Stored("starting_params.consumable_slots"),
    "starting_params.discards": Stored("starting_params.discards"),
    "starting_params.dollars": Stored("starting_params.dollars"),
    "starting_params.erratic_suits_and_ranks": Stored("starting_params.erratic_suits_and_ranks"),
    "starting_params.hand_size": Stored("starting_params.hand_size"),
    "starting_params.hands": Stored("starting_params.hands"),
    "starting_params.joker_slots": Stored("starting_params.joker_slots"),
    "starting_params.no_faces": Stored("starting_params.no_faces"),
    "starting_params.reroll_cost": Stored("starting_params.reroll_cost"),
    "starting_voucher_count": _STATS,
    "subhash": _UI,
    "tag_tally": OutOfScope("tag identity counter; the engine keeps tags as an ordered list"),
    "tags": Stored(
        "awarded_tags",
        lazy=True,
        note="engine name for Lua G.GAME.tags (the dead init key 'tags' was removed)",
    ),
    "tarot_rate": Stored("tarot_rate"),
    "unused_discards": Stored("unused_discards"),
    "used_jokers": Stored("used_jokers"),
    "used_vouchers": Stored("used_vouchers"),
    "used_vouchers.v_observatory": Grabber("has_voucher"),
    "used_vouchers.v_omen_globe": Grabber("has_voucher"),
    "used_vouchers.v_telescope": Grabber("has_voucher"),
    "viewed_back": _UI,
    "viewed_back.effect": _UI,
    "viewed_back.name": _UI,
    "voucher_restock": OutOfScope("never read in vanilla (nil-assign only)"),
    "win_ante": Stored("win_ante"),
    "win_notified": _UI,
    "won": Stored("won"),
}

# ---------------------------------------------------------------------------
# Card fields (``self.<x> =`` in card.lua, plus fields Lua sets from outside)
# ---------------------------------------------------------------------------

CARD_FIELDS: dict[str, Entry] = {
    "ability": Stored("ability"),
    "added_to_deck": Stored("added_to_deck", note="writer: Phase 3 lifecycle"),
    "area": Grabber("card_area", note="membership in a gs card list"),
    "base": Stored("base"),
    "base_cost": Stored("base_cost"),
    "config": Stored("center_key", note="config.center is the prototype, found by key"),
    "cost": Stored("cost"),
    "debuff": Stored("debuff"),
    "destroyed": Stored("destroyed", note="writer: Phase 3 lifecycle"),
    "edition": Stored("edition"),
    "eligible_editionless_jokers": Grabber("editionless_jokers", note="D39"),
    "eligible_strength_jokers": Grabber("editionless_jokers", note="Wheel; D39"),
    "extra_cost": Stored("extra_cost"),
    "facing": Stored("facing"),
    "getting_sliced": Stored("getting_sliced", note="writer: Phase 4 (D15)"),
    "highlighted": OutOfScope("selection is passed explicitly with each action"),
    "lucky_trigger": Stored("lucky_trigger"),
    "pinned": _CHALLENGE,
    "playing_card": Stored("playing_card"),
    "rank": OutOfScope("CardArea position; the engine uses list order"),
    "removed": Stored("removed", note="writer: Phase 3 lifecycle"),
    "seal": Stored("seal"),
    "sell_cost": Stored("sell_cost"),
    "shattered": Stored("shattered", note="writer: Phase 5 (D17)"),
    "sort_id": Stored("sort_id"),
    "unique_val": Stored("unique_val", note="order-preserving property over sort_id"),
}
for _ui_field in (
    "ability_UIBox_table",
    "ambient_tilt",
    "back_overlay",
    "bypass_discovery_center",
    "bypass_discovery_ui",
    "bypass_lock",
    "children",
    "click_timeout",
    "CT",
    "discard_pos",
    "dissolve",
    "dissolve_colours",
    "flipping",
    "hover_tilt",
    "juice",
    "label",
    "layered_parallax",
    "mouse_damping",
    "no_ui",
    "opening",
    "params",
    "parent",
    "sell_cost_label",
    "shadow_height",
    "sprite_facing",
    "sticker_run",
    "tilt_var",
    "zoom",
):
    CARD_FIELDS[_ui_field] = _UI

# ---------------------------------------------------------------------------
# Blind fields (``self.<x> =`` in blind.lua)
# ---------------------------------------------------------------------------

BLIND_FIELDS: dict[str, Entry] = {
    "blind_set": _UI,
    "block_play": OutOfScope("UI: blocks input while the debuff alert animates"),
    "boss": Stored("boss"),
    "chips": Stored("chips"),
    "config": Stored("key", note="config.blind is the prototype, found by key"),
    "debuff": Stored("debuff_config"),
    "disabled": Stored("disabled"),
    "discards_sub": Stored("discards_sub"),
    "dollars": Stored("dollars"),
    "hands": Stored("hands_used", note="The Eye history; Lua names it self.hands"),
    "hands_sub": Stored("hands_sub"),
    "mult": Stored("mult"),
    "name": Stored("name"),
    "only_hand": Stored("only_hand"),
    "prepped": Stored("prepped"),
    "triggered": Stored("triggered"),
}
for _ui_field in (
    "ambient_tilt",
    "children",
    "chip_text",
    "colour",
    "dark_colour",
    "dissolve",
    "dissolve_colours",
    "hover_tilt",
    "hovering",
    "loc_debuff_lines",
    "loc_debuff_text",
    "loc_name",
    "pos",
    "shadow_height",
    "sound_pings",
    "tilt_var",
    "zoom",
):
    BLIND_FIELDS[_ui_field] = _UI

# ---------------------------------------------------------------------------
# card.ability keys Lua reads or writes
# ---------------------------------------------------------------------------
# ``Card.ability`` is an untyped dict, so a Stored ability key means "held in
# ``card.ability[key]``". Most come from the center's config via set_ability.

ABILITY_KEYS: dict[str, Entry] = {
    k: Stored(f"ability.{k}")
    for k in (
        "bonus",
        "burnt_hand",
        "caino_xmult",
        "consumeable",
        "couponed",
        "d_size",
        "discarded",
        "effect",
        "eternal",
        "extra",
        "extra_value",
        "forced_selection",
        "h_dollars",
        "h_mult",
        "h_size",
        "h_x_mult",
        "hands_played_at_create",
        "invis_rounds",
        "loyalty_remaining",
        "mult",
        "name",
        "order",
        "p_dollars",
        "perish_tally",
        "perishable",
        "perma_bonus",
        "perma_debuff",
        "played_this_ante",
        "queue_negative_removal",
        "rental",
        "set",
        "t_chips",
        "t_mult",
        "to_do_poker_hand",
        "type",
        "wheel_flipped",
        "x_mult",
        "yorick_discards",
    )
}
ABILITY_KEYS.update(
    {
        "blind_type": Stored(
            "awarded_tags[*].blind_type", lazy=True, note="Tag ability, Orbital Tag (D54)"
        ),
        "orbital_hand": Stored(
            "awarded_tags[*].orbital_hand", lazy=True, note="Tag ability, Orbital Tag (D54)"
        ),
        "blueprint_compat": Grabber("resolve_copy_targets"),
        "blueprint_compat_check": _UI,
        "blueprint_compat_ui": _UI,
        "booster_pos": OutOfScope("shop save/reload slot marker for used_packs"),
        "driver_tally": Grabber("enhanced_count", note="Driver's License"),
        "money": Grabber("temperance_money", note="Temperance, per-frame in Card:update"),
        "nine_tally": Grabber("rank_count", note="Cloud 9 (D19)"),
        "steel_tally": Grabber("count_enhancement", note="Steel Joker"),
        "stone_tally": Grabber("count_enhancement", note="Stone Joker"),
    }
)
