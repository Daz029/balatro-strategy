"""Game step function — the heart of the simulator.

Applies a player :data:`~jackdaw.engine.actions.Action` to the game state
and advances the phase.  Each handler validates legality, executes the
action, and transitions to the next phase as needed.

Usage::

    from jackdaw.engine.game import step
    from jackdaw.engine.actions import SelectBlind

    game_state = initialize_run("b_red", 1, "SEED")
    game_state["phase"] = GamePhase.BLIND_SELECT
    game_state["blind_on_deck"] = "Small"

    game_state = step(game_state, SelectBlind())
    assert game_state["phase"] == GamePhase.SELECTING_HAND
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine import lifecycle, read
from jackdaw.engine.actions import (
    Action,
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    RedeemVoucher,
    Reroll,
    SelectBlind,
    SellCard,
    SkipBlind,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
)
from jackdaw.engine.effects import (
    AddTag,
    CreateCard,
    EaseDollars,
    EffectQueue,
    LevelUpHand,
    apply_effects,
)


class IllegalActionError(Exception):
    """Raised when an action is not valid in the current game state."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def step(game_state: dict[str, Any], action: Action) -> dict[str, Any]:
    """Apply *action* to *game_state* in-place and return it.

    Dispatches based on action type.  Raises :class:`IllegalActionError`
    if the action is not valid in the current phase.
    """
    match action:
        case SelectBlind(allow_forced_boss=allow_forced_boss):
            return _handle_select_blind(game_state, allow_forced_boss=allow_forced_boss)
        case SkipBlind():
            return _handle_skip_blind(game_state)
        case PlayHand(card_indices=indices):
            return _handle_play_hand(game_state, indices)
        case Discard(card_indices=indices):
            return _handle_discard(game_state, indices)
        case CashOut():
            return _handle_cash_out(game_state)
        case BuyCard(shop_index=idx):
            return _handle_buy_card(game_state, idx)
        case SellCard(area=area, card_index=idx):
            return _handle_sell_card(game_state, area, idx)
        case UseConsumable(card_index=idx, target_indices=targets):
            return _handle_use_consumable(game_state, idx, targets)
        case RedeemVoucher(card_index=idx):
            return _handle_redeem_voucher(game_state, idx)
        case OpenBooster(card_index=idx):
            return _handle_open_booster(game_state, idx)
        case PickPackCard(card_index=idx, target_indices=targets):
            return _handle_pick_pack_card(game_state, idx, targets)
        case SkipPack():
            return _handle_skip_pack(game_state)
        case Reroll():
            return _handle_reroll(game_state)
        case NextRound():
            return _handle_next_round(game_state)
        case SortHand(mode=mode):
            return _handle_sort_hand(game_state, mode)
        case SwapHandLeft(idx=idx):
            return _handle_swap_hand(game_state, idx, -1)
        case SwapHandRight(idx=idx):
            return _handle_swap_hand(game_state, idx, +1)
        case SwapJokersLeft(idx=idx):
            return _handle_swap_jokers(game_state, idx, -1)
        case SwapJokersRight(idx=idx):
            return _handle_swap_jokers(game_state, idx, +1)
        case _:
            raise IllegalActionError(f"Unknown action type: {type(action).__name__}")


# ---------------------------------------------------------------------------
# Phase validation helper
# ---------------------------------------------------------------------------


def _require_phase(gs: dict[str, Any], *phases: GamePhase) -> GamePhase:
    """Assert the current phase is one of *phases* and return it."""
    raw = gs.get("phase")
    phase = GamePhase(raw) if isinstance(raw, str) else raw
    if phase not in phases:
        raise IllegalActionError(f"Action not valid in phase {phase!r} (expected {phases})")
    return phase


def _gain_joker(gs: dict[str, Any], card: Any) -> bool:
    """Thin compatibility wrapper over :func:`lifecycle.emplace`."""
    jokers: list = gs.setdefault("jokers", [])
    if any(owned is card for owned in jokers):
        return False
    lifecycle.emplace(gs, card, "jokers")
    return True


def _lose_joker(gs: dict[str, Any], card: Any) -> bool:
    """Thin compatibility wrapper over :func:`lifecycle.remove`."""
    jokers: list = gs.get("jokers", [])
    for owned in jokers:
        if owned is card:
            lifecycle.remove(gs, owned)
            return True
    return False


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_select_blind(gs: dict[str, Any], *, allow_forced_boss: bool = False) -> dict[str, Any]:
    """Accept the current blind and start the round.

    Full sequence matching ``game.lua`` select_blind → ``blind.lua``
    ``set_blind`` → ``state_events.lua`` ``new_round``:

    1. Create Blind from blind_choices
    2. Call ``start_round`` (reset counters, targeting cards)
    3. Apply boss blind set-time effects (Water, Needle, Manacle,
       Amber Acorn)
    4. Fire and apply joker ``setting_blind`` effects
    5. Debuff playing cards
    6. Draw hand from deck
    7. Set phase → SELECTING_HAND
    """
    _require_phase(gs, GamePhase.BLIND_SELECT)

    from jackdaw.engine.blind import Blind
    from jackdaw.engine.run_init import start_round

    blind_on_deck = gs.get("blind_on_deck", "Small")
    rr = gs["round_resets"]
    blind_key = rr["blind_choices"].get(blind_on_deck, "bl_small")

    # ------------------------------------------------------------------
    # 1. Create the active Blind
    # ------------------------------------------------------------------
    ante = rr["ante"]
    # Boss eligibility guard (plan Part 4, C08 note): catches forced/sampled
    # bosses that get_new_boss (common_events.lua:2353) could never produce.
    # Lua picks the boss when the ante starts; Hieroglyph / Petroglyph
    # (vouchers.py, card.lua:1958 ease_ante(-1)) can lower the ante AFTER
    # that, so a legitimately chosen boss may face a lower ante here. The
    # guard is therefore skipped once an ante-lowering voucher is redeemed.
    ante_lowered = read.has_voucher(gs, "v_hieroglyph") or read.has_voucher(gs, "v_petroglyph")
    if blind_on_deck == "Boss" and not allow_forced_boss and not ante_lowered:
        from jackdaw.engine.data.prototypes import BLINDS

        boss = BLINDS[blind_key].boss or {}
        eff_ante = max(1, ante)
        showdown_ante = eff_ante % gs.get("win_ante", 8) == 0 and ante >= 2
        assert bool(boss.get("showdown")) == showdown_ante, (
            f"{blind_key} showdown eligibility does not match ante {ante}"
        )
        if not boss.get("showdown"):
            assert boss.get("min", 1) <= eff_ante, (
                f"{blind_key} requires ante {boss.get('min', 1)}, got {ante}"
            )
    scaling = gs.get("modifiers", {}).get("scaling", 1)
    ante_scaling = gs["starting_params"].get("ante_scaling", 1.0)
    no_reward = gs.get("modifiers", {}).get("no_blind_reward", {})
    blind = Blind.create(
        blind_key,
        ante,
        scaling=scaling,
        ante_scaling=ante_scaling,
        no_blind_reward=bool(no_reward.get(blind_on_deck)),
    )
    gs["blind"] = blind
    # Lua sets this as soon as select_blind begins (button_callbacks.lua:2516).
    gs["facing_blind"] = True
    gs["chips"] = 0
    rr["blind_states"][blind_on_deck] = "Current"
    rr["blind"] = blind

    # ------------------------------------------------------------------
    # 2. Start round (reset counters, targeting cards)
    #    Must run BEFORE boss effects so that The Water/Needle/etc.
    #    can decrement from the freshly-set values.
    # ------------------------------------------------------------------
    # Juggle Tag (round_start_bonus): request the one-round hand-size
    # bonus BEFORE start_round, which applies rr["temp_handsize"] to
    # hand_size and records the amount for the end-of-round revert.
    from jackdaw.engine.tags import fire_tag_context

    for _entry, tag_res in fire_tag_context(gs, "round_start_bonus"):
        if tag_res.hand_size_delta:
            rr["temp_handsize"] = (rr.get("temp_handsize") or 0) + tag_res.hand_size_delta

    start_round(gs)

    # ------------------------------------------------------------------
    # 3. Boss blind set-time effects (blind.lua:157-209)
    #    In Lua, set_blind fires inside new_round BEFORE the shuffle.
    #    Order: set_blind → joker setting_blind → shuffle → draw.
    # ------------------------------------------------------------------
    if blind.boss:
        _apply_boss_blind_effects(gs, blind)

    # ------------------------------------------------------------------
    # 4. Joker setting_blind pass. Lua runs this after Blind:set_blind;
    # Chicot therefore reverses any boss set-time effects it disables.
    # ------------------------------------------------------------------
    _fire_setting_blind(gs)

    # Debuff playing cards based on boss blind
    deck: list = gs.get("deck", [])
    active_rules = read.rules(gs)
    for card in deck:
        blind.debuff_card(card, active_rules, gs)

    # ------------------------------------------------------------------
    # 6. Per-round deck shuffle (state_events.lua:344)
    #    Fires AFTER set_blind and joker setting_blind context,
    #    BEFORE draw_to_hand.  Key: 'nr' + str(ante).
    # ------------------------------------------------------------------
    rng = gs.get("rng")
    if rng:
        deck_list: list = gs.get("deck", [])
        nr_seed = rng.seed("nr" + str(ante))
        rng.shuffle(deck_list, nr_seed)

    # ------------------------------------------------------------------
    # 7. Draw hand from deck
    # ------------------------------------------------------------------
    _draw_hand(gs)
    # Debuff hand cards too (they were drawn from the deck)
    for card in gs.get("hand", []):
        blind.debuff_card(card, active_rules, gs)

    # game.lua:3226-3231: the first-hand pass fires after the draw and card
    # debuffs, while facing the blind, before Blind:drawn_to_hand.
    cr = gs.get("current_round", {})
    if (
        cr.get("hands_played", 0) == 0
        and cr.get("discards_used", 0) == 0
        and gs.get("facing_blind")
    ):
        from jackdaw.engine.jokers import fire_jokers

        queue = EffectQueue(gs)
        fire_jokers(gs, queue, first_hand_drawn=True)
        queue.apply()

    # ------------------------------------------------------------------
    # 7b. Boss drawn_to_hand effects (Cerulean Bell, Crimson Heart)
    # ------------------------------------------------------------------
    if blind.boss and not blind.disabled:
        dth = blind.drawn_to_hand(
            hand_cards=gs.get("hand", []),
            joker_cards=gs.get("jokers"),
            rng=rng,
            gs=gs,
        )
        if dth.get("forced_card_index") is not None:
            hand = gs.get("hand", [])
            idx = dth["forced_card_index"]
            if 0 <= idx < len(hand):
                hand[idx].ability["forced_selection"] = True

    # ------------------------------------------------------------------
    # 8. Phase → SELECTING_HAND
    # ------------------------------------------------------------------
    gs["phase"] = GamePhase.SELECTING_HAND
    return gs


def _fire_new_blind_choice_tags(gs: dict[str, Any]) -> None:
    """Fire new_blind_choice context for any awarded tags that need it.

    In Lua, new_blind_choice tags fire when entering the blind select screen
    after a skip.  Pack-creating tags (buffoon, charm, ethereal, meteor,
    standard) open a pack for the player to pick from.  The boss tag rerolls
    the boss blind.

    Since the engine has no interactive player, pack-creating tags populate
    ``gs["pack_cards"]`` and set the phase to PACK_OPENING so the caller
    (or agent) can pick or skip.
    """
    from jackdaw.engine.tags import Tag

    awarded: list[dict] = gs.get("awarded_tags", [])
    rng = gs.get("rng")
    rr = gs.get("round_resets", {})

    for entry in awarded:
        tag_key = entry.get("key", "")
        # Only fire tags that haven't been processed for new_blind_choice yet
        if entry.get("nbc_fired") or entry.get("consumed"):
            continue

        tag = Tag(tag_key)
        result = tag.apply("new_blind_choice", gs, rng=rng)
        entry["nbc_fired"] = True

        if result is None:
            continue

        # Pack/boss tags are single-use: delivered here, consumed here.
        entry["consumed"] = True
        entry["consumed_context"] = "new_blind_choice"

        if result.reroll_boss:
            from jackdaw.engine.blind import get_new_boss

            bosses_used = gs.setdefault("bosses_used", {})
            ante = rr.get("ante", 1)
            new_boss = get_new_boss(ante, bosses_used, rng)
            rr.setdefault("blind_choices", {})["Boss"] = new_boss

        if result.create_pack:
            if gs.get("phase") == GamePhase.PACK_OPENING:
                gs.setdefault("pending_tag_packs", []).append(result.create_pack)
            else:
                _open_tag_pack(gs, result.create_pack)


def _open_tag_pack(gs: dict[str, Any], pack_key: str) -> None:
    """Open a pack from a tag reward, populating pack_cards.

    The caller (or agent) must then pick from the pack or skip it.
    For the engine-only path (no interactive player), we store the pack
    state so that get_legal_actions returns PickPackCard/SkipPack options.
    """
    from jackdaw.engine.data.prototypes import BOOSTERS
    from jackdaw.engine.packs import generate_pack_cards

    rng = gs.get("rng")
    ante = gs.get("round_resets", {}).get("ante", 1)

    pack_cards, choices = generate_pack_cards(pack_key, rng, ante, gs)
    gs["pack_cards"] = []
    for card in pack_cards:
        lifecycle.emplace(gs, card, "pack_cards")
    gs["pack_choices_remaining"] = choices

    # Determine pack kind from prototype
    proto = BOOSTERS.get(pack_key)
    pack_kind = proto.kind if proto else ""
    gs["pack_type"] = pack_kind

    # Tag-awarded packs still run through Card:open in Lua (card.lua:1797).
    from jackdaw.engine.jokers import fire_jokers

    queue = EffectQueue(gs)
    fire_jokers(gs, queue, open_booster=True)
    queue.apply()

    # Save current phase and switch to pack opening
    gs["shop_return_phase"] = gs.get("phase", GamePhase.BLIND_SELECT)

    # For Arcana/Spectral packs: deal hand from deck for targeting
    # Matches _handle_open_booster behavior
    if pack_kind in ("Arcana", "Spectral"):
        deck: list = gs.get("deck", [])
        hand: list = gs.get("hand", [])
        hand_size = gs.get("hand_size", 8)
        to_deal = min(len(deck), hand_size - len(hand))
        pack_hand: list = []
        for _ in range(to_deal):
            if deck:
                card = deck.pop()
                pack_hand.append(card)
        gs["pack_hand"] = pack_hand
        combined_hand = hand + pack_hand
        _sort_hand_desc(combined_hand)
        gs["hand"] = combined_hand

    gs["phase"] = GamePhase.PACK_OPENING


def _apply_tag_result(gs: dict[str, Any], result: Any) -> None:
    """Apply a TagResult's effects to the game state.

    Handles all TagResult fields that produce immediate side-effects.
    """
    queue = EffectQueue(gs)
    if result.dollars:
        queue.add(EaseDollars(amount=result.dollars))
    if result.create_jokers:
        queue.add(
            CreateCard(
                set="Joker",
                count=result.create_jokers,
                rarity="Common",
                append="top",
            )
        )
    if result.level_up is not None:
        hand_type, levels = result.level_up
        queue.add(LevelUpHand(hand=hand_type, amount=levels))
    queue.apply()

    # Context-owned fields: create_pack/reroll_boss are consumed by
    # _fire_new_blind_choice_tags; force_rarity/force_edition by shop card
    # creation; temp_reroll_cost/coupon/create_voucher by shop setup;
    # hand_size_delta by _handle_select_blind; double by _check_double_tag.


def _handle_skip_blind(gs: dict[str, Any]) -> dict[str, Any]:
    """Skip the current blind (Small or Big) and advance.

    Full sequence matching ``button_callbacks.lua:2740-2775``:

    1. Validate: not Boss
    2. Increment skips
    3. Award skip tag from ``blind_tags``
    4. Fire tag ``apply('immediate')`` for immediate tags
    5. Fire joker ``skip_blind`` context (Throwback tracking)
    6. Advance ``blind_on_deck``: Small→Big, Big→Boss
    7. Check Double Tag: if active, duplicate the just-awarded tag
    8. Phase stays BLIND_SELECT
    """
    _require_phase(gs, GamePhase.BLIND_SELECT)

    blind_on_deck = gs.get("blind_on_deck", "Small")
    if blind_on_deck not in ("Small", "Big"):
        raise IllegalActionError("Cannot skip Boss blind")

    rr = gs["round_resets"]

    # ------------------------------------------------------------------
    # 1-2. Increment skips
    # ------------------------------------------------------------------
    gs["skips"] = gs.get("skips", 0) + 1
    rr["blind_states"][blind_on_deck] = "Skipped"

    # ------------------------------------------------------------------
    # 3. Award skip tag
    # ------------------------------------------------------------------
    blind_tags = rr.get("blind_tags", {})
    tag_key = blind_tags.get(blind_on_deck)
    awarded_tags: list[dict[str, Any]] = gs.setdefault("awarded_tags", [])

    awarded_entry: dict[str, Any] | None = None
    if tag_key:
        from jackdaw.engine.tags import Tag

        tag = Tag(tag_key)
        tag_result = tag.apply("immediate", gs, rng=gs.get("rng"))

        awarded_entry = {
            "key": tag_key,
            "result": tag_result,
            "blind": blind_on_deck,
        }
        awarded_tags.append(awarded_entry)

        # Apply immediate tag effects; immediate tags are single-use and
        # deliver right here, so mark them consumed. Deferred tags (shop,
        # eval, round-start contexts) stay un-consumed for later polls.
        if tag_result is not None:
            _apply_tag_result(gs, tag_result)
            awarded_entry["consumed"] = True
            awarded_entry["consumed_context"] = "immediate"

    # ------------------------------------------------------------------
    # 4. Fire joker skip_blind context
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import fire_jokers

    queue = EffectQueue(gs)
    fire_jokers(gs, queue, skip_blind=True)
    queue.apply()

    # ------------------------------------------------------------------
    # 5. Advance blind_on_deck
    # ------------------------------------------------------------------
    if blind_on_deck == "Small":
        gs["blind_on_deck"] = "Big"
        rr["blind_states"]["Big"] = "Select"
    else:
        gs["blind_on_deck"] = "Boss"
        rr["blind_states"]["Boss"] = "Select"

    # ------------------------------------------------------------------
    # 6. Double Tag check
    # ------------------------------------------------------------------
    if awarded_entry is not None:
        _check_double_tag(gs, awarded_entry)

    # ------------------------------------------------------------------
    # 7. Fire new_blind_choice tags from awarded (deferred) tags
    # ------------------------------------------------------------------
    # In Lua, new_blind_choice tags fire when the blind select screen
    # appears after a skip.  Tags like tag_buffoon, tag_charm, etc.
    # open packs; tag_boss rerolls the boss blind.
    _fire_new_blind_choice_tags(gs)

    # Only return to BLIND_SELECT if a tag didn't open a pack
    if gs.get("phase") != GamePhase.PACK_OPENING:
        gs["phase"] = GamePhase.BLIND_SELECT
    return gs


def _handle_play_hand(gs: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    """Play cards from the hand, score them, and check if blind is beaten.

    Full sequence matching ``state_events.lua`` play_cards_from_highlighted
    → evaluate_play:

    1. Validate indices
    2. Move cards from hand to play area (preserve index order)
    3. Decrement hands_left, increment hands_played
    4. Update per-card stats (times_played, played_this_ante)
    5. Fire ``Blind:press_play`` (The Hook, The Tooth)
    6. Call ``score_hand`` (full 14-phase pipeline)
    7. Process scoring side-effects (dollars, card destruction,
       joker removal)
    8. Move surviving played cards to discard pile
    9. Record hand type in hand_levels
    10. Determine next phase: won / continue / game over
    11. If continuing: draw cards, re-debuff for boss
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    # state_events.lua:455 clears this before each play. Without the reset,
    # Matador would keep paying after one debuffed scoring card triggers it.
    blind = gs["blind"]
    blind.triggered = False

    cr = gs["current_round"]
    if cr["hands_left"] <= 0:
        raise IllegalActionError("No hands remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot play more than 5 cards")
    if any(i < 0 or i >= len(hand) for i in indices):
        raise IllegalActionError("Card index out of range")

    # ------------------------------------------------------------------
    # 2. Move cards from hand to play area
    # ------------------------------------------------------------------
    # Preserve SELECTION ORDER (not hand position order).
    # In Balatro, cards are placed left-to-right in click order.
    # The first index in card_indices is the leftmost scored card.
    idx_set = set(indices)
    played = [hand[i] for i in indices]
    held = [c for i, c in enumerate(hand) if i not in idx_set]
    gs["hand"] = held
    gs["played_cards_area"] = played

    # ------------------------------------------------------------------
    # 3. Decrement hands_left
    # ------------------------------------------------------------------
    cr["hands_left"] -= 1

    # ------------------------------------------------------------------
    # 4. Per-card stats
    # ------------------------------------------------------------------
    for card in played:
        base = getattr(card, "base", None)
        if base is not None:
            base.times_played = getattr(base, "times_played", 0) + 1
        ability = getattr(card, "ability", None)
        if isinstance(ability, dict):
            ability["played_this_ante"] = True

    # ------------------------------------------------------------------
    # 5. Blind:press_play (blind.lua:464)
    # ------------------------------------------------------------------
    rng = gs.get("rng")
    _press_play(gs, blind, played, rng)

    # ------------------------------------------------------------------
    # 6. Score the hand (full 14-phase pipeline)
    # ------------------------------------------------------------------
    from jackdaw.engine.scoring import score_hand

    jokers = gs.get("jokers", [])
    hand_levels = gs.get("hand_levels")

    # Lua writes this at the start of evaluate_play, once the played hand
    # type is known and before any scoring effects run (state_events.lua:576).
    from jackdaw.engine.hand_eval import evaluate_hand

    gs["last_hand_played"] = evaluate_hand(played, jokers=jokers).detected_hand

    result = score_hand(
        played_cards=played,
        held_cards=held,
        jokers=jokers,
        hand_levels=hand_levels,
        blind=blind,
        rng=rng,
        probabilities_normal=gs.get("probabilities", {}).get("normal", 1),
        game_state=gs,
        back_key=gs.get("selected_back_key"),
        blind_chips=blind.chips,
    )

    # Lua increments these after evaluate_play has returned, so the joker
    # contexts during scoring still observe the pre-hand counts.
    cr["hands_played"] += 1
    gs["hands_played"] = gs.get("hands_played", 0) + 1

    # ------------------------------------------------------------------
    # 7. Process scoring side-effects
    # ------------------------------------------------------------------
    # Accumulate chips
    gs["chips"] = gs.get("chips", 0) + result.total
    gs["last_score_result"] = result

    # Dollars from scoring (Gold Seal, Lucky Card, joker economy)
    if result.dollars_earned:
        gs["dollars"] = gs.get("dollars", 0) + result.dollars_earned

    # score_hand marks the destroyed cards (shattered/destroyed) and fires the
    # destruction notification at Lua's scoring-time observation point; the
    # canonical removal happens here, on the live state only.
    lifecycle.destroy_playing_cards(gs, result.cards_destroyed, notify=False)
    destroyed_set = set(id(c) for c in result.cards_destroyed)
    played = [c for c in played if id(c) not in destroyed_set]

    apply_effects(gs, result.effects)

    # The Ox: set money to $0 if most-played hand type is played
    # (blind.lua:debuff_hand fires during scoring in Lua)
    if (
        getattr(blind, "name", "") == "The Ox"
        and not getattr(blind, "disabled", False)
        and hand_levels is not None
        and result.hand_type != "NULL"
    ):
        from jackdaw.engine.data.hands import HandType as _HT

        try:
            played_ht = _HT(result.hand_type)
            if played_ht == hand_levels.most_played():
                blind.triggered = True
                gs["dollars"] = 0
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # 8. Move surviving played cards to discard pile
    #
    # In Lua, draw_from_play_to_discard (state_events.lua:522, 1088-1096)
    # moves played cards to the discard area after scoring.
    # ------------------------------------------------------------------
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(played)
    gs["played_cards_area"] = []

    # ------------------------------------------------------------------
    # 10. Determine next phase
    # ------------------------------------------------------------------
    if gs["chips"] >= blind.chips:
        _round_won(gs)
    elif cr["hands_left"] <= 0:
        if not result.saved:
            gs["phase"] = GamePhase.GAME_OVER
            gs["won"] = False
        else:
            _round_won(gs)
    else:
        # ------------------------------------------------------------------
        # 11. More hands — draw cards and stay in SELECTING_HAND
        # ------------------------------------------------------------------
        # The Serpent: draw only 3 cards instead of filling to hand_size
        serpent_play = getattr(blind, "name", "") == "The Serpent" and not getattr(
            blind, "disabled", False
        )
        if serpent_play:
            deck: list = gs.get("deck", [])
            hand_out: list = gs.get("hand", [])
            for _ in range(min(3, len(deck))):
                if deck:
                    hand_out.append(deck.pop())
            _sort_hand_desc(hand_out)
        else:
            _draw_hand(gs)

        # Re-debuff hand cards for boss blind (new cards from deck)
        if blind.boss and not blind.disabled:
            active_rules = read.rules(gs)
            for card in gs.get("hand", []):
                blind.debuff_card(card, active_rules, gs)

        # The Fish: flip newly drawn cards face-down
        if getattr(blind, "name", "") == "The Fish" and getattr(blind, "prepped", False):
            for card in gs.get("hand", []):
                card.facing = "back"

        # Boss drawn_to_hand effects on redraw (Cerulean Bell, Crimson Heart)
        if blind.boss and not blind.disabled:
            dth = blind.drawn_to_hand(
                hand_cards=gs.get("hand", []),
                joker_cards=jokers,
                rng=rng,
                gs=gs,
            )
            if dth.get("forced_card_index") is not None:
                hand = gs.get("hand", [])
                idx = dth["forced_card_index"]
                if 0 <= idx < len(hand):
                    hand[idx].ability["forced_selection"] = True

        _end_round_if_hand_empty(gs)

    return gs


def _handle_discard(gs: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
    """Discard highlighted cards, fire joker contexts, draw replacements.

    Full sequence matching ``state_events.lua:379-448``:

    1. Validate
    2. Sort discarded cards (left-to-right by index)
    3. Fire joker ``pre_discard`` context (Burnt Joker)
    4. Per-card: fire seal effects (Purple Seal → Tarot) + joker ``discard``
       context with ``other_card`` + ``full_hand``
    5. Process side-effects (dollars, card destruction, joker mutations)
    6. Discard cost (Golden Needle challenge)
    7. Decrement discards_left, increment discards_used
    8. Move surviving cards to discard pile
    9. Draw replacements from deck
    10. The Serpent: draw only 3 if not first action
    11. Re-debuff drawn cards for boss blind
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    cr = gs["current_round"]
    if cr["discards_left"] <= 0:
        raise IllegalActionError("No discards remaining")

    hand: list = gs.get("hand", [])
    if not indices or not hand:
        raise IllegalActionError("Must select at least 1 card")
    if len(indices) > 5:
        raise IllegalActionError("Cannot discard more than 5 cards")
    if any(i < 0 or i >= len(hand) for i in indices):
        raise IllegalActionError("Card index out of range")

    # ------------------------------------------------------------------
    # 2. Extract discarded cards in sorted order
    # ------------------------------------------------------------------
    idx_set = set(indices)
    discarded = [hand[i] for i in sorted(indices)]
    gs["hand"] = [c for i, c in enumerate(hand) if i not in idx_set]

    # ------------------------------------------------------------------
    # 3. Fire joker pre_discard context (Burnt Joker: level up hand)
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import calculate_joker, context_for

    jokers: list = gs.get("jokers", [])
    queue = EffectQueue(gs)

    pre_discard_effects: list = []
    for joker in jokers:
        if getattr(joker, "debuff", False):
            continue
        ctx = context_for(
            gs,
            queue=queue,
            pre_discard=True,
            full_hand=discarded,
        )
        result = calculate_joker(joker, ctx)
        if result:
            pre_discard_effects.append(result)

    # Burnt Joker: level up the hand type of discarded cards
    for eff in pre_discard_effects:
        if eff.level_up:
            hand_levels = gs.get("hand_levels")
            if hand_levels is not None:
                from jackdaw.engine.hand_eval import evaluate_hand

                det = evaluate_hand(discarded)
                if det.detected_hand and det.detected_hand != "NULL":
                    hand_levels.level_up(det.detected_hand)

    # ------------------------------------------------------------------
    # 4. Per-card: seal effects + joker discard context
    # ------------------------------------------------------------------
    dollars_earned = 0
    destroyed: list = []

    for card in discarded:
        # Seal: Purple Seal → create random Tarot with append '8ba'
        # (card.lua:2254-2260; slot check gates the roll so no RNG is
        # consumed when consumable slots are full)
        if getattr(card, "seal", None) == "Purple":
            queue.add(CreateCard(set="Tarot", append="8ba"))

        # Fire joker discard context per card
        card_destroyed = False
        for joker in jokers:
            if getattr(joker, "debuff", False):
                continue
            ctx = context_for(
                gs,
                queue=queue,
                discard=True,
                other_card=card,
                full_hand=discarded,
            )
            result = calculate_joker(joker, ctx)
            if result:
                dollars_earned += result.dollars
                if result.level_up:
                    # Burnt Joker: level up the discard hand type
                    hl = gs.get("hand_levels")
                    if hl is not None:
                        from jackdaw.engine.hand_eval import evaluate_hand as _eval

                        det = _eval(discarded)
                        if det.detected_hand and det.detected_hand != "NULL":
                            hl.level_up(det.detected_hand)
                if result.remove:
                    card_destroyed = True

        if card_destroyed:
            destroyed.append(card)

    # ------------------------------------------------------------------
    # 5. Process side-effects
    # ------------------------------------------------------------------
    if dollars_earned:
        gs["dollars"] = gs.get("dollars", 0) + dollars_earned

    lifecycle.destroy_playing_cards(gs, destroyed)
    queue.apply()

    # ------------------------------------------------------------------
    # 6. Discard cost (Golden Needle challenge)
    # ------------------------------------------------------------------
    discard_cost = gs.get("modifiers", {}).get("discard_cost", 0)
    if discard_cost > 0:
        gs["dollars"] = gs.get("dollars", 0) - discard_cost

    # ------------------------------------------------------------------
    # 7. Decrement discards_left, increment discards_used
    # ------------------------------------------------------------------
    cr["discards_left"] -= 1
    cr["discards_used"] += 1

    # ------------------------------------------------------------------
    # 8. Move surviving cards to discard pile
    # ------------------------------------------------------------------
    surviving = [c for c in discarded if c not in destroyed]
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(surviving)

    # Track stat
    gs["round_scores"] = gs.get("round_scores", {})
    gs["round_scores"]["cards_discarded"] = gs["round_scores"].get("cards_discarded", 0) + len(
        discarded
    )

    # ------------------------------------------------------------------
    # 9-10. Draw replacements from deck
    # ------------------------------------------------------------------
    blind = gs.get("blind")
    serpent = (
        blind is not None
        and getattr(blind, "name", "") == "The Serpent"
        and not getattr(blind, "disabled", False)
        and (cr.get("hands_played", 0) > 0 or cr.get("discards_used", 0) > 0)
    )
    if serpent:
        # The Serpent: draw only 3 after first action
        # Lua's draw_card(G.deck, G.hand) pops LAST card from deck
        deck: list = gs.get("deck", [])
        hand_out: list = gs.get("hand", [])
        for _ in range(min(3, len(deck))):
            if deck:
                hand_out.append(deck.pop())
        _sort_hand_desc(hand_out)
    else:
        _draw_hand(gs)

    # ------------------------------------------------------------------
    # 11. Re-debuff drawn cards for boss blind
    # ------------------------------------------------------------------
    if blind and getattr(blind, "boss", False) and not getattr(blind, "disabled", False):
        active_rules = read.rules(gs)
        for card in gs.get("hand", []):
            blind.debuff_card(card, active_rules, gs)

        # Boss drawn_to_hand effects on discard redraw
        rng = gs.get("rng")
        dth = blind.drawn_to_hand(
            hand_cards=gs.get("hand", []),
            joker_cards=jokers,
            rng=rng,
            gs=gs,
        )
        if dth.get("forced_card_index") is not None:
            hand = gs.get("hand", [])
            idx = dth["forced_card_index"]
            if 0 <= idx < len(hand):
                hand[idx].ability["forced_selection"] = True

    _end_round_if_hand_empty(gs)

    return gs


def _handle_cash_out(gs: dict[str, Any]) -> dict[str, Any]:
    """Accept round earnings and proceed to the shop.

    1. Shuffle deck (button_callbacks.lua:2918)
    2. Apply round earnings to dollars
    3. Track previous_round.dollars
    4. Populate shop (jokers, voucher, boosters)
    5. Phase → SHOP
    """
    _require_phase(gs, GamePhase.ROUND_EVAL)

    # End-of-round targeting-card re-roll (state_events.lua:273-276):
    # idol / mail / ancient / castle streams advance once per round END
    # (they are NOT re-rolled at round start; see start_round).
    rng = gs.get("rng")
    if rng:
        from jackdaw.engine.round_lifecycle import reset_round_targets

        ante = gs.get("round_resets", {}).get("ante", 1)
        reset_round_targets(rng, ante, gs)

    # Shuffle deck at cash-out (button_callbacks.lua:2918)
    # G.deck:shuffle('cashout'..G.GAME.round_resets.ante)
    if rng:
        deck: list = gs.get("deck", [])
        ante = gs.get("round_resets", {}).get("ante", 1)
        cashout_seed = rng.seed("cashout" + str(ante))
        rng.shuffle(deck, cashout_seed)

    earnings = gs.get("round_earnings")
    if earnings:
        gs["dollars"] = gs.get("dollars", 0) + earnings.total

    from jackdaw.engine.tags import fire_tag_context

    # Investment Tag (eval context): pays out at cash-out, but ONLY after a
    # Boss blind — the handler returns None otherwise, leaving the tag
    # un-consumed for the next boss.
    last_blind_is_boss = bool(getattr(gs.get("blind"), "boss", False))
    for _entry, tag_res in fire_tag_context(gs, "eval", last_blind_is_boss=last_blind_is_boss):
        _apply_tag_result(gs, tag_res)

    gs["previous_round"] = {"dollars": gs.get("dollars", 0)}

    # D6/Coupon are limited to one trigger per shop. Lua clears these just
    # before the new shop's tag contexts fire (button_callbacks.lua:2932).
    gs["shop_d6ed"] = False
    gs["shop_free"] = False

    # D6 Tag (shop_start context): rerolls start at $0 this shop. The temp
    # base is cleared by the NEXT round's start_round, so it covers exactly
    # this shop session.
    d6_fired = fire_tag_context(gs, "shop_start")
    for _entry, tag_res in d6_fired:
        if tag_res.temp_reroll_cost is not None:
            gs["round_resets"]["temp_reroll_cost"] = tag_res.temp_reroll_cost
    if d6_fired:
        from jackdaw.engine.run_init import _calculate_reroll_cost

        _calculate_reroll_cost(gs, skip_increment=True)

    # Populate shop
    _populate_shop(gs)

    gs["phase"] = GamePhase.SHOP
    return gs


def _handle_buy_card(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase a card from the shop.

    After buying:
    - Joker: add to jokers area, mark in used_jokers
    - Consumable: add to consumables area
    - Playing card: add to deck, fire ``playing_card_added`` joker context
    - Fire ``buying_card`` on all jokers
    """
    _require_phase(gs, GamePhase.SHOP)

    shop_cards: list = gs.get("shop_cards", [])
    if idx < 0 or idx >= len(shop_cards):
        raise IllegalActionError(f"Invalid shop index {idx}")

    card = shop_cards[idx]
    if card.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford card")

    gs["dollars"] -= card.cost
    shop_cards.pop(idx)
    gs["current_round"]["jokers_purchased"] = (
        gs.get("current_round", {}).get("jokers_purchased", 0) + 1
    )

    # Place card in appropriate area
    card_set = _get_card_set(card)
    if card_set == "Joker":
        _gain_joker(gs, card)
    elif card_set in ("Tarot", "Planet", "Spectral"):
        lifecycle.emplace(gs, card, "consumables")
    else:
        lifecycle.add_playing_cards(gs, [card], "deck", notify=False)

    # Fire buying_card joker context
    _fire_shop_joker_context(gs, buying_card=True)

    # Fire playing_card_added if a playing card was bought
    if card_set not in ("Joker", "Tarot", "Planet", "Spectral"):
        lifecycle.fire_joker_context(gs, playing_card_added=True, cards=[card])

    return gs


def _handle_sell_card(gs: dict[str, Any], area: str, idx: int) -> dict[str, Any]:
    """Sell through the single queued implementation in :mod:`shop`."""
    _require_phase(gs, GamePhase.SHOP)

    from jackdaw.engine.shop import sell_card

    result = sell_card(gs, area, idx)
    if not result["ok"]:
        reason = result["reason"]
        if reason == "eternal":
            raise IllegalActionError("Cannot sell eternal card")
        raise IllegalActionError(f"Cannot sell card: {reason}")

    return gs


def _handle_use_consumable(
    gs: dict[str, Any], idx: int, targets: tuple[int, ...] | None
) -> dict[str, Any]:
    """Use a consumable from the player's consumable slots.

    Consumables can be used in BLIND_SELECT, SELECTING_HAND,
    ROUND_EVAL, and SHOP phases.  The phase does NOT change after use.

    Sequence:
    1. Validate phase and index
    2. Pop card from consumables
    3. Use via ``_use_consumable_card`` (builds ConsumableContext,
       applies ConsumableResult effects, and fires the one
       ``using_consumeable`` joker pass)
    4. Track usage stats (last_tarot_planet)
    """
    _require_phase(
        gs, GamePhase.BLIND_SELECT, GamePhase.SELECTING_HAND, GamePhase.ROUND_EVAL, GamePhase.SHOP
    )

    consumables: list = gs.get("consumables", [])
    if idx < 0 or idx >= len(consumables):
        raise IllegalActionError(f"Invalid consumable index {idx}")

    card = consumables[idx]

    # Validate can_use before consuming (matches Balatro's can_use_consumeable)
    from jackdaw.engine.consumables import can_use_consumable

    hand: list = gs.get("hand", [])
    highlighted: list = []
    if targets:
        highlighted = [hand[i] for i in targets if i < len(hand)]

    if not can_use_consumable(
        card,
        highlighted=highlighted,
        hand_cards=hand,
        jokers=gs.get("jokers", []),
        consumables=consumables,
        joker_limit=gs.get("joker_slots", 5),
        consumable_limit=gs.get("consumable_slots", 2),
        game_state=gs,
    ):
        consumable_name = card.ability.get("name", card.center_key)
        raise IllegalActionError(f"Consumable {consumable_name!r} cannot be used at this time")

    consumables.pop(idx)
    _use_consumable_card(gs, card, targets)
    lifecycle.remove(gs, card)

    # Phase does NOT change — returns to whatever it was
    return gs


def _handle_redeem_voucher(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Purchase and activate a voucher."""
    _require_phase(gs, GamePhase.SHOP)

    vouchers: list = gs.get("shop_vouchers", [])
    if idx < 0 or idx >= len(vouchers):
        raise IllegalActionError(f"Invalid voucher index {idx}")

    card = vouchers[idx]
    if card.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford voucher")

    gs["dollars"] -= card.cost
    vouchers.pop(idx)

    from jackdaw.engine.vouchers import apply_voucher

    gs["used_vouchers"][card.center_key] = True
    apply_voucher(card.center_key, gs)
    lifecycle.remove(gs, card)

    # Clear the voucher slot so the next shop doesn't re-offer it.
    # Matches card.lua:1850: G.GAME.current_round.voucher = nil
    gs.get("current_round", {})["voucher"] = None

    return gs


def _handle_open_booster(gs: dict[str, Any], idx: int) -> dict[str, Any]:
    """Open a booster pack — generate cards and transition to PACK_OPENING.

    1. Deduct cost, remove pack from shop
    2. Generate pack cards via :func:`generate_pack_cards`
    3. Set ``pack_cards``, ``pack_choices_remaining``, ``pack_type``
    4. For Arcana/Spectral: deal hand from deck for targeting
    5. Fire ``open_booster`` joker context (Hallucination)
    6. Phase → PACK_OPENING
    """
    _require_phase(gs, GamePhase.SHOP)

    boosters: list = gs.get("shop_boosters", [])
    if idx < 0 or idx >= len(boosters):
        raise IllegalActionError(f"Invalid booster index {idx}")

    pack = boosters[idx]
    if pack.cost > gs.get("dollars", 0):
        raise IllegalActionError("Cannot afford booster")

    gs["dollars"] -= pack.cost
    boosters.pop(idx)
    lifecycle.remove(gs, pack)

    # Generate pack cards
    from jackdaw.engine.data.prototypes import BOOSTERS
    from jackdaw.engine.packs import generate_pack_cards

    pack_key = pack.center_key
    rng = gs.get("rng")
    ante = gs.get("round_resets", {}).get("ante", 1)

    if rng and pack_key in BOOSTERS:
        cards, choose = generate_pack_cards(pack_key, rng, ante, gs)
        gs["pack_cards"] = []
        for card in cards:
            lifecycle.emplace(gs, card, "pack_cards")
        gs["pack_choices_remaining"] = choose
        gs["pack_type"] = BOOSTERS[pack_key].kind
    else:
        gs["pack_cards"] = []
        gs["pack_choices_remaining"] = 1
        gs["pack_type"] = "Unknown"

    gs["shop_return_phase"] = GamePhase.SHOP

    # For Arcana/Spectral packs: deal hand from deck for targeting
    # Cards are drawn from the END of deck (top of visual stack),
    # matching Lua's draw_card(G.deck, G.hand) which pops last card.
    pack_kind = gs.get("pack_type", "")
    if pack_kind in ("Arcana", "Spectral"):
        deck: list = gs.get("deck", [])
        hand: list = gs.get("hand", [])
        hand_size = gs.get("hand_size", 8)
        to_deal = min(len(deck), hand_size - len(hand))
        pack_hand: list = []
        for _ in range(to_deal):
            if deck:
                card = deck.pop()
                pack_hand.append(card)
        gs["pack_hand"] = pack_hand
        # These cards serve as targets for Tarot/Spectral use
        combined_hand = hand + pack_hand
        _sort_hand_desc(combined_hand)
        gs["hand"] = combined_hand

    # Fire after pack generation (card.lua:1797), retaining the pack object
    # in context even though it has already left the shop area.
    _fire_shop_joker_context(gs, open_booster=True, booster=pack)

    gs["phase"] = GamePhase.PACK_OPENING
    return gs


def _handle_pick_pack_card(
    gs: dict[str, Any],
    idx: int,
    targets: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Pick a card from an opened booster pack.

    Matching ``button_callbacks.lua:2155-2247`` use_card:

    - **Consumable** (Arcana/Spectral/Celestial): use immediately via
      ``use_consumeable``.  For Arcana/Spectral, ``targets`` specifies
      which dealt hand cards the consumable should target.  Planets
      are used without targets (level up hand type).
    - **Playing card** (Standard pack): added to deck.  Fires
      ``playing_card_added`` joker context (Hologram).
    - **Joker** (Buffoon pack): added to joker slots.  Marks in
      ``used_jokers``.

    When ``pack_choices_remaining`` hits 0 or pack is empty, the pack
    closes: remaining cards are removed, dealt hand cards (if any)
    return to deck, and phase restores to SHOP.
    """
    _require_phase(gs, GamePhase.PACK_OPENING)

    pack_cards: list = gs.get("pack_cards", [])
    remaining = gs.get("pack_choices_remaining", 0)
    if remaining <= 0:
        raise IllegalActionError("No pack choices remaining")
    if idx < 0 or idx >= len(pack_cards):
        raise IllegalActionError(f"Invalid pack card index {idx}")

    card = pack_cards.pop(idx)
    gs["pack_choices_remaining"] = remaining - 1

    # Determine card type and handle accordingly
    card_set = _get_card_set(card)

    if card_set in ("Tarot", "Planet", "Spectral"):
        # Consumable: use immediately (Arcana/Spectral/Celestial pack)
        _use_consumable_card(gs, card, targets)
        lifecycle.remove(gs, card)

    elif card_set == "Joker":
        # Buffoon pack: add to joker slots
        _gain_joker(gs, card)

    else:
        # Standard pack: playing card → add to deck
        lifecycle.add_playing_cards(gs, [card], "deck")

    # Check if pack should close
    if gs["pack_choices_remaining"] <= 0 or not pack_cards:
        _close_pack(gs)

    return gs


def _handle_skip_pack(gs: dict[str, Any]) -> dict[str, Any]:
    """Skip remaining pack cards.

    Fires ``skipping_booster`` on all jokers (Red Card +mult per skip),
    then closes the pack.
    """
    _require_phase(gs, GamePhase.PACK_OPENING)

    # Fire skipping_booster joker context (Red Card +mult)
    _fire_shop_joker_context(gs, skipping_booster=True)

    _close_pack(gs)
    return gs


def _handle_reroll(gs: dict[str, Any]) -> dict[str, Any]:
    """Reroll the shop.

    After rerolling:
    - Fire ``reroll_shop`` on all jokers (Flash Card +mult)
    - Track times_rerolled stat
    """
    _require_phase(gs, GamePhase.SHOP)

    cr = gs.get("current_round", {})
    free = cr.get("free_rerolls", 0)
    cost = cr.get("reroll_cost", 5)

    if free > 0:
        cr["free_rerolls"] = free - 1
    elif gs.get("dollars", 0) >= cost:
        gs["dollars"] -= cost
    else:
        raise IllegalActionError("Cannot afford reroll")

    # Increment reroll cost for the next reroll. Uses the canonical
    # calculator (mirrors common_events.lua:2263) instead of the old inline
    # `base_reroll_cost + increase`, which silently ignored BOTH the
    # Reroll Surplus/Glut voucher discount (they mutate
    # round_resets.reroll_cost, not base_reroll_cost) AND the D6 Tag's
    # temp_reroll_cost. Also matches vanilla in not escalating the price
    # while free rerolls (Chaos the Clown) remain.
    from jackdaw.engine.run_init import _calculate_reroll_cost

    _calculate_reroll_cost(gs)

    # Track stat
    gs.setdefault("round_scores", {})
    gs["round_scores"]["times_rerolled"] = gs["round_scores"].get("times_rerolled", 0) + 1

    # Regenerate shop joker cards
    _reroll_shop_cards(gs)

    # Fire reroll_shop joker context (Flash Card +mult)
    _fire_shop_joker_context(gs, reroll_shop=True)

    return gs


def _handle_next_round(gs: dict[str, Any]) -> dict[str, Any]:
    """Leave the shop and proceed to the next blind.

    Before leaving:
    - Fire ``ending_shop`` on all jokers (Perkeo copies consumable)
    - Process Perkeo side-effects
    """
    _require_phase(gs, GamePhase.SHOP)

    # Fire ending_shop joker context (Perkeo)
    _fire_shop_joker_context(gs, ending_shop=True)

    # G.shop:remove() removes each remaining card before the CardArea dies.
    for area in ("shop_cards", "shop_vouchers", "shop_boosters"):
        for card in list(gs.get(area, [])):
            lifecycle.remove(gs, card)

    rr = gs["round_resets"]
    blind_on_deck = gs.get("blind_on_deck", "Small")

    # Set next blind to Select
    if blind_on_deck == "Small":
        rr["blind_states"]["Small"] = "Select"
    elif blind_on_deck == "Big":
        rr["blind_states"]["Big"] = "Select"
    else:
        rr["blind_states"]["Boss"] = "Select"

    gs["phase"] = GamePhase.BLIND_SELECT
    return gs


def _handle_sort_hand(gs: dict[str, Any], mode: str) -> dict[str, Any]:
    """Sort the hand by rank or suit."""
    _require_phase(gs, GamePhase.SELECTING_HAND)

    hand: list = gs.get("hand", [])
    if mode == "rank":
        hand.sort(
            key=lambda c: (
                getattr(c.base, "id", 0) if c.base else 0,
                getattr(c.base, "suit_nominal", 0) if c.base else 0,
            )
        )
    elif mode == "suit":
        hand.sort(
            key=lambda c: (
                getattr(c.base, "suit_nominal", 0) if c.base else 0,
                getattr(c.base, "id", 0) if c.base else 0,
            )
        )
    return gs


def _handle_swap_hand(gs: dict[str, Any], idx: int, direction: int) -> dict[str, Any]:
    """Swap a hand card with its neighbor.

    *direction* is ``-1`` (left) or ``+1`` (right).
    Free action — no cost, doesn't consume hands or discards.
    """
    _require_phase(gs, GamePhase.SELECTING_HAND)

    hand: list = gs.get("hand", [])
    other = idx + direction
    if not (0 <= idx < len(hand) and 0 <= other < len(hand)):
        raise IllegalActionError("Swap index out of range")

    hand[idx], hand[other] = hand[other], hand[idx]
    return gs


def _handle_swap_jokers(gs: dict[str, Any], idx: int, direction: int) -> dict[str, Any]:
    """Swap a joker with its neighbor.

    *direction* is ``-1`` (left) or ``+1`` (right).
    """
    _require_phase(gs, GamePhase.SELECTING_HAND, GamePhase.SHOP)

    jokers: list = gs.get("jokers", [])
    other = idx + direction
    if not (0 <= idx < len(jokers) and 0 <= other < len(jokers)):
        raise IllegalActionError("Swap index out of range")

    jokers[idx], jokers[other] = jokers[other], jokers[idx]
    return gs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sort_hand_desc(hand: list) -> None:
    """Sort hand in place, descending by nominal value.

    Matches Lua ``CardArea:sort()`` with default config ``sort='desc'``
    (cardarea.lua:577-580).  Uses ``Card.get_nominal()`` as the sort key,
    which combines rank, suit tiebreaker, face nominal, and a unique
    micro-value so every card gets a distinct position.

    Only sorts cards that have a ``get_nominal`` method (playing cards);
    non-playing-card entries are left at the end.
    """
    hand.sort(key=lambda c: c.get_nominal() if hasattr(c, "get_nominal") else -1e9, reverse=True)


def _draw_hand(gs: dict[str, Any], *, count: int | None = None) -> None:
    """Draw cards from deck to fill the hand up to hand_size.

    Cards are drawn from the END of the deck list (top of the visual
    stack), matching Lua's ``draw_card(G.deck, G.hand, ...)`` which
    pops from the last position.

    After drawing, the hand is sorted descending by nominal value
    (matching Lua's ``draw_from_deck_to_hand`` which passes ``sort=true``
    to ``draw_card``, triggering ``CardArea:sort()`` with default 'desc').
    """
    deck: list = gs.get("deck", [])
    hand: list = gs.setdefault("hand", [])
    hand_size: int = gs.get("hand_size", 8)
    to_draw = min(len(deck), hand_size - len(hand))
    if count is not None:
        to_draw = min(to_draw, count)
    for _ in range(to_draw):
        if deck:
            hand.append(deck.pop())
    # Sort hand descending by nominal (matches Lua CardArea:sort 'desc')
    _sort_hand_desc(hand)


def _end_round_if_hand_empty(gs: dict[str, Any]) -> None:
    """End a lost round when an action exhausts the hand and draw pile.

    Leaving ``SELECTING_HAND`` with no cards exposes no legal play or discard
    action, so an auto-resolved hand policy would be asked to decode an empty
    hand.  The blind cannot be cleared from that state; mark it as a terminal
    loss instead.
    """
    if not gs.get("hand"):
        gs["phase"] = GamePhase.GAME_OVER
        gs["won"] = False


def _round_won(gs: dict[str, Any]) -> None:
    """Handle winning a round — transition to ROUND_EVAL.

    Full sequence matching ``state_events.lua:87-120``:

    1. Fire joker ``end_of_round`` context (economy + scaling)
    2. Process perishable/rental (round_lifecycle)
    3. Gold Seal: +$3 per held card with Gold Seal
    4. Return all cards to deck (hand + played + discard)
    5. Un-debuff all playing cards (blind debuffs don't persist)
    6. Track unused discards (for Garbage Tag)
    7. Mark blind as Defeated
    8. Advance blind progression (Small→Big, Big→Boss)
    9. Boss beaten: check win condition, advance ante
    10. Calculate round earnings
    11. Phase → ROUND_EVAL
    """
    from jackdaw.engine.economy import calculate_round_earnings
    from jackdaw.engine.round_lifecycle import process_round_end_cards

    cr = gs["current_round"]
    blind = gs["blind"]
    jokers = gs.get("jokers", [])
    rng = gs.get("rng")

    # ------------------------------------------------------------------
    # 1. Fire joker end_of_round context
    # ------------------------------------------------------------------
    from jackdaw.engine.jokers import fire_jokers, round_dollar_bonus
    from jackdaw.engine.read import StateView

    joker_dollars = round_dollar_bonus(jokers, StateView(gs, jokers=jokers))
    queue = EffectQueue(gs)
    fire_jokers(gs, queue, end_of_round=True)
    queue.apply()
    # joker_dollars (Golden Joker, Rocket, Cloud 9, Satellite,
    # Delayed Gratification) is deliberately NOT applied here: it flows once
    # through calculate_round_earnings(joker_dollars=...) into
    # earnings.total, applied at CashOut — vanilla pays these as cash-out
    # rows AFTER interest is computed on the pre-payout balance
    # (state_events.lua:1175 vs :1191). Applying it here as well
    # double-counted the payout and leaked it into interest (inherited
    # upstream bug; pinned in tests/engine/test_cashout_ordering.py).
    # ------------------------------------------------------------------
    # 2. Process perishable/rental
    # ------------------------------------------------------------------
    process_round_end_cards(jokers, gs)

    # ------------------------------------------------------------------
    # 3. Gold Card enhancement: +h_dollars ($3) per held gold card
    #    (card.lua:1093). Keyed on ability["h_dollars"], NOT the Gold
    #    Seal — the seal pays on play via get_p_dollars, never when held.
    #    Lands before calculate_round_earnings, so it counts toward
    #    interest (in-blind money class).
    # ------------------------------------------------------------------
    hand: list = gs.get("hand", [])
    held_gold_dollars = sum(
        c.ability.get("h_dollars", 0)
        for c in hand
        if isinstance(getattr(c, "ability", None), dict) and not getattr(c, "debuff", False)
    )
    seal_queue = EffectQueue(gs)
    if held_gold_dollars:
        seal_queue.add(EaseDollars(amount=held_gold_dollars))

    # ------------------------------------------------------------------
    # 3b. Blue Seal: create Planet for the last played hand type
    # ------------------------------------------------------------------
    hand_levels = gs.get("hand_levels")
    for c in hand:
        if getattr(c, "seal", None) == "Blue" and not getattr(c, "debuff", False):
            if gs.get("last_hand_played"):
                last_played = gs["last_hand_played"]
                # Find the planet key for this hand type
                from jackdaw.engine.consumables import _PLANET_HAND

                planet_key = None
                for pk, ht in _PLANET_HAND.items():
                    if ht == last_played:
                        planet_key = pk
                        break
                if planet_key:
                    seal_queue.add(
                        CreateCard(
                            set="Planet",
                            forced_key=planet_key,
                            append="blusl",
                        )
                    )
    seal_queue.apply()

    # ------------------------------------------------------------------
    # 4. Return all cards to deck
    #
    # Lua sequence (state_events.lua:237-250):
    #   a) draw_from_hand_to_discard — hand cards removed first-first,
    #      appended at end of discard
    #   b) draw_from_discard_to_deck — discard cards popped LAST-first
    #      (remove_card on discard type takes #cards), then INSERTED AT
    #      FRONT of deck (emplace on deck type does table.insert(1))
    #
    # Net effect: [old_discard, hand] is prepended to deck front in
    # original order (pop-last + insert-at-front cancel out).
    # ------------------------------------------------------------------
    deck: list = gs.setdefault("deck", [])
    played: list = gs.get("played_cards_area", [])
    discarded: list = gs.get("discard_pile", [])

    # Step a: hand → discard end (forward order)
    discarded.extend(hand)
    # Any leftover played cards (shouldn't normally exist post-scoring)
    discarded.extend(played)
    # Step b: discard → deck FRONT (pop-last + insert-at-front = original order at front)
    deck[:0] = discarded

    gs["hand"] = []
    gs["played_cards_area"] = []
    gs["discard_pile"] = []

    # ------------------------------------------------------------------
    # 5. Un-debuff all playing cards (blind debuffs don't persist)
    # ------------------------------------------------------------------
    for card in deck:
        # Only clear blind-applied debuffs; perishable debuffs are permanent
        if getattr(card, "debuff", False):
            if not (getattr(card, "perishable", False) and getattr(card, "perish_tally", 1) <= 0):
                card.set_debuff(gs, False)

    # ------------------------------------------------------------------
    # 6. Track unused discards / hands played (for Garbage/Handy Tags)
    #    These are run-level cumulative totals matching Lua:
    #    - G.GAME.unused_discards += current_round.discards_left  (state_events.lua:124)
    #    - G.GAME.hands_played += 1 per hand  (state_events.lua:523, tracked at line 522)
    # ------------------------------------------------------------------
    gs["unused_discards"] = gs.get("unused_discards", 0) + cr.get("discards_left", 0)

    # ------------------------------------------------------------------
    # 7. Mark blind as Defeated
    # ------------------------------------------------------------------
    rr = gs["round_resets"]
    blind_on_deck = gs.get("blind_on_deck", "Small")
    rr["blind_states"][blind_on_deck] = "Defeated"
    gs["round"] = gs.get("round", 0) + 1

    # Back:trigger_effect({context='eval'}) runs here, after Blind:defeat and
    # before round earnings (state_events.lua:1163). Anaglyph awards one
    # Double Tag only for a defeated boss.
    from jackdaw.engine.back import Back

    back_result = Back(gs.get("selected_back_key", "b_red")).trigger_effect(
        "eval", boss_defeated=bool(getattr(blind, "boss", False))
    )
    if back_result and back_result.get("create_tag"):
        back_queue = EffectQueue(gs)
        back_queue.add(AddTag(key=back_result["create_tag"]))
        back_queue.apply()

    # On boss defeat Lua snapshots the run-wide most-played hand. Ties go
    # to the stronger hand (the lower display-order value).
    if getattr(blind, "boss", False) and hand_levels is not None:
        from jackdaw.engine.data.hands import HAND_BASE, HandType

        hand_name = HandType.HIGH_CARD
        played_count = -1
        order = 100
        for candidate, base in HAND_BASE.items():
            candidate_played = hand_levels.get_state(candidate).played
            if candidate_played > played_count or (
                candidate_played == played_count and order > base.order
            ):
                hand_name = candidate
                played_count = candidate_played
                order = base.order
        cr["most_played_poker_hand"] = hand_name.value

    # ------------------------------------------------------------------
    # 8-9. Advance blind progression
    # ------------------------------------------------------------------
    if blind_on_deck == "Small":
        gs["blind_on_deck"] = "Big"
    elif blind_on_deck == "Big":
        gs["blind_on_deck"] = "Boss"
    elif blind_on_deck == "Boss":
        # Boss beaten — check win, advance ante
        if rr["ante"] >= gs.get("win_ante", 8):
            gs["won"] = True
        _advance_ante(gs)
        gs["blind_on_deck"] = "Small"

    # The Manacle: restore hand size after boss defeat
    if blind_on_deck == "Boss" and getattr(blind, "name", "") == "The Manacle":
        if not getattr(blind, "disabled", False):
            gs["hand_size"] = gs.get("hand_size", 7) + 1

    # Juggle Tag: revert the one-round hand-size bonus applied by start_round
    temp_hs = cr.get("temp_handsize_applied", 0)
    if temp_hs:
        gs["hand_size"] = gs.get("hand_size", 8) - temp_hs
        cr["temp_handsize_applied"] = 0

    # ------------------------------------------------------------------
    # 10. Calculate round earnings (for cash-out screen)
    # ------------------------------------------------------------------
    earnings = calculate_round_earnings(
        blind=blind,
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        money=gs.get("dollars", 0),
        jokers=jokers,
        game_state=gs,
        rng=rng,
        joker_dollars=joker_dollars,
    )
    gs["round_earnings"] = earnings

    # ------------------------------------------------------------------
    # 11. Phase → ROUND_EVAL
    # ------------------------------------------------------------------
    gs["phase"] = GamePhase.ROUND_EVAL
    # Lua clears this on entry to round evaluation (game.lua:3312).
    gs["facing_blind"] = False


def _advance_ante(gs: dict[str, Any]) -> None:
    """Advance to the next ante after boss is defeated."""
    rr = gs["round_resets"]
    rr["ante"] += 1
    rr["blind_ante"] = rr["ante"]
    rr["blind_states"] = {"Small": "Select", "Big": "Upcoming", "Boss": "Upcoming"}
    rr["boss_rerolled"] = False

    # Generate new boss, tags, voucher for next ante
    from jackdaw.engine.tags import assign_ante_blinds

    rng = gs.get("rng")
    if rng:
        ante_result = assign_ante_blinds(rr["ante"], rng, gs)
        rr["blind_choices"]["Boss"] = ante_result["blind_choices"]["Boss"]
        gs["current_round"]["voucher"] = ante_result["voucher"]


# ---------------------------------------------------------------------------
# setting_blind joker context
# ---------------------------------------------------------------------------


def _fire_setting_blind(gs: dict[str, Any]) -> None:
    """Fire and apply the live-state ``setting_blind`` joker pass."""
    from jackdaw.engine.jokers import fire_jokers

    queue = EffectQueue(gs)
    fire_jokers(gs, queue, setting_blind=True)
    queue.apply()


# ---------------------------------------------------------------------------
# Boss blind set-time effects
# ---------------------------------------------------------------------------


def _apply_boss_blind_effects(gs: dict[str, Any], blind: Any) -> None:
    """Apply boss blind effects at set-time (blind.lua:157-209).

    These are one-time mutations that happen when the blind is set,
    before the round starts.
    """
    cr = gs.get("current_round", {})
    name = getattr(blind, "name", "")

    # The Water: remove all discards
    if name == "The Water":
        current_discards = cr.get("discards_left", 0)
        blind.discards_sub = current_discards
        cr["discards_left"] = 0

    # The Needle: reduce to 1 hand
    elif name == "The Needle":
        rr = gs.get("round_resets", {})
        current_hands = rr.get("hands", 4)
        blind.hands_sub = current_hands - 1
        cr["hands_left"] = max(1, cr.get("hands_left", current_hands) - blind.hands_sub)

    # The Manacle: -1 hand size
    elif name == "The Manacle":
        gs["hand_size"] = gs.get("hand_size", 8) - 1

    # Amber Acorn: shuffle jokers (flip + randomize order)
    elif name == "Amber Acorn":
        jokers: list = gs.get("jokers", [])
        if jokers:
            for j in jokers:
                j.facing = "back"
            rng = gs.get("rng")
            if rng and len(jokers) > 1:
                seed_val = rng.seed("aajk")
                rng.shuffle(jokers, seed_val)

    # The Eye's history lives in Blind.hands_used, reset on Blind creation.

    # The Mouth: reset only_hand
    elif name == "The Mouth":
        blind.only_hand = None

    # The House / The Mark: flip cards face-down (blind.lua:200-203)
    # Cards are flipped at draw_to_hand time, not set_blind time.
    # We handle this in _draw_hand context by checking blind.name.


# ---------------------------------------------------------------------------
# Shop joker context helpers
# ---------------------------------------------------------------------------


def _use_consumable_card(
    gs: dict[str, Any],
    card: Any,
    targets: tuple[int, ...] | None = None,
) -> None:
    """Build a ConsumableContext and use a consumable card.

    Bridges the game_state dict with the ``use_consumable(card, ctx)`` API.

    1. Build ConsumableContext from game_state + target_indices
    2. Call handler → ConsumableResult
    3. Translate and apply every result field through an EffectQueue
    4. Fire ``using_consumeable`` on every joker for every consumable
    5. Track usage (last_tarot_planet)
    """
    from jackdaw.engine.consumables import (
        ConsumableContext,
        consumable_effects,
        record_consumable_usage,
        use_consumable,
    )

    # Lua Card:use_consumeable records usage before anything else, the
    # debuff check included (card.lua:1093).
    record_consumable_usage(gs, card)

    hand: list = gs.get("hand", [])
    highlighted: list = []
    if targets:
        highlighted = [hand[i] for i in targets if i < len(hand)]

    ctx = ConsumableContext(
        card=card,
        highlighted=highlighted or None,
        hand_cards=hand or None,
        jokers=gs.get("jokers") or None,
        consumables=gs.get("consumables") or None,
        playing_cards=gs.get("deck") or None,
        rng=gs.get("rng"),
        game_state=gs,
    )
    result = use_consumable(card, ctx)

    if result is not None:
        # Track last_tarot_planet for The Fool
        card_key = getattr(card, "center_key", None)
        if card_key:
            card_set = _get_card_set(card) if card else ""
            if card_set in ("Tarot", "Planet"):
                gs["last_tarot_planet"] = card_key

        queue = EffectQueue(gs)
        queue.add(consumable_effects(result))
        queue.apply()

    # button_callbacks.lua:2209-2220 dispatches this for every consumable.
    from jackdaw.engine.jokers import fire_jokers

    notify_queue = EffectQueue(gs)
    fire_jokers(
        gs,
        notify_queue,
        using_consumeable=True,
        consumeable=card,
        highlighted=highlighted,
    )
    notify_queue.apply()


# ---------------------------------------------------------------------------
# Shop population helpers
# ---------------------------------------------------------------------------


def _populate_shop(gs: dict[str, Any]) -> None:
    """Generate shop cards using populate_shop and store in game_state.

    Places results in ``gs["shop_cards"]``, ``gs["shop_vouchers"]``,
    ``gs["shop_boosters"]``.
    """
    from jackdaw.engine.shop import populate_shop

    rng = gs.get("rng")
    if rng is None:
        return

    # Sync played hand types for Planet pool softlock filtering
    # (G.GAME.hands[ht].played > 0, common_events.lua:2009).

    ante = gs.get("round_resets", {}).get("ante", 1)
    result = populate_shop(rng, ante, gs)

    gs["shop_cards"] = []
    for card in result.get("jokers", []):
        lifecycle.emplace(gs, card, "shop_cards")
    voucher = result.get("voucher")
    gs["shop_vouchers"] = []
    if voucher:
        lifecycle.emplace(gs, voucher, "shop_vouchers")
    gs["shop_boosters"] = []
    for booster in result.get("boosters", []):
        lifecycle.emplace(gs, booster, "shop_boosters")

    from jackdaw.engine.tags import fire_tag_context

    # Voucher Tag (voucher_add context): each adds one extra voucher,
    # polled from the tag pool key ('Voucher_fromtag' — vouchers.py).
    voucher_fired = fire_tag_context(gs, "voucher_add")
    if voucher_fired:
        from jackdaw.engine.card_factory import create_voucher
        from jackdaw.engine.vouchers import get_next_voucher_key

        for _entry, tag_res in voucher_fired:
            if not tag_res.create_voucher:
                continue
            in_shop = [getattr(v, "center_key", "") for v in gs["shop_vouchers"]]
            v_key = get_next_voucher_key(
                rng,
                gs.get("used_vouchers", {}),
                in_shop=in_shop,
                from_tag=True,
                ante=ante,
            )
            if v_key is None:
                continue
            extra = create_voucher(v_key, game_state=gs)
            extra.set_cost(gs)
            lifecycle.emplace(gs, extra, "shop_vouchers")

    # Coupon Tag (shop_final_pass context): initial shop cards and booster
    # packs become free. Vouchers stay full price, and rerolled cards are
    # NOT free (this fires only on the populate pass, never on rerolls).
    for _entry, tag_res in fire_tag_context(gs, "shop_final_pass"):
        if tag_res.coupon:
            for card in gs["shop_cards"]:
                card.ability["couponed"] = True
                card.set_cost(gs)
            for booster in gs["shop_boosters"]:
                booster.ability["couponed"] = True
                booster.set_cost(gs)


def _reroll_shop_cards(gs: dict[str, Any]) -> None:
    """Regenerate the shop joker/consumable cards (not voucher or boosters).

    Matches the repopulate step of ``reroll_shop``
    (``button_callbacks.lua:2855``).
    """
    from jackdaw.engine.shop import create_shop_slot_card

    rng = gs.get("rng")
    if rng is None:
        return

    # Lua removes every old card before rolling replacements, making its key
    # eligible for the new roll (button_callbacks.lua:2873-2881).
    for old_card in list(gs.get("shop_cards", [])):
        lifecycle.remove(gs, old_card)

    ante = gs.get("round_resets", {}).get("ante", 1)
    shop_joker_max: int = gs.get("shop", {}).get("joker_max", 2)

    # Shared slot-creation path applies pending Rare/Uncommon/edition tags
    # to rerolled cards too (vanilla: an unspent tag catches the next shop
    # Joker created, whether from populate or reroll).
    gs["shop_cards"] = [create_shop_slot_card(rng, ante, gs) for _ in range(shop_joker_max)]


def _get_card_set(card: Any) -> str:
    """Get the set name from a Card's ability dict."""
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return ability.get("set", "")
    return ""


def _fire_shop_joker_context(gs: dict[str, Any], **context_flags: Any) -> None:
    """Fire and apply a joker context during the shop phase.

    Accepts keyword arguments matching :class:`JokerContext` flags
    (e.g. ``buying_card=True``, ``reroll_shop=True``).
    """
    lifecycle.fire_joker_context(gs, **context_flags)


# ---------------------------------------------------------------------------
# Pack close helper
# ---------------------------------------------------------------------------


def _close_pack(gs: dict[str, Any]) -> None:
    """Close the current booster pack and return to the previous phase.

    Matches ``end_consumeable`` in ``button_callbacks.lua:2565``:
    - Remove remaining pack cards
    - Return dealt hand cards to deck (Arcana/Spectral packs deal a hand)
    - Fire ``new_blind_choice`` tags (deferred from skip)
    - Restore phase from ``shop_return_phase``
    """
    # CardArea:remove() calls Card:remove on every unpicked display card.
    for card in list(gs.get("pack_cards", [])):
        lifecycle.remove(gs, card)
    gs["pack_choices_remaining"] = 0

    # Return dealt hand cards to deck (Arcana/Spectral packs deal from deck)
    # Lua's draw_from_hand_to_deck (state_events.lua:1121-1126) removes
    # first card from hand and inserts at position 1 (front) of deck,
    # repeated for all cards.  Net effect: hand cards end up REVERSED
    # at the FRONT of the deck.
    pack_hand: list = gs.get("pack_hand", [])
    if pack_hand:
        deck: list = gs.setdefault("deck", [])
        hand: list = gs.get("hand", [])
        hand_set = set(id(c) for c in hand)
        # Only return cards that are still in the hand (not destroyed)
        surviving = [c for c in pack_hand if id(c) in hand_set]
        deck[:0] = list(reversed(surviving))
        # Remove only the pack_hand cards from hand, preserving any
        # cards that were there before the pack opened.
        pack_ids = set(id(c) for c in pack_hand)
        gs["hand"] = [c for c in hand if id(c) not in pack_ids]
        gs["pack_hand"] = []

    pending_tag_packs = gs.get("pending_tag_packs", [])
    if pending_tag_packs:
        return_phase = gs.get("shop_return_phase", GamePhase.BLIND_SELECT)
        _open_tag_pack(gs, pending_tag_packs.pop(0))
        gs["shop_return_phase"] = return_phase
        return

    # Restore phase
    gs["phase"] = gs.get("shop_return_phase", GamePhase.SHOP)


# ---------------------------------------------------------------------------
# Blind:press_play — blind.lua:464
# ---------------------------------------------------------------------------


def _press_play(
    gs: dict[str, Any],
    blind: Any,
    played: list,
    rng: Any,
) -> None:
    """Fire boss blind press_play effects before scoring.

    Mirrors ``Blind:press_play`` (blind.lua:464-502).
    """
    if getattr(blind, "disabled", False):
        return

    name = getattr(blind, "name", "")

    if name == "The Hook":
        # Discard 2 random cards from hand
        hand: list = gs.get("hand", [])
        discard_pile: list = gs.setdefault("discard_pile", [])
        for _ in range(min(2, len(hand))):
            if hand and rng:
                seed_val = rng.seed("hook")
                target, _ = rng.element(hand, seed_val)
                hand.remove(target)
                discard_pile.append(target)

    elif name == "The Tooth":
        # Lose $1 per card played
        gs["dollars"] = gs.get("dollars", 0) - len(played)

    elif name == "The Fish":
        # Flip all hand cards face-down after play (blind.lua:494-496)
        blind.prepped = True

    elif name == "Crimson Heart":
        # Debuff a random joker each hand (blind.lua:488-493)
        jokers: list = gs.get("jokers", [])
        if jokers and rng:
            blind.triggered = True
            blind.prepped = True


# ---------------------------------------------------------------------------
# Double Tag check
# ---------------------------------------------------------------------------


def _check_double_tag(gs: dict[str, Any], awarded_entry: dict[str, Any]) -> None:
    """If a Double Tag is held, duplicate the just-awarded tag.

    Scans ``awarded_tags`` (the only tag store — the old ``gs["tags"]`` list
    was never populated, so Double Tag could never fire) for an un-consumed
    ``tag_double`` acquired BEFORE the current award. Consumes one Double
    Tag per award; the duplicate entry behaves exactly like a fresh award:
    immediate effects apply in full (dollars, top-up jokers, orbital
    level-ups — the old code applied only dollars), deferred tags stay
    un-consumed for their later context polls.
    """
    awarded_tag_key = awarded_entry.get("key", "")
    if awarded_tag_key == "tag_double":
        return

    from jackdaw.engine.tags import Tag

    awarded_tags: list = gs.setdefault("awarded_tags", [])
    for entry in awarded_tags:
        if entry is awarded_entry:
            continue
        if entry.get("key") != "tag_double" or entry.get("consumed"):
            continue

        # Consume the Double Tag and fire the duplicate award
        entry["consumed"] = True
        entry["consumed_context"] = "tag_add"

        dup_result = Tag(awarded_tag_key).apply("immediate", gs, rng=gs.get("rng"))
        dup_entry: dict[str, Any] = {
            "key": awarded_tag_key,
            "result": dup_result,
            "blind": "double",
        }
        awarded_tags.append(dup_entry)

        if dup_result is not None:
            _apply_tag_result(gs, dup_result)
            dup_entry["consumed"] = True
            dup_entry["consumed_context"] = "immediate"
        break
