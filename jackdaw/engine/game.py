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
    ChangeHandSize,
    ChangeRoundResource,
    CreateCard,
    EaseDollars,
    EffectQueue,
    LevelUpHand,
    ShuffleArea,
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

    # L: blind.lua:157-205 — set_blind emits its boss effects before the
    # setting_blind pass. F5 shares one FIFO queue for both producers.
    set_blind_queue = EffectQueue(gs)
    if blind.boss:
        _apply_boss_blind_effects(gs, blind, queue=set_blind_queue)

    # L: blind.lua:207-215 — set_blind rechecks every playing card and Joker.
    active_rules = read.rules(gs)
    for card in read.playing_cards(gs):
        blind.debuff_card(card, active_rules, gs)
    for joker in gs.get("jokers", []):
        blind.debuff_card(joker, active_rules, gs, is_joker_area=True)

    # L: state_events.lua:333-337 — setting_blind is emitted after set_blind.
    _fire_setting_blind(gs, queue=set_blind_queue)
    # F5: blind/set-blind and Joker events settle before the state shuffle.
    set_blind_queue.apply()

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

    # L: state_events.lua:338-349; game.lua:3208-3244 — enter the one shared
    # DRAW_TO_HAND controller after the nr shuffle.
    _draw_from_deck_to_hand(gs)
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
    """Synchronous port of ``play_cards_from_highlighted`` (Lua 450-537)."""
    _require_phase(gs, GamePhase.SELECTING_HAND)

    # L: state_events.lua:450-463 — validate, stop input, and clear the blind.
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

    # L: state_events.lua:459-483 — move the selected cards into play.
    # Preserve SELECTION ORDER (not hand position order).
    # NEW-P5-1-05: unlike Lua's physical-x sort, solver-selected scoring order
    # is an intentional engine behavior pending the explicit user decision.
    idx_set = set(indices)
    played = [hand[i] for i in indices]
    held = [c for i, c in enumerate(hand) if i not in idx_set]
    gs["hand"] = held
    gs["played_cards_area"] = played

    # L: state_events.lua:465-476 — HAND_PLAYED is intentionally collapsed
    # (NEW-P5-1-06); the queued hand decrement settles at F1.
    cr["hands_left"] -= 1

    # L: state_events.lua:478-484 — per-card play statistics precede press_play.
    for card in played:
        base = getattr(card, "base", None)
        if base is not None:
            base.times_played = getattr(base, "times_played", 0) + 1
        ability = getattr(card, "ability", None)
        if isinstance(ability, dict):
            ability["played_this_ante"] = True

    # L: state_events.lua:488-502 — Blind:press_play, then F1. Hook remains
    # on its current mechanics until P5-3, but this dispatch point is exact.
    rng = gs.get("rng")
    press_queue = EffectQueue(gs)
    _press_play(gs, blind, played, rng, press_queue)
    press_queue.apply()

    # L: state_events.lua:504-515 — evaluate_play is the first child event.
    from jackdaw.engine.scoring import score_hand

    jokers = gs.get("jokers", [])
    hand_levels = gs.get("hand_levels")

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

    # L: state_events.lua:517-527 — the caller's already queued event moves
    # every played card first, then advances both hand counters at F2.
    discard_pile: list = gs.setdefault("discard_pile", [])
    discard_pile.extend(played)
    gs["played_cards_area"] = []
    cr["hands_played"] += 1
    gs["hands_played"] = gs.get("hands_played", 0) + 1

    # L: state_events.lua:657-662,719-730,836-839,985-995,1029-1084 —
    # scoring's queued gameplay events follow the move/counter event at F2.
    gs["last_score_result"] = result

    if result.effect_queue is not None:
        result.effect_queue.apply()

    # L: game.lua:3187-3204 — only after F2 does HAND_PLAYED choose the
    # win, redraw, or game-over branch.
    if gs["chips"] >= blind.chips or cr["hands_left"] <= 0:
        # L: game.lua:3187-3204; state_events.lua:87-287 — every terminal
        # play enters the same accumulated-chip end_round controller.
        end_round(gs)
    else:
        # L: game.lua:3187-3244 — the DRAW_TO_HAND controller owns every
        # post-play redraw, including Serpent and drawn_to_hand effects.
        _draw_from_deck_to_hand(gs)

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

    discarded = [hand[i] for i in sorted(indices)]
    discard_cards_from_highlighted(gs, discarded)
    return gs


def discard_cards_from_highlighted(
    gs: dict[str, Any],
    cards: list[Any],
    *,
    hook: bool = False,
) -> None:
    """Shared synchronous port of Lua's voluntary/Hook discard function."""
    from jackdaw.engine.jokers import calculate_joker, context_for, fire_jokers

    # L: state_events.lua:379-393; game.lua:2247-2251 — clear forced
    # selections, cap by discard room, and restore physical hand order.
    for playing_card in read.playing_cards(gs):
        playing_card.ability.pop("forced_selection", None)
    hand: list = gs.get("hand", [])
    selected_ids = {id(card) for card in cards}
    discard_room = max(0, 500 - len(gs.get("played_cards_area", [])))
    discarded = [card for card in hand if id(card) in selected_ids][:discard_room]
    if not discarded:
        return

    jokers: list = gs.get("jokers", [])
    queue = EffectQueue(gs)

    # L: state_events.lua:394-396; card.lua:2748-2755 — Burnt observes the
    # Hook flag in pre_discard and is the only vanilla discard handler that
    # reads context.hook.
    for joker in list(jokers):
        if joker.debuff:
            continue
        result = calculate_joker(
            joker,
            context_for(
                gs,
                queue=queue,
                pre_discard=True,
                full_hand=discarded,
                hook=hook,
            ),
        )
        if result is not None and result.level_up:
            hand_levels = gs.get("hand_levels")
            if hand_levels is not None:
                from jackdaw.engine.hand_eval import evaluate_hand

                detected = evaluate_hand(discarded).detected_hand
                if detected and detected != "NULL":
                    hand_levels.level_up(detected)

    # L: state_events.lua:397-422 — seal first, then every Joker, per card.
    destroyed: list = []
    for card in discarded:
        if card.seal == "Purple":
            # L: card.lua:2254-2265 — Purple Seal reserves and queues a Tarot.
            queue.add(CreateCard(set="Tarot", append="8ba", source=card))

        removed = False
        for joker in list(jokers):
            if joker.debuff:
                continue
            result = calculate_joker(
                joker,
                context_for(
                    gs,
                    queue=queue,
                    discard=True,
                    other_card=card,
                    full_hand=discarded,
                ),
            )
            if result is None:
                continue
            if result.dollars:
                # L: card.lua:2802-2833,2858-2869 — Mail/Trading queue their
                # ease directly; Faceless emits it from a nested event.
                nested = joker.center_key == "j_faceless"
                queue.add(
                    EaseDollars(
                        amount=result.dollars,
                        order=1 if nested else 0,
                        source=joker,
                    )
                )
            removed = removed or result.remove
        if removed:
            destroyed.append(card)

    # L: state_events.lua:424-428 — remove first, then one notification pass.
    if destroyed:
        lifecycle.destroy_playing_cards(gs, destroyed, notify=False)
        fire_jokers(gs, queue, cards_destroyed=destroyed)

    # L: state_events.lua:410-422 — surviving cards are marked before their
    # queued move to discard settles at F3.
    destroyed_ids = {id(card) for card in destroyed}
    surviving = [card for card in discarded if id(card) not in destroyed_ids]
    for card in surviving:
        card.ability["discarded"] = True
    discarded_ids = {id(card) for card in discarded}
    hand[:] = [card for card in hand if id(card) not in discarded_ids]
    gs.setdefault("discard_pile", []).extend(surviving)

    # L: state_events.lua:430-431 — Hook and voluntary discards both count.
    scores = gs.setdefault("round_scores", {})
    scores["cards_discarded"] = scores.get("cards_discarded", 0) + len(discarded)

    # L: state_events.lua:432-446 — cost/counters/DRAW_TO_HAND are voluntary.
    if not hook:
        discard_cost = gs.get("modifiers", {}).get("discard_cost", 0)
        if discard_cost:
            queue.add(EaseDollars(amount=-discard_cost))
        queue.add(ChangeRoundResource(discards=-1))
        gs["current_round"]["discards_used"] += 1

    # F3: seal/Joker/move/money/resource events settle in FIFO order. Nested
    # Faceless money is order=1, after every first-level discard event.
    queue.apply()

    if not hook:
        # L: game.lua:3208-3244 — only voluntary discard enters DRAW_TO_HAND.
        _draw_from_deck_to_hand(gs)


def _handle_cash_out(gs: dict[str, Any]) -> dict[str, Any]:
    """Synchronous F8 port of ``G.FUNCS.cash_out``."""
    _require_phase(gs, GamePhase.ROUND_EVAL)

    rng = gs.get("rng")
    rr = gs["round_resets"]
    cr = gs["current_round"]

    # L: button_callbacks.lua:2918-2919 — preserve the verified cashout deck
    # order by retaining the same stream and in-place shuffle implementation.
    if rng:
        deck: list = gs.get("deck", [])
        ante = rr.get("ante", 1)
        cashout_seed = rng.seed("cashout" + str(ante))
        rng.shuffle(deck, cashout_seed)

    # L: button_callbacks.lua:2921-2937 — these values are visible in SHOP,
    # so reset them before changing phase or populating shop observations.
    cr["jokers_purchased"] = 0
    cr["discards_left"] = max(0, rr["discards"] + gs["round_bonus"]["discards"])
    cr["hands_left"] = max(1, rr["hands"] + gs["round_bonus"]["next_hands"])
    gs["phase"] = GamePhase.SHOP
    gs["shop_free"] = False
    gs["shop_d6ed"] = False

    # L: button_callbacks.lua:2938-2944 — commit the bottom-row total through
    # the one ledger path, then snapshot the resulting committed balance.
    cashout_queue = EffectQueue(gs)
    cashout_queue.add(EaseDollars(amount=cr.get("dollars", 0)))
    cashout_queue.apply()
    gs["previous_round"] = {"dollars": gs.get("dollars", 0)}

    # L: button_callbacks.lua:2948-2954 — chips clear before post-Boss tags
    # and blind choices are generated.
    gs["chips"] = 0
    if rr["blind_states"].get("Boss") == "Defeated":
        _cashout_reset_boss_choices(gs)

    from jackdaw.engine.tags import fire_tag_context

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
    return gs


def _cashout_reset_boss_choices(gs: dict[str, Any]) -> None:
    """Generate post-Boss tags and the next boss at Lua's F8 point."""
    from jackdaw.engine.blind import get_new_boss
    from jackdaw.engine.pools import pick_card_from_pool

    rr = gs["round_resets"]
    rng = gs.get("rng")
    rr["blind_ante"] = rr["ante"]
    if rng is not None:
        used_vouchers = {key for key, owned in gs.get("used_vouchers", {}).items() if owned}
        # L: button_callbacks.lua:2949-2952 — Small then Big tag.
        rr["blind_tags"] = {
            "Small": pick_card_from_pool(
                "Tag", rng, rr["ante"], used_vouchers=used_vouchers
            ),
            "Big": pick_card_from_pool(
                "Tag", rng, rr["ante"], used_vouchers=used_vouchers
            ),
        }
        # L: common_events.lua:2326-2335 — reset_blinds chooses the boss last.
        rr["blind_choices"]["Boss"] = get_new_boss(
            rr["ante"],
            gs.setdefault("bosses_used", {}),
            rng,
            win_ante=gs.get("win_ante", 8),
            banned_keys=gs.get("banned_keys"),
        )
    rr["blind_states"] = {"Small": "Upcoming", "Big": "Upcoming", "Boss": "Upcoming"}
    rr["boss_rerolled"] = False
    gs["blind_on_deck"] = "Small"


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
    from jackdaw.engine.jokers import fire_jokers

    ending_shop_queue = EffectQueue(gs)
    fire_jokers(gs, ending_shop_queue, ending_shop=True)
    ending_shop_queue.apply()

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


def _draw_card_to_hand(gs: dict[str, Any]) -> Any | None:
    """Resolve one queued ``draw_card(G.deck, G.hand, ..., sort=true)``."""
    deck: list = gs.get("deck", [])
    if not deck:
        return None
    hand: list = gs.setdefault("hand", [])
    card = deck.pop()
    blind = gs.get("blind")
    stay_flipped = False
    if blind is not None:
        cr = gs.get("current_round", {})
        # L: common_events.lua:386-423; blind.lua:605-622.
        stay_flipped = bool(
            blind.stay_flipped(
                card,
                read.rules(gs),
                rng=gs.get("rng"),
                probabilities_normal=gs.get("probabilities", {}).get("normal", 1),
                hands_played=cr.get("hands_played", 0),
                discards_used=cr.get("discards_used", 0),
            )
        )

    # L: common_events.lua:397-400 — the flipped-cards challenge is a second
    # independent reason for a card to stay face-down.
    flipped_cards = gs.get("modifiers", {}).get("flipped_cards")
    rng = gs.get("rng")
    if flipped_cards and rng is not None:
        if rng.random("flipped_card") < 1 / flipped_cards:
            stay_flipped = True

    lifecycle.emplace(gs, card, "hand")
    if stay_flipped:
        card.facing = "back"
        # L: cardarea.lua:39-43 — all stay-flipped hand cards use this marker,
        # despite its Wheel-specific name.
        card.ability["wheel_flipped"] = True
    else:
        card.facing = "front"
        card.ability.pop("wheel_flipped", None)
    if blind is not None:
        blind.debuff_card(card, read.rules(gs), gs)
    _sort_hand_desc(hand)
    return card


def _draw_from_deck_to_hand(gs: dict[str, Any]) -> bool:
    """One synchronous DRAW_TO_HAND controller for round/play/discard draws."""
    hand: list = gs.setdefault("hand", [])
    hand_size = gs.get("hand_size", 8)

    # L: state_events.lua:355-360 — this is Lua's sole empty-draw loss guard.
    if hand_size <= 0 and not hand:
        end_round(gs)
        return False

    deck: list = gs.get("deck", [])
    to_draw = max(0, min(len(deck), hand_size - len(hand)))
    blind = gs.get("blind")
    cr = gs.get("current_round", {})
    # L: state_events.lua:362-368 — after either prior action, Serpent draws
    # exactly up to three rather than filling the available hand space.
    if (
        blind is not None
        and blind.name == "The Serpent"
        and not blind.disabled
        and (cr.get("hands_played", 0) > 0 or cr.get("discards_used", 0) > 0)
    ):
        to_draw = min(len(deck), 3)

    # L: game.lua:3219-3231 — dispatch after draw events have been queued but
    # before they land. In this synchronous port, handler emission happens
    # now; its queued effects remain behind the staged card draws.
    first_hand_queue = EffectQueue(gs)
    if (
        cr.get("hands_played", 0) == 0
        and cr.get("discards_used", 0) == 0
        and gs.get("facing_blind")
    ):
        from jackdaw.engine.jokers import fire_jokers

        fire_jokers(gs, first_hand_queue, first_hand_drawn=True)

    # L: state_events.lua:369-376; common_events.lua:386-423 — F4 resolves
    # each draw FIFO. Certificate's already-emitted child follows these draws.
    for _ in range(to_draw):
        _draw_card_to_hand(gs)
    first_hand_queue.apply()

    # L: game.lua:3233-3241; blind.lua:572-603 — selecting-hand and the blind
    # callback follow all draw/Certificate events. drawn_to_hand owns prepped.
    gs["phase"] = GamePhase.SELECTING_HAND
    if blind is not None:
        dth = blind.drawn_to_hand(
            hand_cards=hand,
            joker_cards=gs.get("jokers"),
            rng=gs.get("rng"),
            gs=gs,
        )
        forced = dth.get("forced_card_index")
        if forced is not None and 0 <= forced < len(hand):
            hand[forced].ability["forced_selection"] = True

    # L: game.lua:3056-3065 — SELECTING_HAND immediately ends a round when
    # hand, deck, and play are all empty. At this stable port point play has
    # already settled to discard, so an uncleared blind is a terminal loss.
    if not hand and not deck and not gs.get("played_cards_area"):
        # L: game.lua:3056-3065 — exhausted areas use the ordinary end_round
        # path, including losing-round Joker maintenance and Mr. Bones.
        end_round(gs)
        return False
    return True


def end_round(gs: dict[str, Any]) -> None:
    """Synchronous F6 port of Lua ``end_round`` for wins, saves, and losses."""
    from jackdaw.engine.jokers import calculate_joker, context_for
    from jackdaw.engine.round_lifecycle import is_rental, process_perishable

    cr = gs["current_round"]
    blind = gs["blind"]
    blind_type = blind.get_type()
    was_boss = blind_type == "Boss"
    blind_target = blind.chips
    blind_reward = blind.dollars
    jokers = gs.get("jokers", [])
    queue = EffectQueue(gs)

    # L: state_events.lua:92-110 — game_over reads committed accumulated
    # chips, then each Joker's EOR, saved flag, rent, and perish step interleave.
    game_over = gs.get("chips", 0) < blind_target
    rental_cost = 0
    for joker in list(jokers):
        result = calculate_joker(
            joker,
            context_for(gs, queue=queue, end_of_round=True, game_over=game_over),
        )
        if result is not None and result.saved:
            game_over = False
        if is_rental(joker):
            rental_cost += gs.get("rental_rate", 3)
            queue.add(
                EaseDollars(
                    amount=-gs.get("rental_rate", 3),
                    source=joker,
                )
            )
        process_perishable(joker, gs)

    # L: state_events.lua:111-123 — even an ordinary loss flushes the EOR
    # maintenance above before entering GAME_OVER.
    if was_boss and gs["round_resets"]["ante"] == gs.get("win_ante", 8):
        gs["won"] = True
    if game_over:
        queue.apply()
        gs["phase"] = GamePhase.GAME_OVER
        gs["won"] = bool(gs.get("won", False))
        gs["facing_blind"] = False
        return

    # L: state_events.lua:124-170 — saved rounds share the won-round branch.
    gs["unused_discards"] = gs.get("unused_discards", 0) + cr.get("discards_left", 0)
    hand_levels = gs.get("hand_levels")
    if was_boss and hand_levels is not None:
        from jackdaw.engine.data.hands import HAND_BASE, HandType

        hand_name = HandType.HIGH_CARD
        played_count = -1
        stale_order = 100
        for candidate, base in HAND_BASE.items():
            candidate_played = hand_levels.get_state(candidate).played
            if candidate_played > played_count or (
                candidate_played == played_count and stale_order > base.order
            ):
                hand_name = candidate
                played_count = candidate_played
                # L: state_events.lua:129-137 — NEW-P5-1-03: Lua never updates
                # `_order`. Preserve that apparent bug instead of fixing ties.
        cr["most_played_poker_hand"] = hand_name.value

    # L: state_events.lua:171-233 — each held card recomputes its own EOR
    # effect and individual Joker effects for the base pass and every repeat.
    _held_cards_end_of_round(gs, queue)

    # L: state_events.lua:237-250 — hand -> discard, then discard -> deck.
    hand: list = gs.get("hand", [])
    deck: list = gs.setdefault("deck", [])
    discarded: list = gs.get("discard_pile", [])
    discarded.extend(hand)
    discarded.extend(gs.get("played_cards_area", []))
    deck[:0] = discarded
    gs["hand"] = []
    gs["played_cards_area"] = []
    gs["discard_pile"] = []

    rr = gs["round_resets"]
    if was_boss:
        _advance_ante(gs)

    # L: state_events.lua:251-283 — the final F6 transition owns all reset
    # state, including Pillar markers and target-card rolls (not cash-out).
    rr["blind_states"][blind_type] = "Defeated"
    if blind_type == "Small":
        gs["blind_on_deck"] = "Big"
    elif blind_type == "Big":
        gs["blind_on_deck"] = "Boss"
    for card in read.playing_cards(gs):
        if was_boss:
            card.ability.pop("played_this_ante", None)
        card.ability.pop("discarded", None)
        card.ability.pop("forced_selection", None)

    temp_hs = cr.get("temp_handsize_applied", 0)
    if temp_hs:
        gs["hand_size"] = gs.get("hand_size", 8) - temp_hs
        cr["temp_handsize_applied"] = 0

    rng = gs.get("rng")
    if rng is not None:
        from jackdaw.engine.round_lifecycle import reset_round_targets

        reset_round_targets(rng, rr["ante"], gs)

    gs["round"] = gs.get("round", 0) + 1
    gs["phase"] = GamePhase.ROUND_EVAL
    gs["facing_blind"] = False

    # F6: commit Joker effects, rent, held repeats, creations, and removals
    # before round evaluation starts (state_events.lua:87-287).
    queue.apply()
    evaluate_round(
        gs,
        blind_target=blind_target,
        blind_reward=blind_reward,
        was_boss=was_boss,
        rental_cost=rental_cost,
    )


def _held_cards_end_of_round(gs: dict[str, Any], queue: EffectQueue) -> None:
    """Emit held-card EOR effects and first-pass repetitions into *queue*."""
    from jackdaw.engine.consumables import _PLANET_HAND
    from jackdaw.engine.jokers import calculate_joker, context_for

    hand = list(gs.get("hand", []))
    jokers = list(gs.get("jokers", []))
    planet_for_hand = {hand_type: key for key, hand_type in _PLANET_HAND.items()}

    for held in hand:
        reps = 1
        repeat_index = 0
        while repeat_index < reps:
            has_card_effect = False
            if not held.debuff:
                h_dollars = held.ability.get("h_dollars", 0)
                if h_dollars:
                    queue.add(EaseDollars(amount=h_dollars, source=held))
                    has_card_effect = True
                planet_key = planet_for_hand.get(gs.get("last_hand_played"))
                if held.seal == "Blue" and planet_key and queue.room("consumables") > 0:
                    queue.add(
                        CreateCard(
                            set="Planet",
                            forced_key=planet_key,
                            append="blusl",
                            source=held,
                        )
                    )
                    has_card_effect = True

            individual_effect = False
            for joker in jokers:
                result = calculate_joker(
                    joker,
                    context_for(
                        gs,
                        queue=queue,
                        individual=True,
                        end_of_round=True,
                        cardarea="hand",
                        other_card=held,
                        held_cards=hand,
                    ),
                )
                individual_effect = individual_effect or result is not None

            # L: state_events.lua:189-209 — Red seal and Joker repetitions are
            # collected only on the first pass and only when the card/Joker has
            # an EOR effect to repeat.
            if repeat_index == 0 and (has_card_effect or individual_effect):
                seal_result = held.calculate_seal(repetition=True)
                if seal_result:
                    reps += seal_result.get("repetitions", 0)
                for joker in jokers:
                    result = calculate_joker(
                        joker,
                        context_for(
                            gs,
                            queue=queue,
                            repetition=True,
                            end_of_round=True,
                            cardarea="hand",
                            other_card=held,
                            held_cards=hand,
                        ),
                    )
                    if result is not None:
                        reps += result.repetitions
            repeat_index += 1


def evaluate_round(
    gs: dict[str, Any],
    *,
    blind_target: int | None = None,
    blind_reward: int | None = None,
    was_boss: bool | None = None,
    rental_cost: int = 0,
) -> None:
    """Synchronous F7 port of ``G.FUNCS.evaluate_round``."""
    from jackdaw.engine.back import Back
    from jackdaw.engine.economy import calculate_round_earnings
    from jackdaw.engine.jokers import round_dollar_bonus
    from jackdaw.engine.read import StateView
    from jackdaw.engine.tags import fire_tag_context

    blind = gs["blind"]
    target = blind.chips if blind_target is None else blind_target
    reward = blind.dollars if blind_reward is None else blind_reward
    boss_defeated = bool(blind.boss) if was_boss is None else was_boss

    # L: state_events.lua:1139-1155 — the blind row is based on committed
    # chips, so a Mr. Bones save below target records $0 before defeat cleanup.
    earned_blind_reward = reward if gs.get("chips", 0) >= target else 0
    blind.defeat(gs)

    # L: state_events.lua:1157-1163 — Back eval follows the queued defeat
    # point. Preserve the defeated-boss fact after the blind setter clears it.
    back_result = Back(gs.get("selected_back_key", "b_red")).trigger_effect(
        "eval", boss_defeated=boss_defeated
    )
    if back_result and back_result.get("create_tag"):
        back_queue = EffectQueue(gs)
        back_queue.add(AddTag(key=back_result["create_tag"]))
        back_queue.apply()

    # L: state_events.lua:1165-1203 — dollar bonuses are calculated NOW,
    # after F6 mutation/perishing; eval tags are rows, not immediate money.
    jokers = gs.get("jokers", [])
    joker_dollars = round_dollar_bonus(jokers, StateView(gs, jokers=jokers))
    tag_dollars = sum(
        result.dollars
        for _entry, result in fire_tag_context(
            gs,
            "eval",
            last_blind_is_boss=boss_defeated,
        )
    )
    cr = gs["current_round"]
    earnings = calculate_round_earnings(
        blind=blind,
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        money=gs.get("dollars", 0),
        jokers=jokers,
        game_state=gs,
        rng=gs.get("rng"),
        joker_dollars=joker_dollars,
        blind_reward=earned_blind_reward,
        tag_dollars=tag_dollars,
        rental_cost=rental_cost,
    )
    gs["round_earnings"] = earnings
    # L: state_events.lua:1205-1208; common_events.lua:1064-1088 — bottom row.
    cr["dollars"] = earnings.total


def _advance_ante(gs: dict[str, Any]) -> None:
    """Run the boss-only ante/voucher part of F6."""
    rr = gs["round_resets"]
    rr["ante"] += 1

    rng = gs.get("rng")
    if rng:
        from jackdaw.engine.vouchers import get_next_voucher_key

        # L: state_events.lua:238-264 — ante changes and next voucher happen
        # at F6. Tags, boss choice, and blind-state reset wait for cash-out.
        gs["current_round"]["voucher"] = get_next_voucher_key(
            rng,
            gs.get("used_vouchers", {}),
            in_shop=None,
            ante=rr["ante"],
        )


# ---------------------------------------------------------------------------
# setting_blind joker context
# ---------------------------------------------------------------------------


def _fire_setting_blind(gs: dict[str, Any], *, queue: EffectQueue | None = None) -> None:
    """Fire the live-state ``setting_blind`` Joker pass."""
    from jackdaw.engine.jokers import fire_jokers

    target_queue = queue or EffectQueue(gs)
    fire_jokers(gs, target_queue, setting_blind=True)
    if queue is None:
        target_queue.apply()


# ---------------------------------------------------------------------------
# Boss blind set-time effects
# ---------------------------------------------------------------------------


def _apply_boss_blind_effects(
    gs: dict[str, Any],
    blind: Any,
    *,
    queue: EffectQueue | None = None,
) -> None:
    """Apply boss blind effects at set-time (blind.lua:157-209).

    These are one-time mutations that happen when the blind is set,
    before the round starts.
    """
    target_queue = queue or EffectQueue(gs)
    cr = gs.get("current_round", {})
    name = getattr(blind, "name", "")

    # L: blind.lua:176-184 — Water and Needle snapshot immediately and queue
    # their resource changes.
    if name == "The Water":
        current_discards = cr.get("discards_left", 0)
        blind.discards_sub = current_discards
        target_queue.add(ChangeRoundResource(discards=-current_discards))

    elif name == "The Needle":
        rr = gs.get("round_resets", {})
        current_hands = rr.get("hands", 4)
        blind.hands_sub = current_hands - 1
        target_queue.add(ChangeRoundResource(hands=-blind.hands_sub))

    # L: blind.lua:186-188 — change_size itself is queued.
    elif name == "The Manacle":
        target_queue.add(ChangeHandSize(delta=-1))

    # L: blind.lua:190-205 — flip now; the parent event later emits three
    # aajk shuffles. order=1 keeps them behind first-level setting effects.
    elif name == "Amber Acorn":
        jokers: list = gs.get("jokers", [])
        if jokers:
            for j in jokers:
                j.facing = "back"
            if len(jokers) > 1:
                for _ in range(3):
                    target_queue.add(ShuffleArea(area="jokers", seed_key="aajk", order=1))

    # The Eye's history lives in Blind.hands_used, reset on Blind creation.

    # The Mouth: reset only_hand
    elif name == "The Mouth":
        blind.only_hand = None

    if queue is None:
        target_queue.apply()


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
    queue: EffectQueue,
) -> None:
    """Fire boss blind press_play effects before scoring.

    Mirrors ``Blind:press_play`` (blind.lua:464-502).
    """
    if getattr(blind, "disabled", False):
        return

    name = getattr(blind, "name", "")

    if name == "The Hook":
        # L: blind.lua:466-487 — copy the hand, take two hook-stream samples
        # in selection order, then route the highlighted cards through the
        # same discard function with hook=true.
        hand: list = gs.get("hand", [])
        candidates = list(hand)
        selected: list = []
        for _ in range(min(2, len(hand))):
            if candidates and rng:
                seed_val = rng.seed("hook")
                target, _ = rng.element(candidates, seed_val)
                selected.append(target)
                candidates.remove(target)
        if selected:
            discard_cards_from_highlighted(gs, selected, hook=True)
        blind.triggered = True

    elif name == "The Tooth":
        # L: blind.lua:497-504 — ordinary (unbuffered) queued ease at F1.
        for card in played:
            queue.add(EaseDollars(amount=-1, source=card))
        blind.triggered = True

    elif name == "The Fish":
        # L: blind.lua:494-496.
        blind.prepped = True

    elif name == "Crimson Heart":
        # L: blind.lua:488-493.
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
