"""Action legality — the single source of truth (engine Phase 6, audit S5).

Decision 2026-10-12: the engine exposes the FULL vanilla action set; the RL
environments restrict it with their own masks (reconstructed later — see
CLAUDE.md TODO 4). Before this module, legality lived in two places that
disagreed: ``actions.get_legal_actions`` enumerated, the ``_handle_*``
functions in ``game.py`` validated, and both were partial (audit D56) or
carried bot-only restrictions (D23, D57).

Now :func:`illegal_reason` is the only predicate. :func:`~jackdaw.engine.game.step`
raises :class:`~jackdaw.engine.game.IllegalActionError` with its reason, and
:func:`~jackdaw.engine.actions.get_legal_actions` enumerates candidates and
keeps exactly those it accepts. Every rule cites the Lua 1.0.1o line it ports.

PlayHand / Discard: :func:`get_legal_actions` keeps its MARKER convention
(empty ``card_indices`` = "playing is possible", see :func:`can_play` /
:func:`can_discard`); a concrete action with indices is checked here.
"""

from __future__ import annotations

from typing import Any

from jackdaw.engine.actions import (
    Action,
    BuyAndUse,
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
    RerollBoss,
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

# Stable decision phases. Lua's sell/use/arrange checks gate on "nothing in
# play and controller unlocked", which holds at every one of these points.
STABLE_PHASES = frozenset(
    {
        GamePhase.BLIND_SELECT,
        GamePhase.SELECTING_HAND,
        GamePhase.ROUND_EVAL,
        GamePhase.SHOP,
        GamePhase.PACK_OPENING,
    }
)

# Pack types whose Lua state shows a hand (TAROT_PACK, SPECTRAL_PACK,
# PLANET_PACK in can_use_consumeable's hand-needing branch, card.lua:1565).
_HAND_PACK_TYPES = frozenset({"Arcana", "Spectral", "Celestial"})

_CONSUMABLE_SETS = frozenset({"Tarot", "Planet", "Spectral"})
_BOSS_REROLL_COST = 10  # button_callbacks.lua:2785, 2803


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _phase(gs: dict[str, Any]) -> GamePhase | None:
    raw = gs.get("phase")
    if raw is None:
        return None
    return GamePhase(raw) if isinstance(raw, str) else raw


def spendable(gs: dict[str, Any]) -> int:
    """``G.GAME.dollars - G.GAME.bankrupt_at`` (Credit Card lowers the floor)."""
    return gs.get("dollars", 0) - gs.get("bankrupt_at", 0)


def affordable(gs: dict[str, Any], cost: int) -> bool:
    """Lua ``can_buy`` / ``can_open`` (button_callbacks.lua:56, 112).

    Unaffordable only when ``cost > dollars - bankrupt_at`` AND ``cost > 0``:
    a free card is always affordable, even in debt (audit D50).
    """
    return cost <= 0 or cost <= spendable(gs)


def _card_set(card: Any) -> str:
    ability = getattr(card, "ability", None)
    return ability.get("set", "") if isinstance(ability, dict) else ""


def _is_negative(card: Any) -> bool:
    edition = getattr(card, "edition", None)
    return bool(isinstance(edition, dict) and edition.get("negative"))


def has_room(gs: dict[str, Any], card: Any) -> bool:
    """``G.FUNCS.check_for_buy_space`` (button_callbacks.lua:2392-2401).

    Vouchers and playing cards always fit. A Joker or consumable needs
    ``#area < limit``, with one extra slot when the card itself is Negative.
    """
    card_set = _card_set(card)
    bonus = 1 if _is_negative(card) else 0
    if card_set == "Joker":
        return len(gs.get("jokers", [])) < gs.get("joker_slots", 5) + bonus
    if card_set in _CONSUMABLE_SETS:
        return len(gs.get("consumables", [])) < gs.get("consumable_slots", 2) + bonus
    return True


def forced_card(gs: dict[str, Any]) -> Any | None:
    """The Cerulean Bell card held highlighted (``blind.lua:575-586``)."""
    for card in gs.get("hand", []):
        ability = getattr(card, "ability", None)
        if isinstance(ability, dict) and ability.get("forced_selection"):
            return card
    return None


def _index_reason(idx: int, n: int, what: str) -> str | None:
    if not isinstance(idx, int) or idx < 0 or idx >= n:
        return f"invalid {what} index {idx!r} (have {n})"
    return None


def _indices_reason(
    indices: tuple[int, ...] | None,
    n: int,
    what: str,
    *,
    lo: int,
    hi: int,
) -> str | None:
    """Indices must be in range, unique and lo..hi long (audit D56)."""
    indices = tuple(indices or ())
    if len(indices) < lo:
        return f"must select at least {lo} {what}"
    if len(indices) > hi:
        return f"cannot select more than {hi} {what}"
    if len(set(indices)) != len(indices):
        return f"duplicate {what} indices {indices}"
    for i in indices:
        reason = _index_reason(i, n, what)
        if reason:
            return reason
    return None


def _hand_state(gs: dict[str, Any], phase: GamePhase | None) -> bool:
    """Lua states where a hand is on screen for selection (card.lua:1565)."""
    if phase == GamePhase.SELECTING_HAND:
        return True
    return phase == GamePhase.PACK_OPENING and gs.get("pack_type", "") in _HAND_PACK_TYPES


def _needs_hand(card: Any) -> bool:
    """Whether ``can_use_consumeable`` reaches its hand-only branch (card.lua:1565-1576)."""
    from jackdaw.engine.consumables import _NEED_HAND_CARDS, _resolve_consumable_config

    if getattr(card, "center_key", "") in _NEED_HAND_CARDS:
        return True
    return bool(_resolve_consumable_config(card).get("max_highlighted"))


def _use_reason(
    gs: dict[str, Any],
    phase: GamePhase | None,
    card: Any,
    targets: tuple[int, ...] | None,
) -> str | None:
    """Lua ``Card:can_use_consumeable`` (card.lua:1523-1579) for *card*.

    Targets are indices into ``gs["hand"]`` (the dealt pack hand while an
    Arcana/Spectral pack is open); any bad index is rejected, never filtered.
    """
    from jackdaw.engine.consumables import can_use_consumable

    hand = gs.get("hand", [])
    targets = tuple(targets or ())
    if targets:
        reason = _indices_reason(targets, len(hand), "target", lo=1, hi=len(hand) or 1)
        if reason:
            return reason
    if _needs_hand(card) and not _hand_state(gs, phase):
        return "consumable needs a hand on screen (card.lua:1565)"
    ok = can_use_consumable(
        card,
        highlighted=[hand[i] for i in targets],
        hand_cards=hand,
        jokers=gs.get("jokers", []),
        consumables=gs.get("consumables", []),
        joker_limit=gs.get("joker_slots", 5),
        consumable_limit=gs.get("consumable_slots", 2),
        game_state=gs,
    )
    if not ok:
        name = card.ability.get("name", card.center_key)
        return f"consumable {name!r} cannot be used now"
    return None


def _reroll_cost(gs: dict[str, Any]) -> int:
    cr = gs.get("current_round", {})
    if cr.get("free_rerolls", 0) > 0:
        return 0
    return cr.get("reroll_cost", 5)


def with_forced_card(gs: dict[str, Any], indices: tuple[int, ...]) -> tuple[int, ...]:
    """Make a hand selection include Cerulean Bell's forced card.

    In Lua the forced card is pre-highlighted and counts toward the 5-card
    limit, so a player's selection always contains it. For a selection that
    omits it: add it, replacing the LAST pick when the selection is already
    5 cards. Index order is otherwise preserved (it is the play order,
    NEW-P5-1-05). No-op without a forced card or when it is already present.
    """
    forced = forced_card(gs)
    if forced is None:
        return tuple(indices)
    hand = gs.get("hand", [])
    forced_idx = next(i for i, c in enumerate(hand) if c is forced)
    picks = list(indices)
    if forced_idx in picks:
        return tuple(picks)
    if len(picks) >= 5:
        picks = picks[:4]
    return (*picks, forced_idx)


# ---------------------------------------------------------------------------
# Marker predicates (get_legal_actions' PlayHand / Discard markers)
# ---------------------------------------------------------------------------


def can_play(gs: dict[str, Any]) -> bool:
    """Some concrete PlayHand is legal (``G.FUNCS.can_play``, l.2048)."""
    return (
        _phase(gs) == GamePhase.SELECTING_HAND
        and bool(gs.get("hand"))
        and gs.get("current_round", {}).get("hands_left", 0) > 0
    )


def can_discard(gs: dict[str, Any]) -> bool:
    """Some concrete Discard is legal (``G.FUNCS.can_discard``, l.2092)."""
    return (
        _phase(gs) == GamePhase.SELECTING_HAND
        and bool(gs.get("hand"))
        and gs.get("current_round", {}).get("discards_left", 0) > 0
    )


# ---------------------------------------------------------------------------
# The predicate
# ---------------------------------------------------------------------------


def illegal_reason(gs: dict[str, Any], action: Action) -> str | None:
    """Return why *action* is illegal in *gs*, or ``None`` when it is legal."""
    phase = _phase(gs)
    if phase is None or phase == GamePhase.GAME_OVER:
        return f"no actions are legal in phase {phase!r}"

    def need(*phases: GamePhase) -> str | None:
        if phase not in phases:
            return f"{type(action).__name__} not valid in phase {phase!r}"
        return None

    match action:
        # -- Blind select --------------------------------------------------
        case SelectBlind():
            return need(GamePhase.BLIND_SELECT)
        case SkipBlind():
            if reason := need(GamePhase.BLIND_SELECT):
                return reason
            if gs.get("blind_on_deck", "Small") == "Boss":
                return "cannot skip the Boss blind"
            return None
        case RerollBoss():
            # L: button_callbacks.lua:2784-2797 (reroll_boss_button).
            if reason := need(GamePhase.BLIND_SELECT):
                return reason
            if spendable(gs) - _BOSS_REROLL_COST < 0:
                return "cannot afford the $10 boss reroll"
            vouchers = gs.get("used_vouchers", {})
            if vouchers.get("v_retcon"):
                return None
            if vouchers.get("v_directors_cut") and not gs.get("round_resets", {}).get(
                "boss_rerolled"
            ):
                return None
            return "boss reroll needs Retcon, or Director's Cut unused this ante"

        # -- Hand ------------------------------------------------------------
        case PlayHand(card_indices=indices) | Discard(card_indices=indices):
            if reason := need(GamePhase.SELECTING_HAND):
                return reason
            cr = gs.get("current_round", {})
            if isinstance(action, PlayHand) and cr.get("hands_left", 0) <= 0:
                return "no hands remaining"
            if isinstance(action, Discard) and cr.get("discards_left", 0) <= 0:
                return "no discards remaining"
            hand = gs.get("hand", [])
            if reason := _indices_reason(indices, len(hand), "card", lo=1, hi=5):
                return reason
            # L: blind.lua:575-586 + cardarea.lua highlight rules — Cerulean
            # Bell's card is highlighted and cannot be unhighlighted, so every
            # play/discard includes it (audit D27).
            forced = forced_card(gs)
            if forced is not None and not any(hand[i] is forced for i in indices):
                return "Cerulean Bell: the forced card must be selected"
            return None
        case SortHand(mode=mode):
            if reason := need(GamePhase.SELECTING_HAND):
                return reason
            return None if mode in ("rank", "suit") else f"unknown sort mode {mode!r}"
        case SwapHandLeft(idx=idx) | SwapHandRight(idx=idx):
            # The hand can be rearranged wherever it is on screen (audit D57).
            if not _hand_state(gs, phase):
                return f"no hand to arrange in phase {phase!r}"
            other = idx + (-1 if isinstance(action, SwapHandLeft) else 1)
            n = len(gs.get("hand", []))
            return _index_reason(idx, n, "hand") or _index_reason(other, n, "hand")
        case SwapJokersLeft(idx=idx) | SwapJokersRight(idx=idx):
            if phase not in STABLE_PHASES:
                return f"cannot arrange jokers in phase {phase!r}"
            other = idx + (-1 if isinstance(action, SwapJokersLeft) else 1)
            n = len(gs.get("jokers", []))
            return _index_reason(idx, n, "joker") or _index_reason(other, n, "joker")

        # -- Owned cards (any stable phase) ----------------------------------
        case SellCard(area=area, card_index=idx):
            # L: card.lua:1640-1652 Card:can_sell_card — any stable state
            # (the BLIND_SELECT exclusion is commented out in Lua), owned
            # joker/consumable rows only (both are type 'joker' areas,
            # game.lua:2235-2245), never Eternal (audit D23, D56).
            if phase not in STABLE_PHASES:
                return f"cannot sell in phase {phase!r}"
            if area not in ("jokers", "consumables"):
                return f"cannot sell from area {area!r}"
            cards = gs.get(area, [])
            if reason := _index_reason(idx, len(cards), area):
                return reason
            if getattr(cards[idx], "eternal", False):
                return "cannot sell an Eternal card"
            return None
        case UseConsumable(card_index=idx, target_indices=targets):
            if phase not in STABLE_PHASES:
                return f"cannot use consumables in phase {phase!r}"
            consumables = gs.get("consumables", [])
            if reason := _index_reason(idx, len(consumables), "consumable"):
                return reason
            return _use_reason(gs, phase, consumables[idx], targets)

        # -- Round eval ------------------------------------------------------
        case CashOut():
            return need(GamePhase.ROUND_EVAL)

        # -- Shop ------------------------------------------------------------
        case BuyCard(shop_index=idx):
            if reason := need(GamePhase.SHOP):
                return reason
            cards = gs.get("shop_cards", [])
            if reason := _index_reason(idx, len(cards), "shop"):
                return reason
            card = cards[idx]
            if not affordable(gs, card.cost):
                return "cannot afford card"
            if not has_room(gs, card):
                return "no room for card"
            return None
        case BuyAndUse(shop_index=idx, target_indices=targets):
            # L: button_callbacks.lua:77-79 can_buy_and_use; buy_from_shop
            # skips check_for_buy_space for buy_and_use (l.2407).
            if reason := need(GamePhase.SHOP):
                return reason
            cards = gs.get("shop_cards", [])
            if reason := _index_reason(idx, len(cards), "shop"):
                return reason
            card = cards[idx]
            if _card_set(card) not in _CONSUMABLE_SETS:
                return "only consumables can be bought and used"
            if not affordable(gs, card.cost):
                return "cannot afford card"
            return _use_reason(gs, phase, card, targets)
        case RedeemVoucher(card_index=idx):
            if reason := need(GamePhase.SHOP):
                return reason
            vouchers = gs.get("shop_vouchers", [])
            if reason := _index_reason(idx, len(vouchers), "voucher"):
                return reason
            # L: button_callbacks.lua:96-103 can_redeem — no free-card exemption.
            if vouchers[idx].cost > spendable(gs):
                return "cannot afford voucher"
            return None
        case OpenBooster(card_index=idx):
            if reason := need(GamePhase.SHOP):
                return reason
            boosters = gs.get("shop_boosters", [])
            if reason := _index_reason(idx, len(boosters), "booster"):
                return reason
            if not affordable(gs, boosters[idx].cost):
                return "cannot afford booster"
            return None
        case Reroll():
            if reason := need(GamePhase.SHOP):
                return reason
            # L: button_callbacks.lua:2076-2090 can_reroll.
            cost = _reroll_cost(gs)
            if spendable(gs) - cost < 0 and cost != 0:
                return "cannot afford reroll"
            return None
        case NextRound():
            return need(GamePhase.SHOP)

        # -- Pack opening ----------------------------------------------------
        case PickPackCard(card_index=idx, target_indices=targets):
            if reason := need(GamePhase.PACK_OPENING):
                return reason
            if gs.get("pack_choices_remaining", 0) <= 0:
                return "no pack choices remaining"
            pack = gs.get("pack_cards", [])
            if reason := _index_reason(idx, len(pack), "pack card"):
                return reason
            card = pack[idx]
            card_set = _card_set(card)
            if card_set == "Joker":
                # L: button_callbacks.lua:2112-2120 can_select_card.
                if not has_room(gs, card):
                    return "no joker slot for this pick"
                return None
            if card_set in _CONSUMABLE_SETS:
                # L: use_card on a pack consumable needs can_use_consumeable
                # (audit D57: an unusable pick is illegal, not a silent no-op).
                return _use_reason(gs, phase, card, targets)
            return None  # playing card (Standard pack)
        case SkipPack():
            return need(GamePhase.PACK_OPENING)

    return f"unknown action type {type(action).__name__}"


def is_legal(gs: dict[str, Any], action: Action) -> bool:
    """``True`` when :func:`illegal_reason` finds nothing wrong."""
    return illegal_reason(gs, action) is None
