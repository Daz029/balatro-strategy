"""Canonical card lifetime operations.

This module centralises the bookkeeping performed by Lua's ``Card`` and
``CardArea`` methods: pool registration in ``Card:set_ability``
(``card.lua:349-354``), copying in ``copy_card``
(``functions/common_events.lua:2156``), and teardown in ``Card:remove``
(``card.lua:4727-4760``).
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from jackdaw.engine import read
from jackdaw.engine.card import Card
from jackdaw.engine.data.prototypes import _load_json


def _build_name_keys() -> dict[str, tuple[str, ...]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for key, center in _load_json("centers.json").items():
        grouped[center.get("name", "")].append(key)
    return {name: tuple(keys) for name, keys in grouped.items()}


_NAME_TO_KEYS = _build_name_keys()


_PLAYING_CARD_SETS = frozenset({"Default", "Enhanced"})


def _tracked(card: Card) -> bool:
    """Whether *card* takes part in pool exclusion.

    Lua's ``set_ability`` also registers playing cards' centers (``c_base``,
    ``m_glass``...), but no pool ever reads those keys: the Enhanced pool
    skips ``used_jokers`` (pools.py) and Base is a forced key. Their Lua value
    ("set since the last removal of any card with that name") is not a
    function of current state and is unobservable, so playing cards are
    left out and the set stays exactly "every non-playing card that exists".
    """
    return card.ability.get("set") not in _PLAYING_CARD_SETS


class PoolTracker:
    """Maintain Lua's current-card ``G.GAME.used_jokers`` exclusion set."""

    @staticmethod
    def register(gs: dict[str, Any], card: Card) -> None:
        """Register every same-named center (``card.lua:349-354``)."""
        if not _tracked(card):
            return
        used = gs.setdefault("used_jokers", {})
        for key in _NAME_TO_KEYS.get(card.ability.get("name", ""), ()):
            used[key] = True

    @staticmethod
    def release(gs: dict[str, Any], card: Card) -> None:
        """Release a removed card's keys (``card.lua:4741-4749``).

        Lua's confusing ``find_joker(name, true)`` includes debuffed owned
        cards, so a debuffed duplicate in either owned area keeps the key.
        The caller must first remove *card* from its area.
        """
        if not _tracked(card):
            return
        name = card.ability.get("name", "")
        if read.find_joker(gs, name, include_debuffed=True):
            return
        used = gs.setdefault("used_jokers", {})
        for key in _NAME_TO_KEYS.get(name, ()):
            used.pop(key, None)

    @staticmethod
    def expected(gs: dict[str, Any]) -> dict[str, bool]:
        """Return registrations implied by every card that currently exists."""
        expected: dict[str, bool] = {}
        for card in read.all_cards(gs):
            if not _tracked(card):
                continue
            for key in _NAME_TO_KEYS.get(card.ability.get("name", ""), ()):
                expected[key] = True
        return expected


def copy_card(
    gs: dict[str, Any],
    source: Card,
    *,
    into: Card | None = None,
    strip_edition: bool = False,
) -> Card:
    """Copy *source* like Lua ``copy_card`` (``common_events.lua:2156``)."""
    result = into if into is not None else Card()
    result.set_ability(source.center_key, gs=gs)
    result.ability["type"] = source.ability.get("type", "")

    if source.base is None:
        result.base = None
        result.card_key = None
    else:
        result.set_base(
            source.card_key or "",
            source.base.suit.value,
            source.base.rank.value,
            gs=gs,
        )

    for key, value in source.ability.items():
        result.ability[key] = copy.deepcopy(value) if isinstance(value, (dict, list)) else value

    if not strip_edition:
        # Lua passes ``other.edition or {}``: the empty table clears and
        # reprices, while nil would return without repricing.
        result.set_edition(gs, source.edition or {})
    # strip_edition: Lua skips set_edition entirely, so a fresh copy keeps
    # its nil edition and a copy-into target keeps its own.
    result.set_seal(gs, source.seal)

    result.eternal = source.eternal
    result.perishable = source.perishable
    result.perish_tally = source.perish_tally
    result.rental = source.rental
    result.set_cost(gs)
    result.debuff = source.debuff
    result.pinned = getattr(source, "pinned", False)
    # Lua's copy_card never touches added_to_deck / removed: a fresh Card
    # starts with both False, and a copy-into target (Death) keeps its own.
    PoolTracker.register(gs, result)
    return result


def emplace(gs: dict[str, Any], card: Card, area: str) -> None:
    """Place *card* in a state area, mirroring ``CardArea:emplace``."""
    cards = gs.setdefault(area, [])
    if any(existing is card for existing in cards):
        return

    # Factories register at set_ability time like Lua. Keeping placement
    # defensive makes restored/debug-created cards obey the same invariant.
    PoolTracker.register(gs, card)
    if area in {"jokers", "consumables", "deck", "hand"}:
        card.add_to_deck(gs)
    if area == "deck":
        # CardArea:emplace inserts at index 1 for deck-type areas
        # (cardarea.lua:33). Our deck draws from the END, so Lua's index 1
        # is our index 0 (the bottom); the position feeds later shuffles.
        cards.insert(0, card)
    else:
        cards.append(card)

    if area == "hand":
        from jackdaw.engine.actions import GamePhase

        if gs.get("phase") == GamePhase.PACK_OPENING:
            pack_hand = gs.setdefault("pack_hand", [])
            if not any(existing is card for existing in pack_hand):
                pack_hand.append(card)
        # No sort here: Lua's emplace does not sort. Callers that re-sort the
        # hand after a batch (as the engine did before Phase 3) do so
        # explicitly.


_AREAS = (
    "jokers",
    "consumables",
    "deck",
    "hand",
    "discard_pile",
    "played_cards_area",
    "shop_cards",
    "shop_vouchers",
    "shop_boosters",
    "pack_cards",
    "pack_hand",
)


def remove(gs: dict[str, Any], card: Card) -> None:
    """Remove *card* like Lua ``Card:remove`` (``card.lua:4727-4760``)."""
    if card.removed:
        return
    card.removed = True

    # ``pack_hand`` aliases cards in ``hand``; remove every identity alias so
    # expected registrations cannot retain a card that no longer exists.
    for area in _AREAS:
        cards = gs.get(area)
        if cards:
            cards[:] = [existing for existing in cards if existing is not card]

    card.remove_from_deck(gs)
    if card.ability.pop("queue_negative_removal", False):
        slot = "consumable_slots" if card.ability.get("consumeable") else "joker_slots"
        gs[slot] = gs.get(slot, 0) - 1
    PoolTracker.release(gs, card)
