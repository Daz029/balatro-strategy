"""Pure, live reads over engine game state.

These getters mirror values that Balatro reads directly or recomputes in
``Card:update``.  They deliberately do not cache across calls.  A missing key
falls back to ``init_game_object``'s value (only hand-built test states lack
one; every real run state carries them).  ``StateView``
adds scoring-call-local laziness by caching each property on one view instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from jackdaw.engine.data.hands import HAND_ORDER

if TYPE_CHECKING:
    from jackdaw.engine.card import Card


def money_committed(gs: dict[str, Any]) -> int:
    """Return committed dollars (Vagabond, ``card.lua:3744``)."""
    return gs.get("dollars", 0)


def money_with_buffer(gs: dict[str, Any]) -> int:
    """Return dollars including pending changes (Bull, ``card.lua:3936-3939``)."""
    return gs.get("dollars", 0) + gs.get("dollar_buffer", 0)


def probability(gs: dict[str, Any]) -> float:
    """Return the normal probability numerator, defaulting to Lua's 1."""
    return float(gs.get("probabilities", {}).get("normal", 1))


def has_voucher(gs: dict[str, Any], key: str) -> bool:
    """Return whether *key* is truthy in ``G.GAME.used_vouchers``."""
    return bool(gs.get("used_vouchers", {}).get(key))


def find_joker(
    gs: dict[str, Any],
    name: str,
    *,
    include_debuffed: bool = False,
) -> list[Card]:
    """Find named jokers and consumables (``misc_functions.lua:903-917``).

    Lua confusingly names its second parameter ``non_debuff``: when true it
    *includes* debuffed cards.  ``include_debuffed`` names that behavior
    directly.  By default debuffed matches are excluded.
    """
    matches: list[Card] = []
    for card in (*gs.get("jokers", []), *gs.get("consumables", [])):
        if card.ability.get("name") == name and (include_debuffed or not card.debuff):
            matches.append(card)
    return matches


def joker_count(jokers: list[Card]) -> int:
    """Return ``#G.jokers.cards``."""
    return len(jokers)


def stencil_xmult(jokers: list[Card], joker_slots: int) -> int:
    """Return Joker Stencil's live Xmult (``card.lua:4203-4207``)."""
    stencils = sum(card.ability.get("name") == "Joker Stencil" for card in jokers)
    return joker_slots - len(jokers) + stencils


def swashbuckler_mult(jokers: list[Card], card: Card) -> int:
    """Return other jokers' sell-cost total (``card.lua:4240-4247``)."""
    return sum(joker.sell_cost for joker in jokers if joker is not card)


def temperance_money(jokers: list[Card], cap: int) -> int:
    """Return capped Joker sell value (``card.lua:4167-4175``)."""
    total = sum(joker.sell_cost for joker in jokers if joker.ability.get("set") == "Joker")
    return min(total, cap)


def editionless_jokers(jokers: list[Card]) -> list[Card]:
    """Return editionless Jokers (``card.lua:4209-4223``), debuffed included."""
    return [joker for joker in jokers if joker.ability.get("set") == "Joker" and not joker.edition]


_PLAYING_CARD_AREAS = ("deck", "hand", "discard_pile", "played_cards_area", "pack_hand")


def playing_cards(gs: dict[str, Any]) -> list[Card]:
    """Return all owned playing cards, deduplicated by identity.

    This reconstructs Lua's ``G.playing_cards`` from the engine's physical
    card areas.  A pack hand can alias the ordinary hand, so equality is not
    sufficient and the first occurrence of each object is retained.
    """
    if not gs.get("pack_hand"):
        # Fast path (the solver's inner loop): without an open pack the areas
        # are disjoint, so a plain concatenation is already duplicate-free.
        return [
            *gs.get("deck", []),
            *gs.get("hand", []),
            *gs.get("discard_pile", []),
            *gs.get("played_cards_area", []),
        ]
    result: list[Card] = []
    seen: set[int] = set()
    for area in _PLAYING_CARD_AREAS:
        for card in gs.get(area, []):
            identity = id(card)
            if identity not in seen:
                seen.add(identity)
                result.append(card)
    return result


def playing_card_count(gs: dict[str, Any]) -> int:
    """Return ``#G.playing_cards`` (for example, Erosion at ``card.lua:3894``)."""
    return len(playing_cards(gs))


def count_enhancement(gs: dict[str, Any], center_key: str) -> int:
    """Count an enhancement as in ``Card:update`` (``card.lua:4185-4201``)."""
    return sum(card.center_key == center_key for card in playing_cards(gs))


def enhanced_count(gs: dict[str, Any]) -> int:
    """Count non-base cards for Driver's License (``card.lua:4179-4183``)."""
    return sum(card.center_key != "c_base" for card in playing_cards(gs))


def rank_count(gs: dict[str, Any], rank_id: int) -> int:
    """Count live card IDs for Cloud 9 (``card.lua:4191-4195``)."""
    return sum(card.get_id() == rank_id for card in playing_cards(gs))


def deck_enhancements(gs: dict[str, Any]) -> set[str]:
    """Return enhancement keys used by the pool gate (``common_events.lua:2012-2018``)."""
    return {card.center_key for card in playing_cards(gs)}


def deck_cards_remaining(gs: dict[str, Any]) -> int:
    """Return ``#G.deck.cards`` (Blue Joker, ``card.lua:3887-3890``)."""
    return len(gs.get("deck", []))


def hands_left(gs: dict[str, Any]) -> int:
    """Return ``G.GAME.current_round.hands_left``."""
    return gs.get("current_round", {}).get("hands_left", 0)


def hands_played_this_round(gs: dict[str, Any]) -> int:
    """Return ``G.GAME.current_round.hands_played``."""
    return gs.get("current_round", {}).get("hands_played", 0)


def discards_left(gs: dict[str, Any]) -> int:
    """Return ``G.GAME.current_round.discards_left``."""
    return gs.get("current_round", {}).get("discards_left", 0)


def discards_used(gs: dict[str, Any]) -> int:
    """Return ``G.GAME.current_round.discards_used``."""
    return gs.get("current_round", {}).get("discards_used", 0)


def last_blind(gs: dict[str, Any]) -> Any:
    """Return the blind that Lua exposes as ``G.GAME.last_blind``.

    Lua sets ``last_blind`` in ``Blind:set_blind`` (``blind.lua:97-99``).
    Its only readers, Investment Tag (``tag.lua:119``) and Anaglyph Deck
    (``back.lua:111``), run synchronously in ``evaluate_round`` before the
    queued ``Blind:defeat`` event resets it.  At every read it is therefore
    the just-defeated blind, which ``gs["blind"]`` still holds.
    """
    return gs.get("blind")


def telescope_hand(gs: dict[str, Any]) -> str | None:
    """Return Telescope's most-played visible hand (``card.lua:1737-1744``)."""
    hand_levels = gs["hand_levels"]
    selected: str | None = None
    tally = 0
    for hand_type in HAND_ORDER:
        hand = hand_levels[hand_type]
        if hand.visible and hand.played > tally:
            selected = hand_type.value
            tally = hand.played
    return selected


def tarot_usage(gs: dict[str, Any]) -> int:
    """Return Fortune Teller's Tarot total (``card.lua:4016-4019``)."""
    return gs.get("consumeable_usage_total", {}).get("tarot", 0)


def planets_used(gs: dict[str, Any]) -> int:
    """Distinct Planet keys used this run (Satellite, ``card.lua:1667-1674``).

    Counts ``consumeable_usage`` entries whose ``set`` is ``"Planet"``, so Black
    Hole (a Spectral) never counts.
    """
    return sum(entry.get("set") == "Planet" for entry in gs.get("consumeable_usage", {}).values())


def idol_card(gs: dict[str, Any]) -> dict[str, Any] | None:
    """Return The Idol target (``card.lua:3127-3129``)."""
    return gs.get("current_round", {}).get("idol_card")


def mail_card_id(gs: dict[str, Any]) -> int | None:
    """Return Mail-In Rebate's rank ID (``card.lua:2825-2827``)."""
    return gs.get("current_round", {}).get("mail_card", {}).get("id")


def ancient_suit(gs: dict[str, Any]) -> str | None:
    """Return Ancient Joker's target suit (``card.lua:3255-3256``)."""
    return gs.get("current_round", {}).get("ancient_card", {}).get("suit")


def castle_suit(gs: dict[str, Any]) -> str | None:
    """Return Castle's target suit (``card.lua:2814-2816``)."""
    return gs.get("current_round", {}).get("castle_card", {}).get("suit")


@dataclass(frozen=True)
class Rules:
    """Active non-debuffed joker rules derived through Lua ``find_joker``."""

    pareidolia: bool = False
    smeared: bool = False
    four_fingers: bool = False
    shortcut: bool = False
    splash: bool = False
    showman: bool = False


_RULE_NAMES = {
    "pareidolia": "Pareidolia",
    "smeared": "Smeared Joker",
    "four_fingers": "Four Fingers",
    "shortcut": "Shortcut",
    "splash": "Splash",
    "showman": "Showman",
}

_RULE_KEYS = {
    "pareidolia": "j_pareidolia",
    "smeared": "j_smeared",
    "four_fingers": "j_four_fingers",
    "shortcut": "j_shortcut",
    "splash": "j_splash",
    "showman": "j_ring_master",
}


def rules_for(jokers: list[Card]) -> Rules:
    """Build active rules from *jokers*, excluding debuffed cards."""
    active = [joker for joker in jokers if not joker.debuff]
    active_names = {joker.ability.get("name") for joker in active}
    active_keys = {joker.center_key for joker in active}
    return Rules(
        **{
            flag: name in active_names or _RULE_KEYS[flag] in active_keys
            for flag, name in _RULE_NAMES.items()
        }
    )


def rules(gs: dict[str, Any]) -> Rules:
    """Build active rules from the state's joker area."""
    return rules_for(gs.get("jokers", []))


class StateView:
    """Lazy, scoring-call-local read-only view over live game state."""

    def __init__(
        self,
        gs: dict[str, Any],
        jokers: list[Card] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> None:
        """Build a view, optionally pinning named properties for this view.

        Overrides are scoring-call-local values supplied explicitly by a
        caller.  Every key must name an existing ``StateView`` attribute;
        misspellings raise instead of silently creating a dead field.
        """
        self._gs = gs
        self._jokers = gs.get("jokers", []) if jokers is None else jokers
        for name, value in (overrides or {}).items():
            if not isinstance(vars(type(self)).get(name), cached_property):
                raise KeyError(name)
            self.__dict__[name] = value

    @cached_property
    def gs(self) -> dict[str, Any]:
        return self._gs

    @cached_property
    def joker_count(self) -> int:
        return joker_count(self._jokers)

    @cached_property
    def joker_slots(self) -> int:
        return self._gs.get("joker_slots", 5)

    @cached_property
    def money(self) -> int:
        return money_committed(self._gs)

    @cached_property
    def money_with_buffer(self) -> int:
        return money_with_buffer(self._gs)

    @cached_property
    def deck_cards_remaining(self) -> int:
        return deck_cards_remaining(self._gs)

    @cached_property
    def starting_deck_size(self) -> int:
        return self._gs.get("starting_deck_size", 52)

    @cached_property
    def playing_cards(self) -> list[Card]:
        # Collected once per view: every tally below counts over this list,
        # so a board of tally jokers pays for one area walk, not one each.
        return playing_cards(self._gs)

    @cached_property
    def playing_cards_count(self) -> int:
        return len(self.playing_cards)

    @cached_property
    def stone_tally(self) -> int:
        return sum(card.center_key == "m_stone" for card in self.playing_cards)

    @cached_property
    def steel_tally(self) -> int:
        return sum(card.center_key == "m_steel" for card in self.playing_cards)

    @cached_property
    def nine_tally(self) -> int:
        return sum(card.get_id() == 9 for card in self.playing_cards)

    @cached_property
    def enhanced_card_count(self) -> int:
        return sum(card.center_key != "c_base" for card in self.playing_cards)

    @cached_property
    def hands_left(self) -> int:
        return hands_left(self._gs)

    @cached_property
    def hands_played(self) -> int:
        return hands_played_this_round(self._gs)

    @cached_property
    def discards_left(self) -> int:
        return discards_left(self._gs)

    @cached_property
    def discards_used(self) -> int:
        return discards_used(self._gs)

    @cached_property
    def probabilities_normal(self) -> float:
        return probability(self._gs)

    @cached_property
    def consumable_usage_tarot(self) -> int:
        return tarot_usage(self._gs)

    @cached_property
    def planets_used(self) -> int:
        return planets_used(self._gs)

    @cached_property
    def mail_card_id(self) -> int | None:
        return mail_card_id(self._gs)

    @cached_property
    def idol_card(self) -> dict[str, Any] | None:
        return idol_card(self._gs)

    @cached_property
    def ancient_suit(self) -> str | None:
        return ancient_suit(self._gs)

    @cached_property
    def castle_card_suit(self) -> str | None:
        return castle_suit(self._gs)

    @cached_property
    def skips(self) -> int:
        return self._gs.get("skips", 0)

    @cached_property
    def rules(self) -> Rules:
        return rules_for(self._jokers)
