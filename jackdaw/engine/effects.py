"""Effects: every non-scoring state change a handler can cause.

Phase 4 of the engine overhaul (S1, ``docs/engine_class_design_2026-10-06.md``
section 6). Handlers (jokers, consumables, tags, seals, decks) no longer return
untyped ``extra`` dicts that each call site interprets with its own partial
applier. They return ``Effect`` objects, and :func:`apply_effects` is the only
code that applies them.

Two Lua rules shape this module.

**Emission vs. event.** Lua handlers decide at *emission* time (inside the
``calculate_joker`` loop) and queue the visible change as an *event* that runs
after the loop. Some state is written at emission so that later handlers in the
same loop see it: the ``joker_buffer`` / ``consumeable_buffer`` room
reservations (``card.lua:2529``, ``3106``) and ``getting_sliced``
(``card.lua:2514``, ``2569``). :meth:`Effect.reserve` is the emission half and
runs when the effect is queued; :meth:`Effect.apply` is the event half.

**Room before RNG.** Every Lua creator checks
``#area.cards + buffer < card_limit`` before it draws a card (and, for 8 Ball,
Hallucination, Vagabond, Superposition and Séance, before its probability roll).
A creation with no room draws no RNG and registers nothing in the pool (D46).
Handlers read room through :meth:`EffectQueue.room`, and :class:`CreateCard`
clamps its count to the room left when it is queued.

Reservations live on the :class:`EffectQueue`, not in ``gs``. Lua's buffers are
zero at every stable decision point (each creation event resets them), so a
pass-local count is the same value. Keeping it local is what makes
counterfactual scoring safe: the solver calls ``score_hand`` against the live
``gs`` and never applies the queue, so nothing it reserves can leak.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jackdaw.engine import lifecycle

if TYPE_CHECKING:
    from jackdaw.engine.card import Card


_AREA_LIMIT_KEYS = {"jokers": "joker_slots", "consumables": "consumable_slots"}
_AREA_BUFFER_KEYS = {"jokers": "joker_buffer", "consumables": "consumeable_buffer"}
_AREA_LIMIT_DEFAULTS = {"jokers": 5, "consumables": 2}


class UnappliedEffectError(RuntimeError):
    """A handler produced effects at a call site that has no queue for them."""


class EffectQueue:
    """The effects emitted during one pass, plus that pass's room reservations."""

    def __init__(self, gs: dict[str, Any]) -> None:
        self.gs = gs
        self.effects: list[Effect] = []
        self.reserved: dict[str, int] = {"jokers": 0, "consumables": 0}

    def room(self, area: str) -> int:
        """Free slots in *area*: ``card_limit - #cards - buffer`` (Lua)."""
        gs = self.gs
        limit = gs.get(_AREA_LIMIT_KEYS[area], _AREA_LIMIT_DEFAULTS[area])
        used = len(gs.get(area, [])) + gs.get(_AREA_BUFFER_KEYS[area], 0)
        return limit - used - self.reserved[area]

    def add(self, effects: Effect | list[Effect] | tuple[Effect, ...] | None) -> None:
        """Queue effects, running each one's emission-time ``reserve``."""
        if effects is None:
            return
        if isinstance(effects, Effect):
            effects = [effects]
        for effect in effects:
            if not isinstance(effect, Effect):
                raise TypeError(f"not an Effect: {effect!r}")
            if effect.reserve(self):
                self.effects.append(effect)

    def apply(self) -> None:
        """Apply and clear everything queued so far."""
        pending, self.effects = self.effects, []
        self.reserved = {"jokers": 0, "consumables": 0}
        apply_effects(self.gs, pending)


def apply_effects(gs: dict[str, Any], effects: list[Effect]) -> None:
    """Apply *effects* in order. The single applier for every call site.

    Effects run in emission order, except that a higher ``order`` runs after
    every lower one (stable). That models Lua's nested events: Chicot and
    Cartomancer queue an event that queues another, so their change lands
    after the rest of the pass's first-level events (``card.lua:2493``,
    ``2547``).
    """
    for effect in sorted(effects, key=lambda e: e.order):
        if not isinstance(effect, Effect):
            raise TypeError(f"not an Effect: {effect!r}")
        effect.apply(gs)


@dataclass(kw_only=True)
class Effect(ABC):
    """One state change. Subclasses must implement :meth:`apply`."""

    source: Card | None = None
    order: int = 0

    def reserve(self, queue: EffectQueue) -> bool:
        """Emission-time half. Return False to drop the effect (no room)."""
        return True

    @abstractmethod
    def apply(self, gs: dict[str, Any]) -> None:
        """Event-time half: change the game state."""


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class EaseDollars(Effect):
    """``ease_dollars(amount)``. Phase 5 adds the buffered/instant split."""

    amount: int

    def apply(self, gs: dict[str, Any]) -> None:
        gs["dollars"] = gs.get("dollars", 0) + self.amount


@dataclass(kw_only=True)
class SetDollars(Effect):
    """Set money to *value* (Wraith: ``ease_dollars(-G.GAME.dollars, true)``)."""

    value: int

    def apply(self, gs: dict[str, Any]) -> None:
        gs["dollars"] = self.value


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class CreateCard(Effect):
    """``create_card(set, area, ...)`` from a pool, up to *count* cards.

    ``count`` is an upper bound. :meth:`reserve` clamps it to the room left
    in the destination (Riff-raff's ``min(2, limit - (#cards + buffer))``)
    and drops the effect when there is none, before any RNG is drawn.
    ``room_checked=False`` is for Lua creators that do not check room
    (none in vanilla today; kept explicit rather than implied).
    """

    set: str
    count: int = 1
    append: str = ""
    rarity: int | str | None = None
    forced_key: str | None = None
    room_checked: bool = True

    @property
    def area(self) -> str:
        return "jokers" if self.set == "Joker" else "consumables"

    def reserve(self, queue: EffectQueue) -> bool:
        if not self.room_checked:
            return self.count > 0
        self.count = min(self.count, queue.room(self.area))
        if self.count <= 0:
            return False
        queue.reserved[self.area] += self.count
        return True

    def apply(self, gs: dict[str, Any]) -> None:
        from jackdaw.engine.card_factory import resolve_create_descriptor

        descriptor: dict[str, Any] = {"type": self.set, "seed": self.append}
        if self.rarity is not None:
            descriptor["rarity"] = self.rarity
        if self.forced_key is not None:
            descriptor["forced_key"] = self.forced_key
        rng = gs.get("rng")
        ante = gs.get("round_resets", {}).get("ante", 1)
        for _ in range(self.count):
            card = resolve_create_descriptor(descriptor, rng, ante, gs)
            if card is not None:
                lifecycle.emplace(gs, card, self.area)


@dataclass(kw_only=True)
class CreatePlayingCard(Effect):
    """Create one playing card and place it in *area*.

    The front is either explicit (``suit`` + ``rank``: Familiar, Grim,
    Incantation) or drawn from ``G.P_CARDS`` with ``front_seed`` (Marble
    ``marb_fr``, Certificate ``cert_fr``). The seal is explicit or rolled with
    ``seal_seed`` (Certificate ``certsl``, ``card.lua:2468``). Lua notifies the
    jokers (``playing_card_joker_effects``) for every creator here, so
    ``notify`` defaults on.
    """

    area: str = "deck"
    suit: str | None = None
    rank: str | None = None
    front_seed: str | None = None
    enhancement: str = "c_base"
    seal: str | None = None
    seal_seed: str | None = None
    notify: bool = True

    def apply(self, gs: dict[str, Any]) -> None:
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit
        from jackdaw.engine.data.prototypes import PLAYING_CARDS

        rng = gs.get("rng")
        if self.front_seed is not None:
            front, _ = rng.element(PLAYING_CARDS, rng.seed(self.front_seed))
            suit, rank = Suit(front.suit), Rank(front.rank)
        else:
            suit, rank = Suit(self.suit), Rank(self.rank)
        seal = self.seal
        if self.seal_seed is not None:
            roll = rng.random(self.seal_seed)
            if roll > 0.75:
                seal = "Red"
            elif roll > 0.5:
                seal = "Blue"
            elif roll > 0.25:
                seal = "Gold"
            else:
                seal = "Purple"
        card = create_playing_card(suit, rank, enhancement=self.enhancement, game_state=gs)
        if seal:
            card.set_seal(gs, seal)
        lifecycle.add_playing_cards(gs, [card], self.area, notify=self.notify)
        if self.area == "hand":
            blind = gs.get("blind")
            if blind is not None:
                from jackdaw.engine import read

                blind.debuff_card(card, read.rules(gs), gs)
            _sort_hand(gs)


@dataclass(kw_only=True)
class CopyCard(Effect):
    """``copy_card(source, into?, ..., strip_edition)`` and placement.

    * ``into`` set: overwrite an existing card in place (Death).
    * otherwise a fresh card is placed in ``area`` (Ankh, Invisible Joker,
      Perkeo, DNA, Cryptid). With ``card=None`` the source is drawn from
      ``pick_area`` with ``pick_seed`` when the effect applies. ``edition`` is
      applied after the copy
      (Perkeo's Negative). ``reset_invis`` zeroes ``invis_rounds`` on the
      copy (Ankh / Invisible, ``card.lua:1447``, ``2386``).
    * Playing-card copies join the deck's card list; ``notify`` fires
      ``playing_card_added`` (Cryptid does, DNA does not).
    """

    card: Card | None = None
    pick_area: str | None = None
    pick_seed: str | None = None
    area: str = ""
    into: Card | None = None
    strip_edition: bool = False
    edition: dict[str, Any] | None = None
    reset_invis: bool = False
    notify: bool = False

    def apply(self, gs: dict[str, Any]) -> None:
        source = self.card
        if source is None:
            # Pick at event time (Perkeo, card.lua:2416). Nothing to copy
            # means no event in Lua, so no RNG is drawn.
            candidates = gs.get(self.pick_area or "", [])
            if not candidates:
                return
            rng = gs["rng"]
            source, _ = rng.element(list(candidates), rng.seed(self.pick_seed or ""))
        if self.into is not None:
            lifecycle.copy_card(gs, source, into=self.into)
            return
        new = lifecycle.copy_card(gs, source, strip_edition=self.strip_edition)
        if self.edition is not None:
            new.set_edition(gs, self.edition)
        if self.reset_invis and "invis_rounds" in new.ability:
            new.ability["invis_rounds"] = 0
        if source.base is not None and self.area in ("hand", "deck"):
            lifecycle.add_playing_cards(gs, [new], self.area, notify=self.notify)
            if self.area == "hand":
                _sort_hand(gs)
        else:
            lifecycle.emplace(gs, new, self.area)


@dataclass(kw_only=True)
class AddTag(Effect):
    """``add_tag(Tag(key))``: award a tag without firing it (Diet Cola, Anaglyph)."""

    key: str

    def apply(self, gs: dict[str, Any]) -> None:
        gs.setdefault("awarded_tags", []).append({"key": self.key, "result": None, "blind": None})


# ---------------------------------------------------------------------------
# Destruction
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class DestroyCard(Effect):
    """Remove a joker or consumable (``start_dissolve`` / ``Card:remove``).

    ``slice`` marks the target ``getting_sliced`` at emission, as Madness and
    Ceremonial Dagger do (``card.lua:2514``, ``2569``), so a joker already
    marked to die does not fire its own ``setting_blind`` later in the same
    loop. ``frees_slot`` is Dagger's ``joker_buffer - 1``: a later creator in
    the same pass sees the slot as free.
    """

    card: Card
    slice: bool = False
    frees_slot: bool = False

    def reserve(self, queue: EffectQueue) -> bool:
        if self.slice:
            self.card.getting_sliced = True
        if self.frees_slot:
            queue.reserved["jokers"] -= 1
        return True

    def apply(self, gs: dict[str, Any]) -> None:
        if self.card.removed:
            return
        lifecycle.remove(gs, self.card)


@dataclass(kw_only=True)
class DestroyPlayingCards(Effect):
    """Destroy a batch of playing cards, then notify jokers once (Hanged Man...)."""

    cards: list[Card]
    notify: bool = True

    def apply(self, gs: dict[str, Any]) -> None:
        alive = [card for card in self.cards if not card.removed]
        lifecycle.destroy_playing_cards(gs, alive, notify=self.notify)


# ---------------------------------------------------------------------------
# Card modification
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SetEnhancement(Effect):
    card: Card
    center: str

    def apply(self, gs: dict[str, Any]) -> None:
        self.card.set_ability(self.center, gs=gs)


@dataclass(kw_only=True)
class ChangeSuit(Effect):
    card: Card
    suit: str

    def apply(self, gs: dict[str, Any]) -> None:
        self.card.change_suit(self.suit, gs=gs)


@dataclass(kw_only=True)
class ChangeRank(Effect):
    card: Card
    delta: int

    def apply(self, gs: dict[str, Any]) -> None:
        self.card.change_rank(self.delta, gs=gs)


@dataclass(kw_only=True)
class SetSeal(Effect):
    card: Card
    seal: str | None

    def apply(self, gs: dict[str, Any]) -> None:
        self.card.set_seal(gs, self.seal)


@dataclass(kw_only=True)
class SetEdition(Effect):
    card: Card
    edition: dict[str, Any]

    def apply(self, gs: dict[str, Any]) -> None:
        self.card.set_edition(gs, self.edition)


# ---------------------------------------------------------------------------
# Run state
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class LevelUpHand(Effect):
    """``level_up_hand(hand, amount)`` outside scoring (Planets, Orbital Tag).

    The scoring pipeline's own level-ups (Space Joker, Burnt Joker, The Arm)
    are pipeline fields, not effects: they must land mid-pass, before base
    chips are read.
    """

    hand: str
    amount: int = 1

    def apply(self, gs: dict[str, Any]) -> None:
        hand_levels = gs.get("hand_levels")
        if hand_levels is not None:
            hand_levels.level_up(self.hand, self.amount)


@dataclass(kw_only=True)
class ChangeRoundResource(Effect):
    """``ease_hands_played`` / ``ease_discard`` (Burglar, ``card.lua:2522``).

    ``zero_discards`` reads ``discards_left`` at event time, as Burglar's
    ``ease_discard(-G.GAME.current_round.discards_left)`` does.
    """

    hands: int = 0
    discards: int = 0
    zero_discards: bool = False

    def apply(self, gs: dict[str, Any]) -> None:
        cr = gs.setdefault("current_round", {})
        if self.zero_discards:
            cr["discards_left"] = 0
        cr["discards_left"] = cr.get("discards_left", 0) + self.discards
        cr["hands_left"] = cr.get("hands_left", 0) + self.hands


@dataclass(kw_only=True)
class ChangeHandSize(Effect):
    """``G.hand:change_size(delta)`` (Turtle Bean decay, Ectoplasm, Ouija)."""

    delta: int

    def apply(self, gs: dict[str, Any]) -> None:
        gs["hand_size"] = gs.get("hand_size", 8) + self.delta


@dataclass(kw_only=True)
class DisableBlind(Effect):
    """``G.GAME.blind:disable()`` (Chicot, Luchador)."""

    def apply(self, gs: dict[str, Any]) -> None:
        blind = gs.get("blind")
        if blind is not None and not blind.disabled:
            blind.disable(gs)


@dataclass(kw_only=True)
class SetPoolFlag(Effect):
    """``G.GAME.pool_flags[flag] = true`` (Gros Michel extinction)."""

    flag: str

    def apply(self, gs: dict[str, Any]) -> None:
        gs.setdefault("pool_flags", {})[self.flag] = True


def _sort_hand(gs: dict[str, Any]) -> None:
    from jackdaw.engine.game import _sort_hand_desc

    _sort_hand_desc(gs.get("hand", []))


EFFECT_TYPES: tuple[type[Effect], ...] = (
    EaseDollars,
    SetDollars,
    CreateCard,
    CreatePlayingCard,
    CopyCard,
    AddTag,
    DestroyCard,
    DestroyPlayingCards,
    SetEnhancement,
    ChangeSuit,
    ChangeRank,
    SetSeal,
    SetEdition,
    LevelUpHand,
    ChangeRoundResource,
    ChangeHandSize,
    DisableBlind,
    SetPoolFlag,
)
"""Every concrete effect, for the structural tests."""
