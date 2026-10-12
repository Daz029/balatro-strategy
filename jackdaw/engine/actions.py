"""Player action types and game phase enum.

Every decision point in a Balatro run is represented as a frozen dataclass
action.  The simulator's step function accepts an :data:`Action` and
advances the game state accordingly.

Game phases
-----------
:class:`GamePhase` enumerates the phases where player input is needed.
The engine exposes the FULL vanilla action set (decision 2026-10-12); the
RL environments restrict it with their own masks. Legality — which action is
valid in which state — lives in exactly one place,
:mod:`jackdaw.engine.legality`, used by both :func:`step` and
:func:`get_legal_actions`. Phase summary (``SellCard``, ``UseConsumable``
and the joker swaps are valid in every stable phase, as in Lua):

+---------------------+------------------------------------------------------+
| Phase               | Phase-specific actions                               |
+=====================+======================================================+
| BLIND_SELECT        | SelectBlind, SkipBlind, RerollBoss                   |
+---------------------+------------------------------------------------------+
| SELECTING_HAND      | PlayHand, Discard, SortHand, SwapHandLeft/Right      |
+---------------------+------------------------------------------------------+
| SHOP                | BuyCard, BuyAndUse, RedeemVoucher, OpenBooster,      |
|                     | Reroll, NextRound                                    |
+---------------------+------------------------------------------------------+
| PACK_OPENING        | PickPackCard, SkipPack, SwapHandLeft/Right           |
+---------------------+------------------------------------------------------+
| ROUND_EVAL          | CashOut                                              |
+---------------------+------------------------------------------------------+
| GAME_OVER           | *(terminal — no valid actions)*                      |
+---------------------+------------------------------------------------------+
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from jackdaw.engine.card import Card


# ---------------------------------------------------------------------------
# Game phase
# ---------------------------------------------------------------------------


class GamePhase(StrEnum):
    """Phases where player input is needed.

    Maps to ``G.STATES`` in the Lua source, filtered to decision points.
    """

    BLIND_SELECT = "blind_select"
    SELECTING_HAND = "selecting_hand"
    ROUND_EVAL = "round_eval"
    SHOP = "shop"
    PACK_OPENING = "pack_opening"
    GAME_OVER = "game_over"


# ---------------------------------------------------------------------------
# Blind selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectBlind:
    """Accept the current blind and start the round."""

    allow_forced_boss: bool = False


@dataclass(frozen=True)
class SkipBlind:
    """Skip the current blind (Small or Big) and collect the tag reward."""


# ---------------------------------------------------------------------------
# Hand play / discard
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayHand:
    """Play selected cards from the hand.

    ``card_indices`` are 0-based indices into ``hand.cards``, 1–5 cards.
    Their ORDER is the left-to-right placement of the played cards, and the
    cards score in that order (NEW-P5-1-05): vanilla lets the player rearrange
    the hand for free before playing, so the caller chooses the order.
    """

    card_indices: tuple[int, ...]


@dataclass(frozen=True)
class Discard:
    """Discard selected cards from the hand.

    ``card_indices`` are 0-based indices into ``hand.cards``, 1–5 cards.
    """

    card_indices: tuple[int, ...]


# ---------------------------------------------------------------------------
# Shop actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuyCard:
    """Purchase a card from the shop.

    ``shop_index`` is the 0-based index into the shop card area
    (jokers, tarots, planets, or playing cards depending on slot).
    """

    shop_index: int


@dataclass(frozen=True)
class SellCard:
    """Sell a card from jokers or consumables for its sell value.

    ``area`` is ``'jokers'`` or ``'consumables'``.
    """

    area: str
    card_index: int


@dataclass(frozen=True)
class UseConsumable:
    """Use a consumable from the consumable area.

    ``target_indices`` are 0-based indices into ``hand.cards`` for
    targeted consumables (tarots that modify specific cards).
    ``None`` for untargeted consumables (planets, most spectrals).
    """

    card_index: int
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RedeemVoucher:
    """Purchase and activate a voucher from the shop."""

    card_index: int


@dataclass(frozen=True)
class OpenBooster:
    """Open a booster pack from the shop."""

    card_index: int


@dataclass(frozen=True)
class Reroll:
    """Reroll the shop (costs reroll_cost dollars)."""


@dataclass(frozen=True)
class NextRound:
    """Leave the shop and proceed to the next round."""


@dataclass(frozen=True)
class BuyAndUse:
    """Buy a shop consumable and use it immediately (``buy_and_use`` button).

    Needs no consumable slot (``button_callbacks.lua:2407`` skips
    ``check_for_buy_space``) but the card must be usable
    (``can_buy_and_use``, ``button_callbacks.lua:77-79``).
    """

    shop_index: int
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class RerollBoss:
    """Reroll the upcoming Boss blind ($10; Retcon or Director's Cut).

    ``button_callbacks.lua:2784-2848`` (``reroll_boss_button`` /
    ``reroll_boss``).
    """


# ---------------------------------------------------------------------------
# Pack opening
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PickPackCard:
    """Select a card from an opened booster pack.

    For Arcana/Spectral packs, ``target_indices`` specifies which dealt
    hand cards the consumable should target (e.g. for The Magician
    enhancing up to 2 cards).  ``None`` for non-targeted picks
    (Celestial, Standard, Buffoon packs).
    """

    card_index: int
    target_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class SkipPack:
    """Skip remaining picks in a booster pack."""


# ---------------------------------------------------------------------------
# Round evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CashOut:
    """Accept end-of-round earnings and proceed to the shop."""


# ---------------------------------------------------------------------------
# Utility actions (available in multiple phases)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SortHand:
    """Sort the hand by rank or suit.

    ``mode`` is ``'rank'`` or ``'suit'``.
    """

    mode: str


@dataclass(frozen=True)
class SwapHandLeft:
    """Swap a hand card one position to the left.

    Swaps ``hand[idx]`` with ``hand[idx - 1]``.  Free action — no cost,
    doesn't consume hands or discards.  Left-to-right order affects which
    card is "first scored" for jokers like Photograph and Hanging Chad.
    """

    idx: int


@dataclass(frozen=True)
class SwapHandRight:
    """Swap a hand card one position to the right.

    Swaps ``hand[idx]`` with ``hand[idx + 1]``.  Free action.
    """

    idx: int


@dataclass(frozen=True)
class SwapJokersLeft:
    """Swap a joker one position to the left.

    Swaps ``jokers[idx]`` with ``jokers[idx - 1]``.
    Left-to-right order affects Phase 9 scoring (additive before
    multiplicative), Blueprint (copies right neighbor), Brainstorm
    (copies leftmost), Ceremonial Dagger (destroys right neighbor).
    """

    idx: int


@dataclass(frozen=True)
class SwapJokersRight:
    """Swap a joker one position to the right.

    Swaps ``jokers[idx]`` with ``jokers[idx + 1]``.
    """

    idx: int


# ---------------------------------------------------------------------------
# Union type
# ---------------------------------------------------------------------------

Action = (
    PlayHand
    | Discard
    | SelectBlind
    | SkipBlind
    | BuyCard
    | SellCard
    | UseConsumable
    | RedeemVoucher
    | OpenBooster
    | PickPackCard
    | SkipPack
    | Reroll
    | NextRound
    | BuyAndUse
    | RerollBoss
    | CashOut
    | SortHand
    | SwapHandLeft
    | SwapHandRight
    | SwapJokersLeft
    | SwapJokersRight
)


# ---------------------------------------------------------------------------
# Legal action generation
# ---------------------------------------------------------------------------


def get_legal_actions(game_state: dict[str, Any]) -> list[Action]:
    """Return every legal action in the current state.

    Candidates are enumerated per phase and kept only if
    :func:`jackdaw.engine.legality.illegal_reason` accepts them, so this list
    and :func:`~jackdaw.engine.game.step` can never disagree. The engine lists
    the FULL vanilla action set; RL environments restrict it with their own
    masks.

    ``PlayHand`` / ``Discard`` are MARKERS: one action with empty
    ``card_indices`` means "some selection is legal" (enumerating C(n, 5)
    subsets is the caller's job; Cerulean Bell's forced card must be in it,
    see :func:`jackdaw.engine.legality.with_forced_card`). Consumable uses and
    pack picks that need targets are listed once per legal target set.
    """
    from jackdaw.engine import legality

    gs = game_state
    phase = legality._phase(gs)
    if phase is None or phase == GamePhase.GAME_OVER:
        return []

    candidates: list[Action] = []
    hand: list[Card] = gs.get("hand", [])
    jokers: list[Card] = gs.get("jokers", [])
    consumables: list[Card] = gs.get("consumables", [])

    if phase == GamePhase.BLIND_SELECT:
        candidates += [SelectBlind(), SkipBlind(), RerollBoss()]
    elif phase == GamePhase.SELECTING_HAND:
        if legality.can_play(gs):
            candidates.append(PlayHand(card_indices=()))
        if legality.can_discard(gs):
            candidates.append(Discard(card_indices=()))
        if len(hand) > 1:
            candidates += [SortHand(mode="rank"), SortHand(mode="suit")]
    elif phase == GamePhase.ROUND_EVAL:
        candidates.append(CashOut())
    elif phase == GamePhase.SHOP:
        for i, card in enumerate(gs.get("shop_cards", [])):
            candidates.append(BuyCard(shop_index=i))
            for targets in _target_sets(card, len(hand)):
                candidates.append(BuyAndUse(shop_index=i, target_indices=targets))
        candidates += [RedeemVoucher(card_index=i) for i in range(len(gs.get("shop_vouchers", [])))]
        candidates += [OpenBooster(card_index=i) for i in range(len(gs.get("shop_boosters", [])))]
        candidates += [Reroll(), NextRound()]
    elif phase == GamePhase.PACK_OPENING:
        for i, card in enumerate(gs.get("pack_cards", [])):
            for targets in _target_sets(card, len(hand)):
                candidates.append(PickPackCard(card_index=i, target_indices=targets))
        candidates.append(SkipPack())

    # Every stable phase: hand arrangement where a hand is on screen, joker
    # arrangement, selling owned cards, using owned consumables.
    candidates += [SwapHandLeft(idx=i) for i in range(1, len(hand))]
    candidates += [SwapHandRight(idx=i) for i in range(len(hand) - 1)]
    for i, card in enumerate(consumables):
        for targets in _target_sets(card, len(hand)):
            candidates.append(UseConsumable(card_index=i, target_indices=targets))
    candidates += [SellCard(area="jokers", card_index=i) for i in range(len(jokers))]
    candidates += [SellCard(area="consumables", card_index=i) for i in range(len(consumables))]
    candidates += [SwapJokersLeft(idx=i) for i in range(1, len(jokers))]
    candidates += [SwapJokersRight(idx=i) for i in range(len(jokers) - 1)]

    return [
        a
        for a in candidates
        if (isinstance(a, (PlayHand, Discard)) and not a.card_indices)
        or legality.is_legal(gs, a)
    ]


def _target_sets(card: Card, hand_size: int) -> list[tuple[int, ...] | None]:
    """Candidate target selections for using *card* (``None`` = untargeted).

    Vanilla targeting selects at most 3 hand cards (``max_highlighted``), so
    enumerating every 1..max subset of the hand is cheap; legality filters
    them. Untargeted cards and non-consumables get the single ``None``.
    """
    from itertools import combinations

    from jackdaw.engine.consumables import _resolve_consumable_config

    ability = getattr(card, "ability", None)
    if not isinstance(ability, dict) or ability.get("set") not in ("Tarot", "Planet", "Spectral"):
        return [None]
    cfg = _resolve_consumable_config(card)
    top = cfg.get("max_highlighted") or (1 if card.center_key == "c_aura" else 0)
    if not top or hand_size == 0:
        return [None]
    sets: list[tuple[int, ...] | None] = []
    for k in range(1, min(top, hand_size) + 1):
        sets.extend(combinations(range(hand_size), k))
    return sets
