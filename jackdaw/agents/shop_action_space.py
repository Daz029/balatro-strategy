"""Canonical fixed action space for the shop agent (appended at 436).

Extends the frozen hand-play block (``hand_action_space``, indices 0..435)
with the shop decision surface, honoring the APPEND-ONLY contract: indices
never reorder or renumber; new families only ever append at the end.

Index layout (canonical indices, all frozen forever):

    [436, 440)  BuyCard slot 0-3        (4 = Overstock Plus max card slots)
    [440, 444)  RedeemVoucher slot 0-3  (4 = base 1 + stacked Voucher-tag headroom)
    [444, 446)  OpenBooster slot 0-1    (engine hardcodes exactly 2 pack slots)
    [446, 454)  SellJoker slot 0-7      (8 = 5 base + Antimatter + negative-edition
                                         headroom; obs joker rows match)
    [454, 457)  SellConsumable slot 0-2 (3 = 2 base + Crystal Ball voucher)
    [457, 460)  UseConsumable slot 0-2
    [460]       Reroll
    [461]       NextRound (leave shop)
    [462, 467)  PickPackCard slot 0-4   (Mega packs show 5)
    [467]       SkipPack
    [468, 686)  SelectTarget combo i    (reuses hand_action_space.COMBOS verbatim --
                                         one combo table everywhere; vanilla
                                         consumables target at most 3 cards, so
                                         size-4/5 rows are permanently masked
                                         future-proofing)

Total: 250 shop actions; the canonical (s0) space is Discrete(686).

Targeting is a two-step scheme (grilled decision -- see CLAUDE.md shop-agent
design): a carrier action (PickPackCard / UseConsumable) that needs card
targets puts the env into a pending-target state where ONLY SelectTarget
combos are legal; the chosen combo completes the engine action. Targetable
cards always live in the hand-card obs rows (the dealt ``pack_hand`` during
shop pack-opening; the real hand in-blind after the merge).

s1 append (CLAUDE.md "s1" / "MAX_JOKER_ROWS" open items, DECIDED
2026-07-16; docs/post-regen-training-plan.md sections 6-7) -- these families
are BUILT here but only ever reachable when ``ShopRunConfig.s1_schema`` /
``s1_schema=True`` is threaded through the env; s0's Discrete(686) space,
every existing offset, and every existing mask/decode call keep behaving
exactly as before with the flag off:

    [686]       SkipBlind             (cold head row; blind-select stops
                                        being auto-resolved on non-boss
                                        blinds under s1 -- "select"/proceed
                                        reuses the NextRound row instead of
                                        a new index, see shop_gym.py)
    [687, 694)  SellJoker slot 8-14   (seven cold rows; obs joker block
                                        widens 8->15 to match MAX_JOKERS_V2,
                                        see :func:`sell_joker_action`)

Total with s1 appended: Discrete(694) (``NUM_TOTAL_ACTIONS_S1``).
``NUM_TOTAL_ACTIONS``/``NUM_SHOP_ACTIONS`` stay FROZEN at their s0 values --
they are not derived from iterating the (now longer) enum, precisely so
existing callers/tests that reference them never observe the s1 append.

No torch dependency; shared by the shop gym env and training scripts.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from jackdaw.agents.hand_action_space import COMBOS, NUM_COMBOS, NUM_HAND_ACTIONS
from jackdaw.env.action_space import ActionMask, ActionType
from jackdaw.env.hand_play_gym import MAX_JOKERS_V2

# Family sizes (headroom rationale in the module docstring / CLAUDE.md)
MAX_SHOP_CARD_SLOTS = 4
MAX_SHOP_VOUCHERS = 4
NUM_BOOSTER_SLOTS = 2
MAX_JOKER_ROWS = 8
MAX_CONSUMABLE_ROWS = 3
MAX_PACK_CARDS = 5

# s1: obs joker rows widen 8 -> 15 to match the hand-side v2 joker cap.
# Imported (not re-declared as a literal) so the two constants can never
# drift apart -- see the module docstring's s1 append section.
MAX_JOKER_ROWS_S1 = MAX_JOKERS_V2


class ShopActionFamily(IntEnum):
    """The shop action families, in frozen canonical order.

    ``SkipBlind``/``SellJokerExt`` are the s1 append (see module docstring)
    -- declared last so the existing families' relative order and the
    append-only loop below never move an existing offset.
    """

    BuyCard = 0
    RedeemVoucher = 1
    OpenBooster = 2
    SellJoker = 3
    SellConsumable = 4
    UseConsumable = 5
    Reroll = 6
    NextRound = 7
    PickPackCard = 8
    SkipPack = 9
    SelectTarget = 10
    SkipBlind = 11  # s1: canonical index 686
    SellJokerExt = 12  # s1: canonical indices [687, 694) -- joker rows 8-14


# (family, size) in canonical order -- APPEND-ONLY, never reorder.
FAMILY_SIZES: dict[ShopActionFamily, int] = {
    ShopActionFamily.BuyCard: MAX_SHOP_CARD_SLOTS,
    ShopActionFamily.RedeemVoucher: MAX_SHOP_VOUCHERS,
    ShopActionFamily.OpenBooster: NUM_BOOSTER_SLOTS,
    ShopActionFamily.SellJoker: MAX_JOKER_ROWS,
    ShopActionFamily.SellConsumable: MAX_CONSUMABLE_ROWS,
    ShopActionFamily.UseConsumable: MAX_CONSUMABLE_ROWS,
    ShopActionFamily.Reroll: 1,
    ShopActionFamily.NextRound: 1,
    ShopActionFamily.PickPackCard: MAX_PACK_CARDS,
    ShopActionFamily.SkipPack: 1,
    ShopActionFamily.SelectTarget: NUM_COMBOS,
    ShopActionFamily.SkipBlind: 1,
    ShopActionFamily.SellJokerExt: MAX_JOKER_ROWS_S1 - MAX_JOKER_ROWS,
}

SHOP_ACTION_BASE = NUM_HAND_ACTIONS  # 436

# This loop spans ALL families, s1 included (they're declared last in the
# enum), so FAMILY_OFFSETS carries valid offsets for SkipBlind/SellJokerExt
# too. The two size/total constants below are deliberately NOT read off
# the loop's final `_offset` (which now reflects the s1 span) -- they are
# pinned explicitly to the s0 span so nothing downstream that imports them
# ever sees a different number with the flag off.
FAMILY_OFFSETS: dict[ShopActionFamily, int] = {}
_offset = SHOP_ACTION_BASE
for _family in ShopActionFamily:
    FAMILY_OFFSETS[_family] = _offset
    _offset += FAMILY_SIZES[_family]

SELECT_TARGET_BASE = FAMILY_OFFSETS[ShopActionFamily.SelectTarget]
NUM_TOTAL_ACTIONS = SELECT_TARGET_BASE + NUM_COMBOS  # 686, FROZEN (s0 checkpoint width)
NUM_SHOP_ACTIONS = NUM_TOTAL_ACTIONS - SHOP_ACTION_BASE  # 250, FROZEN
NUM_TOTAL_ACTIONS_S1 = _offset  # 694 (s0 span + SkipBlind + SellJoker ext)

# Combo sizes, for SelectTarget legality (min/max highlighted per consumable)
_COMBO_SIZES: np.ndarray = np.array([len(c) for c in COMBOS], dtype=np.int64)
_COMBO_MAX_INDEX: np.ndarray = np.array([c[-1] for c in COMBOS], dtype=np.int64)

# Families whose slots map 1:1 onto an entity list's per-index legality mask
# from the engine's ActionMask.entity_masks.
_ENTITY_FAMILIES: dict[ShopActionFamily, ActionType] = {
    ShopActionFamily.BuyCard: ActionType.BuyCard,
    ShopActionFamily.RedeemVoucher: ActionType.RedeemVoucher,
    ShopActionFamily.OpenBooster: ActionType.OpenBooster,
    ShopActionFamily.SellJoker: ActionType.SellJoker,
    ShopActionFamily.SellConsumable: ActionType.SellConsumable,
    ShopActionFamily.UseConsumable: ActionType.UseConsumable,
    ShopActionFamily.PickPackCard: ActionType.PickPackCard,
}

# Families that are single actions gated only by the type mask.
_SINGLETON_FAMILIES: dict[ShopActionFamily, ActionType] = {
    ShopActionFamily.Reroll: ActionType.Reroll,
    ShopActionFamily.NextRound: ActionType.NextRound,
    ShopActionFamily.SkipPack: ActionType.SkipPack,
}


def shop_action(family: ShopActionFamily, slot: int = 0) -> int:
    """Encode (family, slot) into the canonical action index."""
    size = FAMILY_SIZES[family]
    if not 0 <= slot < size:
        raise ValueError(f"slot {slot} outside [0, {size}) for {family.name}")
    return FAMILY_OFFSETS[family] + slot


def decode_shop_action(action: int) -> tuple[ShopActionFamily, int]:
    """Decode a canonical shop action index into (family, slot).

    For ``SelectTarget``, the slot indexes ``COMBOS`` -- use
    :func:`target_combo_for_action` to get the card positions directly.

    The structural bound is ``NUM_TOTAL_ACTIONS_S1`` (694) unconditionally
    -- this is a pure index->(family, slot) decoder with no notion of
    "schema"; whether index 686+ is ever actually SAMPLED is a legality
    question the mask (and the env's Discrete(686) vs Discrete(694) action
    space) answers, not this function.
    """
    if not SHOP_ACTION_BASE <= action < NUM_TOTAL_ACTIONS_S1:
        raise ValueError(f"action {action} outside [{SHOP_ACTION_BASE}, {NUM_TOTAL_ACTIONS_S1})")
    for family in reversed(ShopActionFamily):
        offset = FAMILY_OFFSETS[family]
        if action >= offset:
            return family, action - offset
    raise AssertionError("unreachable")  # pragma: no cover


def target_combo_for_action(action: int) -> tuple[int, ...]:
    """Card positions for a SelectTarget action."""
    family, slot = decode_shop_action(action)
    if family is not ShopActionFamily.SelectTarget:
        raise ValueError(f"action {action} is {family.name}, not SelectTarget")
    return COMBOS[slot]


def sell_joker_action(slot: int) -> int:
    """Canonical action index that sells the joker at obs row ``slot``.

    ``slot`` spans ``[0, MAX_JOKER_ROWS_S1)`` (0-14) across TWO
    non-contiguous families: rows 0-7 are the frozen s0 ``SellJoker`` block
    [446, 454); rows 8-14 are the s1 ``SellJokerExt`` cold rows [687, 694).
    The single mapping lives here so the mask builder
    (:func:`shop_action_mask`) and the env's action decode
    (``shop_gym.py::_resolve_action``) never duplicate the arithmetic --
    see the module docstring's s1 append section and the CLAUDE.md
    ``MAX_JOKER_ROWS`` open item.
    """
    if 0 <= slot < MAX_JOKER_ROWS:
        return shop_action(ShopActionFamily.SellJoker, slot)
    if MAX_JOKER_ROWS <= slot < MAX_JOKER_ROWS_S1:
        return shop_action(ShopActionFamily.SellJokerExt, slot - MAX_JOKER_ROWS)
    raise ValueError(f"joker row {slot} outside [0, {MAX_JOKER_ROWS_S1})")


def joker_row_for_sell_action(action: int) -> int:
    """Inverse of :func:`sell_joker_action`: canonical index -> joker row.

    Raises if ``action`` decodes to neither ``SellJoker`` nor
    ``SellJokerExt``.
    """
    family, slot = decode_shop_action(action)
    if family is ShopActionFamily.SellJoker:
        return slot
    if family is ShopActionFamily.SellJokerExt:
        return slot + MAX_JOKER_ROWS
    raise ValueError(f"action {action} is {family.name}, not a SellJoker family")


_COMBO_TO_INDEX: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(COMBOS)}


def select_target_action(card_indices: tuple[int, ...] | list[int]) -> int:
    """Canonical SelectTarget index for a set of card positions."""
    combo = tuple(sorted(int(i) for i in card_indices))
    idx = _COMBO_TO_INDEX.get(combo)
    if idx is None:
        raise ValueError(f"card indices {card_indices!r} are not a valid 1-5 card combo")
    return SELECT_TARGET_BASE + idx


def select_target_mask(
    n_targetable: int,
    min_cards: int,
    max_cards: int,
    *,
    s1_schema: bool = False,
) -> np.ndarray:
    """Legality mask for a pending-target state: ONLY SelectTarget combos
    whose size is within ``[min_cards, max_cards]`` and whose every
    position is a dealt card.

    ``min_cards``/``max_cards`` come from the pending consumable's config
    (``jackdaw.env.action_space.get_consumable_target_info``).

    Shape is ``(NUM_TOTAL_ACTIONS,)`` by default (686) or
    ``(NUM_TOTAL_ACTIONS_S1,)`` (694) when ``s1_schema=True`` -- must match
    whichever action space the caller's env exposes, or MaskablePPO's
    per-step mask/action-space shapes disagree. The s1 tail (SkipBlind,
    SellJoker slots 8-14) is always False here: nothing but a SelectTarget
    combo is ever legal while a carrier is pending.
    """
    n = NUM_TOTAL_ACTIONS_S1 if s1_schema else NUM_TOTAL_ACTIONS
    mask = np.zeros(n, dtype=bool)
    legal = (
        (_COMBO_MAX_INDEX < n_targetable)
        & (_COMBO_SIZES >= max(1, min_cards))
        & (_COMBO_SIZES <= max_cards)
    )
    mask[SELECT_TARGET_BASE : SELECT_TARGET_BASE + NUM_COMBOS] = legal
    return mask


def shop_action_mask(action_mask: ActionMask, *, s1_schema: bool = False) -> np.ndarray:
    """Map the engine's per-phase ``ActionMask`` onto the canonical space.

    Returns shape ``(NUM_TOTAL_ACTIONS,)`` by default (686, byte-identical
    to the pre-s1 behavior) or ``(NUM_TOTAL_ACTIONS_S1,)`` (694) when
    ``s1_schema=True``. The hand block [0, 436) is always False here -- the
    shop agent never emits hand actions (they stay permanently masked in
    s0; the merge replaces this function, not the indices). Entity lists
    longer than their family block are clipped: rows beyond the block
    (e.g. a 9th negative-edition joker, when ``s1_schema=False``) are
    unreachable, a documented shared limitation with the obs entity rows.

    ``s1_schema=True`` additionally spreads the ``SellJoker`` entity mask's
    rows 8-14 onto the ``SellJokerExt`` block via :func:`sell_joker_action`
    (the same k->index mapping the env's action decode uses -- see its
    docstring). BLIND_SELECT-phase legality (SkipBlind + the reused
    NextRound "select/proceed" row) is NOT built here: it has no analog in
    the generic per-phase ``ActionMask`` (BLIND_SELECT never sets
    ``type_mask[ActionType.NextRound]``), so it is synthesized directly in
    ``shop_gym.py::action_masks`` instead, the same way PACK_OPENING's
    PickPackCard rows already get a bespoke rebuild there.
    """
    n = NUM_TOTAL_ACTIONS_S1 if s1_schema else NUM_TOTAL_ACTIONS
    mask = np.zeros(n, dtype=bool)

    for family, action_type in _ENTITY_FAMILIES.items():
        if not action_mask.type_mask[action_type]:
            continue
        entity = action_mask.entity_masks.get(int(action_type))
        if entity is None:
            continue
        size = FAMILY_SIZES[family]
        offset = FAMILY_OFFSETS[family]
        n_rows = min(len(entity), size)
        mask[offset : offset + n_rows] = entity[:n_rows]

    if s1_schema and action_mask.type_mask[ActionType.SellJoker]:
        entity = action_mask.entity_masks.get(int(ActionType.SellJoker))
        if entity is not None:
            for row in range(MAX_JOKER_ROWS, min(len(entity), MAX_JOKER_ROWS_S1)):
                mask[sell_joker_action(row)] = entity[row]

    for family, action_type in _SINGLETON_FAMILIES.items():
        if action_mask.type_mask[action_type]:
            mask[FAMILY_OFFSETS[family]] = True

    return mask
