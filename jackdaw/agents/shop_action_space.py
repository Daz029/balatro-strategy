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

Total: 250 shop actions; the canonical space is now Discrete(686).

Targeting is a two-step scheme (grilled decision -- see CLAUDE.md shop-agent
design): a carrier action (PickPackCard / UseConsumable) that needs card
targets puts the env into a pending-target state where ONLY SelectTarget
combos are legal; the chosen combo completes the engine action. Targetable
cards always live in the hand-card obs rows (the dealt ``pack_hand`` during
shop pack-opening; the real hand in-blind after the merge).

No torch dependency; shared by the shop gym env and training scripts.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from jackdaw.agents.hand_action_space import COMBOS, NUM_COMBOS, NUM_HAND_ACTIONS
from jackdaw.env.action_space import ActionMask, ActionType

# Family sizes (headroom rationale in the module docstring / CLAUDE.md)
MAX_SHOP_CARD_SLOTS = 4
MAX_SHOP_VOUCHERS = 4
NUM_BOOSTER_SLOTS = 2
MAX_JOKER_ROWS = 8
MAX_CONSUMABLE_ROWS = 3
MAX_PACK_CARDS = 5


class ShopActionFamily(IntEnum):
    """The shop action families, in frozen canonical order."""

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
}

SHOP_ACTION_BASE = NUM_HAND_ACTIONS  # 436

FAMILY_OFFSETS: dict[ShopActionFamily, int] = {}
_offset = SHOP_ACTION_BASE
for _family in ShopActionFamily:
    FAMILY_OFFSETS[_family] = _offset
    _offset += FAMILY_SIZES[_family]

NUM_SHOP_ACTIONS = _offset - SHOP_ACTION_BASE  # 250
NUM_TOTAL_ACTIONS = _offset  # 686

SELECT_TARGET_BASE = FAMILY_OFFSETS[ShopActionFamily.SelectTarget]

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
    """
    if not SHOP_ACTION_BASE <= action < NUM_TOTAL_ACTIONS:
        raise ValueError(f"action {action} outside [{SHOP_ACTION_BASE}, {NUM_TOTAL_ACTIONS})")
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


_COMBO_TO_INDEX: dict[tuple[int, ...], int] = {c: i for i, c in enumerate(COMBOS)}


def select_target_action(card_indices: tuple[int, ...] | list[int]) -> int:
    """Canonical SelectTarget index for a set of card positions."""
    combo = tuple(sorted(int(i) for i in card_indices))
    idx = _COMBO_TO_INDEX.get(combo)
    if idx is None:
        raise ValueError(f"card indices {card_indices!r} are not a valid 1-5 card combo")
    return SELECT_TARGET_BASE + idx


def select_target_mask(n_targetable: int, min_cards: int, max_cards: int) -> np.ndarray:
    """Legality mask of shape ``(NUM_TOTAL_ACTIONS,)`` for a pending-target
    state: ONLY SelectTarget combos whose size is within
    ``[min_cards, max_cards]`` and whose every position is a dealt card.

    ``min_cards``/``max_cards`` come from the pending consumable's config
    (``jackdaw.env.action_space.get_consumable_target_info``).
    """
    mask = np.zeros(NUM_TOTAL_ACTIONS, dtype=bool)
    legal = (
        (_COMBO_MAX_INDEX < n_targetable)
        & (_COMBO_SIZES >= max(1, min_cards))
        & (_COMBO_SIZES <= max_cards)
    )
    mask[SELECT_TARGET_BASE : SELECT_TARGET_BASE + NUM_COMBOS] = legal
    return mask


def shop_action_mask(action_mask: ActionMask) -> np.ndarray:
    """Map the engine's per-phase ``ActionMask`` onto the canonical space.

    Returns shape ``(NUM_TOTAL_ACTIONS,)``. The hand block [0, 436) is
    always False here -- the shop agent never emits hand actions (they stay
    permanently masked in s0; the merge replaces this function, not the
    indices). Entity lists longer than their family block are clipped: rows
    beyond the block (e.g. a 9th negative-edition joker) are unreachable, a
    documented shared limitation with the obs entity rows.
    """
    mask = np.zeros(NUM_TOTAL_ACTIONS, dtype=bool)

    for family, action_type in _ENTITY_FAMILIES.items():
        if not action_mask.type_mask[action_type]:
            continue
        entity = action_mask.entity_masks.get(int(action_type))
        if entity is None:
            continue
        size = FAMILY_SIZES[family]
        offset = FAMILY_OFFSETS[family]
        n = min(len(entity), size)
        mask[offset : offset + n] = entity[:n]

    for family, action_type in _SINGLETON_FAMILIES.items():
        if action_mask.type_mask[action_type]:
            mask[FAMILY_OFFSETS[family]] = True

    return mask
