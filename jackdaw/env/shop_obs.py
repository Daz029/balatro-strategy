"""Observation encoding for the shop agent.

Extends the hand-agent obs pattern (per-entity blocks + masks + reused
``observation.py`` encoders) with the shop decision surface. Every entity
row that has an identity carries a parallel ``*_ids`` integer array
(``observation.center_key_id`` — ONE vocabulary for all item types); the
policy net turns ids into learned embeddings + static descriptors
(``joker_descriptors.DESCRIPTOR_MATRIX``). Keeping ids out of the float
rows means the obs stays compact and the embedding/descriptor lookup lives
net-side where it's differentiable/bufferable.

Row-count invariants (mirror ``shop_action_space`` family sizes — the mask
for "sell joker 5" must have an obs row 5 to look at):

    hand/pack_hand 8x15, jokers 8x15, consumables 3x8, shop items 4x16,
    pack contents 5x16, vouchers 4x4, boosters 2x8.

The dealt ``pack_hand`` occupies the hand rows during PACK_OPENING (the
engine already merges it into ``gs["hand"]``): "targetable cards live in
the hand rows" is the invariant that makes in-blind consumable targeting
at the merge pure reuse. Entity lists longer than their block are clipped
(documented shared limitation with the action space).

The pending-target state (two-step targeting, grilled decision) must be
OBSERVABLE, not mask-only: ``shop_context`` carries the pending flag and
min/max target counts, and the carrier row gets a ``selected`` bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gymnasium import spaces

from jackdaw.engine.data.prototypes import BOOSTERS, CENTER_POOLS, JOKERS
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    NUM_CENTER_KEYS,
    center_key_id,
    encode_consumable,
    encode_global_context,
    encode_jokers_batch,
    encode_playing_cards_batch,
)

MAX_HAND_ROWS = 8
MAX_JOKER_ROWS = 8  # matches SellJoker family size
MAX_CONSUMABLE_ROWS = 3  # matches Sell/UseConsumable family size
MAX_SHOP_ITEM_ROWS = 4  # matches BuyCard family size
MAX_VOUCHER_ROWS = 4  # matches RedeemVoucher family size
NUM_BOOSTER_ROWS = 2  # engine-fixed pack slots
MAX_PACK_ROWS = 5  # matches PickPackCard family size

D_CONSUMABLE_ROW = D_CONSUMABLE + 1  # + selected bit (pending carrier)
D_ITEM = 16
D_VOUCHER_ROW = 4
D_BOOSTER_ROW = 8
D_SHOP_CONTEXT = 12

_EDITION_ORD = {"foil": 1, "holo": 2, "polychrome": 3, "negative": 4}
_ENHANCEMENT_ORD = {k: i + 1 for i, k in enumerate(CENTER_POOLS.get("Enhanced", []))}
_SEAL_ORD = {"Red": 1, "Blue": 2, "Gold": 3, "Purple": 4}
_SUIT_ORD = {"Hearts": 0, "Diamonds": 1, "Clubs": 2, "Spades": 3}
_PACK_KIND_ORD = {"Arcana": 0, "Celestial": 1, "Spectral": 2, "Standard": 3, "Buffoon": 4}
# Union-row type slots (D_ITEM features 0-4)
_SET_SLOT = {"Joker": 0, "Tarot": 1, "Planet": 2, "Spectral": 3}


@dataclass
class PendingTarget:
    """Two-step targeting: the carrier action awaiting a SelectTarget combo.

    kind:
        ``"pack"`` (PickPackCard) or ``"consumable"`` (UseConsumable).
    slot:
        Row index of the carrier in its entity block.
    min_cards / max_cards:
        Legal target-count bounds from the consumable's config
        (``get_consumable_target_info``).
    """

    kind: str
    slot: int
    min_cards: int
    max_cards: int


def _edition_ord(card: Any) -> int:
    ed = getattr(card, "edition", None)
    if isinstance(ed, dict):
        return _EDITION_ORD.get(ed.get("type", ""), 0)
    return 0


def _card_set(card: Any) -> str:
    ability = getattr(card, "ability", None)
    if isinstance(ability, dict):
        return ability.get("set", "")
    return ""


def _encode_item_row(card: Any, selected: bool) -> np.ndarray:
    """Union encoding for a mixed-type item (shop slot / pack content).

    Layout (16): [0:5] type one-hot (Joker/Tarot/Planet/Spectral/
    PlayingCard), 5 cost, 6 edition, 7 negative, 8 joker rarity,
    9 rank, 10 suit, 11 enhancement, 12 seal, 13 is_face, 14 selected
    (pending carrier), 15 reserved.
    """
    v = np.zeros(D_ITEM, dtype=np.float32)
    card_set = _card_set(card)
    slot = _SET_SLOT.get(card_set)
    if slot is not None:
        v[slot] = 1.0
    else:
        v[4] = 1.0  # playing card (Default/Enhanced)

    v[5] = getattr(card, "cost", 0) / 10.0
    ed = _edition_ord(card)
    v[6] = ed / 4.0
    v[7] = float(ed == _EDITION_ORD["negative"])

    if card_set == "Joker":
        proto = JOKERS.get(card.center_key)
        if proto is not None:
            v[8] = proto.rarity / 4.0

    base = getattr(card, "base", None)
    if slot is None and base is not None:  # playing card
        v[9] = getattr(base, "id", 0) / 14.0
        v[10] = _SUIT_ORD.get(getattr(base, "suit", None) and base.suit.value, 0) / 3.0
        v[11] = _ENHANCEMENT_ORD.get(getattr(card, "center_key", ""), 0) / 8.0
        v[12] = _SEAL_ORD.get(getattr(card, "seal", None), 0) / 4.0
        v[13] = float(getattr(base, "id", 0) in (11, 12, 13))

    v[14] = float(selected)
    return v


def _pad_rows(rows: list[np.ndarray], max_n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((max_n, dim), dtype=np.float32)
    mask = np.zeros(max_n, dtype=np.float32)
    n = min(len(rows), max_n)
    for i in range(n):
        padded[i] = rows[i]
        mask[i] = 1.0
    return padded, mask


def _ids(cards: list, max_n: int) -> np.ndarray:
    out = np.zeros(max_n, dtype=np.int64)
    for i, c in enumerate(cards[:max_n]):
        out[i] = center_key_id(getattr(c, "center_key", ""))
    return out


def build_shop_observation(
    gs: dict[str, Any],
    pending: PendingTarget | None = None,
) -> dict[str, np.ndarray]:
    """Encode the full shop-agent observation from a live engine state."""
    cr = gs.get("current_round", {})
    rr = gs.get("round_resets", {})
    phase = gs.get("phase")
    phase_str = getattr(phase, "value", phase)

    # -- reused hand-agent blocks ------------------------------------------
    hand: list = gs.get("hand", [])[:MAX_HAND_ROWS]
    hand_arr = encode_playing_cards_batch(hand, gs)
    hand_cards, hand_mask = _pad_rows(list(hand_arr), MAX_HAND_ROWS, D_PLAYING_CARD)

    jokers: list = gs.get("jokers", [])[:MAX_JOKER_ROWS]
    joker_arr = encode_jokers_batch(jokers, gs)
    joker_rows, joker_mask = _pad_rows(list(joker_arr), MAX_JOKER_ROWS, D_JOKER)

    consumables: list = gs.get("consumables", [])[:MAX_CONSUMABLE_ROWS]
    cons_rows = []
    for i, c in enumerate(consumables):
        row = np.zeros(D_CONSUMABLE_ROW, dtype=np.float32)
        row[:D_CONSUMABLE] = encode_consumable(c, gs)
        row[D_CONSUMABLE] = float(
            pending is not None and pending.kind == "consumable" and pending.slot == i
        )
        cons_rows.append(row)
    cons_block, cons_mask = _pad_rows(cons_rows, MAX_CONSUMABLE_ROWS, D_CONSUMABLE_ROW)

    # -- shop inventory -------------------------------------------------------
    shop_cards: list = gs.get("shop_cards", [])[:MAX_SHOP_ITEM_ROWS]
    shop_rows = [_encode_item_row(c, selected=False) for c in shop_cards]
    shop_block, shop_mask = _pad_rows(shop_rows, MAX_SHOP_ITEM_ROWS, D_ITEM)

    pack_cards: list = gs.get("pack_cards", [])[:MAX_PACK_ROWS]
    pack_rows = [
        _encode_item_row(
            c,
            selected=(pending is not None and pending.kind == "pack" and pending.slot == i),
        )
        for i, c in enumerate(pack_cards)
    ]
    pack_block, pack_mask = _pad_rows(pack_rows, MAX_PACK_ROWS, D_ITEM)

    dollars = gs.get("dollars", 0)
    vouchers: list = gs.get("shop_vouchers", [])[:MAX_VOUCHER_ROWS]
    voucher_rows = []
    for v in vouchers:
        row = np.zeros(D_VOUCHER_ROW, dtype=np.float32)
        row[0] = getattr(v, "cost", 0) / 10.0
        row[1] = float(getattr(v, "cost", 0) <= dollars)
        voucher_rows.append(row)
    voucher_block, voucher_mask = _pad_rows(voucher_rows, MAX_VOUCHER_ROWS, D_VOUCHER_ROW)

    boosters: list = gs.get("shop_boosters", [])[:NUM_BOOSTER_ROWS]
    booster_rows = []
    for b in boosters:
        row = np.zeros(D_BOOSTER_ROW, dtype=np.float32)
        row[0] = getattr(b, "cost", 0) / 10.0
        proto = BOOSTERS.get(getattr(b, "center_key", ""))
        if proto is not None:
            cfg = proto.config or {}
            row[1] = cfg.get("choose", 1) / 5.0
            row[2] = cfg.get("extra", 2) / 5.0
            kind_slot = _PACK_KIND_ORD.get(proto.kind)
            if kind_slot is not None:
                row[3 + kind_slot] = 1.0
        booster_rows.append(row)
    booster_block, booster_mask = _pad_rows(booster_rows, NUM_BOOSTER_ROWS, D_BOOSTER_ROW)

    # -- shop context ----------------------------------------------------------
    ctx = np.zeros(D_SHOP_CONTEXT, dtype=np.float32)
    ctx[0] = dollars / 50.0
    ctx[1] = cr.get("reroll_cost", 5) / 10.0
    ctx[2] = float(cr.get("free_rerolls", 0) > 0)
    ctx[3] = rr.get("ante", 1) / 8.0
    ctx[4] = gs.get("round", 0) / 24.0
    ctx[6] = gs.get("pack_choices_remaining", 0) / 5.0
    ctx[7] = float(pending is not None)
    if pending is not None:
        ctx[8] = pending.min_cards / 5.0
        ctx[9] = pending.max_cards / 5.0
    ctx[10] = gs.get("win_ante", 8) / 8.0
    ctx[11] = float(phase_str == "pack_opening")

    return {
        "global_context": encode_global_context(gs),
        "shop_context": ctx,
        "hand_cards": hand_cards,
        "hand_mask": hand_mask,
        "jokers": joker_rows,
        "joker_mask": joker_mask,
        "joker_ids": _ids(jokers, MAX_JOKER_ROWS),
        "consumables": cons_block,
        "consumable_mask": cons_mask,
        "consumable_ids": _ids(consumables, MAX_CONSUMABLE_ROWS),
        "shop_items": shop_block,
        "shop_item_mask": shop_mask,
        "shop_item_ids": _ids(shop_cards, MAX_SHOP_ITEM_ROWS),
        "pack_items": pack_block,
        "pack_item_mask": pack_mask,
        "pack_item_ids": _ids(pack_cards, MAX_PACK_ROWS),
        "vouchers": voucher_block,
        "voucher_mask": voucher_mask,
        "voucher_ids": _ids(vouchers, MAX_VOUCHER_ROWS),
        "boosters": booster_block,
        "booster_mask": booster_mask,
        "booster_ids": _ids(boosters, NUM_BOOSTER_ROWS),
    }


def observation_space() -> spaces.Dict:
    def box(*shape: int) -> spaces.Box:
        return spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    def mask(n: int) -> spaces.Box:
        return spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)

    def ids(n: int) -> spaces.Box:
        return spaces.Box(0, NUM_CENTER_KEYS, shape=(n,), dtype=np.int64)

    return spaces.Dict(
        {
            "global_context": box(D_GLOBAL),
            "shop_context": box(D_SHOP_CONTEXT),
            "hand_cards": box(MAX_HAND_ROWS, D_PLAYING_CARD),
            "hand_mask": mask(MAX_HAND_ROWS),
            "jokers": box(MAX_JOKER_ROWS, D_JOKER),
            "joker_mask": mask(MAX_JOKER_ROWS),
            "joker_ids": ids(MAX_JOKER_ROWS),
            "consumables": box(MAX_CONSUMABLE_ROWS, D_CONSUMABLE_ROW),
            "consumable_mask": mask(MAX_CONSUMABLE_ROWS),
            "consumable_ids": ids(MAX_CONSUMABLE_ROWS),
            "shop_items": box(MAX_SHOP_ITEM_ROWS, D_ITEM),
            "shop_item_mask": mask(MAX_SHOP_ITEM_ROWS),
            "shop_item_ids": ids(MAX_SHOP_ITEM_ROWS),
            "pack_items": box(MAX_PACK_ROWS, D_ITEM),
            "pack_item_mask": mask(MAX_PACK_ROWS),
            "pack_item_ids": ids(MAX_PACK_ROWS),
            "vouchers": box(MAX_VOUCHER_ROWS, D_VOUCHER_ROW),
            "voucher_mask": mask(MAX_VOUCHER_ROWS),
            "voucher_ids": ids(MAX_VOUCHER_ROWS),
            "boosters": box(NUM_BOOSTER_ROWS, D_BOOSTER_ROW),
            "booster_mask": mask(NUM_BOOSTER_ROWS),
            "booster_ids": ids(NUM_BOOSTER_ROWS),
        }
    )
