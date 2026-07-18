"""Tests for the canonical shop action space (appended at 436).

The 250-index shop block extends the frozen hand block under the same
append-only contract: index *i* means the same (family, slot) forever, and
checkpoints' policy-head rows depend on it. These tests pin the exact
offsets, round-trips, and mask builders against brute-force references.
"""

from __future__ import annotations

import numpy as np
import pytest

from jackdaw.agents.hand_action_space import COMBOS, NUM_COMBOS, NUM_HAND_ACTIONS
from jackdaw.agents.shop_action_space import (
    FAMILY_OFFSETS,
    FAMILY_SIZES,
    MAX_JOKER_ROWS,
    MAX_JOKER_ROWS_S1,
    NUM_SHOP_ACTIONS,
    NUM_TOTAL_ACTIONS,
    NUM_TOTAL_ACTIONS_S1,
    SELECT_TARGET_BASE,
    SHOP_ACTION_BASE,
    ShopActionFamily,
    decode_shop_action,
    joker_row_for_sell_action,
    select_target_action,
    select_target_mask,
    sell_joker_action,
    shop_action,
    shop_action_mask,
    target_combo_for_action,
)
from jackdaw.env.action_space import ActionMask, ActionType


class TestFrozenLayout:
    def test_totals(self):
        assert SHOP_ACTION_BASE == NUM_HAND_ACTIONS == 436
        assert NUM_SHOP_ACTIONS == 250
        assert NUM_TOTAL_ACTIONS == 686  # s0, FROZEN forever (checkpoint width)

    def test_s1_total_appends_skipblind_and_selljoker_ext(self):
        # docs/post-regen-training-plan.md section 7 / CLAUDE.md
        # MAX_JOKER_ROWS: one SkipBlind row + seven SellJoker ext rows,
        # strictly appended after the s0 span -- NUM_TOTAL_ACTIONS (the s0
        # constant) must NOT move when this grows.
        assert NUM_TOTAL_ACTIONS_S1 == NUM_TOTAL_ACTIONS + 1 + 7 == 694
        assert MAX_JOKER_ROWS_S1 == 15

    def test_exact_offsets_are_frozen(self):
        # Pin every family boundary; renumbering silently corrupts every
        # existing checkpoint's action-head rows.
        expected = {
            ShopActionFamily.BuyCard: 436,
            ShopActionFamily.RedeemVoucher: 440,
            ShopActionFamily.OpenBooster: 444,
            ShopActionFamily.SellJoker: 446,
            ShopActionFamily.SellConsumable: 454,
            ShopActionFamily.UseConsumable: 457,
            ShopActionFamily.Reroll: 460,
            ShopActionFamily.NextRound: 461,
            ShopActionFamily.PickPackCard: 462,
            ShopActionFamily.SkipPack: 467,
            ShopActionFamily.SelectTarget: 468,
            # s1 append (DECIDED 2026-07-16) -- new offsets only, never a
            # renumbering of the eleven above.
            ShopActionFamily.SkipBlind: 686,
            ShopActionFamily.SellJokerExt: 687,
        }
        assert FAMILY_OFFSETS == expected
        assert SELECT_TARGET_BASE + NUM_COMBOS == NUM_TOTAL_ACTIONS

    def test_families_tile_block_exactly(self):
        covered = np.zeros(NUM_TOTAL_ACTIONS - SHOP_ACTION_BASE, dtype=int)
        for family in ShopActionFamily:
            off = FAMILY_OFFSETS[family] - SHOP_ACTION_BASE
            covered[off : off + FAMILY_SIZES[family]] += 1
        assert (covered == 1).all()  # no gaps, no overlaps

    def test_no_overlap_with_hand_block(self):
        for family in ShopActionFamily:
            assert FAMILY_OFFSETS[family] >= NUM_HAND_ACTIONS


class TestRoundTrip:
    def test_every_shop_action_round_trips(self):
        for action in range(SHOP_ACTION_BASE, NUM_TOTAL_ACTIONS):
            family, slot = decode_shop_action(action)
            assert shop_action(family, slot) == action

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            decode_shop_action(NUM_HAND_ACTIONS - 1)  # hand block
        with pytest.raises(ValueError):
            # NUM_TOTAL_ACTIONS (686) is now the valid SkipBlind index (s1
            # append); the true structural ceiling is NUM_TOTAL_ACTIONS_S1.
            decode_shop_action(NUM_TOTAL_ACTIONS_S1)
        with pytest.raises(ValueError):
            shop_action(ShopActionFamily.BuyCard, 4)

    def test_s1_indices_decode(self):
        # 686/687-693 are only reachable through the flag-gated s1 path,
        # but decode_shop_action is schema-agnostic -- legality lives in
        # the mask/action-space size, not here.
        assert decode_shop_action(NUM_TOTAL_ACTIONS) == (ShopActionFamily.SkipBlind, 0)
        assert decode_shop_action(NUM_TOTAL_ACTIONS_S1 - 1) == (ShopActionFamily.SellJokerExt, 6)


class TestSellJokerMapping:
    def test_round_trip_all_rows(self):
        for row in range(MAX_JOKER_ROWS_S1):
            action = sell_joker_action(row)
            assert joker_row_for_sell_action(action) == row

    def test_low_rows_hit_the_frozen_s0_block(self):
        for row in range(MAX_JOKER_ROWS):
            assert sell_joker_action(row) == shop_action(ShopActionFamily.SellJoker, row)

    def test_high_rows_hit_the_s1_ext_block(self):
        for row in range(MAX_JOKER_ROWS, MAX_JOKER_ROWS_S1):
            assert sell_joker_action(row) == shop_action(
                ShopActionFamily.SellJokerExt, row - MAX_JOKER_ROWS
            )

    def test_out_of_range_slot_rejected(self):
        with pytest.raises(ValueError):
            sell_joker_action(-1)
        with pytest.raises(ValueError):
            sell_joker_action(MAX_JOKER_ROWS_S1)

    def test_inverse_rejects_non_selljoker_family(self):
        with pytest.raises(ValueError):
            joker_row_for_sell_action(shop_action(ShopActionFamily.Reroll))


class TestSelectTargetRoundTrip:
    def test_select_target_combo_round_trip(self):
        for combo in COMBOS:
            action = select_target_action(combo)
            assert target_combo_for_action(action) == combo

    def test_select_target_unsorted_input(self):
        assert select_target_action([3, 0, 5]) == select_target_action((0, 3, 5))

    def test_target_combo_rejects_other_families(self):
        with pytest.raises(ValueError):
            target_combo_for_action(shop_action(ShopActionFamily.Reroll))


class TestSelectTargetMask:
    def test_matches_brute_force(self):
        for n_targetable, min_c, max_c in [(8, 1, 3), (5, 1, 1), (7, 2, 2), (8, 1, 5), (0, 1, 3)]:
            mask = select_target_mask(n_targetable, min_c, max_c)
            assert mask[:SELECT_TARGET_BASE].sum() == 0
            for i, combo in enumerate(COMBOS):
                expected = combo[-1] < n_targetable and max(1, min_c) <= len(combo) <= max_c
                assert mask[SELECT_TARGET_BASE + i] == expected, (
                    f"combo {combo} n={n_targetable} min={min_c} max={max_c}"
                )

    def test_only_select_target_block_set(self):
        mask = select_target_mask(8, 1, 3)
        assert not mask[:SELECT_TARGET_BASE].any()
        assert mask.shape == (NUM_TOTAL_ACTIONS,)

    def test_default_is_s0_sized_and_s1_tail_is_false(self):
        # Default (s1_schema=False) must be byte-identical to pre-s1: same
        # shape, same content.
        mask = select_target_mask(8, 1, 3)
        assert mask.shape == (NUM_TOTAL_ACTIONS,)

        s1_mask = select_target_mask(8, 1, 3, s1_schema=True)
        assert s1_mask.shape == (NUM_TOTAL_ACTIONS_S1,)
        np.testing.assert_array_equal(s1_mask[:NUM_TOTAL_ACTIONS], mask)
        # Nothing but a SelectTarget combo is ever legal while pending --
        # the s1 tail (SkipBlind, SellJoker ext) stays False.
        assert not s1_mask[NUM_TOTAL_ACTIONS:].any()


def _empty_action_mask() -> ActionMask:
    return ActionMask(
        type_mask=np.zeros(21, dtype=bool),
        card_mask=np.zeros(0, dtype=bool),
        entity_masks={},
        max_card_select=5,
        min_card_select=1,
    )


class TestShopActionMask:
    def test_empty_mask_all_false(self):
        mask = shop_action_mask(_empty_action_mask())
        assert mask.shape == (NUM_TOTAL_ACTIONS,)
        assert not mask.any()

    def test_entity_and_singleton_mapping(self):
        am = _empty_action_mask()
        am.type_mask[ActionType.BuyCard] = True
        am.entity_masks[int(ActionType.BuyCard)] = np.array([True, False, True])
        am.type_mask[ActionType.NextRound] = True
        am.type_mask[ActionType.Reroll] = True

        mask = shop_action_mask(am)
        buy = FAMILY_OFFSETS[ShopActionFamily.BuyCard]
        assert mask[buy] and not mask[buy + 1] and mask[buy + 2] and not mask[buy + 3]
        assert mask[FAMILY_OFFSETS[ShopActionFamily.NextRound]]
        assert mask[FAMILY_OFFSETS[ShopActionFamily.Reroll]]
        # nothing else set
        assert mask.sum() == 4
        # hand block always False for the shop agent
        assert not mask[:NUM_HAND_ACTIONS].any()

    def test_oversized_entity_list_clipped(self):
        # 9 sellable jokers (negative editions) but only 8 canonical rows:
        # row 9 is unreachable, by documented design.
        am = _empty_action_mask()
        am.type_mask[ActionType.SellJoker] = True
        am.entity_masks[int(ActionType.SellJoker)] = np.ones(9, dtype=bool)

        mask = shop_action_mask(am)
        off = FAMILY_OFFSETS[ShopActionFamily.SellJoker]
        assert mask[off : off + 8].all()
        assert mask.sum() == 8

    def test_type_gate_required(self):
        # entity mask present but type flag off -> nothing legal
        am = _empty_action_mask()
        am.entity_masks[int(ActionType.BuyCard)] = np.ones(2, dtype=bool)
        assert not shop_action_mask(am).any()

    def test_s1_default_off_is_byte_identical(self):
        am = _empty_action_mask()
        am.type_mask[ActionType.BuyCard] = True
        am.entity_masks[int(ActionType.BuyCard)] = np.array([True, False, True])
        am.type_mask[ActionType.SellJoker] = True
        am.entity_masks[int(ActionType.SellJoker)] = np.ones(9, dtype=bool)

        default_mask = shop_action_mask(am)
        explicit_off_mask = shop_action_mask(am, s1_schema=False)
        np.testing.assert_array_equal(default_mask, explicit_off_mask)
        assert default_mask.shape == (NUM_TOTAL_ACTIONS,)

    def test_s1_spreads_selljoker_rows_8_to_14(self):
        # 15 sellable jokers (negative editions past the s0 8-row cap):
        # s1_schema=True must widen onto SellJokerExt via the SAME
        # k->index mapping the env's action decode uses.
        am = _empty_action_mask()
        am.type_mask[ActionType.SellJoker] = True
        entity = np.ones(15, dtype=bool)
        entity[10] = False  # row 10 not sellable (e.g. eternal)
        am.entity_masks[int(ActionType.SellJoker)] = entity

        mask = shop_action_mask(am, s1_schema=True)
        assert mask.shape == (NUM_TOTAL_ACTIONS_S1,)
        off = FAMILY_OFFSETS[ShopActionFamily.SellJoker]
        assert mask[off : off + 8].all()  # rows 0-7
        ext_off = FAMILY_OFFSETS[ShopActionFamily.SellJokerExt]
        assert mask[ext_off : ext_off + 7].tolist() == [True, True, False, True, True, True, True]
        assert mask.sum() == 8 + 6  # 15 sellable jokers minus row 10

    def test_s1_selljoker_ext_untouched_when_not_overflowing(self):
        am = _empty_action_mask()
        am.type_mask[ActionType.SellJoker] = True
        am.entity_masks[int(ActionType.SellJoker)] = np.ones(3, dtype=bool)

        mask = shop_action_mask(am, s1_schema=True)
        ext_off = FAMILY_OFFSETS[ShopActionFamily.SellJokerExt]
        assert not mask[ext_off : ext_off + 7].any()

    def test_real_shop_state_produces_sane_mask(self):
        # End-to-end sanity: real engine shop state through get_action_mask.
        from jackdaw.engine.actions import CashOut, PlayHand, SelectBlind
        from jackdaw.engine.game import step
        from jackdaw.engine.run_init import initialize_run
        from jackdaw.env.action_space import get_action_mask

        gs = initialize_run("b_red", 1, "SHOP_MASK_TEST")
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())

        mask = shop_action_mask(get_action_mask(gs))
        assert mask[FAMILY_OFFSETS[ShopActionFamily.NextRound]]  # always legal in shop
        assert not mask[:NUM_HAND_ACTIONS].any()
        assert not mask[SELECT_TARGET_BASE:].any()  # no pending target
