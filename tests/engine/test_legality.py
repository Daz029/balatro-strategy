"""Engine Phase 6 — one legality predicate, full vanilla action set.

Every finding is driven through ``step()`` / ``get_legal_actions`` (the
integration path), one class per audit ID. Lua 1.0.1o lines are cited per
test. The environments restrict this vanilla set with their own masks
(rebuilt later, CLAUDE.md TODO 4; pinned by tests/env/test_mask_pin.py).
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jackdaw.engine import lifecycle
from jackdaw.engine.actions import (
    BuyCard,
    Discard,
    GamePhase,
    PickPackCard,
    PlayHand,
    Reroll,
    SelectBlind,
    SellCard,
    UseConsumable,
    get_legal_actions,
)
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run


def _gs(seed: str, phase: GamePhase | None = None) -> dict[str, Any]:
    gs = initialize_run("b_red", 1, seed)
    if phase is GamePhase.SELECTING_HAND:
        step(gs, SelectBlind())
    elif phase is not None:
        gs["phase"] = phase
    return gs


def _own_joker(gs: dict[str, Any], key: str = "j_joker") -> Any:
    card = create_joker(key, game_state=gs)
    lifecycle.emplace(gs, card, "jokers")
    return card


def _own_consumable(gs: dict[str, Any], key: str) -> Any:
    card = create_consumable(key, game_state=gs)
    lifecycle.emplace(gs, card, "consumables")
    return card


def _shop_card(gs: dict[str, Any], card: Any, cost: int) -> Any:
    lifecycle.emplace(gs, card, "shop_cards")
    card.cost = cost
    return card


def _open_pack(gs: dict[str, Any], pack_type: str, cards: list[Any], hand: int = 0) -> None:
    gs["phase"] = GamePhase.PACK_OPENING
    gs["pack_type"] = pack_type
    gs["pack_choices_remaining"] = 1
    gs["pack_cards"] = []
    for card in cards:
        lifecycle.emplace(gs, card, "pack_cards")
    if hand:
        gs["hand"] = [gs["deck"].pop() for _ in range(hand)]


class TestD56StepRejectsImpossibleActions:
    """``button_callbacks.lua`` / ``cardarea.lua`` highlight rules (audit D56)."""

    def test_duplicate_play_indices_are_illegal(self) -> None:
        gs = _gs("P6_D56_DUP", GamePhase.SELECTING_HAND)
        with pytest.raises(IllegalActionError, match="duplicate"):
            step(gs, PlayHand(card_indices=(0, 0)))

    def test_duplicate_discard_indices_are_illegal(self) -> None:
        gs = _gs("P6_D56_DUPD", GamePhase.SELECTING_HAND)
        with pytest.raises(IllegalActionError, match="duplicate"):
            step(gs, Discard(card_indices=(1, 1)))

    def test_negative_target_index_is_rejected_not_wrapped(self) -> None:
        gs = _gs("P6_D56_NEG", GamePhase.SELECTING_HAND)
        _own_consumable(gs, "c_strength")
        with pytest.raises(IllegalActionError, match="target"):
            step(gs, UseConsumable(card_index=0, target_indices=(-1,)))

    def test_out_of_range_target_is_rejected_not_filtered(self) -> None:
        gs = _gs("P6_D56_OOR", GamePhase.SELECTING_HAND)
        _own_consumable(gs, "c_strength")
        with pytest.raises(IllegalActionError, match="target"):
            step(gs, UseConsumable(card_index=0, target_indices=(0, 99)))

    def test_buying_a_joker_into_a_full_row_is_illegal(self) -> None:
        # L: button_callbacks.lua:2392-2401 check_for_buy_space.
        gs = _gs("P6_D56_FULL", GamePhase.SHOP)
        gs["dollars"] = 50
        for _ in range(gs["joker_slots"]):
            _own_joker(gs)
        _shop_card(gs, create_joker("j_greedy_joker", game_state=gs), 5)
        with pytest.raises(IllegalActionError, match="no room"):
            step(gs, BuyCard(0))

    def test_buffoon_pick_into_a_full_row_is_illegal(self) -> None:
        # L: button_callbacks.lua:2112-2120 can_select_card.
        gs = _gs("P6_D56_BUFF")
        for _ in range(gs["joker_slots"]):
            _own_joker(gs)
        _open_pack(gs, "Buffoon", [create_joker("j_greedy_joker", game_state=gs)])
        with pytest.raises(IllegalActionError, match="joker slot"):
            step(gs, PickPackCard(card_index=0))

    def test_negative_joker_fits_a_full_row(self) -> None:
        gs = _gs("P6_D56_NEGJ", GamePhase.SHOP)
        gs["dollars"] = 50
        for _ in range(gs["joker_slots"]):
            _own_joker(gs)
        neg = create_joker("j_greedy_joker", {"negative": True}, game_state=gs)
        _shop_card(gs, neg, 5)
        step(gs, BuyCard(0))
        assert gs["jokers"][-1] is neg

    def test_selling_from_a_non_owned_area_is_illegal(self) -> None:
        gs = _gs("P6_D56_AREA", GamePhase.SHOP)
        with pytest.raises(IllegalActionError, match="area"):
            step(gs, SellCard(area="deck", card_index=0))


class TestD50CreditCardAffordability:
    """``can_buy`` / ``can_open`` / ``can_reroll`` (l.55, 111, 2076) use
    ``dollars - bankrupt_at`` and always allow a free card."""

    def test_credit_card_allows_buying_into_debt(self) -> None:
        gs = _gs("P6_D50_CREDIT", GamePhase.SHOP)
        _own_joker(gs, "j_credit_card")
        assert gs["bankrupt_at"] == -20
        gs["dollars"] = 0
        _shop_card(gs, create_joker("j_greedy_joker", game_state=gs), 6)
        step(gs, BuyCard(0))
        assert gs["dollars"] == -6

    def test_free_card_is_affordable_while_in_debt(self) -> None:
        gs = _gs("P6_D50_FREE", GamePhase.SHOP)
        gs["dollars"] = -3
        _shop_card(gs, create_joker("j_greedy_joker", game_state=gs), 0)
        step(gs, BuyCard(0))
        assert gs["dollars"] == -3

    def test_reroll_into_debt_with_credit_card(self) -> None:
        gs = _gs("P6_D50_REROLL", GamePhase.SHOP)
        _own_joker(gs, "j_credit_card")
        gs["dollars"] = 2
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["dollars"] == -3

    def test_still_cannot_buy_past_the_credit_floor(self) -> None:
        gs = _gs("P6_D50_FLOOR", GamePhase.SHOP)
        _own_joker(gs, "j_credit_card")
        gs["dollars"] = -18
        _shop_card(gs, create_joker("j_greedy_joker", game_state=gs), 5)
        with pytest.raises(IllegalActionError, match="afford"):
            step(gs, BuyCard(0))


class TestD23SellOutsideTheShop:
    """``Card:can_sell_card`` (card.lua:1640-1652): any stable state."""

    @pytest.mark.parametrize(
        "phase", [GamePhase.SELECTING_HAND, GamePhase.BLIND_SELECT, GamePhase.ROUND_EVAL]
    )
    def test_sell_joker_in_phase(self, phase: GamePhase) -> None:
        gs = _gs(f"P6_D23_{phase}", phase if phase is GamePhase.SELECTING_HAND else None)
        if phase is not GamePhase.SELECTING_HAND:
            gs["phase"] = phase
        joker = _own_joker(gs)
        before = gs["dollars"]
        assert SellCard(area="jokers", card_index=0) in get_legal_actions(gs)
        step(gs, SellCard(area="jokers", card_index=0))
        assert joker not in gs["jokers"]
        assert gs["dollars"] == before + joker.sell_cost

    def test_sell_consumable_while_a_pack_is_open(self) -> None:
        gs = _gs("P6_D23_PACK")
        _own_consumable(gs, "c_pluto")
        _open_pack(gs, "Buffoon", [create_joker("j_joker", game_state=gs)])
        step(gs, SellCard(area="consumables", card_index=0))
        assert gs["consumables"] == []


class TestD27CeruleanBell:
    """``blind.lua:575-586``: the forced card is highlighted and cannot be
    unhighlighted, so every play and discard includes it."""

    def _forced(self) -> dict[str, Any]:
        gs = _gs("P6_D27", GamePhase.SELECTING_HAND)
        gs["hand"][2].ability["forced_selection"] = True
        return gs

    def test_play_without_the_forced_card_is_illegal(self) -> None:
        with pytest.raises(IllegalActionError, match="Cerulean"):
            step(self._forced(), PlayHand(card_indices=(0, 1)))

    def test_discard_without_the_forced_card_is_illegal(self) -> None:
        with pytest.raises(IllegalActionError, match="Cerulean"):
            step(self._forced(), Discard(card_indices=(0,)))

    def test_play_with_the_forced_card_is_legal(self) -> None:
        gs = self._forced()
        gs["blind"].chips = 10**9
        step(gs, PlayHand(card_indices=(0, 2)))
        assert gs["current_round"]["hands_left"] == 3

    def test_with_forced_card_completes_a_selection(self) -> None:
        from jackdaw.engine.legality import with_forced_card

        gs = self._forced()
        assert with_forced_card(gs, (0, 1)) == (0, 1, 2)
        assert with_forced_card(gs, (0, 1, 3, 4, 5)) == (0, 1, 3, 4, 2)
        assert with_forced_card(gs, (2, 0)) == (2, 0)


class TestD52BossReroll:
    """``reroll_boss_button`` / ``reroll_boss`` (button_callbacks.lua:2784-2848)."""

    def _blind_select(self, seed: str, voucher: str | None) -> dict[str, Any]:
        gs = _gs(seed)
        gs["dollars"] = 30
        if voucher:
            gs["used_vouchers"][voucher] = True
        return gs

    def test_needs_retcon_or_directors_cut(self) -> None:
        from jackdaw.engine.actions import RerollBoss

        gs = self._blind_select("P6_D52_NONE", None)
        assert RerollBoss() not in get_legal_actions(gs)
        with pytest.raises(IllegalActionError, match="Retcon"):
            step(gs, RerollBoss())

    def test_directors_cut_once_per_ante_for_ten_dollars(self) -> None:
        from jackdaw.engine.actions import RerollBoss

        gs = self._blind_select("P6_D52_DC", "v_directors_cut")
        step(gs, RerollBoss())
        assert gs["dollars"] == 20
        assert gs["round_resets"]["boss_rerolled"] is True
        with pytest.raises(IllegalActionError):
            step(gs, RerollBoss())

    def test_retcon_is_unlimited_and_costs_ten_each(self) -> None:
        from jackdaw.engine.actions import RerollBoss

        gs = self._blind_select("P6_D52_RETCON", "v_retcon")
        bosses = set()
        for _ in range(3):
            step(gs, RerollBoss())
            bosses.add(gs["round_resets"]["blind_choices"]["Boss"])
        assert gs["dollars"] == 0  # NOT free (audit D52)

    def test_needs_ten_dollars_above_the_debt_floor(self) -> None:
        from jackdaw.engine.actions import RerollBoss

        gs = self._blind_select("P6_D52_BROKE", "v_retcon")
        gs["dollars"] = 9
        with pytest.raises(IllegalActionError, match=r"\$10"):
            step(gs, RerollBoss())

    def test_boss_tag_uses_up_directors_cut(self) -> None:
        # L: tag.lua:284-289 — Boss Tag goes through reroll_boss with
        # G.from_boss_tag: free, but it sets round_resets.boss_rerolled.
        from jackdaw.engine.game import _fire_new_blind_choice_tags

        gs = self._blind_select("P6_D52_TAG", "v_directors_cut")
        gs.setdefault("awarded_tags", []).append({"key": "tag_boss", "result": None})
        _fire_new_blind_choice_tags(gs)
        assert gs["dollars"] == 30
        assert gs["round_resets"]["boss_rerolled"] is True


class TestD57PackDecisions:
    """``use_card`` / ``can_use_consumeable`` (button_callbacks.lua:2155;
    card.lua:1523) — audit D57."""

    def test_spectral_picks_are_legal_in_the_engine(self) -> None:
        gs = _gs("P6_D57_SPEC")
        _open_pack(gs, "Spectral", [create_consumable("c_immolate", game_state=gs)], hand=8)
        assert any(isinstance(a, PickPackCard) for a in get_legal_actions(gs))
        step(gs, PickPackCard(card_index=0))

    def test_unusable_pick_is_illegal_not_a_silent_no_op(self) -> None:
        gs = _gs("P6_D57_FOOL")
        gs["last_tarot_planet"] = None
        _open_pack(gs, "Arcana", [create_consumable("c_fool", game_state=gs)], hand=8)
        with pytest.raises(IllegalActionError, match="cannot be used"):
            step(gs, PickPackCard(card_index=0))

    def test_targeted_pick_without_targets_is_illegal(self) -> None:
        gs = _gs("P6_D57_TGT")
        _open_pack(gs, "Arcana", [create_consumable("c_strength", game_state=gs)], hand=8)
        with pytest.raises(IllegalActionError):
            step(gs, PickPackCard(card_index=0))
        step(gs, PickPackCard(card_index=0, target_indices=(0,)))

    def test_owned_consumable_usable_while_a_pack_is_open(self) -> None:
        gs = _gs("P6_D57_USE")
        _own_consumable(gs, "c_pluto")
        _open_pack(gs, "Buffoon", [create_joker("j_joker", game_state=gs)])
        level = gs["hand_levels"]["High Card"].level
        step(gs, UseConsumable(card_index=0))
        assert gs["hand_levels"]["High Card"].level == level + 1
        assert gs["phase"] == GamePhase.PACK_OPENING

    def test_buy_and_use_needs_no_consumable_slot(self) -> None:
        # L: button_callbacks.lua:2407 — buy_and_use skips check_for_buy_space.
        from jackdaw.engine.actions import BuyAndUse

        gs = _gs("P6_D57_BAU", GamePhase.SHOP)
        gs["dollars"] = 20
        for _ in range(gs["consumable_slots"]):
            _own_consumable(gs, "c_pluto")
        _shop_card(gs, create_consumable("c_mercury", game_state=gs), 3)
        with pytest.raises(IllegalActionError, match="no room"):
            step(gs, BuyCard(0))
        level = gs["hand_levels"]["Pair"].level
        step(gs, BuyAndUse(shop_index=0))
        assert gs["hand_levels"]["Pair"].level == level + 1
        assert gs["dollars"] == 17
        assert len(gs["consumables"]) == gs["consumable_slots"]


class TestD39UsabilityCountsDebuffedJokers:
    """``card.lua:4209-4223`` + ``1547-1549``: Hex / Ectoplasm / Wheel are usable
    when an editionless joker exists, debuffed or not."""

    def test_hex_usable_with_only_a_debuffed_editionless_joker(self) -> None:
        gs = _gs("P6_D39_HEX", GamePhase.SHOP)
        joker = _own_joker(gs)
        joker.debuff = True  # fixture: plain Joker has no passive to toggle
        _own_consumable(gs, "c_hex")
        step(gs, UseConsumable(card_index=0))
        assert joker.edition and joker.edition.get("polychrome")


class TestLegalityIsTheOnlySource:
    """Every action get_legal_actions lists is accepted by step(), over real
    seeded runs (the shop-heavy rollout driver steps each one it picks)."""

    def test_rollout_lists_only_legal_actions(self) -> None:
        from jackdaw.engine.legality import is_legal

        from ._rollout import drive

        checked = 0
        for i in range(8):
            for gs, _ in drive(f"P6_ROLLOUT_{i}", random.Random(i), max_steps=300):
                for action in get_legal_actions(gs):
                    if isinstance(action, (PlayHand, Discard)) and not action.card_indices:
                        continue  # marker
                    assert is_legal(gs, action), action
                    checked += 1
        assert checked > 1000
