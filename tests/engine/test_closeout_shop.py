"""Engine CO-2 integration regressions for shop, tags, run init, and RNG."""

from __future__ import annotations

import math
from typing import Any

from jackdaw.engine import lifecycle, read
from jackdaw.engine.actions import (
    BuyCard,
    CashOut,
    GamePhase,
    NextRound,
    OpenBooster,
    PlayHand,
    RedeemVoucher,
    Reroll,
    SelectBlind,
    SkipBlind,
)
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.game import step
from jackdaw.engine.pools import check_soul_chance
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run


def _beat_blind_to_shop(gs: dict[str, Any]) -> None:
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0,)))
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP


def _shop_with_voucher(
    voucher_key: str,
    seed: str,
    *,
    voucher_tag: bool = False,
) -> dict[str, Any]:
    gs = initialize_run("b_red", 1, seed)
    gs["current_round"]["voucher"] = voucher_key
    if voucher_tag:
        gs.setdefault("awarded_tags", []).append(
            {"key": "tag_voucher", "result": None, "blind": "Small"}
        )
    gs["dollars"] = 100
    _beat_blind_to_shop(gs)
    return gs


def _open_pack(seed: str, pack_key: str, *, showman: bool, existing_soul: bool) -> dict[str, Any]:
    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.SHOP
    gs["dollars"] = 100
    if existing_soul:
        lifecycle.emplace(gs, create_consumable("c_soul", game_state=gs), "consumables")
    if showman:
        lifecycle.emplace(gs, create_joker("j_ring_master", game_state=gs), "jokers")
    pack = Card(center_key=pack_key, cost=0)
    pack.ability = {"set": "Booster", "name": pack_key}
    gs["shop_boosters"] = [pack]
    step(gs, OpenBooster(card_index=0))
    return gs


class TestC14StartingDeckEffects:
    """Magic/Ghost creation is back.lua:184-195; Anaglyph is back.lua:111-119."""

    def test_magic_deck_starts_with_two_real_fools(self) -> None:
        gs = initialize_run("b_magic", 1, "CO2_MAGIC")

        assert [card.center_key for card in gs["consumables"]] == ["c_fool", "c_fool"]
        assert all(card.added_to_deck for card in gs["consumables"])
        assert all(card in read.all_cards(gs) for card in gs["consumables"])
        assert not any(key.startswith("soul_") for key in gs["rng"].get_state())

    def test_ghost_deck_starts_with_real_hex_without_rng(self) -> None:
        gs = initialize_run("b_ghost", 1, "CO2_GHOST")

        assert [card.center_key for card in gs["consumables"]] == ["c_hex"]
        assert gs["consumables"][0].ability["set"] == "Spectral"
        assert gs["consumables"][0].added_to_deck is True
        assert not any(key.startswith("soul_") for key in gs["rng"].get_state())

    def test_green_deck_run_init_keeps_its_economy_modifiers(self) -> None:
        gs = initialize_run("b_green", 1, "CO2_GREEN")

        assert gs["modifiers"]["money_per_hand"] == 2
        assert gs["modifiers"]["money_per_discard"] == 1
        assert gs["modifiers"]["no_interest"] is True

    def test_anaglyph_boss_defeat_awards_double_tag(self) -> None:
        gs = initialize_run("b_anaglyph", 1, "CO2_ANAGLYPH")
        gs["blind_on_deck"] = "Boss"
        gs["round_resets"]["blind_states"]["Boss"] = "Select"
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_arm"

        step(gs, SelectBlind(allow_forced_boss=True))
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0,)))

        assert "tag_double" in [entry["key"] for entry in gs.get("awarded_tags", [])]


class TestD45ShopHiddenCards:
    """create_card_for_shop omits soulable at UI_definitions.lua:742-776."""

    def test_seed_sweep_shop_tarot_and_planet_slots_never_become_hidden_cards(self) -> None:
        cases = (("Tarot", "c_soul"), ("Planet", "c_black_hole"))
        for card_type, hidden_key in cases:
            # These are the first seeds in this sweep whose source hidden-card
            # stream exceeds 0.997. With the buggy soulable shop call they
            # produce Soul / Black Hole instead of an ordinary shop card.
            seed = next(
                candidate
                for i in range(500)
                if PseudoRandom(candidate := f"CO2_{card_type.upper()}_{i}").random(
                    f"soul_{card_type}1"
                )
                > 0.997
            )
            gs = initialize_run("b_red", 1, seed)
            gs["shop"]["joker_max"] = 1
            gs["joker_rate"] = 0
            gs["tarot_rate"] = int(card_type == "Tarot")
            gs["planet_rate"] = int(card_type == "Planet")
            gs["spectral_rate"] = 0
            gs["playing_card_rate"] = 0

            _beat_blind_to_shop(gs)

            assert [card.center_key for card in gs["shop_cards"]] != [hidden_key]
            assert f"soul_{card_type}1" not in gs["rng"].get_state()


class TestD48RngCallSites:
    """Rental short-circuits at common_events.lua:2144; Hallucination is card.lua:2337."""

    def test_white_stake_shop_does_not_advance_rental_stream(self) -> None:
        white = initialize_run("b_red", 1, "CO2_RENTAL_WHITE")
        white["shop"]["joker_max"] = 1
        white["joker_rate"] = 1
        white["tarot_rate"] = white["planet_rate"] = white["spectral_rate"] = 0
        white["playing_card_rate"] = 0
        _beat_blind_to_shop(white)
        assert "ssjr1" not in white["rng"].get_state()

        gold = initialize_run("b_red", 8, "CO2_RENTAL_GOLD")
        gold["shop"]["joker_max"] = 1
        gold["joker_rate"] = 1
        gold["tarot_rate"] = gold["planet_rate"] = gold["spectral_rate"] = 0
        gold["playing_card_rate"] = 0
        _beat_blind_to_shop(gold)
        assert "ssjr1" in gold["rng"].get_state()

    def test_hallucination_uses_halu_plus_ante(self) -> None:
        gs = initialize_run("b_red", 1, "CO2_HALU")
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 100
        lifecycle.emplace(gs, create_joker("j_hallucination", game_state=gs), "jokers")
        pack = Card(center_key="p_arcana_normal_1", cost=0)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]

        step(gs, OpenBooster(card_index=0))

        assert "halu1" in gs["rng"].get_state()
        assert "hallucination" not in gs["rng"].get_state()


class TestD51VoucherShopRefresh:
    """Overstock and discounts refresh live cards at card.lua:1885-1923."""

    def test_overstock_adds_the_new_current_shop_slot(self) -> None:
        gs = _shop_with_voucher("v_overstock_norm", "CO2_OVERSTOCK")
        before = len(gs["shop_cards"])
        idx = next(
            i
            for i, card in enumerate(gs["shop_vouchers"])
            if card.center_key == "v_overstock_norm"
        )

        step(gs, RedeemVoucher(card_index=idx))

        assert gs["shop"]["joker_max"] == before + 1
        assert len(gs["shop_cards"]) == before + 1

    def test_clearance_reprices_every_live_area_and_owned_sell_value(self) -> None:
        gs = initialize_run("b_red", 1, "CO2_CLEARANCE")
        owned_joker = create_joker("j_stuntman", game_state=gs)
        owned_consumable = create_consumable("c_ankh", game_state=gs)
        lifecycle.emplace(gs, owned_joker, "jokers")
        lifecycle.emplace(gs, owned_consumable, "consumables")
        gs["current_round"]["voucher"] = "v_clearance_sale"
        gs.setdefault("awarded_tags", []).append(
            {"key": "tag_voucher", "result": None, "blind": "Small"}
        )
        gs["dollars"] = 100
        _beat_blind_to_shop(gs)
        clearance = next(
            card for card in gs["shop_vouchers"] if card.center_key == "v_clearance_sale"
        )
        repriced = [
            *gs["shop_cards"],
            *(card for card in gs["shop_vouchers"] if card is not clearance),
            *gs["shop_boosters"],
            owned_joker,
            owned_consumable,
        ]
        before = {id(card): (card.cost, card.sell_cost) for card in repriced}
        idx = gs["shop_vouchers"].index(clearance)

        step(gs, RedeemVoucher(card_index=idx))

        assert gs["discount_percent"] == 25
        for card in repriced:
            expected_cost = max(1, math.floor((card.base_cost + card.extra_cost + 0.5) * 0.75))
            expected_sell = max(1, math.floor(expected_cost / 2)) + card.ability.get(
                "extra_value", 0
            )
            assert (card.cost, card.sell_cost) == (expected_cost, expected_sell)
        assert any((card.cost, card.sell_cost) != before[id(card)] for card in repriced)
        assert owned_joker.sell_cost != before[id(owned_joker)][1]


class TestD54OrbitalTagOffer:
    """Orbital stores at tag.lua:103-111 and Double Tag copies at tag.lua:320-328."""

    def test_offer_fixes_visible_hand_and_double_tag_copies_it(self) -> None:
        gs = initialize_run("b_red", 1, "CO2_ORBITAL_2_28")
        for hand_type in HandType:
            gs["hand_levels"].get_state(hand_type).visible = hand_type is HandType.PAIR
        gs["blind_on_deck"] = "Boss"
        gs["round_resets"]["blind_states"]["Boss"] = "Select"
        step(gs, SelectBlind(allow_forced_boss=True))
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0,)))
        # Playing the winning High Card reveals it; isolate the offer-time
        # eligibility assertion immediately before the blind choices appear.
        gs["hand_levels"].get_state(HandType.HIGH_CARD).visible = False
        step(gs, CashOut())

        assert gs["round_resets"]["ante"] == 2
        assert gs["round_resets"]["blind_tags"]["Small"] == "tag_orbital"
        assert gs["orbital_choices"][2]["Small"] == HandType.PAIR.value

        # Change visibility after the offer. Redemption must still use Pair.
        gs["hand_levels"].get_state(HandType.PAIR).visible = False
        gs["hand_levels"].get_state(HandType.HIGH_CARD).visible = True
        gs.setdefault("awarded_tags", []).append(
            {"key": "tag_double", "result": None, "blind": None}
        )
        before = gs["hand_levels"].get_state(HandType.PAIR).level
        step(gs, NextRound())
        step(gs, SkipBlind())

        assert gs["hand_levels"].get_state(HandType.PAIR).level == before + 6
        orbitals = [entry for entry in gs["awarded_tags"] if entry["key"] == "tag_orbital"]
        assert len(orbitals) == 2
        assert {entry["orbital_hand"] for entry in orbitals} == {HandType.PAIR.value}


class TestD55VoucherTagRedemption:
    """Card:redeem clears current_round.voucher at card.lua:1850."""

    def test_extra_voucher_clears_current_voucher_like_lua(self) -> None:
        ordinary_key = "v_clearance_sale"
        gs = _shop_with_voucher(
            ordinary_key,
            "CO2_VOUCHER_EXTRA",
            voucher_tag=True,
        )
        extra_idx = next(
            i for i, card in enumerate(gs["shop_vouchers"]) if card.center_key != ordinary_key
        )

        step(gs, RedeemVoucher(card_index=extra_idx))
        assert gs["current_round"]["voucher"] is None


class TestD43HiddenCardChecks:
    """Soul/Black Hole existing-card and Showman gates are common_events.lua:2087-2100."""

    def test_existing_soul_blocks_duplicate_unless_showman_is_owned(self) -> None:
        blocked = _open_pack(
            "CO2_SPECTRAL_410",
            "p_spectral_normal_1",
            showman=False,
            existing_soul=True,
        )
        assert "c_soul" not in [card.center_key for card in blocked["pack_cards"]]

        allowed = _open_pack(
            "CO2_SPECTRAL_410",
            "p_spectral_normal_1",
            showman=True,
            existing_soul=True,
        )
        assert "c_soul" in [card.center_key for card in allowed["pack_cards"]]

    def test_spectral_rolls_both_streams_and_black_hole_overrides_soul(self) -> None:
        """``common_events.lua:2088-2099``: no early exit between the two rolls.

        A scripted RNG makes both rolls hit (a seed for that is ~1 in 1.1e5):
        Lua reassigns ``forced_key`` to Black Hole and has drawn twice.
        """

        class _ScriptedRng:
            def __init__(self) -> None:
                self.keys: list[str] = []

            def random(self, key: str) -> float:
                self.keys.append(key)
                return 0.999

        rng = _ScriptedRng()
        assert check_soul_chance("Spectral", rng, 3) == "c_black_hole"
        assert rng.keys == ["soul_Spectral3", "soul_Spectral3"]

    def test_tarot_pack_can_roll_soul(self) -> None:
        gs = _open_pack(
            "CO2_TAROT_145",
            "p_arcana_normal_1",
            showman=False,
            existing_soul=False,
        )
        assert "c_soul" in [card.center_key for card in gs["pack_cards"]]


class TestD59JokerPurchaseCounter:
    """Only Joker purchases increment at button_callbacks.lua:2440-2449."""

    def test_counter_ignores_consumables_and_counts_jokers(self) -> None:
        gs = initialize_run("b_red", 1, "CO2_PURCHASE_COUNTER")
        gs["shop"]["joker_max"] = 1
        gs["joker_rate"] = 0
        gs["tarot_rate"] = 1
        gs["planet_rate"] = gs["spectral_rate"] = gs["playing_card_rate"] = 0
        gs["dollars"] = 100
        _beat_blind_to_shop(gs)

        step(gs, BuyCard(shop_index=0))
        assert gs["current_round"]["jokers_purchased"] == 0

        gs["joker_rate"] = 1
        gs["tarot_rate"] = 0
        step(gs, Reroll())
        assert gs["shop_cards"][0].ability["set"] == "Joker"
        step(gs, BuyCard(shop_index=0))
        assert gs["current_round"]["jokers_purchased"] == 1
