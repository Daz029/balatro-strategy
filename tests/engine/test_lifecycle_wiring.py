"""End-to-end regressions for Phase 3 lifecycle wiring."""

from __future__ import annotations

import math
import random
from typing import Any

from jackdaw.engine.actions import (
    BuyCard,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    SelectBlind,
    SellCard,
    UseConsumable,
)
from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_card, create_consumable, create_joker
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import _use_consumable_card, step
from jackdaw.engine.lifecycle import emplace, remove
from jackdaw.engine.run_init import initialize_run
from tests.engine._rollout import drive


def _owned_joker(gs: dict[str, Any], key: str) -> Card:
    joker = create_joker(key, game_state=gs)
    emplace(gs, joker, "jokers")
    return joker


def _owned_consumable(gs: dict[str, Any], key: str) -> Card:
    card = create_consumable(key, game_state=gs)
    emplace(gs, card, "consumables")
    return card


def _award(gs: dict[str, Any], key: str) -> None:
    gs.setdefault("awarded_tags", []).append({"key": key, "result": None, "blind": "Small"})


def _deal_one(gs: dict[str, Any]) -> Card:
    card = gs["deck"].pop()
    gs.setdefault("hand", []).append(card)
    return card


def _to_shop(gs: dict[str, Any]) -> None:
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0,)))
    step(gs, CashOut())


class TestD17Notifications:
    def test_cryptid_notifies_hologram_once_with_the_whole_batch(self) -> None:
        gs = initialize_run("b_red", 1, "D17_CRYPTID")
        gs["phase"] = GamePhase.SELECTING_HAND
        hologram = _owned_joker(gs, "j_hologram")
        cryptid = create_consumable("c_cryptid", game_state=gs)
        _deal_one(gs)
        before_ids = {id(card) for card in gs["hand"]}

        _use_consumable_card(gs, cryptid, (0,))

        created = [card for card in gs["hand"] if id(card) not in before_ids]
        assert len(created) == 2
        assert hologram.ability["x_mult"] == 1.5

    def test_marble_notifies_hologram(self) -> None:
        gs = initialize_run("b_red", 1, "D17_MARBLE")
        hologram = _owned_joker(gs, "j_hologram")
        _owned_joker(gs, "j_marble")

        step(gs, SelectBlind())

        assert hologram.ability["x_mult"] == 1.25

    def test_hanged_man_notifies_caino_for_a_face_card(self) -> None:
        gs = initialize_run("b_red", 1, "D17_CAINO")
        gs["phase"] = GamePhase.SELECTING_HAND
        target = _deal_one(gs)
        target.set_base("S_K", "Spades", "King", gs=gs)
        caino = _owned_joker(gs, "j_caino")
        _owned_consumable(gs, "c_hanged_man")

        step(gs, UseConsumable(card_index=0, target_indices=(0,)))

        assert caino.ability["caino_xmult"] == 2
        assert target.removed is True

    def test_glass_joker_counts_scoring_shatter_not_hanged_man(self) -> None:
        gs = initialize_run("b_red", 1, "D17_GLASS_SCORE")
        glass_joker = _owned_joker(gs, "j_glass")
        step(gs, SelectBlind())
        target = gs["hand"][0]
        target.set_ability("m_glass", gs=gs)
        gs["probabilities"]["normal"] = 4
        gs["blind"].chips = 10**9

        step(gs, PlayHand(card_indices=(0,)))

        assert target.shattered is True
        assert target.removed is True
        assert glass_joker.ability["x_mult"] == 1.75

        gs2 = initialize_run("b_red", 1, "D17_GLASS_HANGED")
        gs2["phase"] = GamePhase.SELECTING_HAND
        glass_joker2 = _owned_joker(gs2, "j_glass")
        hanged_target = _deal_one(gs2)
        hanged_target.set_ability("m_glass", gs=gs2)
        _owned_consumable(gs2, "c_hanged_man")
        step(gs2, UseConsumable(card_index=0, target_indices=(0,)))
        assert glass_joker2.ability["x_mult"] == 1

    def test_trading_card_destruction_notifies(self) -> None:
        gs = initialize_run("b_red", 1, "D17_TRADING")
        caino = _owned_joker(gs, "j_caino")
        _owned_joker(gs, "j_trading")
        step(gs, SelectBlind())
        target = gs["hand"][0]
        target.set_base("H_Q", "Hearts", "Queen", gs=gs)

        step(gs, Discard(card_indices=(0,)))

        assert target.removed is True
        assert caino.ability["caino_xmult"] == 2


class TestD37CreatedPlayingCardAreas:
    def test_familiar_created_cards_stay_targetable_until_spectral_pack_closes(self) -> None:
        gs = initialize_run("b_red", 1, "D37_PACK")
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 999
        booster = create_card(
            "Booster",
            gs["rng"],
            1,
            area="shop",
            forced_key="p_spectral_mega_1",
            game_state=gs,
        )
        emplace(gs, booster, "shop_boosters")
        step(gs, OpenBooster(card_index=0))
        for displayed in list(gs["pack_cards"]):
            remove(gs, displayed)
        familiar = create_consumable("c_familiar", game_state=gs)
        aura = create_consumable("c_aura", game_state=gs)
        emplace(gs, familiar, "pack_cards")
        emplace(gs, aura, "pack_cards")
        gs["pack_choices_remaining"] = 2
        before = {id(card) for card in gs["pack_hand"]}

        step(gs, PickPackCard(card_index=0))

        created = [card for card in gs["hand"] if id(card) not in before]
        assert len(created) == 3
        assert all(any(card is packed for packed in gs["pack_hand"]) for card in created)
        target_index = gs["hand"].index(created[0])
        step(gs, PickPackCard(card_index=0, target_indices=(target_index,)))
        assert created[0].edition
        assert gs["phase"] == GamePhase.SHOP
        assert all(any(card is deck_card for deck_card in gs["deck"]) for card in created)


class TestC12EditionSetters:
    def test_two_ectoplasms_each_add_a_live_negative_joker_slot(self) -> None:
        gs = initialize_run("b_red", 1, "C12_ECTO")
        first = _owned_joker(gs, "j_joker")
        second = _owned_joker(gs, "j_greedy_joker")
        ectoplasm = create_consumable("c_ectoplasm", game_state=gs)

        _use_consumable_card(gs, ectoplasm)
        _use_consumable_card(gs, ectoplasm)

        assert first.edition and first.edition.get("negative")
        assert second.edition and second.edition.get("negative")
        assert gs["joker_slots"] == 7

    def test_hex_preserves_eternal_jokers_and_destroys_other_non_eternals(self) -> None:
        gs = initialize_run("b_red", 1, "C12_HEX")
        chosen_candidate = _owned_joker(gs, "j_joker")
        doomed = _owned_joker(gs, "j_greedy_joker")
        eternal = _owned_joker(gs, "j_lusty_joker")
        eternal.set_eternal(True)
        hex_card = create_consumable("c_hex", game_state=gs)

        _use_consumable_card(gs, hex_card)

        assert eternal in gs["jokers"]
        assert len(gs["jokers"]) == 2
        assert doomed.removed or chosen_candidate.removed
        assert any(j.edition and j.edition.get("polychrome") for j in gs["jokers"])


class TestD33DeathCopy:
    def test_death_uses_rightmost_hand_position_and_deep_copies_full_state(self) -> None:
        gs = initialize_run("b_red", 1, "D33_DEATH")
        gs["phase"] = GamePhase.SELECTING_HAND
        older_ace = _deal_one(gs)
        older_ace.set_base("H_A", "Hearts", "Ace", gs=gs)
        older_ace.ability["perma_bonus"] = 25
        newer_two = gs["deck"].pop()
        newer_two.set_base("S_2", "Spades", "2", gs=gs)
        gs["hand"].insert(0, newer_two)
        death = create_consumable("c_death", game_state=gs)

        _use_consumable_card(gs, death, (0, 1))

        assert newer_two.base is not None and newer_two.base.rank is Rank.ACE
        assert older_ace.base is not None and older_ace.base.rank is Rank.ACE
        assert newer_two.ability["perma_bonus"] == older_ace.ability["perma_bonus"] == 25
        assert newer_two.ability is not older_ace.ability


class TestD35AnkhCopy:
    def test_ankh_at_normal_slot_cap_removes_before_copying(self) -> None:
        gs = initialize_run("b_red", 1, "D35_ANKH_FULL")
        for key in (
            "j_joker",
            "j_greedy_joker",
            "j_lusty_joker",
            "j_wrathful_joker",
            "j_gluttenous_joker",
        ):
            _owned_joker(gs, key)

        _use_consumable_card(gs, create_consumable("c_ankh", game_state=gs))

        assert len(gs["jokers"]) == 2

    def test_ankh_removes_others_then_creates_fresh_nonnegative_copy(self) -> None:
        gs = initialize_run("b_red", 1, "D35_ANKH")
        originals = [
            _owned_joker(gs, key)
            for key in (
                "j_joker",
                "j_greedy_joker",
                "j_lusty_joker",
                "j_wrathful_joker",
                "j_gluttenous_joker",
            )
        ]
        for joker in originals:
            joker.set_edition(gs, {"negative": True})
        eternal = _owned_joker(gs, "j_half")
        eternal.set_eternal(True)
        ankh = create_consumable("c_ankh", game_state=gs)
        old_ids = {j.sort_id for j in originals}

        _use_consumable_card(gs, ankh)

        assert eternal in gs["jokers"]
        assert len(gs["jokers"]) == 3
        copies = [j for j in gs["jokers"] if j is not eternal and j.sort_id not in old_ids]
        assert len(copies) == 1
        copy = copies[0]
        source = next(j for j in originals if not j.removed)
        assert not copy.edition or not copy.edition.get("negative")
        assert copy.ability == source.ability
        assert copy.ability is not source.ability


class TestD36MutationDebuffRefresh:
    def test_consumable_suit_change_rechecks_the_live_boss_debuff(self) -> None:
        gs = initialize_run("b_red", 1, "D36_STAR")
        gs["phase"] = GamePhase.SELECTING_HAND
        target = _deal_one(gs)
        target.set_base("C_5", "Clubs", "5", gs=gs)
        gs["blind"] = Blind.create("bl_club", ante=1)
        gs["blind"].refresh_debuffs(gs)
        assert target.debuff is True

        _use_consumable_card(gs, create_consumable("c_star", game_state=gs), (0,))

        assert target.base is not None and target.base.suit is Suit.DIAMONDS
        assert target.debuff is False


class TestD24PerkeoCopy:
    def test_perkeo_negative_copy_owns_and_returns_its_consumable_slot(self) -> None:
        gs = initialize_run("b_red", 1, "D24_PERKEO")
        gs["phase"] = GamePhase.SHOP
        perkeo = _owned_joker(gs, "j_perkeo")
        original = _owned_consumable(gs, "c_hermit")
        slots = gs["consumable_slots"]

        step(gs, NextRound())

        assert perkeo in gs["jokers"]
        assert len(gs["consumables"]) == 2
        duplicate = next(card for card in gs["consumables"] if card is not original)
        assert gs["consumable_slots"] == slots + 1
        assert duplicate.ability == original.ability
        assert duplicate.ability is not original.ability
        gs["phase"] = GamePhase.SHOP
        step(gs, SellCard(area="consumables", card_index=gs["consumables"].index(duplicate)))
        assert gs["consumable_slots"] == slots


class TestC10BlueSealInitialization:
    def test_blue_seal_uses_last_played_hand_and_factory_initializes_planet(self) -> None:
        gs = initialize_run("b_red", 1, "C10_BLUE")
        gs["hand_levels"].record_play("Pair")
        gs["hand_levels"].record_play("Pair")
        step(gs, SelectBlind())
        held_blue = gs["hand"][1]
        held_blue.set_seal(gs, "Blue")
        gs["blind"].chips = 1

        step(gs, PlayHand(card_indices=(0,)))

        assert gs["last_hand_played"] == "High Card"
        assert len(gs["consumables"]) == 1
        planet = gs["consumables"][0]
        assert planet.center_key == "c_pluto"
        assert planet.ability.get("consumeable", {}).get("hand_type") == "High Card"
        assert planet.cost > 0
        assert planet.added_to_deck is True


class TestD20TurtleBeanDecay:
    def test_live_hand_size_decays_and_final_removal_returns_exactly_to_base(self) -> None:
        gs = initialize_run("b_red", 1, "D20_BEAN")
        base = gs["hand_size"]
        bean = _owned_joker(gs, "j_turtle_bean")
        assert gs["hand_size"] == base + 5

        for round_index in range(5):
            step(gs, SelectBlind())
            gs["blind"].chips = 1
            gs["chips"] = 1
            step(gs, PlayHand(card_indices=(0,)))
            if round_index == 1:
                assert gs["hand_size"] == base + 3
            if round_index < 4:
                step(gs, CashOut())
                step(gs, NextRound())

        assert bean.removed is True
        assert gs["hand_size"] == base


class TestD53TagPricing:
    def test_uncommon_tag_joker_stays_free_when_astronomer_reprices_shop(self) -> None:
        gs = initialize_run("b_red", 1, "D53_UNCOMMON")
        _award(gs, "tag_uncommon")
        _to_shop(gs)
        tagged = gs["shop_cards"][0]
        assert tagged.ability.get("couponed") is True
        assert tagged.cost == 0
        astronomer = create_joker("j_astronomer", game_state=gs)
        emplace(gs, astronomer, "shop_cards")
        gs["dollars"] = 999

        step(gs, BuyCard(shop_index=gs["shop_cards"].index(astronomer)))

        assert tagged.cost == 0

    def test_edition_tag_coupon_reprices_sell_value_with_lua_formula(self) -> None:
        gs = initialize_run("b_red", 1, "D53_FOIL")
        _award(gs, "tag_uncommon")
        _award(gs, "tag_foil")
        _to_shop(gs)
        tagged = gs["shop_cards"][0]

        assert tagged.ability.get("couponed") is True
        assert tagged.cost == 0
        expected_sell = max(1, math.floor((tagged.base_cost + tagged.extra_cost) / 2))
        assert tagged.sell_cost == expected_sell


class TestStructuralLifecycleRollout:
    def test_owned_area_and_slot_invariants_after_every_step(self) -> None:
        for seed_index in range(20):
            for gs, action in drive(f"P3_3_{seed_index}", random.Random(seed_index)):
                owned = (*gs["jokers"], *gs["consumables"])
                assert all(card.added_to_deck for card in owned), action

                areas = (
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
                )
                seen: dict[int, str] = {}
                for area in areas:
                    for card in gs.get(area, []):
                        previous = seen.setdefault(id(card), area)
                        assert previous == area, (action, previous, area, card)
                assert all(
                    any(card is hand_card for hand_card in gs["hand"])
                    for card in gs.get("pack_hand", [])
                )

                negative_jokers = sum(
                    bool(card.edition and card.edition.get("negative")) for card in gs["jokers"]
                )
                negative_consumables = sum(
                    bool(card.edition and card.edition.get("negative"))
                    for card in gs["consumables"]
                )
                base_joker_slots = gs["starting_params"]["joker_slots"] + int(
                    "v_antimatter" in gs["used_vouchers"]
                )
                base_consumable_slots = gs["starting_params"]["consumable_slots"] + int(
                    "v_crystal_ball" in gs["used_vouchers"]
                )
                assert gs["joker_slots"] == base_joker_slots + negative_jokers, action
                assert gs["consumable_slots"] == base_consumable_slots + negative_consumables, (
                    action
                )
