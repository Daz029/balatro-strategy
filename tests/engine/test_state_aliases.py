"""Regression coverage for canonical stored-state keys and their writers."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    PlayHand,
    RedeemVoucher,
    SelectBlind,
    UseConsumable,
)
from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import (
    create_card,
    create_consumable,
    create_joker,
    create_playing_card,
    create_voucher,
)
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.economy import calculate_round_earnings
from jackdaw.engine.game import step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import init_game_object, initialize_run
from jackdaw.engine.scoring import score_hand


def _award(gs: dict[str, Any], key: str) -> dict[str, Any]:
    entry: dict[str, Any] = {"key": key, "result": None, "blind": "Small"}
    gs.setdefault("awarded_tags", []).append(entry)
    return entry


def _beat_blind(gs: dict[str, Any]) -> None:
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0,)))


def _cash_out_to_shop(gs: dict[str, Any]) -> None:
    _beat_blind(gs)
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP


def _wheel_seed_between_old_and_oops_thresholds() -> str:
    for index in range(1000):
        seed = f"WHEEL_ALIAS_{index}"
        roll = PseudoRandom(seed).random("wheel_of_fortune")
        if 0.25 <= roll < 0.5:
            return seed
    raise AssertionError("could not find deterministic Wheel threshold seed")


class TestC04Probabilities:
    def test_oops_updates_only_canonical_probability_table(self):
        gs = init_game_object()
        gs["probabilities"]["custom"] = 1.5
        oops = create_joker("j_oops")

        oops.add_to_deck(gs)
        assert gs["probabilities"] == {"normal": 2, "custom": 3.0}
        assert "probabilities_normal" not in gs

        oops.remove_from_deck(gs)
        assert gs["probabilities"] == {"normal": 1, "custom": 1.5}

    def test_wheel_reads_probability_doubled_by_oops(self):
        gs = initialize_run("b_red", 1, _wheel_seed_between_old_and_oops_thresholds())
        oops = create_joker("j_oops")
        target = create_joker("j_joker")
        gs["jokers"] = [oops, target]
        oops.add_to_deck(gs)
        gs["consumables"] = [create_consumable("c_wheel_of_fortune")]
        gs["phase"] = GamePhase.SHOP

        step(gs, UseConsumable(card_index=0))

        assert gs["probabilities"]["normal"] == 2
        assert "probabilities_normal" not in gs
        assert any(joker.edition is not None for joker in (oops, target))


class TestC07ShopStickerFlags:
    def test_stake_eight_modifiers_reach_shop_joker_factory(self):
        gs = initialize_run("b_red", 8, "S15")

        joker = create_card(
            "Joker",
            PseudoRandom("S15"),
            1,
            forced_key="j_joker",
            game_state=gs,
        )

        assert joker.eternal is True
        assert joker.rental is True


class TestC14GreenDeckEconomy:
    def test_green_deck_uses_only_modifier_economy_keys(self):
        gs = initialize_run("b_green", 1, "GREEN_ALIAS")

        assert gs["modifiers"]["money_per_hand"] == 2
        assert gs["modifiers"]["money_per_discard"] == 1
        assert gs["modifiers"]["no_interest"] is True
        assert "money_per_hand" not in gs
        assert "money_per_discard" not in gs
        assert "no_interest" not in gs

        earnings = calculate_round_earnings(
            blind=Blind.create("bl_small", ante=1),
            hands_left=2,
            discards_left=3,
            money=20,
            jokers=[],
            game_state=gs,
        )
        assert earnings.unused_hands_bonus == 4
        assert earnings.unused_discards_bonus == 3
        assert earnings.interest == 0


class TestD26ChaosFreeRerolls:
    def test_chaos_reprices_current_shop_rerolls(self):
        gs = initialize_run("b_red", 1, "CHAOS_ALIAS")
        gs["phase"] = GamePhase.SHOP
        chaos = create_joker("j_chaos")

        chaos.add_to_deck(gs)

        assert gs["current_round"]["free_rerolls"] == 1
        assert gs["current_round"]["reroll_cost"] == 0
        assert "free_rerolls" not in gs


class TestD40ConsumeableUsageSpelling:
    def test_planet_step_updates_only_lua_spelled_totals(self):
        gs = initialize_run("b_red", 1, "PLANET_ALIAS")
        gs["phase"] = GamePhase.SHOP
        gs["consumables"] = [create_consumable("c_mercury")]

        step(gs, UseConsumable(card_index=0))

        assert gs["consumeable_usage_total"]["planet"] == 1
        assert gs["consumeable_usage_total"]["all"] == 1
        assert "consumable_usage_total" not in gs


class TestD04LuckyTrigger:
    def test_score_hand_exposes_and_then_clears_top_level_lucky_trigger(self):
        lucky = create_playing_card(Suit.HEARTS, Rank.FIVE, enhancement="m_lucky")
        lucky_cat = create_joker("j_lucky_cat")

        score_hand(
            played_cards=[lucky],
            held_cards=[],
            jokers=[lucky_cat],
            hand_levels=HandLevels(),
            blind=Blind.create("bl_small", ante=1),
            rng=PseudoRandom("LUCKY_ALIAS"),
            probabilities_normal=100,
            game_state={"probabilities": {"normal": 0}},
        )

        assert lucky_cat.ability["x_mult"] == pytest.approx(1.25)
        assert lucky.lucky_trigger is False
        assert "lucky_trigger" not in lucky.ability


class TestC15ChallengeModifierAliases:
    def test_booster_ante_scaling_is_read_from_modifiers(self):
        gs = init_game_object()
        gs["modifiers"]["booster_ante_scaling"] = True
        gs["round_resets"]["ante"] = 3

        booster = create_card(
            "Booster",
            PseudoRandom("BOOSTER_ALIAS"),
            3,
            forced_key="p_arcana_normal_1",
            game_state=gs,
        )

        assert booster.cost == 6


class TestC13OmenGlobeCanonicalFlag:
    def test_redeem_omen_globe_uses_used_vouchers_only(self):
        gs = initialize_run("b_red", 1, "OMEN_ALIAS")
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 20
        gs["shop_vouchers"] = [create_voucher("v_omen_globe")]

        step(gs, RedeemVoucher(card_index=0))

        assert gs["used_vouchers"]["v_omen_globe"] is True
        assert "omen_globe" not in gs


class TestShopTagOnePerShopFlags:
    @pytest.mark.parametrize(
        ("tag_key", "flag"),
        [("tag_d_six", "shop_d6ed"), ("tag_coupon", "shop_free")],
    )
    def test_second_same_tag_waits_for_next_shop(self, tag_key: str, flag: str):
        gs = initialize_run("b_red", 1, f"DOUBLE_{tag_key}")
        assert gs["shop_d6ed"] is False
        assert gs["shop_free"] is False
        first = _award(gs, tag_key)
        second = _award(gs, tag_key)

        _cash_out_to_shop(gs)

        assert gs[flag] is True
        assert first["consumed"] is True
        assert not second.get("consumed", False)

        step(gs, NextRound())
        _cash_out_to_shop(gs)

        assert second["consumed"] is True


class TestC10bD29NewStoredFieldWriters:
    def test_play_hand_writes_last_hand_played_before_scoring_returns(self):
        gs = initialize_run("b_red", 1, "LAST_HAND_WRITER")
        step(gs, SelectBlind())
        gs["blind"].chips = 10**9

        step(gs, PlayHand(card_indices=(0,)))

        assert gs["last_hand_played"] == gs["last_score_result"].hand_type

    def test_boss_win_writes_most_played_poker_hand(self):
        gs = initialize_run("b_red", 1, "BOSS_HAND_WRITER")
        gs["blind_on_deck"] = "Boss"
        gs["hand_levels"].get_state(HandType.PAIR).played = 5
        step(gs, SelectBlind())
        gs["blind"].chips = 1

        step(gs, PlayHand(card_indices=(0,)))

        assert gs["current_round"]["most_played_poker_hand"] == "Pair"

    def test_non_boss_win_leaves_most_played_poker_hand_unchanged(self):
        gs = initialize_run("b_red", 1, "SMALL_HAND_WRITER")
        gs["current_round"]["most_played_poker_hand"] = "Straight"

        _beat_blind(gs)

        assert gs["current_round"]["most_played_poker_hand"] == "Straight"

    def test_boss_tie_uses_stronger_lower_order_hand(self):
        gs = initialize_run("b_red", 1, "BOSS_HAND_TIE")
        gs["blind_on_deck"] = "Boss"
        gs["hand_levels"].get_state(HandType.PAIR).played = 5
        gs["hand_levels"].get_state(HandType.TWO_PAIR).played = 5
        step(gs, SelectBlind())
        gs["blind"].chips = 1

        step(gs, PlayHand(card_indices=(0,)))

        assert gs["current_round"]["most_played_poker_hand"] == "Two Pair"


class TestStateAliasMigration:
    def test_migrate_all_aliases_in_place_and_idempotently(self):
        from jackdaw.engine.state import migrate_state

        cards = [Card() for _ in range(9)]
        for card in cards:
            card.ability["lucky_trigger"] = True
        blind = Blind.create("bl_small", ante=1)
        reset_blind = Blind.create("bl_big", ante=1)
        blind.hands = {"Pair": False}
        reset_blind.hands = {"Flush": False}
        gs: dict[str, Any] = {
            "probabilities": {"normal": 1},
            "probabilities_normal": 4,
            "modifiers": {"money_per_hand": 9},
            "money_per_hand": 2,
            "money_per_discard": 1,
            "no_interest": True,
            "free_rerolls": 7,
            "consumeable_usage_total": {"tarot": 3},
            "consumable_usage_total": {"planet": 8, "all": 8},
            "omen_globe": True,
            "used_vouchers": {},
            "blind": blind,
            "round_resets": {"blind": reset_blind},
        }
        for area, card in zip(
            (
                "hand",
                "deck",
                "discard_pile",
                "jokers",
                "consumables",
                "played_cards_area",
                "shop_cards",
                "pack_cards",
                "pack_hand",
            ),
            cards,
            strict=True,
        ):
            gs[area] = [card]

        migrated = migrate_state(gs)

        assert migrated is gs
        assert gs["probabilities"]["normal"] == 4
        assert gs["modifiers"] == {
            "money_per_hand": 9,
            "money_per_discard": 1,
            "no_interest": True,
        }
        assert gs["consumeable_usage_total"] == {
            "tarot": 3,
            "planet": 0,
            "spectral": 0,
            "tarot_planet": 0,
            "all": 0,
        }
        assert gs["used_vouchers"]["v_omen_globe"] is True
        for alias in (
            "probabilities_normal",
            "money_per_hand",
            "money_per_discard",
            "no_interest",
            "free_rerolls",
            "consumable_usage_total",
            "omen_globe",
        ):
            assert alias not in gs
        assert gs["dollar_buffer"] == 0
        assert gs["last_hand_played"] is None
        assert gs["orbital_choices"] == {}
        assert gs["facing_blind"] is False
        assert gs["shop_free"] is False
        assert gs["shop_d6ed"] is False
        assert all("lucky_trigger" not in card.ability for card in cards)
        assert "hands" not in blind.__dict__
        assert "hands" not in reset_blind.__dict__

        once = pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL)
        assert migrate_state(gs) is gs
        assert pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL) == once
