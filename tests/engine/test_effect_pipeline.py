"""Integration regressions for the joker Effect pipeline (engine P4-2)."""

from __future__ import annotations

import pytest

from jackdaw.engine import lifecycle, read
from jackdaw.engine.actions import CashOut, NextRound, PlayHand, SelectBlind
from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.engine.game import step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand


def _owned_joker(gs: dict, key: str) -> Card:
    joker = create_joker(key, game_state=gs)
    lifecycle.emplace(gs, joker, "jokers")
    return joker


def _select_with_single_card(gs: dict, rank: str) -> Card:
    step(gs, SelectBlind())
    hand = gs["hand"]
    card = hand[0]
    rank_key = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}.get(rank, rank)
    card.set_base(f"H_{rank_key}", "Hearts", rank, gs=gs)
    gs["deck"].extend(hand[1:])
    gs["hand"] = [card]
    gs["blind"].chips = 10**9
    return card


def _playing_count(gs: dict) -> int:
    return len(read.playing_cards(gs))


class TestScoringEffects:
    def test_dna_first_single_hand_copies_into_hand(self):
        gs = initialize_run("b_red", 1, "P42_DNA")
        _owned_joker(gs, "j_dna")
        played = _select_with_single_card(gs, "Ace")
        before = _playing_count(gs)

        step(gs, PlayHand(card_indices=(0,)))

        assert _playing_count(gs) == before + 1
        assert any(
            card is not played
            and (card.base.suit, card.base.rank) == (played.base.suit, played.base.rank)
            for card in gs["hand"]
        )

    @pytest.mark.parametrize("full_consumables", [False, True])
    def test_sixth_sense_always_destroys_six_and_only_creates_with_room(
        self, full_consumables: bool
    ):
        gs = initialize_run("b_red", 1, f"P42_SIXTH_{full_consumables}")
        _owned_joker(gs, "j_sixth_sense")
        six = _select_with_single_card(gs, "6")
        if full_consumables:
            for key in ("c_fool", "c_magician"):
                lifecycle.emplace(gs, create_consumable(key, game_state=gs), "consumables")
        before_consumables = len(gs["consumables"])

        step(gs, PlayHand(card_indices=(0,)))

        assert six.removed is True
        assert six not in read.playing_cards(gs)
        expected = before_consumables if full_consumables else before_consumables + 1
        assert len(gs["consumables"]) == expected
        if not full_consumables:
            assert gs["consumables"][-1].ability["set"] == "Spectral"

    def test_eight_ball_full_consumables_draws_no_rng(self):
        gs = initialize_run("b_red", 1, "P42_8BALL")
        _owned_joker(gs, "j_8_ball")
        _select_with_single_card(gs, "8")
        for key in ("c_fool", "c_magician"):
            lifecycle.emplace(gs, create_consumable(key, game_state=gs), "consumables")

        step(gs, PlayHand(card_indices=(0,)))

        assert "8ball" not in gs["rng"].get_state()


class TestSettingBlindEffects:
    def test_riff_raff_and_cartomancer_share_one_capacity_aware_pass(self):
        states = []
        for second_joker in ("j_cartomancer", "j_joker"):
            gs = initialize_run("b_red", 1, "P42_SHARED_ROOM")
            _owned_joker(gs, "j_riff_raff")
            _owned_joker(gs, second_joker)
            gs["joker_slots"] = 3
            gs["consumable_slots"] = 1
            lifecycle.emplace(gs, create_consumable("c_fool", game_state=gs), "consumables")

            step(gs, SelectBlind())

            # Riff-raff has one free Joker slot and must create exactly one;
            # Cartomancer has no consumable room and must not roll a card.
            assert len(gs["jokers"]) == 3
            assert len(gs["consumables"]) == 1
            states.append(gs["rng"].get_state())

        assert states[0] == states[1]

    def test_dagger_removes_neighbor_and_riff_raff_sees_freed_slot(self):
        gs = initialize_run("b_red", 1, "P42_DAGGER_RIFF")
        dagger = _owned_joker(gs, "j_ceremonial")
        victim = _owned_joker(gs, "j_joker")
        victim.sell_cost = 3
        riff_raff = _owned_joker(gs, "j_riff_raff")
        gs["joker_slots"] = 3

        step(gs, SelectBlind())

        assert victim.removed is True
        assert dagger.ability["mult"] == 6
        assert dagger in gs["jokers"] and riff_raff in gs["jokers"]
        assert len(gs["jokers"]) == 3

        eternal_gs = initialize_run("b_red", 1, "P42_DAGGER_ETERNAL")
        eternal_dagger = _owned_joker(eternal_gs, "j_ceremonial")
        eternal_victim = _owned_joker(eternal_gs, "j_joker")
        eternal_victim.eternal = True
        eternal_victim.sell_cost = 3
        step(eternal_gs, SelectBlind())
        assert eternal_victim in eternal_gs["jokers"]
        assert eternal_victim.removed is False
        assert eternal_dagger.ability["mult"] == 0

    @pytest.mark.parametrize("madness_position", [0, 1])
    def test_madness_never_destroys_itself_or_eternal(self, madness_position: int):
        gs = initialize_run("b_red", 1, f"P42_MADNESS_{madness_position}")
        madness = create_joker("j_madness", game_state=gs)
        eternal = create_joker("j_joker", game_state=gs)
        eternal.eternal = True
        cards = [eternal]
        cards.insert(madness_position, madness)
        for card in cards:
            lifecycle.emplace(gs, card, "jokers")

        step(gs, SelectBlind())

        assert madness in gs["jokers"]
        assert eternal in gs["jokers"]

    def test_burglar_applies_after_round_reset(self):
        gs = initialize_run("b_red", 1, "P42_BURGLAR")
        _owned_joker(gs, "j_burglar")

        step(gs, SelectBlind())

        assert gs["current_round"]["hands_left"] == gs["round_resets"]["hands"] + 3
        assert gs["current_round"]["discards_left"] == 0


class TestRunWideAndBlindTiming:
    def test_loyalty_card_fires_on_sixth_run_wide_hand_across_rounds(self):
        gs = initialize_run("b_red", 1, "P42_LOYALTY")
        _owned_joker(gs, "j_loyalty_card")

        for hand_number in range(1, 7):
            step(gs, SelectBlind())
            gs["blind"].chips = 1
            step(gs, PlayHand(card_indices=(0,)))
            result = gs["last_score_result"]
            assert result.mult == (4 if hand_number == 6 else 1)
            if hand_number < 6:
                step(gs, CashOut())
                step(gs, NextRound())

    @pytest.mark.parametrize("pair_level, expected_dollars", [(2, 8), (1, 0)])
    def test_matador_only_pays_when_the_arm_actually_triggers(
        self, pair_level: int, expected_dollars: int
    ):
        levels = HandLevels()
        if pair_level == 2:
            levels.level_up("Pair")
        blind = Blind.create("bl_arm", ante=1)
        matador = create_joker("j_matador")
        first = Card()
        first.set_base("H_A", "Hearts", "Ace")
        second = Card()
        second.set_base("S_A", "Spades", "Ace")

        result = score_hand(
            [first, second],
            [],
            [matador],
            levels,
            blind,
            PseudoRandom(f"P42_MATADOR_{pair_level}"),
            game_state={"jokers": [matador], "hand_levels": levels, "blind": blind},
        )

        assert result.debuffed is False
        assert result.dollars_earned == expected_dollars
        assert blind.triggered is (pair_level == 2)

    def test_turtle_bean_decays_hand_size_through_step(self, monkeypatch):
        from jackdaw.engine import effects

        applied = []
        original_apply = effects.apply_effects

        def capture_applied(gs, pending):
            applied.extend(pending)
            original_apply(gs, pending)

        monkeypatch.setattr(effects, "apply_effects", capture_applied)
        gs = initialize_run("b_red", 1, "P42_TURTLE")
        turtle = _owned_joker(gs, "j_turtle_bean")
        before = gs["hand_size"]
        step(gs, SelectBlind())
        gs["blind"].chips = 1

        step(gs, PlayHand(card_indices=(0,)))

        assert turtle in gs["jokers"]
        assert gs["hand_size"] == before - 1
        assert turtle.ability["extra"]["h_size"] == 4
        assert any(
            isinstance(effect, effects.ChangeHandSize) and effect.source is turtle
            for effect in applied
        )
