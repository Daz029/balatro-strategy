"""Regression coverage for Phase 3 blind lifecycle bookkeeping."""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import GamePhase, SelectBlind
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import _apply_boss_blind_effects, _gain_joker, step
from jackdaw.engine.run_init import initialize_run, start_round


def _state(seed: str = "P31_BLIND") -> dict:
    gs = initialize_run("b_red", 1, seed)
    gs["jokers"] = []
    gs["consumables"] = []
    return gs


class TestC08ChicotDisable:
    @pytest.mark.parametrize(
        ("key", "expected"),
        [("bl_wall", "wall"), ("bl_needle", "needle"), ("bl_manacle", "manacle")],
    )
    def test_owned_chicot_undoes_boss_during_step(self, key: str, expected: str) -> None:
        gs = _state(f"STEP_{key}")
        gs["round_resets"]["ante"] = 2
        gs["round_resets"]["blind_ante"] = 2
        gs["round_resets"]["blind_choices"]["Boss"] = key
        gs["blind_on_deck"] = "Boss"
        gs["phase"] = GamePhase.BLIND_SELECT
        _gain_joker(gs, create_joker("j_chicot"))

        step(gs, SelectBlind(allow_forced_boss=True))

        blind = gs["blind"]
        assert blind.disabled is True
        if expected == "wall":
            assert blind.chips == 1600  # ante-2 base 800 * Wall 4 / 2
        elif expected == "needle":
            assert gs["current_round"]["hands_left"] == gs["round_resets"]["hands"]
        else:
            assert gs["hand_size"] == 8
            assert len(gs["hand"]) == 8

    def test_wall_needle_and_manacle_are_undone_in_state(self) -> None:
        for key in ("bl_wall", "bl_needle", "bl_manacle"):
            gs = _state(key)
            gs["round_resets"]["ante"] = 2
            start_round(gs)
            blind = Blind.create(key, ante=2)
            gs["blind"] = blind
            _apply_boss_blind_effects(gs, blind)
            before_hand = len(gs["hand"])
            original_chips = blind.chips
            blind.disable(gs)
            if key == "bl_wall":
                assert blind.chips == original_chips / 2
            elif key == "bl_needle":
                assert gs["current_round"]["hands_left"] == gs["round_resets"]["hands"]
            else:
                assert gs["hand_size"] == 8
                assert len(gs["hand"]) == min(before_hand + 1, gs["hand_size"])

    def test_chicot_acquired_during_boss_disables_it(self) -> None:
        gs = _state()
        gs["round_resets"]["ante"] = 2
        gs["blind"] = Blind.create("bl_wall", ante=2)
        chips = gs["blind"].chips
        _gain_joker(gs, create_joker("j_chicot"))
        assert gs["blind"].disabled is True
        assert gs["blind"].chips == chips / 2

    def test_disable_clears_pillar_but_not_expired_perishable(self) -> None:
        gs = _state()
        card = create_playing_card(Suit.HEARTS, Rank.FIVE)
        card.ability["played_this_ante"] = True
        gs["deck"] = [card]
        blind = Blind.create("bl_pillar", ante=2)
        gs["blind"] = blind
        blind.refresh_debuffs(gs)
        assert card.debuff is True
        blind.disable(gs)
        assert card.debuff is False

        joker = create_joker("j_juggler")
        _gain_joker(gs, joker)
        joker.perishable = True
        joker.perish_tally = 0
        joker.set_debuff(gs, True)
        blind.debuff_card(joker, None, gs)
        assert joker.debuff is True


class TestC08BossAnteGuard:
    def test_forced_low_ante_boss_requires_explicit_override(self) -> None:
        gs = _state()
        gs["phase"] = GamePhase.BLIND_SELECT
        gs["blind_on_deck"] = "Boss"
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_wall"
        with pytest.raises(AssertionError):
            step(gs, SelectBlind())
        step(gs, SelectBlind(allow_forced_boss=True))
        assert gs["blind"].name == "The Wall"

    def test_generated_bosses_obey_min_and_showdown_constraints(self) -> None:
        from jackdaw.engine.data.prototypes import BLINDS

        for ante in range(1, 17):
            for seed in range(12):
                gs = initialize_run("b_red", 1, f"BOSS_GUARD_{ante}_{seed}")
                if ante != 1:
                    from jackdaw.engine.blind import get_new_boss

                    key = get_new_boss(ante, gs["bosses_used"], gs["rng"])
                else:
                    key = gs["round_resets"]["blind_choices"]["Boss"]
                boss = BLINDS[key].boss or {}
                showdown_ante = ante >= 2 and ante % gs["win_ante"] == 0
                assert bool(boss.get("showdown")) == showdown_ante
                if not boss.get("showdown"):
                    assert boss.get("min", 1) <= ante
