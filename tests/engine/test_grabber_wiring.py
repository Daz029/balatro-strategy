"""Regression coverage for Phase 2 live-state getter wiring."""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine import pools
from jackdaw.engine.actions import Discard, PlayHand, SelectBlind
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import (
    create_card,
    create_consumable,
    create_joker,
    create_playing_card,
)
from jackdaw.engine.consumables import ConsumableContext, use_consumable
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.packs import generate_pack_cards
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


def _high_card():
    return create_playing_card(Suit.HEARTS, Rank.SEVEN)


class _FirstRng:
    """Minimal deterministic RNG for consumable target selection."""

    def __init__(self, random_values: list[float] | None = None) -> None:
        self._values = iter(random_values or [])

    def random(self, _key: str) -> float:
        return next(self._values)

    def seed(self, _key: str) -> float:
        return 0.5

    def element(self, table: list, _seed: float) -> tuple[Any, int]:
        return table[0], 0


class TestD19DelayedGratification:
    def test_two_used_discards_prevent_end_of_round_payout(self):
        gs = initialize_run("b_red", 1, "D19_DELAYED")
        gs["jokers"].append(create_joker("j_delayed_grat"))
        step(gs, SelectBlind())
        step(gs, Discard(card_indices=(0,)))
        step(gs, Discard(card_indices=(0,)))
        assert gs["current_round"]["discards_used"] == 2
        assert gs["current_round"]["discards_left"] > 0

        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0,)))

        assert gs["round_earnings"].joker_dollars == 0


class TestStaleMirrorReads:
    def test_score_hand_sees_live_money_and_stone_count_after_prior_play(self):
        gs = initialize_run("b_red", 1, "STALE_MIRRORS")
        step(gs, SelectBlind())
        gs["blind"].chips = 10**9
        step(gs, PlayHand(card_indices=(0,)))

        gs["dollars"] = 10
        stone = create_playing_card(Suit.CLUBS, Rank.TWO, enhancement="m_stone")
        gs["deck"].append(stone)
        played = [_high_card()]

        baseline = score_hand(
            played,
            [],
            [],
            gs["hand_levels"],
            gs["blind"],
            PseudoRandom("STALE_BASE"),
            game_state=gs,
        )
        result = score_hand(
            played,
            [],
            [create_joker("j_stone"), create_joker("j_bull")],
            gs["hand_levels"],
            gs["blind"],
            PseudoRandom("STALE_LIVE"),
            game_state=gs,
        )

        assert result.chips - baseline.chips == 25 + 2 * 10


class TestC13VoucherAndAstronomerFlags:
    def test_omen_globe_can_put_a_spectral_in_an_arcana_pack(self):
        found = False
        for i in range(100):
            gs = initialize_run("b_red", 1, f"C13_OMEN_{i}")
            gs["used_vouchers"]["v_omen_globe"] = True
            cards, _ = generate_pack_cards(
                "p_arcana_normal_1", PseudoRandom(f"C13_OMEN_{i}"), 1, gs
            )
            if any(card.ability.get("set") == "Spectral" for card in cards):
                found = True
                break
        assert found

    def test_telescope_slot_zero_matches_most_played_visible_hand(self):
        gs = initialize_run("b_red", 1, "C13_TELESCOPE")
        gs["used_vouchers"]["v_telescope"] = True
        gs["hand_levels"].record_play("Flush")
        gs["hand_levels"].record_play("Flush")
        gs["hand_levels"].record_play("Pair")

        cards, _ = generate_pack_cards("p_celestial_normal_1", PseudoRandom("C13_TELESCOPE"), 1, gs)

        assert cards[0].center_key == "c_jupiter"

    def test_owned_astronomer_makes_a_shop_planet_free(self):
        gs = {
            "jokers": [create_joker("j_astronomer")],
            "joker_rate": 0,
            "tarot_rate": 0,
            "planet_rate": 1,
            "spectral_rate": 0,
            "playing_card_rate": 0,
            "used_jokers": {},
            "used_vouchers": {},
            "pool_flags": {},
            "modifiers": {},
        }

        from jackdaw.engine.shop import create_shop_slot_card

        planet = create_shop_slot_card(PseudoRandom("ASTRO"), 1, gs)
        assert planet.ability["set"] == "Planet"
        assert planet.cost == 0


class TestD42DeckEnhancementGate:
    def test_steel_joker_pool_gate_uses_live_playing_cards(self, monkeypatch):
        eligible: list[bool] = []

        def capture_pool(pool_type, rng, ante, **kwargs):
            pool, _ = pools.get_current_pool(pool_type, rng, ante, **kwargs)
            steel_index = list(pools.JOKER_RARITY_POOLS[2]).index("j_steel_joker")
            eligible.append(pool[steel_index] != pools.UNAVAILABLE)
            return "j_joker"

        monkeypatch.setattr(pools, "pick_card_from_pool", capture_pool)
        normal = create_playing_card(Suit.HEARTS, Rank.TWO)
        steel = create_playing_card(Suit.HEARTS, Rank.TWO, enhancement="m_steel")

        create_card(
            "Joker",
            PseudoRandom("D42_NO_STEEL"),
            1,
            forced_rarity=2,
            game_state={"deck": [normal]},
        )
        create_card(
            "Joker",
            PseudoRandom("D42_STEEL"),
            1,
            forced_rarity=2,
            game_state={"deck": [steel]},
        )

        assert eligible == [False, True]


class TestD10JokerStencil:
    def test_debuffed_stencil_counts_as_a_stencil_and_an_occupied_slot(self):
        active = create_joker("j_stencil")
        debuffed = create_joker("j_stencil")
        debuffed.debuff = True

        result = score_hand(
            [_high_card()],
            [],
            [active, debuffed],
            HandLevels(),
            _small_blind(),
            PseudoRandom("D10_STENCIL"),
            game_state={"joker_slots": 5},
        )

        assert result.mult == 5

    def test_full_row_gives_nothing_even_with_two_stencils(self):
        # card.lua:3967 fires only while card_limit - #jokers > 0; the x_mult
        # (= 2 here, one per Stencil) is never applied with a full row.
        stencils = [create_joker("j_stencil"), create_joker("j_stencil")]

        result = score_hand(
            [_high_card()],
            [],
            stencils,
            HandLevels(),
            _small_blind(),
            PseudoRandom("D10_STENCIL_FULL"),
            game_state={"joker_slots": 2},
        )

        assert result.mult == 1


class TestD10Swashbuckler:
    def test_debuffed_joker_sell_value_is_included(self):
        swashbuckler = create_joker("j_swashbuckler")
        debuffed = create_joker("j_joker")
        debuffed.sell_cost = 7
        debuffed.debuff = True

        result = score_hand(
            [_high_card()],
            [],
            [swashbuckler, debuffed],
            HandLevels(),
            _small_blind(),
            PseudoRandom("D10_SWASH"),
        )

        assert result.mult == 1 + 7


class TestD19Cloud9:
    def test_end_of_round_pays_one_dollar_per_nine_in_full_deck(self):
        gs = initialize_run("b_red", 1, "D19_CLOUD9")
        gs["jokers"].append(create_joker("j_cloud_9"))
        step(gs, SelectBlind())
        gs["blind"].chips = 1

        step(gs, PlayHand(card_indices=(0,)))

        assert gs["round_earnings"].joker_dollars == 4


class TestD39EditionlessCandidates:
    @pytest.mark.parametrize(
        ("consumable_key", "random_values", "expected_edition"),
        [
            ("c_ectoplasm", [], {"negative": True}),
            ("c_hex", [], {"polychrome": True}),
            ("c_wheel_of_fortune", [0.0, 0.6], {"holo": True}),
        ],
    )
    def test_debuffed_editionless_joker_can_be_selected(
        self, consumable_key, random_values, expected_edition
    ):
        consumable = create_consumable(consumable_key)
        joker = create_joker("j_joker")
        joker.debuff = True

        result = use_consumable(
            consumable,
            ConsumableContext(
                card=consumable,
                jokers=[joker],
                rng=_FirstRng(random_values),
                game_state={"probabilities": {"normal": 1}},
            ),
        )

        assert result is not None
        assert result.add_edition == {"target": joker, "edition": expected_edition}
