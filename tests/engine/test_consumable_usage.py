"""Lua ``set_consumeable_usage`` port (engine overhaul Phase 2, P2-4).

``Card:use_consumeable`` (card.lua:1091-1093) records usage before anything
else; ``set_consumeable_usage`` (misc_functions.lua:1184-1228) keeps a per-key
``{count, order, set}`` table and per-set totals. Black Hole is a Spectral, so
it counts as spectral and never becomes ``last_tarot_planet``. Satellite pays
per DISTINCT Planet key used (card.lua:1667-1674). All driven through ``step``.
"""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase, UseConsumable
from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_consumable, create_joker
from jackdaw.engine.game import step
from jackdaw.engine.jokers import on_end_of_round
from jackdaw.engine.read import StateView
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand


def _shop_state(seed: str, *keys: str) -> dict:
    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.SHOP
    gs["consumable_slots"] = max(gs.get("consumable_slots", 2), len(keys))
    gs["consumables"] = [create_consumable(k) for k in keys]
    return gs


def _use_first(gs: dict) -> None:
    step(gs, UseConsumable(card_index=0))


class TestD40TarotTally:
    def test_tarot_use_records_totals_and_per_key_entry(self):
        gs = _shop_state("USAGE_TAROT", "c_hermit")

        _use_first(gs)

        totals = gs["consumeable_usage_total"]
        assert (totals["tarot"], totals["tarot_planet"], totals["all"]) == (1, 1, 1)
        assert (totals["planet"], totals["spectral"]) == (0, 0)
        assert gs["consumeable_usage"]["c_hermit"] == {"count": 1, "order": 10, "set": "Tarot"}

    def test_repeat_use_increments_count(self):
        gs = _shop_state("USAGE_REPEAT", "c_hermit", "c_hermit")

        _use_first(gs)
        _use_first(gs)

        assert gs["consumeable_usage"]["c_hermit"]["count"] == 2
        assert gs["consumeable_usage_total"]["tarot"] == 2

    def test_fortune_teller_scores_the_recorded_tarots(self):
        gs = _shop_state("USAGE_FORTUNE", "c_hermit")
        _use_first(gs)
        card = Card()
        card.set_base("H_2", "Hearts", "2")

        result = score_hand(
            [card],
            [],
            [create_joker("j_fortune_teller")],
            gs["hand_levels"],
            Blind.create("bl_small", ante=1),
            gs["rng"],
            game_state=gs,
        )

        assert result.mult == 1 + 1  # High Card base mult + one Tarot


class TestD40BlackHoleIsSpectral:
    def test_black_hole_counts_as_spectral_and_keeps_last_tarot_planet(self):
        gs = _shop_state("USAGE_BLACK_HOLE", "c_black_hole")
        gs["last_tarot_planet"] = "c_hermit"

        _use_first(gs)

        totals = gs["consumeable_usage_total"]
        assert (totals["spectral"], totals["all"]) == (1, 1)
        assert (totals["planet"], totals["tarot_planet"]) == (0, 0)
        assert gs["consumeable_usage"]["c_black_hole"]["set"] == "Spectral"
        assert gs["last_tarot_planet"] == "c_hermit"


class TestD19Satellite:
    def test_satellite_pays_per_distinct_planet(self):
        gs = _shop_state("USAGE_SATELLITE", "c_mercury", "c_mercury", "c_venus")
        for _ in range(3):
            _use_first(gs)
        satellite = create_joker("j_satellite")

        eor = on_end_of_round([satellite], StateView(gs, jokers=[satellite]), gs["rng"])

        assert StateView(gs).planets_used == 2
        assert eor["dollars_earned"] == satellite.ability["extra"] * 2

    def test_black_hole_does_not_count_for_satellite(self):
        gs = _shop_state("USAGE_SATELLITE_BH", "c_black_hole")
        _use_first(gs)

        assert StateView(gs).planets_used == 0


class TestFoolRecordsAfterItsEffect:
    def test_fool_recreates_previous_tarot_then_becomes_last(self):
        gs = _shop_state("USAGE_FOOL", "c_hermit", "c_fool")

        _use_first(gs)  # The Hermit
        _use_first(gs)  # The Fool reads c_hermit, then becomes last_tarot_planet

        assert [c.center_key for c in gs["consumables"]] == ["c_hermit"]
        assert gs["last_tarot_planet"] == "c_fool"
