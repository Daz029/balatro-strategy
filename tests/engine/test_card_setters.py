"""Regression coverage for Phase 3 card-owned bookkeeping."""

from __future__ import annotations

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_consumable, create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.deck_builder import build_deck
from jackdaw.engine.game import _gain_joker, _use_consumable_card
from jackdaw.engine.read import area_of
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.state import migrate_state


def _state(seed: str = "P31") -> dict:
    gs = initialize_run("b_red", 1, seed)
    gs["jokers"] = []
    gs["consumables"] = []
    gs["shop_cards"] = []
    gs["shop_boosters"] = []
    return gs


class TestC12EditionSetter:
    def test_negative_owned_joker_adds_slot_but_shop_joker_waits_for_purchase(self) -> None:
        gs = _state()
        owned = create_joker("j_joker")
        _gain_joker(gs, owned)
        before = gs["joker_slots"]
        owned.set_edition(gs, {"negative": True})
        assert gs["joker_slots"] == before + 1

        shop = create_joker("j_joker")
        gs["shop_cards"].append(shop)
        before = gs["joker_slots"]
        shop.set_edition(gs, {"negative": True})
        assert gs["joker_slots"] == before
        _gain_joker(gs, shop)
        assert gs["joker_slots"] == before + 1

    def test_replacing_negative_does_not_restore_slot_in_setter(self) -> None:
        """Lua card.lua:387-421 nils edition first; removal is card.lua:689."""
        gs = _state()
        joker = create_joker("j_joker")
        _gain_joker(gs, joker)
        joker.set_edition(gs, {"negative": True})
        slots = gs["joker_slots"]
        joker.set_edition(gs, {"foil": True})
        assert gs["joker_slots"] == slots

    def test_polychrome_value_comes_from_center(self) -> None:
        gs = _state()
        card = create_joker("j_joker")
        card.set_edition(gs, {"polychrome": True})
        assert card.get_edition()["x_mult_mod"] == 1.5


class TestD53CostSetter:
    def test_couponed_card_remains_free_only_in_shop_area(self) -> None:
        gs = _state()
        card = create_joker("j_joker")
        card.ability["couponed"] = True
        gs["shop_cards"] = [card]
        assert area_of(gs, card) == "shop_cards"
        card.set_cost(gs)
        assert card.cost == 0
        gs["shop_cards"].clear()
        card.set_cost(gs)
        assert card.cost > 0


class TestC13AstronomerCost:
    def test_only_active_astronomer_makes_planet_free(self) -> None:
        gs = _state()
        astronomer = create_joker("j_astronomer")
        gs["jokers"] = [astronomer]
        planet = create_consumable("c_mercury")
        gs["shop_cards"] = [planet]
        planet.set_cost(gs)
        assert planet.cost == 0
        astronomer.set_debuff(gs, True)
        planet.set_cost(gs)
        assert planet.cost > 0


class TestD47StickerSetters:
    def test_compatibility_and_mutual_exclusion(self) -> None:
        gs = _state()
        incompatible = create_joker("j_cavendish")
        incompatible.set_eternal(True)
        assert incompatible.eternal is False

        joker = create_joker("j_joker")
        joker.set_eternal(True)
        joker.set_perishable(gs, True)
        assert joker.eternal is True
        assert joker.perishable is False
        joker.set_eternal(False)
        joker.set_perishable(gs, True)
        assert joker.perishable is True
        joker.set_eternal(True)
        assert joker.eternal is False
        assert joker.perish_tally == gs["perishable_rounds"]


class TestD25DebuffSetter:
    def test_debuff_toggles_juggler_and_oops_passives(self) -> None:
        gs = _state()
        juggler = create_joker("j_juggler")
        oops = create_joker("j_oops")
        _gain_joker(gs, juggler)
        _gain_joker(gs, oops)
        assert (gs["hand_size"], gs["probabilities"]["normal"]) == (9, 2)
        juggler.set_debuff(gs, True)
        oops.set_debuff(gs, True)
        assert (gs["hand_size"], gs["probabilities"]["normal"]) == (8, 1)
        juggler.set_debuff(gs, False)
        assert gs["hand_size"] == 9

    def test_expired_perishable_cannot_be_cleared(self) -> None:
        gs = _state()
        joker = create_joker("j_juggler")
        _gain_joker(gs, joker)
        joker.perishable = True
        joker.perish_tally = 0
        joker.set_debuff(gs, False)
        assert joker.debuff is True
        assert gs["hand_size"] == 8


class TestD41BaseSetter:
    def test_suit_change_updates_key_and_preserves_original_nominal(self) -> None:
        gs = _state()
        card = create_playing_card(Suit.CLUBS, Rank.ACE)
        original = card.base.suit_nominal_original
        card.change_suit("Diamonds", gs=gs)
        assert card.card_key == "D_A"
        assert card.base.suit_nominal_original == original

    def test_base_change_resets_times_played(self) -> None:
        # card.lua:111-120 rebuilds self.base with times_played = 0.
        gs = _state()
        card = create_playing_card(Suit.CLUBS, Rank.ACE)
        card.base.times_played = 3
        card.change_rank("King", gs=gs)
        assert card.base.times_played == 0

    def test_nil_edition_returns_before_set_cost(self) -> None:
        # card.lua:388-389: set_edition(nil) clears and returns before set_cost.
        gs = _state()
        joker = create_joker("j_joker")
        joker.cost = 99
        joker.set_edition(gs, None)
        assert joker.edition is None
        assert joker.cost == 99

    def test_checkered_deck_keys_match_converted_suits(self) -> None:
        cards = build_deck("b_checkered", PseudoRandom("CHECKERED_KEYS"))
        prefix = {Suit.SPADES: "S_", Suit.HEARTS: "H_"}
        assert all(card.card_key.startswith(prefix[card.base.suit]) for card in cards)


class TestD36AbilityAndBaseRefresh:
    def test_suit_and_enhancement_changes_refresh_blind_debuff(self) -> None:
        gs = _state()
        card = create_playing_card(Suit.CLUBS, Rank.FIVE)
        gs["hand"] = [card]
        gs["blind"] = Blind.create("bl_club", ante=1)
        gs["blind"].refresh_debuffs(gs)
        assert card.debuff is True
        card.change_suit("Diamonds", gs=gs)
        assert card.debuff is False
        card.change_suit("Clubs", gs=gs)
        card.enhance("m_stone", gs=gs)
        assert card.debuff is False

    def test_star_use_and_applier_clear_club_debuff_immediately(self) -> None:
        gs = _state("STAR_REFRESH")
        card = create_playing_card(Suit.CLUBS, Rank.FIVE)
        gs["hand"] = [card]
        gs["blind"] = Blind.create("bl_club", ante=1)
        gs["blind"].refresh_debuffs(gs)
        assert card.debuff is True

        _use_consumable_card(gs, create_consumable("c_star"), (0,))

        assert card.base.suit is Suit.DIAMONDS
        assert card.debuff is False

    def test_wild_rechecks_as_every_suit(self) -> None:
        gs = _state("WILD_REFRESH")
        card = create_playing_card(Suit.DIAMONDS, Rank.FIVE)
        gs["hand"] = [card]
        gs["blind"] = Blind.create("bl_club", ante=1)
        gs["blind"].refresh_debuffs(gs)
        assert card.debuff is False
        card.enhance("m_wild", gs=gs)
        assert card.debuff is True


class TestStickerMigration:
    def test_legacy_ability_stickers_move_to_fields_and_owned_flags_are_set(self) -> None:
        gs = _state()
        joker = create_joker("j_joker")
        joker.ability.update(rental=True, perishable=True, perish_tally=2)
        gs["jokers"] = [joker]
        migrate_state(gs)
        assert (joker.rental, joker.perishable, joker.perish_tally) == (True, True, 2)
        assert not {"rental", "perishable", "perish_tally"} & joker.ability.keys()
        assert joker.added_to_deck is True
