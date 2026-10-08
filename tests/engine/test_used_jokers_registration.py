"""Current-card duplicate exclusion: registration, pack lifetime, and Showman.

Covers the ``used_jokers`` half of the engine PR-2 fixes.

Background
----------
``pools.py`` has always consulted ``used_jokers`` correctly, but nothing
*populated* it for cards that were merely displayed — only buy/pick sites did.
A joker you saw in a shop and declined therefore stayed fully eligible forever.
``create_card`` now registers every created key at creation, mirroring
``Card:set_ability`` (card.lua:349-354).

Pack contents remain registered while displayed and are released by the
canonical lifecycle only when picked/used or when the pack closes
(card.lua:4741-4748).
"""

from __future__ import annotations

from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import _has_showman, create_card
from jackdaw.engine.lifecycle import remove
from jackdaw.engine.packs import generate_pack_cards
from jackdaw.engine.rng import PseudoRandom


def _joker(key: str) -> Card:
    c = Card()
    c.set_ability(key)
    return c


class TestCreateCardRegisters:
    def test_registers_created_key(self):
        """Every created key lands in used_jokers — not just bought ones."""
        gs: dict = {}
        card = create_card("Tarot", PseudoRandom("REG1"), 1, area="shop", game_state=gs)
        assert gs["used_jokers"].get(card.center_key) is True

    def test_no_game_state_is_tolerated(self):
        """create_card must still work when no run state is threaded through."""
        card = create_card("Tarot", PseudoRandom("REG2"), 1, area="shop")
        assert card.center_key

    def test_registration_excludes_the_key_from_a_later_roll(self):
        """The whole point: a registered key cannot be rolled again."""
        gs: dict = {}
        first = create_card("Tarot", PseudoRandom("REG3"), 1, area="shop", game_state=gs)
        # Same seed and stream would otherwise reproduce the same key.
        second = create_card("Tarot", PseudoRandom("REG3"), 1, area="shop", game_state=gs)
        assert first.center_key != second.center_key


class TestPackLifetime:
    """Displayed cards stay registered exactly until they stop existing."""

    def test_pack_cards_stay_registered_until_removed(self):
        gs: dict = {"used_jokers": {}}
        cards, _ = generate_pack_cards("p_arcana_normal_1", PseudoRandom("PACK1"), 1, gs)
        assert cards, "pack generated no cards — fixture assumption broken"
        assert {card.center_key for card in cards} <= set(gs["used_jokers"])
        for card in cards:
            remove(gs, card)
        assert gs["used_jokers"] == {}

    def test_pre_existing_registration_survives_a_pack(self):
        """An owned copy must survive removal of unrelated pack cards."""
        gs: dict = {"used_jokers": {}, "consumables": []}
        owned = create_card(
            "Tarot",
            PseudoRandom("PACK2_OWNED"),
            1,
            forced_key="c_fool",
            game_state=gs,
        )
        gs["consumables"].append(owned)
        cards, _ = generate_pack_cards("p_arcana_normal_1", PseudoRandom("PACK2"), 1, gs)
        for card in cards:
            remove(gs, card)
        assert gs["used_jokers"].get("c_fool") is True

    def test_showman_duplicates_in_pack_do_not_double_delete(self):
        """Showman legalises duplicates within one pack.

        Two cards sharing a key means one registration, so cleanup must delete
        exactly once.  Deleting per generated card instead would raise KeyError.
        """
        gs: dict = {"used_jokers": {}, "jokers": [_joker("j_ring_master")]}
        cards, _ = generate_pack_cards("p_arcana_normal_1", PseudoRandom("PACK3"), 1, gs)
        for card in cards:
            remove(gs, card)
        assert gs["used_jokers"] == {}


class TestShowmanFlag:
    def test_absent_without_the_joker(self):
        assert _has_showman({"jokers": [_joker("j_joker")]}) is False

    def test_derived_from_owned_jokers(self):
        """Regression: nothing in the engine ever wrote gs['has_showman'], so
        it was permanently False and Showman did nothing at all."""
        assert _has_showman({"jokers": [_joker("j_ring_master")]}) is True

    def test_debuffed_showman_is_inactive_in_rules_view(self):
        j = _joker("j_ring_master")
        j.debuff = True
        assert _has_showman({"jokers": [j]}) is False

    def test_owned_showman_is_the_only_source(self):
        assert _has_showman({"jokers": [_joker("j_ring_master")]}) is True

    def test_empty_state_is_false(self):
        assert _has_showman({}) is False

    def test_showman_permits_a_repeat_roll(self):
        """With Showman owned, a registered key stays eligible."""
        gs: dict = {"jokers": [_joker("j_ring_master")]}
        first = create_card("Tarot", PseudoRandom("SHOW1"), 1, area="shop", game_state=gs)
        second = create_card("Tarot", PseudoRandom("SHOW1"), 1, area="shop", game_state=gs)
        # Duplicate exclusion is bypassed, so the identical stream repeats.
        assert first.center_key == second.center_key
