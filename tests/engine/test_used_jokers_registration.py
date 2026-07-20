"""Run-wide duplicate exclusion: registration, pack cleanup, and Showman.

Covers the ``used_jokers`` half of the engine PR-2 fixes.

Background
----------
``pools.py`` has always consulted ``used_jokers`` correctly, but nothing
*populated* it for cards that were merely displayed — only buy/pick sites did.
A joker you saw in a shop and declined therefore stayed fully eligible forever.
``create_card`` now registers every created key at creation, mirroring
``Card:set_ability`` (card.lua:349-354).

That change collides with the pack layer, which temporarily registers keys and
deletes them again so unpicked pack cards don't poison the pool
(card.lua:4741-4748).  The tests below pin the resulting add/delete accounting,
including the Showman case where duplicates within a single pack are legal.
"""

from __future__ import annotations

from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import _has_showman, create_card
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


class TestPackCleanup:
    """Unpicked pack cards must not permanently register."""

    def test_pack_leaves_used_jokers_unchanged(self):
        gs: dict = {"used_jokers": {}}
        cards, _ = generate_pack_cards("p_arcana_normal_1", PseudoRandom("PACK1"), 1, gs)
        assert cards, "pack generated no cards — fixture assumption broken"
        # Regression: with post-hoc membership testing, create_card's own
        # registration made the cleanup list always empty and every displayed
        # pack card leaked into the pool permanently.
        assert gs["used_jokers"] == {}

    def test_pre_existing_registration_survives_a_pack(self):
        """A key registered before the pack must never be erased by cleanup."""
        gs: dict = {"used_jokers": {"c_fool": True}}
        generate_pack_cards("p_arcana_normal_1", PseudoRandom("PACK2"), 1, gs)
        assert gs["used_jokers"].get("c_fool") is True

    def test_showman_duplicates_in_pack_do_not_double_delete(self):
        """Showman legalises duplicates within one pack.

        Two cards sharing a key means one registration, so cleanup must delete
        exactly once.  Deleting per generated card instead would raise KeyError.
        """
        gs: dict = {"used_jokers": {}, "jokers": [_joker("j_ring_master")]}
        # Must not raise, and must leave the pool clean.
        generate_pack_cards("p_arcana_normal_1", PseudoRandom("PACK3"), 1, gs)
        assert gs["used_jokers"] == {}


class TestShowmanFlag:
    def test_absent_without_the_joker(self):
        assert _has_showman({"jokers": [_joker("j_joker")]}) is False

    def test_derived_from_owned_jokers(self):
        """Regression: nothing in the engine ever wrote gs['has_showman'], so
        it was permanently False and Showman did nothing at all."""
        assert _has_showman({"jokers": [_joker("j_ring_master")]}) is True

    def test_debuffed_showman_still_counts(self):
        """Vanilla's find_joker does not filter debuffed jokers."""
        j = _joker("j_ring_master")
        j.debuff = True
        assert _has_showman({"jokers": [j]}) is True

    def test_explicit_flag_wins(self):
        """Callers and tests can still force the flag either way."""
        assert _has_showman({"has_showman": True, "jokers": []}) is True
        assert _has_showman({"has_showman": False, "jokers": [_joker("j_ring_master")]}) is False

    def test_empty_state_is_false(self):
        assert _has_showman({}) is False

    def test_showman_permits_a_repeat_roll(self):
        """With Showman owned, a registered key stays eligible."""
        gs: dict = {"jokers": [_joker("j_ring_master")]}
        first = create_card("Tarot", PseudoRandom("SHOW1"), 1, area="shop", game_state=gs)
        second = create_card("Tarot", PseudoRandom("SHOW1"), 1, area="shop", game_state=gs)
        # Duplicate exclusion is bypassed, so the identical stream repeats.
        assert first.center_key == second.center_key
