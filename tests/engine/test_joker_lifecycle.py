"""Joker ownership transitions apply acquisition passives exactly once."""

from __future__ import annotations

import copy
from typing import Any

from jackdaw.engine.actions import BuyCard, GamePhase, PickPackCard, SellCard
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import _gain_joker, _lose_joker, _resolve_create_descriptors, step
from jackdaw.engine.run_init import initialize_run


def _state(phase: GamePhase) -> dict[str, Any]:
    gs = initialize_run("b_red", 1, "JOKER_LIFECYCLE")
    gs.update(
        {
            "phase": phase,
            "hand_size": 8,
            "jokers": [],
            "used_jokers": {},
            "dollars": 100,
        }
    )
    return gs


def test_buying_juggler_applies_hand_size_passive() -> None:
    gs = _state(GamePhase.SHOP)
    gs["shop_cards"] = [create_joker("j_juggler")]

    step(gs, BuyCard(0))

    assert gs["hand_size"] == 9


def test_picking_juggler_from_buffoon_pack_applies_hand_size_passive() -> None:
    gs = _state(GamePhase.PACK_OPENING)
    gs["pack_cards"] = [create_joker("j_juggler"), create_joker("j_joker")]
    gs["pack_choices_remaining"] = 2

    step(gs, PickPackCard(0))

    assert gs["hand_size"] == 9
    assert gs["used_jokers"] == {"j_juggler": True}


def test_selling_juggler_reverses_hand_size_passive() -> None:
    gs = _state(GamePhase.SHOP)
    juggler = create_joker("j_juggler")
    juggler.add_to_deck(gs)
    gs["jokers"] = [juggler]

    step(gs, SellCard("jokers", 0))

    assert gs["hand_size"] == 8


def test_distinct_duplicate_jokers_each_apply_their_passive() -> None:
    """Showman duplicates share a key but remain distinct owned cards."""
    gs = _state(GamePhase.SHOP)
    gs["shop_cards"] = [create_joker("j_juggler"), create_joker("j_juggler")]

    step(gs, BuyCard(0))
    step(gs, BuyCard(0))

    assert gs["hand_size"] == 10
    assert gs["used_jokers"] == {"j_juggler": True}


def test_selling_duplicate_removes_only_that_cards_passive() -> None:
    gs = _state(GamePhase.SHOP)
    first = create_joker("j_juggler")
    second = create_joker("j_juggler")
    for joker in (first, second):
        joker.add_to_deck(gs)
    gs["jokers"] = [first, second]
    gs["used_jokers"] = {"j_juggler": True}

    step(gs, SellCard("jokers", 1))

    assert gs["jokers"] == [first]
    assert gs["hand_size"] == 9
    # Run-wide duplicate exclusion records seen keys, not current ownership.
    assert gs["used_jokers"] == {"j_juggler": True}


def test_same_card_object_cannot_apply_its_passive_twice() -> None:
    """Restore/re-observation seams may encounter an already-owned object."""
    gs = _state(GamePhase.SHOP)
    juggler = create_joker("j_juggler")

    assert _gain_joker(gs, juggler) is True
    assert _gain_joker(gs, juggler) is False

    assert gs["jokers"] == [juggler]
    assert gs["hand_size"] == 9


def test_removal_uses_identity_when_duplicate_jokers_compare_equal() -> None:
    gs = _state(GamePhase.SHOP)
    first = create_joker("j_juggler")
    # Ankh creates its duplicate with deepcopy, preserving dataclass equality.
    second = copy.deepcopy(first)
    _gain_joker(gs, first)
    _gain_joker(gs, second)

    assert first == second
    assert _lose_joker(gs, second) is True

    assert len(gs["jokers"]) == 1
    assert gs["jokers"][0] is first
    assert gs["hand_size"] == 9


def test_consumable_created_juggler_applies_passive_and_registers_key() -> None:
    gs = _state(GamePhase.SHOP)

    _resolve_create_descriptors(
        gs,
        [{"type": "Joker", "forced_key": "j_juggler"}],
    )

    assert [joker.center_key for joker in gs["jokers"]] == ["j_juggler"]
    assert gs["hand_size"] == 9
    assert gs["used_jokers"] == {"j_juggler": True}


def test_negative_edition_slot_passive_is_symmetric() -> None:
    gs = _state(GamePhase.SHOP)
    starting_slots = gs["joker_slots"]
    gs["shop_cards"] = [create_joker("j_joker", edition={"negative": True})]

    step(gs, BuyCard(0))
    assert gs["joker_slots"] == starting_slots + 1

    step(gs, SellCard("jokers", 0))
    assert gs["joker_slots"] == starting_slots
