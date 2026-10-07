"""Tests for Lua-compatible state declarations and pickle fallbacks."""

from __future__ import annotations

import pickle

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import LUA_RUNTIME_ABILITY_KEYS, Card
from jackdaw.engine.run_init import init_game_object

_CARD_STATE_FIELDS = (
    "added_to_deck",
    "getting_sliced",
    "shattered",
    "destroyed",
    "removed",
    "lucky_trigger",
)


def test_init_game_object_declares_lua_state() -> None:
    gs = init_game_object()

    assert gs["dollar_buffer"] == 0
    assert gs["last_hand_played"] is None
    assert gs["consumeable_usage_total"] == {
        "tarot": 0,
        "planet": 0,
        "spectral": 0,
        "tarot_planet": 0,
        "all": 0,
    }
    assert gs["orbital_choices"] == {}
    assert gs["facing_blind"] is False


def test_card_declares_lua_state_and_ordered_unique_value() -> None:
    earlier = Card()
    later = Card()

    for field_name in _CARD_STATE_FIELDS:
        assert getattr(earlier, field_name) is False
    assert 0 < earlier.unique_val <= 1
    assert earlier.unique_val > later.unique_val


def test_old_pickled_card_uses_class_defaults_for_new_fields() -> None:
    card = Card()
    for field_name in _CARD_STATE_FIELDS:
        del card.__dict__[field_name]

    restored = pickle.loads(pickle.dumps(card))

    for field_name in _CARD_STATE_FIELDS:
        assert getattr(restored, field_name) is False


def test_blind_prepped_defaults_false_and_old_pickle_falls_back() -> None:
    blind = Blind.create("bl_small", ante=1)
    assert blind.prepped is False

    del blind.__dict__["prepped"]
    restored = pickle.loads(pickle.dumps(blind))

    assert restored.prepped is False


def test_lua_runtime_ability_keys_are_complete() -> None:
    assert LUA_RUNTIME_ABILITY_KEYS == frozenset(
        {
            "perma_debuff",
            "couponed",
            "wheel_flipped",
            "discarded",
            "queue_negative_removal",
        }
    )


def test_dead_tags_key_removed_and_migrated() -> None:
    from jackdaw.engine.run_init import init_game_object
    from jackdaw.engine.state import migrate_state

    assert "tags" not in init_game_object()
    old = init_game_object()
    old["tags"] = {}
    assert "tags" not in migrate_state(old)
