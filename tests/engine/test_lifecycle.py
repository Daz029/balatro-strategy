"""Regression tests for the Phase 3 card lifecycle and pool tracker."""

from __future__ import annotations

import random

from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    OpenBooster,
    Reroll,
    SelectBlind,
    SellCard,
    SkipPack,
    UseConsumable,
)
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_card, create_joker
from jackdaw.engine.data.prototypes import JOKERS
from jackdaw.engine.game import step
from jackdaw.engine.lifecycle import emplace
from jackdaw.engine.pools import get_current_pool
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.state import migrate_state


def _shop_state(seed: str) -> dict:
    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.SHOP
    gs["dollars"] = 999
    return gs


def _owned_joker(gs: dict, key: str) -> Card:
    card = create_card(
        "Joker",
        gs["rng"],
        gs["round_resets"]["ante"],
        area="",
        soulable=False,
        forced_key=key,
        game_state=gs,
    )
    emplace(gs, card, "jokers")
    return card


def _joker_is_available(gs: dict, key: str) -> bool:
    pool, _ = get_current_pool(
        "Joker",
        gs["rng"],
        gs["round_resets"]["ante"],
        rarity=JOKERS[key].rarity,
        used_jokers=gs["used_jokers"],
        banned_keys=gs.get("banned_keys"),
        pool_flags=gs.get("pool_flags"),
    )
    return key in pool


class TestC06CurrentlyExists:
    def test_selling_joker_releases_its_pool_key(self):
        gs = _shop_state("C06_SELL")
        joker = _owned_joker(gs, "j_joker")

        step(gs, SellCard(area="jokers", card_index=0))

        assert joker.removed is True
        assert "j_joker" not in gs["used_jokers"]

    def test_reroll_releases_displayed_joker_before_repopulation(self):
        gs = _shop_state("C06_REROLL")
        shown = create_card(
            "Joker",
            gs["rng"],
            1,
            area="shop",
            forced_key="j_joker",
            game_state=gs,
        )
        gs["shop_cards"] = [shown]
        gs["shop"]["joker_max"] = 1
        gs["current_round"]["reroll_cost"] = 0

        step(gs, Reroll())

        assert shown.removed is True
        # The old display must no longer be the reason this key is excluded.
        if all(card.center_key != "j_joker" for card in gs["shop_cards"]):
            assert _joker_is_available(gs, "j_joker")
        else:
            # It was released early enough to be rolled straight back in.
            assert any(card.center_key == "j_joker" for card in gs["shop_cards"])

    def test_buffoon_display_is_excluded_until_pack_closes(self):
        gs = _shop_state("C06_BUFFOON")
        booster = create_card(
            "Booster",
            gs["rng"],
            1,
            area="shop",
            forced_key="p_buffoon_normal_1",
            game_state=gs,
        )
        gs["shop_boosters"] = [booster]

        step(gs, OpenBooster(card_index=0))
        displayed_cards = list(gs["pack_cards"])
        displayed_keys = {card.center_key for card in displayed_cards}
        assert booster.removed is True
        assert displayed_keys
        assert displayed_keys <= set(gs["used_jokers"])
        assert not any(_joker_is_available(gs, key) for key in displayed_keys)

        step(gs, SkipPack())
        assert all(card.removed for card in displayed_cards)
        assert displayed_keys.isdisjoint(gs["used_jokers"])
        assert any(_joker_is_available(gs, key) for key in displayed_keys)

    def test_leaving_shop_removes_all_unsold_displays(self):
        gs = _shop_state("C06_LEAVE_SHOP")
        cards = [
            create_card(
                card_type,
                gs["rng"],
                1,
                area="shop",
                soulable=False,
                forced_key=key,
                game_state=gs,
            )
            for card_type, key in (
                ("Joker", "j_joker"),
                ("Voucher", "v_overstock_norm"),
                ("Booster", "p_arcana_normal_1"),
            )
        ]
        gs["shop_cards"] = [cards[0]]
        gs["shop_vouchers"] = [cards[1]]
        gs["shop_boosters"] = [cards[2]]

        step(gs, NextRound())

        assert all(card.removed for card in cards)
        assert all(not gs[area] for area in ("shop_cards", "shop_vouchers", "shop_boosters"))

    def test_used_tarot_is_released(self):
        gs = _shop_state("C06_TAROT")
        tarot = create_card(
            "Tarot",
            gs["rng"],
            1,
            area="",
            soulable=False,
            forced_key="c_judgement",
            game_state=gs,
        )
        emplace(gs, tarot, "consumables")

        step(gs, UseConsumable(card_index=0))

        assert tarot.removed is True
        assert "c_judgement" not in gs["used_jokers"]

    def test_debuffed_owned_copy_keeps_key_excluded(self):
        gs = _shop_state("C06_DEBUFF")
        first = _owned_joker(gs, "j_joker")
        second = _owned_joker(gs, "j_joker")
        second.set_debuff(gs, True)

        step(gs, SellCard(area="jokers", card_index=0))

        assert first.removed is True
        assert gs["jokers"] == [second]
        assert gs["used_jokers"]["j_joker"] is True


class TestD42CreatedJokerEditions:
    def test_judgement_rolls_edition_and_advances_edition_stream(self):
        found_edition = False
        found_stream = False
        for index in range(300):
            gs = _shop_state(f"D42_{index}")
            judgement = create_card(
                "Tarot",
                gs["rng"],
                1,
                area="",
                soulable=False,
                forced_key="c_judgement",
                game_state=gs,
            )
            emplace(gs, judgement, "consumables")

            step(gs, UseConsumable(card_index=0))

            found_stream |= "edijud1" in gs["rng"].get_state()
            found_edition |= bool(gs["jokers"] and gs["jokers"][0].edition)
            if found_stream and found_edition:
                break

        assert found_stream
        assert found_edition


class TestC13MagicTrickIllusion:
    def test_magic_trick_shop_playing_card_has_front(self):
        gs = initialize_run("b_red", 1, "C13_MAGIC")
        gs["phase"] = GamePhase.ROUND_EVAL
        gs["joker_rate"] = gs["tarot_rate"] = gs["planet_rate"] = 0
        gs["spectral_rate"] = 0
        gs["playing_card_rate"] = 1
        gs["shop"]["joker_max"] = 1

        step(gs, CashOut())

        assert gs["shop_cards"][0].base is not None

    def test_illusion_uses_lua_stream_order_and_can_modify_card(self):
        saw_enhancement = saw_edition = False
        for index in range(300):
            seed = f"C13_ILLUSION_{index}"
            gs = initialize_run("b_red", 1, seed)
            gs["phase"] = GamePhase.ROUND_EVAL
            gs["used_vouchers"]["v_illusion"] = True
            gs["joker_rate"] = gs["tarot_rate"] = gs["planet_rate"] = 0
            gs["spectral_rate"] = 0
            gs["playing_card_rate"] = 1
            gs["shop"]["joker_max"] = 1

            step(gs, CashOut())

            card = gs["shop_cards"][0]
            saw_enhancement |= card.center_key != "c_base"
            saw_edition |= card.edition is not None
            calls = 3 if card.edition is not None else 2
            reference = PseudoRandom(seed)
            for _ in range(calls):
                reference.random("illusion")
            assert gs["rng"].get_state()["illusion"] == reference.get_state()["illusion"]
            if saw_enhancement and saw_edition:
                break

        assert saw_enhancement
        assert saw_edition


class TestD16MarbleFront:
    def test_setting_blind_stone_has_front_from_marb_fr(self):
        gs = initialize_run("b_red", 1, "D16_MARBLE")
        _owned_joker(gs, "j_marble")
        before_ids = {id(card) for card in gs["deck"]}

        step(gs, SelectBlind())

        created = [card for card in (*gs["deck"], *gs["hand"]) if id(card) not in before_ids]
        assert len(created) == 1
        assert created[0].center_key == "m_stone"
        assert created[0].base is not None
        assert "marb_fr" in gs["rng"].get_state()


class TestLifecycleCopyAndRouting:
    def test_copy_card_has_fresh_identity_and_deep_mutables(self):
        from jackdaw.engine.lifecycle import copy_card

        gs = initialize_run("b_red", 1, "COPY")
        source = create_joker("j_ride_the_bus")
        source.ability["nested"] = {"values": [1]}
        source.eternal = True
        source.perishable = False
        source.perish_tally = 3
        source.set_rental(gs, True)
        source.debuff = True
        source.set_edition(gs, {"negative": True})

        copied = copy_card(gs, source)

        assert copied.sort_id != source.sort_id
        assert copied.added_to_deck is False
        assert copied.eternal is True
        assert copied.perish_tally == 3
        assert copied.rental is True
        assert copied.cost == 1
        assert copied.debuff is True
        copied.ability["nested"]["values"].append(2)
        assert source.ability["nested"]["values"] == [1]

    def test_copy_can_strip_negative_edition(self):
        from jackdaw.engine.lifecycle import copy_card

        gs = initialize_run("b_red", 1, "COPY_STRIP")
        source = create_joker("j_joker", game_state=gs)
        source.set_edition(gs, {"negative": True})

        copied = copy_card(gs, source, strip_edition=True)

        assert copied.edition is None
        assert copied.added_to_deck is False

    def test_copy_into_target_keeps_target_ownership_flags(self):
        # Lua copy_card never touches added_to_deck/removed, and with
        # strip_edition it skips set_edition, so the target keeps its own.
        from jackdaw.engine.lifecycle import copy_card

        gs = initialize_run("b_red", 1, "COPY_INTO")
        source, target = gs["deck"][0], gs["deck"][1]
        emplace(gs, target, "deck")
        target.set_edition(gs, {"foil": True})
        assert target.added_to_deck is True

        copy_card(gs, source, into=target, strip_edition=True)

        assert target.added_to_deck is True
        assert target.removed is False
        assert target.edition is not None and target.edition.get("foil")

    def test_deck_emplace_inserts_at_bottom_like_lua(self):
        # cardarea.lua:33 inserts deck-type emplaces at index 1; our deck
        # draws from the end, so that is index 0. The position feeds shuffles.
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit

        gs = initialize_run("b_red", 1, "DECK_EMPLACE")
        card = create_playing_card(Suit.HEARTS, Rank.ACE, game_state=gs)
        emplace(gs, card, "deck")
        assert gs["deck"][0] is card

    def test_pack_opening_hand_emplace_tracks_pack_hand(self):
        gs = initialize_run("b_red", 1, "EMPLACE")
        gs["phase"] = GamePhase.PACK_OPENING
        card = gs["deck"].pop()

        emplace(gs, card, "hand")

        assert card in gs["hand"]
        assert card in gs["pack_hand"]

    def test_debuffed_negative_removal_consumes_queued_slot_once(self):
        from jackdaw.engine.lifecycle import emplace, remove

        gs = initialize_run("b_red", 1, "NEGATIVE_REMOVE")
        joker = create_joker("j_joker", edition={"negative": True}, game_state=gs)
        original_slots = gs["joker_slots"]
        emplace(gs, joker, "jokers")
        assert gs["joker_slots"] == original_slots + 1

        joker.set_debuff(gs, True)
        assert gs["joker_slots"] == original_slots + 1
        remove(gs, joker)

        assert gs["joker_slots"] == original_slots


class TestPoolMigration:
    def test_old_historical_set_is_rebuilt_idempotently(self):
        gs = initialize_run("b_red", 1, "MIGRATE_POOL")
        joker = create_joker("j_joker")
        gs["jokers"].append(joker)
        gs["used_jokers"] = {"j_gros_michel": True}

        migrate_state(gs)
        once = dict(gs["used_jokers"])
        migrate_state(gs)

        assert gs["used_jokers"] == once
        assert "j_gros_michel" not in once
        assert once["j_joker"] is True


class TestPoolInvariant:
    def test_used_jokers_bounded_by_owned_and_existing_after_every_step(self):
        """owned keys <= used_jokers <= keys of cards that currently exist.

        Upper bound: no historical keys survive removal (C06). It is not an
        equality: Lua's Card:remove (card.lua:4741-4749) releases a name
        unless an OWNED copy exists, so removing one of two displayed
        same-named cards (e.g. opening one of two Arcana packs) clears the
        key while the sibling is still on display. Lower bound: an owned
        card always blocks its own release.
        """
        from jackdaw.engine.lifecycle import PoolTracker
        from tests.engine._rollout import drive

        rnd = random.Random(732)
        steps = 0
        for seed_index in range(20):
            for gs, _action in drive(f"POOL_INV_{seed_index}", rnd):
                steps += 1
                actual = {key for key, value in gs["used_jokers"].items() if value}
                existing = set(PoolTracker.expected(gs))
                owned = set(
                    PoolTracker.expected(
                        {"jokers": gs.get("jokers", []), "consumables": gs.get("consumables", [])}
                    )
                )
                assert owned <= actual <= existing, (
                    f"seed {seed_index}: missing {owned - actual}, stale {actual - existing}"
                )
        assert steps > 1000
