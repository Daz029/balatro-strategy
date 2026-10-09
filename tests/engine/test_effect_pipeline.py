"""Integration regressions for the joker Effect pipeline (engine P4-2)."""

from __future__ import annotations

import pytest

from jackdaw.engine import lifecycle, read
from jackdaw.engine.actions import (
    CashOut,
    GamePhase,
    NextRound,
    OpenBooster,
    PickPackCard,
    PlayHand,
    SelectBlind,
    SellCard,
    SkipBlind,
    UseConsumable,
)
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


class TestMissingDispatches:
    def test_certificate_creates_one_sealed_hand_card_only_on_first_draw(self):
        gs = initialize_run("b_red", 1, "P43_CERTIFICATE")
        _owned_joker(gs, "j_certificate")
        # This seed creates a Diamond; The Window must debuff the new card
        # before its drawn_to_hand hook runs.
        gs["round_resets"]["blind_choices"]["Small"] = "bl_window"
        before = _playing_count(gs)

        step(gs, SelectBlind())

        assert _playing_count(gs) == before + 1
        certificate_cards = [card for card in gs["hand"] if card.seal is not None]
        assert len(certificate_cards) == 1
        assert certificate_cards[0].base.suit.value == "Diamonds"
        assert certificate_cards[0].debuff is True

        gs["blind"].chips = 10**9
        step(gs, PlayHand(card_indices=(0,)))
        assert _playing_count(gs) == before + 1

    def test_hallucination_opens_shop_and_tag_packs_but_skips_rng_when_full(self):
        def shop_state(seed: str) -> dict:
            state = initialize_run("b_red", 1, seed)
            state["phase"] = GamePhase.SHOP
            state["dollars"] = 20
            _owned_joker(state, "j_hallucination")
            state["rng"] = PseudoRandom("HALU3")
            pack = Card(center_key="p_arcana_normal_1", cost=4)
            pack.ability = {"set": "Booster", "name": "Arcana Pack"}
            state["shop_boosters"] = [pack]
            return state

        gs = shop_state("P43_HALLUCINATION")
        step(gs, OpenBooster(card_index=0))
        assert [card.ability["set"] for card in gs["consumables"]] == ["Tarot"]

        full = shop_state("P43_HALLUCINATION_FULL")
        for key in ("c_fool", "c_magician"):
            lifecycle.emplace(full, create_consumable(key, game_state=full), "consumables")
        assert "halu1" not in full["rng"].get_state()
        step(full, OpenBooster(card_index=0))
        assert "halu1" not in full["rng"].get_state()

        tagged = initialize_run("b_red", 1, "P43_HALLUCINATION_TAG")
        _owned_joker(tagged, "j_hallucination")
        tagged["rng"] = PseudoRandom("HALU3")
        tagged["round_resets"]["blind_tags"]["Small"] = "tag_charm"
        step(tagged, SkipBlind())
        assert [card.ability["set"] for card in tagged["consumables"]] == ["Tarot"]

    def test_anaglyph_awards_double_tag_only_for_boss(self):
        outcomes = {}
        for blind_on_deck in ("Small", "Boss"):
            gs = initialize_run("b_anaglyph", 1, f"P43_ANAGLYPH_{blind_on_deck}")
            gs["blind_on_deck"] = blind_on_deck
            if blind_on_deck == "Boss":
                gs["round_resets"]["blind_choices"]["Boss"] = "bl_arm"
            step(gs, SelectBlind(allow_forced_boss=True))
            gs["blind"].chips = 1

            step(gs, PlayHand(card_indices=(0,)))

            outcomes[blind_on_deck] = [entry["key"] for entry in gs.get("awarded_tags", [])]
        assert "tag_double" not in outcomes["Small"]
        assert "tag_double" in outcomes["Boss"]


class TestQueuedSales:
    @staticmethod
    def _shop_state(seed: str) -> dict:
        gs = initialize_run("b_red", 1, seed)
        gs["phase"] = GamePhase.SHOP
        return gs

    def test_diet_cola_sale_awards_double_tag(self):
        gs = self._shop_state("P43_DIET_COLA")
        _owned_joker(gs, "j_diet_cola")

        step(gs, SellCard(area="jokers", card_index=0))

        assert [entry["key"] for entry in gs["awarded_tags"]] == ["tag_double"]

    def test_luchador_sale_disables_boss(self):
        gs = self._shop_state("P43_LUCHADOR")
        gs["blind"] = Blind.create("bl_arm", ante=1)
        _owned_joker(gs, "j_luchador")

        step(gs, SellCard(area="jokers", card_index=0))

        assert gs["blind"].disabled is True

    def test_invisible_sale_copies_other_joker_and_strips_negative(self):
        gs = self._shop_state("P43_INVISIBLE")
        invisible = _owned_joker(gs, "j_invisible")
        invisible.ability["invis_rounds"] = invisible.ability.get("extra", 2)
        source = create_joker("j_joker", edition={"negative": True}, game_state=gs)
        lifecycle.emplace(gs, source, "jokers")

        step(gs, SellCard(area="jokers", card_index=0))

        assert len(gs["jokers"]) == 2
        copy = next(joker for joker in gs["jokers"] if joker is not source)
        assert copy.center_key == source.center_key
        assert not copy.edition or not copy.edition.get("negative")

    def test_verdant_leaf_disables_when_a_joker_is_sold(self):
        gs = self._shop_state("P43_VERDANT")
        gs["blind"] = Blind.create("bl_final_leaf", ante=8)
        _owned_joker(gs, "j_joker")

        step(gs, SellCard(area="jokers", card_index=0))

        assert gs["blind"].disabled is True


class TestConsumableEffects:
    @staticmethod
    def _shop_state(seed: str) -> dict:
        gs = initialize_run("b_red", 1, seed)
        gs["phase"] = GamePhase.SHOP
        return gs

    def test_ankh_with_full_ordinary_row_leaves_survivor_and_copy(self, monkeypatch):
        from jackdaw.engine import effects

        gs = self._shop_state("P43_ANKH_FULL")
        for key in (
            "j_joker",
            "j_greedy_joker",
            "j_lusty_joker",
            "j_wrathful_joker",
            "j_gluttenous_joker",
        ):
            _owned_joker(gs, key)
        lifecycle.emplace(gs, create_consumable("c_ankh", game_state=gs), "consumables")
        applied = []
        original = effects.apply_effects

        def capture(state, pending):
            applied.extend(pending)
            original(state, pending)

        monkeypatch.setattr(effects, "apply_effects", capture)

        step(gs, UseConsumable(card_index=0))

        assert len(gs["jokers"]) == 2
        assert gs["jokers"][0].center_key == gs["jokers"][1].center_key
        assert sum(isinstance(effect, effects.DestroyCard) for effect in applied) == 4
        assert sum(isinstance(effect, effects.CopyCard) for effect in applied) == 1

    def test_ankh_strips_negative_from_copy(self, monkeypatch):
        from jackdaw.engine import effects

        gs = self._shop_state("P43_ANKH_NEGATIVE")
        chosen = create_joker("j_joker", edition={"negative": True}, game_state=gs)
        lifecycle.emplace(gs, chosen, "jokers")
        lifecycle.emplace(gs, create_consumable("c_ankh", game_state=gs), "consumables")
        applied = []
        original = effects.apply_effects

        def capture(state, pending):
            applied.extend(pending)
            original(state, pending)

        monkeypatch.setattr(effects, "apply_effects", capture)

        step(gs, UseConsumable(card_index=0))

        assert len(gs["jokers"]) == 2
        copy = next(joker for joker in gs["jokers"] if joker is not chosen)
        assert chosen.edition and chosen.edition.get("negative")
        assert not copy.edition or not copy.edition.get("negative")
        copy_effect = next(effect for effect in applied if isinstance(effect, effects.CopyCard))
        assert copy_effect.strip_edition is True

    def test_high_priestess_with_one_slot_draws_exactly_one_planet(self, monkeypatch):
        from jackdaw.engine import card_factory

        gs = self._shop_state("P43_PRIESTESS")
        filler = create_consumable("c_fool", game_state=gs)
        priestess = create_consumable("c_high_priestess", game_state=gs)
        lifecycle.emplace(gs, filler, "consumables")
        lifecycle.emplace(gs, priestess, "consumables")
        calls = []
        original = card_factory.resolve_create_descriptor

        def counted(*args, **kwargs):
            calls.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr(card_factory, "resolve_create_descriptor", counted)
        step(gs, UseConsumable(card_index=1))

        assert len(calls) == 1
        assert len(gs["consumables"]) == 2
        assert sum(card.ability["set"] == "Planet" for card in gs["consumables"]) == 1

    def test_glass_joker_hanged_man_counts_each_glass_card_once(self):
        gs = initialize_run("b_red", 1, "P43_GLASS_HANGED")
        glass_joker = _owned_joker(gs, "j_glass")
        step(gs, SelectBlind())
        for card in gs["hand"][:2]:
            card.set_ability("m_glass", gs=gs)
        hanged = create_consumable("c_hanged_man", game_state=gs)
        lifecycle.emplace(gs, hanged, "consumables")
        before = glass_joker.ability["x_mult"]

        step(gs, UseConsumable(card_index=0, target_indices=(0, 1)))

        assert glass_joker.ability["x_mult"] == before + 2 * glass_joker.ability["extra"]

    def test_pack_consumable_dispatches_with_card_and_highlighted_context(self):
        gs = initialize_run("b_red", 1, "P43_PACK_CONSUMABLE")
        glass_joker = _owned_joker(gs, "j_glass")
        targets = [gs["deck"].pop(), gs["deck"].pop()]
        for card in targets:
            card.set_ability("m_glass", gs=gs)
        gs["hand"] = targets
        hanged = create_consumable("c_hanged_man", game_state=gs)
        lifecycle.emplace(gs, hanged, "pack_cards")
        gs["pack_choices_remaining"] = 1
        gs["pack_type"] = "Arcana"
        gs["pack_hand"] = []
        gs["shop_return_phase"] = GamePhase.SHOP
        gs["phase"] = GamePhase.PACK_OPENING
        before = glass_joker.ability["x_mult"]

        step(gs, PickPackCard(card_index=0, target_indices=(0, 1)))

        assert glass_joker.ability["x_mult"] == before + 2 * glass_joker.ability["extra"]

    @pytest.mark.parametrize(
        "consumable_key,created_count",
        [("c_familiar", 3), ("c_grim", 2), ("c_incantation", 4)],
    )
    def test_spectral_playing_cards_send_one_batch_notification(
        self, consumable_key: str, created_count: int, monkeypatch
    ):
        from jackdaw.engine import effects

        gs = initialize_run("b_red", 1, f"P43_BATCH_{consumable_key}")
        step(gs, SelectBlind())
        consumable = create_consumable(consumable_key, game_state=gs)
        lifecycle.emplace(gs, consumable, "consumables")
        notifications = []
        applied = []
        original_fire = lifecycle.fire_joker_context
        original_apply = effects.apply_effects

        def capture(state, **fields):
            if fields.get("playing_card_added"):
                notifications.append(list(fields["cards"]))
            return original_fire(state, **fields)

        def capture_applied(state, pending):
            applied.extend(pending)
            original_apply(state, pending)

        monkeypatch.setattr(lifecycle, "fire_joker_context", capture)
        monkeypatch.setattr(effects, "apply_effects", capture_applied)
        step(gs, UseConsumable(card_index=0))

        assert [len(cards) for cards in notifications] == [created_count]
        assert sum(isinstance(effect, effects.CreatePlayingCards) for effect in applied) == 1


def test_blind_triggered_is_reset_before_each_play_for_matador() -> None:
    gs = initialize_run("b_red", 1, "P43_MATADOR_RESET")
    _owned_joker(gs, "j_matador")
    step(gs, SelectBlind())
    gs["blind"].chips = 10**9
    gs["hand"][0].debuff = True
    before = gs["dollars"]

    step(gs, PlayHand(card_indices=(0,)))
    after_trigger = gs["dollars"]
    assert after_trigger == before + 8

    clean_index = next(i for i, card in enumerate(gs["hand"]) if not card.debuff)
    step(gs, PlayHand(card_indices=(clean_index,)))
    assert gs["dollars"] == after_trigger
