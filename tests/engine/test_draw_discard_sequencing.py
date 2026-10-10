"""Line-order regressions for draw, discard, and new_round (P5-3)."""

from __future__ import annotations

import pytest

from jackdaw.engine import game, lifecycle
from jackdaw.engine.actions import Discard, GamePhase, PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run


def _card(gs: dict, suit: Suit, rank: Rank):
    return create_playing_card(suit, rank, game_state=gs)


def _joker(gs: dict, key: str):
    card = create_joker(key, game_state=gs)
    lifecycle.emplace(gs, card, "jokers")
    return card


def _select(gs: dict, blind_key: str, *, allow_forced_boss: bool = True) -> None:
    gs["blind_on_deck"] = "Boss"
    gs["round_resets"]["blind_choices"]["Boss"] = blind_key
    step(gs, SelectBlind(allow_forced_boss=allow_forced_boss))


@pytest.mark.parametrize("blind_key", ["bl_house", "bl_wheel"])
def test_c09_house_and_wheel_flip_real_initial_draws(
    blind_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # L: blind.lua:605-613; common_events.lua:386-423.
    if blind_key == "bl_wheel":
        original = PseudoRandom.random

        def force_wheel(self, key, min_val=None, max_val=None):
            if key == "wheel":
                self.seed("wheel")
                return 0.0
            return original(self, key, min_val, max_val)

        monkeypatch.setattr(PseudoRandom, "random", force_wheel)

    gs = initialize_run("b_red", 1, f"P53_{blind_key}")
    cards = [_card(gs, Suit.HEARTS, Rank.TWO), _card(gs, Suit.CLUBS, Rank.THREE)]
    gs["deck"] = cards

    _select(gs, blind_key)

    assert gs["hand"]
    assert all(card.facing == "back" for card in gs["hand"])
    assert all(card.ability.get("wheel_flipped") is True for card in gs["hand"])


def test_c09_mark_flips_only_faces_on_a_real_initial_draw() -> None:
    # L: blind.lua:614-616; cardarea.lua:32-43.
    gs = initialize_run("b_red", 1, "P53_MARK")
    king = _card(gs, Suit.SPADES, Rank.KING)
    two = _card(gs, Suit.HEARTS, Rank.TWO)
    gs["deck"] = [king, two]

    _select(gs, "bl_mark")

    assert king.facing == "back"
    assert king.ability.get("wheel_flipped") is True
    assert two.facing == "front"
    assert "wheel_flipped" not in two.ability


def test_c09_fish_flips_only_cards_drawn_after_play() -> None:
    # L: blind.lua:494-496,618-620; common_events.lua:386-423.
    gs = initialize_run("b_red", 1, "P53_FISH")
    _select(gs, "bl_fish")
    held_before = list(gs["hand"][1:])
    deck_before = set(map(id, gs["deck"]))
    gs["blind"].chips = 10**9

    step(gs, PlayHand(card_indices=(0,)))

    drawn = [card for card in gs["hand"] if id(card) in deck_before]
    assert held_before
    assert drawn
    assert all(card.facing == "front" for card in held_before)
    assert all(card.facing == "back" for card in drawn)
    assert all(card.ability.get("wheel_flipped") is True for card in drawn)
    assert gs["blind"].prepped is None


def test_d28_discard_does_not_reroll_crimson_heart_or_advance_stream() -> None:
    # L: blind.lua:588-603: Heart requires prepped; every draw clears it.
    gs = initialize_run("b_red", 1, "P53_HEART")
    left = _joker(gs, "j_joker")
    right = _joker(gs, "j_mime")
    _select(gs, "bl_final_heart")
    before = gs["rng"].state.get("crimson_heart")
    debuffs = (left.debuff, right.debuff)

    step(gs, Discard(card_indices=(0,)))

    assert gs["rng"].state.get("crimson_heart") == before
    assert (left.debuff, right.debuff) == debuffs
    assert gs["blind"].prepped is None


def test_d31_hook_uses_discard_pipeline_but_burnt_guard_blocks_level() -> None:
    # L: blind.lua:466-487; state_events.lua:394-430; card.lua:2748-2755.
    gs = initialize_run("b_red", 1, "P53_HOOK")
    _select(gs, "bl_hook")
    mail = _joker(gs, "j_mail")
    burnt = _joker(gs, "j_burnt")
    del mail, burnt
    played = _card(gs, Suit.CLUBS, Rank.TWO)
    purple_ace = _card(gs, Suit.SPADES, Rank.ACE)
    other = _card(gs, Suit.DIAMONDS, Rank.THREE)
    purple_ace.set_seal(gs, "Purple")
    gs["hand"] = [played, purple_ace, other]
    gs["deck"] = []
    gs["current_round"]["mail_card"] = {"rank": "Ace", "id": 14}
    gs["blind"].chips = 10**9
    dollars = gs["dollars"]

    step(gs, PlayHand(card_indices=(0,)))

    assert len(gs["consumables"]) == 1
    assert gs["dollars"] == dollars + 5
    assert gs["hand_levels"]["High Card"].level == 1
    assert gs["current_round"]["discards_used"] == 0


def test_d32_amber_acorn_runs_three_aajk_shuffles() -> None:
    # L: blind.lua:190-205. pseudoshuffle pre-sorts by sort_id each time
    # (misc_functions.lua:206-218). For P53_ACORN the three orders are
    # abdce -> eabdc -> ecbad, so the final exact order is e,c,b,a,d.
    gs = initialize_run("b_red", 1, "P53_ACORN")
    keys = ["j_joker", "j_bull", "j_mime", "j_mail", "j_burnt"]
    jokers = [_joker(gs, key) for key in keys]
    names = dict(zip(map(id, jokers), "abcde", strict=True))

    _select(gs, "bl_final_acorn")

    assert "".join(names[id(card)] for card in gs["jokers"]) == "ecbad"
    assert all(card.facing == "back" for card in gs["jokers"])
    assert gs["rng"].state["aajk"] == pytest.approx(0.5256759670741)


def test_new_p5_1_04_new_round_clears_wheel_flipped_markers() -> None:
    # L: state_events.lua:307-309.
    gs = initialize_run("b_red", 1, "P53_WHEEL_RESET")
    card = _card(gs, Suit.HEARTS, Rank.TWO)
    card.ability["wheel_flipped"] = True
    card.facing = "back"
    gs["deck"] = [card]

    step(gs, SelectBlind())

    assert "wheel_flipped" not in card.ability


def test_set_blind_redebuffs_jokers_and_certificate_lands_after_draws() -> None:
    # L: blind.lua:207-215; game.lua:3219-3241; card.lua:2462-2481.
    gs = initialize_run("b_red", 1, "P53_CERTIFICATE")
    ordinary = _joker(gs, "j_joker")
    ordinary.debuff = True
    _joker(gs, "j_certificate")
    gs["hand_size"] = 2
    gs["deck"] = [
        _card(gs, Suit.HEARTS, Rank.TWO),
        _card(gs, Suit.SPADES, Rank.THREE),
    ]

    step(gs, SelectBlind())

    assert ordinary.debuff is False
    # draw_from_deck_to_hand snapshots two draw events before Certificate is
    # dispatched, so its queued card arrives third even though capacity is 2.
    assert len(gs["hand"]) == 3
    assert gs["deck"] == []


def test_new_p5_1_07_zero_capacity_empty_initial_hand_loses() -> None:
    # L: state_events.lua:355-360.
    gs = initialize_run("b_red", 1, "P53_ZERO_HAND")
    gs["hand_size"] = 0
    gs["deck"] = [_card(gs, Suit.HEARTS, Rank.TWO)]

    step(gs, SelectBlind())

    assert gs["phase"] == GamePhase.GAME_OVER
    assert gs["won"] is False


@pytest.mark.parametrize("action", ["discard", "play"])
def test_serpent_draws_exactly_three_through_the_shared_draw(
    action: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # L: state_events.lua:362-376; game.lua:3208-3244.
    shared_draw = game._draw_from_deck_to_hand
    calls = 0

    def count_shared_draw(gs):
        nonlocal calls
        calls += 1
        return shared_draw(gs)

    monkeypatch.setattr(game, "_draw_from_deck_to_hand", count_shared_draw)
    gs = initialize_run("b_red", 1, f"P53_SERPENT_{action}")
    _select(gs, "bl_serpent")
    # Ignore the initial shared draw; this assertion concerns the action draw.
    calls = 0
    original_hand = list(gs["hand"])
    gs["blind"].chips = 10**9

    if action == "discard":
        step(gs, Discard(card_indices=(0,)))
        expected = len(original_hand) - 1 + 3
    else:
        step(gs, PlayHand(card_indices=(0,)))
        expected = len(original_hand) - 1 + 3

    assert calls == 1
    assert len(gs["hand"]) == expected
