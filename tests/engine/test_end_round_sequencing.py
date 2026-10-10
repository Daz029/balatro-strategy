"""Step-level regressions for end_round, round evaluation, and cash-out (P5-4)."""

from __future__ import annotations

from typing import Any

from jackdaw.engine import lifecycle
from jackdaw.engine.actions import CashOut, GamePhase, NextRound, PlayHand, SelectBlind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.hands import HandType
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.observation import encode_global_context


def _card(gs: dict[str, Any], rank: Rank = Rank.TWO):
    return create_playing_card(Suit.SPADES, rank, game_state=gs)


def _joker(gs: dict[str, Any], key: str):
    card = create_joker(key, game_state=gs)
    lifecycle.emplace(gs, card, "jokers")
    return card


def _select(gs: dict[str, Any], blind_key: str = "bl_small") -> None:
    blind_type = "Small" if blind_key == "bl_small" else "Big" if blind_key == "bl_big" else "Boss"
    gs["blind_on_deck"] = blind_type
    gs["round_resets"]["blind_choices"][blind_type] = blind_key
    step(gs, SelectBlind(allow_forced_boss=True))


def _one_card_round(
    seed: str,
    *,
    blind_key: str = "bl_small",
    held: list | None = None,
) -> dict[str, Any]:
    gs = initialize_run("b_red", 1, seed)
    _select(gs, blind_key)
    gs["hand"] = [_card(gs), *(held or [])]
    gs["deck"] = []
    gs["discard_pile"] = []
    gs["blind"].chips = 1
    return gs


def test_c10_held_gold_red_seal_and_mime_pay_nine_dollars() -> None:
    # L: state_events.lua:171-233; card.lua:1033-1064,3386-3394.
    gs = initialize_run("b_red", 1, "P54_GOLD_MIME")
    mime = _joker(gs, "j_mime")
    del mime
    _select(gs)
    gold = create_playing_card(
        Suit.HEARTS,
        Rank.KING,
        enhancement="m_gold",
        seal="Red",
        game_state=gs,
    )
    gs["hand"] = [_card(gs), gold]
    gs["deck"] = []
    gs["blind"].chips = 1
    before = gs["dollars"]

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["dollars"] == before + 9


def test_c10_blue_seal_creates_initialized_pluto_for_last_played_hand() -> None:
    # L: card.lua:1040-1057 uses last_hand_played and the ordinary create_card path.
    gs = initialize_run("b_red", 1, "P54_BLUE_PLUTO")
    gs["hand_levels"].get_state(HandType.PAIR).played = 20
    _select(gs)
    blue = create_playing_card(Suit.HEARTS, Rank.KING, seal="Blue", game_state=gs)
    gs["hand"] = [_card(gs), blue]
    gs["deck"] = []
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["last_hand_played"] == "High Card"
    assert len(gs["consumables"]) == 1
    pluto = gs["consumables"][0]
    assert pluto.center_key == "c_pluto"
    assert pluto.ability["consumeable"]["hand_type"] == "High Card"
    assert pluto.cost == 3
    assert gs["used_jokers"]["c_pluto"] is True


def test_c11_seven_committed_chips_do_not_consume_bones_or_save() -> None:
    # L: state_events.lua:92-105; card.lua:3047-3062.
    gs = initialize_run("b_red", 1, "P54_BONES_LOW")
    bones = _joker(gs, "j_mr_bones")
    _select(gs)
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["chips"] = 0
    gs["blind"].chips = 300
    gs["current_round"]["hands_left"] = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["chips"] == 7
    assert gs["phase"] == GamePhase.GAME_OVER
    assert bones in gs["jokers"]
    assert bones.removed is False


def test_c11_bones_saves_at_eighty_committed_chips_but_blind_row_is_zero() -> None:
    # L: state_events.lua:92-105,1139-1146; card.lua:3047-3062.
    gs = initialize_run("b_red", 1, "P54_BONES_SAVE")
    bones = _joker(gs, "j_mr_bones")
    _select(gs)
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["chips"] = 73
    gs["blind"].chips = 300
    gs["current_round"]["hands_left"] = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["chips"] == 80
    assert gs["phase"] == GamePhase.ROUND_EVAL
    assert bones not in gs["jokers"]
    assert gs["round_earnings"].blind_reward == 0


def test_d18_fresh_rocket_pays_three_on_the_boss_it_scales_from() -> None:
    # L: state_events.lua:99-110,1175-1182; card.lua:1664-1666,2896-2901.
    gs = initialize_run("b_red", 1, "P54_ROCKET")
    rocket = _joker(gs, "j_rocket")
    _select(gs, "bl_hook")
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert rocket.ability["extra"]["dollars"] == 3
    assert gs["round_earnings"].joker_dollars == 3


def test_d18_golden_joker_perishing_this_round_pays_nothing() -> None:
    # L: state_events.lua:99-110,1175-1182; card.lua:1655-1659,2278-2288.
    gs = initialize_run("b_red", 1, "P54_GOLDEN_PERISH")
    golden = _joker(gs, "j_golden")
    golden.set_perishable(gs, True)
    golden.set_perish_tally(1)
    _select(gs)
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert golden.debuff is True
    assert golden.perish_tally == 0
    assert gs["round_earnings"].joker_dollars == 0


def test_d22_self_destroying_rental_still_pays_rent() -> None:
    # L: state_events.lua:99-110; card.lua:2271-2288,2903-2908.
    gs = initialize_run("b_red", 1, "P54_RENTAL_DESTROY")
    bean = _joker(gs, "j_turtle_bean")
    bean.set_rental(gs, True)
    bean.ability["extra"]["h_size"] = 1
    _select(gs)
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["blind"].chips = 1
    before = gs["dollars"]

    step(gs, PlayHand(card_indices=(0,)))

    assert bean.removed is True
    assert bean not in gs["jokers"]
    assert gs["dollars"] == before - gs["rental_rate"]


def test_d22_losing_round_runs_maintenance_and_charges_rent() -> None:
    # L: state_events.lua:92-123 still runs the per-Joker loop before GAME_OVER.
    gs = initialize_run("b_red", 1, "P54_RENTAL_LOSS")
    rental = _joker(gs, "j_joker")
    rental.set_rental(gs, True)
    _select(gs)
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["blind"].chips = 300
    gs["current_round"]["hands_left"] = 1
    before = gs["dollars"]

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["phase"] == GamePhase.GAME_OVER
    assert gs["dollars"] == before - gs["rental_rate"]


def test_empty_hand_and_deck_loss_uses_end_round_maintenance() -> None:
    # L: game.lua:3056-3065 routes the exhausted state to end_round.
    gs = initialize_run("b_red", 1, "P54_EMPTY_LOSS")
    rental = _joker(gs, "j_joker")
    rental.set_rental(gs, True)
    gs["hand_size"] = 0
    gs["deck"] = []
    before = gs["dollars"]

    step(gs, SelectBlind())

    assert gs["phase"] == GamePhase.GAME_OVER
    assert gs["dollars"] == before - gs["rental_rate"]


def test_d30_boss_defeat_clears_old_marker_before_a_later_pillar() -> None:
    # L: state_events.lua:258-267.
    gs = initialize_run("b_red", 1, "P54_PILLAR")
    _select(gs, "bl_hook")
    old = _card(gs, Rank.KING)
    old.ability["played_this_ante"] = True
    gs["hand"] = [_card(gs), old]
    gs["deck"] = []
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))
    assert "played_this_ante" not in old.ability
    step(gs, CashOut())
    step(gs, NextRound())
    gs["blind_on_deck"] = "Boss"
    gs["round_resets"]["blind_choices"]["Boss"] = "bl_pillar"
    step(gs, SelectBlind(allow_forced_boss=True))

    assert old.debuff is False


def test_d32_defeat_flips_jokers_and_clears_boss_debuffs() -> None:
    # L: blind.lua:276-344,78-215,624-652.
    gs = initialize_run("b_red", 1, "P54_DEFEAT")
    joker = _joker(gs, "j_joker")
    _select(gs, "bl_final_acorn")
    playing = _card(gs, Rank.KING)
    gs["hand"] = [_card(gs), playing]
    gs["deck"] = []
    playing.set_debuff(gs, True)
    joker.set_debuff(gs, True)
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert joker.facing == "front"
    assert joker.debuff is False
    assert playing.debuff is False
    assert gs["blind"].name == ""
    assert gs["blind"].chips == 0


def test_d32_defeat_restores_manacle_hand_size_once() -> None:
    # L: blind.lua:187-189,341-343.
    gs = initialize_run("b_red", 1, "P54_MANACLE")
    base_size = gs["hand_size"]
    _select(gs, "bl_manacle")
    assert gs["hand_size"] == base_size - 1
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["hand_size"] == base_size


def test_new_p5_1_02_cashout_resets_resources_before_shop_observation() -> None:
    # L: button_callbacks.lua:2918-2937.
    gs = _one_card_round("P54_SHOP_RESET")
    step(gs, PlayHand(card_indices=(0,)))
    gs["current_round"]["hands_left"] = 0
    gs["current_round"]["discards_left"] = 0
    gs["current_round"]["jokers_purchased"] = 3

    step(gs, CashOut())

    expected_hands = max(1, gs["round_resets"]["hands"] + gs["round_bonus"]["next_hands"])
    expected_discards = max(0, gs["round_resets"]["discards"] + gs["round_bonus"]["discards"])
    assert gs["phase"] == GamePhase.SHOP
    assert gs["current_round"]["hands_left"] == expected_hands
    assert gs["current_round"]["discards_left"] == expected_discards
    assert gs["current_round"]["jokers_purchased"] == 0
    obs = encode_global_context(gs)
    assert obs[13] == expected_hands / 10
    assert obs[14] == expected_discards / 10


def test_new_p5_1_03_boss_tie_replicates_luas_stale_order_bug() -> None:
    # L: state_events.lua:129-137. Lua never updates _order; preserve that bug.
    gs = initialize_run("b_red", 1, "P54_TIE_BUG")
    gs["hand_levels"].get_state(HandType.PAIR).played = 5
    gs["hand_levels"].get_state(HandType.TWO_PAIR).played = 5
    _select(gs, "bl_hook")
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["current_round"]["most_played_poker_hand"] == "Pair"


def test_new_p5_1_09_target_cards_reset_at_end_round_not_cashout() -> None:
    # L: state_events.lua:251-276.
    gs = _one_card_round("P54_TARGET_RESET")
    streams = ("idol1", "mail1", "anc1", "cas1")
    before = {key: gs["rng"].state.get(key) for key in streams}

    step(gs, PlayHand(card_indices=(0,)))

    at_eval = {key: gs["rng"].state.get(key) for key in streams}
    assert all(at_eval[key] != before[key] for key in streams)
    targets_at_eval = {
        key: dict(gs["current_round"][key])
        for key in ("idol_card", "mail_card", "ancient_card", "castle_card")
    }
    step(gs, CashOut())
    assert {key: gs["rng"].state.get(key) for key in streams} == at_eval
    assert {
        key: dict(gs["current_round"][key])
        for key in ("idol_card", "mail_card", "ancient_card", "castle_card")
    } == targets_at_eval


def test_f8_commits_bottom_row_then_snapshots_and_clears_chips() -> None:
    # L: button_callbacks.lua:2928-2954.
    gs = _one_card_round("P54_F8")
    step(gs, PlayHand(card_indices=(0,)))
    before = gs["dollars"]
    bottom_row = gs["current_round"]["dollars"]
    assert bottom_row == gs["round_earnings"].total

    step(gs, CashOut())

    assert gs["dollars"] == before + bottom_row
    assert gs["previous_round"]["dollars"] == gs["dollars"]
    assert gs["chips"] == 0


def test_defeat_runs_after_reward_rows_so_boss_debuffed_golden_pays_nothing() -> None:
    # L: state_events.lua:1148-1155 QUEUES Blind:defeat, so the synchronous
    # calc_dollar_bonus rows (:1175-1182; card.lua:1655 debuff gate) still see
    # the boss's joker debuff. Defeat then clears it (blind.lua:330-337).
    gs = initialize_run("b_red", 1, "P54_HEART_GOLDEN")
    golden = _joker(gs, "j_golden")
    _select(gs, "bl_final_heart")
    golden.set_debuff(gs, True)
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["blind"].chips = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["phase"] == GamePhase.ROUND_EVAL
    assert gs["round_earnings"].joker_dollars == 0
    assert golden.debuff is False


def test_losing_the_win_ante_boss_is_not_a_win() -> None:
    # Lua sets G.GAME.won before the game_over branch even on a loss
    # (state_events.lua:111-114); the engine's `won` means "run won".
    gs = initialize_run("b_red", 1, "P54_FINAL_LOSS")
    gs["round_resets"]["ante"] = gs.get("win_ante", 8)
    _select(gs, "bl_club")
    gs["hand"] = [_card(gs)]
    gs["deck"] = []
    gs["blind"].chips = 10_000
    gs["current_round"]["hands_left"] = 1

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["phase"] == GamePhase.GAME_OVER
    assert gs.get("won") is False


def test_held_card_eor_passes_do_not_refire_plain_end_of_round_handlers() -> None:
    # L: card.lua:2874-2887 — end_of_round is tested before individual /
    # repetition; its individual branch is empty and repetition is Mime-only.
    # Rocket must scale exactly once however many cards are held.
    gs = _one_card_round(
        "P54_ROCKET_HELD",
        blind_key="bl_club",
        held=[],
    )
    rocket = _joker(gs, "j_rocket")
    gs["hand"].extend(_card(gs, rank) for rank in (Rank.THREE, Rank.FOUR, Rank.FIVE))
    start = rocket.ability["extra"]["dollars"]
    increase = rocket.ability["extra"]["increase"]

    step(gs, PlayHand(card_indices=(0,)))

    assert rocket.ability["extra"]["dollars"] == start + increase
