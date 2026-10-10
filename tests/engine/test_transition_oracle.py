"""Whole-transition checks against Balatro 1.0.1o's real Lua code."""

from __future__ import annotations

import pytest

from tests.oracle import (
    Action,
    BlindSpec,
    CardSpec,
    OracleTrace,
    Scenario,
    compare,
    run_engine,
    run_lua_oracle,
)
from tests.oracle.transition import lua_oracle_unavailable_reason

_unavailable = lua_oracle_unavailable_reason()
pytestmark = pytest.mark.skipif(
    _unavailable is not None,
    reason=_unavailable or "oracle unavailable",
)


def card(card_id: str, front: str = "H_2", **kwargs: object) -> CardSpec:
    return CardSpec(card_id, front=front, **kwargs)


def joker(card_id: str, key: str, **kwargs: object) -> CardSpec:
    return CardSpec(card_id, key=key, **kwargs)


def assert_matches(scenario: Scenario, fields: tuple[str, ...]) -> None:
    diffs = compare(run_lua_oracle(scenario), run_engine(scenario), fields=fields)
    assert not diffs, "\n".join(str(diff) for diff in diffs)


def assert_traces_match(
    lua_trace: OracleTrace,
    engine_trace: OracleTrace,
    fields: tuple[str, ...],
) -> None:
    diffs = compare(lua_trace, engine_trace, fields=fields)
    assert not diffs, "\n".join(str(diff) for diff in diffs)


# ---------------------------------------------------------------------------
# Harness smoke and timing
# ---------------------------------------------------------------------------


def test_lua_smoke_high_card_play() -> None:
    scenario = Scenario(
        areas={
            "hand": [card("played")],
            "deck": [card("drawn", "S_3")],
        },
        action=Action.play(0),
    )
    trace = run_lua_oracle(scenario)

    assert trace.final["chips"] == 7
    assert [item["id"] for item in trace.final["areas"]["discard"]] == ["played"]
    assert [item["id"] for item in trace.final["areas"]["hand"]] == ["drawn"]
    assert trace.final["events_pending"] == 0


def test_lua_smoke_discard_two_cards() -> None:
    scenario = Scenario(
        areas={
            "hand": [card("a"), card("b", "S_3"), card("c", "D_4")],
            "deck": [card("d", "C_5"), card("e", "H_6")],
        },
        action=Action.discard(0, 2),
    )
    trace = run_lua_oracle(scenario)

    assert {item["id"] for item in trace.final["areas"]["discard"]} == {"a", "c"}
    assert len(trace.final["areas"]["hand"]) == 3
    assert trace.final["current_round"]["discards_left"] == 2
    assert trace.final["events_pending"] == 0


def test_lua_smoke_winning_play_runs_end_round() -> None:
    scenario = Scenario(
        blind=BlindSpec("bl_small", {"chips": 7}),
        areas={"hand": [card("winner")]},
        action=Action.play(0),
    )
    trace = run_lua_oracle(scenario)

    assert trace.final["chips"] == 7
    assert trace.final["phase"] == "round_eval"
    assert [item["id"] for item in trace.final["areas"]["deck"]] == ["winner"]
    assert trace.final["events_pending"] == 0


def test_c05_queued_gold_seal_money_is_visible_to_bull() -> None:
    scenario = Scenario(
        dollars=4,
        areas={"hand": [card("gold-two", seal="Gold")]},
        jokers=[joker("bull", "j_bull")],
        action=Action.play(0),
    )
    trace = run_lua_oracle(scenario)

    assert trace.synchronous["dollars"] == 4
    assert trace.synchronous["dollar_buffer"] == 3
    assert trace.final["dollars"] == 7
    assert trace.final["dollar_buffer"] == 0
    assert trace.final["chips"] == 21


# ---------------------------------------------------------------------------
# Scenarios that already agree (validates comparison and state mapping)
# ---------------------------------------------------------------------------


def test_differential_plain_high_card_already_matches() -> None:
    scenario = Scenario(
        areas={"hand": [card("p")], "deck": [card("d", "S_3")]},
        action=Action.play(0),
    )
    assert_matches(scenario, ("chips", "dollars", "hands_played", "last_hand_played", "hands"))


def test_differential_plain_pair_already_matches() -> None:
    scenario = Scenario(
        areas={
            "hand": [card("h2", "H_2"), card("s2", "S_2")],
            "deck": [card("d", "C_3")],
        },
        action=Action.play(0, 1),
    )
    assert_matches(scenario, ("chips", "dollars", "hands_played", "last_hand_played", "hands"))


def test_differential_flush_plus_mult_joker_already_matches() -> None:
    scenario = Scenario(
        areas={
            "hand": [
                card("h2", "H_2"),
                card("h3", "H_3"),
                card("h4", "H_4"),
                card("h5", "H_5"),
                card("h6", "H_6"),
            ]
        },
        jokers=[joker("joker", "j_joker")],
        action=Action.play(0, 1, 2, 3, 4),
    )
    assert_matches(scenario, ("chips", "dollars", "hands_played", "last_hand_played", "hands"))


def test_differential_discard_without_jokers_already_matches() -> None:
    scenario = Scenario(
        areas={
            "hand": [card("a"), card("b", "S_3"), card("c", "D_4")],
            "deck": [card("d", "C_5"), card("e", "H_6")],
        },
        action=Action.discard(0, 2),
    )
    assert_matches(scenario, ("dollars", "current_round", "hands_played", "pseudorandom"))


def test_differential_small_blind_end_round_interest_already_matches() -> None:
    scenario = Scenario(
        dollars=10,
        chips=1_000,
        hands_left=3,
        discards_left=2,
        areas={"hand": [card("held")]},
        action=Action.end_round(),
    )
    assert_matches(scenario, ("phase", "dollars", "current_round", "hands_played"))


def test_lua_round_eval_rows_capture_gameplay_amounts() -> None:
    scenario = Scenario(
        dollars=10,
        chips=1_000,
        hands_left=3,
        discards_left=2,
        areas={"hand": [card("held")]},
        action=Action.end_round(),
    )

    assert run_lua_oracle(scenario).final["round_eval_rows"] == [
        {"type": "blind1", "amount": 3},
        {"type": "hands", "amount": 3},
        {"type": "interest", "amount": 2},
        {"type": "bottom", "amount": 8},
    ]


def test_differential_gold_seal_without_money_reader_already_matches() -> None:
    scenario = Scenario(
        areas={"hand": [card("gold-two", seal="Gold")], "deck": [card("d", "S_3")]},
        action=Action.play(0),
    )
    assert_matches(scenario, ("chips", "dollars", "hands_played", "last_hand_played"))


def test_differential_plain_cash_out_already_matches() -> None:
    scenario = Scenario(
        dollars=10,
        round_dollars=5,
        areas={"deck": [card("two"), card("king", "S_K")]},
        action=Action.cash_out(),
    )
    assert_matches(scenario, ("phase", "dollars", "current_round"))


# ---------------------------------------------------------------------------
# Phase 5 red tests.  Strict XPASS means removing a bug requires removing the
# marker (and, where appropriate, adding engine-side internal observations).
# ---------------------------------------------------------------------------


def test_c01_hand_counter_observation_points() -> None:
    scenario = Scenario(
        areas={"hand": [card("p")], "deck": [card("drawn", "S_3")]},
        action=Action.play(0),
    )
    lua_trace = run_lua_oracle(scenario)
    engine_trace = run_engine(scenario)

    assert_traces_match(
        lua_trace,
        engine_trace,
        ("chips", "current_round", "hands_played", "last_hand_played", "hands"),
    )
    assert lua_trace.synchronous["hands_played"] == 0
    assert lua_trace.synchronous["current_round"]["hands_played"] == 0
    # NEW-P5-1-06 deliberately has no synchronous HAND_PLAYED observation in
    # this engine. tests/engine/test_play_sequencing.py pins that scoring sees
    # hand 0 and the stable post-step state advances both counters exactly once.
    assert engine_trace.final["hands_played"] == 1


def test_c03_vampire_before_pass() -> None:
    scenario = Scenario(
        areas={"hand": [card("mult-two", center="m_mult")]},
        jokers=[joker("vampire", "j_vampire")],
        action=Action.play(0),
    )
    assert_matches(scenario, ("chips",))


def test_c05_bull_reads_pending_gold_seal_money() -> None:
    scenario = Scenario(
        dollars=4,
        areas={"hand": [card("gold-two", seal="Gold")], "deck": [card("drawn", "S_3")]},
        jokers=[joker("bull", "j_bull")],
        action=Action.play(0),
    )
    lua_trace = run_lua_oracle(scenario)
    assert lua_trace.synchronous["dollars"] == 4
    assert lua_trace.synchronous["dollar_buffer"] == 3
    assert_traces_match(lua_trace, run_engine(scenario), ("chips",))


@pytest.mark.xfail(strict=True, reason="C09: House and Mark call stay_flipped on every draw")
@pytest.mark.parametrize("blind_key", ["bl_house", "bl_mark"])
def test_c09_face_down_boss_draw(blind_key: str) -> None:
    scenario = Scenario(
        blind=BlindSpec(blind_key),
        areas={"deck": [card("king", "S_K"), card("two", "H_2")]},
        action=Action.set_blind(blind_key),
    )
    assert_matches(scenario, ("areas.hand",))


@pytest.mark.xfail(
    strict=True,
    reason="C10: held-card Red seal and Mime repetitions run at end of round",
)
def test_c10_held_gold_card_retriggers() -> None:
    scenario = Scenario(
        chips=1_000,
        areas={"hand": [card("gold", center="m_gold", seal="Red")]},
        jokers=[joker("mime", "j_mime")],
        action=Action.end_round(),
    )
    assert_matches(scenario, ("dollars",))


@pytest.mark.xfail(
    strict=True,
    reason="C11: Mr. Bones requires 25 percent of accumulated blind chips",
)
def test_c11_mr_bones_below_threshold_does_not_save() -> None:
    scenario = Scenario(
        chips=7,
        hands_left=0,
        blind=BlindSpec("bl_small", {"chips": 300}),
        jokers=[joker("bones", "j_mr_bones")],
        action=Action.end_round(),
    )
    assert_matches(scenario, ("phase",))


def test_d12_blocked_hand_decays_ice_cream() -> None:
    scenario = Scenario(
        blind=BlindSpec("bl_psychic"),
        areas={"hand": [card("two")]},
        jokers=[joker("ice-cream", "j_ice_cream")],
        action=Action.play(0),
    )
    assert_matches(scenario, ("areas.jokers",))


@pytest.mark.xfail(strict=True, reason="D18: Rocket scales before its cash-out bonus is calculated")
def test_d18_rocket_boss_payout_uses_new_value() -> None:
    scenario = Scenario(
        chips=1_000,
        blind=BlindSpec("bl_hook"),
        jokers=[joker("rocket", "j_rocket")],
        action=Action.end_round(),
    )
    assert_matches(scenario, ("current_round.dollars",))


@pytest.mark.xfail(
    strict=True,
    reason="D22: rental is charged during end-of-round maintenance, including losses",
)
def test_d22_losing_round_charges_rental_immediately() -> None:
    scenario = Scenario(
        dollars=10,
        chips=0,
        jokers=[joker("rental", "j_joker", ability={"rental": True})],
        action=Action.end_round(),
    )
    assert_matches(scenario, ("dollars",))


@pytest.mark.xfail(strict=True, reason="D28: Crimson Heart rerolls only after a prepared play draw")
def test_d28_crimson_heart_does_not_reroll_on_discard() -> None:
    scenario = Scenario(
        blind=BlindSpec("bl_final_heart", {"prepped": False}),
        areas={
            "hand": [card("h1"), card("h2", "S_3")],
            "deck": [card("drawn", "D_4")],
        },
        jokers=[joker("left", "j_joker"), joker("right", "j_joker")],
        action=Action.discard(0),
    )
    assert_matches(scenario, ("pseudorandom.crimson_heart",))


def test_d29_ox_does_not_wipe_gold_seal_payout() -> None:
    scenario = Scenario(
        dollars=4,
        most_played_poker_hand="High Card",
        blind=BlindSpec("bl_ox"),
        areas={"hand": [card("gold-two", seal="Gold")], "deck": [card("drawn", "S_3")]},
        action=Action.play(0),
    )
    lua_trace = run_lua_oracle(scenario)
    assert lua_trace.synchronous["dollars"] == 0
    assert lua_trace.synchronous["dollar_buffer"] == 3
    assert_traces_match(lua_trace, run_engine(scenario), ("dollars",))


@pytest.mark.xfail(strict=True, reason="D30: boss defeat clears played_this_ante on playing cards")
def test_d30_pillar_marker_resets_at_new_ante() -> None:
    scenario = Scenario(
        chips=1_000,
        blind=BlindSpec("bl_pillar"),
        areas={"hand": [card("old-play", ability={"played_this_ante": True})]},
        action=Action.end_round(),
    )
    assert_matches(scenario, ("areas.deck",))


@pytest.mark.xfail(
    strict=True,
    reason="D31: Hook uses the discard pipeline, including Purple seals",
)
def test_d31_hook_forced_discard_fires_purple_seals() -> None:
    scenario = Scenario(
        blind=BlindSpec("bl_hook"),
        areas={
            "hand": [
                card("played"),
                card("purple-a", "S_3", seal="Purple"),
                card("purple-b", "D_4", seal="Purple"),
            ]
        },
        action=Action.play(0),
    )
    lua_trace, engine_trace = run_lua_oracle(scenario), run_engine(scenario)
    assert len(lua_trace.final["areas"]["consumeables"]) == len(
        engine_trace.final["areas"]["consumeables"]
    )


@pytest.mark.xfail(strict=True, reason="D32: Amber Acorn shuffles three times on the aajk stream")
def test_d32_amber_acorn_shuffle_count() -> None:
    scenario = Scenario(
        blind=BlindSpec("bl_final_acorn"),
        jokers=[
            joker("a", "j_joker"),
            joker("b", "j_bull"),
            joker("c", "j_mime"),
        ],
        action=Action.set_blind("bl_final_acorn"),
    )
    lua_trace = run_lua_oracle(scenario)
    assert all(card_state["facing"] == "back" for card_state in lua_trace.final["areas"]["jokers"])
    assert_traces_match(lua_trace, run_engine(scenario), ("pseudorandom.aajk",))


@pytest.mark.xfail(
    strict=True,
    reason="D32: boss defeat flips Jokers face-up and clears blind debuffs",
)
def test_d32_boss_defeat_restores_cards_and_jokers() -> None:
    scenario = Scenario(
        chips=1_000,
        blind=BlindSpec("bl_final_acorn"),
        areas={"hand": [card("debuffed", debuff=True)]},
        jokers=[joker("flipped", "j_joker", facing="back", debuff=True)],
        action=Action.end_round(),
    )
    lua_trace = run_lua_oracle(scenario)
    assert lua_trace.final["areas"]["jokers"][0]["facing"] == "front"
    assert lua_trace.final["areas"]["jokers"][0]["debuff"] is False
    assert lua_trace.final["areas"]["deck"][0]["debuff"] is False
    assert_traces_match(lua_trace, run_engine(scenario), ("areas.jokers",))


@pytest.mark.xfail(
    strict=True,
    reason="NEW-P5-O2-01: discard does not preserve gameplay ability.discarded marker",
)
def test_new_p5_o2_discard_marker_is_preserved() -> None:
    scenario = Scenario(
        areas={"hand": [card("discarded"), card("held", "S_3")]},
        action=Action.discard(0),
    )
    assert_matches(scenario, ("areas.discard",))
