"""Line-order regressions for play_cards_from_highlighted/evaluate_play (P5-2)."""

from __future__ import annotations

import random

import pytest

from jackdaw.engine import lifecycle
from jackdaw.engine.actions import PlayHand, SelectBlind
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.effects import CreateCard, EaseDollars
from jackdaw.engine.game import step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand
from tests.engine._rollout import drive


def _owned_joker(gs: dict, key: str):
    joker = create_joker(key, game_state=gs)
    lifecycle.emplace(gs, joker, "jokers")
    return joker


def _single_card_play(
    seed: str,
    *,
    rank: Rank = Rank.TWO,
    suit: Suit = Suit.HEARTS,
    center: str = "c_base",
    seal: str | None = None,
    blind_key: str | None = None,
) -> tuple[dict, object]:
    gs = initialize_run("b_red", 1, seed)
    step(gs, SelectBlind())
    if blind_key is not None:
        gs["blind"] = Blind.create(blind_key, ante=1)
    card = create_playing_card(suit, rank, enhancement=center, game_state=gs)
    if seal is not None:
        card.set_seal(gs, seal)
    gs["deck"].extend(gs["hand"])
    gs["hand"] = [card]
    gs["blind"].chips = 10**9
    return gs, card


def _score(
    played,
    held,
    jokers,
    *,
    gs: dict | None = None,
    blind: Blind | None = None,
    levels: HandLevels | None = None,
    seed: str = "P52_SCORE",
):
    state = gs if gs is not None else {}
    active_blind = blind or Blind.create("bl_small", ante=1)
    hand_levels = levels or HandLevels()
    state.setdefault("jokers", jokers)
    state.setdefault("blind", active_blind)
    state.setdefault("hand_levels", hand_levels)
    state.setdefault("current_round", {"hands_left": 3, "hands_played": 0})
    state.setdefault("dollars", 0)
    state.setdefault("dollar_buffer", 0)
    return score_hand(
        played,
        held,
        jokers,
        hand_levels,
        active_blind,
        PseudoRandom(seed),
        game_state=state,
        blind_chips=active_blind.chips,
    )


def test_c01_hand_stats_are_recorded_once_before_first_hand_jokers() -> None:
    # state_events.lua:574-578 records the hand once before the blind branch.
    gs, _ = _single_card_play("P52_C01")
    dna = _owned_joker(gs, "j_dna")
    sixth = _owned_joker(gs, "j_sixth_sense")
    del dna, sixth

    step(gs, PlayHand(card_indices=(0,)))

    high_card = gs["hand_levels"]["High Card"]
    assert high_card.played == 1
    assert high_card.played_this_round == 1
    assert gs["hands_played"] == 1
    assert gs["current_round"]["hands_played"] == 1
    # DNA proves that the before pass observed hand 0; it copies only there.
    assert len(gs["hand"]) >= 1

    sixth_gs, six = _single_card_play("P52_C01_SIXTH", rank=Rank.SIX)
    _owned_joker(sixth_gs, "j_sixth_sense")
    step(sixth_gs, PlayHand(card_indices=(0,)))
    assert six.removed is True
    assert len(sixth_gs["consumables"]) == 1

    # state_events.lua:578 makes even an unlevelled secret hand visible.
    visible_gs = initialize_run("b_red", 1, "P52_C01_VISIBLE")
    step(visible_gs, SelectBlind())
    suits = (Suit.HEARTS, Suit.SPADES, Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS)
    visible_gs["deck"].extend(visible_gs["hand"])
    visible_gs["hand"] = [
        create_playing_card(suit, Rank.TWO, game_state=visible_gs) for suit in suits
    ]
    visible_gs["blind"].chips = 10**9

    step(visible_gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))

    five = visible_gs["hand_levels"]["Five of a Kind"]
    assert five.played == 1
    assert five.visible is True


def test_c03_vampire_strips_before_scoring_and_multiplies_once() -> None:
    # card.lua:3443-3487: strip in before; main XMult is applied later once.
    card = create_playing_card(Suit.HEARTS, Rank.TWO, enhancement="m_mult")
    vampire = create_joker("j_vampire")

    result = _score([card], [], [vampire])

    assert result.total == 7
    assert card.center_key == "c_base"
    assert not getattr(card, "vampired", False)


@pytest.mark.parametrize(
    ("joker_key", "expected_total"),
    [("j_bull", 21), ("j_bootstraps", 21)],
)
def test_c05_buffer_consumers_see_gold_seal_money(joker_key: str, expected_total: int) -> None:
    # card.lua:3936-3939,4046-4049 read dollars + dollar_buffer.
    gs, _ = _single_card_play("P52_BUFFER_" + joker_key, seal="Gold")
    _owned_joker(gs, joker_key)
    gs["dollars"] = 4

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["last_score_result"].total == expected_total
    assert gs["dollars"] == 7
    assert gs["dollar_buffer"] == 0


def test_c05_vagabond_reads_committed_money_but_sees_ox_instant_drain() -> None:
    # card.lua:3744 reads committed dollars; blind.lua:565 changes that view.
    ordinary, _ = _single_card_play("P52_VAG_ORDINARY", seal="Gold")
    _owned_joker(ordinary, "j_vagabond")
    ordinary["dollars"] = 5
    step(ordinary, PlayHand(card_indices=(0,)))
    assert ordinary["dollars"] == 8
    assert ordinary["consumables"] == []

    ox, _ = _single_card_play("P52_VAG_OX", seal="Gold", blind_key="bl_ox")
    _owned_joker(ox, "j_vagabond")
    ox["dollars"] = 5
    ox["current_round"]["most_played_poker_hand"] = "High Card"
    step(ox, PlayHand(card_indices=(0,)))
    assert ox["dollars"] == 3
    assert len(ox["consumables"]) == 1


@pytest.mark.parametrize(
    ("producer", "expected"),
    [
        ("gold_seal", 3),
        ("lucky", 20),
        ("golden_ticket", 4),
        ("business", 2),
        ("rough_gem", 1),
        ("reserved_parking", 1),
        ("to_do_list", 4),
    ],
)
def test_c05_scoring_money_producers_emit_buffered_effects(
    producer: str,
    expected: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # card.lua:1084-1087,3150-3183,3224-3231,3302-3317,3491-3499.
    monkeypatch.setattr(PseudoRandom, "random", lambda _self, _key: 0.0)
    played = create_playing_card(Suit.DIAMONDS, Rank.KING)
    held = []
    jokers = []
    if producer == "gold_seal":
        played.set_seal({}, "Gold")
    elif producer == "lucky":
        played.set_ability("m_lucky", gs={})
    elif producer == "golden_ticket":
        played.set_ability("m_gold", gs={})
        jokers = [create_joker("j_ticket")]
    elif producer == "business":
        jokers = [create_joker("j_business")]
    elif producer == "rough_gem":
        jokers = [create_joker("j_rough_gem")]
    elif producer == "reserved_parking":
        played = create_playing_card(Suit.HEARTS, Rank.TWO)
        held = [create_playing_card(Suit.SPADES, Rank.KING)]
        jokers = [create_joker("j_reserved_parking")]
    else:
        todo = create_joker("j_todo_list")
        todo.ability["extra"]["poker_hand"] = "High Card"
        played = create_playing_card(Suit.HEARTS, Rank.TWO)
        jokers = [todo]

    result = _score(played=[played], held=held, jokers=jokers, seed="P52_" + producer)
    payouts = [
        effect
        for effect in result.effects
        if isinstance(effect, EaseDollars) and not effect.instant
    ]

    assert sum(effect.amount for effect in payouts) == expected
    assert all(effect.buffered for effect in payouts)
    assert result.dollars_earned == expected


def test_d12_blocked_hand_still_runs_after_pass() -> None:
    # state_events.lua:1068-1075 is outside the allowed/blocked branch.
    gs, _ = _single_card_play("P52_D12", blind_key="bl_psychic")
    ice_cream = _owned_joker(gs, "j_ice_cream")
    before = ice_cream.ability["extra"]["chips"]

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["last_score_result"].debuffed is True
    assert ice_cream.ability["extra"]["chips"] == before - 5


def test_d29_ox_uses_stored_most_played_not_the_hand_becoming_most_played() -> None:
    # blind.lua:562 compares current_round.most_played_poker_hand.
    gs, _ = _single_card_play("P52_D29_STORED", blind_key="bl_ox")
    gs["dollars"] = 9
    gs["current_round"]["most_played_poker_hand"] = "Pair"

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["blind"].triggered is False
    assert gs["dollars"] == 9


def test_d29_ox_drain_precedes_scoring_and_gold_money_survives() -> None:
    # blind.lua:560-567 drains instantly; state_events.lua:719-724 pays later.
    gs, _ = _single_card_play("P52_D29_GOLD", seal="Gold", blind_key="bl_ox")
    gs["dollars"] = 9
    gs["current_round"]["most_played_poker_hand"] = "High Card"

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["blind"].triggered is True
    assert gs["dollars"] == 3


def test_d29_matador_main_pass_sees_ox_trigger() -> None:
    # card.lua:3719-3728 reads blind.triggered in the main pass.
    gs = {
        "dollars": 9,
        "dollar_buffer": 0,
        "current_round": {
            "hands_left": 3,
            "hands_played": 0,
            "most_played_poker_hand": "High Card",
        },
    }
    blind = Blind.create("bl_ox", ante=1)
    matador = create_joker("j_matador", game_state=gs)
    played = create_playing_card(Suit.HEARTS, Rank.TWO, game_state=gs)

    result = _score([played], [], [matador], gs=gs, blind=blind)

    assert blind.triggered is True
    assert result.dollars_earned == 8
    assert any(
        isinstance(effect, EaseDollars) and effect.amount == -9 and effect.instant
        for effect in result.effects
    )


def test_dna_copy_scores_as_a_held_card_in_the_same_hand() -> None:
    # card.lua:3501-3511 emplaces the copy before the held-card loop at 782.
    gs, played = _single_card_play("P52_DNA_HELD", center="m_steel")
    _owned_joker(gs, "j_dna")

    step(gs, PlayHand(card_indices=(0,)))

    assert gs["last_score_result"].total == 10  # floor((5 + 2) * 1.5)
    assert any(card is not played and card.center_key == "m_steel" for card in gs["hand"])


@pytest.mark.parametrize("consumer", ["j_bull", "j_bootstraps", "j_vagabond"])
def test_money_consumers_leave_live_solver_state_untouched(consumer: str) -> None:
    live = {
        "dollars": 4 if consumer != "j_vagabond" else 5,
        "dollar_buffer": 0,
        "last_hand_played": "Pair",
        "consumables": [],
        "consumable_slots": 2,
        "current_round": {
            "hands_left": 3,
            "hands_played": 0,
            "most_played_poker_hand": "High Card",
        },
    }
    joker = create_joker(consumer, game_state=live)
    played = create_playing_card(Suit.HEARTS, Rank.TWO, game_state=live)
    played.set_seal(live, "Gold")
    blind = Blind.create("bl_ox" if consumer == "j_vagabond" else "bl_small", ante=1)
    before = (
        live["dollars"],
        live["dollar_buffer"],
        list(live["consumables"]),
        live["last_hand_played"],
    )

    result = _score([played], [], [joker], gs=live, blind=blind, seed="P52_ISO_" + consumer)

    assert (
        live["dollars"],
        live["dollar_buffer"],
        live["consumables"],
        live["last_hand_played"],
    ) == before
    if consumer == "j_bull":
        assert result.total == 21
    elif consumer == "j_bootstraps":
        assert result.total == 21
    else:
        assert any(isinstance(effect, CreateCard) for effect in result.effects)


def test_dollar_buffer_is_zero_after_every_seeded_rollout_step() -> None:
    # The pass-local ledger must never leak into a stable engine state.
    payout_gs, _ = _single_card_play("P52_LEDGER_PAYOUT", seal="Gold")
    _owned_joker(payout_gs, "j_bull")
    payout_gs["dollars"] = 4
    step(payout_gs, PlayHand(card_indices=(0,)))
    assert payout_gs["last_score_result"].total == 21
    assert payout_gs["dollar_buffer"] == 0

    for gs, _action in drive("P52_LEDGER", random.Random(52), max_steps=80):
        assert gs["dollar_buffer"] == 0
