"""Regression test: `evaluate_value` must never mutate the caller's real
cards or jokers.

`score_hand` mutates card/joker state in place for several effects (Vampire
strips enhancements off scored cards, Wee Joker/Lucky Cat accumulate onto
`ability["extra"]`/`ability["x_mult"]`, Lucky Card sets a `lucky_trigger`
flag). Since a solver evaluates many hypothetical plays that are never
actually taken, any of these leaking onto the real hand/joker objects would
silently corrupt the live run.
"""

from __future__ import annotations

import copy

from hand_solver import evaluate_value

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _setup():
    hand_levels = HandLevels()
    blind = Blind.create("bl_small", ante=1)
    rng = PseudoRandom("MUTATION_CHECK")
    return hand_levels, blind, rng


def test_vampire_does_not_strip_caller_enhancements() -> None:
    played = [
        create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_glass"),
        create_playing_card(Suit.SPADES, Rank.KING, enhancement="m_mult"),
    ]
    jokers = [create_joker("j_vampire")]
    played_snapshot = copy.deepcopy(played)
    jokers_snapshot = copy.deepcopy(jokers)

    hand_levels, blind, rng = _setup()
    evaluate_value(played, [], jokers, hand_levels, blind, rng)

    for before, after in zip(played_snapshot, played):
        assert after.ability == before.ability, "played card ability mutated"
        assert getattr(after, "vampired", False) == getattr(before, "vampired", False)
    for before, after in zip(jokers_snapshot, jokers):
        assert after.ability == before.ability, "joker ability mutated"


def test_lucky_card_trigger_flag_not_left_on_caller_card() -> None:
    played = [create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_lucky")]
    played_snapshot = copy.deepcopy(played)

    hand_levels, blind, rng = _setup()
    evaluate_value(played, [], [], hand_levels, blind, rng)

    assert played[0].ability == played_snapshot[0].ability
    assert not getattr(played[0], "lucky_trigger", False)


def test_wee_joker_and_lucky_cat_do_not_accumulate_on_caller_joker() -> None:
    played = [
        create_playing_card(Suit.HEARTS, Rank.TWO),
        create_playing_card(Suit.SPADES, Rank.TWO),
    ]
    jokers = [create_joker("j_wee"), create_joker("j_lucky_cat")]
    jokers_snapshot = copy.deepcopy(jokers)

    hand_levels, blind, rng = _setup()
    evaluate_value(played, [], jokers, hand_levels, blind, rng)

    for before, after in zip(jokers_snapshot, jokers):
        assert after.ability == before.ability, "joker ability mutated"


def test_eye_history_not_corrupted_by_hypothetical_evals() -> None:
    """Regression: score_hand mutates Blind.hands_used on EVERY call (The
    Eye), hypothetical or not. Before fast_clone_blind existed, two
    unrelated hypothetical evaluations of the same hand type -- neither
    ever actually played -- would corrupt each other: the second would
    incorrectly read as already-used and score 0."""
    hand_levels = HandLevels()
    blind = Blind.create("bl_eye", ante=1)
    rng = PseudoRandom("EYE_MUTATION_CHECK")

    pair_a = [
        create_playing_card(Suit.HEARTS, Rank.ACE),
        create_playing_card(Suit.SPADES, Rank.ACE),
    ]
    pair_b = [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.KING),
    ]

    first = evaluate_value(pair_a, [], [], hand_levels, blind, rng, search_orderings=False)
    second = evaluate_value(pair_b, [], [], hand_levels, blind, rng, search_orderings=False)

    assert first.debuffed is False and first.total > 0
    assert second.debuffed is False and second.total > 0
    assert blind.hands_used == {}, "hypothetical evals must never touch the real Blind"


def test_mouth_history_not_corrupted_by_hypothetical_evals() -> None:
    """Same regression as the Eye test above, for The Mouth's only_hand."""
    hand_levels = HandLevels()
    blind = Blind.create("bl_mouth", ante=1)
    rng = PseudoRandom("MOUTH_MUTATION_CHECK")

    pair = [
        create_playing_card(Suit.HEARTS, Rank.ACE),
        create_playing_card(Suit.SPADES, Rank.ACE),
    ]
    three_of_a_kind = [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.KING),
    ]

    first = evaluate_value(pair, [], [], hand_levels, blind, rng, search_orderings=False)
    second = evaluate_value(
        three_of_a_kind, [], [], hand_levels, blind, rng, search_orderings=False
    )

    assert first.debuffed is False and first.total > 0
    assert second.debuffed is False and second.total > 0
    assert blind.only_hand is None, "hypothetical evals must never touch the real Blind"


def test_repeated_evaluation_is_idempotent() -> None:
    """Calling evaluate_value twice on the same objects must give the same
    result -- if state leaked across calls, the second call would see
    already-accumulated joker/card state from the first."""
    played = [
        create_playing_card(Suit.HEARTS, Rank.TWO),
        create_playing_card(Suit.SPADES, Rank.TWO),
        create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_glass"),
    ]
    jokers = [create_joker("j_wee"), create_joker("j_vampire")]

    hand_levels, blind, rng = _setup()
    first = evaluate_value(
        copy.deepcopy(played), [], copy.deepcopy(jokers), copy.deepcopy(hand_levels), blind, rng
    )
    second = evaluate_value(
        copy.deepcopy(played), [], copy.deepcopy(jokers), copy.deepcopy(hand_levels), blind, rng
    )
    assert first.total == second.total
