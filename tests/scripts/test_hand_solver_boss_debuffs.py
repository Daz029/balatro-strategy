"""Solver-level boss debuff correctness (the stage-4 boss curriculum's
prerequisite: CLAUDE.md's "verify the solver handles boss debuffs through
score_hand"). ``Blind.debuff_hand``/``modify_hand`` are already unit-tested
directly (tests/engine/test_blind.py, tests/engine/test_scoring.py) -- these
tests instead cover the solver's own call sites into score_hand
(``evaluate_value``, ``rank_templates_cheaply``), confirming boss debuffs are
correctly reflected in what the solver actually sees and decide on, and that
history-dependent bosses (The Eye, The Mouth) don't get corrupted by the
solver's many hypothetical evaluations of candidates it never chooses (see
tests/scripts/test_hand_solver_mutation.py for the underlying mutation-safety
regression this depends on).
"""

from __future__ import annotations

from hand_solver import DeckComposition, evaluate_value, rank_templates_cheaply

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _setup(blind_key: str, ante: int = 1) -> tuple[HandLevels, Blind, PseudoRandom]:
    return HandLevels(), Blind.create(blind_key, ante=ante), PseudoRandom("BOSS_DEBUFF_TEST")


def _full_deck_minus(hand: list) -> DeckComposition:
    all_cards = [create_playing_card(s, r) for s in Suit for r in Rank]
    held = {(c.base.id, c.base.suit.value) for c in hand}
    remaining = [c for c in all_cards if (c.base.id, c.base.suit.value) not in held]
    return DeckComposition.from_deck(remaining)


class TestPsychicMinimumCardCount:
    def test_blocks_hand_under_five_cards(self) -> None:
        hand_levels, blind, rng = _setup("bl_psychic")
        played = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.SPADES, Rank.KING),
        ]
        result = evaluate_value(played, [], [], hand_levels, blind, rng)
        assert result.debuffed is True
        assert result.total == 0

    def test_allows_five_or_more_cards(self) -> None:
        hand_levels, blind, rng = _setup("bl_psychic")
        played = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.SPADES, Rank.KING),
            create_playing_card(Suit.CLUBS, Rank.NINE),
            create_playing_card(Suit.DIAMONDS, Rank.SEVEN),
            create_playing_card(Suit.HEARTS, Rank.FIVE),
        ]
        result = evaluate_value(played, [], [], hand_levels, blind, rng)
        assert result.debuffed is False
        assert result.total > 0


class TestFlintHalving:
    def test_halves_score_relative_to_small_blind(self) -> None:
        played = [
            create_playing_card(Suit.HEARTS, Rank.TWO),
            create_playing_card(Suit.SPADES, Rank.TWO),
        ]
        small_hl, small_blind, small_rng = _setup("bl_small")
        flint_hl, flint_blind, flint_rng = _setup("bl_flint")

        small_result = evaluate_value(
            list(played), [], [], small_hl, small_blind, small_rng, search_orderings=False
        )
        flint_result = evaluate_value(
            list(played), [], [], flint_hl, flint_blind, flint_rng, search_orderings=False
        )

        assert small_result.debuffed is False
        assert flint_result.debuffed is False
        assert 0 < flint_result.total < small_result.total


class TestEyeBlocksAlreadyUsedHandType:
    def test_pre_marked_type_is_blocked(self) -> None:
        hand_levels, blind, rng = _setup("bl_eye")
        blind.hands_used["Pair"] = True

        pair = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.SPADES, Rank.ACE),
        ]
        result = evaluate_value(pair, [], [], hand_levels, blind, rng, search_orderings=False)
        assert result.debuffed is True
        assert result.total == 0

    def test_unused_type_is_not_blocked(self) -> None:
        hand_levels, blind, rng = _setup("bl_eye")
        blind.hands_used["Pair"] = True

        two_pair = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.SPADES, Rank.ACE),
            create_playing_card(Suit.HEARTS, Rank.KING),
            create_playing_card(Suit.SPADES, Rank.KING),
        ]
        result = evaluate_value(
            two_pair, [], [], hand_levels, blind, rng, search_orderings=False
        )
        assert result.debuffed is False
        assert result.total > 0


class TestMouthBlocksMismatchedHandType:
    def test_mismatched_type_is_blocked(self) -> None:
        hand_levels, blind, rng = _setup("bl_mouth")
        blind.only_hand = "Flush"

        pair = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.SPADES, Rank.ACE),
        ]
        result = evaluate_value(pair, [], [], hand_levels, blind, rng, search_orderings=False)
        assert result.debuffed is True
        assert result.total == 0

    def test_matching_type_is_not_blocked(self) -> None:
        hand_levels, blind, rng = _setup("bl_mouth")
        blind.only_hand = "Pair"

        pair = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.SPADES, Rank.ACE),
        ]
        result = evaluate_value(pair, [], [], hand_levels, blind, rng, search_orderings=False)
        assert result.debuffed is False
        assert result.total > 0


class TestRankTemplatesCheaplyDoesNotCorruptSharedBlind:
    def test_eye_history_untouched_after_ranking_many_candidates(self) -> None:
        # A hand with both pair and flush potential -- multiple templates
        # get cheaply scored (hypothetically) in one rank_templates_cheaply
        # call, none of which is an actually-played hand.
        hand = [
            create_playing_card(Suit.HEARTS, Rank.ACE),
            create_playing_card(Suit.HEARTS, Rank.KING),
            create_playing_card(Suit.HEARTS, Rank.NINE),
            create_playing_card(Suit.HEARTS, Rank.SIX),
            create_playing_card(Suit.SPADES, Rank.ACE),
            create_playing_card(Suit.CLUBS, Rank.FOUR),
            create_playing_card(Suit.DIAMONDS, Rank.TWO),
            create_playing_card(Suit.CLUBS, Rank.SEVEN),
        ]
        hand_levels, blind, rng = _setup("bl_eye")
        deck = _full_deck_minus(hand)

        results = rank_templates_cheaply(hand, deck, hand_levels, blind, rng)

        assert results, "expected at least one candidate template"
        assert any(r[5] > 0 for r in results), "every candidate scored 0 -- likely corrupted"
        assert blind.hands_used == {}, "cheap ranking must never touch the real Blind"
