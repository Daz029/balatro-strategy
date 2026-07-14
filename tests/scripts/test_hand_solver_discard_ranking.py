"""Tests for B7 -- joker/held-aware discard-branch ranking.

`rank_templates_cheaply` decides WHICH top-k templates the discard
recursion ever explores. The historical scorer was jokerless and
held-empty (`score_hand_base(eval_cards, [])`), which has the exact
failure shape the B5 play prescreen fixed: a joker-favored line (Greedy's
suit) or a held-favored branch (Baron's held Kings) ranks below its true
value and never reaches the exact hit/miss recursion. These tests pin:

  - a suit joker flips the template ranking toward its suit's flush line;
  - held cards retained by the discard cap (`kept`) contribute to the
    ranking value (Baron + a kept King);
  - `joker_aware=False` (the validation harness's comparison arm) and
    `jokers=None` (legacy callers) reproduce the old scorer exactly;
  - repeated calls with jokers do not corrupt history-dependent boss
    state (the stage-4 shared-blind bug class).

Only the RANKING scorer changed -- reachability math and the exact
recursion valuation are untouched (see the B7 spec in
docs/pre-regen-handoff.md).
"""

from __future__ import annotations

from hand_solver import DeckComposition, rank_templates_cheaply

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _card(suit: Suit, rank: Rank) -> Card:
    return create_playing_card(suit, rank)


def _full_deck_minus(hand: list[Card]) -> DeckComposition:
    all_cards = [create_playing_card(s, r) for s in Suit for r in Rank]
    held = {(c.base.id, c.base.suit.value) for c in hand}
    remaining = [c for c in all_cards if (c.base.id, c.base.suit.value) not in held]
    return DeckComposition.from_deck(remaining)


def _fixtures(seed: str = "b7") -> tuple[HandLevels, Blind, PseudoRandom]:
    return HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom(seed)


def _diamond_draw_vs_trips_hand() -> list[Card]:
    """4-to-a-diamond-flush plus complete spade trips: jokerless, the
    trips line outranks the flush draw; Greedy Joker (+3 mult per scored
    Diamond) makes the flush line the true best target."""
    return [
        _card(Suit.DIAMONDS, Rank.TWO),
        _card(Suit.DIAMONDS, Rank.THREE),
        _card(Suit.DIAMONDS, Rank.FOUR),
        _card(Suit.DIAMONDS, Rank.SIX),
        _card(Suit.SPADES, Rank.KING),
        _card(Suit.HEARTS, Rank.KING),
        _card(Suit.CLUBS, Rank.KING),
        _card(Suit.SPADES, Rank.NINE),
    ]


class TestJokerAwareTemplateRanking:
    def test_suit_joker_flips_ranking_toward_its_flush(self):
        hand = _diamond_draw_vs_trips_hand()
        deck = _full_deck_minus(hand)
        hand_levels, blind, rng = _fixtures()
        greedy = [create_joker("j_greedy_joker")]

        without = rank_templates_cheaply(
            hand, deck, hand_levels, blind, rng, jokers=[], top_k=4
        )
        with_greedy = rank_templates_cheaply(
            hand, deck, hand_levels, blind, rng, jokers=greedy, top_k=4
        )
        assert without[0][0].name != "flush_Diamonds"
        assert with_greedy[0][0].name == "flush_Diamonds"

    def test_kept_cards_contribute_to_ranking_value(self):
        # pair_3's discard cap keeps one King in hand (highest nominal of
        # six non-matches); Baron multiplies mult per HELD King, so the
        # pair_3 branch must rank higher with Baron than without.
        hand = [
            _card(Suit.SPADES, Rank.THREE),
            _card(Suit.CLUBS, Rank.THREE),
            _card(Suit.SPADES, Rank.KING),
            _card(Suit.HEARTS, Rank.KING),
            _card(Suit.DIAMONDS, Rank.SEVEN),
            _card(Suit.CLUBS, Rank.SIX),
            _card(Suit.HEARTS, Rank.FIVE),
            _card(Suit.DIAMONDS, Rank.FOUR),
        ]
        deck = _full_deck_minus(hand)
        hand_levels, blind, rng = _fixtures()

        def pair3_value(jokers: list[Card]) -> float:
            results = rank_templates_cheaply(
                hand, deck, hand_levels, blind, rng, jokers=jokers, top_k=50
            )
            for template, _hold, kept, _discard, _p, value, _needed in results:
                if template.name == "pair_3":
                    assert any(c.base.id == 13 for c in kept), (
                        "test premise: the discard cap must keep a King in hand"
                    )
                    return value
            raise AssertionError("pair_3 template missing from ranking")

        assert pair3_value([create_joker("j_baron")]) > pair3_value([])


class TestOldScorerEscapeHatch:
    def _outputs(self, **kwargs):
        hand = _diamond_draw_vs_trips_hand()
        deck = _full_deck_minus(hand)
        hand_levels, blind, rng = _fixtures()
        results = rank_templates_cheaply(hand, deck, hand_levels, blind, rng, **kwargs)
        return [(t.name, p, v) for t, _h, _k, _d, p, v, _n in results]

    def test_joker_aware_false_matches_legacy_none(self):
        greedy = [create_joker("j_greedy_joker")]
        assert self._outputs(jokers=greedy, joker_aware=False) == self._outputs(jokers=None)

    def test_legacy_positional_call_unchanged(self):
        # Old call shape (no joker kwargs at all) must keep working.
        assert self._outputs() == self._outputs(jokers=None)


class TestStateSafety:
    def test_repeated_calls_identical_on_history_boss_with_jokers(self):
        hand = _diamond_draw_vs_trips_hand()
        deck = _full_deck_minus(hand)
        hand_levels = HandLevels()
        blind = Blind.create("bl_eye", ante=2)
        rng = PseudoRandom("b7-eye")
        greedy = [create_joker("j_greedy_joker")]

        def snap():
            results = rank_templates_cheaply(
                hand, deck, hand_levels, blind, rng, jokers=greedy, top_k=8
            )
            return [(t.name, p, v) for t, _h, _k, _d, p, v, _n in results]

        assert snap() == snap()
