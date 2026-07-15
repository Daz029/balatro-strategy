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

import numpy as np

from hand_solver import (
    DeckComposition,
    best_immediate_play,
    rank_templates_cheaply,
    solve_hand_turn,
)

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


class TestFullSolverExistenceProof:
    """The RANKING-flip tests above prove the box reorders; these prove the
    flip changes the EMITTED discard through the whole `solve_hand_turn`
    recursion AND that the new discard is genuinely better -- judged by a
    FAITHFUL Monte Carlo model, because the solver's own value model cannot
    see it. A still_needed==0 line (discard around trip Kings) refills to a
    monster via optimistic filler and saturates at p_clear=1.0, so the solver
    ties it with Greedy's flush line even though real draws make the flush
    strictly better. This is the whole reason B7 needs an external arbiter."""

    def _discard_ids(self, choice) -> frozenset[int]:
        return frozenset(id(c) for c in choice.discard)

    def _emit(self, hand, jokers, deck, hand_levels, blind, rng, chips, *, top_k, discards_left, joker_aware):
        return solve_hand_turn(
            hand, jokers, hand_levels, blind, rng, deck, chips,
            1, discards_left, [], None, 0,
            top_k=top_k, hand_size=len(hand), joker_aware=joker_aware,
        )

    def test_greedy_flip_is_faithfully_better_d1(self):
        from validate_discard_ranking import (
            _action_key,
            _deck_pool,
            _value_at,
            faithful_totals,
        )

        hand = _diamond_draw_vs_trips_hand()
        jokers = [create_joker("j_greedy_joker")]
        deck = _full_deck_minus(hand)
        hand_levels, blind, rng = _fixtures()
        play_now = best_immediate_play(hand, jokers, hand_levels, blind, rng)[1].total
        chips = play_now * 1.5  # above play-now -> the label must be a discard

        old = self._emit(hand, jokers, deck, hand_levels, blind, rng, chips, top_k=1, discards_left=1, joker_aware=False)
        new = self._emit(hand, jokers, deck, hand_levels, blind, rng, chips, top_k=1, discards_left=1, joker_aware=True)

        # B7 flips the emitted discard toward Greedy's flush line.
        assert new.template_name == "flush_Diamonds"
        assert old.template_name != "flush_Diamonds"
        assert self._discard_ids(new) != self._discard_ids(old)

        # The solver's own model does NOT prefer the new action (optimistic
        # refill saturates both) -- so faithful judgment is necessary.
        assert new.p_clear <= old.p_clear + 1e-9

        # Faithful MC over real draws: the flush line clears strictly more
        # often than the trips-refill line somewhere above play-now.
        pool = _deck_pool(deck)

        def totals(choice, tag):
            return faithful_totals(
                hand=hand, discard_key=_action_key(hand, choice.discard), jokers=jokers,
                hand_levels=hand_levels, blind=blind, rng=rng, pool=pool,
                game_state=None, blind_chips=0, n_samples=200, mc_seed=f"ex:{tag}",
            )

        new_tot, old_tot = totals(new, "new"), totals(old, "old")
        pooled = new_tot + old_tot
        goals = [g for g in np.quantile(pooled, [0.5, 0.7, 0.85, 0.95]) if g > play_now]
        gaps = [_value_at(new_tot, g) - _value_at(old_tot, g) for g in goals]
        assert max(gaps) > 0.05, f"faithful gaps {gaps} do not favor the flush line"

    def test_greedy_flip_threads_through_recursion_d2(self):
        # Two discards remaining: the joker_aware flag must reach the deeper
        # shortlist cut, not just the root, for new to still emit the flush.
        hand = _diamond_draw_vs_trips_hand()
        jokers = [create_joker("j_greedy_joker")]
        deck = _full_deck_minus(hand)
        hand_levels, blind, rng = _fixtures()
        play_now = best_immediate_play(hand, jokers, hand_levels, blind, rng)[1].total
        new = self._emit(hand, jokers, deck, hand_levels, blind, rng, play_now * 1.5, top_k=1, discards_left=2, joker_aware=True)
        assert new.action == "discard"
        assert new.template_name == "flush_Diamonds"


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
