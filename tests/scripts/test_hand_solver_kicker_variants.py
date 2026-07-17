"""Tests for the K1 kicker-variant emitter in the big-hand play prescreen.

The prescreen proposed the right scoring LINE and the wrong KICKERS.
Measured at n=8 on stage2's joker-dense pool: 0.845 score-capture, misses
up to 90% of the play's value, and INVARIANT in k (27/27 misses had the
right line at true_size 5) -- so the candidate GENERATOR was starved and no
k rescues it. `_kicker_pad`'s keep-priority nominal-best padding assumed
kickers are inert filler, which is false under Splash (kickers score) and
Raised Fist (the cards you RETAIN set the mult).

K1 turns the pad into a hypothesis-gated variant emitter. A variant is the
GREEDY argmax completion of a line under ONE hypothesis about where kicker
value lives; there are no magnitudes anywhere, because the exact pass
arbitrates. These tests pin the gates, the four hypotheses, the
config-derived held-enhancement test, and the redefined family key.

Full measurement record: docs/bruteforce_speedup_and_kicker_design.md.
Decision record: CLAUDE.md, "Kicker variants + prescreen-at-n=8".
"""

from __future__ import annotations

import itertools

import hand_solver
from hand_solver import best_immediate_play, evaluate_value, prescreen_play_candidates

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_eval import get_hand_eval_flags
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _fixtures() -> tuple[HandLevels, Blind, PseudoRandom]:
    return HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom("kicker")


def _enhanced(suit: Suit, rank: Rank, enhancement: str) -> Card:
    """Enhancement via the engine's own `set_ability`, so held-channel
    config (h_x_mult / h_dollars) actually populates. Poking `center_key`
    directly -- as the sibling prescreen suite's `_card` helper does --
    leaves `ability` untouched, and every held test here would pass
    vacuously."""
    c = create_playing_card(suit, rank)
    c.set_ability(enhancement)
    return c


def _flush_plus_junk_hand() -> list[Card]:
    """10 cards: a complete heart flush plus low offsuit junk."""
    return [
        create_playing_card(Suit.HEARTS, Rank.ACE),
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.HEARTS, Rank.NINE),
        create_playing_card(Suit.HEARTS, Rank.SEVEN),
        create_playing_card(Suit.HEARTS, Rank.FOUR),
        create_playing_card(Suit.SPADES, Rank.TWO),
        create_playing_card(Suit.CLUBS, Rank.THREE),
        create_playing_card(Suit.DIAMONDS, Rank.FIVE),
        create_playing_card(Suit.SPADES, Rank.SIX),
        create_playing_card(Suit.CLUBS, Rank.EIGHT),
    ]


def _quads_and_flush_hand() -> list[Card]:
    """11 cards: four Kings AND five non-King hearts -- two strong families."""
    return [
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.KING),
        create_playing_card(Suit.DIAMONDS, Rank.KING),
        create_playing_card(Suit.HEARTS, Rank.QUEEN),
        create_playing_card(Suit.HEARTS, Rank.TEN),
        create_playing_card(Suit.HEARTS, Rank.EIGHT),
        create_playing_card(Suit.HEARTS, Rank.SIX),
        create_playing_card(Suit.HEARTS, Rank.THREE),
        create_playing_card(Suit.SPADES, Rank.FOUR),
        create_playing_card(Suit.CLUBS, Rank.SEVEN),
    ]


def _brute_force_best(hand: list[Card], jokers: list[Card]) -> float:
    """The pre-prescreen behaviour: exact evaluation of every subset."""
    hand_levels, blind, rng = _fixtures()
    best = None
    for size in range(1, min(5, len(hand)) + 1):
        for combo in itertools.combinations(hand, size):
            combo_ids = {id(c) for c in combo}
            held = [c for c in hand if id(c) not in combo_ids]
            result = evaluate_value(list(combo), held, jokers, hand_levels, blind, rng)
            if best is None or result.total > best.total:
                best = result
    assert best is not None
    return best.total


def _gates_for(hand: list[Card], jokers: list[Card]) -> hand_solver._KickerGates:
    flags = get_hand_eval_flags(jokers)
    views = hand_solver._resolved_joker_views(jokers)
    counts = hand_solver._card_channel_counts(hand, views, {}, flags)
    return hand_solver._kicker_gates(hand, views, flags, counts)


class TestGates:
    def test_plain_board_gates_all_off(self):
        """No Splash, no held-channel joker, no held enhancement -> every
        hypothesis collapses onto nominal-best, so the fix costs a plain
        board nothing. This is what keeps the budget adaptive, not 3x flat."""
        gates = _gates_for(_flush_plus_junk_hand(), [])
        assert not gates.scored_value
        assert not gates.held_value
        assert not gates.play_away_lowest

    def test_splash_opens_scored_value_gate(self):
        """Splash is the only way a non-line card scores. It is class-3
        (all-zero) in the trigger matrix BY DESIGN, so the gate reads
        `get_hand_eval_flags` -- the matrix cannot answer this."""
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_splash")])
        assert gates.scored_value

    def test_raised_fist_opens_play_away_gate(self):
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_raised_fist")])
        assert gates.play_away_lowest

    def test_held_channel_joker_opens_held_gate(self):
        """Baron reads HELD Kings -- the taxonomy's held channel, which is
        what makes hypothesis 3 principled rather than a joker wishlist."""
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_baron")])
        assert gates.held_value

    def test_steel_and_gold_open_held_gate_without_any_joker(self):
        for enhancement in ("m_steel", "m_gold"):
            hand = _flush_plus_junk_hand()[:-1] + [
                _enhanced(Suit.CLUBS, Rank.EIGHT, enhancement)
            ]
            assert _gates_for(hand, []).held_value, enhancement

    def test_debuffed_joker_gates_nothing(self):
        fist = create_joker("j_raised_fist")
        fist.debuff = True
        assert not _gates_for(_flush_plus_junk_hand(), [fist]).play_away_lowest


class TestHeldEnhancementDetection:
    """The held test is read from the engine's own config
    (h_x_mult / h_mult / h_dollars), never a {"Steel Card", "Gold Card"}
    name set -- which would be correct on today's content and rot the
    moment an enhancement is added or rebalanced."""

    def test_steel_detected_via_h_x_mult(self):
        steel = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_steel")
        assert steel.get_chip_h_x_mult() > 0
        assert hand_solver._has_held_enhancement(steel)

    def test_gold_detected_via_h_dollars(self):
        """Gold has no accessor -- its held payout is a raw ability read."""
        gold = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_gold")
        assert gold.ability.get("h_dollars", 0) > 0
        assert hand_solver._has_held_enhancement(gold)

    def test_scored_channel_enhancement_has_no_held_value(self):
        """Glass is x_mult when PLAYED -- worth nothing held, so it must
        stay fair game as a kicker."""
        glass = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_glass")
        assert not hand_solver._has_held_enhancement(glass)

    def test_plain_card_has_no_held_value(self):
        assert not hand_solver._has_held_enhancement(
            create_playing_card(Suit.CLUBS, Rank.EIGHT)
        )

    def test_debuffed_steel_has_no_held_value(self):
        """The accessors return 0 for a debuffed card, so a debuffed Steel
        card is correctly spendable as a kicker."""
        steel = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_steel")
        steel.debuff = True
        assert not hand_solver._has_held_enhancement(steel)


class TestCopyResolution:
    """Gates read RESOLVED joker identities through the ENGINE's own copy
    path, never raw center keys."""

    def test_blueprint_gates_as_the_joker_it_copies(self):
        jokers = [create_joker("j_blueprint"), create_joker("j_lusty_joker")]
        views = hand_solver._resolved_joker_views(jokers)
        assert [k for k, _ in views] == ["j_lusty_joker", "j_lusty_joker"]

    def test_blueprint_cannot_manufacture_a_splash_gate(self):
        """Splash is on the 29-joker blueprint_compat incompat list, so a
        copy can never produce one. That comes free from resolving through
        the engine; a hand-rolled resolver would have to remember it."""
        jokers = [create_joker("j_blueprint"), create_joker("j_splash")]
        views = hand_solver._resolved_joker_views(jokers)
        assert views[0][0] != "j_splash", "Blueprint must not resolve to Splash"

    def test_unresolved_copy_joker_gates_nothing(self):
        """A lone Blueprint has no right-neighbour: it keeps its own key,
        which carries no predicate (class 4), so it gates nothing."""
        gates = _gates_for(_flush_plus_junk_hand(), [create_joker("j_blueprint")])
        assert not gates.scored_value
        assert not gates.play_away_lowest


class TestHypotheses:
    def test_raised_fist_variant_spends_the_lowest_cards(self):
        """Hypothesis 4 is exact for Raised Fist's min term: to raise the
        lowest HELD rank you spend the lowest cards as kickers."""
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:3]
        gates = hand_solver._KickerGates(False, False, True)
        variants = hand_solver._kicker_variants(line, hand, gates, counts)
        line_ids = {id(c) for c in line}
        kickers = [c for c in variants[-1] if id(c) not in line_ids]
        junk_ids = sorted(
            c.base.id for c in hand if c.base.suit != Suit.HEARTS.value
        )
        assert sorted(c.base.id for c in kickers) == junk_ids[: len(kickers)]

    def test_raised_fist_key_ranks_stone_cards_last(self):
        """A Stone card is excluded from Raised Fist's minimum, so playing
        one away cannot raise it -- it must never be the spent kicker."""
        stone = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_stone")
        two = create_playing_card(Suit.SPADES, Rank.TWO)
        assert hand_solver._raised_fist_key(two) < hand_solver._raised_fist_key(stone)

    def test_raised_fist_key_uses_rank_id_not_nominal(self):
        """`nominal` ties J/Q/K at 10; the handler compares `get_id()`,
        which distinguishes them. Using nominal would make the min term
        wrong on face-heavy hands."""
        jack = create_playing_card(Suit.SPADES, Rank.JACK)
        king = create_playing_card(Suit.SPADES, Rank.KING)
        assert jack.base.nominal == king.base.nominal
        assert hand_solver._raised_fist_key(jack) < hand_solver._raised_fist_key(king)

    def test_scored_variant_prefers_edition_kickers(self):
        """Editions fire on the SCORED channel, so under Splash a
        Polychrome 3 beats plain junk as a kicker. They are absent from
        trigger_match by design (it is a card x JOKER matrix), so the key
        carries the term itself -- caught in review 2026-07-16, and missing
        from the locked spec's "chips + enhancement + candidacy bits"."""
        hand = _flush_plus_junk_hand()
        junk = [c for c in hand if c.base.suit != Suit.HEARTS.value]
        poly = min(junk, key=lambda c: c.base.nominal)  # the WORST junk card
        poly.set_edition({"polychrome": True})
        counts = {id(c): (0, 0) for c in hand}
        key = hand_solver._scored_kicker_key(counts)
        assert all(key(poly) > key(o) for o in junk if o is not poly), (
            "a Polychrome kicker must outrank higher plain junk under Splash"
        )

    def test_held_variant_keeps_baron_kings_in_hand(self):
        """Hypothesis 3 pads with the cards LEAST valuable held, so Baron's
        Kings stay in hand rather than being spent as kickers."""
        hand = _quads_and_flush_hand()
        jokers = [create_joker("j_baron")]
        flags = get_hand_eval_flags(jokers)
        views = hand_solver._resolved_joker_views(jokers)
        counts = hand_solver._card_channel_counts(hand, views, {}, flags)
        line = [
            c
            for c in hand
            if c.base.suit == Suit.HEARTS.value and c.base.id != 13
        ][:3]
        gates = hand_solver._KickerGates(False, True, False)
        variants = hand_solver._kicker_variants(line, hand, gates, counts)
        line_ids = {id(c) for c in line}
        kickers = [c for c in variants[-1] if id(c) not in line_ids]
        assert all(c.base.id != 13 for c in kickers), (
            "held-value variant must not spend Baron's Kings as kickers"
        )

    def test_nominal_best_variant_is_always_first(self):
        """The pre-K1 behaviour is kept as variant 1 on every board."""
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:3]
        all_on = hand_solver._KickerGates(True, True, True)
        off = hand_solver._KickerGates(False, False, False)
        assert (
            hand_solver._kicker_variants(line, hand, all_on, counts)[0]
            == hand_solver._kicker_variants(line, hand, off, counts)[0]
        )

    def test_variants_dedupe_by_card_identity(self):
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:3]
        all_on = hand_solver._KickerGates(True, True, True)
        variants = hand_solver._kicker_variants(line, hand, all_on, counts)
        keys = [frozenset(id(c) for c in v) for v in variants]
        assert len(keys) == len(set(keys))

    def test_full_line_emits_no_variants(self):
        hand = _flush_plus_junk_hand()
        counts = {id(c): (0, 0) for c in hand}
        line = [c for c in hand if c.base.suit == Suit.HEARTS.value][:5]
        all_on = hand_solver._KickerGates(True, True, True)
        assert hand_solver._kicker_variants(line, hand, all_on, counts) == [line]


class TestFamilyKey:
    def test_family_key_is_splash_agnostic(self):
        """THE pitfall-#13 guard. Under Splash `score_hand`'s scoring_cards
        is all 5 played cards, so keying on it would give every kicker
        variant of one line a DIFFERENT family -- the variants would crowd
        out genuinely distinct lines, recreating the exact bug the family
        pass exists to prevent, on precisely the boards K1 targets. The
        engine's detection scan ignores Splash, so variants collide."""
        hand = _quads_and_flush_hand()
        kings = [c for c in hand if c.base.id == 13]
        junk = [c for c in hand if c.base.suit != Suit.HEARTS.value and c.base.id != 13]
        hearts = [c for c in hand if c.base.suit == Suit.HEARTS.value and c.base.id != 13]

        family_a = hand_solver._line_family(
            kings + junk[:1], four_fingers=False, shortcut=False, smeared=False
        )
        family_b = hand_solver._line_family(
            kings + hearts[:1], four_fingers=False, shortcut=False, smeared=False
        )
        assert family_a == family_b, "same line, different kicker -> same family"
        assert family_a[0] == "Four of a Kind"

    def test_family_key_separates_distinct_rank_lines(self):
        """The card SET stays in the key: hand type alone would collapse a
        pair of Kings and a pair of 3s into one family."""
        hand = _quads_and_flush_hand()
        kings = [c for c in hand if c.base.id == 13][:2]
        hearts = [c for c in hand if c.base.suit == Suit.HEARTS.value and c.base.id != 13]
        pair_of_kings = hand_solver._line_family(
            kings, four_fingers=False, shortcut=False, smeared=False
        )
        high_card = hand_solver._line_family(
            hearts[:1], four_fingers=False, shortcut=False, smeared=False
        )
        assert pair_of_kings != high_card

    def test_family_key_tolerates_baseless_stone_cards(self):
        """Stone cards have no base; `get_best_hand` groups but never emits
        them, so the scan must not raise (the existing fallback path)."""
        stone = _enhanced(Suit.CLUBS, Rank.EIGHT, "m_stone")
        stone.base = None
        family = hand_solver._line_family(
            [stone], four_fingers=False, shortcut=False, smeared=False
        )
        assert isinstance(family[0], str)


class TestVariantsRideTheirLine:
    def test_top_k_counts_lines_not_candidates(self):
        """`top_k` caps FAMILIES; every surviving line carries all of its
        variants into the exact pass, which arbitrates between them."""
        hand = _quads_and_flush_hand()
        jokers = [create_joker("j_splash")]
        hand_levels, blind, rng = _fixtures()
        for k in (1, 3, 5):
            candidates = prescreen_play_candidates(
                hand, jokers, hand_levels, blind, rng, top_k=k, game_state={}
            )
            families = {
                hand_solver._line_family(
                    cards, four_fingers=False, shortcut=False, smeared=False
                )
                for cards in candidates
            }
            assert len(families) <= k

    def test_budget_stays_far_under_brute_force(self):
        """The whole point: ~15 candidates against brute force's 218."""
        hand = _quads_and_flush_hand()
        hand_levels, blind, rng = _fixtures()
        jokers = [create_joker("j_splash"), create_joker("j_raised_fist")]
        candidates = prescreen_play_candidates(
            hand, jokers, hand_levels, blind, rng, top_k=5, game_state={}
        )
        assert len(candidates) < 60, (
            f"variant fan-out must stay well under 218; got {len(candidates)}"
        )


def _trips_plus_kicker_bait_hand() -> list[Card]:
    """9 cards engineered to reproduce the K1 miss.

    Trip Kings are the LINE and are never in doubt -- what varies is the two
    KICKERS. Nominal-best padding grabs the Ace and Queen (highest
    keep-priority); the low hearts are what a suit joker under Splash
    actually wants, and the low cards are what Raised Fist wants played
    away. So a generator that pads by nominal rank proposes the right line
    at the right size and still loses badly, which is exactly the measured
    signature (27/27 misses had `true_size = 5`).

    Note the sibling suite's `_flush_plus_junk_hand` CANNOT show this: its
    flush is complete and dominant, so no kicker choice exists to get wrong
    -- those tests pass on pre-K1 code and prove nothing about the fix.
    """
    return [
        create_playing_card(Suit.SPADES, Rank.KING),
        create_playing_card(Suit.CLUBS, Rank.KING),
        create_playing_card(Suit.DIAMONDS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.ACE),
        create_playing_card(Suit.CLUBS, Rank.QUEEN),
        create_playing_card(Suit.SPADES, Rank.JACK),
        create_playing_card(Suit.HEARTS, Rank.TWO),
        create_playing_card(Suit.HEARTS, Rank.THREE),
        create_playing_card(Suit.DIAMONDS, Rank.FOUR),
    ]


class TestRegressionAgainstBruteForce:
    """The tests that actually catch the bug: each one FAILS on pre-K1 code
    with the regret recorded in its docstring, measured on this branch.

    A prescreened label is "exact among prescreened candidates", so
    agreement with full brute force on these boards is the property K1
    exists to restore.
    """

    def test_splash_plus_suit_joker(self):
        """Lusty under Splash -- kicker SUIT feeds the mult. Pre-K1 pads the
        trip Kings with the Ace by nominal rank and scores 444 against brute
        force's 585: REGRET 141, 24.1% of the play's value. The same
        interaction produced the doc's seed 879 (regret 279).
        """
        hand = _trips_plus_kicker_bait_hand()
        jokers = [create_joker("j_splash"), create_joker("j_lusty_joker")]
        hand_levels, blind, rng = _fixtures()
        subset, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)
        # The fix is specifically that BOTH hearts get played as kickers.
        assert sum(1 for c in subset if c.base.suit == Suit.HEARTS.value) == 2

    def test_raised_fist(self):
        """Raised Fist reads the lowest HELD card, so the cards you DON'T
        play set the mult -- it was 16 of the 27 measured misses. Pre-K1
        strands the 2 in hand and scores 540 against brute force's 660:
        REGRET 120, 18.2% of the play's value.
        """
        hand = _trips_plus_kicker_bait_hand()
        jokers = [create_joker("j_raised_fist")]
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)

    def test_plain_board_unchanged(self):
        """The other side of the contract: with every gate shut, K1 must not
        move a plain board's label at all."""
        hand = _trips_plus_kicker_bait_hand()
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, [], hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, [])

    def test_dominant_flush_board_unchanged(self):
        """A complete dominant flush has no kicker choice to get wrong, so
        this passes pre-K1 too -- kept as a don't-break-the-easy-case guard,
        NOT as evidence for the fix."""
        hand = _flush_plus_junk_hand()
        jokers = [create_joker("j_splash"), create_joker("j_lusty_joker")]
        hand_levels, blind, rng = _fixtures()
        _, result = best_immediate_play(hand, jokers, hand_levels, blind, rng)
        assert result.total == _brute_force_best(hand, jokers)
