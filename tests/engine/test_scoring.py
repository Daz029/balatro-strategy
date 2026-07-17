"""Scoring pipeline tests.

Consolidated tests covering score_hand_base (Phases 1-8, 12) and
score_hand (Phases 1-14) including joker effects, editions, retriggers,
boss blinds, Plasma Deck, Glass destruction, and joker decay/save.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import ScoreResult, score_hand, score_hand_base


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_SL = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RL = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "Jack": "J",
    "Queen": "Q",
    "King": "K",
    "Ace": "A",
}


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    c = Card()
    c.set_base(f"{_SL[suit]}_{_RL[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    return c


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# 1. Baseline arithmetic (score_hand_base)
# ============================================================================


class TestBaseline:
    """Plain pair of Aces, level 1 — the reference score."""

    def test_pair_of_aces(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        result = score_hand_base(
            played,
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.hand_type == "Pair"
        assert result.chips == 32.0  # 10 + 11 + 11
        assert result.mult == 2.0
        assert result.total == 64
        assert result.debuffed is False
        assert isinstance(result, ScoreResult)
        assert isinstance(result.breakdown, list)
        assert len(result.breakdown) > 0


# ============================================================================
# 2. Enhancements — one test per type that affects scoring
# ============================================================================


class TestEnhancements:
    def test_bonus_card(self):
        """Bonus: +30 chips."""
        played = [_card("Hearts", "5", "m_bonus"), _card("Spades", "5")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.chips == 50.0  # 10 + 35 + 5
        assert r.total == 100

    def test_mult_card(self):
        """Mult Card: +4 mult."""
        played = [_card("Hearts", "5", "m_mult"), _card("Spades", "5")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == 6.0  # 2 + 4
        assert r.total == 120

    def test_glass_card(self):
        """Glass Card: x2 mult when scored."""
        glass_5 = _card("Hearts", "5", "m_glass")
        played = [glass_5, _card("Spades", "5")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == 4.0  # 2 × 2
        assert r.total == 80

    def test_steel_card_held(self):
        """Steel Card: x1.5 mult when held."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [_card("Diamonds", "5", "m_steel")]
        r = score_hand_base(played, held, HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == pytest.approx(3.0)  # 2 × 1.5
        assert r.total == 96

    def test_stone_card(self):
        """Stone Card: 50 chips, ignores rank nominal."""
        sc = _card("Hearts", "Ace", "m_stone")
        played = [_card("Hearts", "5"), _card("Spades", "5"), sc]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        # Pair: 10 base + 5 + 5 = 20 from pair. Stone adds 50.
        assert r.chips == 70.0
        assert r.total == 140


# ============================================================================
# 3. Editions — one test per type
# ============================================================================


class TestEditions:
    def test_foil(self):
        """Foil: +50 chips from edition."""
        c = _card("Hearts", "Ace")
        c.set_edition({"foil": True})
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.chips == 82.0  # 32 + 50
        assert r.total == 164

    def test_holo(self):
        """Holo: +10 mult from edition."""
        c = _card("Hearts", "Ace")
        c.set_edition({"holo": True})
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == 12.0  # 2 + 10

    def test_polychrome(self):
        """Polychrome: x1.5 mult from edition."""
        c = _card("Hearts", "Ace")
        c.set_edition({"polychrome": True})
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        assert r.mult == pytest.approx(3.0)  # 2 × 1.5


# ============================================================================
# 4. Red Seal retrigger
# ============================================================================


class TestRedSealRetrigger:
    def test_effects_double(self):
        """Red Seal: card evaluated twice — chips from both reps."""
        c = _card("Hearts", "Ace")
        c.set_seal("Red")
        played = [c, _card("Spades", "Ace")]
        r = score_hand_base(played, [], HandLevels(), _small_blind(), PseudoRandom("T"))
        # 10 + 11×2 + 11 = 43 chips, 2 mult
        assert r.chips == 43.0
        assert r.total == 86


# ============================================================================
# 5. Boss blinds
# ============================================================================


class TestBossBlind:
    def test_flint_halves(self):
        """The Flint: halves base chips and mult."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        blind = Blind.create("bl_flint", ante=1)
        r = score_hand_base(played, [], HandLevels(), blind, PseudoRandom("T"))
        # Pair base 10→5 chips, 2→1 mult; per card +22
        assert r.chips == 27.0
        assert r.mult == 1.0
        assert r.total == 27

    def test_eye_debuffs_repeat_hand(self):
        """The Eye: repeat hand type → debuffed (score = 0)."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        blind = Blind.create("bl_eye", ante=1)
        # First Pair: allowed
        r1 = score_hand_base(played, [], HandLevels(), blind, PseudoRandom("T"))
        assert r1.debuffed is False
        assert r1.total > 0
        # Second Pair: blocked
        reset_sort_id_counter()
        played2 = [_card("Hearts", "King"), _card("Spades", "King")]
        r2 = score_hand_base(played2, [], HandLevels(), blind, PseudoRandom("T"))
        assert r2.debuffed is True
        assert r2.total == 0


# ============================================================================
# 6. Jokers in pipeline — ordering matters
# ============================================================================


class TestJokerOrdering:
    def test_additive_then_multiplicative(self):
        """j_joker (+4) then j_blackboard (x3): (2+4)×3 = 18."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]  # all black for Blackboard
        joker = _joker("j_joker", mult=4)
        bb = _joker("j_blackboard", extra=3)
        r = score_hand(
            played,
            held,
            [joker, bb],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == pytest.approx(18.0)
        assert r.total == 576

    def test_multiplicative_then_additive(self):
        """j_blackboard (x3) then j_joker (+4): (2×3)+4 = 10. DIFFERENT."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]
        bb = _joker("j_blackboard", extra=3)
        joker = _joker("j_joker", mult=4)
        r = score_hand(
            played,
            held,
            [bb, joker],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == pytest.approx(10.0)
        assert r.total == 320


# ============================================================================
# 7. Joker edition effects
# ============================================================================


class TestJokerEdition:
    def test_foil_adds_chips(self):
        """Foil joker: +50 chips BEFORE joker effect."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"foil": True})
        r = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.chips == 82.0
        assert r.mult == 6.0
        assert r.total == 492

    def test_holo_adds_mult(self):
        """Holo joker: +10 mult BEFORE joker effect."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"holo": True})
        r = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == 16.0
        assert r.total == 512

    def test_polychrome_multiplies_after(self):
        """Polychrome joker: x1.5 AFTER joker effect."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"polychrome": True})
        r = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.mult == pytest.approx(9.0)
        assert r.total == 288


# ============================================================================
# 8. Phase 10: Plasma Deck averaging
# ============================================================================


class TestPlasmaDeck:
    def test_averages_chips_and_mult(self):
        """Plasma Deck: (chips+mult)/2 for both. 32+2=34, each=17. 17×17=289."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        r = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            back_key="b_plasma",
        )
        assert r.chips == 17.0
        assert r.mult == 17.0
        assert r.total == 289


# ============================================================================
# 9. Phase 11: Glass Card shatter + joker reaction
# ============================================================================


class TestGlassDestruction:
    def test_glass_guaranteed_shatter(self):
        """With high probabilities, Glass Card always shatters."""
        glass_5 = _card("Hearts", "5", "m_glass")
        played = [glass_5, _card("Spades", "5")]
        r = score_hand(
            played,
            [],
            [],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        assert glass_5 in r.cards_destroyed

    def test_caino_reacts_to_shattered_face(self):
        """Glass King shatters → Caino gains +1 xMult."""
        glass_king = _card("Hearts", "King", "m_glass")
        played = [
            glass_king,
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        caino = _joker("j_caino", caino_xmult=1, extra=1)
        r = score_hand(
            played,
            [],
            [caino],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        assert glass_king in r.cards_destroyed
        assert caino.ability["caino_xmult"] == 2

    def test_glass_joker_reacts_to_shatter(self):
        """Glass Card shatters → Glass Joker gains +0.75 xMult."""
        glass_5 = _card("Hearts", "5", "m_glass")
        played = [glass_5, _card("Spades", "5")]
        glass_j = _joker("j_glass", x_mult=1, extra=0.75)
        r = score_hand(
            played,
            [],
            [glass_j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("SHATTER"),
            probabilities_normal=100.0,
        )
        assert glass_5 in r.cards_destroyed
        assert glass_j.ability["x_mult"] == pytest.approx(1.75)


# ============================================================================
# 10. Phase 13: Ice Cream decay + Mr. Bones save
# ============================================================================


class TestAfterPhase:
    def test_ice_cream_decays(self):
        """Ice Cream: +100 chips in Phase 9, then -5 in Phase 13."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        ice = _joker("j_ice_cream", extra={"chips": 100, "chip_mod": 5})
        r = score_hand(
            played,
            [],
            [ice],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert r.total == 264  # (32+100) × 2
        assert ice.ability["extra"]["chips"] == 95

    def test_ice_cream_self_destructs(self):
        """Ice Cream at 5 chips → decays to 0 → removed."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        ice = _joker("j_ice_cream", extra={"chips": 5, "chip_mod": 5})
        r = score_hand(
            played,
            [],
            [ice],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )
        assert ice in r.jokers_removed

    def test_mr_bones_saves_losing_hand(self):
        """Mr. Bones saves when score < blind_chips and hands_left == 0."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bones = _joker("j_mr_bones")
        r = score_hand(
            played,
            [],
            [bones],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            blind_chips=300,
            game_state={"hands_left": 0},
        )
        assert r.total == 64
        assert r.saved is True
        assert bones in r.jokers_removed


# ============================================================================
# Throwback — skips must flow from game_state into GameSnapshot
# ============================================================================


class TestThrowbackIntegration:
    """The handler was unit-tested with a hand-built JokerContext, but
    score_hand's GameSnapshot construction silently dropped ``skips`` —
    Throwback could never fire through the real scoring pipeline. This
    covers the integration path end to end."""

    def test_throwback_fires_through_score_hand(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        throwback = _joker("j_throwback", extra=0.25)
        base = score_hand(
            played, [], [], HandLevels(), _small_blind(), PseudoRandom("T")
        )
        r = score_hand(
            played,
            [],
            [throwback],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            game_state={"skips": 4},
        )
        # 4 skips x 0.25 = +1.0 -> x2 mult overall
        assert r.total == base.total * 2

    def test_throwback_inert_with_zero_skips(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        throwback = _joker("j_throwback", extra=0.25)
        base = score_hand(
            played, [], [], HandLevels(), _small_blind(), PseudoRandom("T")
        )
        r = score_hand(
            played,
            [],
            [throwback],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            game_state={"skips": 0},
        )
        assert r.total == base.total


# ============================================================================
# The Idol — idol_card["id"] must flow from reset_round_targets to the handler
# ============================================================================


class TestIdolIntegration:
    """Same bug class as Throwback: the j_idol handler compares
    ``other_card.get_id() == idol_card.get("id")``, but ``reset_round_targets``
    stored only rank/suit — no "id" — so The Idol could never fire through the
    real pipeline while its hand-built-ctx unit tests passed. This drives the
    reset-produced dict through score_hand end to end."""

    def _idol_card_from_reset(self) -> dict:
        from jackdaw.engine.round_lifecycle import reset_round_targets
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "IDOL_INTEG")
        reset_round_targets(PseudoRandom("IDOL_INTEG"), 1, gs)
        return gs["current_round"]["idol_card"]

    def test_idol_fires_through_score_hand_with_reset_dict(self):
        idol_card = self._idol_card_from_reset()
        assert idol_card.get("id") is not None  # the missing field
        played = [_card(idol_card["suit"], idol_card["rank"])]
        idol = _joker("j_idol", extra=2)
        base = score_hand(played, [], [], HandLevels(), _small_blind(), PseudoRandom("T"))
        r = score_hand(
            played,
            [],
            [idol],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            game_state={"idol_card": idol_card},
        )
        assert r.total == base.total * 2

    def test_idol_inert_on_non_matching_card(self):
        idol_card = self._idol_card_from_reset()
        # Same suit, different rank — must not fire
        wrong_rank = "Ace" if idol_card["rank"] != "Ace" else "King"
        played = [_card(idol_card["suit"], wrong_rank)]
        idol = _joker("j_idol", extra=2)
        base = score_hand(played, [], [], HandLevels(), _small_blind(), PseudoRandom("T"))
        r = score_hand(
            played,
            [],
            [idol],
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
            game_state={"idol_card": idol_card},
        )
        assert r.total == base.total


# ============================================================================
# N. Hand-eval modifier flags reach the SCORING pipeline (Throwback class)
# ============================================================================


class TestHandEvalFlagsIntegration:
    """Four Fingers / Shortcut / Smeared must change hand DETECTION inside
    `score_hand`, not merely inside `hand_eval`.

    Regression for a real bug: `score_hand` called
    `evaluate_hand(played_cards, jokers=None)` -- a copy-paste carryover
    from `score_hand_base` (the deliberately jokerless scorer, 09105e3)
    into the joker-aware scorer (7633d34). `evaluate_hand` derives every
    detection flag from that list, so all three jokers were INERT
    everywhere: in-game, in the env, and in every solver label. Splash
    alone survived, because `score_hand` re-implements it independently at
    Phase 3c.

    These assertions deliberately go through `score_hand` and NOT through
    `evaluate_hand`/`get_flush`/`evaluate_poker_hand`. Unit tests at those
    levels already existed and PASSED throughout -- the defect was purely
    in the integration seam, which is the same gap that hid The Idol and
    Throwback itself. A test that constructs the flags by hand cannot fail
    on this bug and is worthless here.
    """

    def _score(self, played, jokers):
        return score_hand(
            played,
            [],
            jokers,
            HandLevels(),
            _small_blind(),
            PseudoRandom("T"),
        )

    def test_four_fingers_four_card_straight_detected(self):
        played = [
            _card("Clubs", "10"),
            _card("Diamonds", "9"),
            _card("Diamonds", "8"),
            _card("Clubs", "7"),
        ]
        assert self._score(played, []).hand_type == "High Card"
        assert self._score(played, [_joker("j_four_fingers")]).hand_type == "Straight"

    def test_four_fingers_four_card_flush_detected(self):
        played = [
            _card("Hearts", "2"),
            _card("Hearts", "5"),
            _card("Hearts", "9"),
            _card("Hearts", "King"),
        ]
        assert self._score(played, []).hand_type == "High Card"
        assert self._score(played, [_joker("j_four_fingers")]).hand_type == "Flush"

    def test_shortcut_gapped_straight_detected(self):
        played = [
            _card("Diamonds", "9"),
            _card("Clubs", "7"),
            _card("Hearts", "5"),
            _card("Spades", "4"),
            _card("Diamonds", "3"),
        ]
        assert self._score(played, []).hand_type == "High Card"
        assert self._score(played, [_joker("j_shortcut")]).hand_type == "Straight"

    def test_smeared_mixed_suit_flush_detected(self):
        played = [
            _card("Hearts", "2"),
            _card("Diamonds", "5"),
            _card("Hearts", "9"),
            _card("Diamonds", "King"),
            _card("Hearts", "3"),
        ]
        assert self._score(played, []).hand_type == "High Card"
        assert self._score(played, [_joker("j_smeared")]).hand_type == "Flush"

    def test_debuffed_four_fingers_does_not_enable(self):
        """`get_hand_eval_flags` skips debuffed jokers (find_joker semantics).
        Pinned at the integration level so passing the list can never
        smuggle a debuffed joker's flag through."""
        played = [
            _card("Clubs", "10"),
            _card("Diamonds", "9"),
            _card("Diamonds", "8"),
            _card("Clubs", "7"),
        ]
        ff = _joker("j_four_fingers")
        ff.debuff = True
        assert self._score(played, [ff]).hand_type == "High Card"

    def test_four_fingers_straight_scores_more_than_high_card(self):
        """The detection change must reach the SCORE, not just the label --
        a Straight's base chips/mult are what the solver's labels read."""
        played = [
            _card("Clubs", "10"),
            _card("Diamonds", "9"),
            _card("Diamonds", "8"),
            _card("Clubs", "7"),
        ]
        assert (
            self._score(played, [_joker("j_four_fingers")]).total
            > self._score(played, []).total
        )

    def test_splash_still_scores_all_played_cards(self):
        """Splash is applied TWICE once jokers are passed (evaluate_hand's
        augmentation + Phase 3c). Both produce `list(played_cards)` in
        played order, so the second is a redundant no-op -- pinned so a
        future Phase 3c removal, or an evaluate_hand change, cannot
        silently double-count or drop it."""
        played = [
            _card("Hearts", "Ace"),
            _card("Spades", "Ace"),
            _card("Clubs", "2"),
        ]
        plain = self._score(played, [])
        splash = self._score(played, [_joker("j_splash")])
        assert plain.hand_type == splash.hand_type == "Pair"
        assert len(plain.scoring_cards) == 2  # the two Aces
        assert len(splash.scoring_cards) == 3  # + the off-line 2
        assert splash.total > plain.total

    def test_four_fingers_and_shortcut_stack(self):
        """Two detection flags on one board must COMPOSE, not race.

        10-8-6-4 is a straight only if the hand is allowed to be 4 long
        (Four Fingers) AND to skip a rank at every step (Shortcut). Either
        joker alone leaves it High Card, so this fails unless both flags
        survive the same `get_hand_eval_flags` call and both reach
        detection. The flags are independent booleans, so composition is
        expected -- but "expected to compose" is precisely the reasoning
        that let jokers=None ship, hence a board only the conjunction can
        score.
        """
        played = [
            _card("Clubs", "10"),
            _card("Diamonds", "8"),
            _card("Hearts", "6"),
            _card("Spades", "4"),
        ]
        ff = _joker("j_four_fingers")
        sc = _joker("j_shortcut")
        assert self._score(played, []).hand_type == "High Card"
        assert self._score(played, [ff]).hand_type == "High Card"
        assert self._score(played, [sc]).hand_type == "High Card"
        both = self._score(played, [ff, sc])
        assert both.hand_type == "Straight"
        assert len(both.scoring_cards) == 4
        assert both.total > self._score(played, [ff]).total

    def test_splash_composes_with_four_fingers(self):
        """Splash's two application sites must agree once a SECOND flag is
        live on the same board.

        The redundant-restatement argument for keeping Phase 3c rests on
        both sites producing `list(played_cards)`. That is trivially true
        when Splash is the only flag; this pins it when Four Fingers has
        already rewritten `scoring_cards` upstream (4-card flush + 1
        off-suit card). If Phase 3c ever stopped being a pure restatement,
        the off-suit 5th card is where it would show: dropped (Phase 3c
        overwritten by FF's line) or double-counted.
        """
        played = [
            _card("Hearts", "2"),
            _card("Hearts", "5"),
            _card("Hearts", "9"),
            _card("Hearts", "King"),
            _card("Clubs", "3"),
        ]
        ff = _joker("j_four_fingers")
        ff_only = self._score(played, [ff])
        assert ff_only.hand_type == "Flush"
        assert len(ff_only.scoring_cards) == 4  # the off-suit 3 sits out

        both = self._score(played, [ff, _joker("j_splash")])
        assert both.hand_type == "Flush"
        assert len(both.scoring_cards) == 5  # ... and now it scores
        assert both.scoring_cards == played  # played order, no dupes
        assert both.total > ff_only.total
