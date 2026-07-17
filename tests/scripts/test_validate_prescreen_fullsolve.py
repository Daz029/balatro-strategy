"""Tests for the K2 full-solve gate arm (`validate_prescreen_fullsolve`).

The probe measures the prescreen counterfactually at every
`best_immediate_play` call inside a REAL solve. Two properties are
load-bearing and pinned here:

  * NON-PERTURBATION: an instrumented solve returns byte-identical results
    to a bare one (the wrappers call the originals first and everything the
    measurement touches is clone-safe). A violation would mean the gate
    measures a solve that never happens in production.
  * TRUE-DEPTH ATTRIBUTION: nodes land in the right stratum
    (discards_left at the node, "mc" for the future-hand sampler, roots
    flagged) -- the whole point of the full-solve arm over root-only
    measurement.

Plus the free-oracle shortcut (at n <= PRESCREEN_HAND_LIMIT the original
call's own result IS the brute-force truth) and clean install/uninstall.
"""

from __future__ import annotations

import hand_solver
from hand_solver import DeckComposition, solve_hand_for_ante_clear
from validate_prescreen_fullsolve import PrescreenNodeProbe, run_smoke_solve
from validate_prescreen_n8 import _brute_argmax, _solver_args

from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom


def _deck() -> DeckComposition:
    return DeckComposition.from_deck(
        [create_playing_card(s, r) for s in Suit for r in Rank]
    )


def _hand() -> list:
    return [
        create_playing_card(Suit.HEARTS, Rank.ACE),
        create_playing_card(Suit.HEARTS, Rank.KING),
        create_playing_card(Suit.SPADES, Rank.TWO),
        create_playing_card(Suit.CLUBS, Rank.FOUR),
        create_playing_card(Suit.DIAMONDS, Rank.SIX),
        create_playing_card(Suit.SPADES, Rank.EIGHT),
        create_playing_card(Suit.CLUBS, Rank.NINE),
        create_playing_card(Suit.DIAMONDS, Rank.JACK),
    ]


def _solve(hand, discards_left=1, mc_seed="fullsolve_test"):
    return solve_hand_for_ante_clear(
        hand, [], HandLevels(), Blind.create("bl_small", ante=1),
        PseudoRandom("fullsolve"), _deck(),
        chips_needed=300.0, hands_left=2, discards_left=discards_left,
        blind_chips=300, mc_seed=mc_seed,
    )


class TestNonPerturbation:
    def test_instrumented_solve_identical_to_bare(self):
        hand = _hand()
        bare = _solve(hand)
        with PrescreenNodeProbe(ks=[4]):
            probed = _solve(hand)
        assert probed.action == bare.action
        assert probed.template_name == bare.template_name
        assert probed.p_clear == bare.p_clear
        assert [id(c) for c in probed.hold] == [id(c) for c in bare.hold]
        assert [id(c) for c in probed.discard] == [id(c) for c in bare.discard]

    def test_uninstall_restores_originals(self):
        orig_bip = hand_solver.best_immediate_play
        orig_sht = hand_solver.solve_hand_turn
        orig_mc = hand_solver.estimate_future_hand_distribution
        with PrescreenNodeProbe(ks=[4]):
            assert hand_solver.best_immediate_play is not orig_bip
        assert hand_solver.best_immediate_play is orig_bip
        assert hand_solver.solve_hand_turn is orig_sht
        assert hand_solver.estimate_future_hand_distribution is orig_mc


class TestDepthAttribution:
    def test_strata_cover_recursion_and_mc(self):
        hand = _hand()
        probe = PrescreenNodeProbe(ks=[4])
        with probe:
            _solve(hand, discards_left=2)
        depths = {r["depth"] for r in probe.records}
        # Root node at the starting discards_left, at least one recursed
        # level below it, and the MC future-hand sampler's stratum.
        assert 2 in depths
        assert 1 in depths
        assert "mc" in depths
        roots = [r for r in probe.records if r["is_root"]]
        assert roots and all(r["depth"] == 2 for r in roots)
        assert all(not r["is_root"] for r in probe.records if r["depth"] == "mc")

    def test_record_shape(self):
        hand = _hand()
        probe = PrescreenNodeProbe(ks=[3, 8])
        with probe:
            _solve(hand)
        assert probe.records
        for rec in probe.records:
            assert rec["hand_size"] == 8
            assert rec["truth_total"] > 0
            for k in (3, 8):
                row = rec["by_k"][k]
                assert row["regret"] >= 0.0
                assert row["captured_by_value"] == (row["regret"] <= 1e-6)
                assert 0 < row["box_size"] < 218


class TestFreeOracle:
    def test_truth_is_the_original_result_at_n8(self):
        # At n <= PRESCREEN_HAND_LIMIT the original call brute-forces, so
        # the probe must reuse its result as truth (no re-enumeration).
        hand = _hand()
        hl, blind, rng = HandLevels(), Blind.create("bl_small", ante=1), PseudoRandom("fs2")
        probe = PrescreenNodeProbe(ks=[4])
        with probe:
            subset, result = hand_solver.best_immediate_play(hand, [], hl, blind, rng)
        assert len(probe.records) == 1
        rec = probe.records[0]
        assert rec["truth_total"] == float(result.total)
        assert rec["depth"] == "bare"  # no solve_hand_turn frame on the stack


class TestBruteArgmaxHelper:
    def test_matches_best_immediate_play_at_n8(self):
        # The n8 harness's explicit enumeration must be byte-identical to
        # best_immediate_play's n<=8 branch (same order, same tie rule).
        hand = _hand()
        gs = {
            "hand": hand,
            "jokers": [],
            "hand_levels": HandLevels(),
            "blind": Blind.create("bl_small", ante=1),
            "rng": PseudoRandom("brute"),
        }
        args = _solver_args(gs)
        subset, total = _brute_argmax(args)
        bip_subset, bip_result = hand_solver.best_immediate_play(
            hand, [], args["hand_levels"], args["blind"], args["rng"],
            args["game_state"], args["blind_chips"],
        )
        assert total == float(bip_result.total)
        assert [id(c) for c in subset] == [id(c) for c in bip_subset]


class TestSmokeSolve:
    def test_forced_prescreen_solve_restores_globals(self):
        hand = _hand()
        old_limit = hand_solver.PRESCREEN_HAND_LIMIT
        old_k = hand_solver.PRESCREEN_TOP_K
        gs = {
            "hand": hand,
            "jokers": [],
            "hand_levels": HandLevels(),
            "blind": Blind.create("bl_small", ante=1),
            "rng": PseudoRandom("smoke"),
            "deck": [create_playing_card(s, r) for s in Suit for r in Rank],
            "chips": 0,
            "current_round": {"hands_left": 2, "discards_left": 1},
        }
        gs["blind"].chips = 300
        choice = run_smoke_solve(gs, "smoke_test", k=4)
        assert choice.action in ("play", "discard")
        assert hand_solver.PRESCREEN_HAND_LIMIT == old_limit
        assert hand_solver.PRESCREEN_TOP_K == old_k
