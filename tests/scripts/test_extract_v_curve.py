"""Tests for the V_curve extraction harness (wave-0 item, CLAUDE.md "Money/
dollar handling" + docs/post-regen-training-plan.md section 3).

No real checkpoint or 25k-state corpus here -- a tiny corpus is harvested
via the real ``harvest_s0_rollouts`` pipeline (so blobs restore through the
real ``harvest_restore.restore_state`` path, exactly like production data),
and the critic is a scriptable stub (``DollarProbeCritic``) that recovers
the swept dollar value from the real encoded observation and applies a
caller-supplied function of it -- so gut-check behaviour (monotonicity
violations, range violations, kink diagnostics) can be pinned exactly
without depending on real network weights.

The one test that matters most (:class:`TestHardRuleRegression`) is the
regression test for the project's stated hard rule: "counterfactuals edit
engine state, never obs vectors." It asserts the sweep's actual mechanism
(mutate ``gs["dollars"]``, then fully re-encode) changes MULTIPLE obs
feature families, and contrasts it against the wrong shortcut (poking one
array in place) to show that shortcut leaves other families stale.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from extract_v_curve import (
    CellAccumulator,
    build_artifact,
    iter_restored_states,
    kink_diagnostic,
    load_shop_records,
    monotonicity_violations,
    range_summary,
    run_sweep,
    sorted_for_locality,
    sparse_antes,
    stack_obs,
    subsample_records,
)
from harvest_s0_rollouts import HarvestSink, NextRoundPolicy, harvest_runs

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy
from jackdaw.agents.v_curve import VCurve, load_v_curve
from jackdaw.env.shop_obs import build_shop_observation

# ---------------------------------------------------------------------------
# Shared tiny corpus -- real engine states via the real harvest pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A tiny real harvested corpus: ~30 shop-state records over antes 1-2.

    NextRoundPolicy (shop) + GreedyHandPolicy (hand) needs no checkpoint at
    all, so this runs in well under a second and produces real, restorable
    engine blobs -- exactly the shape ``extract_v_curve.py`` consumes in
    production, just smaller.
    """
    tmp_path = tmp_path_factory.mktemp("harvest_corpus")
    sink = HarvestSink(tmp_path, git_sha_stamp="testsha")
    harvest_runs(
        NextRoundPolicy(),
        GreedyHandPolicy(),
        sink,
        n_runs=15,
        seed_prefix="HARVEST",
        source="det",
        win_ante=4,
        max_steps=64,
    )
    sink.close()
    rows = load_shop_records(tmp_path, source_filter="det")
    assert len(rows) >= 20, "fixture assumption: enough shop states for the tests below"
    assert {r["ante"] for r in rows} >= {1, 2}, "fixture assumption: multiple antes present"
    return tmp_path, sorted_for_locality(rows)


@pytest.fixture(scope="module")
def blob_dir(corpus):
    tmp_path, _rows = corpus
    return tmp_path / "blobs"


# ---------------------------------------------------------------------------
# The hard rule: counterfactuals edit engine state, never obs vectors
# ---------------------------------------------------------------------------


class TestHardRuleRegression:
    def test_full_reencode_changes_multiple_obs_families(self, corpus, blob_dir):
        _tmp_path, rows = corpus
        row, gs = next(iter_restored_states([rows[0]], blob_dir))

        gs["dollars"] = 5
        obs_lo = build_shop_observation(gs, None)
        gs["dollars"] = 55
        obs_hi = build_shop_observation(gs, None)

        changed = [k for k in obs_lo if not np.array_equal(obs_lo[k], obs_hi[k])]
        # global_context carries the GC log-dollars feature; shop_context
        # carries the raw dollars/50 feature -- both must move on a real
        # re-encode, independent of what this particular state's shop
        # inventory happens to contain.
        assert "global_context" in changed
        assert "shop_context" in changed
        assert len(changed) >= 2

    def test_naive_single_field_poke_is_inconsistent_with_honest_reencode(self, corpus, blob_dir):
        """Documents WHY the hard rule exists: patching just the obvious
        ``shop_context`` dollars feature (the wrong shortcut) leaves every
        OTHER dollar-derived family (e.g. global_context's log-dollars)
        stale -- an internally contradictory, OOD observation."""
        _tmp_path, rows = corpus
        row, gs = next(iter_restored_states([rows[0]], blob_dir))

        gs["dollars"] = 5
        obs = build_shop_observation(gs, None)
        naive = {k: v.copy() for k, v in obs.items()}
        naive["shop_context"][0] = 55.0 / 50.0  # only the "obvious" feature

        gs["dollars"] = 55
        honest = build_shop_observation(gs, None)

        # shop_context[0] agrees by construction (that's the one field the
        # naive edit patched) -- but the rest of the state disagrees.
        assert naive["shop_context"][0] == pytest.approx(honest["shop_context"][0])
        assert not np.array_equal(naive["global_context"], honest["global_context"])

    def test_dollars_edit_does_not_mutate_other_engine_state(self, corpus, blob_dir):
        """The sweep must only ever touch ``gs["dollars"]`` -- everything
        else (ante, jokers, hand) must be exactly what was harvested."""
        _tmp_path, rows = corpus
        row, gs = next(iter_restored_states([rows[0]], blob_dir))
        ante_before = gs["round_resets"]["ante"]
        jokers_before = list(gs.get("jokers", []))

        gs["dollars"] = 37
        build_shop_observation(gs, None)

        assert gs["round_resets"]["ante"] == ante_before
        assert gs.get("jokers", []) == jokers_before


# ---------------------------------------------------------------------------
# Corpus loading / subsampling
# ---------------------------------------------------------------------------


class TestLoadShopRecords:
    def test_filters_to_shop_kind_only(self, corpus):
        tmp_path, rows = corpus
        assert rows, "fixture produced no shop rows"
        assert all(r["kind"] == "shop" for r in rows)

    def test_unknown_source_filter_yields_nothing(self, corpus):
        tmp_path, _rows = corpus
        assert load_shop_records(tmp_path, source_filter="sampled") == []

    def test_no_source_filter_keeps_everything(self, corpus):
        tmp_path, rows = corpus
        assert len(load_shop_records(tmp_path, source_filter=None)) == len(rows)


class TestSubsampleRecords:
    def test_cap_above_population_returns_everything(self, corpus):
        _tmp_path, rows = corpus
        assert subsample_records(rows, max_states=None, seed=0) == rows
        assert subsample_records(rows, max_states=len(rows) + 5, seed=0) == rows

    def test_deterministic_for_a_fixed_seed(self, corpus):
        _tmp_path, rows = corpus
        a = subsample_records(rows, max_states=5, seed=42)
        b = subsample_records(rows, max_states=5, seed=42)
        assert a == b
        assert len(a) == 5

    def test_is_a_subset(self, corpus):
        _tmp_path, rows = corpus
        sub = subsample_records(rows, max_states=5, seed=7)
        ids = {r["record_id"] for r in rows}
        assert all(r["record_id"] in ids for r in sub)


# ---------------------------------------------------------------------------
# Sweep-loop mechanics with a scriptable stub critic
# ---------------------------------------------------------------------------


class DollarProbeCritic:
    """Stub critic: recovers the swept dollar value from the REAL encoded
    ``shop_context[:,0]`` feature (``dollars / 50.0``) and applies a
    caller-supplied function of it. Lets tests pin exact per-dollar values
    without a real checkpoint, while still exercising the real
    ``build_shop_observation`` encode path end to end.
    """

    def __init__(self, fn):
        self._fn = fn

    def values(self, obs_batch: dict[str, np.ndarray]) -> np.ndarray:
        dollars = np.round(obs_batch["shop_context"][:, 0] * 50.0).astype(int)
        return np.array([self._fn(int(d)) for d in dollars], dtype=np.float32)


class TestStackObs:
    def test_stacks_matching_keys_with_leading_batch_dim(self, corpus, blob_dir):
        _tmp_path, rows = corpus
        _row, gs = next(iter_restored_states([rows[0]], blob_dir))
        obs_a = build_shop_observation(gs, None)
        gs["dollars"] = gs["dollars"] + 1
        obs_b = build_shop_observation(gs, None)

        batch = stack_obs([obs_a, obs_b])
        assert set(batch) == set(obs_a)
        for key in obs_a:
            assert batch[key].shape == (2, *obs_a[key].shape)


class TestRunSweepMechanics:
    def test_every_ante_and_dollar_cell_gets_the_full_state_count(self, corpus, blob_dir):
        tmp_path, rows = corpus
        critic = DollarProbeCritic(lambda d: d / 100.0)
        dollar_values = list(range(-2, 3))
        acc, failures = run_sweep(rows, blob_dir, dollar_values, critic, batch_size=7)

        assert failures == []
        cells = acc.cells()
        n_by_ante: dict[int, int] = {}
        for r in rows:
            n_by_ante[r["ante"]] = n_by_ante.get(r["ante"], 0) + 1

        for ante, expected_count in n_by_ante.items():
            assert set(cells[ante]) == set(dollar_values)
            for dollar in dollar_values:
                assert cells[ante][dollar]["count"] == expected_count

    def test_cell_means_match_the_probe_function(self, corpus, blob_dir):
        tmp_path, rows = corpus
        critic = DollarProbeCritic(lambda d: d / 10.0)
        dollar_values = [0, 5, 10]
        acc, failures = run_sweep(rows, blob_dir, dollar_values, critic, batch_size=13)
        assert failures == []
        cells = acc.cells()
        for ante, dollar_map in cells.items():
            for dollar, cell in dollar_map.items():
                assert cell["mean"] == pytest.approx(dollar / 10.0, abs=1e-4)

    def test_batch_size_does_not_affect_results(self, corpus, blob_dir):
        tmp_path, rows = corpus
        dollar_values = [-1, 0, 1, 4]
        acc_small, _ = run_sweep(
            rows, blob_dir, dollar_values, DollarProbeCritic(lambda d: d / 7.0), batch_size=1
        )
        acc_big, _ = run_sweep(
            rows, blob_dir, dollar_values, DollarProbeCritic(lambda d: d / 7.0), batch_size=10_000
        )
        assert acc_small.cells() == acc_big.cells()


# ---------------------------------------------------------------------------
# Gut-checks
# ---------------------------------------------------------------------------


class TestMonotonicityCheck:
    def test_monotone_curve_has_no_violations(self, corpus, blob_dir):
        rows = corpus[1]
        critic = DollarProbeCritic(lambda d: 1.0 / (1.0 + np.exp(-d / 10.0)))
        acc, _ = run_sweep(rows, blob_dir, list(range(-5, 6)), critic, batch_size=32)
        assert monotonicity_violations(acc.cells()) == []

    def test_a_deliberate_dip_is_reported_not_crashed(self, corpus, blob_dir):
        rows = corpus[1]

        def fn(d: int) -> float:
            return -0.5 if d == 2 else d / 10.0

        acc, _ = run_sweep(rows, blob_dir, [0, 1, 2, 3, 4], DollarProbeCritic(fn), batch_size=32)
        violations = monotonicity_violations(acc.cells())

        assert violations  # reported...
        for v in violations:
            # ...and localized exactly at the dip (1 -> 2), not (2 -> 3):
            # v(1)=0.1 > v(2)=-0.5 (violation), v(2)=-0.5 < v(3)=0.3 (fine).
            assert v["dollar_from"] == 1
            assert v["dollar_to"] == 2
            assert v["delta"] == pytest.approx(-0.5 - 0.1, abs=1e-4)

    def test_pure_function_on_handcrafted_cells(self):
        cells = {
            1: {
                0: {"mean": 0.1, "count": 5},
                1: {"mean": 0.05, "count": 5},
                2: {"mean": 0.3, "count": 5},
            },
            2: {0: {"mean": 0.2, "count": 5}, 1: {"mean": 0.4, "count": 5}},
        }
        violations = monotonicity_violations(cells)
        assert len(violations) == 1
        v = violations[0]
        assert v["ante"] == 1 and v["dollar_from"] == 0 and v["dollar_to"] == 1
        assert v["delta"] == pytest.approx(0.05 - 0.1)


class TestRangeCheck:
    def test_in_range_curve_flags_nothing(self, corpus, blob_dir):
        rows = corpus[1]
        critic = DollarProbeCritic(lambda d: 1.0 / (1.0 + np.exp(-d / 10.0)))
        acc, _ = run_sweep(rows, blob_dir, list(range(-5, 6)), critic, batch_size=32)
        summary = range_summary(acc)
        assert summary["n_out_of_range"] == 0
        assert 0.0 <= summary["min_observed"] <= summary["max_observed"] <= 1.0

    def test_out_of_range_values_are_flagged_not_fatal(self, corpus, blob_dir):
        rows = corpus[1]
        critic = DollarProbeCritic(lambda d: d / 10.0)  # exceeds [0,1] outside d in [0,10]
        dollar_values = [-20, -5, 0, 15, 40]
        acc, failures = run_sweep(rows, blob_dir, dollar_values, critic, batch_size=32)
        assert failures == []  # never fatal
        summary = range_summary(acc)
        assert summary["n_out_of_range"] > 0
        assert summary["max_observed"] == pytest.approx(4.0, abs=1e-4)
        assert summary["min_observed"] == pytest.approx(-2.0, abs=1e-4)


class TestSparseAntes:
    def test_flags_only_when_below_threshold(self):
        cells = {
            1: {0: {"mean": 0.1, "count": 50}},
            2: {0: {"mean": 0.1, "count": 3}},
        }
        flagged = sparse_antes(cells, threshold=10)
        assert [f["ante"] for f in flagged] == [2]
        assert flagged[0]["min_count"] == 3

    def test_zero_threshold_flags_nothing(self):
        cells = {1: {0: {"mean": 0.1, "count": 1}}}
        assert sparse_antes(cells, threshold=0) == []

    def test_real_corpus_ante2_is_sparser_than_ante1(self, corpus, blob_dir):
        rows = corpus[1]
        acc, _ = run_sweep(rows, blob_dir, [0], DollarProbeCritic(lambda d: 0.5), batch_size=32)
        cells = acc.cells()
        count_ante1 = cells[1][0]["count"]
        count_ante2 = cells[2][0]["count"]
        assert count_ante2 < count_ante1  # fixture assumption backing this test


class TestKinkDiagnostic:
    def test_linear_curve_has_near_zero_second_difference(self):
        cells = {1: {d: {"mean": d / 10.0, "count": 5} for d in range(-10, 11)}}
        diag = kink_diagnostic(cells, dollar_min=-10, dollar_max=10)
        for entry in diag[1]:
            assert entry["second_diff"] == pytest.approx(0.0, abs=1e-9)

    def test_step_function_shows_a_kink_at_the_boundary(self):
        # A step exactly at multiples of 5 produces a nonzero second
        # difference precisely at that boundary.
        cells = {1: {d: {"mean": float(d >= 5), "count": 5} for d in range(0, 11)}}
        diag = kink_diagnostic(cells, dollar_min=0, dollar_max=10)
        by_dollar = {e["dollar"]: e["second_diff"] for e in diag[1]}
        assert by_dollar[5] != 0.0

    def test_never_raises_on_missing_neighbors(self):
        cells = {1: {5: {"mean": 0.5, "count": 1}}}  # no d-1 or d+1
        diag = kink_diagnostic(cells, dollar_min=0, dollar_max=10)
        assert diag == {}


# ---------------------------------------------------------------------------
# Artifact assembly
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    def _artifact(self, corpus, blob_dir):
        tmp_path, rows = corpus
        critic = DollarProbeCritic(lambda d: 1.0 / (1.0 + np.exp(-d / 10.0)))
        dollar_values = list(range(-3, 4))
        acc, failures = run_sweep(rows, blob_dir, dollar_values, critic, batch_size=16)
        cells = acc.cells()
        return build_artifact(
            cells,
            acc,
            checkpoint="fake.zip",
            harvest_dir=str(tmp_path),
            source_filter="det",
            dollar_min=-3,
            dollar_max=3,
            n_source_records=len(rows),
            n_states=len(rows) - len(failures),
            max_states=None,
            seed=0,
            batch_size=16,
            sparse_threshold=1000,
            n_failures=len(failures),
        )

    def test_schema_has_the_documented_top_level_keys(self, corpus, blob_dir):
        artifact = self._artifact(corpus, blob_dir)
        assert set(artifact) == {"metadata", "cells", "gut_checks"}
        meta = artifact["metadata"]
        for key in (
            "checkpoint",
            "git_sha",
            "harvest_dir",
            "source_filter",
            "dollar_min",
            "dollar_max",
            "n_source_records",
            "n_states",
            "created_at",
        ):
            assert key in meta
        gc = artifact["gut_checks"]
        assert "hard" in gc and "sparse" in gc and "soft_kink_diagnostic" in gc
        assert "range" in gc["hard"] and "monotonicity_violations" in gc["hard"]

    def test_is_json_serializable_round_trip(self, corpus, blob_dir, tmp_path):
        artifact = self._artifact(corpus, blob_dir)
        out = tmp_path / "v_curve.json"
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        assert reloaded == artifact

    def test_sparse_threshold_1000_flags_every_ante(self, corpus, blob_dir):
        """With a deliberately absurd threshold, every ante in this tiny
        corpus should be flagged -- confirms the flag actually wires
        through build_artifact, not just the standalone function."""
        artifact = self._artifact(corpus, blob_dir)
        flagged_antes = {f["ante"] for f in artifact["gut_checks"]["sparse"]["flagged_antes"]}
        assert flagged_antes == {int(a) for a in artifact["cells"]}


# ---------------------------------------------------------------------------
# Loader helper (jackdaw.agents.v_curve)
# ---------------------------------------------------------------------------


def _write_artifact(path, cells_by_ante, dollar_min, dollar_max):
    artifact = {
        "metadata": {
            "dollar_min": dollar_min,
            "dollar_max": dollar_max,
        },
        "cells": {
            str(ante): {str(d): {"mean": mean, "count": 5} for d, mean in dmap.items()}
            for ante, dmap in cells_by_ante.items()
        },
        "gut_checks": {},
    }
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


class TestVCurveLoader:
    def test_exact_cell_lookup(self, tmp_path):
        path = _write_artifact(
            tmp_path / "v.json",
            {1: {-5: 0.01, 0: 0.2, 5: 0.4, 10: 0.6}},
            dollar_min=-5,
            dollar_max=10,
        )
        curve = load_v_curve(path)
        assert curve.value(1, 5) == pytest.approx(0.4)

    def test_clamps_above_swept_range(self, tmp_path):
        path = _write_artifact(
            tmp_path / "v.json", {1: {-5: 0.01, 10: 0.6}}, dollar_min=-5, dollar_max=10
        )
        curve = load_v_curve(path)
        assert curve.value(1, 10_000) == pytest.approx(0.6)

    def test_clamps_below_swept_range(self, tmp_path):
        path = _write_artifact(
            tmp_path / "v.json", {1: {-5: 0.01, 10: 0.6}}, dollar_min=-5, dollar_max=10
        )
        curve = load_v_curve(path)
        assert curve.value(1, -10_000) == pytest.approx(0.01)

    def test_rounds_to_nearest_whole_dollar(self, tmp_path):
        path = _write_artifact(
            tmp_path / "v.json", {1: {4: 0.4, 5: 0.5}}, dollar_min=0, dollar_max=10
        )
        curve = load_v_curve(path)
        assert curve.value(1, 4.6) == pytest.approx(0.5)  # rounds to 5
        assert curve.value(1, 4.4) == pytest.approx(0.4)  # rounds to 4

    def test_nearest_ante_fallback_when_ante_absent(self, tmp_path):
        path = _write_artifact(
            tmp_path / "v.json",
            {1: {5: 0.11}, 9: {5: 0.99}},
            dollar_min=0,
            dollar_max=10,
        )
        curve = load_v_curve(path)
        # ante 3 is closer to 1 (distance 2) than to 9 (distance 6).
        assert curve.value(3, 5) == pytest.approx(0.11)
        # ante 5 is a tie (distance 4 to both) -- min() over the sorted
        # ante list resolves the tie to the smaller ante.
        assert curve.value(5, 5) == pytest.approx(0.11)

    def test_nearest_dollar_fallback_for_a_missing_cell_within_range(self, tmp_path):
        path = _write_artifact(
            tmp_path / "v.json", {1: {0: 0.0, 10: 1.0}}, dollar_min=0, dollar_max=10
        )
        curve = load_v_curve(path)
        # dollar 6 isn't a stored cell (only 0 and 10 are); nearer to 10.
        assert curve.value(1, 6) == pytest.approx(1.0)

    def test_empty_cells_is_rejected(self):
        with pytest.raises(ValueError, match="no cells"):
            VCurve({}, dollar_min=0, dollar_max=10)

    def test_antes_property_reports_available_antes(self, tmp_path):
        path = _write_artifact(
            tmp_path / "v.json", {1: {0: 0.1}, 4: {0: 0.2}}, dollar_min=0, dollar_max=10
        )
        curve = load_v_curve(path)
        assert curve.antes == [1, 4]


# ---------------------------------------------------------------------------
# CellAccumulator basics
# ---------------------------------------------------------------------------


class TestCellAccumulator:
    def test_add_batch_matches_sequential_add(self):
        acc_a = CellAccumulator()
        acc_a.add(1, 0, 0.2)
        acc_a.add(1, 0, 0.4)
        acc_a.add(1, 1, 0.9)

        acc_b = CellAccumulator()
        acc_b.add_batch([(1, 0), (1, 0), (1, 1)], np.array([0.2, 0.4, 0.9]))

        assert acc_a.cells() == acc_b.cells()

    def test_tracks_raw_min_max_and_out_of_range_count(self):
        acc = CellAccumulator()
        acc.add(1, 0, -0.5)
        acc.add(1, 1, 0.5)
        acc.add(1, 2, 1.5)
        assert acc.raw_min == pytest.approx(-0.5)
        assert acc.raw_max == pytest.approx(1.5)
        assert acc.n_out_of_range == 2
        assert acc.n_total == 3
