"""Tests for C1 (`scripts/select_harvest_manifest.py`).

The manifest is a checked-in versioned artifact that decides which ~8k of the
harvested corpus gets ~12s/example of solver time, so the properties under test
are: byte-level determinism (a dataset must be traceable to the exact query
that produced it), the within-run correlation caps actually binding, and the
seed-hygiene guard that keeps the held-out EVAL_ suite out of training data.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest
from select_harvest_manifest import (
    SelectionError,
    build_candidate_pool,
    build_manifest,
    hand_records,
    load_manifest,
    select_records,
    thin_to_target,
    write_manifest,
)


def _record(
    run_seed: str,
    turn_idx: int,
    ante: int,
    *,
    source: str = "det",
    kind: str = "hand",
    git_sha: str = "abc123",
) -> dict:
    return {
        "record_id": f"{run_seed}_{turn_idx}",
        "run_seed": run_seed,
        "turn_idx": turn_idx,
        "kind": kind,
        "source": source,
        "git_sha": git_sha,
        "ante": ante,
        "blind_type": "Small",
        "boss_key": "",
        "hand_size": 8,
        "dollars": 4,
        "hands_left": 4,
        "discards_left": 3,
        "owned_jokers": [],
    }


def _corpus(
    n_runs: int = 40,
    *,
    source: str = "det",
    turns_per_ante: int = 10,
    max_ante: int = 3,
) -> list[dict]:
    """A synthetic corpus shaped like the real one: many correlated turns per
    (run, ante), several antes per run."""
    rows: list[dict] = []
    for run in range(n_runs):
        run_seed = f"HARVEST_{run:08d}"
        turn = 0
        for ante in range(1, max_ante + 1):
            for _ in range(turns_per_ante):
                rows.append(_record(run_seed, turn, ante, source=source))
                turn += 1
    return rows


class TestFiltering:
    def test_shop_records_are_not_selectable(self):
        rows = _corpus(4) + [
            _record("HARVEST_00000000", 99, 1, kind="shop"),
            _record("HARVEST_00000001", 99, 2, kind="shop"),
        ]
        assert all(r["kind"] == "hand" for r in hand_records(rows))

    def test_eval_seeds_are_rejected_loudly(self):
        """Pitfall #6: an EVAL_ record in the manifest leaks the held-out
        policy-eval suite into h1's training data. It must raise, not skip."""
        rows = _corpus(4) + [_record("EVAL_00000007", 0, 1)]
        with pytest.raises(SelectionError, match="EVAL_"):
            hand_records(rows)

    def test_empty_corpus_raises(self):
        with pytest.raises(SelectionError, match="no hand records"):
            select_records([_record("HARVEST_00000000", 0, 1, kind="shop")])


class TestCaps:
    def test_per_ante_cap_binds(self):
        rows = _corpus(20, turns_per_ante=10, max_ante=3)
        pool = build_candidate_pool(
            rows, per_run_cap=99, per_ante_cap=3, rng=random.Random(0)
        )
        buckets = Counter((r["run_seed"], r["ante"]) for r in pool)
        assert buckets, "expected a non-empty pool"
        assert max(buckets.values()) <= 3

    def test_per_run_cap_binds(self):
        rows = _corpus(20, turns_per_ante=10, max_ante=3)
        pool = build_candidate_pool(
            rows, per_run_cap=8, per_ante_cap=3, rng=random.Random(0)
        )
        per_run = Counter(r["run_seed"] for r in pool)
        assert max(per_run.values()) <= 8

    def test_deepest_ante_first_when_run_cap_binds(self):
        """3 antes x cap 3 = 9 candidates into 8 slots: round-robin from the
        deepest ante means ante 1 (the abundant one) loses the slot, not the
        scarce deep ante."""
        rows = _corpus(1, turns_per_ante=10, max_ante=3)
        pool = build_candidate_pool(
            rows, per_run_cap=8, per_ante_cap=3, rng=random.Random(0)
        )
        by_ante = Counter(r["ante"] for r in pool)
        assert len(pool) == 8
        assert by_ante[3] == 3
        assert by_ante[2] == 3
        assert by_ante[1] == 2

    def test_run_shorter_than_cap_contributes_everything_it_has(self):
        """A run that died at ante 1 has only 3 capped candidates; it must not
        be padded, and must not be dropped."""
        rows = [_record("HARVEST_00000000", t, 1) for t in range(10)]
        pool = build_candidate_pool(
            rows, per_run_cap=8, per_ante_cap=3, rng=random.Random(0)
        )
        assert len(pool) == 3

    def test_bucket_draw_is_not_a_turn_prefix(self):
        """Early turns of a blind are systematically different states (full
        budgets) from later ones, so a prefix draw would bias the stage."""
        rows = [_record("HARVEST_00000000", t, 1) for t in range(20)]
        pool = build_candidate_pool(
            rows, per_run_cap=8, per_ante_cap=3, rng=random.Random(0)
        )
        assert [r["turn_idx"] for r in pool] != [0, 1, 2]


class TestThinning:
    def test_pool_below_target_passes_through(self):
        pool = [_record("HARVEST_00000000", t, 1) for t in range(5)]
        assert len(thin_to_target(pool, 10, random.Random(0))) == 5

    def test_thinning_hits_target_exactly(self):
        pool = [_record(f"HARVEST_{i:08d}", 0, 1 + i % 4) for i in range(100)]
        assert len(thin_to_target(pool, 37, random.Random(0))) == 37

    def test_thinning_preserves_ante_mix_and_keeps_scarce_buckets(self):
        """Largest-remainder stratification, not a uniform draw: a scarce deep
        bucket must survive rather than being randomly wiped."""
        pool = [_record(f"HARVEST_{i:08d}", 0, 1) for i in range(900)]
        pool += [_record(f"HARVEST_{900 + i:08d}", 0, 7) for i in range(4)]
        thinned = thin_to_target(pool, 452, random.Random(0))
        by_ante = Counter(r["ante"] for r in thinned)
        assert by_ante[7] == 2  # 4 * (452/904) == 2, exactly
        assert by_ante[1] == 450


class TestDeterminism:
    def test_same_seed_same_selection(self):
        rows = _corpus(30) + _corpus(10, source="sampled")
        a = select_records(rows, n_total=60, seed=0)
        b = select_records(rows, n_total=60, seed=0)
        assert [r["record_id"] for r in a] == [r["record_id"] for r in b]

    def test_selection_is_independent_of_input_row_order(self):
        """The metadata table's order is an artifact of capture scheduling; it
        must not leak into the manifest."""
        rows = _corpus(30) + _corpus(10, source="sampled")
        shuffled = list(rows)
        random.Random(99).shuffle(shuffled)
        a = select_records(rows, n_total=60, seed=0)
        b = select_records(shuffled, n_total=60, seed=0)
        assert [r["record_id"] for r in a] == [r["record_id"] for r in b]

    def test_different_seed_changes_selection(self):
        rows = _corpus(30) + _corpus(10, source="sampled")
        a = select_records(rows, n_total=60, seed=0)
        b = select_records(rows, n_total=60, seed=1)
        assert [r["record_id"] for r in a] != [r["record_id"] for r in b]

    def test_one_sources_supply_does_not_shift_the_others_draws(self):
        """Per-source RNG streams: adding sampled runs must not perturb which
        det records were chosen."""
        det = _corpus(30)
        a = select_records(det + _corpus(5, source="sampled"), n_total=60, seed=0)
        b = select_records(det + _corpus(20, source="sampled"), n_total=60, seed=0)
        det_a = sorted(r["record_id"] for r in a if r["source"] == "det")
        det_b = sorted(r["record_id"] for r in b if r["source"] == "det")
        assert det_a == det_b


class TestSourceRatio:
    def test_det_sampled_ratio_honoured_when_supply_allows(self):
        rows = _corpus(200) + _corpus(200, source="sampled")
        selected = select_records(rows, n_total=800, det_frac=0.75, seed=0)
        by_source = Counter(r["source"] for r in selected)
        assert by_source["det"] == 600
        assert by_source["sampled"] == 200

    def test_short_det_supply_is_not_backfilled_from_sampled(self):
        """Targets are ceilings. The 75:25 split is a deployed-distribution
        anchor, so topping det up with sampled records would silently make the
        stage MORE sampled-heavy than the decision that set the ratio."""
        rows = _corpus(2) + _corpus(200, source="sampled")
        selected = select_records(rows, n_total=800, det_frac=0.75, seed=0)
        by_source = Counter(r["source"] for r in selected)
        assert by_source["det"] < 600
        assert by_source["sampled"] == 200

    def test_bad_det_frac_raises(self):
        with pytest.raises(SelectionError, match="det-frac"):
            select_records(_corpus(4), n_total=10, det_frac=1.5)


class TestManifest:
    def _manifest(self, tmp_path, **kw):
        rows = _corpus(30) + _corpus(10, source="sampled")
        selected = select_records(rows, n_total=60, seed=0, **kw)
        return build_manifest(
            selected,
            params={"seed": 0, "n_total": 60},
            corpus={"n_hand_records": len(rows)},
        )

    def test_round_trips_as_valid_json(self, tmp_path):
        manifest = self._manifest(tmp_path)
        path = tmp_path / "m.json"
        write_manifest(path, manifest)
        loaded = load_manifest(path)
        assert loaded["records"] == manifest["records"]
        assert loaded["params"]["seed"] == 0
        assert loaded["stats"]["n_records"] == len(manifest["records"])

    def test_written_file_is_byte_identical_across_runs(self, tmp_path):
        """The artifact is checked in; a no-op re-run must produce a no-op diff
        (created_at excluded, since it is provenance, not selection)."""
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        for path in (a, b):
            m = self._manifest(tmp_path)
            m["created_at"] = "FIXED"
            write_manifest(path, m)
        assert a.read_bytes() == b.read_bytes()

    def test_records_carry_run_seed_explicitly(self, tmp_path):
        """C2 resolves blobs/{run_seed}.pkl from this; both seed prefixes
        (HARVEST_ / HARVEST_S_) contain underscores, so it is carried, not
        parsed back out of record_id."""
        manifest = self._manifest(tmp_path)
        for record in manifest["records"]:
            assert record["run_seed"]
            assert record["record_id"].startswith(record["run_seed"] + "_")

    def test_records_carry_git_sha_for_the_skew_check(self, tmp_path):
        manifest = self._manifest(tmp_path)
        assert all(r["git_sha"] for r in manifest["records"])

    def test_record_ids_are_unique(self, tmp_path):
        manifest = self._manifest(tmp_path)
        ids = [r["record_id"] for r in manifest["records"]]
        assert len(ids) == len(set(ids))

    def test_one_line_per_record(self, tmp_path):
        """Keeps an 8k-record artifact's diffs readable."""
        manifest = self._manifest(tmp_path)
        path = tmp_path / "m.json"
        write_manifest(path, manifest)
        body = path.read_text(encoding="utf-8")
        assert body.count('{"ante"') == len(manifest["records"])

    def test_order_is_shuffled_not_grouped_by_run(self, tmp_path):
        """C2 partitions workers over manifest order; grouped order would pile
        the slow deep-ante states into one worker."""
        manifest = self._manifest(tmp_path)
        seeds = [r["run_seed"] for r in manifest["records"]]
        assert seeds != sorted(seeds)
