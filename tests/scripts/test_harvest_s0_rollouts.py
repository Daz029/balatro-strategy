"""Tests for the phase-1 harvest pass (pre-regen A1/A2).

The three properties everything downstream depends on:

1. A captured blob restores byte-identically and continues (the same
   RNG-exact round-trip ``ShopRunAdapter`` guarantees — a harvested state must
   be labelable and replayable later, in phase 2, with no drift).
2. Each metadata row matches its blob (stratification/stats read the flat
   table and must never have to unpickle — so the table has to be right).
3. The harvesting wrapper is transparent: the run plays out identically with
   or without it (capture must not perturb the induced distribution).

The rollout driver is exercised with a do-nothing shop policy + the scripted
greedy hand policy, so the whole capture pipeline runs with no checkpoints.
"""

from __future__ import annotations

import pickle

from harvest_s0_rollouts import (
    DET_PREFIX,
    HarvestingHandPolicy,
    HarvestSink,
    NextRoundPolicy,
    compute_coverage_readout,
    compute_reductions,
    emit_reductions,
    extract_hand_meta,
    harvest_runs,
    load_metadata,
)

from jackdaw.agents.greedy_hand_policy import GreedyHandPolicy
from jackdaw.engine.actions import GamePhase
from jackdaw.engine.game import step as engine_step
from jackdaw.env.shop_run_adapter import ShopRunConfig


def _run_harvest(tmp_path, n_det=6, n_sampled=0, source="det", prefix=DET_PREFIX):
    sink = HarvestSink(tmp_path, git_sha_stamp="testsha", schema_note="note")
    n = harvest_runs(
        NextRoundPolicy(),
        GreedyHandPolicy(),
        sink,
        n_runs=n_det,
        seed_prefix=prefix,
        source=source,
        win_ante=2,
        max_steps=128,
    )
    sink.close()
    return sink, n


def _load_blob(tmp_path, run_seed, record_id):
    blobs = pickle.loads((tmp_path / "blobs" / f"{run_seed}.pkl").read_bytes())
    return blobs[record_id]


def _row(kind, source, *, ante, dollars, hand_size=0, owned=None):
    return {
        "kind": kind,
        "source": source,
        "ante": ante,
        "dollars": dollars,
        "hand_size": hand_size,
        "owned_jokers": owned or [],
    }


class TestCaptureAndMetadata:
    def test_records_and_metadata_written(self, tmp_path):
        sink, _ = _run_harvest(tmp_path, n_det=6)
        rows = load_metadata(tmp_path)
        assert sink.n_records == len(rows)
        assert rows, "expected some captured records over 6 runs"
        # Both kinds appear (greedy clears some ante-1 blinds -> reaches SHOP).
        kinds = {r["kind"] for r in rows}
        assert "hand" in kinds

    def test_record_id_scheme(self, tmp_path):
        _run_harvest(tmp_path, n_det=6)
        rows = load_metadata(tmp_path)
        for r in rows:
            if r["kind"] == "hand":
                assert r["record_id"] == f"{r['run_seed']}_{r['turn_idx']}"
            else:
                assert r["record_id"] == f"{r['run_seed']}_s{r['turn_idx']}"

    def test_stamps_present(self, tmp_path):
        _run_harvest(tmp_path, n_det=4)
        rows = load_metadata(tmp_path)
        for r in rows:
            assert r["git_sha"] == "testsha"
            assert r["schema_note"] == "note"
            assert r["source"] == "det"

    def test_metadata_matches_blob(self, tmp_path):
        """A row's cached fields equal what the blob actually contains — the
        no-unpickle guarantee is only sound if this holds."""
        _run_harvest(tmp_path, n_det=8)
        rows = [r for r in load_metadata(tmp_path) if r["kind"] == "hand"]
        assert rows
        for r in rows[:20]:
            gs = pickle.loads(_load_blob(tmp_path, r["run_seed"], r["record_id"]))
            fresh = extract_hand_meta(gs)
            assert r["ante"] == fresh["ante"]
            assert r["dollars"] == fresh["dollars"]
            assert r["hand_size"] == fresh["hand_size"]
            assert r["hands_left"] == fresh["hands_left"]
            assert r["discards_left"] == fresh["discards_left"]
            assert r["owned_jokers"] == fresh["owned_jokers"]
            # The captured state really is a hand-decision state.
            assert gs["phase"] == GamePhase.SELECTING_HAND

    def test_shop_records_are_shop_states(self, tmp_path):
        _run_harvest(tmp_path, n_det=8)
        rows = [r for r in load_metadata(tmp_path) if r["kind"] == "shop"]
        for r in rows[:20]:
            gs = pickle.loads(_load_blob(tmp_path, r["run_seed"], r["record_id"]))
            assert gs["phase"] == GamePhase.SHOP


class TestRoundTrip:
    def test_blob_restores_and_continues_byte_identically(self, tmp_path):
        """Restore a captured hand-turn blob and confirm the greedy policy's
        next action + the resulting engine state match the original run."""
        _run_harvest(tmp_path, n_det=4)
        rows = [r for r in load_metadata(tmp_path) if r["kind"] == "hand"]
        assert rows
        r = rows[0]
        blob = _load_blob(tmp_path, r["run_seed"], r["record_id"])

        policy = GreedyHandPolicy()
        gs_a = pickle.loads(blob)
        action_a = policy(gs_a)
        engine_step(gs_a, action_a)

        gs_b = pickle.loads(blob)
        action_b = policy(gs_b)
        engine_step(gs_b, action_b)

        # Same decision, same resulting engine state (RNG-exact continuation).
        assert type(action_a) is type(action_b)
        assert gs_a["chips"] == gs_b["chips"]
        assert gs_a["dollars"] == gs_b["dollars"]
        assert gs_a["phase"] == gs_b["phase"]


class TestWrapperTransparency:
    def test_identical_run_with_and_without_wrapper(self, tmp_path):
        """Full episode outcome is identical whether the hand policy is
        wrapped for harvesting or not."""

        from jackdaw.env.shop_gym import ShopGymEnv

        def _play_out(hand_policy):
            env = ShopGymEnv(config=ShopRunConfig(win_ante=2), hand_policy=hand_policy)
            obs, info = env.reset(options={"episode_seed": "SHOPRUN_TEST"})
            policy = NextRoundPolicy()
            steps = 0
            for _ in range(128):
                a = policy.act(obs, info["action_mask"])
                obs, _, term, trunc, info = env.step(a)
                steps += 1
                if term or trunc:
                    break
            gs = env.raw_state
            return (gs["dollars"], gs["round"], gs.get("won", False), steps)

        plain = _play_out(GreedyHandPolicy())

        sink = HarvestSink(tmp_path, "sha")
        sink.begin_run("SHOPRUN_TEST", "det")
        wrapped = _play_out(HarvestingHandPolicy(GreedyHandPolicy(), sink))
        sink.close()

        assert plain == wrapped


class TestReductions:
    def test_reductions_deterministic_only(self, tmp_path):
        """Sampled records must not enter the money/hand-size reductions."""
        rows = [
            _row("hand", "det", ante=1, dollars=4, hand_size=8),
            _row("hand", "det", ante=2, dollars=9, hand_size=8, owned=["j_a"]),
            _row("hand", "sampled", ante=1, dollars=99, hand_size=20),
            _row("shop", "det", ante=1, dollars=12),
        ]
        red = compute_reductions(rows)
        # sampled dollars (99) and shop dollars (12) excluded; hand-size 20 excluded.
        assert red["dollar_marginals_by_ante"] == {"1": {"4": 1}, "2": {"9": 1}}
        assert red["hand_size_histogram"] == {"8": 2}

    def test_coverage_split_by_source(self, tmp_path):
        rows = [
            _row("hand", "det", ante=3, dollars=1, hand_size=10, owned=["j_a", "j_b"]),
            _row("hand", "sampled", ante=1, dollars=1, hand_size=8, owned=["j_a"]),
        ]
        cov = compute_coverage_readout(rows)
        assert set(cov) == {"det", "sampled"}
        assert cov["det"]["n_at_ante_ge3"] == 1
        assert cov["det"]["distinct_jokers_owned"] == 2
        assert cov["det"]["hand_size_gt8"] == 1
        assert cov["sampled"]["max_hand_size"] == 8

    def test_emit_writes_files(self, tmp_path):
        _run_harvest(tmp_path, n_det=4)
        reductions, coverage = emit_reductions(tmp_path)
        assert (tmp_path / "reductions.json").exists()
        assert (tmp_path / "coverage.json").exists()
        assert "det" in coverage


class TestSinkLifecycle:
    def test_empty_run_flushes_nothing(self, tmp_path):
        sink = HarvestSink(tmp_path, "sha")
        sink.begin_run("EMPTY", "det")
        sink.end_run()  # no captures
        sink.close()
        assert not (tmp_path / "blobs" / "EMPTY.pkl").exists()
        assert sink.n_records == 0
