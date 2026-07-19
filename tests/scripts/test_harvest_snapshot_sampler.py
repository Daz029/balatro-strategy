"""Tests for the harvested hand snapshot mixture sampler."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from harvest_restore import restore_state
from harvest_snapshot_sampler import HarvestSnapshotSampler

from jackdaw.engine.actions import GamePhase
from jackdaw.env.hand_play_adapter import HandPlayAdapter, HandPlayConfig
from jackdaw.env.hand_play_gym import HandPlayGymEnv


def _write_harvest(path: Path) -> None:
    blob_dir = path / "blobs"
    blob_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    shards: dict[str, dict[str, bytes]] = {}

    def add(run_seed: str, record_id: str, source: str, kind: str, dollars: int) -> None:
        adapter = HandPlayAdapter(
            HandPlayConfig(
                ante_range=(1, 1),
                hands_range=(1, 1),
                discards_range=(0, 0),
                dollars_range=(dollars, dollars),
            )
        )
        adapter.reset("b_red", 1, f"SYNTHETIC_{record_id}")
        raw_state = adapter.raw_state
        assert raw_state["phase"] == GamePhase.SELECTING_HAND
        assert "id" in raw_state["current_round"]["idol_card"]
        blob = adapter.snapshot_state()
        repaired = restore_state(blob)
        assert repaired["current_round"]["idol_card"]["id"] == (
            raw_state["current_round"]["idol_card"]["id"]
        )
        shards.setdefault(run_seed, {})[record_id] = blob
        rows.append(
            {"record_id": record_id, "run_seed": run_seed, "kind": kind, "source": source}
        )

    add("run_det", "det_0", "det", "hand", 7)
    add("run_sampled", "sampled_0", "sampled", "hand", 8)
    add("run_ignored", "ignored_0", "other", "hand", 90)
    add("run_shop", "shop_0", "det", "shop", 91)

    for run_seed, shard in shards.items():
        (blob_dir / f"{run_seed}.pkl").write_bytes(pickle.dumps(shard))
    (path / "metadata.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_anchor_fraction_and_seeded_sequence(tmp_path: Path) -> None:
    """The anchor count and full draw sequence are reproducible."""
    _write_harvest(tmp_path)
    first = HarvestSnapshotSampler(tmp_path, seed=123)
    second = HarvestSnapshotSampler(tmp_path, seed=123)

    first_draws = [first() for _ in range(400)]
    second_draws = [second() for _ in range(400)]

    assert sum(draw is None for draw in first_draws) == 214
    assert first_draws == second_draws


def test_filtering_and_draws_restore_into_v2_env(tmp_path: Path) -> None:
    """Only eligible hand records are returned and v2 accepts every draw."""
    _write_harvest(tmp_path)
    sampler = HarvestSnapshotSampler(tmp_path, config_anchor_frac=0.01, seed=4)
    env = HandPlayGymEnv(obs_version=2, action_version=2)
    observed_dollars: set[int] = set()

    for _ in range(200):
        blob = sampler()
        if blob is None:
            continue
        state = pickle.loads(blob)
        assert state["phase"] == GamePhase.SELECTING_HAND
        observed_dollars.add(state["dollars"])
        env.reset(options={"snapshot": blob})

    assert observed_dollars == {7, 8}


def test_env_mixture_has_config_and_restored_resets(tmp_path: Path) -> None:
    """The env receives both branches of the training mixture."""
    _write_harvest(tmp_path)
    sampler = HarvestSnapshotSampler(tmp_path, seed=123)
    env = HandPlayGymEnv(start_state_sampler=sampler)

    episode_seeds = [env.reset()[1]["episode_seed"] for _ in range(30)]

    assert "<restored>" in episode_seeds
    assert any(seed != "<restored>" for seed in episode_seeds)


def test_invalid_anchor_and_empty_corpus_are_rejected(tmp_path: Path) -> None:
    """The anchor is strictly nonzero and strictly below one."""
    _write_harvest(tmp_path)
    for fraction in (0.0, 1.0):
        try:
            HarvestSnapshotSampler(tmp_path, config_anchor_frac=fraction)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid anchor fraction {fraction}")

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "metadata.jsonl").write_text("", encoding="utf-8")
    (empty / "blobs").mkdir()
    try:
        HarvestSnapshotSampler(empty)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an empty harvest corpus")
