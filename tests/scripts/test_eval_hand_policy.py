"""Tests for version-aware hand policy evaluation."""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import eval_hand_policy  # noqa: E402
from eval_hand_policy import (  # noqa: E402
    _BCPolicy,
    _PointerBCPolicy,
    load_policy,
    run_suite,
)
from fingerprint_discard_bias import _GreedyGymPolicy, run_episodes  # noqa: E402
from generate_hand_demos import stage_presets  # noqa: E402
from train_hand_ppo_b import build_model  # noqa: E402

from jackdaw.agents.hand_pointer_head import HandPointerBCModel  # noqa: E402
from jackdaw.agents.hand_policy import HandPlayBCModel  # noqa: E402
from jackdaw.env.action_space import ActionType  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import observation_space, observation_space_v2  # noqa: E402


@pytest.fixture()
def v1_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "v1.pt"
    torch.save({"model_state_dict": HandPlayBCModel(observation_space()).state_dict()}, path)
    return path


@pytest.fixture()
def pointer_bc_checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "pointer.pt"
    torch.save(
        {
            "model_state_dict": HandPointerBCModel(observation_space_v2()).state_dict(),
            "metadata": {"head": "pointer"},
        },
        path,
    )
    return path


@pytest.fixture()
def pointer_ppo_checkpoint(pointer_bc_checkpoint: Path, tmp_path: Path) -> Path:
    model = build_model(
        pointer_bc_checkpoint,
        HandPlayConfig(),
        seed=7,
        n_envs=1,
        n_steps=8,
        batch_size=8,
        device="cpu",
    )
    path = tmp_path / "pointer_ppo"
    model.save(path)
    return path.with_suffix(".zip")


def _metadata_zip(path: Path, policy_class: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"policy_class": policy_class}))
    return path


def test_load_policy_dispatches_bc_heads(v1_checkpoint, pointer_bc_checkpoint):
    assert isinstance(load_policy(v1_checkpoint, "cpu"), _BCPolicy)
    assert isinstance(load_policy(pointer_bc_checkpoint, "cpu"), _PointerBCPolicy)


def test_load_policy_dispatches_zip_metadata_without_model_construction(tmp_path, monkeypatch):
    v1_zip = _metadata_zip(tmp_path / "v1.zip", "MaskableActorCriticPolicy")
    pointer_zip = _metadata_zip(tmp_path / "pointer.zip", "PointerPPOPolicy")
    monkeypatch.setattr(eval_hand_policy, "_PPOPolicy", lambda *args: "v1")
    monkeypatch.setattr(eval_hand_policy, "_PointerPPOPolicy", lambda *args: "pointer")
    assert load_policy(v1_zip, "cpu") == "v1"
    assert load_policy(pointer_zip, "cpu") == "pointer"


def test_load_policy_rejects_ambiguous_checkpoints(tmp_path):
    bad_pt = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}, "metadata": {"head": "flat"}}, bad_pt)
    with pytest.raises(ValueError):
        load_policy(bad_pt, "cpu")

    bad_zip = _metadata_zip(tmp_path / "bad.zip", "ActorCriticPolicy")
    with pytest.raises(ValueError):
        load_policy(bad_zip, "cpu")


@pytest.mark.parametrize("checkpoint_name", ["pointer_bc_checkpoint", "pointer_ppo_checkpoint"])
def test_pointer_run_suite_is_valid_and_deterministic(request, checkpoint_name):
    checkpoint = request.getfixturevalue(checkpoint_name)
    policy = load_policy(checkpoint, "cpu")
    result_a = run_suite(policy, HandPlayConfig(), n_episodes=3)
    result_b = run_suite(policy, HandPlayConfig(), n_episodes=3)
    assert set(result_a) == {"n_episodes", "clear_rate", "mean_steps"}
    assert result_a == result_b


class _ForcedPointerPolicy:
    obs_version = 2
    action_version = 2

    def __init__(self):
        self._calls = 0

    def act(self, obs):
        self._calls += 1
        if self._calls == 1:
            return np.array([int(ActionType.Discard), 0, 40, 40, 40, 40], dtype=np.int64)
        return np.array([int(ActionType.PlayHand), 0, 40, 40, 40, 40], dtype=np.int64)


def test_pointer_fingerprint_uses_type_token_for_first_discard():
    rows = run_episodes(lambda env: _ForcedPointerPolicy(), HandPlayConfig(), n_episodes=3)
    assert rows[0]["first_discard"] is True
    assert all("first_discard" in row for row in rows)


def test_fingerprint_greedy_control_runs_in_fingerprint_env():
    rows = run_episodes(lambda env: _GreedyGymPolicy(env), HandPlayConfig(), n_episodes=1)
    assert len(rows) == 1


def test_fingerprint_greedy_control_handles_h1_wide_stage1_hands():
    # The h1 regen preset's hand-size tail deliberately creates 9-12-card
    # hands. The greedy control must use the pointer action path, not the
    # frozen v1/436-action space whose positions stop at 7.
    config = stage_presets()["stage1_no_jokers"].config
    rows = run_episodes(lambda env: _GreedyGymPolicy(env), config, n_episodes=6)
    assert len(rows) == 6
