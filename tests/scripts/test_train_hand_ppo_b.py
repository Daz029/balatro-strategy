"""Tests for the pointer PPO KL-leash training script."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import stable_baselines3  # noqa: E402
from harvest_snapshot_sampler import HarvestSnapshotSampler  # noqa: E402
from train_hand_ppo_b import (  # noqa: E402
    KLToBCPointerPPO,
    build_model,
    load_trained_pointer_policy,
    main,
    make_vec_env,
)

from jackdaw.agents.hand_pointer_head import HandPointerBCModel  # noqa: E402
from jackdaw.agents.v_curve import VCurve  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayAdapter, HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import observation_space_v2  # noqa: E402


@pytest.fixture()
def bc_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(30)
    checkpoint = tmp_path / "bc_v3_pointer.pt"
    model = HandPointerBCModel(observation_space_v2())
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": {"head": "pointer"}},
        checkpoint,
    )
    return checkpoint


def _tiny_model(checkpoint: Path, *, n_steps: int = 8, n_envs: int = 2):
    return build_model(
        checkpoint,
        HandPlayConfig(),
        seed=31,
        n_envs=n_envs,
        n_steps=n_steps,
        batch_size=n_steps * n_envs,
        device="cpu",
    )


@pytest.fixture()
def trained_checkpoint(bc_checkpoint: Path, tmp_path: Path) -> Path:
    source = _tiny_model(bc_checkpoint)
    checkpoint = tmp_path / "trained_hand_agent.zip"
    source.save(checkpoint)
    return checkpoint


def _single_snapshot_harvest(path: Path) -> None:
    """Write one post-fix synthetic hand record."""
    blob_dir = path / "blobs"
    blob_dir.mkdir(parents=True)
    adapter = HandPlayAdapter(
        HandPlayConfig(ante_range=(1, 1), hands_range=(1, 1), discards_range=(0, 0))
    )
    adapter.reset("b_red", 1, "TRAINING_SYNTHETIC")
    raw_state = adapter.raw_state
    assert "id" in raw_state["current_round"]["idol_card"]
    record_id = "run_0"
    (blob_dir / "run.pkl").write_bytes(pickle.dumps({record_id: adapter.snapshot_state()}))
    (path / "metadata.jsonl").write_text(
        json.dumps(
            {"record_id": record_id, "run_seed": "run", "kind": "hand", "source": "det"}
        )
        + "\n",
        encoding="utf-8",
    )


def test_sb3_version_pin_matches_the_vendored_train_loop():
    assert stable_baselines3.__version__.startswith("2.7"), (
        f"stable-baselines3 {stable_baselines3.__version__}: re-diff "
        "KLToBCPointerPPO.train() before changing the pin"
    )


def test_pointer_kl_is_zero_for_self_and_positive_after_a_perturbation(bc_checkpoint):
    model = _tiny_model(bc_checkpoint)
    obs = model.env.reset()
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    actions = model.policy.act(obs_tensor)[0]
    assert model._kl_to_bc(obs_tensor, actions).item() == pytest.approx(0.0, abs=1e-7)
    with torch.no_grad():
        model.policy.pointer_head.type_head.bias[0] += 0.5
    assert model._kl_to_bc(obs_tensor, actions).item() > 0.0


def test_pointer_ppo_short_run_logs_kl_and_round_trips_checkpoint(bc_checkpoint, tmp_path):
    model = _tiny_model(bc_checkpoint, n_steps=16)
    model.learn(total_timesteps=256)
    logs = model.logger.name_to_value
    assert "train/kl_bc" in logs
    assert np.isfinite(logs["train/kl_bc"])
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())

    save_path = tmp_path / "pointer_ppo"
    model.save(save_path)
    loaded = KLToBCPointerPPO.load(
        save_path,
        env=make_vec_env(HandPlayConfig(), seed_prefix="LOAD", n_envs=2),
        device="cpu",
    )
    loaded.set_bc_model(model.bc_model)
    comparison_env = make_vec_env(HandPlayConfig(), seed_prefix="COMPARE", n_envs=2)
    comparison_obs = comparison_env.reset()
    original_action = model.policy.predict(comparison_obs, deterministic=True)[0]
    loaded_action = loaded.policy.predict(comparison_obs, deterministic=True)[0]
    assert np.array_equal(original_action, loaded_action)


def test_descendant_init_matches_source_and_logs_frozen_policy_kl(trained_checkpoint):
    source = load_trained_pointer_policy(trained_checkpoint)
    model = build_model(
        None,
        HandPlayConfig(),
        init_from=trained_checkpoint,
        seed=31,
        n_envs=2,
        n_steps=8,
        batch_size=16,
        device="cpu",
    )

    assert model.leash_policy is not None
    assert model.leash_policy is not model.policy
    assert not model.leash_policy.training
    assert all(not parameter.requires_grad for parameter in model.leash_policy.parameters())

    obs = model.env.reset()
    source_action = source.predict(obs, deterministic=True)[0]
    descendant_action = model.policy.predict(obs, deterministic=True)[0]
    assert np.array_equal(source_action, descendant_action)

    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    actions = model.policy.act(obs_tensor)[0]
    assert model._kl_to_bc(obs_tensor, actions).item() == pytest.approx(0.0, abs=1e-7)
    with torch.no_grad():
        model.policy.pointer_head.type_head.bias[0] += 0.5
    assert model._kl_to_bc(obs_tensor, actions).item() > 0.0

    model.learn(total_timesteps=128)
    logs = model.logger.name_to_value
    assert "train/kl_bc" in logs
    assert np.isfinite(logs["train/kl_bc"])
    assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())


def test_build_model_rejects_missing_or_multiple_reference_checkpoints(bc_checkpoint, tmp_path):
    with pytest.raises(AssertionError, match="exactly one"):
        build_model(None, HandPlayConfig(), device="cpu")
    with pytest.raises(AssertionError, match="exactly one"):
        build_model(
            bc_checkpoint,
            HandPlayConfig(),
            init_from=tmp_path / "trained.zip",
            device="cpu",
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["train_hand_ppo_b.py"],
        ["train_hand_ppo_b.py", "--bc-checkpoint", "model.pt", "--init-from", "model.zip"],
    ],
)
def test_main_requires_exactly_one_reference_checkpoint(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()


def test_build_model_keeps_training_knobs_out_of_eval(bc_checkpoint):
    """Training envs carry both optional inputs; eval defaults carry neither."""
    curve = VCurve({1: {0: 0.25}}, dollar_min=0, dollar_max=50)

    def sampler():
        return None

    model = build_model(
        bc_checkpoint,
        HandPlayConfig(),
        seed=32,
        n_envs=2,
        n_steps=8,
        batch_size=16,
        device="cpu",
        v_curve=curve,
        start_state_sampler=sampler,
    )

    assert all(env._v_curve is curve for env in model.env.envs)
    assert all(env._sampler is sampler for env in model.env.envs)

    eval_env = make_vec_env(HandPlayConfig(), seed_prefix="EVAL", n_envs=2)
    assert all(env._v_curve is None for env in eval_env.envs)
    assert all(env._sampler is None for env in eval_env.envs)


def test_pointer_ppo_short_run_with_harvest_sampler_and_v_curve(bc_checkpoint, tmp_path):
    """A short PPO rollout tolerates repaired snapshot starts and terminal value."""
    harvest_dir = tmp_path / "harvest"
    _single_snapshot_harvest(harvest_dir)
    sampler = HarvestSnapshotSampler(harvest_dir, seed=33)
    curve = VCurve({1: {0: 0.25}}, dollar_min=0, dollar_max=50)
    model = build_model(
        bc_checkpoint,
        HandPlayConfig(),
        seed=33,
        n_envs=2,
        n_steps=8,
        batch_size=16,
        device="cpu",
        v_curve=curve,
        start_state_sampler=sampler,
    )

    model.learn(total_timesteps=128)
