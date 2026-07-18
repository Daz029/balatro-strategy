"""Tests for the pointer PPO KL-leash training script."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import stable_baselines3  # noqa: E402
from train_hand_ppo_b import (  # noqa: E402
    KLToBCPointerPPO,
    build_model,
    make_vec_env,
)

from jackdaw.agents.hand_pointer_head import HandPointerBCModel  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
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
