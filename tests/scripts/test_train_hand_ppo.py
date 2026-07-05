"""Tests for the PPO fine-tune pipeline (`scripts/train_hand_ppo.py`).

The critical invariants:
- BC weights actually land in the PPO policy (KL(pi||pi_BC) == 0 at init,
  end-to-end through build_model, not just the unit-level transfer test)
- the copied-from-sb3 train() keeps the KL-leash path live through a real
  learn() call (kl multiplier adapts / kl stats populated)
- the sb3-contrib version pin: train() was copied from 2.7.1; a version
  bump must fail here loudly so someone re-diffs the method
- eval suite determinism on the reserved EVAL_ seed stream
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sb3_contrib = pytest.importorskip("sb3_contrib")

from eval_hand_policy import eval_seeds, load_policy, run_suite  # noqa: E402
from train_hand_ppo import KLToBCMaskablePPO, build_model  # noqa: E402

from jackdaw.agents.hand_policy import HandPlayBCModel  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import HandPlayGymEnv, observation_space  # noqa: E402


@pytest.fixture(scope="module")
def bc_checkpoint(tmp_path_factory):
    """A BC checkpoint with untrained-but-real weights (transfer semantics
    don't depend on training quality)."""
    torch.manual_seed(0)
    model = HandPlayBCModel(observation_space())
    path = tmp_path_factory.mktemp("bc") / "bc_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": {}}, path)
    return path


@pytest.fixture(scope="module")
def tiny_model(bc_checkpoint):
    return build_model(
        bc_checkpoint,
        HandPlayConfig(),
        seed=0,
        n_envs=2,
        n_steps=32,
        batch_size=32,
        device="cpu",
    )


class TestVersionPin:
    def test_sb3_contrib_version_matches_copied_train(self):
        # train() in train_hand_ppo.py is copied from sb3-contrib 2.7.x.
        # If this fails: re-diff KLToBCMaskablePPO.train() against the new
        # sb3_contrib/ppo_mask/ppo_mask.py before bumping this assertion.
        assert sb3_contrib.__version__.startswith("2.7"), (
            f"sb3-contrib {sb3_contrib.__version__}: re-diff the copied train() "
            "in scripts/train_hand_ppo.py before updating this pin"
        )


class TestBCWarmStart:
    def _rollout_batch(self, model: KLToBCMaskablePPO, n: int = 6):
        env = HandPlayGymEnv(config=HandPlayConfig(), seed_prefix="PPOTEST")
        obs_list, mask_list = [], []
        for seed in range(n):
            obs, info = env.reset(seed=seed)
            obs_list.append(obs)
            mask_list.append(info["action_mask"])
        obs_t = {
            k: torch.as_tensor(np.stack([o[k] for o in obs_list]), device=model.device)
            for k in obs_list[0]
        }
        masks = torch.as_tensor(
            np.stack(mask_list), dtype=torch.float32, device=model.device
        )
        return obs_t, masks

    def test_kl_to_bc_is_zero_at_init(self, tiny_model):
        obs, masks = self._rollout_batch(tiny_model)
        kl = tiny_model._kl_to_bc(obs, masks)
        assert abs(kl.item()) < 1e-5

    def test_value_head_transferred(self, tiny_model):
        obs, _ = self._rollout_batch(tiny_model)
        with torch.no_grad():
            ppo_values = tiny_model.policy.predict_values(obs).squeeze(-1)
            bc_values = tiny_model.bc_model(obs)[1]
        assert torch.allclose(ppo_values, bc_values, atol=1e-5)

    def test_beta_schedule_decays_with_progress(self, tiny_model):
        tiny_model._current_progress_remaining = 1.0
        beta_start = tiny_model._kl_beta_eff
        tiny_model._current_progress_remaining = 0.0
        assert tiny_model._kl_beta_eff == 0.0
        tiny_model._current_progress_remaining = 1.0
        assert beta_start == tiny_model.kl_beta0 * tiny_model._kl_multiplier


class TestLearnSmoke:
    def test_one_update_exercises_kl_leash(self, tiny_model):
        params_before = [p.detach().clone() for p in tiny_model.policy.parameters()]
        tiny_model.learn(total_timesteps=64)  # exactly one rollout+update
        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, tiny_model.policy.parameters())
        )
        # The KL machinery ran: multiplier is a positive finite value that
        # the adaptation clamps into [0.1, 10].
        assert 0.1 <= tiny_model._kl_multiplier <= 10.0
        logs = tiny_model.logger.name_to_value
        assert "train/kl_bc" in logs
        assert np.isfinite(logs["train/kl_bc"])

    def test_learn_requires_bc_model(self, bc_checkpoint):
        model = build_model(
            bc_checkpoint,
            HandPlayConfig(),
            seed=1,
            n_envs=1,
            n_steps=8,
            batch_size=8,
            device="cpu",
        )
        model.bc_model = None
        with pytest.raises(AssertionError, match="set_bc_model"):
            model.learn(total_timesteps=8)


class TestEvalSuite:
    def test_eval_seeds_reserved_prefix(self):
        seeds = eval_seeds(3)
        assert seeds == ["EVAL_00000000", "EVAL_00000001", "EVAL_00000002"]

    def test_bc_policy_eval_deterministic(self, bc_checkpoint):
        policy = load_policy(bc_checkpoint, "cpu")
        result_a = run_suite(policy, HandPlayConfig(), n_episodes=4)
        result_b = run_suite(policy, HandPlayConfig(), n_episodes=4)
        assert result_a["clear_rate"] == result_b["clear_rate"]
        assert result_a["mean_steps"] == result_b["mean_steps"]
        assert 0.0 <= result_a["clear_rate"] <= 1.0
