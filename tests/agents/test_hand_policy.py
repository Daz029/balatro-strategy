"""Tests for the shared hand-policy network and BC->PPO weight transfer.

The transfer test is the load-bearing one: it builds a real
MaskableActorCriticPolicy exactly the way scripts/train_hand_ppo.py does,
loads BC weights into it, and asserts the two produce IDENTICAL masked
action distributions and values on real environment observations. If SB3's
policy decomposition ever drifts (net_arch semantics, extractor sharing),
this fails loudly before a training run silently starts from random
weights.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy  # noqa: E402

from jackdaw.agents.hand_action_space import NUM_HAND_ACTIONS  # noqa: E402
from jackdaw.agents.hand_policy import (  # noqa: E402
    HandPlayBCModel,
    HandPlayFeaturesExtractor,
    load_bc_weights_into_policy,
    masked_pool,
)
from jackdaw.env.hand_play_adapter import HandPlayConfig  # noqa: E402
from jackdaw.env.hand_play_gym import HandPlayGymEnv  # noqa: E402


def _obs_batch(n: int = 4) -> tuple[dict[str, torch.Tensor], torch.Tensor, HandPlayGymEnv]:
    """Real observations + action masks from seeded env resets."""
    env = HandPlayGymEnv(
        config=HandPlayConfig(joker_pool=("j_jolly", "j_photograph"), joker_count_range=(0, 2)),
        seed_prefix="POLTEST",
    )
    obs_list, mask_list = [], []
    for seed in range(n):
        obs, info = env.reset(seed=seed)
        obs_list.append(obs)
        mask_list.append(info["action_mask"])
    batch = {
        k: torch.as_tensor(np.stack([o[k] for o in obs_list])) for k in obs_list[0]
    }
    masks = torch.as_tensor(np.stack(mask_list))
    return batch, masks, env


class TestMaskedPool:
    def test_fully_masked_type_contributes_exact_zeros(self):
        x = torch.randn(3, 2, 8)
        mask = torch.zeros(3, 2)
        out = masked_pool(x, mask)
        assert out.shape == (3, 16)
        assert torch.equal(out, torch.zeros(3, 16))

    def test_partial_mask_ignores_padding(self):
        x = torch.randn(1, 4, 8)
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        out = masked_pool(x, mask)
        expected_mean = x[0, :2].mean(dim=0)
        expected_max = x[0, :2].amax(dim=0)
        assert torch.allclose(out[0, :8], expected_mean)
        assert torch.allclose(out[0, 8:], expected_max)


class TestBCModel:
    def test_forward_shapes(self):
        batch, masks, env = _obs_batch(4)
        model = HandPlayBCModel(env.observation_space)
        logits, values = model(batch)
        assert logits.shape == (4, NUM_HAND_ACTIONS)
        assert values.shape == (4,)

    def test_masked_log_probs_zero_out_illegal(self):
        batch, masks, env = _obs_batch(4)
        model = HandPlayBCModel(env.observation_space)
        log_probs = model.masked_log_probs(batch, masks)
        assert torch.isneginf(log_probs[~masks]).all()
        probs = log_probs.exp()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)


class TestBCToPPOTransfer:
    def _build_policy(self, env: HandPlayGymEnv) -> MaskableActorCriticPolicy:
        # Mirrors scripts/train_hand_ppo.py's policy_kwargs exactly.
        return MaskableActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lambda _: 3e-5,
            net_arch=[],
            features_extractor_class=HandPlayFeaturesExtractor,
        )

    def test_transfer_reproduces_bc_distribution_and_value(self):
        batch, masks, env = _obs_batch(4)
        bc_model = HandPlayBCModel(env.observation_space)
        policy = self._build_policy(env)
        load_bc_weights_into_policy(policy, bc_model)
        policy.set_training_mode(False)

        with torch.no_grad():
            bc_log_probs = bc_model.masked_log_probs(batch, masks)
            bc_values = bc_model(batch)[1]
            dist = policy.get_distribution(batch, action_masks=masks.numpy())
            ppo_log_probs = dist.distribution.logits  # normalized log-probs
            ppo_values = policy.predict_values(batch).squeeze(-1)

        legal = masks
        assert torch.allclose(bc_log_probs[legal], ppo_log_probs[legal], atol=1e-5)
        assert torch.allclose(bc_values, ppo_values, atol=1e-5)

    def test_transfer_rejects_mismatched_architecture(self):
        _, _, env = _obs_batch(1)
        bc_model = HandPlayBCModel(env.observation_space)
        policy = MaskableActorCriticPolicy(
            env.observation_space,
            env.action_space,
            lambda _: 3e-5,
            net_arch=[64],  # WRONG: adds an mlp_extractor layer
            features_extractor_class=HandPlayFeaturesExtractor,
        )
        with pytest.raises(RuntimeError):
            load_bc_weights_into_policy(policy, bc_model)
