"""Public behavior tests for the v2 pointer PPO policy adapter."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from jackdaw.agents.hand_pointer_head import HandPointerBCModel
from jackdaw.agents.pointer_ppo_policy import (
    PointerPPOPolicy,
    _action_vector_from_decode,
    load_bc_weights_into_policy,
)
from jackdaw.env.hand_play_gym import observation_space_v2


def _obs_batch(batch_size: int = 32) -> dict[str, torch.Tensor]:
    space = observation_space_v2()
    samples = [space.sample() for _ in range(batch_size)]
    obs = {
        key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
        for key in samples[0]
    }
    generator = torch.Generator().manual_seed(101)
    sizes = torch.randint(1, 13, (batch_size,), generator=generator)
    obs["hand_mask"].zero_()
    obs["hand_mask"] = torch.arange(40).expand(batch_size, -1) < sizes[:, None]
    hands_left = torch.randint(0, 4, (batch_size,), generator=generator)
    discards_left = torch.randint(0, 4, (batch_size,), generator=generator)
    neither = (hands_left == 0) & (discards_left == 0)
    hands_left[neither] = 1
    obs["global_context"][:, 13] = hands_left.float() / 10
    obs["global_context"][:, 14] = discards_left.float() / 10
    obs["joker_mask"].zero_()
    obs["consumable_mask"].zero_()
    obs["joker_ids"] = obs["joker_ids"].long()
    obs["copy_target_ids"] = obs["copy_target_ids"].long()
    return obs


def _policy() -> PointerPPOPolicy:
    return PointerPPOPolicy(
        observation_space_v2(), spaces.MultiDiscrete([2] + [41] * 5), lambda _: 1e-3
    )


def test_sampled_actions_are_valid_and_evaluate_to_the_same_sequence_log_prob():
    torch.manual_seed(20)
    policy = _policy()
    obs = _obs_batch()
    actions, values, sampled_log_prob = policy.act(obs)
    evaluated_values, evaluated_log_prob, entropy = policy.evaluate_actions(obs, actions)

    assert actions.shape == (32, 6)
    assert values.shape == (32,)
    assert torch.allclose(sampled_log_prob, evaluated_log_prob)
    assert torch.isfinite(evaluated_values).all()
    assert torch.isfinite(entropy).all()
    for action, hand, global_context in zip(
        actions, obs["hand_mask"], obs["global_context"]
    ):
        assert int(action[0]) in (0, 1)
        assert int(global_context[13] * 10 + 0.5) >= (1 if int(action[0]) == 0 else 0)
        picks = action[1:].tolist()
        stop = picks.index(40) if 40 in picks else 5
        selected = picks[:stop]
        assert 1 <= len(selected) <= 5
        assert selected == sorted(selected)
        assert all(hand[index] for index in selected)
        assert all(token == 40 for token in picks[stop:])


def test_entropy_sum_is_the_manual_active_step_entropy():
    torch.manual_seed(21)
    policy = _policy()
    obs = _obs_batch(8)
    actions = policy.act(obs)[0]
    _, _, entropy_sum = policy.evaluate_actions(obs, actions)
    type_log_probs, pointer_log_probs, active = policy.teacher_forced_step_distributions(
        obs, actions
    )
    type_mask = torch.isfinite(type_log_probs)
    pointer_mask = torch.isfinite(pointer_log_probs)
    type_entropy = -(
        type_log_probs.exp()
        * torch.where(type_mask, type_log_probs, torch.zeros_like(type_log_probs))
    ).sum(-1)
    pointer_entropy = -(
        pointer_log_probs.exp()
        * torch.where(pointer_mask, pointer_log_probs, torch.zeros_like(pointer_log_probs))
    ).sum(-1)
    expected = type_entropy + (pointer_entropy * active[:, 1:]).sum(-1)
    assert torch.allclose(entropy_sum, expected)


def test_loading_complete_bc_weights_preserves_greedy_decode():
    torch.manual_seed(22)
    obs = _obs_batch(4)
    bc_model = HandPointerBCModel(observation_space_v2())
    policy = _policy()
    load_bc_weights_into_policy(policy, bc_model)
    action = policy.predict_deterministic(obs)
    types, picked = bc_model.decode(obs)
    expected = _action_vector_from_decode(types, picked)
    assert torch.equal(action, expected)
