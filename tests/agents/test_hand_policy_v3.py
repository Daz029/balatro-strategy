"""Tests for the v3 hand trunk's identity and fixed-attention seam."""

from __future__ import annotations

import numpy as np
import torch

from jackdaw.agents.hand_policy_v3 import HandPlayFeaturesExtractorV3
from jackdaw.env.hand_play_gym import observation_space_v2
from jackdaw.env.observation import NUM_CENTER_KEYS


def _sample_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    space = observation_space_v2()
    samples = [space.sample() for _ in range(batch_size)]
    obs = {
        key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
        for key in samples[0]
    }
    for key in ("joker_ids", "copy_target_ids"):
        obs[key] = obs[key].clamp(0, NUM_CENTER_KEYS).to(torch.int64)
    return obs


def _real_batch() -> dict[str, torch.Tensor]:
    obs = _sample_batch(1)
    obs["hand_mask"].zero_()
    obs["hand_mask"][0, :3] = 1.0
    obs["joker_mask"].zero_()
    obs["joker_mask"][0, :3] = 1.0
    obs["consumable_mask"].zero_()
    obs["consumable_mask"][0, :2] = 1.0
    obs["joker_ids"][0, :3] = torch.tensor([10, 11, 12])
    obs["copy_target_ids"][0, :3] = torch.tensor([20, 21, 22])
    obs["copy_active"][0, :3] = 1.0
    obs["trigger_match"].zero_()
    return obs


def _outputs(model: HandPlayFeaturesExtractorV3, obs: dict[str, torch.Tensor]):
    model.eval()
    with torch.no_grad():
        return model(obs)


def test_forward_shapes_from_observation_space_samples():
    torch.manual_seed(1)
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    cards, pooled = _outputs(model, _sample_batch(4))
    assert cards.shape == (4, 40, 64)
    assert pooled.shape == (4, 256)


def test_masked_rows_are_invariant_in_both_outputs():
    torch.manual_seed(2)
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    obs = _real_batch()
    base_cards, base_pooled = _outputs(model, obs)

    perturbed = {key: value.clone() for key, value in obs.items()}
    perturbed["hand_cards"][0, 3:] = torch.randn_like(perturbed["hand_cards"][0, 3:])
    perturbed["joker_ids"][0, 3:] = torch.randint(1, NUM_CENTER_KEYS + 1, (12,))
    perturbed["copy_target_ids"][0, 3:] = torch.randint(1, NUM_CENTER_KEYS + 1, (12,))
    perturbed["jokers"][0, 3:] = torch.randn_like(perturbed["jokers"][0, 3:])
    perturbed["copy_active"][0, 3:] = 1.0
    perturbed["trigger_match"][0, 3:, :] = torch.randn_like(
        perturbed["trigger_match"][0, 3:, :]
    )
    perturbed["trigger_match"][0, :, 3:, :] = torch.randn_like(
        perturbed["trigger_match"][0, :, 3:, :]
    )
    changed_cards, changed_pooled = _outputs(model, perturbed)

    assert torch.equal(base_cards, changed_cards)
    assert torch.equal(base_pooled, changed_pooled)


def test_flipping_scored_trigger_changes_only_that_card():
    torch.manual_seed(3)
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    obs = _real_batch()
    base_cards, _ = _outputs(model, obs)
    changed = {key: value.clone() for key, value in obs.items()}
    changed["trigger_match"][0, 0, 0, 0] = 1.0
    changed_cards, _ = _outputs(model, changed)

    assert not torch.equal(base_cards[0, 0], changed_cards[0, 0])
    assert torch.equal(base_cards[0, 1], changed_cards[0, 1])
    assert torch.equal(base_cards[0, 2], changed_cards[0, 2])


def test_scored_and_held_channels_are_distinct_inputs():
    torch.manual_seed(4)
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    scored = _real_batch()
    scored["trigger_match"][0, 0, 0, 0] = 1.0
    held = {key: value.clone() for key, value in scored.items()}
    held["trigger_match"][0, 0, 0, 0] = 0.0
    held["trigger_match"][0, 0, 0, 1] = 1.0

    scored_cards, _ = _outputs(model, scored)
    held_cards, _ = _outputs(model, held)
    assert not torch.equal(scored_cards[0, 0], held_cards[0, 0])


def test_copy_fields_change_unmasked_joker_representation():
    torch.manual_seed(5)
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    obs = _real_batch()
    _, base_pooled = _outputs(model, obs)

    target_changed = {key: value.clone() for key, value in obs.items()}
    target_changed["copy_target_ids"][0, 0] = 25
    _, target_pooled = _outputs(model, target_changed)
    active_changed = {key: value.clone() for key, value in obs.items()}
    active_changed["copy_active"][0, 0] = 0.0
    _, active_pooled = _outputs(model, active_changed)

    assert not torch.equal(base_pooled, target_pooled)
    assert not torch.equal(base_pooled, active_pooled)


def test_padding_identity_is_exactly_zero():
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    ids = torch.tensor([0], dtype=torch.int64)
    identity = torch.cat([model.embedding(ids), model.descriptors[ids]], dim=-1)
    assert torch.equal(identity, torch.zeros(1, 40))


def test_embedding_gradients_flow_and_padding_row_stays_zero():
    torch.manual_seed(6)
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    obs = _real_batch()
    obs["trigger_match"][0, :3, :3] = 1.0
    cards, pooled = model(obs)
    (cards.sum() + pooled.sum()).backward()

    grad = model.embedding.weight.grad
    assert grad is not None
    used_ids = torch.tensor([10, 11, 12, 20, 21, 22])
    assert (grad[used_ids].abs().sum(dim=1) > 0).all()
    assert torch.equal(grad[0], torch.zeros_like(grad[0]))


def test_embedding_and_descriptor_vocab_sizes_are_pinned():
    model = HandPlayFeaturesExtractorV3(observation_space_v2())
    assert model.embedding.num_embeddings == NUM_CENTER_KEYS + 1
    assert model.descriptors.shape == (NUM_CENTER_KEYS + 1, 24)
