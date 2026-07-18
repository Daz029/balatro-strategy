"""Tests for Candidate B's compound distribution and shared mask contract."""

from __future__ import annotations

from itertools import combinations, product

import numpy as np
import pytest
import torch

from jackdaw.agents.hand_pointer_head import (
    CARD_SLOTS,
    MAX_PICKS,
    STOP_INDEX,
    HandPointerBCModel,
    PointerActionHead,
    initial_type_mask,
    pick_step_mask,
)
from jackdaw.env.hand_play_gym import observation_space_v2
from jackdaw.env.observation import NUM_CENTER_KEYS


def _enumerate_sequences(hand_size: int) -> list[tuple[tuple[int, ...], ...]]:
    hand_mask = torch.zeros(CARD_SLOTS, dtype=torch.bool)
    hand_mask[:hand_size] = True
    sequences: list[tuple[tuple[int, ...], ...]] = []

    def visit(picks: tuple[int, ...], last_pick: int) -> None:
        if len(picks) == MAX_PICKS:
            sequences.append((picks,))
            return
        mask = pick_step_mask(hand_mask, last_pick, len(picks))
        if bool(mask[STOP_INDEX]):
            sequences.append((picks,))
        for index in torch.where(mask[:CARD_SLOTS])[0].tolist():
            visit((*picks, index), index)

    visit((), -1)
    return sequences


@pytest.mark.parametrize("hand_size", [3, 5, 8])
def test_mask_reachable_sequences_are_a_bijection_with_card_sets(hand_size: int):
    reachable = {picks for (picks,) in _enumerate_sequences(hand_size)}
    expected = {
        combo
        for size in range(1, min(5, hand_size) + 1)
        for combo in combinations(range(hand_size), size)
    }
    assert reachable == expected
    assert len(reachable) == len(expected)
    assert len(_enumerate_sequences(hand_size)) == len(expected)
    assert {(action_type, picks) for action_type, picks in product(range(2), reachable)} == {
        (action_type, combo)
        for action_type in range(2)
        for combo in expected
    }


def test_random_compound_distribution_normalizes_per_type_and_overall():
    torch.manual_seed(10)
    hand_size = 3
    n_sets = sum(
        len(tuple(combinations(range(hand_size), size)))
        for size in range(1, hand_size + 1)
    )
    labels = torch.full((2 * n_sets, MAX_PICKS), -1, dtype=torch.long)
    types = torch.repeat_interleave(torch.arange(2), n_sets)
    row = 0
    for action_type in range(2):
        for size in range(1, hand_size + 1):
            for combo in combinations(range(hand_size), size):
                labels[row, :size] = torch.tensor(combo)
                row += 1

    head = PointerActionHead()
    cards = torch.randn(1, CARD_SLOTS, 64).expand(len(labels), -1, -1)
    pooled = torch.randn(1, 256).expand(len(labels), -1)
    hand_mask = torch.zeros(len(labels), CARD_SLOTS, dtype=torch.bool)
    hand_mask[:, :hand_size] = True
    _, sequence_log_prob, _ = head.teacher_forced_log_probs(
        cards, pooled, hand_mask, 1, 1, types, labels
    )
    probabilities = sequence_log_prob.exp()
    type_probabilities = head.type_head(pooled[:1]).softmax(-1)
    assert torch.allclose(probabilities.sum(), torch.ones(()), atol=1e-5)
    assert torch.allclose(
        (probabilities[types == 0] / type_probabilities[0, 0]).sum(),
        torch.ones(()),
        atol=1e-5,
    )
    assert torch.allclose(
        (probabilities[types == 1] / type_probabilities[0, 1]).sum(),
        torch.ones(()),
        atol=1e-5,
    )


def test_teacher_forcing_and_rigged_greedy_decode_round_trip(monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(11)
    head = PointerActionHead()
    hand_mask = torch.zeros(1, CARD_SLOTS, dtype=torch.bool)
    hand_mask[0, :8] = True
    cards = torch.randn(1, CARD_SLOTS, 64)
    pooled = torch.randn(1, 256)
    label = torch.tensor([[1, 4, 7, -1, -1]])
    action_type = torch.tensor([1])
    _, sequence_log_prob, entropies = head.teacher_forced_log_probs(
        cards, pooled, hand_mask, 1, 1, action_type, label
    )
    assert torch.isfinite(sequence_log_prob).all()
    assert entropies.shape == (1, 6)

    target = [1, 4, 7, STOP_INDEX]
    call_count = 0

    def rigged_pointer_logits(state: torch.Tensor, card_latents: torch.Tensor) -> torch.Tensor:
        nonlocal call_count
        logits = torch.full((1, CARD_SLOTS + 1), -100.0, device=state.device)
        logits[:, target[min(call_count, len(target) - 1)]] = 100.0
        call_count += 1
        return logits

    monkeypatch.setattr(head, "_pointer_logits", rigged_pointer_logits)
    with torch.no_grad():
        head.type_head.weight.zero_()
        head.type_head.bias[:] = torch.tensor([-100.0, 100.0])
    decoded_type, decoded_indices = head.greedy_decode(cards, pooled, hand_mask, 1, 1)
    assert decoded_type.tolist() == [1]
    assert decoded_indices[0].tolist() == [1, 4, 7]


def test_monotone_mask_cap_and_stop_conventions():
    hand_mask = torch.ones(CARD_SLOTS, dtype=torch.bool)
    mask = pick_step_mask(hand_mask, 12, 1)
    assert not mask[:13].any()
    assert mask[13:CARD_SLOTS].all()

    head = PointerActionHead()
    cards = torch.randn(2, CARD_SLOTS, 64)
    pooled = torch.randn(2, 256)
    labels = torch.tensor([[0, 1, 2, 3, 4], [0, -1, -1, -1, -1]])
    hand = torch.ones(2, CARD_SLOTS, dtype=torch.bool)
    per_step, sequence, _ = head.teacher_forced_log_probs(
        cards, pooled, hand, 1, 1, torch.tensor([0, 0]), labels
    )
    assert per_step.shape == (2, 6)  # type + five pointer steps; no sixth stop step
    assert torch.allclose(sequence, per_step.sum(-1))

    exhausted_hand = torch.zeros(1, CARD_SLOTS, dtype=torch.bool)
    exhausted_hand[0, :2] = True
    exhausted_label = torch.tensor([[0, 1, -1, -1, -1]])
    exhausted_steps, _, _ = head.teacher_forced_log_probs(
        cards[:1], pooled[:1], exhausted_hand, 1, 1, torch.tensor([0]), exhausted_label
    )
    assert exhausted_steps[0, 3].item() == 0.0  # stop is the only legal token


def test_budget_masking_and_no_budget_error():
    assert initial_type_mask(0, 2).tolist() == [False, True]
    assert initial_type_mask(3, 0).tolist() == [True, False]
    with pytest.raises(ValueError, match="both.*zero"):
        initial_type_mask(0, 0)


def _assert_decodes_valid(
    decoded: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
    hand_masks: torch.Tensor,
    budgets: torch.Tensor,
) -> None:
    action_types, indices = decoded
    assert len(indices) == hand_masks.shape[0]
    for row, (action_type, picked) in enumerate(zip(action_types.tolist(), indices)):
        assert budgets[row, action_type].item()
        values = picked.tolist()
        assert 1 <= len(values) <= 5
        assert values == sorted(values)
        assert len(values) == len(set(values))
        assert all(hand_masks[row, index] for index in values)


def test_free_running_greedy_and_sample_are_valid_for_500_random_cases():
    torch.manual_seed(12)
    batch = 500
    head = PointerActionHead()
    cards = torch.randn(batch, CARD_SLOTS, 64)
    pooled = torch.randn(batch, 256)
    sizes = torch.randint(1, CARD_SLOTS + 1, (batch,))
    hand_masks = torch.arange(CARD_SLOTS).expand(batch, -1) < sizes[:, None]
    hands_left = torch.randint(0, 3, (batch,))
    discards_left = torch.randint(0, 3, (batch,))
    both_zero = (hands_left == 0) & (discards_left == 0)
    discards_left[both_zero] = 1
    budgets = torch.stack((hands_left >= 1, discards_left >= 1), dim=-1)

    _assert_decodes_valid(
        head.greedy_decode(cards, pooled, hand_masks, hands_left, discards_left),
        hand_masks,
        budgets,
    )
    _assert_decodes_valid(
        head.sample(cards, pooled, hand_masks, hands_left, discards_left),
        hand_masks,
        budgets,
    )


def test_greedy_decode_is_deterministic():
    torch.manual_seed(13)
    head = PointerActionHead()
    cards = torch.randn(8, CARD_SLOTS, 64)
    pooled = torch.randn(8, 256)
    hand_masks = torch.zeros(8, CARD_SLOTS, dtype=torch.bool)
    hand_masks[:, :5] = True
    first = head.greedy_decode(cards, pooled, hand_masks, 1, 1)
    second = head.greedy_decode(cards, pooled, hand_masks, 1, 1)
    assert torch.equal(first[0], second[0])
    assert all(torch.equal(left, right) for left, right in zip(first[1], second[1]))


def _model_obs(batch: int = 2) -> dict[str, torch.Tensor]:
    samples = [observation_space_v2().sample() for _ in range(batch)]
    obs = {
        key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
        for key in samples[0]
    }
    obs["hand_mask"].zero_()
    obs["hand_mask"][:, :5] = 1.0
    obs["joker_mask"].zero_()
    obs["joker_mask"][:, :2] = 1.0
    obs["consumable_mask"].zero_()
    obs["global_context"][:, 13] = 1.0
    obs["global_context"][:, 14] = 1.0
    obs["joker_ids"][:, :2] = torch.tensor([10, 11])
    obs["copy_target_ids"][:, :2] = torch.tensor([20, 21])
    obs["copy_active"][:, :2] = 1.0
    obs["trigger_match"].zero_()
    obs["trigger_match"][:, :5, :2, :] = 1.0
    obs["joker_ids"] = obs["joker_ids"].clamp(0, NUM_CENTER_KEYS).long()
    obs["copy_target_ids"] = obs["copy_target_ids"].clamp(0, NUM_CENTER_KEYS).long()
    return obs


def test_model_forward_gradient_reaches_embedding_and_gru():
    torch.manual_seed(14)
    model = HandPointerBCModel(observation_space_v2())
    obs = _model_obs()
    labels = torch.tensor([[0, 1, 2, -1, -1], [1, 2, 3, -1, -1]])
    types = torch.tensor([0, 1])
    sequence_log_prob, value = model(obs, types, labels)
    (-(sequence_log_prob.mean()) + value.square().mean()).backward()
    embedding_grad = model.features_extractor.embedding.weight.grad
    gru_grad = model.pointer_head.gru.weight_hh.grad
    assert embedding_grad is not None and embedding_grad.abs().sum() > 0
    assert gru_grad is not None and gru_grad.abs().sum() > 0
