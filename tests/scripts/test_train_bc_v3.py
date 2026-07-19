"""Synthetic tests for the shared v3 pointer/flat BC trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from train_bc import DemoDataset, split_train_val
from train_bc_v3 import (
    evaluate_pointer,
    mean_non_padding_token_ce,
    split_for_head,
    train,
)

from jackdaw.agents.hand_action_space import combo_to_action, legal_action_mask
from jackdaw.agents.hand_pointer_head import (
    HandPointerBCModel,
    PointerActionHead,
    pick_step_mask,
)
from jackdaw.agents.hand_policy_v3 import FlatV3BCModel
from jackdaw.env.hand_play_gym import observation_space_v2


def _synthetic_dataset(n: int = 64, *, repeated: bool = False) -> DemoDataset:
    space = observation_space_v2()
    # Seeded: gate tests assert verdict properties on models trained on this
    # data, and an unseeded space.sample() makes those verdicts per-process
    # coin flips (observed: order-dependent PASS/FAIL in test_eval_bc_gate).
    space.seed(1234 + n + int(repeated))
    samples = [space.sample() for _ in range(1 if repeated else n)]
    obs = {
        key: torch.as_tensor(np.stack([sample[key] for sample in samples])) for key in samples[0]
    }
    if repeated:
        obs = {key: value.expand(n, *value.shape[1:]).clone() for key, value in obs.items()}

    action_types = torch.zeros(n, dtype=torch.long)
    card_indices = torch.full((n, 5), -1, dtype=torch.long)
    actions = torch.full((n,), -1, dtype=torch.long)
    legal_masks = []
    for row in range(n):
        is_wide = not repeated and row % 2 == 1
        hand_size = 10 if is_wide else (1 if repeated else 8)
        obs["hand_mask"][row].zero_()
        obs["hand_mask"][row, :hand_size] = 1.0
        obs["global_context"][row, 13] = 0.1
        obs["global_context"][row, 14] = 0.0
        obs["joker_mask"][row].zero_()
        obs["consumable_mask"][row].zero_()
        obs["trigger_match"][row].zero_()
        card_indices[row, 0] = 8 if is_wide else 0
        if not is_wide:
            actions[row] = combo_to_action(0, (0,))
        legal_masks.append(legal_action_mask(hand_size, 1, 0))

    return DemoDataset(
        obs=obs,
        action_types=action_types,
        card_indices=card_indices,
        actions=actions,
        legal_masks=torch.as_tensor(np.stack(legal_masks)),
        p_clear=torch.zeros(n, dtype=torch.float32),
        sample_weights=torch.ones(n, dtype=torch.float32),
        seeds=[f"v3_synthetic_{row:04d}" for row in range(n)],
    )


def test_pointer_uniform_term_matches_forced_stop_and_hand_computation():
    torch.manual_seed(20)
    head = PointerActionHead()
    cards = torch.randn(1, 40, 64)
    pooled = torch.randn(1, 256)
    hand = torch.zeros(1, 40, dtype=torch.bool)
    hand[0, :2] = True
    labels = torch.tensor([[0, -1, -1, -1, -1]])
    per_step, _, _, uniform = head.teacher_forced_log_probs(
        cards,
        pooled,
        hand,
        1,
        0,
        torch.tensor([0]),
        labels,
        return_uniform_log_probs=True,
    )
    assert torch.equal(uniform[:, 1], per_step[:, 1]) is False

    state = head._state_from_type(pooled, torch.tensor([0]))
    mask = pick_step_mask(hand, -1, 0)
    logits = head._pointer_logits(state, cards)
    log_probs = torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
    expected = log_probs.masked_select(mask).mean()
    assert torch.equal(uniform[0, 1], expected)

    forced_hand = torch.zeros(1, 40, dtype=torch.bool)
    forced_hand[0, 0] = True
    forced_label = torch.tensor([[0, -1, -1, -1, -1]])
    forced_per_step, _, _, forced_uniform = head.teacher_forced_log_probs(
        cards,
        pooled,
        forced_hand,
        1,
        0,
        torch.tensor([0]),
        forced_label,
        return_uniform_log_probs=True,
    )
    assert forced_per_step[0, 1].item() == 0.0
    assert forced_uniform[0, 1].item() == 0.0


def test_pointer_arm_writes_checkpoint_and_history(tmp_path):
    dataset = _synthetic_dataset()
    checkpoint = train(
        dataset,
        tmp_path / "pointer",
        head="pointer",
        max_epochs=2,
        patience=2,
        batch_size=64,
        val_fraction=0.2,
        device_str="cpu",
        seed=3,
        data_dirs=["synthetic"],
    )
    metadata = torch.load(checkpoint, weights_only=False)["metadata"]
    assert checkpoint.name == "bc_v3_pointer.pt"
    assert metadata["head"] == "pointer"
    assert metadata["data_dirs"] == ["synthetic"]
    assert metadata["history"]
    assert "sequence_nll" in metadata["history"][0]
    assert metadata["canary_final_ce"] < 0.05
    assert metadata["canary_epochs"] >= 1
    assert metadata["canary_passed"] is True
    assert metadata["memorization_canary_mean_non_padding_token_ce"] == pytest.approx(
        metadata["canary_final_ce"]
    )


def test_flat_arm_drops_exactly_wide_rows_and_writes_checkpoint(tmp_path):
    dataset = _synthetic_dataset()
    checkpoint = train(
        dataset,
        tmp_path / "flat",
        head="flat",
        max_epochs=2,
        patience=2,
        batch_size=64,
        val_fraction=0.2,
        device_str="cpu",
        seed=3,
    )
    metadata = torch.load(checkpoint, weights_only=False)["metadata"]
    expected_count = int((dataset.actions < 0).sum())
    assert checkpoint.name == "bc_v3_flat.pt"
    assert metadata["head"] == "flat"
    assert metadata["dropped_wide_count"] == expected_count
    assert metadata["dropped_wide_fraction"] == pytest.approx(expected_count / len(dataset))
    assert metadata["num_train"] + metadata["num_val"] == len(dataset) - expected_count
    assert not any("canary" in key for key in metadata)


def test_pointer_canary_does_not_change_saved_model_weights(tmp_path):
    dataset = _synthetic_dataset(32, repeated=True)
    common = {
        "head": "pointer",
        "max_epochs": 2,
        "patience": 2,
        "batch_size": 32,
        "val_fraction": 0.2,
        "device_str": "cpu",
        "seed": 8,
    }
    with_canary = train(dataset, tmp_path / "with_canary", **common)
    without_canary = train(dataset, tmp_path / "without_canary", _run_canary=False, **common)
    with_state = torch.load(with_canary, weights_only=False)["model_state_dict"]
    without_state = torch.load(without_canary, weights_only=False)["model_state_dict"]
    assert with_state.keys() == without_state.keys()
    assert all(torch.equal(with_state[key], without_state[key]) for key in with_state)


def test_pointer_memorization_canary_uses_non_padding_token_ce(tmp_path):
    dataset = _synthetic_dataset(32, repeated=True)
    checkpoint = train(
        dataset,
        tmp_path / "canary",
        head="pointer",
        max_epochs=50,
        patience=50,
        batch_size=32,
        lr=3e-3,
        val_fraction=0.2,
        device_str="cpu",
        seed=4,
    )
    metadata = torch.load(checkpoint, weights_only=False)["metadata"]
    train_set, _ = split_for_head(dataset, "pointer", 0.2)
    model = HandPointerBCModel(observation_space_v2())
    model.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"])
    metrics = evaluate_pointer(model, train_set, 32, torch.device("cpu"))
    assert metrics["mean_non_padding_token_ce"] < 0.05
    assert metadata["history"]


def test_split_membership_is_shared_before_flat_filtering():
    dataset = _synthetic_dataset()
    raw_train, raw_val = split_train_val(dataset, 0.2)
    pointer_train, pointer_val = split_for_head(dataset, "pointer", 0.2)
    flat_train, flat_val = split_for_head(dataset, "flat", 0.2)
    assert pointer_train.seeds == raw_train.seeds
    assert pointer_val.seeds == raw_val.seeds
    expected_train = [
        seed for seed, action in zip(raw_train.seeds, raw_train.actions) if action >= 0
    ]
    expected_val = [seed for seed, action in zip(raw_val.seeds, raw_val.actions) if action >= 0]
    assert flat_train.seeds == expected_train
    assert flat_val.seeds == expected_val


def test_checkpoint_round_trip_preserves_pointer_validation_metrics(tmp_path):
    dataset = _synthetic_dataset()
    checkpoint = train(
        dataset,
        tmp_path / "round_trip",
        head="pointer",
        max_epochs=2,
        patience=2,
        batch_size=64,
        val_fraction=0.2,
        device_str="cpu",
        seed=5,
    )
    payload = torch.load(checkpoint, weights_only=False)
    model = FlatV3BCModel(observation_space_v2())
    assert model.action_head.out_features == 436
    pointer_model = HandPointerBCModel(observation_space_v2())
    pointer_model.load_state_dict(payload["model_state_dict"])
    _, val_set = split_for_head(dataset, "pointer", 0.2)
    metrics = evaluate_pointer(pointer_model, val_set, 64, torch.device("cpu"))
    best = min(payload["metadata"]["history"], key=lambda row: row["sequence_nll"])
    for key in ("sequence_nll", "exact_set_match_accuracy", "mean_per_step_entropy", "value_mse"):
        assert metrics[key] == pytest.approx(best[key], abs=1e-7)


def test_canary_metric_excludes_padding_steps():
    per_step = torch.tensor([[-1.0, -2.0, 0.0, 0.0, 0.0, 0.0]])
    labels = torch.tensor([[0, -1, -1, -1, -1]])
    assert mean_non_padding_token_ce(per_step, labels).item() == pytest.approx(1.0)
