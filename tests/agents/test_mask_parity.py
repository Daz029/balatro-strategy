"""Mask-parity harness for BC and the future PPO/eval call sites."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_bc_v3  # noqa: E402

from jackdaw.agents import (  # noqa: E402
    hand_pointer_head,  # noqa: E402
    pointer_ppo_policy,  # noqa: E402
)
from jackdaw.agents.hand_checkpoint_policy import HandCheckpointPolicy  # noqa: E402
from jackdaw.agents.hand_pointer_head import (  # noqa: E402
    CARD_SLOTS,
    MAX_PICKS,
    STOP_INDEX,
    HandPointerBCModel,
    PointerActionHead,
)
from jackdaw.engine.actions import GamePhase, SelectBlind  # noqa: E402
from jackdaw.engine.game import step as engine_step  # noqa: E402
from jackdaw.engine.run_init import initialize_run  # noqa: E402
from jackdaw.env.hand_play_gym import observation_space_v2  # noqa: E402


def _teacher_forced_masks(hand: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    length = int((label >= 0).sum())
    last_pick = -1
    n_picked = 0
    masks = []
    for step in range(MAX_PICKS):
        mask_last = CARD_SLOTS - 1 if step > length else last_pick
        mask_count = MAX_PICKS - 1 if step > length else n_picked
        masks.append(train_bc_v3.pick_step_mask(hand, mask_last, mask_count))
        if step < length:
            last_pick = int(label[step])
            n_picked += 1
    return torch.stack(masks)


def _free_running_masks(hand: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    length = int((label >= 0).sum())
    last_pick = -1
    n_picked = 0
    active = True
    masks = []
    for step in range(MAX_PICKS):
        masks.append(
            train_bc_v3.pick_step_mask(
                hand,
                last_pick if active else CARD_SLOTS - 1,
                n_picked if active else MAX_PICKS - 1,
            )
        )
        if active and step < length:
            last_pick = int(label[step])
            n_picked += 1
        elif active:
            active = False
    return torch.stack(masks)


def _random_corpus(rows: int = 128):
    generator = torch.Generator().manual_seed(21)
    hands = []
    labels = []
    types = []
    hands_left = []
    discards_left = []
    for _ in range(rows):
        hand_size = int(torch.randint(1, 13, (), generator=generator))
        hand = torch.zeros(CARD_SLOTS, dtype=torch.bool)
        hand[:hand_size] = True
        hands.append(hand)
        hands_budget = int(torch.randint(0, 5, (), generator=generator))
        discards_budget = int(torch.randint(0, 5, (), generator=generator))
        if hands_budget == 0 and discards_budget == 0:
            hands_budget = 1
        hands_left.append(hands_budget)
        discards_left.append(discards_budget)
        if discards_budget == 0:
            types.append(0)
        elif hands_budget == 0:
            types.append(1)
        else:
            types.append(int(torch.randint(0, 2, (), generator=generator)))
        length = int(torch.randint(1, min(5, hand_size) + 1, (), generator=generator))
        picks = torch.randperm(hand_size, generator=generator)[:length].sort().values
        label = torch.full((MAX_PICKS,), -1, dtype=torch.long)
        label[:length] = picks
        labels.append(label)
    return (
        torch.stack(hands),
        torch.stack(labels),
        torch.tensor(types),
        torch.tensor(hands_left),
        torch.tensor(discards_left),
    )


def test_trainer_and_pointer_head_share_mask_function_objects():
    assert train_bc_v3.initial_type_mask is hand_pointer_head.initial_type_mask
    assert train_bc_v3.pick_step_mask is hand_pointer_head.pick_step_mask
    assert train_bc_v3.MASK_FUNCTIONS == (
        hand_pointer_head.initial_type_mask,
        hand_pointer_head.pick_step_mask,
    )


def test_teacher_forcing_and_free_running_masks_are_byte_identical(
    monkeypatch: pytest.MonkeyPatch,
):
    hands, labels, types, hands_left, discards_left = _random_corpus()
    head = PointerActionHead()
    cards = torch.randn(len(labels), CARD_SLOTS, 64)
    pooled = torch.randn(len(labels), 256)
    original_mask = hand_pointer_head.pick_step_mask
    recorded_masks: list[torch.Tensor] = []

    def recording_mask(*args, **kwargs):
        mask = original_mask(*args, **kwargs)
        recorded_masks.append(mask.detach().clone())
        return mask

    monkeypatch.setattr(hand_pointer_head, "pick_step_mask", recording_mask)
    head.teacher_forced_log_probs(
        cards,
        pooled,
        hands,
        hands_left,
        discards_left,
        types,
        labels,
    )
    teacher_masks = recorded_masks.copy()
    assert len(teacher_masks) == MAX_PICKS

    type_logits = torch.full((len(labels), 2), -100.0)
    type_logits.scatter_(1, types[:, None], 100.0)

    def rigged_type_logits(_pooled: torch.Tensor) -> torch.Tensor:
        return type_logits

    monkeypatch.setattr(head.type_head, "forward", rigged_type_logits)
    pointer_step = 0

    def rigged_pointer_logits(state: torch.Tensor, card_latents: torch.Tensor) -> torch.Tensor:
        nonlocal pointer_step
        logits = torch.full((len(labels), CARD_SLOTS + 1), -100.0, device=state.device)
        lengths = (labels >= 0).sum(dim=-1)
        target = torch.where(
            pointer_step < lengths,
            labels[:, pointer_step],
            torch.full_like(lengths, STOP_INDEX),
        ).to(state.device)
        logits.scatter_(1, target[:, None], 100.0)
        pointer_step += 1
        return logits

    monkeypatch.setattr(head, "_pointer_logits", rigged_pointer_logits)
    recorded_masks.clear()
    decoded_types, decoded_indices = head.greedy_decode(
        cards, pooled, hands, hands_left, discards_left
    )
    free_masks = recorded_masks.copy()
    assert torch.equal(decoded_types, types)
    assert all(
        torch.equal(picked, label[label >= 0]) for picked, label in zip(decoded_indices, labels)
    )
    assert len(free_masks) == MAX_PICKS
    for hand, label in zip(hands, labels):
        expected_teacher = _teacher_forced_masks(hand, label)
        expected_free = _free_running_masks(hand, label)
        assert torch.equal(expected_teacher, expected_free)
    assert all(torch.equal(left, right) for left, right in zip(teacher_masks, free_masks))


def test_every_teacher_forced_label_token_is_legal_at_its_prefix():
    hands, labels, types, hands_left, discards_left = _random_corpus(256)
    for hand, label, action_type, hands_budget, discards_budget in zip(
        hands, labels, types, hands_left, discards_left
    ):
        type_mask = train_bc_v3.initial_type_mask(hands_budget, discards_budget)
        assert type_mask[action_type]
        length = int((label >= 0).sum())
        last_pick = -1
        n_picked = 0
        for step in range(length):
            mask = train_bc_v3.pick_step_mask(hand, last_pick, n_picked)
            assert mask[label[step]], f"forbidden label token at step {step}"
            last_pick = int(label[step])
            n_picked += 1
        if length < MAX_PICKS:
            assert train_bc_v3.pick_step_mask(hand, last_pick, n_picked)[STOP_INDEX]


def test_pointer_ppo_policy_uses_head_masks_for_identical_taken_prefixes():
    assert pointer_ppo_policy.initial_type_mask is hand_pointer_head.initial_type_mask
    assert pointer_ppo_policy.pick_step_mask is hand_pointer_head.pick_step_mask
    space = observation_space_v2()
    samples = [space.sample() for _ in range(2)]
    obs = {
        key: torch.as_tensor(np.stack([sample[key] for sample in samples]))
        for key in samples[0]
    }
    obs["hand_mask"].zero_()
    obs["hand_mask"][:, :5] = 1
    obs["global_context"][:, 13] = 0.1
    obs["global_context"][:, 14] = 0.1
    obs["joker_mask"].zero_()
    obs["consumable_mask"].zero_()
    actions = torch.tensor([[0, 0, 2, 40, 40, 40], [1, 1, 3, 4, 40, 40]])
    policy = pointer_ppo_policy.PointerPPOPolicy(
        space, spaces.MultiDiscrete([2] + [41] * 5), lambda _: 1e-3
    )
    policy_type_mask, policy_pointer_mask, policy_active = policy.teacher_forced_step_masks(
        obs, actions
    )
    cards, pooled = policy._features(obs)
    hands_left, discards_left = policy._budgets(obs)
    head_type, head_pointer, head_active = policy.pointer_head.teacher_forced_step_distributions(
        cards,
        pooled,
        obs["hand_mask"],
        hands_left,
        discards_left,
        actions[:, 0],
        torch.where(actions[:, 1:] == STOP_INDEX, -1, actions[:, 1:]),
    )
    assert torch.equal(policy_type_mask, torch.isfinite(head_type))
    assert torch.equal(policy_pointer_mask, torch.isfinite(head_pointer))
    assert torch.equal(policy_active, head_active)


def test_partner_wrapper_uses_the_shared_pointer_decoder(tmp_path, monkeypatch):
    checkpoint = tmp_path / "pointer.pt"
    model = HandPointerBCModel(observation_space_v2())
    torch.save(
        {"model_state_dict": model.state_dict(), "metadata": {"head": "pointer"}}, checkpoint
    )
    calls = 0
    original_decode = HandPointerBCModel.decode

    def recording_decode(self, obs):
        nonlocal calls
        calls += 1
        return original_decode(self, obs)

    monkeypatch.setattr(HandPointerBCModel, "decode", recording_decode)
    gs = initialize_run("b_red", 1, "MASK_PARITY_PARTNER")
    engine_step(gs, SelectBlind())
    assert gs["phase"] == GamePhase.SELECTING_HAND
    HandCheckpointPolicy(checkpoint)(gs)
    assert calls == 1
