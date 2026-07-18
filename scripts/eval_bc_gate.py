"""Execute the pre-registered Candidate-B BC gate.

This script is the executable pre-registration: thresholds are transcribed from
the plan document and MUST NOT be edited after the first gate run. The dated
2026-07-19 AMENDMENT in docs/post-regen-training-plan.md records the user's
ruling that the (b) overrun/stop bars and (d) wide-NLL ratio are diagnostics;
this is a recorded amendment, not a silent edit. Empty strata mean INCOMPLETE,
never silent pooling.

The evaluator deliberately keeps the pointer decoder's teacher-forced and
free-running paths separate.  It uses the pointer module's legality functions
for every type and pointer mask, and contains the offline beam decoder here
only as a review input; it does not alter deployment decoding.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from train_bc import DemoDataset, load_dataset, split_train_val  # noqa: E402
from train_bc_v3 import pointer_step_active_mask  # noqa: E402

from jackdaw.agents.hand_action_space import (  # noqa: E402
    NUM_COMBOS,
)
from jackdaw.agents.hand_pointer_head import (  # noqa: E402
    CARD_SLOTS,
    MAX_PICKS,
    STOP_INDEX,
    HandPointerBCModel,
    _entropy,
    _masked_log_probs,
    initial_type_mask,
    pick_step_mask,
)
from jackdaw.agents.hand_policy_v3 import FlatV3BCModel  # noqa: E402
from jackdaw.env.hand_play_gym import observation_space_v2  # noqa: E402

SET_SIZES = tuple(range(1, MAX_PICKS + 1))
ACTION_TYPES = (0, 1)
TYPE_NAMES = {0: "play", 1: "discard"}
LOW_N = 50


def _device(device_str: str) -> torch.device:
    return torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str
    )


def _load_checkpoint(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict"))
    if state is None:
        raise ValueError(f"{path}: checkpoint has no model_state_dict/state_dict")
    model.load_state_dict(state)
    model.to(device).eval()
    return payload.get("metadata", {})


def _flat_compatible(dataset: DemoDataset) -> DemoDataset:
    """Apply T3's exact loader representability sentinel (``actions >= 0``)."""

    return dataset.slice(torch.nonzero(dataset.actions >= 0).squeeze(-1))


def _label_record(dataset: DemoDataset, row: int) -> dict[str, Any]:
    indices = [int(i) for i in dataset.card_indices[row].tolist() if i >= 0]
    action_type = int(dataset.action_types[row])
    return {
        "type": TYPE_NAMES[action_type],
        "type_index": action_type,
        "card_indices": indices,
        "set_size": len(indices),
    }


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _row(n: int, values: dict[str, list[float]], **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"n": n, **extra}
    result["flags"] = ["LOW_N"] if 0 < n < LOW_N else (["EMPTY"] if n == 0 else [])
    for key, items in values.items():
        result[key] = _mean(items)
    return result


def _stratified(
    records: list[dict[str, Any]],
    metric_keys: tuple[str, ...],
    *,
    include_types: bool = True,
) -> dict[str, Any]:
    def make(items: list[dict[str, Any]]) -> dict[str, Any]:
        return _row(
            len(items),
            {key: [r[key] for r in items if r.get(key) is not None] for key in metric_keys},
        )

    result: dict[str, Any] = {
        "aggregate": make(records),
        "by_set_size": {
            str(size): make([r for r in records if r["label"]["set_size"] == size])
            for size in SET_SIZES
        },
    }
    if include_types:
        result["by_type"] = {
            TYPE_NAMES[action_type]: make(
                [r for r in records if r["label"]["type_index"] == action_type]
            )
            for action_type in ACTION_TYPES
        }
    return result


def _pointer_batch(
    model: HandPointerBCModel, batch: DemoDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = {key: value.to(device) for key, value in batch.obs.items()}
    card_latents, pooled = model.features_extractor(obs)
    hands_left, discards_left = model._budgets(obs)
    return card_latents, pooled, obs["hand_mask"], hands_left, discards_left


@torch.no_grad()
def _teacher_stop_and_type(
    model: HandPointerBCModel,
    card_latents: torch.Tensor,
    pooled: torch.Tensor,
    hand_mask: torch.Tensor,
    hands_left: torch.Tensor,
    discards_left: torch.Tensor,
    action_types: torch.Tensor,
    card_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return teacher-forced type argmax and stop argmax for each example.

    Stop argmax is -1 for size five, where no stop token is defined.
    """

    head = model.pointer_head
    lengths = (card_indices >= 0).sum(dim=-1)
    type_mask = initial_type_mask(hands_left, discards_left).to(pooled.device)
    type_log_probs = _masked_log_probs(head.type_head(pooled), type_mask)
    type_argmax = type_log_probs.argmax(dim=-1)
    state = head._state_from_type(pooled, action_types)
    last_pick = torch.full_like(lengths, -1)
    n_picked = torch.zeros_like(lengths)
    stop_argmax = torch.full_like(lengths, -1)
    for step in range(MAX_PICKS):
        active_pick = step < lengths
        after_stop = step > lengths
        mask_last = torch.where(after_stop, torch.full_like(last_pick, CARD_SLOTS - 1), last_pick)
        mask_count = torch.where(after_stop, torch.full_like(n_picked, MAX_PICKS - 1), n_picked)
        mask = pick_step_mask(hand_mask, mask_last, mask_count)
        log_probs = _masked_log_probs(head._pointer_logits(state, card_latents), mask)
        at_stop = (lengths == step) & (lengths < MAX_PICKS)
        stop_argmax = torch.where(at_stop, log_probs.argmax(dim=-1), stop_argmax)
        target = torch.where(
            active_pick,
            card_indices[:, step],
            torch.full_like(card_indices[:, step], STOP_INDEX),
        )
        safe_target = target.clamp(0, CARD_SLOTS - 1)
        picked_latent = card_latents.gather(
            1, safe_target[:, None, None].expand(-1, 1, card_latents.shape[-1])
        ).squeeze(1)
        proposed_state = head.gru(picked_latent, state)
        state = torch.where(active_pick.unsqueeze(-1), proposed_state, state)
        last_pick = torch.where(active_pick, target, last_pick)
        n_picked = n_picked + active_pick.long()
    return type_argmax, stop_argmax


@torch.no_grad()
def _greedy_decode_one(
    model: HandPointerBCModel,
    card_latents: torch.Tensor,
    pooled: torch.Tensor,
    hand_mask: torch.Tensor,
    hands_left: int,
    discards_left: int,
) -> dict[str, Any]:
    """Run the locked greedy decoder while retaining free-running entropy."""

    head = model.pointer_head
    type_mask = initial_type_mask(hands_left, discards_left).to(pooled.device)
    type_log_probs = _masked_log_probs(head.type_head(pooled), type_mask)
    action_type = int(type_log_probs.argmax(dim=-1).item())
    state = head._state_from_type(pooled, torch.tensor([action_type], device=pooled.device))
    last_pick = -1
    n_picked = 0
    picked: list[int] = []
    entropies = [float(_entropy(type_log_probs, type_mask).item())]
    termination = "unknown"

    for _step in range(MAX_PICKS):
        mask = pick_step_mask(hand_mask, last_pick, n_picked)
        log_probs = _masked_log_probs(head._pointer_logits(state, card_latents), mask)
        entropies.append(float(_entropy(log_probs, mask).item()))
        token = int(log_probs.argmax(dim=-1).item())
        if token == STOP_INDEX:
            termination = (
                "pick_exhaustion" if not bool(mask[..., :CARD_SLOTS].any()) else "explicit_stop"
            )
            break
        picked.append(token)
        last_pick = token
        n_picked += 1
        picked_latent = card_latents[:, token, :]
        state = head.gru(picked_latent, state)
        if n_picked == MAX_PICKS:
            termination = "cap"
            break
    if termination == "unknown":
        termination = "cap"
    return {
        "type_index": action_type,
        "type": TYPE_NAMES[action_type],
        "card_indices": picked,
        "entropies": entropies,
        "termination": termination,
    }


def _candidate_tokens(log_probs: torch.Tensor, mask: torch.Tensor) -> list[int]:
    legal = [int(token) for token in torch.nonzero(mask.squeeze(0), as_tuple=False).flatten()]
    return sorted(legal, key=lambda token: (-float(log_probs[0, token]), token))


@torch.no_grad()
def _beam_decode_one(
    model: HandPointerBCModel,
    card_latents: torch.Tensor,
    pooled: torch.Tensor,
    hand_mask: torch.Tensor,
    hands_left: int,
    discards_left: int,
    beam_width: int,
) -> dict[str, Any]:
    """Beam decode for the offline review check; width one is greedy."""

    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    head = model.pointer_head
    type_mask = initial_type_mask(hands_left, discards_left).to(pooled.device)
    type_log_probs = _masked_log_probs(head.type_head(pooled), type_mask)
    beams: list[dict[str, Any]] = []
    for token in _candidate_tokens(type_log_probs, type_mask):
        action_type = torch.tensor([token], device=pooled.device)
        beams.append(
            {
                "score": float(type_log_probs[0, token]),
                "type_index": token,
                "card_indices": [],
                "state": head._state_from_type(pooled, action_type),
                "last_pick": -1,
                "n_picked": 0,
                "active": True,
                "termination": "unknown",
            }
        )
    beams = beams[:beam_width]

    for _step in range(MAX_PICKS):
        expanded: list[dict[str, Any]] = []
        for beam in beams:
            if not beam["active"]:
                expanded.append(beam)
                continue
            mask = pick_step_mask(hand_mask, beam["last_pick"], beam["n_picked"])
            log_probs = _masked_log_probs(head._pointer_logits(beam["state"], card_latents), mask)
            for token in _candidate_tokens(log_probs, mask):
                next_beam = dict(beam)
                next_beam["score"] = beam["score"] + float(log_probs[0, token])
                if token == STOP_INDEX:
                    next_beam["active"] = False
                    next_beam["termination"] = (
                        "pick_exhaustion"
                        if not bool(mask[..., :CARD_SLOTS].any())
                        else "explicit_stop"
                    )
                else:
                    next_beam["card_indices"] = [*beam["card_indices"], token]
                    next_beam["last_pick"] = token
                    next_beam["n_picked"] = beam["n_picked"] + 1
                    picked_latent = card_latents[:, token, :]
                    next_beam["state"] = head.gru(picked_latent, beam["state"])
                    if next_beam["n_picked"] == MAX_PICKS:
                        next_beam["active"] = False
                        next_beam["termination"] = "cap"
                expanded.append(next_beam)
        beams = sorted(
            expanded,
            key=lambda beam: (
                -beam["score"],
                beam["type_index"],
                tuple(beam["card_indices"]),
            ),
        )[:beam_width]
        if not any(beam["active"] for beam in beams):
            break

    best = beams[0]
    if best["termination"] == "unknown":
        best["termination"] = "cap"
    return {
        "type_index": best["type_index"],
        "type": TYPE_NAMES[best["type_index"]],
        "card_indices": best["card_indices"],
        "termination": best["termination"],
        "score": best["score"],
    }


def _decoded_action(decoded: dict[str, Any]) -> dict[str, Any]:
    return {"type": decoded["type"], "card_indices": decoded["card_indices"]}


def _validate_decoded(
    decoded: dict[str, Any], hand_mask: torch.Tensor, hands_left: int, discards_left: int
) -> tuple[bool, str | None]:
    indices = decoded["card_indices"]
    action_type = decoded["type_index"]
    if (
        action_type not in ACTION_TYPES
        or (action_type == 0 and hands_left < 1)
        or (action_type == 1 and discards_left < 1)
    ):
        return False, "illegal action type or budget"
    if not 1 <= len(indices) <= MAX_PICKS:
        return False, "set size outside 1-5"
    if indices != sorted(indices) or len(set(indices)) != len(indices):
        return False, "indices are not strictly ascending"
    live = hand_mask.detach().cpu().bool()
    if any(index < 0 or index >= live.numel() or not bool(live[index]) for index in indices):
        return False, "index is not live"
    return True, None


def _score_decoded(
    model: HandPointerBCModel,
    card_latents: torch.Tensor,
    pooled: torch.Tensor,
    hand_mask: torch.Tensor,
    hands_left: torch.Tensor,
    discards_left: torch.Tensor,
    decoded: list[dict[str, Any]],
    valid: list[bool],
) -> torch.Tensor:
    scores = torch.full((len(decoded),), float("nan"), device=pooled.device)
    valid_rows = [row for row, is_valid in enumerate(valid) if is_valid]
    if not valid_rows:
        return scores
    action_types = torch.tensor(
        [decoded[row]["type_index"] for row in valid_rows], device=pooled.device
    )
    card_indices = torch.full(
        (len(valid_rows), MAX_PICKS), -1, dtype=torch.long, device=pooled.device
    )
    for output_row, source_row in enumerate(valid_rows):
        action = decoded[source_row]
        card_indices[output_row, : len(action["card_indices"])] = torch.tensor(
            action["card_indices"], dtype=torch.long, device=pooled.device
        )
    _, sequence_log_prob, _ = model.pointer_head.teacher_forced_log_probs(
        card_latents[valid_rows],
        pooled[valid_rows],
        hand_mask[valid_rows],
        hands_left[valid_rows],
        discards_left[valid_rows],
        action_types,
        card_indices,
    )
    scores[valid_rows] = -sequence_log_prob
    return scores


@torch.no_grad()
def _make_records(
    pointer: HandPointerBCModel,
    flat: FlatV3BCModel,
    dataset: DemoDataset,
    device: torch.device,
    beam_width: int,
    batch_size: int = 256,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pointer.eval()
    flat.eval()
    for start in range(0, len(dataset), batch_size):
        batch = dataset.slice(torch.arange(start, min(start + batch_size, len(dataset))))
        obs = {key: value.to(device) for key, value in batch.obs.items()}
        card_latents, pooled = pointer.features_extractor(obs)
        hands_left, discards_left = pointer._budgets(obs)
        hand_mask = obs["hand_mask"].bool()
        per_step, sequence_log_prob, teacher_entropies = (
            pointer.pointer_head.teacher_forced_log_probs(
                card_latents,
                pooled,
                hand_mask,
                hands_left,
                discards_left,
                batch.action_types.to(device),
                batch.card_indices.to(device),
            )
        )
        type_argmax, stop_argmax = _teacher_stop_and_type(
            pointer,
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            batch.action_types.to(device),
            batch.card_indices.to(device),
        )
        flat_logits, flat_values = flat(obs)
        flat_compatible = batch.actions >= 0
        flat_nll = torch.full((len(batch),), float("nan"), device=device)
        flat_exact = torch.full((len(batch),), float("nan"), device=device)
        flat_type_accuracy = torch.full((len(batch),), float("nan"), device=device)
        flat_value_mse = (flat_values - batch.p_clear.to(device)).square()
        flat_pred = torch.full((len(batch),), -1, dtype=torch.long, device=device)
        if bool(flat_compatible.any()):
            comp = flat_compatible.to(device)
            legal = batch.legal_masks.to(device)
            masked = flat_logits.masked_fill(~legal, float("-inf"))
            flat_log_probs = torch.log_softmax(masked, dim=-1)
            flat_nll[comp] = (
                -flat_log_probs[comp]
                .gather(1, batch.actions.to(device)[comp].unsqueeze(-1))
                .squeeze(-1)
            )
            flat_pred[comp] = masked[comp].argmax(dim=-1)
            flat_exact[comp] = (flat_pred[comp] == batch.actions.to(device)[comp]).float()
            type_probs = flat_log_probs[comp].exp()
            play_mass = type_probs[:, :NUM_COMBOS].sum(dim=-1)
            discard_mass = type_probs[:, NUM_COMBOS:].sum(dim=-1)
            type_pred = (discard_mass > play_mass).long()
            flat_type_accuracy[comp] = (type_pred == batch.action_types.to(device)[comp]).float()

        free_decoded: list[dict[str, Any]] = []
        beam_decoded: list[dict[str, Any]] = []
        for row in range(len(batch)):
            free_decoded.append(
                _greedy_decode_one(
                    pointer,
                    card_latents[row : row + 1],
                    pooled[row : row + 1],
                    hand_mask[row : row + 1],
                    int(hands_left[row]),
                    int(discards_left[row]),
                )
            )
            beam_decoded.append(
                _beam_decode_one(
                    pointer,
                    card_latents[row : row + 1],
                    pooled[row : row + 1],
                    hand_mask[row : row + 1],
                    int(hands_left[row]),
                    int(discards_left[row]),
                    beam_width,
                )
            )
        free_validity: list[bool] = []
        beam_validity: list[bool] = []
        free_reasons: list[str | None] = []
        beam_reasons: list[str | None] = []
        for row in range(len(batch)):
            free_valid, free_reason = _validate_decoded(
                free_decoded[row],
                hand_mask[row],
                int(hands_left[row]),
                int(discards_left[row]),
            )
            beam_valid, beam_reason = _validate_decoded(
                beam_decoded[row],
                hand_mask[row],
                int(hands_left[row]),
                int(discards_left[row]),
            )
            free_validity.append(free_valid)
            beam_validity.append(beam_valid)
            free_reasons.append(free_reason)
            beam_reasons.append(beam_reason)
        free_nll = _score_decoded(
            pointer,
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            free_decoded,
            free_validity,
        )
        beam_nll = _score_decoded(
            pointer,
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            beam_decoded,
            beam_validity,
        )
        active = pointer_step_active_mask(batch.card_indices.to(device))
        for row in range(len(batch)):
            label = _label_record(batch, row)
            size = label["set_size"]
            wide = not bool(flat_compatible[row])
            free = free_decoded[row]
            beam = beam_decoded[row]
            valid = free_validity[row]
            reason = free_reasons[row]
            beam_valid = beam_validity[row]
            beam_reason = beam_reasons[row]
            record: dict[str, Any] = {
                "seed": batch.seeds[row],
                "label": label,
                "wide": wide,
                "pointer_joint_nll": float(-sequence_log_prob[row]),
                "pointer_exact": float(
                    valid
                    and free["type_index"] == label["type_index"]
                    and free["card_indices"] == label["card_indices"]
                ),
                "pointer_type_accuracy": float(
                    type_argmax[row] == batch.action_types[row].to(device)
                ),
                "stop_accuracy": (
                    None if size == MAX_PICKS else float(stop_argmax[row] == STOP_INDEX)
                ),
                "pointer_value_mse": float(
                    (
                        pointer.value_net(pooled[row : row + 1]).squeeze(-1)[0]
                        - batch.p_clear[row].to(device)
                    ).square()
                ),
                "pick_nll": [
                    float(-per_step[row, step])
                    for step in range(1, MAX_PICKS + 1)
                    if bool(active[row, step])
                ],
                "teacher_entropies": [
                    float(teacher_entropies[row, step])
                    for step in range(per_step.shape[1])
                    if bool(active[row, step])
                ],
                "free": free,
                "free_nll": None if torch.isnan(free_nll[row]) else float(free_nll[row]),
                "free_valid": valid,
                "free_invalid_reason": reason,
                "beam": beam,
                "beam_nll": None if torch.isnan(beam_nll[row]) else float(beam_nll[row]),
                "beam_valid": beam_valid,
                "beam_invalid_reason": beam_reason,
                "beam_action_disagreement": float(
                    free["type_index"] != beam["type_index"]
                    or free["card_indices"] != beam["card_indices"]
                ),
                "beam_set_disagreement": float(free["card_indices"] != beam["card_indices"]),
                "flat_nll": None if math.isnan(float(flat_nll[row])) else float(flat_nll[row]),
                "flat_exact": None
                if math.isnan(float(flat_exact[row]))
                else float(flat_exact[row]),
                "flat_type_accuracy": (
                    None
                    if math.isnan(float(flat_type_accuracy[row]))
                    else float(flat_type_accuracy[row])
                ),
                "flat_value_mse": (
                    None if math.isnan(float(flat_value_mse[row])) else float(flat_value_mse[row])
                ),
            }
            records.append(record)
    return records


def _distribution_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    def make(items: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(len(row["free"]["card_indices"]) for row in items)
        n = len(items)
        return {
            "n": n,
            "flags": ["LOW_N"] if 0 < n < LOW_N else (["EMPTY"] if n == 0 else []),
            "predicted_counts": {str(size): counts.get(size, 0) for size in SET_SIZES},
            "predicted_rates": {
                str(size): (counts.get(size, 0) / n if n else None) for size in SET_SIZES
            },
        }

    return {
        "semantics": "free_running_greedy_decode, stratified by true set size",
        "aggregate": make(records),
        "by_true_set_size": {
            str(size): make([r for r in records if r["label"]["set_size"] == size])
            for size in SET_SIZES
        },
        "by_type": {
            TYPE_NAMES[action_type]: make(
                [r for r in records if r["label"]["type_index"] == action_type]
            )
            for action_type in ACTION_TYPES
        },
    }


def _termination_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    def make(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        invalid = [r for r in items if not r["free_valid"]]
        overrun = [
            r
            for r in items
            if r["free"]["termination"] in {"cap", "pick_exhaustion"}
            and len(r["free"]["card_indices"]) > r["label"]["set_size"]
        ]
        benign = [
            r
            for r in items
            if r["free"]["termination"] in {"cap", "pick_exhaustion"}
            and len(r["free"]["card_indices"]) == r["label"]["set_size"]
        ]
        return {
            "n": n,
            "flags": ["LOW_N"] if 0 < n < LOW_N else (["EMPTY"] if n == 0 else []),
            "invalid_count": len(invalid),
            "invalid_rate": len(invalid) / n if n else None,
            "overrun_termination_count": len(overrun),
            "overrun_termination_rate": len(overrun) / n if n else None,
            "benign_cap_termination_count": len(benign),
            "benign_cap_termination_rate": len(benign) / n if n else None,
            "explicit_stop_count": sum(r["free"]["termination"] == "explicit_stop" for r in items),
            "invalid_seeds": [r["seed"] for r in invalid],
        }

    overrun_cases = [
        {
            "seed": row["seed"],
            "true_label": row["label"],
            "decoded_action": _decoded_action(row["free"]),
            "termination": row["free"]["termination"],
        }
        for row in records
        if row["free"]["termination"] in {"cap", "pick_exhaustion"}
        and len(row["free"]["card_indices"]) > row["label"]["set_size"]
    ]
    decoded_actions = [
        {
            "seed": row["seed"],
            "true_label": row["label"],
            "decoded_action": _decoded_action(row["free"]),
            "valid": row["free_valid"],
            "termination": row["free"]["termination"],
        }
        for row in records
    ]
    return {
        "semantics": "free_running_greedy_decode",
        "aggregate": make(records),
        "by_true_set_size": {
            str(size): make([r for r in records if r["label"]["set_size"] == size])
            for size in SET_SIZES
        },
        "overrun_cases": overrun_cases,
        "decoded_actions": decoded_actions,
        "by_type": {
            TYPE_NAMES[action_type]: make(
                [r for r in records if r["label"]["type_index"] == action_type]
            )
            for action_type in ACTION_TYPES
        },
    }


def _per_pick_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for size in SET_SIZES:
        for step in range(1, MAX_PICKS + 1):
            items = [
                r
                for r in records
                if r["label"]["set_size"] == size and step <= r["label"]["set_size"]
            ]
            values = [r["pick_nll"][step - 1] for r in items]
            rows[f"{size}:{step}"] = _row(
                len(items), {"mean_nll": values}, true_set_size=size, step_index=step
            )
    by_type = {}
    for action_type in ACTION_TYPES:
        type_rows = {}
        type_records = [r for r in records if r["label"]["type_index"] == action_type]
        for step in range(1, MAX_PICKS + 1):
            values = [
                r["pick_nll"][step - 1] for r in type_records if step <= r["label"]["set_size"]
            ]
            type_rows[str(step)] = _row(len(values), {"mean_nll": values}, step_index=step)
        by_type[TYPE_NAMES[action_type]] = type_rows
    return {
        "semantics": "teacher_forced labeled pick-token NLL",
        "rows": rows,
        "step_indices": list(range(1, MAX_PICKS + 1)),
        "by_type": by_type,
    }


def _entropy_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    for size in SET_SIZES:
        items = [r for r in records if r["label"]["set_size"] == size]
        for step in range(0, MAX_PICKS + 1):
            values = [
                r["free"]["entropies"][step] for r in items if len(r["free"]["entropies"]) > step
            ]
            rows[f"{size}:{step}"] = _row(
                len(values), {"mean_entropy": values}, true_set_size=size, step_index=step
            )
    by_type = {}
    for action_type in ACTION_TYPES:
        type_rows = {}
        type_records = [r for r in records if r["label"]["type_index"] == action_type]
        for step in range(0, MAX_PICKS + 1):
            values = [
                r["free"]["entropies"][step]
                for r in type_records
                if len(r["free"]["entropies"]) > step
            ]
            type_rows[str(step)] = _row(len(values), {"mean_entropy": values}, step_index=step)
        by_type[TYPE_NAMES[action_type]] = type_rows
    return {
        "semantics": "free_running greedy masked-token entropy",
        "rows": rows,
        "step_indices": list(range(0, MAX_PICKS + 1)),
        "by_type": by_type,
    }


def _beam_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    def make(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        return {
            "n": n,
            "flags": ["LOW_N"] if 0 < n < LOW_N else (["EMPTY"] if n == 0 else []),
            "greedy_sequence_validity": sum(r["free_valid"] for r in items) / n if n else None,
            "beam_sequence_validity": sum(r["beam_valid"] for r in items) / n if n else None,
            "greedy_sequence_nll": _mean([r["free_nll"] for r in items]),
            "beam_sequence_nll": _mean([r["beam_nll"] for r in items]),
            "greedy_action_disagreement_rate": _mean(
                [r["beam_action_disagreement"] for r in items]
            ),
            "greedy_set_disagreement_rate": _mean([r["beam_set_disagreement"] for r in items]),
            "greedy_predicted_set_size_counts": {
                str(size): sum(len(r["free"]["card_indices"]) == size for r in items)
                for size in SET_SIZES
            },
            "beam_predicted_set_size_counts": {
                str(size): sum(len(r["beam"]["card_indices"]) == size for r in items)
                for size in SET_SIZES
            },
            "cases": [
                {
                    "seed": r["seed"],
                    "true_set_size": r["label"]["set_size"],
                    "greedy_action": _decoded_action(r["free"]),
                    "beam_action": _decoded_action(r["beam"]),
                    "greedy_valid": r["free_valid"],
                    "beam_valid": r["beam_valid"],
                    "greedy_sequence_nll": r["free_nll"],
                    "beam_sequence_nll": r["beam_nll"],
                }
                for r in items
            ],
        }

    return {
        "semantics": "free_running greedy versus in-script masked beam decode",
        "aggregate": make(records),
        "by_true_set_size": {
            str(size): make([r for r in records if r["label"]["set_size"] == size])
            for size in SET_SIZES
        },
        "by_type": {
            TYPE_NAMES[action_type]: make(
                [r for r in records if r["label"]["type_index"] == action_type]
            )
            for action_type in ACTION_TYPES
        },
    }


def _canary(metadata: dict[str, Any]) -> float | None:
    candidates = (
        "memorization_canary_mean_non_padding_token_ce",
        "canary_mean_non_padding_token_ce",
        "mean_non_padding_token_ce",
    )
    for key in candidates:
        if metadata.get(key) is not None:
            return float(metadata[key])
    nested = metadata.get("memorization_canary") or metadata.get("canary")
    if isinstance(nested, dict):
        for key in candidates + ("value", "mean_ce"):
            if nested.get(key) is not None:
                return float(nested[key])
    return None


def _verdict(
    tables: dict[str, Any], pointer_metadata: dict[str, Any], flat_metadata: dict[str, Any]
) -> dict[str, Any]:
    head = tables["head_to_head"]
    wide = tables["wide"]
    type_table = tables["type_token_accuracy"]
    stop = tables["stop_token_accuracy"]
    pclear = tables["p_clear_head_mse"]
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str, description: str, passed: bool | None, measured: Any, threshold: str
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "description": description,
                "status": "PASS"
                if passed is True
                else ("FAIL" if passed is False else "INCOMPLETE"),
                "measured": measured,
                "threshold": threshold,
            }
        )

    def add_diagnostic(
        check_id: str, description: str, measured: Any, threshold: str
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "description": description,
                "status": "DIAGNOSTIC",
                "measured": measured,
                "threshold": threshold,
            }
        )

    def required(values: list[Any]) -> bool:
        return bool(values) and all(value is not None for value in values)

    hagg = head["aggregate"]
    hsets = [head["by_set_size"][str(size)] for size in SET_SIZES]
    nll_values = [hagg.get("pointer_joint_nll"), hagg.get("flat_nll")] + [
        value for row in hsets for value in (row.get("pointer_joint_nll"), row.get("flat_nll"))
    ]
    exact_values = [hagg.get("pointer_exact"), hagg.get("flat_exact")] + [
        value for row in hsets for value in (row.get("pointer_exact"), row.get("flat_exact"))
    ]
    nll_pass = None
    exact_pass = None
    if required(nll_values):
        nll_pass = hagg["pointer_joint_nll"] <= 1.05 * hagg["flat_nll"] and all(
            row["pointer_joint_nll"] <= 1.10 * row["flat_nll"] for row in hsets
        )
    if required(exact_values):
        exact_pass = hagg["pointer_exact"] >= hagg["flat_exact"] - 0.01 and all(
            row["pointer_exact"] >= row["flat_exact"] - 0.02 for row in hsets
        )
    add(
        "(a) shared_nll",
        "B joint NLL versus flat masked NLL on shared flat-compatible support",
        nll_pass,
        {"aggregate": {"B": hagg.get("pointer_joint_nll"), "flat": hagg.get("flat_nll")}},
        "B <= 1.05x flat aggregate and <= 1.10x flat per set-size stratum",
    )
    add(
        "(a) exact_set_match",
        "B versus flat exact-set match on shared support",
        exact_pass,
        {"aggregate": {"B": hagg.get("pointer_exact"), "flat": hagg.get("flat_exact")}},
        "B >= flat - 0.01 aggregate and >= flat - 0.02 per set-size stratum",
    )

    invalid = tables["free_running_termination_audit"]["aggregate"].get("invalid_rate")
    overrun = tables["free_running_termination_audit"]["aggregate"].get("overrun_termination_rate")
    stop_values = [stop["by_set_size"][str(size)].get("accuracy") for size in range(1, MAX_PICKS)]
    b_type = type_table["aggregate"].get("B_accuracy")
    flat_type = type_table["aggregate"].get("flat_accuracy")
    b_parts = [invalid, b_type, flat_type]
    b_pass = None
    if invalid not in (None, 0):
        b_pass = False
    elif required(b_parts):
        b_pass = invalid == 0 and b_type >= flat_type - 0.01
    add(
        "(b) free_running_and_tokens",
        "B invalid-rate and type-token hard thresholds",
        b_pass,
        {
            "invalid_rate": invalid,
            "B_type": b_type,
            "flat_type": flat_type,
        },
        "invalid = 0 HARD, B type >= flat - 0.01 HARD",
    )
    add_diagnostic(
        "(b) overrun_rate",
        "B overrun termination rate",
        {"overrun_rate": overrun},
        "reported diagnostic; excluded from the overall verdict",
    )
    add_diagnostic(
        "(b) stop_token_accuracy",
        "B teacher-forced stop-token accuracy by true set size",
        {"stop_by_size": stop_values},
        "reported diagnostic; 0.85 reference for sizes 1-4, excluded from the overall verdict",
    )

    b_mse = pclear["aggregate"].get("B_value_mse")
    flat_mse = pclear["aggregate"].get("flat_value_mse")
    add(
        "(c) p_clear_mse",
        "B p_clear-head MSE versus flat control",
        None if not required([b_mse, flat_mse]) else b_mse <= 1.10 * flat_mse,
        {"B": b_mse, "flat": flat_mse},
        "B <= 1.10x flat",
    )

    wide_pick = wide["aggregate"].get("wide_per_pick_nll")
    flat_pick = wide["aggregate"].get("B_flat_compatible_per_pick_nll")
    wide_ratio = None
    if required([wide_pick, flat_pick]) and flat_pick != 0:
        wide_ratio = wide_pick / flat_pick
    add_diagnostic(
        "(d) wide_per_pick_nll",
        "B wide-stratum per-pick NLL versus B flat-compatible per-pick NLL",
        {"wide": wide_pick, "ratio": wide_ratio, "B_flat_compatible": flat_pick},
        (
            "reported diagnostic; wide <= 1.5x B flat-compatible reference, "
            "excluded from the overall verdict"
        ),
    )

    canary = _canary(pointer_metadata)
    if canary is None:
        canary = _canary(flat_metadata)
    checks.append(
        {
            "id": "(e) memorization_canary",
            "description": "Mean non-padding per-token CE memorization canary",
            "status": "NOT_RUN" if canary is None else ("PASS" if canary < 0.05 else "FAIL"),
            "measured": canary,
            "threshold": "< 0.05",
        }
    )
    incomplete = tables.get("incomplete_requirements", [])
    structural_fail = invalid not in (None, 0)
    any_fail = any(check["status"] == "FAIL" for check in checks)
    any_incomplete = bool(incomplete) or any(
        check["status"] in {"INCOMPLETE", "NOT_RUN"} for check in checks
    )
    overall = (
        "FAIL"
        if structural_fail or (any_fail and not any_incomplete)
        else ("INCOMPLETE" if any_incomplete else "PASS")
    )
    return {
        "checks": checks,
        "overall": overall,
        "incomplete_requirements": incomplete,
        "winrate": "REFERENCE_ONLY_NOT_COMPUTED",
    }


def _build_report(
    records: list[dict[str, Any]],
    pointer_metadata: dict[str, Any],
    flat_metadata: dict[str, Any],
    beam_width: int,
) -> dict[str, Any]:
    shared = [r for r in records if not r["wide"]]
    wide_records = [r for r in records if r["wide"]]
    head = _stratified(
        shared,
        ("pointer_joint_nll", "flat_nll", "pointer_exact", "flat_exact"),
    )
    head["semantics"] = {
        "pointer_joint_nll": "teacher_forced labeled sequence NLL",
        "flat_nll": "teacher_forced masked labeled-action NLL",
        "pointer_exact": "free_running greedy decoded type and card sequence",
        "flat_exact": "teacher_forced masked argmax action index",
    }
    wide = {
        **_stratified(wide_records, ("pointer_joint_nll", "pointer_exact")),
        "aggregate": {
            **_row(
                len(wide_records),
                {
                    "sequence_nll": [r["pointer_joint_nll"] for r in wide_records],
                    "exact_set_match": [r["pointer_exact"] for r in wide_records],
                },
            ),
            "wide_per_pick_nll": _mean([v for r in wide_records for v in r["pick_nll"]]),
            "B_flat_compatible_per_pick_nll": _mean([v for r in shared for v in r["pick_nll"]]),
        },
    }
    wide["semantics"] = {
        "sequence_nll": "teacher_forced labeled pointer sequence NLL",
        "per_pick_nll": "teacher_forced labeled pick-token NLL",
        "exact_set_match": "free_running greedy decoded type and card sequence",
    }
    wide["aggregate"]["per_pick_nll"] = wide["aggregate"]["wide_per_pick_nll"]
    # Keep the names in the wide table explicit: this arm has no flat score.
    for key in ("by_set_size", "by_type"):
        for stratum, row in wide[key].items():
            items = [
                record
                for record in wide_records
                if (
                    record["label"]["set_size"] == int(stratum)
                    if key == "by_set_size"
                    else TYPE_NAMES[record["label"]["type_index"]] == stratum
                )
            ]
            row["sequence_nll"] = row.pop("pointer_joint_nll")
            row["exact_set_match"] = row.pop("pointer_exact")
            row["per_pick_nll"] = _mean([value for item in items for value in item["pick_nll"]])

    type_table = _stratified(shared, ("pointer_type_accuracy", "flat_type_accuracy"))
    type_table["semantics"] = "teacher_forced type-token/marginal argmax accuracy"
    for container in (
        type_table["aggregate"],
        *type_table["by_set_size"].values(),
        *type_table["by_type"].values(),
    ):
        container["B_accuracy"] = container.pop("pointer_type_accuracy")
        container["flat_accuracy"] = container.pop("flat_type_accuracy")

    stop_table = {
        "semantics": "teacher_forced stop-token argmax among legal tokens; size 5 is N/A",
        "aggregate": _row(
            len(records),
            {"accuracy": [r["stop_accuracy"] for r in records if r["stop_accuracy"] is not None]},
        ),
        "by_set_size": {},
        "by_type": {},
    }
    for size in SET_SIZES:
        items = [r for r in records if r["label"]["set_size"] == size]
        stop_table["by_set_size"][str(size)] = _row(
            len(items),
            {"accuracy": [r["stop_accuracy"] for r in items if r["stop_accuracy"] is not None]},
            status="N/A" if size == MAX_PICKS else None,
        )
    for action_type in ACTION_TYPES:
        items = [r for r in records if r["label"]["type_index"] == action_type]
        stop_table["by_type"][TYPE_NAMES[action_type]] = _row(
            len(items),
            {"accuracy": [r["stop_accuracy"] for r in items if r["stop_accuracy"] is not None]},
        )

    pclear = _stratified(records, ("pointer_value_mse", "flat_value_mse"))
    pclear["semantics"] = "forward p_clear-head squared error"
    for container in (
        pclear["aggregate"],
        *pclear["by_set_size"].values(),
        *pclear["by_type"].values(),
    ):
        container["B_value_mse"] = container.pop("pointer_value_mse")
        container["flat_value_mse"] = container.pop("flat_value_mse")

    wide_required = bool(wide_records)
    incomplete: list[str] = []
    if not shared:
        incomplete.append("shared flat-compatible support is empty")
    for size in SET_SIZES:
        if head["by_set_size"][str(size)]["n"] == 0:
            incomplete.append(f"shared set-size stratum {size} is empty")
        if not [r for r in records if r["label"]["set_size"] == size]:
            incomplete.append(f"all-example set-size stratum {size} is empty")
        if stop_table["by_set_size"][str(size)]["n"] == 0:
            incomplete.append(f"stop set-size stratum {size} is empty")
    if not wide_required:
        incomplete.append("wide stratum is empty")
    for size in SET_SIZES:
        if wide["by_set_size"][str(size)]["n"] == 0:
            incomplete.append(f"wide set-size stratum {size} is empty")
    for action_type in ACTION_TYPES:
        if not [r for r in shared if r["label"]["type_index"] == action_type]:
            incomplete.append(f"shared {TYPE_NAMES[action_type]} stratum is empty")
    if pclear["aggregate"]["flat_value_mse"] is None:
        incomplete.append("flat p_clear MSE is missing")
    if _canary(pointer_metadata) is None and _canary(flat_metadata) is None:
        incomplete.append("memorization canary metadata is absent")
    tables: dict[str, Any] = {
        "head_to_head": head,
        "wide": wide,
        "type_token_accuracy": type_table,
        "stop_token_accuracy": stop_table,
        "per_pick_position_nll": _per_pick_table(records),
        "predicted_vs_true_set_size": _distribution_table(records),
        "free_running_termination_audit": _termination_table(records),
        "entropy_by_decode_step": _entropy_table(records),
        "p_clear_head_mse": pclear,
        "greedy_vs_beam_decode": _beam_table(records),
        "incomplete_requirements": sorted(set(incomplete)),
    }
    report = {
        "schema_version": 1,
        "pre_registration": {
            "source": "docs/post-regen-training-plan.md section 1",
            "thresholds_locked": "2026-07-18",
            "beam_width": beam_width,
            "flat_compatible_definition": (
                "loader actions >= 0, equivalent to every label index <= 7"
            ),
            "discrepancies": [
                (
                    "T3 DemoDataset has no literal flat_compatible field; this uses its "
                    "exact actions >= 0 representability sentinel."
                ),
                (
                    "The plan calls the offline decode input representative; the T4 CLI "
                    "has no subset selector, so it evaluates the reconstructed val split."
                ),
            ],
        },
        "tables": tables,
        "readouts": {
            "flat_dropped_label_count": len(wide_records),
            "flat_validation_dropped_label_fraction": len(wide_records) / max(len(records), 1),
            "flat_dropped_label_fraction": len(wide_records) / max(len(records), 1),
            "flat_training_dropped_label_fraction": flat_metadata.get("dropped_wide_fraction"),
        },
    }
    # Keep the required table names addressable both under ``tables`` and as
    # top-level JSON keys; consumers can choose a flat or namespaced schema.
    report.update(
        {name: table for name, table in tables.items() if name != "incomplete_requirements"}
    )
    report["verdict"] = _verdict(tables, pointer_metadata, flat_metadata)
    return report


def _markdown_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"### {title}\n\n_No rows._\n"
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lines = [
        f"### {title}",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join("---" for _ in keys) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return "\n".join(lines) + "\n"


def _markdown_report(report: dict[str, Any]) -> str:
    tables = report["tables"]
    chunks = [
        "# BC gate report",
        "",
        f"Overall verdict: **{report['verdict']['overall']}**",
        "",
        (
            "Flat dropped-label rider: "
            f"n={report['readouts']['flat_dropped_label_count']} "
            f"validation_fraction={report['readouts']['flat_validation_dropped_label_fraction']} "
            f"training_fraction={report['readouts']['flat_training_dropped_label_fraction']}"
        ),
        "",
        "Winrate: reference only; not computed.",
        "",
    ]
    for check in report["verdict"]["checks"]:
        chunks.append(
            f"- {check['id']}: **{check['status']}** — {check['measured']} ({check['threshold']})"
        )
    chunks.extend(["", "## Tables", ""])
    for name, table in tables.items():
        if name == "incomplete_requirements":
            continue
        if "semantics" in table:
            chunks.append(f"Semantics ({name}): {table['semantics']}")
        stratum_key = "by_set_size" if "by_set_size" in table else "by_true_set_size"
        if "aggregate" in table and stratum_key in table:
            rows = [{"stratum": "aggregate", **table["aggregate"]}]
            rows.extend(
                {"stratum": f"true_set_size={key}", **value}
                for key, value in table[stratum_key].items()
            )
            chunks.append(_markdown_table(name, rows))
        elif "rows" in table:
            chunks.append(_markdown_table(name, list(table["rows"].values())))
        else:
            chunks.append(_markdown_table(name, [{"value": table}]))
        if "by_type" in table:
            for action_type, type_table in table["by_type"].items():
                if "rows" in type_table:
                    type_rows = list(type_table["rows"].values())
                elif "n" in type_table:
                    type_rows = [{"stratum": action_type, **type_table}]
                else:
                    type_rows = [{"value": type_table}]
                chunks.append(_markdown_table(f"{name} ({action_type})", type_rows))
    return "\n".join(chunks)


def evaluate_gate(
    pointer_checkpoint: Path,
    flat_checkpoint: Path,
    data_dirs: list[Path],
    output: Path,
    *,
    val_fraction: float = 0.10,
    beam_width: int = 4,
    device_str: str = "auto",
) -> dict[str, Any]:
    """Run the gate and write deterministic JSON and Markdown artifacts."""

    device = _device(device_str)
    dataset = load_dataset(data_dirs, {})
    _, val_set = split_train_val(dataset, val_fraction)
    if not len(val_set):
        raise ValueError("the reconstructed validation split is empty")
    pointer = HandPointerBCModel(observation_space_v2())
    flat = FlatV3BCModel(observation_space_v2())
    pointer_metadata = _load_checkpoint(pointer_checkpoint, pointer, device)
    flat_metadata = _load_checkpoint(flat_checkpoint, flat, device)
    records = _make_records(pointer, flat, val_set, device, beam_width)
    report = _build_report(records, pointer_metadata, flat_metadata, beam_width)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True, allow_nan=False)
        file.write("\n")
    markdown = _markdown_report(report)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer-checkpoint", type=Path, required=True)
    parser.add_argument("--flat-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_gate(
        args.pointer_checkpoint,
        args.flat_checkpoint,
        args.data_dirs,
        args.output,
        val_fraction=args.val_fraction,
        beam_width=args.beam_width,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
