"""Sequence-CE BC for the v3 pointer arm and its flat-head control.

This is one shared trainer: ``--head`` is the only arm difference, which is
the gate's comparability requirement. Both arms use the imported legacy loader,
CRC32 split, AdamW loop, value regression, best-epoch checkpoint, and metrics
conventions. The flat arm is deliberately trained only on the flat-compatible
rows and loudly records the dropped-wide rider. Pointer label smoothing is
applied independently at every active step over that step's legal token set;
masked tokens never receive smoothing mass. Early stopping uses UNSMOOTHED
validation NLL (sequence NLL for the pointer arm and flat CE for the control).

Usage::

    uv run python scripts/train_bc_v3.py --head pointer \
        --data-dirs data/hand_agent_demos/stage2_curated \
        --output runs/bc_v3/pointer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from train_bc import (  # noqa: E402
    DemoDataset,
    evaluate,
    load_dataset,
    masked_smoothed_ce,
    split_train_val,
)

from jackdaw.agents.hand_pointer_head import (  # noqa: E402
    MAX_PICKS,
    HandPointerBCModel,
    initial_type_mask,
    pick_step_mask,
)
from jackdaw.agents.hand_policy_v3 import FlatV3BCModel  # noqa: E402
from jackdaw.env.hand_play_gym import observation_space_v2  # noqa: E402

MASK_FUNCTIONS = (initial_type_mask, pick_step_mask)


def _device(device_str: str) -> torch.device:
    return torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str
    )


def _flat_compatible(dataset: DemoDataset) -> DemoDataset:
    """Keep exactly the rows the legacy flat head can represent."""

    return dataset.slice(torch.nonzero(dataset.actions >= 0).squeeze(-1))


def split_for_head(
    dataset: DemoDataset, head: str, val_fraction: float
) -> tuple[DemoDataset, DemoDataset]:
    """Split first, then apply the flat arm's representability filter."""

    train_set, val_set = split_train_val(dataset, val_fraction)
    if head == "flat":
        return _flat_compatible(train_set), _flat_compatible(val_set)
    if head == "pointer":
        return train_set, val_set
    raise ValueError(f"unknown head {head!r}; expected 'pointer' or 'flat'")


def pointer_step_active_mask(card_indices: torch.Tensor) -> torch.Tensor:
    """Return the non-padding mask for type plus pointer/stop steps."""

    lengths = (card_indices >= 0).sum(dim=-1)
    positions = torch.arange(MAX_PICKS, device=card_indices.device).unsqueeze(0)
    pointer_active = positions < lengths.unsqueeze(-1)
    pointer_active |= (positions == lengths.unsqueeze(-1)) & (lengths.unsqueeze(-1) < MAX_PICKS)
    type_active = torch.ones(
        (card_indices.shape[0], 1), dtype=torch.bool, device=card_indices.device
    )
    return torch.cat(
        [type_active, pointer_active],
        dim=-1,
    )


def mean_non_padding_token_ce(
    per_step_log_probs: torch.Tensor, card_indices: torch.Tensor
) -> torch.Tensor:
    """Mean UNSMOOTHED CE over real type, pick, and stop tokens only."""

    active = pointer_step_active_mask(card_indices)
    return (-(per_step_log_probs * active)).sum() / active.sum().clamp_min(1)


def pointer_smoothed_sequence_loss(
    per_step_log_probs: torch.Tensor,
    legal_uniform_log_probs: torch.Tensor,
    card_indices: torch.Tensor,
    label_smoothing: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-example sequence CE with per-step legal-set smoothing."""

    active = pointer_step_active_mask(card_indices)
    step_loss = (1.0 - label_smoothing) * (-per_step_log_probs)
    step_loss += label_smoothing * (-legal_uniform_log_probs)
    sequence_loss = (step_loss * active).sum(dim=-1)
    if sample_weights is not None:
        return (sequence_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)
    return sequence_loss.mean()


def _pointer_batch(
    model: HandPointerBCModel, batch: DemoDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = {key: value.to(device) for key, value in batch.obs.items()}
    card_latents, pooled = model.features_extractor(obs)
    hands_left, discards_left = model._budgets(obs)
    return (
        card_latents,
        pooled,
        obs["hand_mask"],
        hands_left,
        discards_left,
    )


@torch.no_grad()
def evaluate_pointer(
    model: HandPointerBCModel,
    dataset: DemoDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate teacher-forced NLL, greedy exact-set match, entropy, and VMSE."""

    model.eval()
    if not len(dataset):
        raise ValueError("cannot evaluate a pointer model on an empty dataset")
    total_nll = total_exact = total_entropy = total_vmse = total_token_ce = 0.0
    total_steps = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset.slice(torch.arange(start, min(start + batch_size, len(dataset))))
        card_latents, pooled, hand_mask, hands_left, discards_left = _pointer_batch(
            model, batch, device
        )
        per_step, sequence_log_prob, entropies, _ = model.pointer_head.teacher_forced_log_probs(
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            batch.action_types.to(device),
            batch.card_indices.to(device),
            return_uniform_log_probs=True,
        )
        active = pointer_step_active_mask(batch.card_indices.to(device))
        decoded_types, decoded_indices = model.decode(
            {key: value.to(device) for key, value in batch.obs.items()}
        )
        for row, picked in enumerate(decoded_indices):
            length = int((batch.card_indices[row] >= 0).sum())
            exact = decoded_types[row].item() == batch.action_types[row].item()
            exact = exact and torch.equal(picked, batch.card_indices[row, :length].to(device))
            total_exact += float(exact)
        total_nll += (-sequence_log_prob).sum().item()
        total_entropy += (entropies * active).sum().item()
        total_steps += int(active.sum())
        total_vmse += F.mse_loss(
            model.value_net(pooled).squeeze(-1), batch.p_clear.to(device), reduction="sum"
        ).item()
        total_token_ce += -(per_step * active).sum().item()
    n = len(dataset)
    return {
        "sequence_nll": total_nll / n,
        "exact_set_match_accuracy": total_exact / n,
        "mean_per_step_entropy": total_entropy / max(total_steps, 1),
        "value_mse": total_vmse / n,
        "mean_non_padding_token_ce": total_token_ce / max(total_steps, 1),
    }


def evaluate_flat(
    model: FlatV3BCModel,
    dataset: DemoDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Use the legacy flat evaluation convention with the v3 control model."""

    if not len(dataset):
        raise ValueError("cannot evaluate the flat model on an empty dataset")
    return evaluate(model, dataset, batch_size, device)


def _train_epoch_pointer(
    model: HandPointerBCModel,
    train_set: DemoDataset,
    batch_size: int,
    device: torch.device,
    label_smoothing: float,
    value_coef: float,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    perm = torch.randperm(len(train_set))
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, len(train_set), batch_size):
        batch = train_set.slice(perm[start : start + batch_size])
        card_latents, pooled, hand_mask, hands_left, discards_left = _pointer_batch(
            model, batch, device
        )
        per_step, _, _, uniform = model.pointer_head.teacher_forced_log_probs(
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            batch.action_types.to(device),
            batch.card_indices.to(device),
            return_uniform_log_probs=True,
        )
        sequence_ce = pointer_smoothed_sequence_loss(
            per_step,
            uniform,
            batch.card_indices.to(device),
            label_smoothing,
            batch.sample_weights.to(device),
        )
        value_mse = F.mse_loss(model.value_net(pooled).squeeze(-1), batch.p_clear.to(device))
        loss = sequence_ce + value_coef * value_mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


def _train_epoch_flat(
    model: FlatV3BCModel,
    train_set: DemoDataset,
    batch_size: int,
    device: torch.device,
    label_smoothing: float,
    value_coef: float,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    perm = torch.randperm(len(train_set))
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, len(train_set), batch_size):
        batch = train_set.slice(perm[start : start + batch_size])
        obs = {key: value.to(device) for key, value in batch.obs.items()}
        logits, values = model(obs)
        ce = masked_smoothed_ce(
            logits,
            batch.actions.to(device),
            batch.legal_masks.to(device),
            label_smoothing,
            batch.sample_weights.to(device),
        )
        value_mse = F.mse_loss(values, batch.p_clear.to(device))
        loss = ce + value_coef * value_mse
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


@torch.no_grad()
def _mean_pointer_token_ce(
    model: HandPointerBCModel,
    dataset: DemoDataset,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_ce = 0.0
    total_steps = 0
    for start in range(0, len(dataset), batch_size):
        batch = dataset.slice(torch.arange(start, min(start + batch_size, len(dataset))))
        card_latents, pooled, hand_mask, hands_left, discards_left = _pointer_batch(
            model, batch, device
        )
        per_step, _, _, _ = model.pointer_head.teacher_forced_log_probs(
            card_latents,
            pooled,
            hand_mask,
            hands_left,
            discards_left,
            batch.action_types.to(device),
            batch.card_indices.to(device),
            return_uniform_log_probs=True,
        )
        active = pointer_step_active_mask(batch.card_indices.to(device))
        total_ce += float((-(per_step * active)).sum().item())
        total_steps += int(active.sum())
    return total_ce / max(total_steps, 1)


def _run_pointer_memorization_canary(
    train_set: DemoDataset,
    *,
    device: torch.device,
    seed: int,
    lr: float,
    max_epochs: int = 200,
    target_ce: float = 0.05,
) -> dict[str, float | int | bool]:
    """Overfit a fresh pointer model on the first 50 training examples."""

    canary_set = train_set.slice(torch.arange(min(50, len(train_set))))
    if not len(canary_set):
        raise ValueError("memorization canary requires at least one training example")

    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        model = HandPointerBCModel(observation_space_v2()).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=max(lr, 3e-3), weight_decay=0.0)
        final_ce = float("inf")
        epochs = 0
        for epoch in range(max_epochs):
            _train_epoch_pointer(
                model,
                canary_set,
                len(canary_set),
                device,
                label_smoothing=0.0,
                value_coef=0.0,
                optimizer=optimizer,
            )
            final_ce = _mean_pointer_token_ce(model, canary_set, len(canary_set), device)
            epochs = epoch + 1
            if final_ce < target_ce:
                break

    return {
        "final_ce": final_ce,
        "epochs": epochs,
        "passed": final_ce < target_ce,
    }


def train(
    dataset: DemoDataset,
    output_dir: Path,
    *,
    head: str = "pointer",
    max_epochs: int = 10,
    patience: int = 2,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    value_coef: float = 0.5,
    val_fraction: float = 0.10,
    device_str: str = "auto",
    seed: int = 0,
    data_dirs: list[str | Path] | None = None,
    stage_weights: dict[str, float] | None = None,
    metadata_extra: dict | None = None,
    _run_canary: bool = True,
) -> Path:
    """Train one arm and save its best validation checkpoint."""

    if head not in {"pointer", "flat"}:
        raise ValueError(f"unknown head {head!r}; expected 'pointer' or 'flat'")
    torch.manual_seed(seed)
    device = _device(device_str)
    full_train_set, full_val_set = split_train_val(dataset, val_fraction)
    train_set, val_set = (
        (full_train_set, full_val_set)
        if head == "pointer"
        else (_flat_compatible(full_train_set), _flat_compatible(full_val_set))
    )
    dropped_wide_count = int((dataset.actions < 0).sum()) if head == "flat" else 0
    dropped_wide_fraction = dropped_wide_count / max(len(dataset), 1) if head == "flat" else 0.0
    if head == "flat":
        print(
            f"DROPPED-WIDE FRACTION: {dropped_wide_fraction:.6f} "
            f"({dropped_wide_count}/{len(dataset)})"
        )
    if not len(train_set) or not len(val_set):
        raise ValueError(
            f"{head} arm requires non-empty train and validation sets "
            f"(train={len(train_set)}, val={len(val_set)})"
        )

    model: HandPointerBCModel | FlatV3BCModel
    if head == "pointer":
        model = HandPointerBCModel(observation_space_v2()).to(device)
    else:
        model = FlatV3BCModel(observation_space_v2()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        if head == "pointer":
            train_loss = _train_epoch_pointer(
                model,
                train_set,
                batch_size,
                device,
                label_smoothing,
                value_coef,
                optimizer,
            )
            val_metrics = evaluate_pointer(model, val_set, batch_size, device)
            val_nll = val_metrics["sequence_nll"]
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_sequence_nll={val_nll:.4f} "
                f"val_exact_set_match={val_metrics['exact_set_match_accuracy']:.3f} "
                f"val_entropy={val_metrics['mean_per_step_entropy']:.3f} "
                f"val_vmse={val_metrics['value_mse']:.4f}"
            )
        else:
            train_loss = _train_epoch_flat(
                model,
                train_set,
                batch_size,
                device,
                label_smoothing,
                value_coef,
                optimizer,
            )
            val_metrics = evaluate_flat(model, val_set, batch_size, device)
            val_nll = val_metrics["ce"]
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} "
                f"val_ce={val_nll:.4f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_entropy={val_metrics['entropy']:.3f} "
                f"val_vmse={val_metrics['value_mse']:.4f}"
            )
        val_metrics["train_loss"] = train_loss
        history.append(val_metrics)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stop after epoch {epoch} (best epoch {best_epoch})")
                break

    assert best_state is not None
    canary: dict[str, float | int | bool] | None = None
    if head == "pointer" and _run_canary:
        canary = _run_pointer_memorization_canary(
            train_set,
            device=device,
            seed=seed,
            lr=lr,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"bc_v3_{head}.pt"
    metadata = {
        "head": head,
        "data_dirs": [str(path) for path in (data_dirs or [])],
        "stage_weights": stage_weights or {},
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "best_val_sequence_nll": best_val_nll if head == "pointer" else None,
        "best_val_ce": best_val_nll if head == "flat" else None,
        "history": history,
        "num_train": len(train_set),
        "num_val": len(val_set),
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "value_coef": value_coef,
        "val_fraction": val_fraction,
        "seed": seed,
        "device": str(device),
        "num_actions": 436,
        "dropped_wide_fraction": dropped_wide_fraction,
        "dropped_wide_count": dropped_wide_count,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **(metadata_extra or {}),
    }
    if canary is not None:
        metadata.update(
            {
                "canary_final_ce": canary["final_ce"],
                "canary_epochs": canary["epochs"],
                "canary_passed": canary["passed"],
                "memorization_canary_mean_non_padding_token_ce": canary["final_ce"],
            }
        )
    torch.save({"model_state_dict": best_state, "metadata": metadata}, checkpoint_path)
    metrics_payload = {
        "head": head,
        "best_epoch": best_epoch,
        "best_val_nll": best_val_nll,
        "history": history,
    }
    for metrics_path in (
        output_dir / f"bc_v3_{head}_metrics.json",
        output_dir / "bc_metrics.json",
    ):
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics_payload, file, indent=2)
    print(f"checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=("pointer", "flat"), required=True)
    parser.add_argument("--data-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--stage-weight", action="append", default=[], metavar="STAGE=WEIGHT")
    parser.add_argument("--output", type=Path, default=Path("runs/bc_v3/default"))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stage_weights = {}
    for spec in args.stage_weight:
        name, _, value = spec.partition("=")
        stage_weights[name] = float(value)
    dataset = load_dataset(args.data_dirs, stage_weights)
    train(
        dataset,
        args.output,
        head=args.head,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        value_coef=args.value_coef,
        val_fraction=args.val_fraction,
        device_str=args.device,
        seed=args.seed,
        data_dirs=args.data_dirs,
        stage_weights=stage_weights,
    )


if __name__ == "__main__":
    main()
