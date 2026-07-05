"""Behavior-cloning pretraining for the hand-play agent.

Trains ``HandPlayBCModel`` (the exact three modules the PPO policy will
hold -- see ``jackdaw/agents/hand_policy.py``) on solver-labeled demo
shards from ``generate_hand_demos.py``:

  - **Data pooling**: all listed stage directories pool into one dataset
    (BC is plain supervised learning -- sequential stage training would
    just forget earlier stages; the per-stage split exists for generation
    reproducibility, not BC ordering). Optional per-stage sampling weights.
  - **Deliberate partial convergence** (CLAUDE.md: preserve entropy for
    PPO): early stop on val CE with patience 2, capped at --max-epochs,
    best-val-epoch checkpoint kept; label smoothing 0.05 bounds how peaked
    the policy can get regardless of epochs; per-epoch val entropy is
    logged AND stored in checkpoint metadata so a misbehaving PPO run can
    be diagnosed against over-sharpened BC after the fact.
  - **Masked cross-entropy**: illegal actions (reconstructed from each
    example's hand size and hands/discards-left) are excluded from both the
    softmax and the smoothing mass, matching MaskablePPO's masking
    semantics -- BC never places probability on actions PPO can't see.
  - **Value head**: MSE regression on the solver's ``p_clear`` (critic
    warm start; with terminal 1/0 reward and gamma=1.0 the PPO critic
    target is the same quantity).
  - **Val split**: deterministic by CRC32 of the example's seed string --
    reproducible across reruns and independent of shard layout.

Usage::

    uv run python scripts/train_bc.py \
        --data-dirs data/hand_agent_demos/stage1_no_jokers \
                    data/hand_agent_demos/stage2_curated \
        --output runs/bc/run1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from jackdaw.agents.hand_action_space import (  # noqa: E402
    NUM_HAND_ACTIONS,
    legal_action_mask,
    mask_to_action,
)
from jackdaw.agents.hand_policy import HandPlayBCModel  # noqa: E402
from jackdaw.env.hand_play_gym import (  # noqa: E402
    MAX_CONSUMABLES,
    observation_space,
)
from jackdaw.env.observation import D_CONSUMABLE  # noqa: E402

EXPECTED_SCHEMA_VERSION = 1

# Indices of hands_left / discards_left in the global context vector
# (scalars block, see observation.py encode_global_context: v[13], v[14],
# both stored divided by 10).
_GC_HANDS_LEFT_IDX = 13
_GC_DISCARDS_LEFT_IDX = 14


@dataclass
class DemoDataset:
    """All demo examples in memory as tensors (26k examples ~ 40 MB)."""

    obs: dict[str, torch.Tensor]  # each (N, ...)
    actions: torch.Tensor  # (N,) long — canonical action indices
    legal_masks: torch.Tensor  # (N, NUM_HAND_ACTIONS) bool
    p_clear: torch.Tensor  # (N,) float
    sample_weights: torch.Tensor  # (N,) float — per-stage weighting
    seeds: list[str]

    def __len__(self) -> int:
        return self.actions.shape[0]

    def slice(self, idx: torch.Tensor) -> DemoDataset:
        return DemoDataset(
            obs={k: v[idx] for k, v in self.obs.items()},
            actions=self.actions[idx],
            legal_masks=self.legal_masks[idx],
            p_clear=self.p_clear[idx],
            sample_weights=self.sample_weights[idx],
            seeds=[self.seeds[i] for i in idx.tolist()],
        )


def _reconstruct_legal_mask(global_context: np.ndarray, hand_mask: np.ndarray) -> np.ndarray:
    """Rebuild the (NUM_HAND_ACTIONS,) legality mask for one example from
    quantities encoded in the observation itself."""
    hand_size = int(hand_mask.sum())
    hands_left = int(round(float(global_context[_GC_HANDS_LEFT_IDX]) * 10))
    discards_left = int(round(float(global_context[_GC_DISCARDS_LEFT_IDX]) * 10))
    return legal_action_mask(hand_size, hands_left, discards_left)


def load_dataset(data_dirs: list[Path], stage_weights: dict[str, float]) -> DemoDataset:
    """Load and pool every shard from the given stage directories.

    Raises on schema mismatch or on a label that is illegal under its own
    reconstructed mask -- either means the dataset and this code have
    drifted, and training on it would be silently wrong.
    """
    obs_parts: dict[str, list[np.ndarray]] = {
        "global_context": [],
        "hand_cards": [],
        "hand_mask": [],
        "jokers": [],
        "joker_mask": [],
    }
    actions: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    p_clear: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    seeds: list[str] = []

    for data_dir in data_dirs:
        shard_paths = sorted(data_dir.glob("worker_*_shard_*.npz"))
        if not shard_paths:
            raise FileNotFoundError(f"no shards found in {data_dir}")
        stage_weight = stage_weights.get(data_dir.name, 1.0)
        for shard_path in shard_paths:
            shard = np.load(shard_path)
            version = int(shard["schema_version"][0])
            if version != EXPECTED_SCHEMA_VERSION:
                raise ValueError(
                    f"{shard_path}: schema_version {version} != {EXPECTED_SCHEMA_VERSION}"
                )
            n = shard["action_type"].shape[0]
            for key in obs_parts:
                obs_parts[key].append(shard[key])
            shard_actions = np.array(
                [
                    mask_to_action(int(shard["action_type"][i]), shard["card_target_mask"][i])
                    for i in range(n)
                ],
                dtype=np.int64,
            )
            shard_legal = np.stack(
                [
                    _reconstruct_legal_mask(shard["global_context"][i], shard["hand_mask"][i])
                    for i in range(n)
                ]
            )
            illegal_labels = ~shard_legal[np.arange(n), shard_actions]
            if illegal_labels.any():
                bad = shard["seed"][illegal_labels][:3]
                raise ValueError(
                    f"{shard_path}: {int(illegal_labels.sum())} labels illegal under "
                    f"their reconstructed masks (e.g. {bad}) -- dataset/code drift"
                )
            actions.append(shard_actions)
            legal_masks.append(shard_legal)
            p_clear.append(shard["p_clear"])
            weights.append(np.full(n, stage_weight, dtype=np.float32))
            seeds.extend(str(s) for s in shard["seed"])

    n_total = len(seeds)
    obs = {
        "global_context": torch.as_tensor(
            np.concatenate(obs_parts["global_context"]), dtype=torch.float32
        ),
        "hand_cards": torch.as_tensor(
            np.concatenate(obs_parts["hand_cards"]), dtype=torch.float32
        ),
        "hand_mask": torch.as_tensor(
            np.concatenate(obs_parts["hand_mask"]), dtype=torch.float32
        ),
        "jokers": torch.as_tensor(np.concatenate(obs_parts["jokers"]), dtype=torch.float32),
        "joker_mask": torch.as_tensor(
            np.concatenate(obs_parts["joker_mask"]), dtype=torch.float32
        ),
        # Dormant consumable block (see hand_play_gym.py): shards don't
        # carry it, synthesize the always-empty arrays.
        "consumables": torch.zeros(n_total, MAX_CONSUMABLES, D_CONSUMABLE),
        "consumable_mask": torch.zeros(n_total, MAX_CONSUMABLES),
    }
    return DemoDataset(
        obs=obs,
        actions=torch.as_tensor(np.concatenate(actions)),
        legal_masks=torch.as_tensor(np.concatenate(legal_masks)),
        p_clear=torch.as_tensor(np.concatenate(p_clear), dtype=torch.float32),
        sample_weights=torch.as_tensor(np.concatenate(weights)),
        seeds=seeds,
    )


def split_train_val(dataset: DemoDataset, val_fraction: float) -> tuple[DemoDataset, DemoDataset]:
    """Deterministic split by CRC32 of the seed string."""
    buckets = 1000
    threshold = int(val_fraction * buckets)
    is_val = torch.tensor(
        [zlib.crc32(s.encode()) % buckets < threshold for s in dataset.seeds]
    )
    return dataset.slice(torch.nonzero(~is_val).squeeze(-1)), dataset.slice(
        torch.nonzero(is_val).squeeze(-1)
    )


def masked_smoothed_ce(
    logits: torch.Tensor,
    actions: torch.Tensor,
    legal_masks: torch.Tensor,
    label_smoothing: float,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over legal actions only, smoothing mass spread over the
    legal set only (uniform smoothing would leak probability onto actions
    MaskablePPO will never expose)."""
    masked_logits = logits.masked_fill(~legal_masks, float("-inf"))
    log_probs = torch.log_softmax(masked_logits, dim=-1)
    nll = -log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    legal_f = legal_masks.float()
    smooth_term = -(log_probs.masked_fill(~legal_masks, 0.0) * legal_f).sum(1) / legal_f.sum(
        1
    ).clamp(min=1.0)
    loss = (1.0 - label_smoothing) * nll + label_smoothing * smooth_term
    if sample_weights is not None:
        return (loss * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)
    return loss.mean()


@torch.no_grad()
def evaluate(
    model: HandPlayBCModel,
    dataset: DemoDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_ce = total_acc = total_ent = total_vmse = 0.0
    n = len(dataset)
    for start in range(0, n, batch_size):
        idx = torch.arange(start, min(start + batch_size, n))
        batch = dataset.slice(idx)
        obs = {k: v.to(device) for k, v in batch.obs.items()}
        legal = batch.legal_masks.to(device)
        actions = batch.actions.to(device)
        logits, values = model(obs)
        masked_logits = logits.masked_fill(~legal, float("-inf"))
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        ce = -log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        acc = (masked_logits.argmax(dim=-1) == actions).float()
        probs = log_probs.exp()
        ent = -(probs * log_probs.masked_fill(~legal, 0.0)).sum(dim=-1)
        vmse = (values - batch.p_clear.to(device)) ** 2
        total_ce += ce.sum().item()
        total_acc += acc.sum().item()
        total_ent += ent.sum().item()
        total_vmse += vmse.sum().item()
    return {
        "ce": total_ce / n,
        "accuracy": total_acc / n,
        "entropy": total_ent / n,
        "value_mse": total_vmse / n,
    }


def train(
    dataset: DemoDataset,
    output_dir: Path,
    *,
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
    metadata_extra: dict | None = None,
) -> Path:
    """Run BC training; returns the path of the saved checkpoint."""
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else device_str
    )
    torch.manual_seed(seed)

    train_set, val_set = split_train_val(dataset, val_fraction)
    print(f"train={len(train_set)} val={len(val_set)} device={device}")

    model = HandPlayBCModel(observation_space()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_ce = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(train_set))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_set), batch_size):
            batch = train_set.slice(perm[start : start + batch_size])
            obs = {k: v.to(device) for k, v in batch.obs.items()}
            logits, values = model(obs)
            ce = masked_smoothed_ce(
                logits,
                batch.actions.to(device),
                batch.legal_masks.to(device),
                label_smoothing,
                batch.sample_weights.to(device),
            )
            vmse = torch.nn.functional.mse_loss(values, batch.p_clear.to(device))
            loss = ce + value_coef * vmse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_metrics = evaluate(model, val_set, batch_size, device)
        val_metrics["train_loss"] = epoch_loss / max(n_batches, 1)
        history.append(val_metrics)
        print(
            f"epoch {epoch}: train_loss={val_metrics['train_loss']:.4f} "
            f"val_ce={val_metrics['ce']:.4f} val_acc={val_metrics['accuracy']:.3f} "
            f"val_entropy={val_metrics['entropy']:.3f} val_vmse={val_metrics['value_mse']:.4f}"
        )

        if val_metrics["ce"] < best_val_ce:
            best_val_ce = val_metrics["ce"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early stop after epoch {epoch} (best epoch {best_epoch})")
                break

    assert best_state is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "bc_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "metadata": {
                "best_epoch": best_epoch,
                "best_val_ce": best_val_ce,
                "history": history,  # per-epoch entropy lives here (see docstring)
                "num_train": len(train_set),
                "num_val": len(val_set),
                "label_smoothing": label_smoothing,
                "value_coef": value_coef,
                "lr": lr,
                "batch_size": batch_size,
                "seed": seed,
                "num_actions": NUM_HAND_ACTIONS,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **(metadata_extra or {}),
            },
        },
        checkpoint_path,
    )
    with open(output_dir / "bc_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_epoch": best_epoch, "best_val_ce": best_val_ce, "history": history},
            f,
            indent=2,
        )
    print(f"checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Stage directories of demo shards; pooled into one dataset",
    )
    parser.add_argument(
        "--stage-weight",
        action="append",
        default=[],
        metavar="STAGE=WEIGHT",
        help="Per-stage loss weight, e.g. --stage-weight stage2_curated=2.0",
    )
    parser.add_argument("--output", type=Path, default=Path("runs/bc/default"))
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
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
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        label_smoothing=args.label_smoothing,
        value_coef=args.value_coef,
        val_fraction=args.val_fraction,
        device_str=args.device,
        seed=args.seed,
        metadata_extra={"data_dirs": [str(d) for d in args.data_dirs]},
    )


if __name__ == "__main__":
    main()
