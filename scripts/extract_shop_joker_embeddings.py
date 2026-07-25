"""Extract and visualize learned Joker embeddings from a shop PPO checkpoint.

Example::

    python scripts/extract_shop_joker_embeddings.py \
        --checkpoint runs/shop_ppo/s2_a4/shop_ppo_final.zip \
        --output data/shop_joker_embeddings.npz \
        --image-output data/shop_joker_embeddings_umap.png

The checkpoint's learned identity vectors are exported separately from the
frozen engine-derived descriptors. The UMAP plot is computed from the learned
vectors only, so its geometry reflects the PPO trunk's embedding table rather
than the hand-authored descriptor features.

UMAP and matplotlib are imported lazily. Loading/extracting the vectors can
therefore be used independently, while plotting reports an actionable install
hint when ``umap-learn`` is not available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _path in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from jackdaw.env.observation import center_key_id, center_key_vocabulary  # noqa: E402


def joker_vocabulary() -> tuple[tuple[str, ...], np.ndarray]:
    """Return Joker keys and their frozen center-key ids in vocabulary order."""
    keys = tuple(key for key in center_key_vocabulary() if key.startswith("j_"))
    ids = np.asarray([center_key_id(key) for key in keys], dtype=np.int64)
    return keys, ids


def _load_shop_model(checkpoint: Path, device: str) -> Any:
    """Load a checkpoint and verify that it contains the shop PPO trunk."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError(
            "Shop PPO checkpoint loading requires the training dependencies; "
            "install the project's train extra first."
        ) from exc

    model = MaskablePPO.load(str(checkpoint), device=device)
    from jackdaw.agents.shop_policy import ShopFeaturesExtractor

    extractor = model.policy.features_extractor
    if not isinstance(extractor, ShopFeaturesExtractor):
        raise TypeError(
            "checkpoint does not contain ShopFeaturesExtractor; "
            f"found {type(extractor).__name__}"
        )
    return model


def extract_joker_embeddings(checkpoint: Path, device: str = "cpu") -> dict[str, Any]:
    """Extract learned Joker embeddings and static descriptors from a checkpoint."""
    import torch

    model = _load_shop_model(checkpoint, device)
    extractor = model.policy.features_extractor
    keys, ids = joker_vocabulary()
    id_tensor = torch.as_tensor(ids, dtype=torch.long, device=extractor.embedding.weight.device)

    with torch.no_grad():
        embeddings = extractor.embedding(id_tensor).cpu().numpy()
        descriptors = extractor.descriptors[id_tensor].cpu().numpy()

    expected_width = int(extractor.embedding.embedding_dim)
    if embeddings.shape != (len(keys), expected_width):
        raise RuntimeError(
            f"unexpected Joker embedding shape {embeddings.shape}; "
            f"expected ({len(keys)}, {expected_width})"
        )

    return {
        "center_keys": np.asarray(keys),
        "center_key_ids": ids,
        "embeddings": embeddings.astype(np.float32, copy=False),
        "descriptors": descriptors.astype(np.float32, copy=False),
        "embedding_dim": np.asarray(expected_width, dtype=np.int64),
        "checkpoint": np.asarray(str(checkpoint)),
    }


def save_embeddings(output: Path, data: dict[str, Any]) -> None:
    """Write extracted embeddings and metadata as a compressed NumPy archive."""
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)


def project_umap(
    embeddings: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Project embedding rows to two dimensions with reproducible UMAP."""
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be a 2D array, got shape {embeddings.shape}")
    if len(embeddings) < 3:
        raise ValueError("UMAP requires at least three embedding rows")
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be at least 2, got {n_neighbors}")
    if not 0.0 <= min_dist:
        raise ValueError(f"min_dist must be non-negative, got {min_dist}")

    try:
        from umap import UMAP
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "UMAP plotting requires 'umap-learn'. Install it with "
            "'.venv\\Scripts\\python -m pip install umap-learn'."
        ) from exc

    reducer = UMAP(
        n_components=2,
        n_neighbors=min(n_neighbors, len(embeddings) - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def save_umap_plot(
    plot_path: Path,
    coordinates: np.ndarray,
    center_keys: np.ndarray,
    *,
    label_points: bool = True,
    title: str = "Shop PPO Joker embeddings (UMAP)",
) -> None:
    """Render and save a labeled 2D UMAP scatter plot."""
    try:
        import matplotlib

        # This script writes an artifact and never opens a GUI window. Using a
        # non-interactive backend keeps it usable on CI and headless servers.
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError(
            "Plotting requires matplotlib. Install the project's train extra first."
        ) from exc

    if coordinates.shape != (len(center_keys), 2):
        raise ValueError(
            f"coordinates must have shape ({len(center_keys)}, 2), "
            f"got {coordinates.shape}"
        )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=34, alpha=0.85)
    if label_points:
        for (x, y), key in zip(coordinates, center_keys, strict=True):
            label = str(key)
            if label.startswith("j_"):
                label = label[2:]
            ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="shop PPO .zip checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="compressed .npz output path")
    parser.add_argument(
        "--plot",
        "--image-output",
        dest="plot",
        type=Path,
        default=None,
        help="UMAP PNG path; defaults to the output path with a .png suffix",
    )
    parser.add_argument("--device", default="cpu", help="Torch device used to load the checkpoint")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--seed", type=int, default=42, help="UMAP random seed")
    parser.add_argument(
        "--no-labels", action="store_true", help="omit Joker names from the scatter plot"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data = extract_joker_embeddings(args.checkpoint, args.device)
    save_embeddings(args.output, data)

    plot_path = args.plot or args.output.with_suffix(".png")
    coordinates = project_umap(
        data["embeddings"],
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed,
    )
    data["umap_coordinates"] = coordinates
    # Rewrite the archive after projection so it contains the exact points
    # used for the PNG as well as the original embedding matrix.
    save_embeddings(args.output, data)
    save_umap_plot(
        plot_path,
        coordinates,
        data["center_keys"],
        label_points=not args.no_labels,
    )
    print(f"Extracted {len(data['center_keys'])} Joker embeddings to {args.output}")
    print(f"Saved UMAP plot to {plot_path}")


if __name__ == "__main__":
    main()
