"""Tests for shop PPO Joker embedding extraction."""

from __future__ import annotations

from types import SimpleNamespace

import extract_shop_joker_embeddings as extract  # noqa: E402
import numpy as np
import torch

from jackdaw.agents.joker_descriptors import DESCRIPTOR_DIM
from jackdaw.env.observation import NUM_CENTER_KEYS


def test_joker_vocabulary_uses_frozen_ids():
    keys, ids = extract.joker_vocabulary()

    assert len(keys) == len(ids)
    assert len(keys) > 0
    assert all(key.startswith("j_") for key in keys)
    assert np.all(ids > 0)
    assert np.all(ids[:-1] < ids[1:])


def test_extract_returns_learned_embeddings_and_descriptors(monkeypatch):
    embedding = torch.nn.Embedding(NUM_CENTER_KEYS + 1, 16, padding_idx=0)
    descriptors = torch.zeros(NUM_CENTER_KEYS + 1, DESCRIPTOR_DIM)
    extractor = SimpleNamespace(embedding=embedding, descriptors=descriptors)
    model = SimpleNamespace(policy=SimpleNamespace(features_extractor=extractor))
    monkeypatch.setattr(extract, "_load_shop_model", lambda checkpoint, device: model)

    result = extract.extract_joker_embeddings("checkpoint.zip")

    assert result["embeddings"].shape == (len(result["center_keys"]), 16)
    assert result["descriptors"].shape == (len(result["center_keys"]), DESCRIPTOR_DIM)
    assert result["center_key_ids"].dtype == np.int64
    np.testing.assert_allclose(
        result["embeddings"],
        embedding(torch.as_tensor(result["center_key_ids"])).detach().numpy(),
    )


def test_save_embeddings_round_trip(tmp_path):
    data = {
        "center_keys": np.asarray(["j_joker", "j_greedy_joker"]),
        "center_key_ids": np.asarray([146, 133], dtype=np.int64),
        "embeddings": np.zeros((2, 16), dtype=np.float32),
        "descriptors": np.zeros((2, DESCRIPTOR_DIM), dtype=np.float32),
    }
    output = tmp_path / "joker_embeddings.npz"

    extract.save_embeddings(output, data)

    with np.load(output) as loaded:
        np.testing.assert_array_equal(loaded["center_keys"], data["center_keys"])
        np.testing.assert_array_equal(loaded["center_key_ids"], data["center_key_ids"])
        np.testing.assert_array_equal(loaded["embeddings"], data["embeddings"])


def test_image_output_alias_sets_plot_path():
    args = extract.build_parser().parse_args(
        ["--checkpoint", "model.zip", "--output", "vectors.npz", "--image-output", "plot.png"]
    )

    assert args.plot.name == "plot.png"
