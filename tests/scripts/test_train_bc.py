"""Tests for the BC training pipeline (`scripts/train_bc.py`).

Uses fast synthetic shards (correct schema, random features, guaranteed-
legal labels) rather than solver-labeled ones -- generation is ~12s/example
and irrelevant to what's under test here: shard loading and label mapping,
legality reconstruction/drift guards, the deterministic val split, masked-
smoothed CE semantics, and an end-to-end 2-epoch training smoke that
produces a loadable checkpoint with entropy history in its metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_bc import (  # noqa: E402
    EXPECTED_SCHEMA_VERSION,
    load_dataset,
    masked_smoothed_ce,
    split_train_val,
    train,
)

from jackdaw.agents.hand_action_space import (  # noqa: E402
    NUM_COMBOS,
    NUM_HAND_ACTIONS,
    action_to_combo,
)
from jackdaw.agents.hand_policy import HandPlayBCModel  # noqa: E402
from jackdaw.env.hand_play_gym import (  # noqa: E402
    MAX_CONSUMABLES_V2,
    MAX_HAND_CARDS_OBS,
    MAX_JOKERS_V2,
    observation_space_v2,
)
from jackdaw.env.observation import (  # noqa: E402
    D_CONSUMABLE,
    D_HAND_CARD,
    D_HAND_GLOBAL,
    D_JOKER,
)

# Shards write 8-wide hand blocks (the action space's position count); the
# loader up-pads them to MAX_HAND_CARDS_OBS.
MAX_HAND_CARDS = 8
# The demo writer emits joker blocks at the fixed v2 width; the loader does
# NOT up-pad the joker axis (nor trigger_match's joker axis), so synthetic
# shards must match the obs width exactly, as real shards do.
MAX_JOKERS = MAX_JOKERS_V2


def _write_synthetic_shard(
    path,
    n: int,
    *,
    rng: np.random.Generator,
    hands_left: int = 2,
    discards_left: int = 1,
    seed_prefix: str = "synth",
    start_idx: int = 0,
    schema_version: int = EXPECTED_SCHEMA_VERSION,
) -> None:
    """Emit a shard matching generate_hand_demos.write_shard's schema."""
    global_context = np.zeros((n, D_HAND_GLOBAL), dtype=np.float32)
    global_context[:, 13] = hands_left / 10.0
    global_context[:, 14] = discards_left / 10.0
    global_context[:, :] += rng.normal(0, 0.01, size=global_context.shape).astype(np.float32)
    # Keep the two legality scalars exact after noise:
    global_context[:, 13] = hands_left / 10.0
    global_context[:, 14] = discards_left / 10.0

    hand_mask = np.ones((n, MAX_HAND_CARDS), dtype=bool)
    action_type = np.zeros(n, dtype=np.int64)
    card_indices = np.full((n, 5), -1, dtype=np.int64)
    for i in range(n):
        action_type[i] = rng.integers(0, 2) if discards_left > 0 else 0
        k = int(rng.integers(1, 6))
        card_indices[i, :k] = np.sort(rng.choice(MAX_HAND_CARDS, size=k, replace=False))

    np.savez_compressed(
        path,
        schema_version=np.array([schema_version]),
        global_context=global_context,
        hand_cards=rng.normal(size=(n, MAX_HAND_CARDS, D_HAND_CARD)).astype(np.float32),
        hand_mask=hand_mask,
        jokers=rng.normal(size=(n, MAX_JOKERS, D_JOKER)).astype(np.float32),
        joker_mask=np.ones((n, MAX_JOKERS), dtype=bool),
        joker_ids=rng.integers(0, 300, size=(n, MAX_JOKERS)).astype(np.int64),
        copy_active=np.zeros((n, MAX_JOKERS), dtype=np.float32),
        copy_target_ids=np.zeros((n, MAX_JOKERS), dtype=np.int64),
        trigger_match=rng.integers(0, 2, size=(n, MAX_HAND_CARDS, MAX_JOKERS, 2)).astype(bool),
        consumables=rng.normal(size=(n, MAX_CONSUMABLES_V2, D_CONSUMABLE)).astype(np.float32),
        consumable_mask=np.zeros((n, MAX_CONSUMABLES_V2), dtype=bool),
        action_type=action_type,
        card_indices=card_indices,
        p_clear=rng.uniform(0, 1, size=n).astype(np.float32),
        seed=np.array([f"{seed_prefix}_{start_idx + i:08d}" for i in range(n)]),
    )


@pytest.fixture
def stage_dir(tmp_path):
    d = tmp_path / "stage_synth"
    d.mkdir()
    rng = np.random.default_rng(0)
    _write_synthetic_shard(d / "worker_000_shard_00000.npz", 40, rng=rng, start_idx=0)
    _write_synthetic_shard(d / "worker_000_shard_00001.npz", 40, rng=rng, start_idx=40)
    return d


class TestLoadDataset:
    def test_shapes_and_label_mapping(self, stage_dir):
        ds = load_dataset([stage_dir], {})
        assert len(ds) == 80
        assert ds.obs["global_context"].shape == (80, D_HAND_GLOBAL)
        # 8-wide shard hand blocks are up-padded to the observation width --
        # including trigger_match's hand axis.
        assert ds.obs["hand_cards"].shape == (80, MAX_HAND_CARDS_OBS, D_HAND_CARD)
        assert ds.obs["hand_mask"].shape == (80, MAX_HAND_CARDS_OBS)
        assert torch.all(ds.obs["hand_mask"][:, MAX_HAND_CARDS:] == 0)
        assert ds.obs["trigger_match"].shape == (80, MAX_HAND_CARDS_OBS, MAX_JOKERS, 2)
        assert torch.all(ds.obs["trigger_match"][:, MAX_HAND_CARDS:] == 0)
        # Real consumable block loaded from the shard, not synthesized.
        assert ds.obs["consumables"].shape == (80, MAX_CONSUMABLES_V2, D_CONSUMABLE)
        assert ds.obs["joker_ids"].dtype == torch.int64
        assert ds.obs["copy_target_ids"].dtype == torch.int64
        assert ds.action_types.shape == (80,)
        assert ds.card_indices.shape == (80, 5)
        assert ds.legal_masks.shape == (80, NUM_HAND_ACTIONS)
        # Every label decodes to a 1-5 card combo and is legal.
        for i in range(0, 80, 7):
            action_type, combo = action_to_combo(int(ds.actions[i]))
            assert 1 <= len(combo) <= 5
            assert ds.legal_masks[i, ds.actions[i]]

    def test_rejects_wrong_schema_version(self, tmp_path):
        d = tmp_path / "bad_schema"
        d.mkdir()
        _write_synthetic_shard(
            d / "worker_000_shard_00000.npz",
            4,
            rng=np.random.default_rng(1),
            schema_version=99,
        )
        with pytest.raises(ValueError, match="schema_version"):
            load_dataset([d], {})

    def test_rejects_v1_shards(self, tmp_path):
        # Pre-regen (v1) datasets have a different feature layout; loading
        # them must fail loudly, not train on misaligned features.
        d = tmp_path / "v1_data"
        d.mkdir()
        _write_synthetic_shard(
            d / "worker_000_shard_00000.npz",
            4,
            rng=np.random.default_rng(4),
            schema_version=2,
        )
        with pytest.raises(ValueError, match="schema_version"):
            load_dataset([d], {})

    def test_rejects_illegal_label(self, tmp_path):
        # discards_left=0 in the global context but Discard-labeled examples
        # (action_type forced by hand) -> drift guard must fire.
        d = tmp_path / "drift"
        d.mkdir()
        rng = np.random.default_rng(2)
        path = d / "worker_000_shard_00000.npz"
        _write_synthetic_shard(path, 4, rng=rng, discards_left=1)
        data = dict(np.load(path))
        data["global_context"][:, 14] = 0.0  # discards_left -> 0
        data["action_type"][:] = 1  # all Discard labels
        np.savez_compressed(path, **data)
        with pytest.raises(ValueError, match="illegal"):
            load_dataset([d], {})

    def test_rejects_non_integer_card_indices(self, tmp_path):
        d = tmp_path / "non_integer_label"
        d.mkdir()
        path = d / "worker_000_shard_00000.npz"
        _write_synthetic_shard(path, 4, rng=np.random.default_rng(5))
        data = dict(np.load(path))
        data["card_indices"] = data["card_indices"].astype(np.float32)
        data["card_indices"][0, 0] = 0.5
        np.savez_compressed(path, **data)
        with pytest.raises(ValueError, match="integer dtype"):
            load_dataset([d], {})

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset([tmp_path / "empty"], {})

    def test_rejects_overwide_hand_block(self, tmp_path):
        # A shard wider than the observation space means the widths have
        # drifted the wrong way round -- must fail, not silently truncate.
        d = tmp_path / "wide"
        d.mkdir()
        rng = np.random.default_rng(3)
        path = d / "worker_000_shard_00000.npz"
        _write_synthetic_shard(path, 4, rng=rng)
        data = dict(np.load(path))
        extra = MAX_HAND_CARDS_OBS - MAX_HAND_CARDS + 1
        data["hand_cards"] = np.concatenate(
            [data["hand_cards"], np.zeros((4, extra, D_HAND_CARD), np.float32)], axis=1
        )
        data["hand_mask"] = np.concatenate([data["hand_mask"], np.zeros((4, extra), bool)], axis=1)
        np.savez_compressed(path, **data)
        with pytest.raises(ValueError, match="exceeds obs width"):
            load_dataset([d], {})

    def test_stage_weights_applied(self, stage_dir):
        ds = load_dataset([stage_dir], {"stage_synth": 2.5})
        assert torch.all(ds.sample_weights == 2.5)


class TestSplit:
    def test_split_deterministic_and_disjoint(self, stage_dir):
        ds = load_dataset([stage_dir], {})
        train_a, val_a = split_train_val(ds, 0.2)
        train_b, val_b = split_train_val(ds, 0.2)
        assert train_a.seeds == train_b.seeds
        assert val_a.seeds == val_b.seeds
        assert set(train_a.seeds).isdisjoint(val_a.seeds)
        assert len(train_a) + len(val_a) == len(ds)
        assert len(val_a) > 0


class TestMaskedSmoothedCE:
    def test_smoothing_mass_stays_on_legal_set(self):
        # With logits uniform over legal actions, CE should equal
        # log(n_legal) regardless of smoothing (smoothing redistributes
        # within the legal set only).
        n_legal = 10
        legal = torch.zeros(1, NUM_HAND_ACTIONS, dtype=torch.bool)
        legal[0, :n_legal] = True
        logits = torch.zeros(1, NUM_HAND_ACTIONS)
        actions = torch.tensor([3])
        loss = masked_smoothed_ce(logits, actions, legal, label_smoothing=0.05)
        assert torch.isclose(loss, torch.log(torch.tensor(float(n_legal))), atol=1e-5)

    def test_perfect_logits_penalized_by_smoothing_only(self):
        legal = torch.zeros(1, NUM_HAND_ACTIONS, dtype=torch.bool)
        legal[0, :4] = True
        logits = torch.full((1, NUM_HAND_ACTIONS), -30.0)
        logits[0, 2] = 30.0
        actions = torch.tensor([2])
        sharp = masked_smoothed_ce(logits, actions, legal, label_smoothing=0.0)
        smoothed = masked_smoothed_ce(logits, actions, legal, label_smoothing=0.05)
        assert sharp < 1e-4
        assert smoothed > sharp  # smoothing bounds peakedness


class TestTrainSmoke:
    def test_two_epoch_run_saves_loadable_checkpoint(self, stage_dir, tmp_path):
        ds = load_dataset([stage_dir], {})
        out = tmp_path / "bc_out"
        ckpt_path = train(
            ds,
            out,
            max_epochs=2,
            patience=2,
            batch_size=32,
            val_fraction=0.2,
            device_str="cpu",
            seed=0,
        )
        assert ckpt_path.exists()
        ckpt = torch.load(ckpt_path, weights_only=False)
        meta = ckpt["metadata"]
        assert meta["num_actions"] == NUM_HAND_ACTIONS
        assert len(meta["history"]) >= 1
        assert "entropy" in meta["history"][0]  # over-sharpening diagnosable later
        model = HandPlayBCModel(observation_space_v2())
        model.load_state_dict(ckpt["model_state_dict"])  # loads cleanly
        assert (out / "bc_metrics.json").exists()

    def test_discard_block_gets_probability_when_labels_include_discards(self, stage_dir, tmp_path):
        # Sanity: a trained model puts nonzero mass on the discard block
        # (labels are ~50/50 play/discard in the synthetic set).
        ds = load_dataset([stage_dir], {})
        ckpt_path = train(
            ds, tmp_path / "bc2", max_epochs=2, batch_size=32, device_str="cpu", seed=1
        )
        model = HandPlayBCModel(observation_space_v2())
        model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
        model.eval()
        with torch.no_grad():
            obs = {k: v[:16] for k, v in ds.obs.items()}
            log_probs = model.masked_log_probs(obs, ds.legal_masks[:16])
        discard_mass = log_probs.exp()[:, NUM_COMBOS:].sum(dim=-1)
        assert (discard_mass > 0.05).any()
