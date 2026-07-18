"""Contract tests for the pre-registered BC gate harness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import eval_bc_gate as gate  # noqa: E402
from test_train_bc_v3 import _synthetic_dataset  # noqa: E402
from train_bc import EXPECTED_SCHEMA_VERSION  # noqa: E402
from train_bc_v3 import train  # noqa: E402

from jackdaw.agents.hand_pointer_head import PointerActionHead  # noqa: E402


def _write_dataset_shard(path: Path, dataset) -> None:
    payload = {key: value.detach().cpu().numpy() for key, value in dataset.obs.items()}
    payload.update(
        schema_version=np.array([EXPECTED_SCHEMA_VERSION]),
        action_type=dataset.action_types.numpy(),
        card_indices=dataset.card_indices.numpy(),
        p_clear=dataset.p_clear.numpy(),
        seed=np.asarray(dataset.seeds),
    )
    np.savez_compressed(path, **payload)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("bc_gate")
    dataset = _synthetic_dataset(64)
    data_dir = root / "stage_synthetic"
    data_dir.mkdir()
    _write_dataset_shard(data_dir / "worker_000_shard_00000.npz", dataset)
    pointer_checkpoint = train(
        dataset,
        root / "pointer",
        head="pointer",
        max_epochs=1,
        patience=1,
        batch_size=64,
        val_fraction=0.2,
        device_str="cpu",
        seed=11,
    )
    flat_checkpoint = train(
        dataset,
        root / "flat",
        head="flat",
        max_epochs=1,
        patience=1,
        batch_size=64,
        val_fraction=0.2,
        device_str="cpu",
        seed=12,
    )
    output = root / "report"
    report = gate.evaluate_gate(
        pointer_checkpoint,
        flat_checkpoint,
        [data_dir],
        output,
        val_fraction=0.2,
        beam_width=4,
        device_str="cpu",
    )
    return {
        "root": root,
        "dataset": dataset,
        "data_dir": data_dir,
        "pointer": pointer_checkpoint,
        "flat": flat_checkpoint,
        "output": output,
        "report": report,
    }


def test_gate_writes_all_required_tables_and_verdict(artifacts):
    report = artifacts["report"]
    assert (artifacts["output"] / "report.json").exists()
    assert (artifacts["output"] / "report.md").exists()
    assert {
        "head_to_head",
        "wide",
        "type_token_accuracy",
        "stop_token_accuracy",
        "per_pick_position_nll",
        "predicted_vs_true_set_size",
        "free_running_termination_audit",
        "entropy_by_decode_step",
        "p_clear_head_mse",
        "greedy_vs_beam_decode",
    } <= set(report["tables"])
    assert {"checks", "overall", "winrate"} <= set(report["verdict"])
    for size in range(1, 6):
        assert str(size) in report["tables"]["head_to_head"]["by_set_size"]
        assert str(size) in report["tables"]["predicted_vs_true_set_size"]["by_true_set_size"]
        assert "n" in report["tables"]["head_to_head"]["by_set_size"][str(size)]


def test_free_running_actions_are_all_valid(artifacts):
    audit = artifacts["report"]["tables"]["free_running_termination_audit"]
    assert audit["aggregate"]["invalid_rate"] == 0
    assert all(case["valid"] for case in audit["decoded_actions"])


def test_empty_wide_stratum_is_incomplete(artifacts, monkeypatch, tmp_path):
    no_wide = _synthetic_dataset(32, repeated=True)
    monkeypatch.setattr(gate, "load_dataset", lambda *_args: no_wide)
    report = gate.evaluate_gate(
        artifacts["pointer"],
        artifacts["flat"],
        [tmp_path / "not_used"],
        tmp_path / "empty_wide_report",
        val_fraction=0.2,
        device_str="cpu",
    )
    wide = report["tables"]["wide"]["aggregate"]
    assert wide["n"] == 0
    assert report["verdict"]["overall"] == "INCOMPLETE"


def test_overrun_is_dumped_and_fails_verdict(artifacts, monkeypatch, tmp_path):
    real_pointer_logits = PointerActionHead._pointer_logits

    def pick_favoring_logits(self, state, card_latents):
        logits = real_pointer_logits(self, state, card_latents)
        logits[..., : gate.CARD_SLOTS] = 1.0
        logits[..., gate.STOP_INDEX] = -1.0
        return logits

    monkeypatch.setattr(PointerActionHead, "_pointer_logits", pick_favoring_logits)
    report = gate.evaluate_gate(
        artifacts["pointer"],
        artifacts["flat"],
        [artifacts["data_dir"]],
        tmp_path / "overrun_report",
        val_fraction=0.2,
        device_str="cpu",
    )
    cases = report["tables"]["free_running_termination_audit"]["overrun_cases"]
    assert cases and {"seed", "true_label", "decoded_action"} <= set(cases[0])
    assert any(
        check["id"] == "(b) free_running_and_tokens" and check["status"] == "FAIL"
        for check in report["verdict"]["checks"]
    )
    assert report["verdict"]["overall"] == "FAIL"


def test_beam_width_one_equals_greedy(artifacts, tmp_path):
    report = gate.evaluate_gate(
        artifacts["pointer"],
        artifacts["flat"],
        [artifacts["data_dir"]],
        tmp_path / "beam_one_report",
        val_fraction=0.2,
        beam_width=1,
        device_str="cpu",
    )
    beam = report["tables"]["greedy_vs_beam_decode"]["aggregate"]
    assert beam["greedy_action_disagreement_rate"] == 0
    assert beam["greedy_set_disagreement_rate"] == 0
    assert beam["beam_sequence_validity"] == 1


def test_gate_report_is_deterministic(artifacts, tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    gate.evaluate_gate(
        artifacts["pointer"],
        artifacts["flat"],
        [artifacts["data_dir"]],
        first_dir,
        val_fraction=0.2,
        device_str="cpu",
    )
    gate.evaluate_gate(
        artifacts["pointer"],
        artifacts["flat"],
        [artifacts["data_dir"]],
        second_dir,
        val_fraction=0.2,
        device_str="cpu",
    )
    assert json.loads((first_dir / "report.json").read_text()) == json.loads(
        (second_dir / "report.json").read_text()
    )
