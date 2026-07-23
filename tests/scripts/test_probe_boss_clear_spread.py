from __future__ import annotations

import dataclasses
import json
import pickle

import numpy as np
import pytest
from probe_boss_clear_spread import (
    build_probe_record,
    build_summary,
    detect_shop_s1_schema,
    prepare_redeal,
    reseed_deal_stream,
)

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.run_init import initialize_run
from jackdaw.env.shop_obs import D_SHOP_CONTEXT, D_SHOP_CONTEXT_S1


class StubPartner:
    obs_version = 1
    action_version = 1

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self._legal_action_offsets = iter([0] * 64)

    def predict_value(self, obs: dict[str, np.ndarray]) -> float:
        return next(self._values)

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray | None) -> int:
        assert mask is not None
        legal_actions = np.flatnonzero(mask)
        return int(legal_actions[next(self._legal_action_offsets)])


class _CtxSpace:
    def __init__(self, width: int) -> None:
        self.shape = (width,)


def test_detect_shop_s1_schema_reads_context_width() -> None:
    assert detect_shop_s1_schema({"shop_context": _CtxSpace(D_SHOP_CONTEXT)}) is False
    assert detect_shop_s1_schema({"shop_context": _CtxSpace(D_SHOP_CONTEXT_S1)}) is True
    with pytest.raises(ValueError, match="unrecognized shop_context width"):
        detect_shop_s1_schema({"shop_context": _CtxSpace(99)})


def _predeal_boss_blob() -> bytes:
    gs = initialize_run("b_red", 1, "BOSSPROBE_FIXTURE")
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Boss"
    joker = create_joker("j_green_joker")
    joker.ability["mult"] = 17
    joker.set_edition({"foil": True})
    joker.eternal = True
    joker.perishable = True
    joker.perish_tally = 2
    joker.rental = True
    gs["jokers"] = [joker]
    return pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL)


def _card_ids(gs: dict) -> tuple[int, ...]:
    return tuple(sorted(card.sort_id for card in gs["hand"]))


def test_redeal_changes_only_the_deal_stream_and_preserves_the_build() -> None:
    predeal = pickle.loads(_predeal_boss_blob())
    before_rng = dict(predeal["rng"].state)
    stream = reseed_deal_stream(predeal, "BOSSPROBE_FIXTURE_REDEAL_0000")

    assert stream == "nr1"
    assert {key: value for key, value in predeal["rng"].state.items() if key != stream} == {
        key: value for key, value in before_rng.items() if key != stream
    }
    assert predeal["rng"].state[stream] != before_rng.get(stream)

    blob = _predeal_boss_blob()
    opening_a = pickle.loads(prepare_redeal(blob, "BOSSPROBE_FIXTURE_REDEAL_0000"))
    opening_b = pickle.loads(prepare_redeal(blob, "BOSSPROBE_FIXTURE_REDEAL_0001"))

    assert opening_a["phase"] == GamePhase.SELECTING_HAND
    assert opening_b["phase"] == GamePhase.SELECTING_HAND
    assert _card_ids(opening_a) != _card_ids(opening_b)
    assert opening_a["jokers"] == opening_b["jokers"]
    assert opening_a["hand_levels"]._hands == opening_b["hand_levels"]._hands
    assert opening_a["dollars"] == opening_b["dollars"]
    assert opening_a["current_round"] == opening_b["current_round"]
    assert opening_a["blind"].key == opening_b["blind"].key


def test_stub_partner_builds_complete_raw_record() -> None:
    values = [0.2, 0.4, 0.6]
    record = build_probe_record(
        _predeal_boss_blob(),
        run_seed="BOSSPROBE_00000000",
        n_redeals=3,
        partner=StubPartner(values),
        keep_blob=True,
    )

    assert record["run_seed"] == "BOSSPROBE_00000000"
    assert record["terminal_ante"] == 1
    assert record["boss_key"].startswith("bl_")
    assert set(record) == {
        "run_seed",
        "terminal_ante",
        "boss_key",
        "build",
        "redeals",
        "sampled_clear",
        "critic_mean",
        "n",
        "predeal_blob_b64",
        "predeal_sort_id_counter",
    }
    assert set(record["build"]) == {
        "jokers",
        "hand_levels",
        "vouchers",
        "deck_signature",
        "dollars",
        "hands_left",
        "discards_left",
    }
    assert len(record["redeals"]) == 3
    assert record["critic_mean"] == pytest.approx(0.4)
    assert record["sampled_clear"] == sum(row["cleared"] for row in record["redeals"]) / 3
    assert record["n"] == 3
    assert isinstance(record["predeal_blob_b64"], str)
    assert record["predeal_sort_id_counter"] is None
    assert [row["v"] for row in record["redeals"]] == values
    assert [row["redeal_seed"] for row in record["redeals"]] == [
        f"BOSSPROBE_00000000_REDEAL_{index:04d}" for index in range(3)
    ]
    assert all(set(row) == {"redeal_seed", "v", "cleared"} for row in record["redeals"])

    joker = record["build"]["jokers"][0]
    assert set(joker) == {field.name for field in dataclasses.fields(create_joker("j_joker"))}
    assert joker["center_key"] == "j_green_joker"
    assert joker["ability"]["mult"] == 17
    assert joker["edition"]["foil"] is True
    assert joker["eternal"] is True
    assert joker["perishable"] is True
    assert joker["perish_tally"] == 2
    assert joker["rental"] is True
    assert record["build"]["hand_levels"]
    assert record["build"]["vouchers"] == {}
    assert isinstance(record["build"]["dollars"], int)
    assert isinstance(record["build"]["hands_left"], int)
    assert isinstance(record["build"]["discards_left"], int)
    assert record["build"]["deck_signature"]["cards"]
    assert json.loads(json.dumps(record)) == record


def test_summary_matches_hand_computed_fixture() -> None:
    records = [
        {
            "sampled_clear": 0.0,
            "critic_mean": 0.2,
            "build": {"jokers": [{"center_key": "j_a"}]},
        },
        {
            "sampled_clear": 0.5,
            "critic_mean": 0.5,
            "build": {"jokers": [{"center_key": "j_a"}, {"center_key": "j_b"}]},
        },
        {
            "sampled_clear": 1.0,
            "critic_mean": 0.8,
            "build": {"jokers": [{"center_key": "j_c"}]},
        },
    ]

    summary = build_summary(records)

    assert summary["sampled_clear_distribution"] == {
        "n_builds": 3,
        "min": 0.0,
        "p25": 0.25,
        "median": 0.5,
        "p75": 0.75,
        "max": 1.0,
        "mean": 0.5,
        "spread": 1.0,
    }
    assert summary["critic_vs_sampled"] == {"correlation": 1.0, "mean_abs_error": 2 / 15}
    assert summary["coverage"] == {
        "distinct_jokers": 3,
        "appearance_counts": {"j_a": 2, "j_b": 1, "j_c": 1},
    }
    assert "on-policy" in summary["coverage_caveat"]
    assert "synthetic random-subset" in summary["coverage_caveat"]
