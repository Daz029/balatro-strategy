"""Tests for the rich shop trace CSV exporter."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import export_shop_decisions_csv  # noqa: E402


def _card(name: str, center_key: str, set_name: str = "Joker", cost: int = 4) -> dict:
    return {
        "name": name,
        "center_key": center_key,
        "card_key": None,
        "set": set_name,
        "cost": cost,
        "sell_cost": 2,
        "edition": None,
        "debuff": False,
    }


def _record(phase: str = "shop") -> dict:
    joker = _card("Sly Joker", "j_sly")
    offered = _card("Mad Joker", "j_mad")
    return {
        "seed": "EVAL_00000000",
        "step": 7,
        "ante": 2,
        "round": 4,
        "action": 500,
        "action_family": "BuyCard",
        "action_slot": 0,
        "action_label": "BuyCard[0]",
        "n_legal": 2,
        "legal_actions": [461, 500],
        "action_target": {"kind": "shop_card", "slot": 0, "action_slot": 0, "card": offered},
        "terminal": False,
        "won": None,
        "pre_state": {
            "phase": phase,
            "ante": 2,
            "round": 4,
            "dollars": 10,
            "blind_on_deck": "Boss",
            "resources": {"joker_slots": 5},
            "inventory": {"jokers": [joker], "consumables": []},
            "shop": {"cards": [offered], "vouchers": [], "boosters": []},
            "pack": {"cards": [], "hand": []},
        },
        "post_state": {
            "ante": 2,
            "round": 4,
            "dollars": 6,
            "inventory": {"jokers": [joker, offered], "consumables": []},
        },
    }


def test_convert_trace_flattens_shop_decision(tmp_path: Path):
    source = tmp_path / "trace.jsonl"
    output = tmp_path / "decisions.csv"
    source.write_text(
        "\n".join(json.dumps(record) for record in (_record(), _record("blind_select"))) + "\n",
        encoding="utf-8",
    )

    written, skipped = export_shop_decisions_csv.convert_trace(source, output)

    assert (written, skipped) == (1, 1)
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    row = rows[0]
    assert row["jokers_before"] == "[0] Sly Joker (j_sly) $4"
    assert row["shop_cards"] == "[0] Mad Joker (j_mad) $4"
    assert row["bought_joker"] == "[0] Mad Joker (j_mad) $4"
    assert row["dollars_before"] == "10"
    assert row["dollars_after"] == "6"
    assert row["money_delta"] == "-4"
    assert json.loads(row["available_cards_json"])["shop_cards"][0]["center_key"] == "j_mad"


def test_all_phases_can_be_exported(tmp_path: Path):
    source = tmp_path / "trace.jsonl"
    output = tmp_path / "decisions.csv"
    source.write_text(json.dumps(_record("blind_select")) + "\n", encoding="utf-8")

    assert export_shop_decisions_csv.convert_trace(source, output, phases=None) == (1, 0)
