"""Focused CLI wiring tests for the shop eval script."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import eval_shop_policy  # noqa: E402
import pytest

from jackdaw.agents.shop_action_space import ShopActionFamily, shop_action  # noqa: E402


def test_s1_schema_flag_reaches_run_suite(monkeypatch):
    captured = {}

    monkeypatch.setattr(eval_shop_policy, "load_policy", lambda policy, device: object())

    def fake_run_suite(policy, win_ante, n_episodes, **kwargs):
        captured.update(kwargs)
        return {"n_played": 0, "n_dead_at_reset": 0}

    monkeypatch.setattr(eval_shop_policy, "run_suite", fake_run_suite)
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_shop_policy.py", "--policy", "nextround", "--s1-schema"],
    )

    eval_shop_policy.main()

    assert captured["s1_schema"] is True


def test_partner_money_ordering_requires_hand_policy(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_shop_policy.py", "--policy", "nextround", "--partner-money-ordering"],
    )

    with pytest.raises(SystemExit):
        eval_shop_policy.main()


def test_dump_decisions_writes_full_trace_without_changing_metrics(tmp_path: Path):
    trace_path = tmp_path / "trace.jsonl"
    nextround_policy = eval_shop_policy.load_policy("nextround", "cpu")
    eval_shop_policy.run_suite(
        nextround_policy,
        win_ante=2,
        n_episodes=2,
        dump_decisions=trace_path,
    )

    assert trace_path.exists()
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 1
    records = [json.loads(line) for line in lines]
    required_keys = {
        "seed",
        "step",
        "ante",
        "round",
        "dollars",
        "pending_target",
        "action",
        "action_family",
        "action_slot",
        "action_label",
        "n_legal",
        "legal_actions",
        "terminal",
        "won",
    }
    assert required_keys <= records[0].keys()
    assert len({record["seed"] for record in records}) == 2
    assert all(
        sum(record["terminal"] for record in records if record["seed"] == seed) == 1
        for seed in {record["seed"] for record in records}
    )
    assert all(
        record["action_family"] in {family.name for family in ShopActionFamily}
        for record in records
    )
    assert all(record["n_legal"] == len(record["legal_actions"]) for record in records)

    baseline = eval_shop_policy.run_suite(
        eval_shop_policy.load_policy("nextround", "cpu"),
        win_ante=2,
        n_episodes=2,
    )
    without_dump = eval_shop_policy.run_suite(
        eval_shop_policy.load_policy("nextround", "cpu"),
        win_ante=2,
        n_episodes=2,
    )
    assert without_dump == baseline


def test_dump_decisions_captures_exact_cards_and_game_state(tmp_path: Path):
    class BuyFirstPolicy:
        def act(self, obs, mask):
            buy = shop_action(ShopActionFamily.BuyCard, 0)
            if mask[buy]:
                return buy
            for family in (ShopActionFamily.NextRound, ShopActionFamily.SkipPack):
                action = shop_action(family)
                if mask[action]:
                    return action
            return int(mask.nonzero()[0][0])

    trace_path = tmp_path / "rich-trace.jsonl"
    eval_shop_policy.run_suite(
        BuyFirstPolicy(),
        win_ante=2,
        n_episodes=1,
        dump_decisions=trace_path,
    )

    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    record = next(item for item in records if item["action_family"] == "BuyCard")
    pre_state = record["pre_state"]
    post_state = record["post_state"]

    assert pre_state["shop"]["cards"]
    assert pre_state["shop"]["cards"][0]["center_key"]
    assert isinstance(pre_state["inventory"]["jokers"], list)
    assert isinstance(post_state["inventory"]["jokers"], list)
    assert pre_state["hand_levels"]
    assert pre_state["current_round"]
    assert record["action_target"]["kind"] == "shop_card"
    assert record["action_target"]["card"] == pre_state["shop"]["cards"][0]


def test_extended_sell_joker_target_uses_absolute_joker_row():
    jokers = [
        SimpleNamespace(
            sort_id=i,
            center_key=f"j_test_{i}",
            card_key=None,
            ability={"name": f"Joker {i}", "set": "Joker"},
            base=None,
            edition=None,
            seal=None,
            debuff=False,
            base_cost=0,
            cost=0,
            sell_cost=0,
            extra_cost=0,
            eternal=False,
            perishable=False,
            perish_tally=5,
            rental=False,
        )
        for i in range(9)
    ]
    action = shop_action(ShopActionFamily.SellJokerExt, 0)

    target = eval_shop_policy._serialize_action_target(
        ShopActionFamily.SellJokerExt, 0, action, {"jokers": jokers}
    )

    assert target["slot"] == 8
    assert target["action_slot"] == 0
    assert target["card"]["center_key"] == "j_test_8"
