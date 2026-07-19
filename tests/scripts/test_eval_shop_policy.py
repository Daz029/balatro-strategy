"""Focused CLI wiring tests for the shop eval script."""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("torch")

import eval_shop_policy  # noqa: E402


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
