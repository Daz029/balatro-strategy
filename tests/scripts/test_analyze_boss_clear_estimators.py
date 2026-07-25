from __future__ import annotations

import json

import numpy as np
import pytest
from analyze_boss_clear_estimators import (
    HALF,
    REDEALS,
    Record,
    corr,
    head_deal_sensitivity,
    load_corpus,
    money_correction,
    pooled_within_cell,
    retracted_decomposition,
    sampling_noise,
    spearman,
    spearman_brown,
    split_half,
)

from jackdaw.agents.v_curve import VCurve


def _clear_vector(rate: float, rng: np.random.Generator) -> np.ndarray:
    """40 outcomes averaging exactly ``rate``, so load_corpus's shuffle is harmless."""
    outcomes = np.zeros(REDEALS)
    outcomes[: round(rate * REDEALS)] = 1.0
    rng.shuffle(outcomes)
    return outcomes


def _record(*, ante: int, boss: str, dollars: float, head: float, target: float) -> Record:
    """A Record whose held-out target is EXACTLY ``target`` (no shuffle applied).

    Structural tests build these directly rather than round-tripping through
    load_corpus, whose permutation would randomise the half a designed target
    lives in. ``target`` must be a multiple of 1/HALF.
    """
    second = np.zeros(HALF)
    second[: round(target * HALF)] = 1.0
    return Record(
        ante=ante,
        boss=boss,
        dollars=dollars,
        v=np.full(REDEALS, head),
        c=np.concatenate([np.zeros(HALF), second]),
    )


def _write_corpus(path, rows) -> str:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "terminal_ante": row["ante"],
                        "boss_key": row["boss"],
                        "build": {"dollars": row["dollars"]},
                        "redeals": [
                            {"v": float(v), "cleared": float(c)}
                            for v, c in zip(row["v"], row["c"])
                        ],
                    }
                )
                + "\n"
            )
    return str(path)


@pytest.fixture
def corpus_path(tmp_path):
    """60 builds whose head is deal-blind, mirroring the real corpus's failure mode."""
    rng = np.random.default_rng(7)
    rows = []
    for index in range(60):
        p = (index % 20) / 20.0
        cleared = _clear_vector(p, rng)
        head_value = p + rng.normal(0, 0.05)
        rows.append(
            {
                "ante": 1 + index % 3,
                "boss": f"bl_{index % 2}",
                "dollars": float(index % 15),
                "v": np.full(REDEALS, head_value) + rng.normal(0, 0.01, REDEALS),
                "c": cleared,
            }
        )
    return _write_corpus(tmp_path / "corpus.jsonl", rows)


def test_load_corpus_is_seed_deterministic_and_skips_short_records(tmp_path):
    """Permutation comes from one seeded stream; short records can't form halves."""
    rng = np.random.default_rng(0)
    rows = [
        {
            "ante": 1,
            "boss": "bl_a",
            "dollars": 5.0,
            "v": rng.normal(size=REDEALS),
            "c": rng.integers(0, 2, REDEALS).astype(float),
        },
        {"ante": 2, "boss": "bl_b", "dollars": 6.0, "v": np.zeros(9), "c": np.zeros(9)},
    ]
    path = _write_corpus(tmp_path / "c.jsonl", rows)

    first = load_corpus(path, seed=3)
    assert len(first) == 1, "the 9-redeal record must be skipped"
    assert np.array_equal(first[0].v, load_corpus(path, seed=3)[0].v)
    assert not np.array_equal(first[0].v, load_corpus(path, seed=4)[0].v)


def test_spearman_absorbs_monotone_distortion():
    """Rank correlation is the model-free control for the multiplicative squash."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = x**3
    assert corr(x, y) < 0.95
    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_brown_extrapolates_to_double_length():
    assert spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
    assert spearman_brown(0.908) == pytest.approx(0.9518, abs=1e-4)


def test_playout_improves_with_n_while_deal_blind_head_saturates(corpus_path):
    """The core finding: sampling buys accuracy per sample, querying the head does not."""
    result = split_half(load_corpus(corpus_path), ns=(1, 4, 20))
    playout = {row["n"]: row["playout"] for row in result["rows"]}
    head = {row["n"]: row["head"] for row in result["rows"]}

    assert playout[1] < playout[4] < playout[20]
    assert abs(head[20] - head[1]) < 0.02


def test_head_deal_sensitivity_detects_a_deal_blind_head(corpus_path):
    """A head that ignores the deal shows near-zero within-build variance share."""
    result = head_deal_sensitivity(load_corpus(corpus_path))
    assert result["within_share"] < 0.05
    assert abs(result["within_build_corr"]) < 0.2


def test_retracted_decomposition_reproduces_its_own_artifact(corpus_path):
    """The deal-blind head scores absurd 'efficiency' -- why the section is retracted."""
    result = retracted_decomposition(load_corpus(corpus_path))
    assert result["median_efficiency"] > 10.0


def test_within_cell_pooling_removes_cell_offsets():
    """Cell means are what inflate pooled correlations; centring must strip them."""
    records = [
        _record(ante=ante, boss=boss, dollars=base + offset, head=0.0, target=rate)
        for ante, boss, base, rates in (
            (1, "bl_a", 0.0, [0.15, 0.10, 0.05, 0.00]),
            (2, "bl_b", 10.0, [0.95, 0.90, 0.85, 0.80]),
        )
        for offset, rate in enumerate(rates)
    ]

    dollars = [r.dollars for r in records]
    targets = [r.target for r in records]
    # Ignoring cells, dollars look strongly POSITIVELY related to clearing...
    assert corr(dollars, targets) > 0.9
    # ...but inside every cell the relationship is exactly inverted.
    assert pooled_within_cell(
        records, lambda r: r.dollars, min_cell=4
    ) == pytest.approx(-1.0)


def test_sampling_noise_overlap_identity(corpus_path):
    """|half - full| == |half - other half| / 2 exactly; the 40-sample figure flatters."""
    result = sampling_noise(load_corpus(corpus_path))
    assert result["overlapping"]["mean_abs_diff"] == pytest.approx(
        result["disjoint"]["mean_abs_diff"] / 2.0
    )
    assert result["overlapping"]["corr"] > result["disjoint"]["corr"]


def test_money_correction_recovers_a_multiplicatively_contaminated_head():
    """Dividing by (1 + v_curve) must undo the contamination when the model holds."""
    v_curve = VCurve(
        {1: {d: 0.1 * d for d in range(16)}, 2: {d: 0.1 * d for d in range(16)}},
        dollar_min=0,
        dollar_max=15,
    )
    records = []
    for index in range(60):
        p = (index % 20) / 20.0
        dollars = float(index % 15)
        ante = 1 + index % 2
        records.append(
            _record(
                ante=ante,
                boss=f"bl_{index % 2}",
                dollars=dollars,
                head=p * (1.0 + v_curve.value(ante, dollars)),
                target=p,
            )
        )

    result = money_correction(records, v_curve, min_cell=5)
    corrected = result["variants"][0]["pearson"]
    assert corrected > result["raw_pearson"] + 0.05
    assert corrected == pytest.approx(1.0)
