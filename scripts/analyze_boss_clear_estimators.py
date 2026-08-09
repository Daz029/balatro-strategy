"""Estimator selection for the a4 reward alteration (docs/a4_reward_alteration.md).

Reproduces Findings 2 and 3 of that decision record from the probe corpus
written by ``probe_boss_clear_spread.py``.  The question it answers: to scale
the terminal shop reward by a build's deal-marginalized clear probability, do we
QUERY the partner's learned value head (one cheap forward pass) or SAMPLE it by
playing the boss out (expensive, exact by construction)?

Each corpus record is one build: a pre-``SelectBlind`` boss snapshot replayed 40
times with only the ``nr{ante}`` shuffle stream reseeded, so the build is held
byte-identical and only the deal varies.  Per replay it stores

* ``v``       -- the HEAD: h2's critic value at the opening state, pre-action,
* ``cleared`` -- the TRUTH sample: the partner actually plays the boss out.

The comparison is split-half and model-free.  Estimators are built from redeals
0-19; the target is the clear rate over the disjoint redeals 20-39.  Scoring an
estimator against its own samples would hand play-out a trivial 1.0.

Three things make the numbers honest, and all three were needed:

* **A ceiling.** The target is itself a 20-sample estimate, so nothing can
  correlate 1.0 with it.  Measured by splitting the held-out half 10-vs-10 and
  applying Spearman-Brown to N=20.
* **Stratification.** Pooling across antes inflates BOTH estimators, because
  ante alone predicts clear rate.  The alteration fires at a fixed horizon boss
  (at ``win_ante=4``, always ante 4), so the pooled number never described the
  use case.  Within-cell figures centre estimator and target inside each
  (ante, boss) cell before pooling.
* **Un-multiplying the head.**  h2's reward is ``1 + v_curve`` on a clear, so
  its head learned ``P(clear) * (1 + E[v_curve|clear])`` -- money MULTIPLIES,
  which is why it cannot be subtracted off.  It can be divided off, since the
  artifact h2 trained with is on disk.  Two model-free cross-checks (Spearman,
  and a gradient-boosted upper bound) run alongside so the verdict does not rest
  on that multiplicative model being exactly right.

The ``retracted-decomposition`` section is preserved deliberately: it reproduces
a law-of-total-variance argument that suggested the head was ~84x more
sample-efficient, which was WRONG because it assumed the head is a correct
conditional mean.  ``head-deal-sensitivity`` is the section that disproves that
assumption.  Keeping both makes the error reproducible rather than folklore.

Usage::

    python scripts/analyze_boss_clear_estimators.py \\
        --corpus data/boss_clear_probe.jsonl --v-curve data/v_curve_s1.json

Needs the ``analysis`` extra (``uv sync --extra analysis``) for scipy and
scikit-learn; nothing in ``jackdaw/`` imports either.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _path in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from jackdaw.agents.v_curve import VCurve, load_v_curve  # noqa: E402

REDEALS = 40
HALF = REDEALS // 2
DEFAULT_NS = (1, 2, 4, 8, 12, 16, 20)
DEFAULT_MIN_CELL = 15
DEFAULT_MIN_ANTE_BUILDS = 25

# Build-driven clear-probability std, net of sampling noise (Findings 1).
SIGNAL_STD = 0.18

SECTIONS = (
    "head-deal-sensitivity",
    "split-half",
    "within-ante",
    "within-cell",
    "money-correction",
    "sampling-noise",
    "retracted-decomposition",
)


@dataclasses.dataclass(frozen=True)
class Record:
    """One build: 40 paired (head value, played-out outcome) redeals."""

    ante: int
    boss: str
    dollars: float
    v: np.ndarray
    c: np.ndarray

    @property
    def target(self) -> float:
        """Held-out clear rate -- the truth every estimator is scored against."""
        return float(self.c[HALF:].mean())

    def head(self, n: int) -> float:
        return float(self.v[:n].mean())

    def playout(self, n: int) -> float:
        return float(self.c[:n].mean())


Getter = Callable[[Record], float]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_corpus(path: str | Path, *, seed: int = 0) -> list[Record]:
    """Load probe records, permuting each build's redeals under one seeded RNG.

    The permutation is what makes the split-half arbitrary rather than
    capture-ordered, and drawing it from a single stream in file order is what
    makes every figure in the decision record reproducible.  Records without a
    full 40 redeals are skipped: the halves and the Spearman-Brown ceiling both
    assume that width.
    """
    rng = np.random.default_rng(seed)
    records: list[Record] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            redeals = raw["redeals"]
            if len(redeals) != REDEALS:
                continue
            order = rng.permutation(REDEALS)
            records.append(
                Record(
                    ante=int(raw["terminal_ante"]),
                    boss=str(raw["boss_key"]),
                    dollars=float(raw["build"]["dollars"]),
                    v=np.asarray([r["v"] for r in redeals], dtype=float)[order],
                    c=np.asarray([r["cleared"] for r in redeals], dtype=float)[order],
                )
            )
    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def corr(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Pearson correlation, ``nan`` when either side is constant or too short."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if len(xs) < 3 or xs.std() == 0 or ys.std() == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def spearman(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Rank correlation -- scale-free, so it absorbs ANY monotone distortion."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if len(xs) < 3:
        return float("nan")
    return float(spearmanr(xs, ys).statistic)


def spearman_brown(reliability: float, factor: float = 2.0) -> float:
    """Reliability of an average ``factor`` times longer than the measured one."""
    if not np.isfinite(reliability):
        return float("nan")
    return float(factor * reliability / (1.0 + (factor - 1.0) * reliability))


def ceiling_20(records: Sequence[Record]) -> float:
    """Best correlation any estimator can reach with the 20-sample target.

    The target is noisy, so this is not 1.0.  Split the held-out half 10-vs-10
    to measure a 10-sample mean's reliability, then extrapolate to 20.
    """
    first = [float(r.c[HALF : HALF + 10].mean()) for r in records]
    second = [float(r.c[HALF + 10 :].mean()) for r in records]
    return spearman_brown(corr(first, second))


def pooled_within_cell(
    records: Iterable[Record],
    getter: Getter,
    *,
    min_cell: int = DEFAULT_MIN_CELL,
) -> float:
    """Correlation with the target after removing (ante, boss) cell means.

    Centring both sides inside the cell strips the ante and boss effects, which
    is the only way to read pure build-vs-build discrimination -- the decisive
    test the probe itself used.
    """
    cells: dict[tuple[int, str], list[Record]] = defaultdict(list)
    for record in records:
        cells[(record.ante, record.boss)].append(record)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for cell in cells.values():
        if len(cell) < min_cell:
            continue
        x = np.asarray([getter(r) for r in cell], dtype=float)
        y = np.asarray([r.target for r in cell], dtype=float)
        if x.std() == 0 or y.std() == 0:
            continue
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    if not xs:
        return float("nan")
    return corr(np.concatenate(xs), np.concatenate(ys))


def cell_count(
    records: Iterable[Record], *, min_cell: int = DEFAULT_MIN_CELL
) -> tuple[int, int]:
    """(kept cells, builds covered) at the given cell-size floor."""
    cells: dict[tuple[int, str], int] = defaultdict(int)
    for record in records:
        cells[(record.ante, record.boss)] += 1
    kept = [n for n in cells.values() if n >= min_cell]
    return len(kept), sum(kept)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def head_deal_sensitivity(records: Sequence[Record]) -> dict[str, Any]:
    """Does the head read the DEAL, or only the build?

    This is the measurement that invalidates the retracted decomposition: if the
    head barely varies across redeals, and what variation it has does not track
    the outcome, then its low across-deal variance is insensitivity rather than
    precision.
    """
    deltas_v: list[np.ndarray] = []
    deltas_c: list[np.ndarray] = []
    per_ante: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for record in records:
        if record.c.std() == 0 or record.v.std() == 0:
            continue
        dv = record.v - record.v.mean()
        dc = record.c - record.c.mean()
        deltas_v.append(dv)
        deltas_c.append(dc)
        per_ante[record.ante].append((dv, dc))

    pooled = (
        corr(np.concatenate(deltas_v), np.concatenate(deltas_c)) if deltas_v else float("nan")
    )
    by_ante = {}
    for ante, pairs in sorted(per_ante.items()):
        dv = np.concatenate([p[0] for p in pairs])
        dc = np.concatenate([p[1] for p in pairs])
        by_ante[ante] = {"corr": corr(dv, dc), "n_redeals": len(dv), "n_builds": len(pairs)}

    total_var = float(np.var(np.concatenate([r.v for r in records])))
    within_var = float(np.mean([np.var(r.v) for r in records]))
    return {
        "within_build_corr": pooled,
        "by_ante": by_ante,
        "total_head_var": total_var,
        "within_build_head_var": within_var,
        "within_share": within_var / total_var if total_var else float("nan"),
    }


def split_half(
    records: Sequence[Record], ns: Sequence[int] = DEFAULT_NS
) -> dict[str, Any]:
    """Model-free estimator curves against the disjoint held-out clear rate."""
    target = np.asarray([r.target for r in records], dtype=float)
    rows = []
    for n in ns:
        rows.append(
            {
                "n": n,
                "playout": corr([r.playout(n) for r in records], target),
                "head": corr([r.head(n) for r in records], target),
            }
        )
    return {"n_builds": len(records), "rows": rows, "ceiling_20": ceiling_20(records)}


def combined_gain(records: Sequence[Record]) -> dict[str, float]:
    """Does the head add anything ON TOP of play-outs? (Least squares, in-sample.)"""
    target = np.asarray([r.target for r in records], dtype=float)
    play = np.asarray([r.playout(HALF) for r in records], dtype=float)
    head = np.asarray([r.head(HALF) for r in records], dtype=float)
    design = np.column_stack([np.ones(len(records)), play, head])
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return {
        "playout_only": corr(play, target),
        "playout_plus_head": corr(design @ coef, target),
    }


def within_ante(
    records: Sequence[Record],
    ns: Sequence[int] = DEFAULT_NS,
    *,
    min_builds: int = DEFAULT_MIN_ANTE_BUILDS,
) -> list[dict[str, Any]]:
    """Split-half curves recomputed inside each ante.

    The alteration fires at a fixed horizon boss, so the pooled figure -- which
    lets both estimators score for merely knowing the depth -- is the wrong one.
    """
    rows = []
    for ante in sorted({r.ante for r in records}):
        subset = [r for r in records if r.ante == ante]
        if len(subset) < min_builds:
            continue
        result = split_half(subset, ns)
        rows.append(
            {
                "ante": ante,
                "n_builds": len(subset),
                "playout": {row["n"]: row["playout"] for row in result["rows"]},
                "head": {row["n"]: row["head"] for row in result["rows"]},
                "ceiling_20": result["ceiling_20"],
            }
        )
    return rows


def within_cell(
    records: Sequence[Record],
    ns: Sequence[int] = DEFAULT_NS,
    *,
    min_cell: int = DEFAULT_MIN_CELL,
) -> dict[str, Any]:
    """The strictest comparison: obstacle held constant, cell means removed."""
    cells, builds = cell_count(records, min_cell=min_cell)
    return {
        "n_cells": cells,
        "n_builds": builds,
        "playout": {
            n: pooled_within_cell(records, lambda r, n=n: r.playout(n), min_cell=min_cell)
            for n in ns
        },
        "head": {
            n: pooled_within_cell(records, lambda r, n=n: r.head(n), min_cell=min_cell)
            for n in ns
        },
    }


def money_correction(
    records: Sequence[Record],
    v_curve: VCurve,
    *,
    min_cell: int = DEFAULT_MIN_CELL,
) -> dict[str, Any]:
    """Un-multiply the head's ``v_curve`` term, and check whether it matters.

    h2's head learned ``P(clear) * (1 + E[v_curve|clear])``.  The money term is
    multiplicative, so it divides out -- given an estimate of it.  The variants
    differ in how they approximate ``E[v_curve|clear]``, whose true arguments are
    the post-cashout dollars and the post-clear ante, neither of which the
    corpus stores.  They land within 0.013 of each other, so the approximation
    is not what decides the verdict.
    """
    target = np.asarray([r.target for r in records], dtype=float)
    head = np.asarray([r.head(HALF) for r in records], dtype=float)

    variants = []
    for label, ante_offset, dollar_offset in (
        ("entry dollars, same ante", 0, 0.0),
        ("entry dollars, ante+1", 1, 0.0),
        ("dollars+5 (crude cashout), ante+1", 1, 5.0),
        ("dollars+10, ante+1", 1, 10.0),
    ):
        money = np.asarray(
            [
                v_curve.value(r.ante + ante_offset, r.dollars + dollar_offset)
                for r in records
            ],
            dtype=float,
        )
        corrected = head / (1.0 + money)
        variants.append(
            {
                "label": label,
                "pearson": corr(corrected, target),
                "spearman": spearman(corrected, target),
                "mean_money_term": float(money.mean()),
            }
        )

    def corrected_getter(record: Record) -> float:
        return record.head(HALF) / (1.0 + v_curve.value(record.ante, record.dollars))

    # Residual structure: if money were the story, residuals would be
    # dollar-shaped.  Quadratic fit so a monotone squash is not mistaken for it.
    shaping = [r for r in records if 1 <= r.ante <= 4]
    shaping_head = np.asarray([r.head(HALF) for r in shaping], dtype=float)
    shaping_target = np.asarray([r.target for r in shaping], dtype=float)
    fitted = np.polyval(np.polyfit(shaping_head, shaping_target, 2), shaping_head)
    residual = shaping_target - fitted

    ante4 = [r for r in records if r.ante == 4]
    return {
        "raw_pearson": corr(head, target),
        "raw_spearman": spearman(head, target),
        "variants": variants,
        "within_cell_raw": pooled_within_cell(
            records, lambda r: r.head(HALF), min_cell=min_cell
        ),
        "within_cell_corrected": pooled_within_cell(
            records, corrected_getter, min_cell=min_cell
        ),
        "ante4_cell_raw": pooled_within_cell(ante4, lambda r: r.head(HALF), min_cell=min_cell),
        "ante4_cell_corrected": pooled_within_cell(ante4, corrected_getter, min_cell=min_cell),
        "residual_vs_dollars": corr(residual, np.asarray([r.dollars for r in shaping])),
        "residual_vs_ante": corr(residual, np.asarray([float(r.ante) for r in shaping])),
    }


def gradient_boosted_bound(records: Sequence[Record]) -> dict[str, float]:
    """Upper bound on what ANY money-aware transform of this head can recover.

    Assumes no functional form, unlike the explicit division -- so if the head
    still cannot reach play-out here, no money correction rescues it.  The
    ``dollars + ante`` row is the load-bearing control: it shows how much of the
    head's apparent score is context rather than build evaluation.
    """
    target = np.asarray([r.target for r in records], dtype=float)
    head = np.asarray([r.head(HALF) for r in records], dtype=float)
    dollars = np.asarray([r.dollars for r in records], dtype=float)
    antes = np.asarray([float(r.ante) for r in records], dtype=float)

    results: dict[str, float] = {}
    for label, columns in (
        ("head only", [head]),
        ("head + dollars", [head, dollars]),
        ("head + dollars + ante", [head, dollars, antes]),
        ("dollars + ante ONLY (no head)", [dollars, antes]),
    ):
        model = HistGradientBoostingRegressor(max_iter=300, random_state=0)
        predicted = cross_val_predict(model, np.column_stack(columns), target, cv=5)
        results[label] = corr(predicted, target)
    return results


def sampling_noise(records: Sequence[Record]) -> dict[str, Any]:
    """How noisy is an N-sample clear rate, against the ~0.18 signal it must resolve?

    Two comparisons, and the gap between them is a trap: a half against the FULL
    40-sample mean is 50% OF that mean, so ``|A - full| == |A - B| / 2``
    identically.  The disjoint half-vs-half figure is the honest one.
    """
    first = np.asarray([r.playout(HALF) for r in records], dtype=float)
    second = np.asarray([r.target for r in records], dtype=float)
    full = np.asarray([float(r.c.mean()) for r in records], dtype=float)
    delta = first - second

    per_n = {}
    for n in (4, 8, 12, 20, 40):
        per_n[n] = float(np.mean(np.sqrt(full * (1.0 - full) / n)))

    by_ante = {}
    for ante in sorted({r.ante for r in records}):
        subset = [r for r in records if r.ante == ante]
        if len(subset) < DEFAULT_MIN_ANTE_BUILDS:
            continue
        a = np.asarray([r.playout(HALF) for r in subset], dtype=float)
        b = np.asarray([r.target for r in subset], dtype=float)
        d = a - b
        by_ante[ante] = {
            "n_builds": len(subset),
            "mean_abs_diff": float(np.abs(d).mean()),
            "corr": corr(a, b),
            "se_20": float(np.sqrt((d**2).mean() / 2.0)),
        }

    return {
        "overlapping": {
            "mean_abs_diff": float(np.abs(second - full).mean()),
            "rmse": float(np.sqrt(((second - full) ** 2).mean())),
            "corr": corr(second, full),
        },
        "disjoint": {
            "mean_abs_diff": float(np.abs(delta).mean()),
            "rmse": float(np.sqrt((delta**2).mean())),
            "corr": corr(first, second),
            "within_005": float((np.abs(delta) <= 0.05).mean()),
            "within_010": float((np.abs(delta) <= 0.10).mean()),
            "within_020": float((np.abs(delta) <= 0.20).mean()),
            "implied_se_20": float(np.sqrt((delta**2).mean() / 2.0)),
        },
        "mean_se_by_n": per_n,
        "by_ante": by_ante,
    }


def retracted_decomposition(records: Sequence[Record]) -> dict[str, Any]:
    """RETRACTED. Preserved so the mistake stays reproducible, not folklore.

    The argument: by the law of total variance over the deal ``s``, with
    ``q(s) = P(clear|s)``,

        Var(1{clear}) = E_s[q(1-q)] + Var_s(q)

    A head returning ``q`` would kill the first term outright, so its per-sample
    variance would be ``Var_s(q)`` against play-out's ``p(1-p)`` -- which
    measured as an ~84x efficiency in the head's favour.

    The flaw: this reads the head's across-deal variance AS ``Var_s(q)``, i.e.
    it assumes the head is a correct conditional mean.  ``head_deal_sensitivity``
    shows it is not (5.6% of variance within-build, correlating 0.105 with the
    outcome), so the small variance measures insensitivity, and low variance
    around the wrong value is not efficiency.  The model-free ``split_half``
    comparison is the one to trust.
    """
    efficiencies = []
    for record in records:
        p = float(record.c.mean())
        mean_v = float(record.v.mean())
        if not 0.0 < p < 1.0 or mean_v <= 0.05:
            continue
        cv_squared = float(record.v.var()) / (mean_v * mean_v)
        var_bernoulli = p * (1.0 - p)
        var_q = min(p * p * cv_squared, var_bernoulli)
        if var_q > 1e-9:
            efficiencies.append(var_bernoulli / var_q)
    values = np.asarray(efficiencies, dtype=float)
    return {
        "n_usable": len(values),
        "median_efficiency": float(np.median(values)) if len(values) else float("nan"),
        "p25": float(np.percentile(values, 25)) if len(values) else float("nan"),
        "p75": float(np.percentile(values, 75)) if len(values) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float, places: int = 3) -> str:
    return "  nan" if not np.isfinite(value) else f"{value:.{places}f}"


def report(
    records: Sequence[Record],
    v_curve: VCurve | None,
    sections: Sequence[str],
    ns: Sequence[int],
    *,
    min_cell: int = DEFAULT_MIN_CELL,
) -> None:
    print(f"corpus: {len(records)} builds x {REDEALS} redeals\n")

    if "head-deal-sensitivity" in sections:
        result = head_deal_sensitivity(records)
        print("HEAD DEAL SENSITIVITY -- does the head read the deal, or the build?")
        print(f"  within-build corr(head, cleared): {_fmt(result['within_build_corr'])}")
        for ante, row in result["by_ante"].items():
            if row["n_redeals"] >= 200:
                print(
                    f"    ante {ante}: {_fmt(row['corr'])}"
                    f"  (builds={row['n_builds']}, redeals={row['n_redeals']})"
                )
        print(
            f"  within-build share of head variance: "
            f"{100 * result['within_share']:.1f}%"
            f"  (within {result['within_build_head_var']:.4f}"
            f" / total {result['total_head_var']:.4f})\n"
        )

    if "split-half" in sections:
        result = split_half(records, ns)
        print("SPLIT-HALF, POOLED (corr with held-out 20-redeal clear rate)")
        print("     N    play-out       head")
        for row in result["rows"]:
            print(f"  {row['n']:>4}       {_fmt(row['playout'])}      {_fmt(row['head'])}")
        print(f"  ceiling(20): {_fmt(result['ceiling_20'])}")
        gain = combined_gain(records)
        print(
            f"  play-out alone {_fmt(gain['playout_only'])}"
            f"  ->  + head {_fmt(gain['playout_plus_head'])}"
            "   (does the head add anything?)\n"
        )

    if "within-ante" in sections:
        # The head column is quoted at the largest N on offer, which is where it
        # looks BEST -- and it still saturates far below play-out.
        head_n = max(ns)
        print(f"WITHIN EACH ANTE  (head column at N={head_n})")
        header = "  ante  builds" + "".join(f"  play{n:<2}" for n in ns)
        print(header + "     HEAD  ceiling")
        for row in within_ante(records, ns):
            cells = "".join(f"  {_fmt(row['playout'][n])}" for n in ns)
            print(
                f"  {row['ante']:>4}  {row['n_builds']:>6}{cells}"
                f"    {_fmt(row['head'][head_n])}    {_fmt(row['ceiling_20'])}"
            )
        print()

    if "within-cell" in sections:
        result = within_cell(records, ns, min_cell=min_cell)
        print(
            f"WITHIN (ANTE, BOSS) CELLS -- {result['n_cells']} cells >= {min_cell} builds, "
            f"{result['n_builds']} builds"
        )
        print("     N    play-out       head")
        for n in ns:
            print(f"  {n:>4}       {_fmt(result['playout'][n])}      {_fmt(result['head'][n])}")
        print()

    if "money-correction" in sections:
        if v_curve is None:
            print("MONEY CORRECTION -- skipped (no --v-curve artifact given)\n")
        else:
            result = money_correction(records, v_curve, min_cell=min_cell)
            print("MONEY CORRECTION -- un-multiplying the head's v_curve term")
            print(f"  head raw          Pearson {_fmt(result['raw_pearson'])}"
                  f"   Spearman {_fmt(result['raw_spearman'])}")
            for variant in result["variants"]:
                print(
                    f"  / (1 + v_curve)   Pearson {_fmt(variant['pearson'])}"
                    f"   Spearman {_fmt(variant['spearman'])}"
                    f"   mean m {variant['mean_money_term']:.3f}"
                    f"   [{variant['label']}]"
                )
            print(
                f"  within cells:  raw {_fmt(result['within_cell_raw'])}"
                f"  ->  corrected {_fmt(result['within_cell_corrected'])}"
            )
            print(
                f"  ante-4 cells:  raw {_fmt(result['ante4_cell_raw'])}"
                f"  ->  corrected {_fmt(result['ante4_cell_corrected'])}"
            )
            print(
                f"  residual (antes 1-4) vs dollars {_fmt(result['residual_vs_dollars'])}"
                f", vs ante {_fmt(result['residual_vs_ante'])}"
                "   <- ante-shaped, not dollar-shaped"
            )
            for label, value in gradient_boosted_bound(records).items():
                print(f"  GBM CV {label:<32} {_fmt(value)}")
            print()

    if "sampling-noise" in sections:
        result = sampling_noise(records)
        print("SAMPLING NOISE -- how many play-outs?")
        over = result["overlapping"]
        dis = result["disjoint"]
        print(
            f"  half vs 40-sample mean (OVERLAPPING, flattering): "
            f"mean|diff| {over['mean_abs_diff']:.4f}, corr {_fmt(over['corr'], 4)}"
        )
        print(
            f"  half A vs half B (DISJOINT, honest):              "
            f"mean|diff| {dis['mean_abs_diff']:.4f}, corr {_fmt(dis['corr'], 4)}"
        )
        print(
            f"    within 0.05 {100 * dis['within_005']:.1f}%   "
            f"within 0.10 {100 * dis['within_010']:.1f}%   "
            f"within 0.20 {100 * dis['within_020']:.1f}%"
        )
        print(f"    implied SE of one 20-sample estimate: {dis['implied_se_20']:.4f}")
        pieces = "   ".join(f"N={n}: {se:.4f}" for n, se in result["mean_se_by_n"].items())
        print(f"  mean SE by N:  {pieces}")
        print(f"  (build-driven signal std to resolve: ~{SIGNAL_STD})\n")

    if "retracted-decomposition" in sections:
        result = retracted_decomposition(records)
        print("=" * 72)
        print("RETRACTED SECTION -- reproduced only to document the error.")
        print("Assumes the head is a correct conditional mean; it is not (see")
        print("HEAD DEAL SENSITIVITY). Low variance around the wrong value is")
        print("not efficiency. Trust the model-free SPLIT-HALF section instead.")
        print("=" * 72)
        print(
            f"  claimed efficiency (play-out var / head var), median "
            f"{result['median_efficiency']:.2f}x"
            f"   p25 {result['p25']:.2f}x   p75 {result['p75']:.2f}x"
            f"   (n={result['n_usable']})\n"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/boss_clear_probe.jsonl"))
    parser.add_argument(
        "--v-curve",
        type=Path,
        default=Path("data/v_curve_s1.json"),
        help="artifact the partner trained with; omit to skip the money correction",
    )
    parser.add_argument("--seed", type=int, default=0, help="redeal permutation seed")
    parser.add_argument("--min-cell", type=int, default=DEFAULT_MIN_CELL)
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=list(DEFAULT_NS),
        help="estimator sample counts to report",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=list(SECTIONS),
        choices=list(SECTIONS),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_corpus(args.corpus, seed=args.seed)
    if not records:
        raise SystemExit(f"no usable records in {args.corpus}")
    v_curve = load_v_curve(args.v_curve) if args.v_curve and args.v_curve.exists() else None
    report(records, v_curve, args.sections, args.ns, min_cell=args.min_cell)


if __name__ == "__main__":
    main()
