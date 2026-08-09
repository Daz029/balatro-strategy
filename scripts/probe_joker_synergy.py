"""Option A of the embedding probe: do descriptor twins share learned synergy?

Answers the open question in ``docs/embedding-analysis.md`` §5 — does the shop
policy COMPOSE to joker pairings it rarely saw, or has it memorized the pairs
its own rollouts happened to cover? The ablation there cannot answer it: it
removes identity wholesale and measures an out-of-distribution state.

THE MEASURED OBJECT — the discrete second difference, not ``ΔV``
----------------------------------------------------------------
``V(O ∪ {c}) − V(O)`` is a MAIN EFFECT (how good joker c is alone) and says
nothing about pairings. Synergy is the non-additive part, so we take the second
difference. This script uses the REPLACEMENT form::

    synergy(a, b) = V(O∪{a,b}) − V(O∪{a,f}) − V(O∪{f,b}) + V(O∪{f,f})

against a fixed filler joker ``f``, rather than the doc's sketched
addition form (which compares 2-joker states against 1- and 0-joker ones).

WHY REPLACEMENT: ``masked_pool`` (``hand_policy.py:46``) is masked mean ⊕
masked max, so the mean's denominator changes when the joker COUNT changes.
The addition form therefore carries a purely structural offset — nonzero even
if the injected joker were a duplicate of one already owned — which the doc
proposed to measure as a floor and subtract. Injecting exactly two jokers in
every arm holds cardinality fixed at 2 and the artifact never arises. This is
still an exact second difference, just with ``f`` as the baseline level instead
of absence, and ``f`` enters all four terms symmetrically so it cancels.

THE ONE HARD RULE, inherited from ``extract_v_curve.py``: counterfactuals edit
ENGINE STATE and re-encode; they never edit an obs vector. A joker fans out
into far more derived obs features than dollars does — the joker's own 15 row
features, ``get_hand_eval_flags`` in the global context (inject Four Fingers by
obs-edit and the state claims Four Fingers while every hand-structure feature
disagrees), the hand-analysis block, and the per-shop-row buyability flag
``len(jokers) < joker_slots``. Writing a ``center_key_id`` into ``joker_ids``
would leave every one of those stale, and the critic would happily score the
resulting contradiction.

Three further construction rules, each a bug this project has already paid for:

* Injected jokers get their ``add_to_deck`` passives applied (``card.py:746``).
  Skipping that is the pre-regen B1 bug verbatim — ``HandPlayAdapter.reset``
  injected with a bare ``create_joker`` and never applied acquisition passives,
  so every demo that sampled Juggler/Turtle Bean carried a hand size the engine
  would never have dealt.
* Injected jokers occupy the SAME slot indices across all four arms.
  ``masked_pool`` is permutation-invariant, but ``encode_joker`` feature 9 is
  ``position / 20`` (``observation.py:443``), so the rows are not.
* Base states are filtered to those with room for two more jokers, so
  ``_check_joker_overfill`` (``shop_obs.py:190``) never fires and no arm is
  silently truncated.

THE TWIN TEST, AND ITS REFERENCE ARM
------------------------------------
51% of the joker pool (77 of 150) shares a BIT-IDENTICAL 24-d descriptor row
with at least one other joker — 30 exact-duplicate groups, e.g.
``(four_fingers, shortcut, midas_mask, smeared)``. So the twin test needs no
similarity threshold: for a twin pair the descriptor channel is provably
identical, and any difference in learned synergy is attributable to the
non-descriptor identity channels (the random embedding, plus the ordinal
``center_key_id`` in row feature 0).

``docs/embedding-analysis.md`` §6.2 phrases the verdict as "close ⟹ transfer,
far apart ⟹ memorization", but close and far need a scale. So every run also
computes the SAME gap statistic over random NON-twin pairs:

* twin gaps ≈ non-twin gaps ⟹ twins are no more alike than arbitrary jokers,
  i.e. the descriptor channel transferred nothing and identity-memorization
  dominates.
* twin gaps ≪ non-twin gaps ⟹ knowledge moved through the descriptor channel;
  the random code is not blocking generalization.

CAVEATS THE OUTPUT DOES NOT ENCODE
----------------------------------
* Φ shaping was live for ``s2_a4`` (``shop/phi_beta`` 0.0999 → 0.00006 in the
  run's TB log), so the critic learned ``V_true − β·Φ`` under γ=1. At the final
  β the residual contributes ~1e-4 to any second difference. It does not cancel
  in principle (Φ is state-dependent and the four arms are different states),
  it is merely numerically negligible — and since the Φ checkpoint still
  exists, the term is directly computable if a result ever turns on it.
* The six pool-gated jokers of §8 (cavendish, glass, lucky_cat, steel_joker,
  stone, ticket) never appeared in ANY observation, so their rows are random
  codes behind an untrained decoder path. They are excluded from every role
  here — as injections, partners and fillers — because their values are
  meaningless rather than merely uncertain.
* Constructed states are off-distribution by construction. A flat synergy on a
  never-built pair is ambiguous between failed generalization and an unvisited
  region of state space. That ambiguity is partly the finding.

Usage::

    uv run python scripts/probe_joker_synergy.py --n-states 40
    uv run python scripts/probe_joker_synergy.py --n-states 8 --max-groups 3  # smoke
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jackdaw.agents.joker_descriptors import DESCRIPTOR_MATRIX  # noqa: E402
from jackdaw.engine.card_factory import create_joker  # noqa: E402
from jackdaw.engine.data.prototypes import CENTER_POOLS  # noqa: E402
from jackdaw.env.observation import center_key_id  # noqa: E402
from jackdaw.env.shop_obs import build_shop_observation  # noqa: E402

DEFAULT_CHECKPOINT = "runs/shop_ppo/s2_a4/best_model/best_model.zip"
DEFAULT_RESERVOIR = "runs/shop_ppo/s2_a4/reservoir.pkl"
DEFAULT_OUT = "data/joker_synergy_probe.json"

# Never received a single gradient (embedding-analysis.md §8): conditionally
# pooled by pools.py::_filter_joker, so they appeared in no observation ever.
# Random code + untrained decoder path == meaningless value, so they are barred
# from every role in this probe.
POOL_GATED = frozenset(
    {"j_cavendish", "j_glass", "j_lucky_cat", "j_steel_joker", "j_stone", "j_ticket"}
)

# The filler defines the baseline level of the second difference. Any fixed
# choice cancels; a plain, common, stateless joker keeps the baseline arms as
# ordinary as possible.
DEFAULT_FILLER = "j_joker"

INJECT_COUNT = 2  # every arm injects exactly two jokers -- see module docstring


# ---------------------------------------------------------------------------
# Vocabulary / descriptor twins
# ---------------------------------------------------------------------------


def joker_keys() -> list[str]:
    """Every joker center key, pool-gated ones excluded."""
    return [k for k in CENTER_POOLS.get("Joker", []) if k not in POOL_GATED]


def descriptor_groups(keys: list[str]) -> dict[bytes, list[str]]:
    """Group keys by BIT-IDENTICAL descriptor row.

    Exact rather than near equality: the duplicate structure is strong enough
    (30 groups covering half the pool) that a similarity threshold would only
    add a knob without adding cells.
    """
    groups: dict[bytes, list[str]] = defaultdict(list)
    for k in keys:
        groups[DESCRIPTOR_MATRIX[center_key_id(k)].tobytes()].append(k)
    return dict(groups)


def twin_pairs(
    keys: list[str], rng: np.random.Generator, max_groups: int | None
) -> list[tuple[str, str]]:
    """One representative pair per exact-descriptor-duplicate group."""
    groups = [sorted(v) for v in descriptor_groups(keys).values() if len(v) > 1]
    groups.sort()  # deterministic order before any sampling
    if max_groups is not None:
        groups = groups[:max_groups]
    pairs = []
    for g in groups:
        i, j = rng.choice(len(g), size=2, replace=False)
        pairs.append((g[int(i)], g[int(j)]))
    return pairs


def nontwin_pairs(
    keys: list[str], n: int, rng: np.random.Generator
) -> list[tuple[str, str]]:
    """Random pairs whose descriptors DIFFER — the reference scale."""
    desc = {k: DESCRIPTOR_MATRIX[center_key_id(k)].tobytes() for k in keys}
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    guard = 0
    while len(out) < n and guard < 100 * n:
        guard += 1
        i, j = rng.choice(len(keys), size=2, replace=False)
        a, b = keys[int(i)], keys[int(j)]
        if desc[a] == desc[b]:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


# ---------------------------------------------------------------------------
# Base states
# ---------------------------------------------------------------------------


@dataclass
class BaseState:
    """A reservoir shop snapshot with room for two injected jokers."""

    stratum: tuple[int, bool]
    index: int
    gs_bytes: bytes
    ante: int
    n_jokers: int
    joker_slots: int
    dollars: int


def _unwrap(blob: bytes) -> tuple[dict[str, Any], Any]:
    """ShopGymEnv._snapshot format: {"engine": adapter bytes, "pending": ...}."""
    payload = pickle.loads(blob)
    return pickle.loads(payload["engine"]), payload["pending"]


def load_base_states(
    reservoir_path: Path, n: int, rng: np.random.Generator
) -> list[BaseState]:
    """Sample SHOP-phase snapshots with >=2 free joker slots.

    Pack-pending strata are excluded: their pending-target state makes the
    obs a different animal, and owned-set synergy is a shop-decision question.
    """
    res = pickle.load(reservoir_path.open("rb"))
    strata: dict[tuple[int, bool], list[bytes]] = res["strata"]

    candidates: list[tuple[tuple[int, bool], int]] = [
        (stratum, i)
        for stratum, blobs in sorted(strata.items())
        if not stratum[1]  # pending==False only
        for i in range(len(blobs))
    ]
    order = rng.permutation(len(candidates))

    out: list[BaseState] = []
    for pos in order:
        stratum, idx = candidates[int(pos)]
        gs, pending = _unwrap(strata[stratum][idx])
        if pending is not None:
            continue
        phase = gs.get("phase")
        if getattr(phase, "value", phase) != "shop":
            continue
        jokers = gs.get("jokers", [])
        slots = gs.get("joker_slots", 5)
        if len(jokers) + INJECT_COUNT > slots:
            continue  # no room; keeps _check_joker_overfill from ever firing
        out.append(
            BaseState(
                stratum=stratum,
                index=idx,
                gs_bytes=pickle.dumps(gs, protocol=pickle.HIGHEST_PROTOCOL),
                ante=gs.get("round_resets", {}).get("ante", 0),
                n_jokers=len(jokers),
                joker_slots=slots,
                dollars=gs.get("dollars", 0),
            )
        )
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Value queries
# ---------------------------------------------------------------------------


class Critic:
    """Frozen MaskablePPO value head. ``predict_values`` only — the action
    head is never touched, same contract as ``extract_v_curve.MaskablePPOCritic``."""

    def __init__(self, checkpoint: Path, device: str = "cpu") -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(checkpoint), device=device)
        self._model.policy.set_training_mode(False)
        ctx = int(self._model.observation_space["shop_context"].shape[0])
        rows = int(self._model.observation_space["jokers"].shape[0])
        # Recover the encoder schema from the checkpoint rather than trusting
        # a flag: a mismatch here is a silent garbage-in measurement.
        self.s1_schema = rows > 8
        self.ctx_width = ctx
        self.joker_rows = rows

    def values(self, obs_list: list[dict[str, np.ndarray]]) -> np.ndarray:
        import torch
        from stable_baselines3.common.utils import obs_as_tensor

        batch = {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        with torch.no_grad():
            v = self._model.policy.predict_values(
                obs_as_tensor(batch, self._model.policy.device)
            )
        return v.squeeze(-1).cpu().numpy().astype(np.float64)


def build_arm_obs(base: BaseState, keys: tuple[str, str], s1_schema: bool) -> dict[str, np.ndarray]:
    """Restore, inject two jokers WITH passives at fixed slots, re-encode."""
    gs = pickle.loads(base.gs_bytes)
    for key in keys:
        j = create_joker(key)
        gs["jokers"].append(j)
        j.add_to_deck(gs)  # acquisition passives -- see module docstring
    return build_shop_observation(gs, None, s1_schema=s1_schema)


@dataclass
class ValueCache:
    """Memoizes V per (base index, injected pair). Arms are shared heavily:
    V(f,f) is common to every cell, V(x,f) to every partner, V(f,b) to every
    injected joker."""

    critic: Critic
    s1_schema: bool
    batch_size: int = 256
    _cache: dict[tuple[int, str, str], float] = field(default_factory=dict)
    n_forward: int = 0

    def request(self, bases: list[BaseState], arms: list[tuple[int, str, str]]) -> None:
        todo = [a for a in dict.fromkeys(arms) if a not in self._cache]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start : start + self.batch_size]
            obs = [build_arm_obs(bases[bi], (x, y), self.s1_schema) for bi, x, y in chunk]
            vals = self.critic.values(obs)
            self.n_forward += len(chunk)
            for arm, v in zip(chunk, vals, strict=True):
                self._cache[arm] = float(v)

    def v(self, base_idx: int, x: str, y: str) -> float:
        return self._cache[(base_idx, x, y)]


def synergy(cache: ValueCache, base_idx: int, a: str, b: str, filler: str) -> float:
    """V(a,b) − V(a,f) − V(f,b) + V(f,f). Cardinality is 2 in every term."""
    return (
        cache.v(base_idx, a, b)
        - cache.v(base_idx, a, filler)
        - cache.v(base_idx, filler, b)
        + cache.v(base_idx, filler, filler)
    )


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def run_probe(
    bases: list[BaseState],
    pairs: list[tuple[str, str]],
    partners: list[str],
    filler: str,
    cache: ValueCache,
) -> list[dict[str, Any]]:
    """For each (a, a') pair and partner b: |synergy(a,b) − synergy(a',b)|."""
    arms: list[tuple[int, str, str]] = []
    for bi in range(len(bases)):
        arms.append((bi, filler, filler))
        for b in partners:
            arms.append((bi, filler, b))
        for a, ap in pairs:
            for x in (a, ap):
                arms.append((bi, x, filler))
                for b in partners:
                    arms.append((bi, x, b))
    cache.request(bases, arms)

    rows: list[dict[str, Any]] = []
    for a, ap in pairs:
        for b in partners:
            syn_a, syn_ap = [], []
            for bi in range(len(bases)):
                syn_a.append(synergy(cache, bi, a, b, filler))
                syn_ap.append(synergy(cache, bi, ap, b, filler))
            sa, sap = np.array(syn_a), np.array(syn_ap)
            rows.append(
                {
                    "a": a,
                    "a_prime": ap,
                    "partner": b,
                    "n_states": len(bases),
                    "synergy_a_mean": float(sa.mean()),
                    "synergy_a_prime_mean": float(sap.mean()),
                    "gap_mean_abs": float(np.abs(sa - sap).mean()),
                    "gap_of_means_abs": float(abs(sa.mean() - sap.mean())),
                    "synergy_abs_mean": float((np.abs(sa).mean() + np.abs(sap).mean()) / 2),
                    "synergy_a_std": float(sa.std()),
                    "synergy_a_prime_std": float(sap.std()),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    gaps = np.array([r["gap_mean_abs"] for r in rows])
    gom = np.array([r["gap_of_means_abs"] for r in rows])
    # Mean of |synergy| over the SAME (cell, state) units the gap is averaged
    # over, so the two numbers are on one scale. (abs-of-mean would not be.)
    mags = np.array([r["synergy_abs_mean"] for r in rows])
    return {
        "arm": label,
        "n_cells": len(rows),
        "gap_mean_abs__mean": float(gaps.mean()) if len(gaps) else 0.0,
        "gap_mean_abs__median": float(np.median(gaps)) if len(gaps) else 0.0,
        "gap_mean_abs__p90": float(np.percentile(gaps, 90)) if len(gaps) else 0.0,
        "gap_of_means_abs__mean": float(gom.mean()) if len(gom) else 0.0,
        "synergy_magnitude__mean": float(mags.mean()) if len(mags) else 0.0,
    }


def split_half_reliability(
    cache: ValueCache, bases: list[BaseState], injected: list[str], partners: list[str], filler: str
) -> dict[str, Any]:
    """POSITIVE CONTROL — is synergy a reproducible function of (a, b) at all?

    Without this the headline ratio is uninterpretable: if the critic's second
    differences were pure noise, the twin and non-twin arms would BOTH be noise
    and the ratio would sit at ~1.0 for reasons that have nothing to do with
    memorization. Splits the base states into two disjoint halves, computes each
    cell's mean synergy in each half, and correlates the two vectors across
    cells. High r ⟹ synergy is a real per-pair quantity that generalizes across
    states ⟹ the twin comparison means something.
    """
    even = [i for i in range(len(bases)) if i % 2 == 0]
    odd = [i for i in range(len(bases)) if i % 2 == 1]
    if len(even) < 2 or len(odd) < 2:
        return {"n_cells": 0, "pearson_r": None, "note": "too few base states to split"}

    lhs, rhs = [], []
    for a in injected:
        for b in partners:
            lhs.append(np.mean([synergy(cache, i, a, b, filler) for i in even]))
            rhs.append(np.mean([synergy(cache, i, a, b, filler) for i in odd]))
    lhs_arr, rhs_arr = np.array(lhs), np.array(rhs)
    if lhs_arr.std() == 0 or rhs_arr.std() == 0:
        r = None
    else:
        r = float(np.corrcoef(lhs_arr, rhs_arr)[0, 1])
    return {
        "n_cells": len(lhs),
        "n_states_per_half": [len(even), len(odd)],
        "pearson_r": r,
        "half_a_std": float(lhs_arr.std()),
        "half_b_std": float(rhs_arr.std()),
    }


def bootstrap_ratio(
    twin_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]], n_boot: int, seed: int
) -> dict[str, float]:
    """Percentile CI on the twin/non-twin gap ratio, resampling CELLS."""
    rng = np.random.default_rng(seed)
    t = np.array([r["gap_mean_abs"] for r in twin_rows])
    n = np.array([r["gap_mean_abs"] for r in ref_rows])
    if not len(t) or not len(n):
        return {}
    ratios = []
    for _ in range(n_boot):
        tb = t[rng.integers(0, len(t), len(t))].mean()
        nb = n[rng.integers(0, len(n), len(n))].mean()
        if nb:
            ratios.append(tb / nb)
    arr = np.array(ratios)
    return {
        "ratio_median": float(np.median(arr)),
        "ratio_ci_lo": float(np.percentile(arr, 2.5)),
        "ratio_ci_hi": float(np.percentile(arr, 97.5)),
        "n_boot": len(arr),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    p.add_argument("--reservoir", type=Path, default=Path(DEFAULT_RESERVOIR))
    p.add_argument("--output", type=Path, default=Path(DEFAULT_OUT))
    p.add_argument("--n-states", type=int, default=40, help="base shop states")
    p.add_argument("--n-partners", type=int, default=12, help="partner jokers b")
    p.add_argument("--max-groups", type=int, default=None, help="cap twin groups (smoke)")
    p.add_argument("--filler", type=str, default=DEFAULT_FILLER)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for the ratio CI")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()

    keys = joker_keys()
    pairs = twin_pairs(keys, rng, args.max_groups)
    partners = [keys[int(i)] for i in rng.choice(len(keys), size=args.n_partners, replace=False)]
    ref_pairs = nontwin_pairs(keys, len(pairs), rng)

    print(f"pool: {len(keys)} jokers ({len(POOL_GATED)} pool-gated excluded)")
    print(f"twin pairs: {len(pairs)} | reference non-twin pairs: {len(ref_pairs)}")
    print(f"partners: {partners}")
    print(f"filler: {args.filler}")

    critic = Critic(args.checkpoint, device=args.device)
    print(
        f"critic: s1_schema={critic.s1_schema} ctx={critic.ctx_width} "
        f"joker_rows={critic.joker_rows}"
    )

    bases = load_base_states(args.reservoir, args.n_states, rng)
    if not bases:
        print("ERROR: no eligible base states (need shop phase + 2 free joker slots)")
        return 1
    print(f"base states: {len(bases)} | antes {sorted({b.ante for b in bases})}")

    cache = ValueCache(critic=critic, s1_schema=critic.s1_schema)
    twin_rows = run_probe(bases, pairs, partners, args.filler, cache)
    ref_rows = run_probe(bases, ref_pairs, partners, args.filler, cache)

    twin_sum = summarize(twin_rows, "descriptor_twins")
    ref_sum = summarize(ref_rows, "random_nontwins")
    ratio = (
        twin_sum["gap_mean_abs__mean"] / ref_sum["gap_mean_abs__mean"]
        if ref_sum["gap_mean_abs__mean"]
        else float("nan")
    )
    boot = bootstrap_ratio(twin_rows, ref_rows, args.n_boot, args.seed)

    injected = sorted({k for pr in pairs + ref_pairs for k in pr})
    reliability = split_half_reliability(cache, bases, injected, partners, args.filler)

    print(f"\n{'arm':<20} {'cells':>6} {'gap mean':>10} {'gap med':>10} {'|synergy|':>10}")
    for s in (twin_sum, ref_sum):
        print(
            f"{s['arm']:<20} {s['n_cells']:>6} {s['gap_mean_abs__mean']:>10.5f} "
            f"{s['gap_mean_abs__median']:>10.5f} {s['synergy_magnitude__mean']:>10.5f}"
        )

    r = reliability.get("pearson_r")
    print(f"\nPOSITIVE CONTROL  split-half reliability of synergy(a,b): "
          f"r={r if r is None else f'{r:.3f}'} over {reliability['n_cells']} cells")
    if r is None or r < 0.3:
        print("  LOW -- synergy estimates do not reproduce across base states.")
        print("  The ratio below is NOT interpretable: both arms would be noise.")
    else:
        print("  OK -- synergy is a reproducible function of the pair.")

    print(f"\ntwin/non-twin gap ratio: {ratio:.3f}", end="")
    if boot:
        print(f"  (95% CI {boot['ratio_ci_lo']:.3f}-{boot['ratio_ci_hi']:.3f})")
    else:
        print()
    print("  ~1.0 => twins no more alike than random pairs => memorization dominates")
    print("  <<1.0 => knowledge transferred through the descriptor channel")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint),
        "reservoir": str(args.reservoir),
        "seed": args.seed,
        "filler": args.filler,
        "s1_schema": critic.s1_schema,
        "n_base_states": len(bases),
        "partners": partners,
        "base_states": [
            {"stratum": list(b.stratum), "index": b.index, "ante": b.ante,
             "n_jokers": b.n_jokers, "joker_slots": b.joker_slots, "dollars": b.dollars}
            for b in bases
        ],
        "summary": {
            "twins": twin_sum,
            "nontwins": ref_sum,
            "twin_over_nontwin_ratio": ratio,
            "ratio_bootstrap": boot,
            "split_half_reliability": reliability,
        },
        "twin_cells": twin_rows,
        "nontwin_cells": ref_rows,
        "n_critic_forwards": cache.n_forward,
        "elapsed_s": time.perf_counter() - t0,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"\nwrote {args.output} ({cache.n_forward} critic forwards, "
        f"{time.perf_counter() - t0:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
