"""Measure whether terminal-boss clear probability separates on-policy builds.

Each full run uses the deterministic s0 shop policy and current hand partner.
The last boss reached is snapshotted immediately before ``SelectBlind``.  The
snapshot is then replayed with only the named per-round deck-shuffle RNG stream
reseeded, producing paired critic values and sampled clear outcomes.
"""

from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
import enum
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _path in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from jackdaw.agents.hand_checkpoint_policy import (  # noqa: E402
    HandCheckpointPolicy,
    _zip_policy_kind,
)
from jackdaw.engine.actions import CashOut, GamePhase, SelectBlind  # noqa: E402
from jackdaw.engine.rng import pseudohash  # noqa: E402
from jackdaw.env.hand_play_adapter import HandPlayAdapter  # noqa: E402
from jackdaw.env.hand_play_gym import HandPlayGymEnv  # noqa: E402
from jackdaw.env.shop_gym import ShopGymEnv  # noqa: E402
from jackdaw.env.shop_run_adapter import (  # noqa: E402
    DECISION_PHASES,
    ShopRunAdapter,
    ShopRunConfig,
)

BOSS_PROBE_PREFIX = "BOSSPROBE"
BOSS_PROBE_WIN_ANTE = 8
DEFAULT_REDEALS = 40
_MAX_AUTO_STEPS = 64
_MAX_SHOP_STEPS = 512


class ProbePartner(Protocol):
    """Observation-level partner interface used by the redeal loop."""

    obs_version: int
    action_version: int

    def predict_value(self, obs: dict[str, np.ndarray]) -> float: ...

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray | None) -> int | np.ndarray: ...


@dataclasses.dataclass(frozen=True)
class BossCapture:
    """All process state needed to replay one pre-deal boss entry exactly."""

    predeal_blob: bytes
    sort_id_counter: int


class CheckpointProbePartner:
    """Load one PPO checkpoint for both engine actions and critic queries."""

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        path = Path(checkpoint_path)
        if path.suffix.lower() != ".zip":
            raise ValueError("--hand-policy must be a PPO .zip checkpoint")

        policy_kind = _zip_policy_kind(path)
        self._checkpoint = HandCheckpointPolicy(path, device=device)
        self.obs_version = 2 if policy_kind == "pointer" else 1
        self.action_version = 2 if policy_kind == "pointer" else 1
        expected_kind = "pointer_ppo" if policy_kind == "pointer" else "ppo"
        if self._checkpoint._kind != expected_kind:
            raise ValueError(
                f"checkpoint dispatch mismatch: archive={policy_kind}, "
                f"loaded={self._checkpoint._kind}"
            )

    def __call__(self, game_state: dict[str, Any]):
        """Engine-action surface consumed by ``ShopGymEnv`` full runs."""
        return self._checkpoint(game_state)

    def _check_observation(self, obs: dict[str, np.ndarray]) -> None:
        spaces = self._checkpoint._model.observation_space.spaces
        if set(obs) != set(spaces):
            missing = sorted(set(spaces) - set(obs))
            extra = sorted(set(obs) - set(spaces))
            raise ValueError(
                f"hand checkpoint observation schema mismatch: missing={missing}, extra={extra}"
            )
        wrong_shapes = {
            key: (tuple(obs[key].shape), tuple(space.shape))
            for key, space in spaces.items()
            if tuple(obs[key].shape) != tuple(space.shape)
        }
        if wrong_shapes:
            raise ValueError(f"hand checkpoint observation shape mismatch: {wrong_shapes}")

    def predict_value(self, obs: dict[str, np.ndarray]) -> float:
        self._check_observation(obs)
        policy = self._checkpoint._model.policy
        obs_tensor, _ = policy.obs_to_tensor(obs)
        with self._checkpoint._torch.no_grad():
            value = policy.predict_values(obs_tensor)
        return float(value.squeeze().item())

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray | None) -> int | np.ndarray:
        self._check_observation(obs)
        if self.action_version == 2:
            return self._checkpoint._infer_pointer(obs)
        if mask is None:
            raise ValueError("v1 hand checkpoint requires an action mask")
        return self._checkpoint._infer(obs, mask)


class BossCapturingShopRunAdapter(ShopRunAdapter):
    """Shop adapter variant that retains the latest pre-boss-select blob.

    This script-local override is intentionally the only interception point:
    the engine and environment remain unchanged, and every other auto-resolve
    branch is the same as ``ShopRunAdapter._advance``.
    """

    def __init__(self, hand_policy: Any, config: ShopRunConfig) -> None:
        super().__init__(hand_policy, config)
        self.last_boss_capture: BossCapture | None = None

    def reset(self, *args: Any, **kwargs: Any):
        self.last_boss_capture = None
        return super().reset(*args, **kwargs)

    def _advance(self) -> None:
        from jackdaw.engine.game import step as engine_step

        for _ in range(_MAX_AUTO_STEPS):
            if self.done:
                return
            phase = self._gs.get("phase")
            if phase in DECISION_PHASES:
                return
            if phase == GamePhase.BLIND_SELECT:
                on_deck = self._gs.get("blind_on_deck", "Small")
                if self._config.s1_schema and on_deck in ("Small", "Big"):
                    return
                if on_deck == "Boss":
                    from jackdaw.engine import card as card_module

                    boss_key = self._gs.get("round_resets", {}).get("blind_choices", {}).get("Boss")
                    if not boss_key:
                        raise AssertionError("boss capture seam reached before boss assignment")
                    self.last_boss_capture = BossCapture(
                        predeal_blob=self.snapshot_state(),
                        sort_id_counter=card_module._sort_id_counter,
                    )
                engine_step(self._gs, SelectBlind())
            elif phase == GamePhase.SELECTING_HAND:
                pre_state = (
                    copy.deepcopy(self._gs) if self._hand_decision_observer is not None else None
                )
                action = self._hand_policy(self._gs)
                engine_step(self._gs, action)
                if self._hand_decision_observer is not None:
                    self._hand_decision_observer(pre_state, action, self._gs)
            elif phase == GamePhase.ROUND_EVAL:
                engine_step(self._gs, CashOut())
            else:
                raise RuntimeError(f"unexpected phase during auto-advance: {phase!r}")
        raise RuntimeError(f"auto-advance exceeded {_MAX_AUTO_STEPS} steps -- hand policy stuck?")


def reseed_deal_stream(gs: dict[str, Any], redeal_seed: str) -> str:
    """Replace only the current ante's named new-round shuffle stream."""
    if gs.get("phase") != GamePhase.BLIND_SELECT or gs.get("blind_on_deck") != "Boss":
        raise ValueError("deal reseed requires a pre-SelectBlind boss snapshot")
    ante = int(gs.get("round_resets", {}).get("ante", 1))
    stream = f"nr{ante}"
    rng = gs.get("rng")
    if rng is None:
        raise ValueError("boss snapshot has no engine RNG")
    rng.state[stream] = pseudohash(stream + redeal_seed)
    return stream


def _max_sort_id(gs: dict[str, Any]) -> int:
    maximum = 0
    for value in gs.values():
        if isinstance(value, list):
            maximum = max(
                maximum,
                max((int(getattr(item, "sort_id", 0)) for item in value), default=0),
            )
    return maximum


def _capture_parts(capture: bytes | BossCapture) -> tuple[bytes, int | None]:
    if isinstance(capture, BossCapture):
        return capture.predeal_blob, capture.sort_id_counter
    return capture, None


def prepare_redeal(capture: bytes | BossCapture, redeal_seed: str) -> bytes:
    """Restore a boss-entry blob, vary its deal stream, and select the boss."""
    predeal_blob, sort_id_counter = _capture_parts(capture)
    adapter = HandPlayAdapter()
    adapter.restore_state(predeal_blob)
    gs = adapter.raw_state
    rr = gs.get("round_resets", {})
    boss_key = rr.get("blind_choices", {}).get("Boss")
    if not boss_key:
        raise ValueError("predeal snapshot has no assigned boss")

    # New cards created by setting_blind read this process-global counter,
    # which is not inside the engine pickle. Full-run capture supplies its
    # exact value; standalone fixture blobs fall back to the largest live id.
    from jackdaw.engine import card as card_module

    card_module._sort_id_counter = _max_sort_id(gs) if sort_id_counter is None else sort_id_counter
    reseed_deal_stream(gs, redeal_seed)
    adapter.step(SelectBlind())
    opening = adapter.raw_state
    if opening.get("phase") != GamePhase.SELECTING_HAND:
        raise AssertionError("SelectBlind did not produce a SELECTING_HAND opening")
    if getattr(opening.get("blind"), "key", None) != boss_key:
        raise AssertionError("deal reseed changed the assigned boss")
    return adapter.snapshot_state()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if hasattr(value, "_hands"):
        return {
            str(getattr(hand_type, "value", hand_type)): _jsonable(state)
            for hand_type, state in value._hands.items()
        }
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot encode {type(value).__name__} in boss probe JSON")


def _serialize_card(card: Any) -> dict[str, Any]:
    if not dataclasses.is_dataclass(card):
        raise TypeError(f"expected dataclass card, got {type(card).__name__}")
    return {field.name: _jsonable(getattr(card, field.name)) for field in dataclasses.fields(card)}


def _deck_signature(predeal: dict[str, Any]) -> dict[str, Any]:
    cards: list[Any] = []
    for area in ("deck", "hand", "discard_pile", "played_cards"):
        cards.extend(predeal.get(area, []))
    cards.sort(key=lambda card: int(getattr(card, "sort_id", 0)))
    return {
        "selected_back_key": predeal.get("selected_back_key"),
        "cards": [_serialize_card(card) for card in cards],
    }


def _opening_build(predeal: dict[str, Any], opening: dict[str, Any]) -> dict[str, Any]:
    current_round = opening.get("current_round", {})
    return {
        "jokers": [_serialize_card(joker) for joker in opening.get("jokers", [])],
        "hand_levels": _jsonable(opening.get("hand_levels")),
        "vouchers": _jsonable(opening.get("used_vouchers", {})),
        "deck_signature": _deck_signature(predeal),
        "dollars": int(opening.get("dollars", 0)),
        "hands_left": int(current_round.get("hands_left", 0)),
        "discards_left": int(current_round.get("discards_left", 0)),
    }


def _build_guard(build: dict[str, Any], boss_key: str) -> bytes:
    guarded = {
        "jokers": build["jokers"],
        "hand_levels": build["hand_levels"],
        "vouchers": build["vouchers"],
        "deck_signature": build["deck_signature"],
        "dollars": build["dollars"],
        "hands_left": build["hands_left"],
        "discards_left": build["discards_left"],
        "boss_key": boss_key,
    }
    return json.dumps(guarded, sort_keys=True, separators=(",", ":")).encode()


def _play_opening(opening_blob: bytes, partner: ProbePartner) -> tuple[float, int, dict[str, Any]]:
    env = HandPlayGymEnv(
        obs_version=partner.obs_version,
        action_version=partner.action_version,
    )
    obs, info = env.reset(options={"snapshot": opening_blob})
    opening = copy.deepcopy(env._adapter.raw_state)
    value = partner.predict_value(obs)

    while True:
        mask = info.get("action_mask")
        action = partner.act(obs, mask)
        obs, _, terminated, truncated, info = env.step(action)
        if truncated:
            raise RuntimeError("boss redeal truncated before round resolution")
        if terminated:
            return value, int(bool(info["balatro/cleared"])), opening


def build_probe_record(
    capture: bytes | BossCapture,
    *,
    run_seed: str,
    n_redeals: int,
    partner: ProbePartner,
    keep_blob: bool = False,
) -> dict[str, Any]:
    """Produce one complete JSON record for a captured terminal boss build."""
    if n_redeals < 1:
        raise ValueError("n_redeals must be positive")
    predeal_blob, sort_id_counter = _capture_parts(capture)
    predeal = pickle.loads(predeal_blob)
    if predeal.get("phase") != GamePhase.BLIND_SELECT:
        raise ValueError("terminal boss snapshot is not pre-SelectBlind")
    rr = predeal.get("round_resets", {})
    terminal_ante = int(rr.get("ante", 1))
    boss_key = str(rr.get("blind_choices", {}).get("Boss", ""))
    if not boss_key:
        raise ValueError("terminal boss snapshot has no boss key")

    redeals: list[dict[str, Any]] = []
    build: dict[str, Any] | None = None
    expected_guard: bytes | None = None
    for index in range(n_redeals):
        redeal_seed = f"{run_seed}_REDEAL_{index:04d}"
        opening_blob = prepare_redeal(capture, redeal_seed)
        value, cleared, opening = _play_opening(opening_blob, partner)
        candidate_build = _opening_build(predeal, opening)
        guard = _build_guard(candidate_build, boss_key)
        if expected_guard is None:
            expected_guard = guard
            build = candidate_build
        elif guard != expected_guard:
            assert build is not None
            changed = [
                key
                for key in build
                if json.dumps(build[key], sort_keys=True)
                != json.dumps(candidate_build[key], sort_keys=True)
            ]
            raise AssertionError(
                f"build drift across redeals for {run_seed} at redeal {index}; "
                f"only the opening deal may vary (changed={changed})"
            )
        redeals.append({"redeal_seed": redeal_seed, "v": value, "cleared": cleared})

    assert build is not None
    record: dict[str, Any] = {
        "run_seed": run_seed,
        "terminal_ante": terminal_ante,
        "boss_key": boss_key,
        "build": build,
        "redeals": redeals,
        "sampled_clear": math.fsum(row["cleared"] for row in redeals) / n_redeals,
        "critic_mean": math.fsum(row["v"] for row in redeals) / n_redeals,
        "n": n_redeals,
    }
    if keep_blob:
        record["predeal_blob_b64"] = base64.b64encode(predeal_blob).decode("ascii")
        record["predeal_sort_id_counter"] = sort_id_counter
    return record


def build_summary(
    records: list[dict[str, Any]], *, requested_runs: int | None = None
) -> dict[str, Any]:
    """Compute the deliberately thin post-run readout."""
    sampled = np.asarray([record["sampled_clear"] for record in records], dtype=float)
    critic = np.asarray([record["critic_mean"] for record in records], dtype=float)
    if len(sampled):
        quantiles = np.quantile(sampled, [0.25, 0.5, 0.75])
        distribution = {
            "n_builds": len(records),
            "min": float(sampled.min()),
            "p25": float(quantiles[0]),
            "median": float(quantiles[1]),
            "p75": float(quantiles[2]),
            "max": float(sampled.max()),
            "mean": float(sampled.mean()),
            "spread": float(sampled.max() - sampled.min()),
        }
        correlation = (
            round(float(np.corrcoef(sampled, critic)[0, 1]), 12)
            if len(sampled) > 1 and sampled.std() > 0 and critic.std() > 0
            else None
        )
        mean_abs_error = float(np.mean(np.abs(critic - sampled)))
    else:
        distribution = {
            "n_builds": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "spread": None,
        }
        correlation = None
        mean_abs_error = None

    joker_counts: Counter[str] = Counter()
    for record in records:
        joker_counts.update(
            joker.get("center_key", "")
            for joker in record.get("build", {}).get("jokers", [])
            if joker.get("center_key")
        )

    summary = {
        "sampled_clear_distribution": distribution,
        "critic_vs_sampled": {
            "correlation": correlation,
            "mean_abs_error": mean_abs_error,
        },
        "coverage": {
            "distinct_jokers": len(joker_counts),
            "appearance_counts": dict(sorted(joker_counts.items())),
        },
        "coverage_caveat": (
            "This is the on-policy build distribution, so jokers s0 rarely buys may be absent. "
            "If coverage is too sparse to read a spread, use a synthetic random-subset "
            "critic-only sweep as the fallback; it is not part of this probe."
        ),
    }
    if requested_runs is not None:
        summary["rollout_coverage"] = {
            "requested_runs": requested_runs,
            "terminal_boss_records": len(records),
            "runs_without_a_reached_boss": requested_runs - len(records),
        }
    return summary


def _shop_env(hand_partner: CheckpointProbePartner) -> ShopGymEnv:
    config = ShopRunConfig(win_ante=BOSS_PROBE_WIN_ANTE)
    env = ShopGymEnv(
        config=config,
        hand_policy=hand_partner,
        max_steps=_MAX_SHOP_STEPS,
    )
    env._adapter = BossCapturingShopRunAdapter(hand_partner, config)
    return env


def run_probe(
    shop_policy: Any,
    partner: CheckpointProbePartner,
    *,
    n_runs: int,
    n_redeals: int,
    out_path: str | Path,
    keep_blobs: bool = False,
) -> list[dict[str, Any]]:
    """Roll out full runs, probe their terminal bosses, and stream JSONL."""
    if n_runs < 1:
        raise ValueError("n_runs must be positive")
    if n_redeals < 1:
        raise ValueError("n_redeals must be positive")

    env = _shop_env(partner)
    adapter = env._adapter
    assert isinstance(adapter, BossCapturingShopRunAdapter)
    records: list[dict[str, Any]] = []
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as output:
        for index in range(n_runs):
            run_seed = f"{BOSS_PROBE_PREFIX}_{index:08d}"
            try:
                obs, info = env.reset(options={"episode_seed": run_seed})
            except RuntimeError:
                continue

            terminated = False
            for _ in range(_MAX_SHOP_STEPS):
                action = shop_policy.act(obs, info["action_mask"])
                obs, _, terminated, truncated, info = env.step(action)
                if truncated:
                    raise RuntimeError(
                        f"shop rollout truncated before natural termination: {run_seed}"
                    )
                if terminated:
                    break
            if not terminated:
                raise RuntimeError(f"shop rollout exceeded step budget: {run_seed}")
            capture = adapter.last_boss_capture
            if capture is None:
                continue

            record = build_probe_record(
                capture,
                run_seed=run_seed,
                n_redeals=n_redeals,
                partner=partner,
                keep_blob=keep_blobs,
            )
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            records.append(record)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shop-policy", type=Path, required=True)
    parser.add_argument("--hand-policy", type=Path, required=True)
    parser.add_argument("--n-runs", type=int, required=True)
    parser.add_argument("--redeals", type=int, default=DEFAULT_REDEALS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-blobs", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    from eval_shop_policy import PPOPolicy

    args = build_parser().parse_args()
    partner = CheckpointProbePartner(args.hand_policy, device=args.device)
    shop_policy = PPOPolicy(args.shop_policy, args.device)
    records = run_probe(
        shop_policy,
        partner,
        n_runs=args.n_runs,
        n_redeals=args.redeals,
        out_path=args.out,
        keep_blobs=args.keep_blobs,
    )
    print(json.dumps(build_summary(records, requested_runs=args.n_runs), indent=2))


if __name__ == "__main__":
    main()
