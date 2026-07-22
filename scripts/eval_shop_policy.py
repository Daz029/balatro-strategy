"""Fixed-suite evaluation for shop policies.

The consistent yardstick across s0 horizon stages and later bootstrap
iterations:

  - **Fixed eval suite**: episode seeds ``EVAL_{i:08d}``. ShopGymEnv resets
    are seed-deterministic, so every policy forever faces the identical run
    distribution (given the same hand-policy partner). The ``EVAL_`` prefix
    is reserved — training rollouts must never use it.
  - **Metrics**: win rate at the given ``--win-ante`` horizon under
    deterministic (masked argmax) actions, plus mean final ante / rounds
    cleared / decision count — the progress fingerprint when win rate is
    still near zero.
  - **Baseline**: ``--policy nextround`` is the do-nothing shop (leave every
    shop immediately, skip every pack). The gap between a trained policy
    and this baseline isolates shop value from hand-play skill (same
    partner on both sides).

Seeds where the hand policy loses the auto-resolved FIRST blind (before
the shop agent's first decision) are excluded from the rates and reported
as ``n_dead_at_reset`` — no shop decision influenced them.

Usage::

    uv run python scripts/eval_shop_policy.py \
        --policy runs/shop_ppo/stage_a2/best_model/best_model.zip \
        --win-ante 2 --n-episodes 200 --output runs/shop_ppo/stage_a2/eval.json

    uv run python scripts/eval_shop_policy.py --policy nextround --win-ante 2
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import enum
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jackdaw.agents.shop_action_space import (  # noqa: E402
    MAX_JOKER_ROWS,
    ShopActionFamily,
    decode_shop_action,
    shop_action,
    target_combo_for_action,
)
from jackdaw.engine.actions import Action, PlayHand  # noqa: E402
from jackdaw.env.maskable_guard import install_stale_probs_guard  # noqa: E402
from jackdaw.env.shop_gym import BACK_KEY, STAKE, ShopGymEnv  # noqa: E402
from jackdaw.env.shop_run_adapter import ShopRunConfig  # noqa: E402

EVAL_SEED_PREFIX = "EVAL"

# Generous per-episode decision budget; the env's own max_steps also caps.
_MAX_EPISODE_STEPS = 512


def _serialize_card(card: Any) -> dict[str, Any]:
    """Return the identity and mutable state needed to audit a card decision.

    Evaluation traces are intentionally self-contained: a later charting tool
    should not have to replay the seed to discover what a shop slot contained.
    """
    base = getattr(card, "base", None)
    base_data = None
    if base is not None:
        base_data = {
            "card_key": getattr(card, "card_key", None),
            "suit": _jsonable(getattr(base, "suit", None)),
            "rank": _jsonable(getattr(base, "rank", None)),
            "id": getattr(base, "id", None),
            "nominal": getattr(base, "nominal", None),
            "times_played": getattr(base, "times_played", None),
        }
    return {
        "sort_id": getattr(card, "sort_id", None),
        "center_key": getattr(card, "center_key", None),
        "card_key": getattr(card, "card_key", None),
        "name": getattr(card, "ability", {}).get("name", ""),
        "set": getattr(card, "ability", {}).get("set", ""),
        "ability": _jsonable(getattr(card, "ability", {})),
        "base": base_data,
        "edition": _jsonable(getattr(card, "edition", None)),
        "seal": getattr(card, "seal", None),
        "debuff": bool(getattr(card, "debuff", False)),
        "base_cost": getattr(card, "base_cost", None),
        "cost": getattr(card, "cost", None),
        "sell_cost": getattr(card, "sell_cost", None),
        "extra_cost": getattr(card, "extra_cost", None),
        "eternal": bool(getattr(card, "eternal", False)),
        "perishable": bool(getattr(card, "perishable", False)),
        "perish_tally": getattr(card, "perish_tally", None),
        "rental": bool(getattr(card, "rental", False)),
    }


def _jsonable(value: Any) -> Any:
    """Convert engine values into stable JSON data for the eval dump."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if hasattr(value, "center_key") and hasattr(value, "ability"):
        return _serialize_card(value)
    if dataclasses.is_dataclass(value):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return repr(value)


def _serialize_hand_levels(levels: Any) -> dict[str, Any] | None:
    if levels is None:
        return None
    hands = getattr(levels, "_hands", None)
    if hands is None:
        return _jsonable(levels)
    return {
        str(getattr(hand_type, "value", hand_type)): _jsonable(state)
        for hand_type, state in hands.items()
    }


def _serialize_state(
    gs: dict[str, Any], pending: Any, *, episode_seed: str, win_ante: int, s1_schema: bool
) -> dict[str, Any]:
    """Capture the decision-visible run state and exact card inventories."""
    rr = gs.get("round_resets", {})
    cr = gs.get("current_round", {})
    blind = gs.get("blind") or rr.get("blind")

    def cards(key: str) -> list[dict[str, Any]]:
        return [_serialize_card(card) for card in gs.get(key, [])]

    return {
        "run": {
            "seed": episode_seed,
            "back_key": BACK_KEY,
            "stake": STAKE,
            "win_ante": win_ante,
            "s1_schema": s1_schema,
        },
        "phase": _jsonable(gs.get("phase")),
        "ante": rr.get("ante", 1),
        "round": gs.get("round", 0),
        "dollars": gs.get("dollars", 0),
        "chips": gs.get("chips", 0),
        "won": bool(gs.get("won", False)),
        "done": bool(gs.get("won", False)) or _jsonable(gs.get("phase")) == "game_over",
        "blind_on_deck": gs.get("blind_on_deck"),
        "blind": _jsonable(blind),
        "current_round": _jsonable(cr),
        "round_resets": _jsonable(rr),
        "last_score_result": _jsonable(gs.get("last_score_result")),
        "round_earnings": _jsonable(gs.get("round_earnings")),
        "hand_levels": _serialize_hand_levels(gs.get("hand_levels")),
        "resources": {
            key: _jsonable(gs.get(key))
            for key in (
                "hand_size",
                "joker_slots",
                "consumable_slots",
                "hands_left",
                "discards_left",
                "pack_choices_remaining",
                "pack_type",
                "used_vouchers",
                "tags",
                "round_scores",
                "hands_played",
                "cards_purchased",
                "times_rerolled",
            )
            if key in gs
        },
        "inventory": {
            "hand": cards("hand"),
            "jokers": cards("jokers"),
            "consumables": cards("consumables"),
        },
        "shop": {
            "config": _jsonable(gs.get("shop", {})),
            "cards": cards("shop_cards"),
            "vouchers": cards("shop_vouchers"),
            "boosters": cards("shop_boosters"),
        },
        "pack": {
            "cards": cards("pack_cards"),
            "hand": cards("pack_hand"),
        },
        "counts": {
            "deck": len(gs.get("deck", [])),
            "discard": len(gs.get("discard_pile", [])),
            "played_cards": len(gs.get("played_cards_area", [])),
        },
        "pending_target": _jsonable(pending),
    }


def _serialize_action_target(
    family: ShopActionFamily,
    slot: int,
    action: int,
    gs: dict[str, Any],
) -> dict[str, Any] | None:
    """Attach the exact card selected by a card-targeting action."""
    sources = {
        ShopActionFamily.BuyCard: ("shop_cards", "shop_card"),
        ShopActionFamily.RedeemVoucher: ("shop_vouchers", "voucher"),
        ShopActionFamily.OpenBooster: ("shop_boosters", "booster"),
        ShopActionFamily.SellJoker: ("jokers", "joker"),
        ShopActionFamily.SellJokerExt: ("jokers", "joker"),
        ShopActionFamily.SellConsumable: ("consumables", "consumable"),
        ShopActionFamily.UseConsumable: ("consumables", "consumable"),
        ShopActionFamily.PickPackCard: ("pack_cards", "pack_card"),
    }
    source = sources.get(family)
    if source is not None:
        key, kind = source
        entity_slot = slot + MAX_JOKER_ROWS if family is ShopActionFamily.SellJokerExt else slot
        items = gs.get(key, [])
        if 0 <= entity_slot < len(items):
            return {
                "kind": kind,
                "slot": entity_slot,
                "action_slot": slot,
                "card": _serialize_card(items[entity_slot]),
            }
        return {"kind": kind, "slot": entity_slot, "action_slot": slot, "card": None}
    if family is ShopActionFamily.SelectTarget:
        combo = target_combo_for_action(action)
        return {
            "kind": "target_cards",
            "slots": list(combo),
            "cards": [
                _serialize_card(gs["hand"][index])
                for index in combo
                if 0 <= index < len(gs.get("hand", []))
            ],
        }
    return None


def _serialize_hand_decision(
    *,
    seed: str,
    hand_decision_index: int,
    pre_state: dict[str, Any],
    action: Action,
    post_state: dict[str, Any],
) -> dict[str, Any]:
    """Build one self-contained fingerprint for an auto-resolved hand decision."""
    selected_indices = list(getattr(action, "card_indices", ()))
    hand = pre_state.get("hand", [])
    selected_cards = [
        _serialize_card(hand[index]) for index in selected_indices if 0 <= index < len(hand)
    ]
    played = isinstance(action, PlayHand)
    score_result = post_state.get("last_score_result") if played else None
    blind = pre_state.get("blind") or pre_state.get("round_resets", {}).get("blind")
    current_round = pre_state.get("current_round", {})

    return {
        "seed": seed,
        "hand_decision_index": hand_decision_index,
        "ante": pre_state.get("round_resets", {}).get("ante", 1),
        "round": pre_state.get("round", 0),
        "blind": _jsonable(blind),
        "blind_points": int(getattr(blind, "chips", 0)),
        "money": int(pre_state.get("dollars", 0)),
        "points": int(pre_state.get("chips", 0)),
        "post_points": int(post_state.get("chips", 0)),
        "hands_left": int(current_round.get("hands_left", 0)),
        "discards_left": int(current_round.get("discards_left", 0)),
        "hand_size": (
            int(pre_state["hand_size"]) if pre_state.get("hand_size") is not None else None
        ),
        "action_type": type(action).__name__,
        "selected_indices": selected_indices,
        "selected_cards": selected_cards,
        "jokers": [_serialize_card(card) for card in pre_state.get("jokers", [])],
        "consumables": [_serialize_card(card) for card in pre_state.get("consumables", [])],
        "cards_in_hand": [_serialize_card(card) for card in hand],
        "cards_in_deck": [_serialize_card(card) for card in pre_state.get("deck", [])],
        "cards_in_discard": [_serialize_card(card) for card in pre_state.get("discard_pile", [])],
        "played_hand": selected_cards if played else [],
        "hand_point_value": int(score_result.total) if score_result is not None else None,
        "hand_type": score_result.hand_type if score_result is not None else None,
        "hand_chips": score_result.chips if score_result is not None else None,
        "hand_mult": score_result.mult if score_result is not None else None,
        "score": _jsonable(score_result),
    }


class _HandDecisionTraceWriter:
    def __init__(self, trace_file: Any) -> None:
        self._trace_file = trace_file
        self._seed = ""
        self._hand_decision_index = 0

    def start_episode(self, seed: str) -> None:
        self._seed = seed
        self._hand_decision_index = 0

    def observe(
        self,
        pre_state: dict[str, Any],
        action: Action,
        post_state: dict[str, Any],
    ) -> None:
        self._hand_decision_index += 1
        record = _serialize_hand_decision(
            seed=self._seed,
            hand_decision_index=self._hand_decision_index,
            pre_state=pre_state,
            action=action,
            post_state=post_state,
        )
        self._trace_file.write(json.dumps(record) + "\n")


def eval_seeds(n_episodes: int) -> list[str]:
    return [f"{EVAL_SEED_PREFIX}_{i:08d}" for i in range(n_episodes)]


class NextRoundPolicy:
    """Do-nothing shop baseline: leave every shop, skip every pack."""

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        for family in (ShopActionFamily.NextRound, ShopActionFamily.SkipPack):
            action = shop_action(family)
            if mask[action]:
                return action
        return int(np.flatnonzero(mask)[0])  # pending-target etc.: first legal


class PPOPolicy:
    """Deterministic wrapper for a saved MaskablePPO .zip."""

    def __init__(self, model_path: Path, device: str) -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(model_path), device=device)
        install_stale_probs_guard()

    def act(self, obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        action, _ = self._model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)


def load_policy(policy: str, device: str):
    if policy == "nextround":
        return NextRoundPolicy()
    return PPOPolicy(Path(policy), device)


def run_suite(
    policy,
    win_ante: int,
    n_episodes: int,
    hand_policy=None,
    s1_schema: bool = False,
    dump_shop_decisions: Path | None = None,
    dump_hand_decisions: Path | None = None,
) -> dict:
    wins: list[bool] = []
    final_antes: list[int] = []
    rounds_cleared: list[int] = []
    steps: list[int] = []
    dead_at_reset = 0

    with contextlib.ExitStack() as stack:
        shop_trace_file = None
        if dump_shop_decisions is not None:
            dump_shop_decisions.parent.mkdir(parents=True, exist_ok=True)
            shop_trace_file = stack.enter_context(dump_shop_decisions.open("w", encoding="utf-8"))
        hand_trace_writer = None
        if dump_hand_decisions is not None:
            dump_hand_decisions.parent.mkdir(parents=True, exist_ok=True)
            hand_trace_file = stack.enter_context(dump_hand_decisions.open("w", encoding="utf-8"))
            hand_trace_writer = _HandDecisionTraceWriter(hand_trace_file)

        env = ShopGymEnv(
            config=ShopRunConfig(win_ante=win_ante, s1_schema=s1_schema),
            hand_policy=hand_policy,
            hand_decision_observer=(
                hand_trace_writer.observe if hand_trace_writer is not None else None
            ),
        )

        for seed in eval_seeds(n_episodes):
            if hand_trace_writer is not None:
                hand_trace_writer.start_episode(seed)
            try:
                obs, info = env.reset(options={"episode_seed": seed})
            except RuntimeError:
                # Hand policy lost the auto-resolved first blind — no shop
                # decision was ever made; not attributable to this policy.
                dead_at_reset += 1
                continue

            for step_count in range(1, _MAX_EPISODE_STEPS + 1):
                mask = info["action_mask"]
                gs = env._adapter.raw_state
                pre_state = _serialize_state(
                    gs,
                    env._pending,
                    episode_seed=seed,
                    win_ante=win_ante,
                    s1_schema=s1_schema,
                )
                action = int(policy.act(obs, mask))
                family, slot = decode_shop_action(action)
                action_label = (
                    f"SelectTarget{list(target_combo_for_action(action))}"
                    if family is ShopActionFamily.SelectTarget
                    else f"{family.name}[{slot}]"
                )
                record = {
                    "seed": seed,
                    "step": step_count,
                    "ante": gs.get("round_resets", {}).get("ante", 1),
                    "round": gs.get("round", 0),
                    "dollars": gs.get("dollars", 0),
                    "pending_target": env._pending is not None,
                    "action": int(action),
                    "action_family": family.name,
                    "action_slot": slot,
                    "action_label": action_label,
                    "n_legal": int(mask.sum()),
                    "legal_actions": [int(i) for i in np.nonzero(mask)[0]],
                    "action_target": _serialize_action_target(family, slot, action, gs),
                    "pre_state": pre_state,
                }
                obs, _, terminated, truncated, info = env.step(action)
                terminal = bool(terminated or truncated)
                record["terminal"] = terminal
                record["won"] = bool(info.get("balatro/won", False)) if terminal else None
                record["post_state"] = _serialize_state(
                    env._adapter.raw_state,
                    env._pending,
                    episode_seed=seed,
                    win_ante=win_ante,
                    s1_schema=s1_schema,
                )
                if shop_trace_file is not None:
                    shop_trace_file.write(json.dumps(record) + "\n")
                if terminal:
                    wins.append(bool(info.get("balatro/won", False)))
                    final_antes.append(int(info.get("balatro/ante", 1)))
                    rounds_cleared.append(int(info.get("balatro/round", 0)))
                    steps.append(step_count)
                    break
            else:
                raise AssertionError(f"episode {seed} did not terminate")

    n_played = len(wins)
    return {
        "win_ante": win_ante,
        "n_episodes": n_episodes,
        "n_played": n_played,
        "n_dead_at_reset": dead_at_reset,
        "win_rate": float(np.mean(wins)) if wins else None,
        "mean_final_ante": float(np.mean(final_antes)) if final_antes else None,
        "mean_rounds_cleared": float(np.mean(rounds_cleared)) if rounds_cleared else None,
        "mean_steps": float(np.mean(steps)) if steps else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        required=True,
        help='MaskablePPO .zip path, or "nextround" for the do-nothing baseline',
    )
    parser.add_argument("--win-ante", type=int, default=2)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dump-shop-decisions",
        "--dump_shop_decisions",
        dest="dump_shop_decisions",
        type=Path,
        default=None,
        help="write a rich per-decision JSONL trace with exact pre/post card "
        "inventories, offerings, targets, money, levels, and run state. "
        "Aggregate metrics are unaffected.",
    )
    parser.add_argument(
        "--dump-hand-decisions",
        "--dump_hand_decisions",
        dest="dump_hand_decisions",
        type=Path,
        default=None,
        help="write one detailed JSONL fingerprint per auto-resolved hand "
        "decision, including ante, blind, money, points, cards, selected "
        "play/discard, and the resulting hand score",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--s1-schema", action="store_true")
    parser.add_argument(
        "--hand-policy",
        type=Path,
        default=None,
        help="hand-partner checkpoint (.pt/.zip); omit for the greedy baseline. "
        "Match this to the partner s0 was TRAINED against.",
    )
    parser.add_argument(
        "--partner-money-ordering",
        action="store_true",
        help="use clear-gated money-aware copy-joker ordering with the hand partner",
    )
    args = parser.parse_args()
    if args.partner_money_ordering and args.hand_policy is None:
        parser.error("--partner-money-ordering requires --hand-policy")

    hand_policy = None
    if args.hand_policy is not None:
        from jackdaw.agents.hand_checkpoint_policy import HandCheckpointPolicy

        hand_policy = HandCheckpointPolicy(
            str(args.hand_policy), money_aware_ordering=args.partner_money_ordering
        )

    policy = load_policy(args.policy, args.device)
    result = run_suite(
        policy,
        args.win_ante,
        args.n_episodes,
        hand_policy=hand_policy,
        s1_schema=args.s1_schema,
        dump_shop_decisions=args.dump_shop_decisions,
        dump_hand_decisions=args.dump_hand_decisions,
    )
    result["policy"] = args.policy
    result["hand_policy"] = str(args.hand_policy) if args.hand_policy is not None else "greedy"
    result["partner_money_ordering"] = bool(args.partner_money_ordering)

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
