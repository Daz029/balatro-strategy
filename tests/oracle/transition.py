"""Python API and schema for the headless Lua transition oracle."""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from jackdaw.engine.actions import CashOut, Discard, GamePhase, PlayHand, SelectBlind
from jackdaw.engine.blind import Blind
from jackdaw.engine.card_factory import create_consumable, create_joker, create_playing_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.game import step
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.run_init import initialize_run

_ROOT = Path(__file__).resolve().parents[2]
_BOOTSTRAP = _ROOT / "scripts" / "lua_transition_oracle" / "bootstrap.lua"
_DEFAULT_SOURCE = Path.home() / "Code/Code/balatro-strategy/balatro_source/Balatro"

ActionName = Literal["play", "discard", "end_round", "set_blind", "cash_out"]


@dataclass(slots=True)
class CardSpec:
    """One stable-identity card in a scenario."""

    id: str
    front: str | None = None
    center: str | None = None
    key: str | None = None
    edition: str | dict[str, Any] | None = None
    seal: str | None = None
    ability: dict[str, Any] = field(default_factory=dict)
    debuff: bool = False
    facing: str = "front"


@dataclass(slots=True)
class BlindSpec:
    key: str = "bl_small"
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Action:
    type: ActionName
    indices: tuple[int, ...] = ()
    key: str | None = None

    @classmethod
    def play(cls, *indices: int) -> Action:
        return cls("play", indices)

    @classmethod
    def discard(cls, *indices: int) -> Action:
        return cls("discard", indices)

    @classmethod
    def end_round(cls) -> Action:
        return cls("end_round")

    @classmethod
    def set_blind(cls, key: str) -> Action:
        return cls("set_blind", key=key)

    @classmethod
    def cash_out(cls) -> Action:
        return cls("cash_out")


@dataclass(slots=True)
class Scenario:
    """JSON-serialisable controlled state shared by Lua and Python."""

    seed: str = "ORACLE"
    stake: int = 1
    back: str = "b_red"
    ante: int = 1
    round: int = 1
    dollars: int = 4
    chips: int = 0
    hands_left: int = 4
    discards_left: int = 3
    hands_per_round: int = 4
    discards_per_round: int = 3
    hands_played: int = 0
    round_hands_played: int = 0
    discards_used: int = 0
    round_dollars: int = 0
    last_hand_played: str | None = None
    most_played_poker_hand: str = "High Card"
    hand_levels: dict[str, dict[str, Any]] = field(default_factory=dict)
    blind: BlindSpec = field(default_factory=BlindSpec)
    areas: dict[str, list[CardSpec]] = field(default_factory=dict)
    jokers: list[CardSpec] = field(default_factory=list)
    consumables: list[CardSpec] = field(default_factory=list)
    modifiers: dict[str, Any] = field(default_factory=dict)
    hand_limit: int = 8
    joker_limit: int = 5
    consumable_limit: int = 2
    action: Action = field(default_factory=lambda: Action.play(0))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(slots=True)
class Observation:
    label: str
    kind: str
    state: dict[str, Any]


@dataclass(slots=True)
class OracleTrace:
    observations: list[Observation]
    source: str
    drain_updates: int = 0

    @property
    def synchronous(self) -> dict[str, Any]:
        return self.observations[0].state

    @property
    def final(self) -> dict[str, Any]:
        return self.observations[-1].state


@dataclass(frozen=True, slots=True)
class Diff:
    path: str
    lua: Any
    engine: Any

    def __str__(self) -> str:
        return f"{self.path}: Lua={self.lua!r}, engine={self.engine!r}"


def _source_path() -> Path:
    return Path(os.environ.get("BALATRO_SOURCE", _DEFAULT_SOURCE)).expanduser()


def lua_oracle_unavailable_reason() -> str | None:
    source = _source_path()
    if not source.joinpath("functions/state_events.lua").is_file():
        return f"Balatro 1.0.1o source not found at {source}"
    try:
        import lupa.lua51  # noqa: F401, PLC0415
    except ImportError:
        return "lupa.lua51 is not installed"
    return None


def _to_lua(runtime: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        table = runtime.table()
        for key, item in value.items():
            if item is not None:
                table[key] = _to_lua(runtime, item)
        return table
    if isinstance(value, (list, tuple)):
        table = runtime.table()
        for index, item in enumerate(value, 1):
            table[index] = _to_lua(runtime, item)
        return table
    return value


@lru_cache(maxsize=4)
def _lua_runtime(source: str) -> Any:
    from lupa.lua51 import LuaRuntime  # type: ignore[import-untyped]  # noqa: PLC0415

    from jackdaw.engine.rng import (  # noqa: PLC0415
        _luajit_random,
        _luajit_random_int,
        _luajit_seed,
    )

    runtime = LuaRuntime(unpack_returned_tuples=True)
    random_state = _luajit_seed(0.5)

    def randomseed(seed: float) -> None:
        nonlocal random_state
        random_state = _luajit_seed(float(seed))

    def random(*bounds: int) -> float | int:
        if not bounds:
            return _luajit_random(random_state)
        if len(bounds) == 1:
            return _luajit_random_int(random_state, 1, int(bounds[0]))
        return _luajit_random_int(random_state, int(bounds[0]), int(bounds[1]))

    runtime.globals().oracle_randomseed = randomseed
    runtime.globals().oracle_random = random
    runtime.globals().BALATRO_SOURCE = source
    runtime.execute(_BOOTSTRAP.read_text())
    return runtime


def _trace_from_payload(payload: dict[str, Any], *, source: str) -> OracleTrace:
    observations = []
    for raw in payload["observations"]:
        state = dict(raw)
        label = state.pop("label", "event")
        kind = state.pop("kind", "event")
        state.setdefault("last_hand_played", None)
        if state.get("pseudorandom") == []:
            state["pseudorandom"] = {}
        observations.append(Observation(label=label, kind=kind, state=state))
    return OracleTrace(
        observations=observations,
        source=payload.get("source", source),
        drain_updates=payload.get("drain_updates", 0),
    )


def run_lua_oracle(scenario: Scenario) -> OracleTrace:
    """Run one transition through real Lua gameplay code using lupa."""

    reason = lua_oracle_unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    source = str(_source_path())
    runtime = _lua_runtime(source)
    encoded = runtime.globals().run_transition_oracle(_to_lua(runtime, scenario))
    return _trace_from_payload(json.loads(encoded), source=source)


_SUITS = {"H": Suit.HEARTS, "D": Suit.DIAMONDS, "C": Suit.CLUBS, "S": Suit.SPADES}
_RANKS = {
    "2": Rank.TWO,
    "3": Rank.THREE,
    "4": Rank.FOUR,
    "5": Rank.FIVE,
    "6": Rank.SIX,
    "7": Rank.SEVEN,
    "8": Rank.EIGHT,
    "9": Rank.NINE,
    "T": Rank.TEN,
    "J": Rank.JACK,
    "Q": Rank.QUEEN,
    "K": Rank.KING,
    "A": Rank.ACE,
}


def _merge(target: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _engine_card(spec: CardSpec, gs: dict[str, Any], playing_index: int | None) -> Any:
    if spec.front:
        suit_code, rank_code = spec.front.split("_", maxsplit=1)
        card = create_playing_card(
            _SUITS[suit_code],
            _RANKS[rank_code],
            enhancement=spec.center or spec.key or "c_base",
            edition={spec.edition: True} if isinstance(spec.edition, str) else spec.edition,
            seal=spec.seal,
            playing_card_index=playing_index,
            hands_played=gs.get("hands_played", 0),
            game_state=gs,
        )
    else:
        key = spec.key or spec.center
        if key is None:
            raise ValueError(f"non-playing card {spec.id!r} requires key or center")
        if key.startswith("j_"):
            card = create_joker(
                key,
                edition={spec.edition: True} if isinstance(spec.edition, str) else spec.edition,
                hands_played=gs.get("hands_played", 0),
                game_state=gs,
            )
        else:
            card = create_consumable(key, hands_played=gs.get("hands_played", 0), game_state=gs)
    _merge(card.ability, spec.ability)
    card.debuff = spec.debuff
    card.facing = spec.facing
    card._oracle_id = spec.id
    return card


def build_engine_state(scenario: Scenario) -> dict[str, Any]:
    """Build the engine equivalent of a :class:`Scenario`."""

    gs = initialize_run(scenario.back, scenario.stake, scenario.seed)
    gs["rng"] = PseudoRandom(scenario.seed)
    gs["round_resets"]["ante"] = scenario.ante
    gs["round_resets"]["hands"] = scenario.hands_per_round
    gs["round_resets"]["discards"] = scenario.discards_per_round
    gs["round"] = scenario.round
    gs["dollars"] = scenario.dollars
    gs["chips"] = scenario.chips
    gs["hands_played"] = scenario.hands_played
    gs["last_hand_played"] = scenario.last_hand_played
    gs["modifiers"].update(copy.deepcopy(scenario.modifiers))
    gs["hand_size"] = scenario.hand_limit
    gs["joker_slots"] = scenario.joker_limit
    gs["consumable_slots"] = scenario.consumable_limit

    cr = gs["current_round"]
    cr.update(
        hands_left=scenario.hands_left,
        discards_left=scenario.discards_left,
        hands_played=scenario.round_hands_played,
        discards_used=scenario.discards_used,
        most_played_poker_hand=scenario.most_played_poker_hand,
        dollars=scenario.round_dollars,
    )
    levels = HandLevels()
    for hand_name, overrides in scenario.hand_levels.items():
        hand = levels.get_state(hand_name)
        for key, value in overrides.items():
            setattr(hand, key, value)
    gs["hand_levels"] = levels

    for name in ("hand", "deck", "jokers", "consumables", "discard_pile", "played_cards_area"):
        gs[name] = []
    gs["playing_cards"] = []
    area_map = {
        "hand": "hand",
        "play": "played_cards_area",
        "deck": "deck",
        "discard": "discard_pile",
        "jokers": "jokers",
        "consumeables": "consumables",
    }
    all_areas = dict(scenario.areas)
    all_areas["jokers"] = scenario.jokers or all_areas.get("jokers", [])
    all_areas["consumeables"] = scenario.consumables or all_areas.get("consumeables", [])
    playing_index = 0
    for lua_name, engine_name in area_map.items():
        for spec in all_areas.get(lua_name, []):
            is_playing = spec.front is not None
            playing_index += int(is_playing)
            card = _engine_card(spec, gs, playing_index if is_playing else None)
            gs[engine_name].append(card)
            if is_playing:
                gs["playing_cards"].append(card)

    key = scenario.blind.key
    blind_on_deck = "Small" if key == "bl_small" else "Big" if key == "bl_big" else "Boss"
    blind = Blind.create(key, scenario.ante)
    for name, value in scenario.blind.overrides.items():
        setattr(blind, name, copy.deepcopy(value))
    gs["blind"] = blind
    gs["round_resets"]["blind"] = blind
    gs["round_resets"]["blind_choices"][blind_on_deck] = key
    gs["blind_on_deck"] = blind_on_deck
    if scenario.action.type == "set_blind":
        gs["phase"] = GamePhase.BLIND_SELECT
    elif scenario.action.type == "cash_out":
        from jackdaw.engine.economy import RoundEarnings  # noqa: PLC0415

        gs["phase"] = GamePhase.ROUND_EVAL
        gs["round_earnings"] = RoundEarnings(total=scenario.round_dollars)
    else:
        gs["phase"] = GamePhase.SELECTING_HAND
    return gs


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _json_safe(getattr(value, item.name)) for item in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _snapshot_card(card: Any) -> dict[str, Any]:
    base = None
    if card.base is not None:
        base = {"rank": card.base.rank.value, "suit": card.base.suit.value}
    return {
        "id": getattr(card, "_oracle_id", None),
        "key": card.center_key or card.card_key,
        "base": base,
        "ability": _json_safe(card.ability),
        "edition": _json_safe(card.edition),
        "seal": card.seal,
        "debuff": bool(card.debuff),
        "facing": card.facing,
    }


def snapshot_engine_state(gs: dict[str, Any]) -> dict[str, Any]:
    """The sole Python-to-oracle field-name mapping."""

    levels: HandLevels = gs["hand_levels"]
    hands = {}
    for hand_name in (
        "Flush Five",
        "Flush House",
        "Five of a Kind",
        "Straight Flush",
        "Four of a Kind",
        "Full House",
        "Flush",
        "Straight",
        "Three of a Kind",
        "Two Pair",
        "Pair",
        "High Card",
    ):
        state = levels.get_state(hand_name)
        hands[hand_name] = {
            "played": state.played,
            "played_this_round": state.played_this_round,
            "level": state.level,
        }
    blind = gs.get("blind")
    rng = gs.get("rng")
    rng_state = rng.get_state() if rng else {}
    rng_state = {
        key: value for key, value in rng_state.items() if key not in {"seed", "hashed_seed"}
    }
    cr = gs["current_round"]
    round_dollars = cr.get("dollars", 0)
    if gs.get("phase") == GamePhase.ROUND_EVAL and gs.get("round_earnings") is not None:
        round_dollars = gs["round_earnings"].total
    return {
        "phase": _json_safe(gs.get("phase")),
        "dollars": gs.get("dollars", 0),
        "dollar_buffer": gs.get("dollar_buffer", 0),
        "joker_buffer": gs.get("joker_buffer", 0),
        "consumeable_buffer": gs.get("consumeable_buffer", 0),
        "chips": gs.get("chips", 0),
        "current_round": {
            "hands_left": cr.get("hands_left", 0),
            "discards_left": cr.get("discards_left", 0),
            "hands_played": cr.get("hands_played", 0),
            "discards_used": cr.get("discards_used", 0),
            "most_played_poker_hand": cr.get("most_played_poker_hand", "High Card"),
            "dollars": round_dollars,
        },
        "hands_played": gs.get("hands_played", 0),
        "last_hand_played": gs.get("last_hand_played"),
        "hands": hands,
        "blind": {
            "name": getattr(blind, "name", None),
            "chips": getattr(blind, "chips", None),
            "triggered": bool(getattr(blind, "triggered", False)),
            "disabled": bool(getattr(blind, "disabled", False)),
            "prepped": bool(getattr(blind, "prepped", False)),
        },
        "areas": {
            "hand": [_snapshot_card(card) for card in gs.get("hand", [])],
            "play": [_snapshot_card(card) for card in gs.get("played_cards_area", [])],
            "deck": [_snapshot_card(card) for card in gs.get("deck", [])],
            "discard": [_snapshot_card(card) for card in gs.get("discard_pile", [])],
            "jokers": [_snapshot_card(card) for card in gs.get("jokers", [])],
            "consumeables": [_snapshot_card(card) for card in gs.get("consumables", [])],
        },
        "pseudorandom": rng_state,
        "events_pending": 0,
    }


EngineProbe = Callable[[str, dict[str, Any]], None]


def run_engine(scenario: Scenario, *, probe: EngineProbe | None = None) -> OracleTrace:
    """Apply one Jackdaw action and return its final observation.

    ``probe`` is an intentionally small seam for Phase 5 to wire to internal
    transition points.  It currently receives the public transition boundary
    states only; no engine instrumentation is required by the oracle.
    """

    gs = build_engine_state(scenario)
    if probe is not None:
        probe("before_action", gs)
    action = scenario.action
    if action.type == "play":
        step(gs, PlayHand(action.indices))
    elif action.type == "discard":
        step(gs, Discard(action.indices))
    elif action.type == "set_blind":
        step(gs, SelectBlind(allow_forced_boss=True))
    elif action.type == "cash_out":
        step(gs, CashOut())
    elif action.type == "end_round":
        # Round completion has no public player Action: it is an automatic
        # consequence of play.  Use the engine's canonical transition body.
        from jackdaw.engine.game import end_round  # noqa: PLC0415

        end_round(gs)
    else:  # pragma: no cover - ActionName makes this unreachable
        raise ValueError(action.type)
    if probe is not None:
        probe("after_action", gs)
    state = snapshot_engine_state(gs)
    return OracleTrace(
        observations=[Observation(label="final", kind="drained", state=state)],
        source="jackdaw.engine.game.step",
    )


_COSMETIC = {
    # CardArea:align_cards flips these piles for rendering only
    # (Balatro 1.0.1o cardarea.lua:410-428). No gameplay branch reads their
    # facing while the cards remain in either pile.
    "areas.deck[*].facing": "cardarea.lua:410-428",
    "areas.discard[*].facing": "cardarea.lua:410-428",
}


def _cosmetic_path(path: str) -> bool:
    normalized = re.sub(r"\[\d+\]", "[*]", path)
    return normalized in _COSMETIC


def _walk_diffs(lua: Any, engine: Any, path: str, out: list[Diff]) -> None:
    if _cosmetic_path(path):
        return
    if isinstance(lua, dict) and isinstance(engine, dict):
        for key in sorted(set(lua) | set(engine)):
            _walk_diffs(lua.get(key), engine.get(key), f"{path}.{key}" if path else key, out)
        return
    if isinstance(lua, list) and isinstance(engine, list):
        for index in range(max(len(lua), len(engine))):
            left = lua[index] if index < len(lua) else None
            right = engine[index] if index < len(engine) else None
            _walk_diffs(left, right, f"{path}[{index}]", out)
        return
    if lua != engine:
        out.append(Diff(path, lua, engine))


def compare(
    lua_trace: OracleTrace,
    engine_trace: OracleTrace,
    fields: Iterable[str] | None = None,
) -> list[Diff]:
    """Return readable field-for-field differences between final snapshots."""

    lua, engine = lua_trace.final, engine_trace.final
    selected = tuple(fields) if fields is not None else tuple(sorted(set(lua) | set(engine)))
    diffs: list[Diff] = []
    for field_name in selected:
        lua_value: Any = lua
        engine_value: Any = engine
        for part in field_name.split("."):
            lua_value = lua_value.get(part) if isinstance(lua_value, dict) else None
            engine_value = engine_value.get(part) if isinstance(engine_value, dict) else None
        _walk_diffs(lua_value, engine_value, field_name, diffs)
    return diffs
