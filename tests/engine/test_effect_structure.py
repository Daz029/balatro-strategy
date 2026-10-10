"""Structural exit gates for the engine's typed Effect pipeline."""

from __future__ import annotations

import ast
import copy
import inspect
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from jackdaw.engine import effects, lifecycle
from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.jokers import JokerResult, calculate_joker, context_for, registered_jokers
from jackdaw.engine.play_ordering import (
    fast_clone_blind,
    fast_clone_card,
    fast_clone_hand_levels,
    fast_clone_rng,
)
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.scoring import score_hand

ROOT = Path(__file__).parents[2]
ENGINE = ROOT / "jackdaw" / "engine"
JACKDAW = ROOT / "jackdaw"


def _live_state() -> dict[str, Any]:
    return initialize_run("b_red", 1, "P44_EFFECT_STRUCTURE")


def _effect_case(
    effect_type: type[effects.Effect],
) -> tuple[dict[str, Any], effects.Effect, Callable[[], None]]:
    """Build one live-state application case for every concrete Effect."""
    gs = _live_state()

    if effect_type is effects.EaseDollars:
        before = gs["dollars"]
        return gs, effects.EaseDollars(amount=3), lambda: _assert_equal(gs["dollars"], before + 3)
    if effect_type is effects.SetDollars:
        return gs, effects.SetDollars(value=17), lambda: _assert_equal(gs["dollars"], 17)
    if effect_type is effects.AddChips:
        before = gs["chips"]
        return gs, effects.AddChips(amount=7), lambda: _assert_equal(gs["chips"], before + 7)
    if effect_type is effects.CreateCard:
        before = len(gs["consumables"])
        effect = effects.CreateCard(set="Tarot", forced_key="c_fool")
        return gs, effect, lambda: _assert_created(gs, "consumables", before, "c_fool")
    if effect_type is effects.CreatePlayingCard:
        before = len(gs["hand"])
        effect = effects.CreatePlayingCard(
            area="hand", suit="Hearts", rank="Ace", notify=False
        )
        return gs, effect, lambda: _assert_playing_card(gs, before, "Ace")
    if effect_type is effects.CreatePlayingCards:
        before = len(gs["hand"])
        effect = effects.CreatePlayingCards(
            cards=[
                {"suit": "Hearts", "rank": "2"},
                {"suit": "Spades", "rank": "3", "enhancement": "m_bonus"},
            ],
            notify=False,
        )
        return gs, effect, lambda: _assert_equal(len(gs["hand"]), before + 2)
    if effect_type is effects.CopyCard:
        source = gs["deck"][-1]
        before = len(gs["hand"])
        effect = effects.CopyCard(card=source, area="hand", notify=False)
        return gs, effect, lambda: _assert_copy(gs, source, before)
    if effect_type is effects.AddTag:
        effect = effects.AddTag(key="tag_double")
        return gs, effect, lambda: _assert_equal(gs["awarded_tags"][-1]["key"], "tag_double")
    if effect_type is effects.DestroyCard:
        target = create_joker("j_joker", game_state=gs)
        lifecycle.emplace(gs, target, "jokers")
        effect = effects.DestroyCard(card=target)
        return gs, effect, lambda: _assert_removed(gs, target, "jokers")
    if effect_type is effects.DestroyPlayingCards:
        targets = list(gs["deck"][-2:])
        effect = effects.DestroyPlayingCards(cards=targets, notify=False)
        return gs, effect, lambda: _assert_playing_cards_removed(gs, targets)
    if effect_type is effects.SetEnhancement:
        target = gs["deck"][-1]
        effect = effects.SetEnhancement(card=target, center="m_bonus")
        return gs, effect, lambda: _assert_equal(target.center_key, "m_bonus")
    if effect_type is effects.ChangeSuit:
        target = gs["deck"][-1]
        effect = effects.ChangeSuit(card=target, suit="Hearts")
        return gs, effect, lambda: _assert_equal(target.base.suit.value, "Hearts")
    if effect_type is effects.ChangeRank:
        target = gs["deck"][-1]
        effect = effects.ChangeRank(card=target, rank="King")
        return gs, effect, lambda: _assert_equal(target.base.rank.value, "King")
    if effect_type is effects.SetSeal:
        target = gs["deck"][-1]
        effect = effects.SetSeal(card=target, seal="Gold")
        return gs, effect, lambda: _assert_equal(target.seal, "Gold")
    if effect_type is effects.SetEdition:
        target = gs["deck"][-1]
        effect = effects.SetEdition(card=target, edition={"foil": True})
        return gs, effect, lambda: _assert_equal(target.edition["foil"], True)
    if effect_type is effects.LevelUpHand:
        before = gs["hand_levels"]["Pair"].level
        effect = effects.LevelUpHand(hand="Pair", amount=2)
        return gs, effect, lambda: _assert_equal(gs["hand_levels"]["Pair"].level, before + 2)
    if effect_type is effects.ChangeRoundResource:
        before_hands = gs["current_round"]["hands_left"]
        effect = effects.ChangeRoundResource(hands=2, discards=1)
        return gs, effect, lambda: _assert_round_resources(gs, before_hands)
    if effect_type is effects.ChangeHandSize:
        before = gs["hand_size"]
        effect = effects.ChangeHandSize(delta=-1)
        return gs, effect, lambda: _assert_equal(gs["hand_size"], before - 1)
    if effect_type is effects.DisableBlind:
        blind = Blind.create("bl_hook", ante=1)
        gs["blind"] = blind
        return gs, effects.DisableBlind(), lambda: _assert_equal(blind.disabled, True)
    if effect_type is effects.SetPoolFlag:
        effect = effects.SetPoolFlag(flag="gros_michel_extinct")
        return gs, effect, lambda: _assert_equal(gs["pool_flags"]["gros_michel_extinct"], True)
    raise AssertionError(f"missing application case for {effect_type.__name__}")


def _assert_equal(actual: Any, expected: Any) -> None:
    assert actual == expected


def _assert_created(gs: dict[str, Any], area: str, before: int, key: str) -> None:
    assert len(gs[area]) == before + 1
    assert gs[area][-1].center_key == key


def _assert_playing_card(gs: dict[str, Any], before: int, rank: str) -> None:
    assert len(gs["hand"]) == before + 1
    assert gs["hand"][-1].base.rank.value == rank


def _assert_copy(gs: dict[str, Any], source: Card, before: int) -> None:
    assert len(gs["hand"]) == before + 1
    clone = gs["hand"][-1]
    assert clone is not source
    assert clone.base == source.base
    assert clone.ability == source.ability


def _assert_removed(gs: dict[str, Any], target: Card, area: str) -> None:
    assert target.removed is True
    assert target not in gs[area]


def _assert_playing_cards_removed(gs: dict[str, Any], targets: list[Card]) -> None:
    assert all(card.removed for card in targets)
    assert not any(card in gs["deck"] for card in targets)


def _assert_round_resources(gs: dict[str, Any], before_hands: int) -> None:
    assert gs["current_round"]["hands_left"] == before_hands + 2
    assert gs["current_round"]["discards_left"] == 1


@pytest.mark.parametrize("effect_type", effects.EFFECT_TYPES, ids=lambda cls: cls.__name__)
def test_every_registered_effect_applies(effect_type: type[effects.Effect]) -> None:
    gs, effect, assert_change = _effect_case(effect_type)

    effects.apply_effects(gs, [effect])

    assert_change()


def test_effect_registry_is_exact_and_base_is_abstract() -> None:
    concrete = {
        cls
        for cls in vars(effects).values()
        if inspect.isclass(cls)
        and cls.__module__ == effects.__name__
        and issubclass(cls, effects.Effect)
        and cls is not effects.Effect
        and not inspect.isabstract(cls)
    }
    assert set(effects.EFFECT_TYPES) == concrete
    with pytest.raises(TypeError):
        effects.Effect()


@pytest.mark.parametrize("consumer", ["apply_effects", "queue"])
def test_effect_consumers_reject_untyped_values(consumer: str) -> None:
    gs = _live_state()
    with pytest.raises(TypeError, match="not an Effect"):
        if consumer == "apply_effects":
            effects.apply_effects(gs, [object()])  # type: ignore[list-item]
        else:
            effects.EffectQueue(gs).add([object()])  # type: ignore[list-item]


def test_calculate_joker_rejects_effects_without_a_queue() -> None:
    gs = _live_state()
    joker = create_joker("j_diet_cola", game_state=gs)

    with pytest.raises(effects.UnappliedEffectError, match="j_diet_cola produced"):
        calculate_joker(joker, context_for(gs, queue=None, selling_self=True))


_EFFECT_PHASE_FIELDS = frozenset(
    {
        "after",
        "before",
        "buying_card",
        "cards_destroyed",
        "debuffed_hand",
        "destroying_card",
        "discard",
        "end_of_round",
        "ending_shop",
        "first_hand_drawn",
        "game_over",
        "individual",
        "joker_main",
        "open_booster",
        "other_joker",
        "playing_card_added",
        "pre_discard",
        "repetition",
        "reroll_shop",
        "selling_card",
        "selling_self",
        "setting_blind",
        "skip_blind",
        "skipping_booster",
        "using_consumeable",
    }
)

_EXPECTED_EFFECT_PRODUCER_CONTEXTS = {
    "j_8_ball": {"individual"},
    "j_burglar": {"setting_blind"},
    "j_cartomancer": {"setting_blind"},
    "j_cavendish": {"end_of_round"},
    "j_ceremonial": {"setting_blind"},
    "j_certificate": {"first_hand_drawn"},
    "j_chicot": {"setting_blind"},
    "j_diet_cola": {"selling_self"},
    "j_gros_michel": {"end_of_round"},
    "j_hallucination": {"open_booster"},
    "j_ice_cream": {"after"},
    "j_invisible": {"selling_self"},
    "j_luchador": {"selling_self"},
    "j_madness": {"setting_blind"},
    "j_marble": {"setting_blind"},
    "j_mr_bones": {"game_over"},
    "j_perkeo": {"ending_shop"},
    "j_popcorn": {"end_of_round"},
    "j_ramen": {"discard"},
    "j_riff_raff": {"setting_blind"},
    "j_seance": {"joker_main"},
    "j_selzer": {"after"},
    "j_sixth_sense": {"destroying_card"},
    "j_superposition": {"joker_main"},
    "j_turtle_bean": {"end_of_round"},
    "j_vagabond": {"joker_main"},
}

# Deliberately undispatched contexts belong here as {flag: "phase P?-?"}.
# Phase 4 has none: adding an entry must document the phase that will remove it.
_UNDISPATCHED_EFFECT_CONTEXTS: dict[str, str] = {}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _registered_key(function: ast.FunctionDef) -> str | None:
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and _call_name(decorator) == "register"
            and decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        ):
            return decorator.args[0].value
    return None


def _effect_producer_contexts() -> tuple[dict[str, set[str]], list[str]]:
    tree = ast.parse((ENGINE / "jokers.py").read_text())
    found: dict[str, set[str]] = {}
    errors: list[str] = []
    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        key = _registered_key(function)
        if key is None:
            continue
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(function):
            parents.update((child, node) for child in ast.iter_child_nodes(node))
        for call in (
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and _call_name(node) == "JokerResult"
        ):
            effect_keywords = [keyword for keyword in call.keywords if keyword.arg == "effects"]
            if not effect_keywords:
                continue
            value = effect_keywords[0].value
            if isinstance(value, (ast.List, ast.Tuple)) and not value.elts:
                continue
            phase_flags: set[str] = set()
            parent = parents.get(call)
            while parent is not None:
                if isinstance(parent, ast.If):
                    phase_flags.update(
                        attribute.attr
                        for attribute in ast.walk(parent.test)
                        if isinstance(attribute, ast.Attribute)
                        and isinstance(attribute.value, ast.Name)
                        and attribute.value.id == "ctx"
                        and attribute.attr in _EFFECT_PHASE_FIELDS
                    )
                parent = parents.get(parent)
            if not phase_flags:
                errors.append(f"{key} line {call.lineno} has effects but no ctx phase guard")
            found.setdefault(key, set()).update(phase_flags)
    return found, errors


def _dispatched_contexts() -> set[str]:
    dispatched: set[str] = set()
    dispatchers = {"JokerContext", "context_for", "fire_jokers"}
    value_contexts = {"cards_destroyed", "destroying_card", "other_joker"}
    for path in ENGINE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if _call_name(call) not in dispatchers:
                continue
            for keyword in call.keywords:
                is_true = isinstance(keyword.value, ast.Constant) and keyword.value.value is True
                if keyword.arg in _EFFECT_PHASE_FIELDS and (
                    keyword.arg in value_contexts or is_true
                ):
                    dispatched.add(keyword.arg)
    return dispatched


def test_every_effect_producer_context_is_dispatched() -> None:
    producers, errors = _effect_producer_contexts()
    assert not errors, errors
    assert producers == _EXPECTED_EFFECT_PRODUCER_CONTEXTS
    assert not _UNDISPATCHED_EFFECT_CONTEXTS

    produced_contexts = set().union(*producers.values())
    missing = produced_contexts - _dispatched_contexts() - _UNDISPATCHED_EFFECT_CONTEXTS.keys()
    assert not missing, f"effect-producing contexts never dispatched: {sorted(missing)}"
    assert set(producers) <= set(registered_jokers())


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def test_contexts_have_one_production_construction_path() -> None:
    allowed_context_modules = {"jackdaw/engine/jokers.py", "jackdaw/engine/scoring.py"}
    bad_context_calls: list[str] = []
    bad_snapshot_calls: list[str] = []
    bad_context_replaces: list[str] = []
    context_fields = {field.name for field in fields(context_for(_live_state(), queue=None))}

    for path in JACKDAW.rglob("*.py"):
        relative = _relative(path)
        tree = ast.parse(path.read_text())
        imported_replace_names = {
            alias.asname or alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "dataclasses"
            for alias in node.names
            if alias.name == "replace"
        }
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = _call_name(call)
            location = f"{relative}:{call.lineno}"
            if name == "JokerContext" and relative not in allowed_context_modules:
                bad_context_calls.append(location)
            if name == "GameSnapshot":
                bad_snapshot_calls.append(location)
            is_replace = name in imported_replace_names or (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "dataclasses"
                and call.func.attr == "replace"
            )
            context_like = (
                call.args
                and isinstance(call.args[0], ast.Name)
                and (call.args[0].id.endswith("ctx") or "context" in call.args[0].id)
            ) or any(keyword.arg in context_fields for keyword in call.keywords)
            if is_replace and context_like and relative != "jackdaw/engine/jokers.py":
                bad_context_replaces.append(location)

    assert not bad_context_calls, f"direct JokerContext construction: {bad_context_calls}"
    assert not bad_snapshot_calls, f"production GameSnapshot construction: {bad_snapshot_calls}"
    assert not bad_context_replaces, (
        f"JokerContext dataclasses.replace outside jokers: {bad_context_replaces}"
    )


def test_test_context_helper_is_not_imported_by_production_or_scripts() -> None:
    bad_imports: list[str] = []
    for root in (JACKDAW, ROOT / "scripts"):
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "tests.engine._joker_ctx":
                    bad_imports.append(f"{_relative(path)}:{node.lineno}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "tests.engine._joker_ctx":
                            bad_imports.append(f"{_relative(path)}:{node.lineno}")
    assert not bad_imports, f"production imports the test context helper: {bad_imports}"


def _contains_dict_type(annotation: Any) -> bool:
    return annotation is dict or get_origin(annotation) is dict or any(
        _contains_dict_type(argument) for argument in get_args(annotation)
    )


def _names_bound_from_handler_results(tree: ast.AST) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        name = _call_name(value)
        is_handler = name in {"calculate_joker", "use_consumable"} or (
            name == "apply" and isinstance(value.func, ast.Attribute)
        )
        if not is_handler:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        bound.update(target.id for target in targets if isinstance(target, ast.Name))
    return bound


def test_handler_results_have_no_untyped_extra_channel() -> None:
    result_fields = {field.name for field in fields(JokerResult)}
    assert "extra" not in result_fields
    assert not any(
        _contains_dict_type(annotation) for annotation in get_type_hints(JokerResult).values()
    )

    reads: list[str] = []
    for path in ENGINE.rglob("*.py"):
        tree = ast.parse(path.read_text())
        result_names = _names_bound_from_handler_results(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "extra"
                and isinstance(node.value, ast.Name)
                and node.value.id in result_names
                and isinstance(node.ctx, ast.Load)
            ):
                reads.append(f"{_relative(path)}:{node.lineno}")
    assert not reads, f"handler result .extra reads: {reads}"


def _solver_snapshot(gs: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "consumables",
        "jokers",
        "deck",
        "hand",
        "dollars",
        "used_jokers",
        "joker_buffer",
        "consumeable_buffer",
    )
    return copy.deepcopy({key: gs[key] for key in keys})


def _hypothetical_score(gs: dict[str, Any], rank: str):
    played = fast_clone_card(gs["deck"][-1])
    rank_key = {"6": "6", "8": "8"}[rank]
    played.set_base(f"H_{rank_key}", "Hearts", rank)
    jokers = [fast_clone_card(joker) for joker in gs["jokers"]]
    return score_hand(
        [played],
        [],
        jokers,
        fast_clone_hand_levels(gs["hand_levels"]),
        fast_clone_blind(gs["blind"]),
        fast_clone_rng(gs["rng"]),
        game_state=gs,
    )


def test_solver_scoring_never_mutates_live_state(monkeypatch: pytest.MonkeyPatch) -> None:
    gs = _live_state()
    gs["blind"] = Blind.create("bl_small", ante=1)
    gs["dollars"] = 4
    gs["joker_slots"] = 8
    gs["consumable_slots"] = 8
    for key in ("j_8_ball", "j_vagabond", "j_dna", "j_sixth_sense", "j_riff_raff"):
        lifecycle.emplace(gs, create_joker(key, game_state=gs), "jokers")

    before = _solver_snapshot(gs)
    rng_before = gs["rng"].get_state()
    monkeypatch.setattr(type(gs["rng"]), "random", lambda _self, _key: 0.0)

    eight_result = _hypothetical_score(gs, "8")
    six_result = _hypothetical_score(gs, "6")

    assert eight_result.effects
    assert six_result.effects
    # DNA is synchronous in Lua's before pass (card.lua:3501-3511), so its
    # hypothetical copy is consumed by the held-card loop rather than left as
    # an F2 CopyCard effect. The live-state snapshot below is the isolation pin.
    assert sum(isinstance(effect, effects.CreateCard) for effect in eight_result.effects) >= 2
    assert any(isinstance(effect, effects.CreateCard) for effect in six_result.effects)
    assert _solver_snapshot(gs) == before
    assert gs["rng"].get_state() == rng_before
