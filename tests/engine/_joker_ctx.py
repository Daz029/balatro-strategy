"""Test-only builder for complete joker contexts."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from jackdaw.engine.effects import EffectQueue
from jackdaw.engine.jokers import GameSnapshot, JokerContext
from jackdaw.engine.read import Rules

_SNAPSHOT_FIELDS = {field.name for field in fields(GameSnapshot)}
_MISSING = object()


def make_ctx(**kwargs: Any) -> JokerContext:
    """Build a handler-test context with a snapshot and effect queue.

    Snapshot attributes remain convenient keyword arguments in tests while
    production keeps the stricter ``JokerContext(game=...)`` API.
    """
    explicit_game = kwargs.pop("game", None)
    explicit_queue = kwargs.pop("queue", _MISSING)
    consumables_owned = kwargs.pop("consumables_owned", [])

    smeared = kwargs.pop("smeared", False)
    pareidolia = kwargs.pop("pareidolia", False)
    snapshot_values = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key in _SNAPSHOT_FIELDS and key != "rules"
    }
    rules = kwargs.pop("rules", Rules(pareidolia=pareidolia, smeared=smeared))
    snapshot_values["rules"] = rules

    jokers = kwargs.get("jokers")
    joker_count = snapshot_values.get("joker_count", len(jokers or []))
    if jokers is None:
        queue_jokers = [object() for _ in range(joker_count)]
    else:
        queue_jokers = jokers
        snapshot_values.setdefault("joker_count", len(jokers))

    if isinstance(consumables_owned, int):
        consumables = [object() for _ in range(consumables_owned)]
    else:
        consumables = list(consumables_owned)

    gs = {
        "jokers": queue_jokers,
        "joker_slots": snapshot_values.get("joker_slots", 5),
        "consumables": consumables,
        "consumable_slots": kwargs.pop("consumable_slots", 2),
    }
    game = explicit_game or GameSnapshot(**snapshot_values)
    queue = EffectQueue(gs) if explicit_queue is _MISSING else explicit_queue
    return JokerContext(game=game, queue=queue, **kwargs)
