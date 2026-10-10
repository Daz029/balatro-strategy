"""Phase 1 exit criterion: every Lua state value has a mapped home.

Two halves:

- Without the Lua source: every non-lazy :class:`Stored` entry in
  ``jackdaw.engine.state_map`` resolves on a fresh run (plus a live ``Blind``),
  so the map cannot point at a key the engine does not hold.
- With the Lua source (``BALATRO_SOURCE``, default the 1.0.1o tree on the dev
  Mac): every ``G.GAME`` path, ``Card`` / ``Blind`` field and ``ability`` key the
  source uses is mapped. Skips without the source, like the LuaJIT oracles.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.state_map import (
    ABILITY_KEYS,
    BLIND_FIELDS,
    CARD_FIELDS,
    GAME_PATHS,
    Grabber,
    OutOfScope,
    Stored,
)
from tests._lua_source import lua_source_missing_reason, resolve_lua_source

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from engine_state_inventory import (  # noqa: E402
    scan_lua,
    scan_lua_ability_keys,
    scan_lua_self_fields,
)

_LUA_SRC = resolve_lua_source(Path(__file__).resolve().parents[2])

_MISSING = object()


def _resolve(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part, _MISSING)
        else:
            obj = getattr(obj, part, _MISSING)
        if obj is _MISSING:
            return _MISSING
    return obj


@pytest.fixture(scope="module")
def run_state() -> dict[str, Any]:
    gs = initialize_run("b_red", 1, "STATE_MAP")
    gs["blind"] = Blind.create("bl_small", ante=1)
    return gs


ALL_MAPS = {
    "GAME_PATHS": GAME_PATHS,
    "CARD_FIELDS": CARD_FIELDS,
    "BLIND_FIELDS": BLIND_FIELDS,
    "ABILITY_KEYS": ABILITY_KEYS,
}


@pytest.mark.parametrize("name", ALL_MAPS)
def test_entries_are_typed(name: str) -> None:
    for key, entry in ALL_MAPS[name].items():
        assert isinstance(entry, Stored | Grabber | OutOfScope), (name, key)
        if isinstance(entry, OutOfScope):
            assert entry.reason, (name, key)


@pytest.mark.parametrize(
    "lua_path",
    sorted(k for k, v in GAME_PATHS.items() if isinstance(v, Stored) and not v.lazy),
)
def test_stored_game_path_resolves(run_state: dict[str, Any], lua_path: str) -> None:
    entry = GAME_PATHS[lua_path]
    assert _resolve(run_state, entry.path) is not _MISSING, (
        f"G.GAME.{lua_path} is mapped to {entry.path!r}, which a fresh run does not hold"
    )


@pytest.mark.parametrize(
    "field_name",
    sorted(k for k, v in CARD_FIELDS.items() if isinstance(v, Stored)),
)
def test_stored_card_field_exists(field_name: str) -> None:
    entry = CARD_FIELDS[field_name]
    assert hasattr(Card(), entry.path), entry.path


@pytest.mark.parametrize(
    "field_name",
    sorted(k for k, v in BLIND_FIELDS.items() if isinstance(v, Stored)),
)
def test_stored_blind_field_exists(field_name: str) -> None:
    entry = BLIND_FIELDS[field_name]
    assert hasattr(Blind.create("bl_hook", ante=2), entry.path), entry.path


# ---------------------------------------------------------------------------
# Lua-source coverage (skips without the source tree)
# ---------------------------------------------------------------------------

_LUA_MISSING_REASON = lua_source_missing_reason(_LUA_SRC)
needs_lua = pytest.mark.skipif(
    _LUA_MISSING_REASON is not None,
    reason=_LUA_MISSING_REASON or "Balatro Lua source is available",
)


@needs_lua
def test_every_game_path_is_mapped() -> None:
    unmapped = sorted(set(scan_lua(_LUA_SRC, 2)) - set(GAME_PATHS))
    assert not unmapped, f"unmapped G.GAME paths: {unmapped}"


@needs_lua
def test_every_card_field_is_mapped() -> None:
    fields = scan_lua_self_fields(_LUA_SRC / "card.lua", statement_start=True)
    unmapped = sorted(fields - set(CARD_FIELDS))
    assert not unmapped, f"unmapped Card fields: {unmapped}"


@needs_lua
def test_every_blind_field_is_mapped() -> None:
    unmapped = sorted(scan_lua_self_fields(_LUA_SRC / "blind.lua") - set(BLIND_FIELDS))
    assert not unmapped, f"unmapped Blind fields: {unmapped}"


@needs_lua
def test_every_ability_key_is_mapped() -> None:
    unmapped = sorted(scan_lua_ability_keys(_LUA_SRC) - set(ABILITY_KEYS))
    assert not unmapped, f"unmapped ability keys: {unmapped}"
