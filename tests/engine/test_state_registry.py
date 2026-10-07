"""Structural guards for game-state keys (engine overhaul Phase 2, S2/S6).

The alias / no-writer bug class (C04, C07, C13, C14, D19, D26, D42, the dead
observation flags) is invisible to handler tests. These tests make it fail
mechanically:

- every key the engine, env or agents read or write is registered in
  ``state.STATE_KEYS`` / ``state.GROUP_KEYS`` (a new name is a new alias);
- every registered key that is read has a writer;
- no registry entry is dead;
- joker / consumable / tag handlers never read raw state (getters only).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from jackdaw.engine import shop
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.state import DYNAMIC_WRITE_GROUPS, GROUP_KEYS, STATE_KEYS

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from engine_state_inventory import scan_python  # noqa: E402

# rng.py: its local ``state`` is the PRNG's own dict, not game state.
# state.py: migrate_state reads removed aliases on purpose.
_SCAN_EXCLUDE = {"rng.py", "state.py"}
# Keys written through a module constant the scanner sees as ``[*]``.
_CONSTANT_KEY_WRITERS = {"first_shop_buffoon": shop._FIRST_SHOP_BUFFOON_KEY}


def _scan_files() -> list[Path]:
    files = [
        p for p in sorted((ROOT / "jackdaw/engine").rglob("*.py")) if p.name not in _SCAN_EXCLUDE
    ]
    for pkg in ("jackdaw/env", "jackdaw/agents"):
        files += sorted((ROOT / pkg).rglob("*.py"))
    return files


@pytest.fixture(scope="module")
def scan() -> dict[str, dict[str, list[str]]]:
    return scan_python(_scan_files(), 2)


@pytest.fixture(scope="module")
def created() -> set[str]:
    """Keys that run creation itself produces, across representative setups."""
    keys: set[str] = set()
    for back, stake in (("b_red", 1), ("b_green", 8), ("b_ghost", 1), ("b_magic", 1)):
        gs = initialize_run(back, stake, "REGISTRY")
        for k, v in gs.items():
            keys.add(k)
            if k in GROUP_KEYS and isinstance(v, dict):
                keys.update(f"{k}.{sub}" for sub in v)
    return keys


def _split(key: str) -> tuple[str, str | None]:
    top, _, sub = key.partition(".")
    return top, (sub or None)


def test_every_engine_key_is_registered(scan) -> None:
    offenders = []
    for key, sites in scan.items():
        top, sub = _split(key)
        locs = sites["r"][:1] + sites["w"][:1]
        if top == "[*]":
            continue
        if top not in STATE_KEYS:
            offenders.append(f"{key}  ({locs})")
        elif sub is not None and sub != "[*]" and top in GROUP_KEYS and sub not in GROUP_KEYS[top]:
            offenders.append(f"{key}  ({locs})")
    assert not offenders, "unregistered state keys (alias? add to state.STATE_KEYS):\n" + "\n".join(
        sorted(offenders)
    )


def test_constant_key_writers_match() -> None:
    for key, constant in _CONSTANT_KEY_WRITERS.items():
        assert key == constant and key in STATE_KEYS


def test_every_read_key_has_a_writer(scan, created) -> None:
    def written(key: str) -> bool:
        top, _ = _split(key)
        if scan.get(key, {}).get("w") or key in created or key in _CONSTANT_KEY_WRITERS:
            return True
        return top in DYNAMIC_WRITE_GROUPS and bool(scan.get(f"{top}.[*]", {}).get("w"))

    no_writer = [
        f"{key}  (read at {sites['r'][0]})"
        for key, sites in scan.items()
        if sites["r"] and "[*]" not in key and not written(key)
    ]
    assert not no_writer, "state keys read but never written:\n" + "\n".join(sorted(no_writer))


def test_registry_has_no_dead_entries(scan, created) -> None:
    used = {_split(k)[0] for k in scan} | {_split(k)[0] for k in created}
    used |= set(_CONSTANT_KEY_WRITERS)
    dead = sorted(set(STATE_KEYS) - used)
    assert not dead, f"registered keys nothing reads or writes: {dead}"


# ---------------------------------------------------------------------------
# Handler lint: no raw state reads inside handlers
# ---------------------------------------------------------------------------

_HANDLER_DECORATORS = {"register", "register_dollars", "register_consumable"}
_STATE_PARAM_NAMES = {"gs", "game_state"}


def _is_ctx_game_state(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "game_state"


def _state_expr(node: ast.AST, names: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in names
    if _is_ctx_game_state(node):
        return True
    if isinstance(node, ast.BoolOp):  # ``ctx.game_state or {}``
        return any(_state_expr(v, names) for v in node.values)
    return False


def _handler_violations(fn: ast.FunctionDef, path: Path) -> list[str]:
    names = {a.arg for a in fn.args.args + fn.args.kwonlyargs if a.arg in _STATE_PARAM_NAMES}
    for node in ast.walk(fn):  # locals bound to state
        if isinstance(node, ast.Assign) and _state_expr(node.value, names):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    out = []
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and _state_expr(node.value, names)
        ):
            out.append(f"{path.name}:{node.lineno} {fn.name}: raw state subscript read")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _state_expr(node.func.value, names)
        ):
            out.append(f"{path.name}:{node.lineno} {fn.name}: raw state .get() read")
    return out


def _handlers(tree: ast.Module) -> list[ast.FunctionDef]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if isinstance(target, ast.Name) and target.id in _HANDLER_DECORATORS:
                    found.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "Tag":
            found += [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "apply"]
    return found


def test_handlers_do_not_read_raw_state() -> None:
    violations = []
    n_handlers = 0
    for rel in ("jokers.py", "consumables.py", "tags.py"):
        path = ROOT / "jackdaw/engine" / rel
        handlers = _handlers(ast.parse(path.read_text()))
        n_handlers += len(handlers)
        for fn in handlers:
            violations += _handler_violations(fn, path)
    assert n_handlers > 150, "handler discovery broke (decorator renamed?)"
    assert not violations, "read through jackdaw.engine.read / ctx.game:\n" + "\n".join(violations)


def test_lint_catches_a_raw_read() -> None:
    src = (
        "@register('j_x')\n"
        "def h(card, ctx):\n"
        "    gs = ctx.game_state or {}\n"
        "    return gs.get('dollars')\n"
    )
    fn = _handlers(ast.parse(src))[0]
    assert _handler_violations(fn, Path("x.py"))
