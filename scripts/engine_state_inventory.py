"""Inventory game-state fields: what Lua reads/writes vs what the engine stores.

Scans the vanilla Lua source for every ``G.GAME.<path>`` read and write, and
AST-scans ``jackdaw/engine`` for every game-state dict key read and write
(resolving ``gs[...]`` / ``gs.get(...)`` / ``gs.setdefault(...)`` chains and the
``cr`` / ``rr`` aliases), plus the keys ``init_game_object`` initialises.
Prints one row per Lua path with read/write counts on both sides, flagged:

  MISSING    Lua uses it; the engine never reads, writes or initialises it.
  NO-WRITER  Lua writes it at runtime; the engine only initialises or reads it.

followed by the engine-only keys (renames, mirrors, caches) and the engine keys
read with no writer anywhere.

This is a coverage aid, not a verdict. Many MISSING rows are UI/profile
fields, and some are renames or live on another object (``G.GAME.blind`` is a
``Blind``, ``G.GAME.hands`` is ``HandLevels``). The curated classification is in
``docs/engine_fixing_plan_2026-10-06.md`` Part 1b. Reads through local
variables (``x = gs["a"]; x["b"]``) beyond the known aliases are not resolved.

Usage::

    uv run python scripts/engine_state_inventory.py --lua-src <path to balatro_source/Balatro>
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

ROOT_NAMES = {"gs", "game_state", "state", "_gs", "run_state"}
ALIASES = {
    "cr": "current_round",
    "current_round": "current_round",
    "rr": "round_resets",
    "round_resets": "round_resets",
    "mods": "modifiers",
    "modifiers": "modifiers",
    "sp": "starting_params",
    "starting_params": "starting_params",
    "rb": "round_bonus",
    "round_bonus": "round_bonus",
}
MUTATORS = {
    "append",
    "extend",
    "insert",
    "remove",
    "pop",
    "clear",
    "update",
    "sort",
    "add",
    "discard",
    "popitem",
    "reverse",
}
LUA_SKIP_DIRS = ("localization", "engine")
_INIT_FUNCS = ("init_game_object", "get_starting_params")
_LUA_PAT = re.compile(r"G\.GAME((?:\.[A-Za-z_]\w*)+)((?:\s*\[[^\]\n]*\])*)(\s*=(?!=))?")


def scan_lua(src: Path, depth: int) -> dict[str, dict[str, list[str]]]:
    res: dict[str, dict[str, list[str]]] = {}
    for f in sorted(src.rglob("*.lua")):
        rel = f.relative_to(src)
        if rel.parts and rel.parts[0] in LUA_SKIP_DIRS:
            continue
        for ln, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
            code = line.split("--", 1)[0]
            for m in _LUA_PAT.finditer(code):
                segs = m.group(1).strip(".").split(".")
                kind = "w" if m.group(3) else "r"
                ent = res.setdefault(".".join(segs[:depth]), {"r": [], "w": []})
                ent[kind].append(f"{rel}:{ln}")
    return res


_SELF_WRITE = re.compile(r"self\.([A-Za-z_]\w*)\s*=(?!=)")
_ABILITY_KEY = re.compile(r"ability\.([A-Za-z_]\w*)")


def scan_lua_self_fields(lua_file: Path, statement_start: bool = False) -> set[str]:
    """Fields assigned as ``self.<x> = ...`` in one Lua class file.

    ``statement_start`` keeps only assignments that begin a line, which drops
    the ``self.x = ...`` writes nested in UI-table constructors (card.lua).
    """
    out: set[str] = set()
    for line in lua_file.read_text(errors="replace").splitlines():
        code = line.split("--", 1)[0]
        if statement_start:
            m = re.match(r"\s*self\.([A-Za-z_]\w*)\s*=(?!=)", code)
            if m:
                out.add(m.group(1))
        else:
            out.update(_SELF_WRITE.findall(code))
    return out


def scan_lua_ability_keys(src: Path) -> set[str]:
    """Every ``ability.<key>`` referenced in the game Lua (UI files included)."""
    out: set[str] = set()
    for f in sorted(src.rglob("*.lua")):
        rel = f.relative_to(src)
        if rel.parts and rel.parts[0] in LUA_SKIP_DIRS:
            continue
        for line in f.read_text(errors="replace").splitlines():
            out.update(_ABILITY_KEY.findall(line.split("--", 1)[0]))
    return out


def _key(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "[*]"


def _path(node: ast.AST) -> list[str] | None:
    if isinstance(node, ast.Name):
        if node.id in ROOT_NAMES:
            return []
        if node.id in ALIASES:
            return [ALIASES[node.id]]
        return None
    if (
        isinstance(node, ast.Attribute)
        and node.attr in ("_gs", "gs", "game_state")
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return []
    if isinstance(node, ast.Subscript):
        base = _path(node.value)
        if base is None:
            return None
        return base + [_key(node.slice)]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("get", "setdefault")
        and node.args
    ):
        base = _path(node.func.value)
        if base is None:
            return None
        return base + [_key(node.args[0])]
    return None


def scan_python(paths: list[Path], depth: int) -> dict[str, dict[str, list[str]]]:
    res: dict[str, dict[str, list[str]]] = {}

    def rec(p: list[str] | None, kind: str, loc: str) -> None:
        if p:
            res.setdefault(".".join(p[:depth]), {"r": [], "w": []})[kind].append(loc)

    for root in paths:
        for f in [root] if root.is_file() else sorted(root.rglob("*.py")):
            tree = ast.parse(f.read_text())
            for node in ast.walk(tree):
                loc = f"{f}:{getattr(node, 'lineno', 0)}"
                if isinstance(node, ast.Subscript):
                    write = isinstance(node.ctx, (ast.Store, ast.Del))
                    rec(_path(node), "w" if write else "r", loc)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    if attr in ("get", "setdefault"):
                        write = attr == "setdefault" and len(node.args) > 1
                        rec(_path(node), "w" if write else "r", loc)
                    elif attr in MUTATORS:
                        rec(_path(node.func.value), "w", loc)
                elif isinstance(node, ast.AugAssign):
                    rec(_path(node.target), "w", loc)
    return res


def init_keys(run_init: Path, depth: int) -> set[str]:
    """Keys created by the dict literals in init_game_object / get_starting_params."""
    keys: set[str] = set()

    def walk(d: ast.Dict, prefix: list[str]) -> None:
        for k, v in zip(d.keys, d.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                p = prefix + [k.value]
                keys.add(".".join(p[:depth]))
                if isinstance(v, ast.Dict) and len(p) < depth:
                    walk(v, p)

    for fn in ast.parse(run_init.read_text()).body:
        if isinstance(fn, ast.FunctionDef) and fn.name in _INIT_FUNCS:
            prefix = ["starting_params"] if fn.name == "get_starting_params" else []
            for n in ast.walk(fn):
                if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
                    walk(n.value, prefix)
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lua-src", type=Path, required=True)
    ap.add_argument("--engine", type=Path, default=Path("jackdaw/engine"))
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()

    lua = scan_lua(args.lua_src, args.depth)
    eng = scan_python([args.engine], args.depth)
    init = init_keys(args.engine / "run_init.py", args.depth)

    def counts(d: dict, k: str) -> tuple[int, int]:
        e = d.get(k, {"r": [], "w": []})
        return len(e["r"]), len(e["w"])

    print(f"{'Lua G.GAME path':44} {'Lr':>4} {'Lw':>4} | {'Pr':>4} {'Pw':>4} init")
    for k in sorted(lua):
        lr, lw = counts(lua, k)
        pr, pw = counts(eng, k)
        flag = ""
        if pr == pw == 0 and k not in init:
            flag = "MISSING"
        elif lw and not pw:
            flag = "NO-WRITER"
        print(f"{k:44} {lr:4} {lw:4} | {pr:4} {pw:4} {'Y' if k in init else ' ':>4}  {flag}")

    print("\nEngine-only keys (not a G.GAME path): renames, mirrors, caches")
    for k in sorted((set(eng) | init) - set(lua)):
        pr, pw = counts(eng, k)
        flag = "  NO-WRITER" if not pw and k not in init else ""
        print(f"{k:44} r={pr:3} w={pw:3} init={'Y' if k in init else ' '}{flag}")


if __name__ == "__main__":
    main()
