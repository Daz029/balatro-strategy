"""Static guardrails for the Phase 3 card lifecycle boundary."""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_SCANNED_ROOTS = (_ROOT / "jackdaw" / "engine", _ROOT / "jackdaw" / "env")
_EXCLUDED_MODULES = {"card.py", "card_factory.py", "lifecycle.py"}
_CARD_ATTRIBUTES = {
    "ability",
    "added_to_deck",
    "base",
    "card_key",
    "center_key",
    "debuff",
    "edition",
    "eternal",
    "perish_tally",
    "perishable",
    "rental",
    "seal",
}
_GS_SETTERS = {"change_rank", "change_suit", "enhance", "set_ability", "set_base"}

# Keys deliberately use the containing function and operation, never a line
# number, so harmless source movement cannot silently invalidate an exception.
_ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "jackdaw/engine/play_ordering.py",
        "fast_clone_card.ability",
    ): "Solver-only scoring clone; it is not an owned card lifecycle event.",
    (
        "jackdaw/engine/state.py",
        "migrate_state.added_to_deck",
    ): "Legacy-save normalization restores an already-applied marker without replaying passives.",
    (
        "jackdaw/env/shop_run_adapter.py",
        "_advance.copy.deepcopy",
    ): "Observer snapshot clones the complete game-state dict, not an individual card.",
}


def _function_names(tree: ast.AST) -> dict[ast.AST, str]:
    names: dict[ast.AST, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = ["<module>"]

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            names[node] = self.stack[-1]
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def generic_visit(self, node: ast.AST) -> None:
            names.setdefault(node, self.stack[-1])
            super().generic_visit(node)

    Visitor().visit(tree)
    return names


def _assignment_targets(node: ast.AST) -> list[ast.expr]:
    def flatten(target: ast.expr) -> list[ast.expr]:
        if isinstance(target, (ast.Tuple, ast.List)):
            return [item for element in target.elts for item in flatten(element)]
        if isinstance(target, ast.Starred):
            return flatten(target.value)
        return [target]

    if isinstance(node, ast.Assign):
        return [item for target in node.targets for item in flatten(target)]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        return flatten(node.target)
    return []


def test_card_lifecycle_boundary() -> None:
    violations: list[str] = []
    used_allowlist: set[tuple[str, str]] = set()

    for root in _SCANNED_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if path.name in _EXCLUDED_MODULES:
                continue
            module = path.relative_to(_ROOT).as_posix()
            tree = ast.parse(path.read_text(), filename=str(path))
            owners = _function_names(tree)

            def record(node: ast.AST, operation: str) -> None:
                key = (module, f"{owners[node]}.{operation}")
                if key in _ALLOWLIST:
                    used_allowlist.add(key)
                    return
                violations.append(f"{module}:{node.lineno}: {key[1]}")

            # Resolve aliases (``import copy as _copy``, ``from copy import
            # deepcopy as dc``) so a renamed import cannot dodge the check.
            copy_modules = {"copy"}
            copy_funcs: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "copy":
                            copy_modules.add(alias.asname or "copy")
                elif isinstance(node, ast.ImportFrom) and node.module == "copy":
                    for alias in node.names:
                        if alias.name in {"copy", "deepcopy"}:
                            copy_funcs[alias.asname or alias.name] = alias.name

            for node in ast.walk(tree):
                for target in _assignment_targets(node):
                    if isinstance(target, ast.Attribute) and target.attr in _CARD_ATTRIBUTES:
                        record(node, target.attr)

                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name) and func.id == "Card":
                    record(node, "Card")
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id in copy_modules
                    and func.attr in {"copy", "deepcopy"}
                ):
                    record(node, f"copy.{func.attr}")
                if isinstance(func, ast.Name) and func.id in copy_funcs:
                    record(node, f"copy.{copy_funcs[func.id]}")
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr in _GS_SETTERS
                    and not any(keyword.arg == "gs" for keyword in node.keywords)
                ):
                    record(node, func.attr)

    stale = set(_ALLOWLIST) - used_allowlist
    assert not violations and not stale, (
        "Lifecycle lint violations:\n"
        + "\n".join(violations)
        + ("\nStale allowlist entries:\n" + "\n".join(map(str, sorted(stale))) if stale else "")
    )
