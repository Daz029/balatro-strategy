"""Shared Balatro source-tree resolution for Lua-backed tests."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_lua_source(project_root: Path | None = None) -> Path:
    """Resolve the read-only 1.0.1o source tree used by Lua oracles."""
    configured = os.environ.get("BALATRO_SOURCE")
    if configured:
        return Path(configured).expanduser()

    root = project_root or Path(__file__).resolve().parents[1]
    checkout_source = root / "balatro_source"
    if (checkout_source / "card.lua").is_file():
        return checkout_source

    return Path.home() / "Code/Code/balatro-strategy/balatro_source/Balatro"


def lua_source_missing_reason(source: Path) -> str | None:
    """Return a clear skip reason when *source* is not a Balatro tree."""
    sentinel = source / "card.lua"
    if sentinel.is_file():
        return None
    return f"Balatro Lua source not found at {source} (set BALATRO_SOURCE)"
