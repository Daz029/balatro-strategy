"""Player profile — the engine supports exactly one: ALL-UNLOCKED, all-discovered.

Balatro's pools consult the player's save profile (``unlocked`` /
``discovered`` on ``G.P_CENTERS``; ``common_events.lua:1988``, tag
``requires``). The engine does not model profile progression (audit D44):
every run assumes a profile with everything unlocked and discovered, which is
what pool filtering hard-codes (``jackdaw.engine.pools``). There is no profile
state in the game-state dict.

Anything that brings a profile in from outside (a real save, a bridge) must
pass it through :func:`require_full_profile`, which raises
:class:`~jackdaw.engine.scope.UnsupportedGameModeError` for any other profile
instead of silently running a different game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jackdaw.engine.data.prototypes import BACKS, TAGS
from jackdaw.engine.scope import UnsupportedGameModeError


def _all_profile_keys() -> frozenset[str]:
    """Every key a profile can unlock or discover (centers, tags, backs)."""
    from jackdaw.engine.data.prototypes import _load_json

    return frozenset(_load_json("centers.json")) | frozenset(TAGS) | frozenset(BACKS)


@dataclass
class Profile:
    """A save profile's unlock / discovery sets."""

    discovered: set[str] = field(default_factory=set)
    unlocked: set[str] = field(default_factory=set)


def default_profile() -> Profile:
    """The only supported profile: everything unlocked and discovered."""
    keys = set(_all_profile_keys())
    return Profile(discovered=set(keys), unlocked=set(keys))


def require_full_profile(profile: Profile) -> None:
    """Raise unless *profile* has every key unlocked AND discovered."""
    keys = _all_profile_keys()
    missing_unlock = keys - profile.unlocked
    missing_discover = keys - profile.discovered
    if missing_unlock or missing_discover:
        raise UnsupportedGameModeError(
            "only an all-unlocked, all-discovered profile is supported "
            f"({len(missing_unlock)} locked, {len(missing_discover)} undiscovered; "
            f"e.g. {sorted(missing_unlock | missing_discover)[:3]})"
        )
