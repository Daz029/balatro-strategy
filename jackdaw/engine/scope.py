"""Game modes the engine does not implement, and the error they raise.

Decision 2026-10-06 (D-scope), confirmed 2026-10-12: anything the engine does
not faithfully implement RAISES instead of silently running a different game.

- **Challenges** (audit C15, D43 bans, D44 voucher timing): ``initialize_run``
  rejects any ``challenge``.
- **Endless** (audit D49): play past the win-ante boss is rejected, and blind
  amounts past ante 16 assert (Lua's doubles diverge from Python ints there).
- **Profiles** (audit D44): every run assumes an ALL-UNLOCKED, all-discovered
  profile; any other profile is rejected.
"""

from __future__ import annotations


class UnsupportedGameModeError(NotImplementedError):
    """Raised when a run would enter a mode the engine does not implement."""
