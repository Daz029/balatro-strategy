# Headless Lua transition oracle

`tests.oracle.run_lua_oracle` embeds Lua 5.1 through `lupa.lua51`, loads Balatro
1.0.1o from `BALATRO_SOURCE` (falling back to
`~/Code/Code/balatro-strategy/balatro_source/Balatro`), and runs the real
`Card`, `Blind`, `Back`, `Event`, `EventManager`, scoring, round, and button
callback code. Scenarios and traces are ordinary JSON-compatible dataclasses
defined in `tests/oracle/transition.py`.

Lua 5.1 preserves Balatro/LuaJIT's all-double number model and legacy library
surface. It is not LuaJIT, however: table hash layout can differ, so `pairs`
iteration order is not assumed portable. Scenarios whose gameplay result can
depend on `pairs` must avoid ties/order dependence or be cross-checked with the
`/opt/homebrew/bin/luajit` subprocess.

The bootstrap replaces LÖVE rendering objects, UI widgets, sound, animation,
unlock/profile persistence, and geometry only. Its `CardArea` stand-in keeps
the source's observable card ordering: decks insert at the front, other areas
append, and deck/discard default removal is from the end. `update_hand_text`
retains the source's `current_hand` state mirror because scoring reads it.

The event manager is the source implementation. A fake 60 Hz clock advances
`G.TIMERS.REAL` and `G.TIMERS.TOTAL`; every queue is updated until empty.
Drain is capped at 20,000 updates and errors with the pending-event count.
Each completed event is snapshotted by a wrapper around `Event:handle`; labels
are the source filename and definition line obtained from `debug.getinfo`.
