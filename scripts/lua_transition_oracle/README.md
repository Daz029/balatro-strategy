# Headless Lua transition oracle

`tests.oracle.run_lua_oracle` embeds LuaJIT through `lupa`, loads Balatro
1.0.1o from `BALATRO_SOURCE` (falling back to
`~/Code/Code/balatro-strategy/balatro_source/Balatro`), and runs the real
`Card`, `Blind`, `Back`, `Event`, `EventManager`, scoring, round, and button
callback code. Scenarios and traces are ordinary JSON-compatible dataclasses
defined in `tests/oracle/transition.py`.

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
