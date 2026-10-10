# Engine Phase 5, ticket P5-O: headless Lua transition oracle

Branch `engine-p5-oracle` (off `engine-phase1-state` @ `2bdbcc2`). Runs in
parallel with the Phase 5 engine tickets, so touch ONLY new files (plus a
`pyproject`/conftest marker registration if you need one). Read first:
`docs/engine_fixing_plan_2026-10-06.md` S6 item 2 and Phase 5,
`docs/engine_class_design_2026-10-06.md` sections 5, 8, 9, 11, and the
existing `scripts/lua_scoring_oracle.lua` (a STRIPPED reimplementation with
an event manager that runs events immediately — exactly what this ticket
must not do).

Lua source (1.0.1o, read-only, gitignored): env var `BALATRO_SOURCE`, else
`~/Code/Code/balatro-strategy/balatro_source/Balatro`. LuaJIT is at
`/opt/homebrew/bin/luajit`; `lupa` 2.6 is installed in the uv env.
Tests needing either SKIP with a reason when absent.

## Goal

Phase 5 exit criterion: "Oracle observation-point tests go green". Build the
harness that makes that measurable: feed the SAME controlled state to the
real Lua game code and (in later tickets) to our engine, run one transition,
compare state at each observation point.

## 1. The Lua side: `scripts/lua_transition_oracle/`

- Load the REAL source files, unmodified, with `dofile`/`loadstring` on the
  file text: at least `engine/object.lua`, `engine/event.lua` (the real
  `EventManager` and `Event`), `functions/misc_functions.lua`,
  `functions/common_events.lua`, `functions/state_events.lua`,
  `functions/button_callbacks.lua` (for `G.FUNCS.cash_out`,
  `evaluate_round` if it lives there — find it), `card.lua`, `blind.lua`,
  `back.lua`, `game.lua` only if you can load what you need (e.g.
  `P_CENTERS`, `init_game_object`) without the LOVE runtime; otherwise
  extract the tables the way `scripts/extract_prototypes.lua` does.
- Stub ONLY the LOVE2D / UI / animation / sound / save / unlock surface
  (`Moveable`, `Sprite`, `CardArea` drawing, `play_sound`,
  `attention_text`, `card_eval_status_text`, `juice_card`,
  `update_hand_text`, `G.ROOM`, `love.*`, ...). Keep every gameplay
  function real. A minimal `CardArea` stand-in is fine if the real one
  needs graphics, but it must keep the real `emplace` / `remove_card` /
  `cards` ordering semantics (deck `emplace` inserts at front, etc. — read
  `cardarea.lua`) and `config.card_limit` / `highlighted`.
- Events: use the REAL `EventManager`. Drive it with a fake clock:
  advance `G.TIMERS.REAL/TOTAL` and call `G.E_MANAGER:update(dt)` in a loop
  until every queue is empty (with an iteration cap that errors loudly).
  Delays (`delay = x`) and `ease` events must complete under the fake clock;
  `condition` events must not deadlock (if one waits on a UI value, stub
  that value, and document each such stub).
- Observation points: a hook that snapshots the observable state to a Lua
  table. Capture at least: (a) immediately after the synchronous call
  returns, before draining (this is where "queued vs instant" shows), (b)
  after each event's `func` completes (a list, with a short label = the
  Lua source line of the function that created the event if you can get it
  via `debug.getinfo`, else an index), (c) after full drain. Snapshot
  fields: `dollars`, `dollar_buffer`, `joker_buffer`, `consumeable_buffer`,
  `chips`, `current_round` (hands_left, discards_left, hands_played,
  discards_used, most_played_poker_hand), `G.GAME.hands_played`,
  `last_hand_played`, per-hand `played` / `played_this_round` / `level`,
  blind (`name`, `chips`, `triggered`, `disabled`, `prepped`), and for each
  area (hand, play, deck, discard, jokers, consumeables) the ordered list of
  cards as `{id, key, base(rank,suit), ability(name, effect, perma_bonus,
  played_this_ante, extra/mult/x_mult/...), edition, seal, debuff, facing}`.
  Give each card a stable `id` from the scenario so both engines can be
  matched card by card.
- Scenario input: JSON (read via a small pure-Lua JSON decoder you add, or
  passed through lupa as a table). Schema below. The harness builds
  `G.GAME` via the real `init_game_object` when possible, then overrides
  fields from the scenario, builds real `Card` objects via `Card(...)` /
  `create_card` with the given centers, editions, seals, abilities, and
  places them in areas in the given order.
- RNG: seed `G.GAME.pseudorandom` from the scenario seed exactly as the
  game does (`G.GAME.pseudorandom.seed`, `hashed_seed`), using the real
  `pseudoseed`/`pseudorandom` from `misc_functions.lua`. Snapshot the
  per-key stream states that were touched (`G.GAME.pseudorandom[key]`).

## 2. Scenario schema + Python API: `jackdaw/oracle/` or `tests/oracle/`

Pick one location (prefer `tests/oracle/` unless something in `jackdaw/`
needs it) and document it.

- `Scenario` (dataclass, JSON-serialisable): seed, stake, back, ante,
  round, dollars, hands_left, discards_left, hand_levels overrides, blind
  (key + overrides), the card areas (each card: id, front key like `H_2`,
  center key like `m_gold`, edition, seal, ability overrides, debuff,
  facing), jokers (key, edition, ability overrides incl. counters),
  consumables, and the ACTION: one of `play(indices)`, `discard(indices)`,
  `end_round()`, `set_blind(key)` (new_round), `cash_out()`.
- `run_lua_oracle(scenario) -> OracleTrace` (observations list as above).
  Use lupa (in-process, faster) or `luajit` subprocess + JSON; either is
  fine, document the choice.
- `build_engine_state(scenario) -> gs` that builds the equivalent state
  for OUR engine (`jackdaw.engine`), and `run_engine(scenario) -> trace`
  that applies the action through `jackdaw.engine.game.step()` and returns
  a trace with the SAME snapshot fields (only the final snapshot is
  required on the engine side for now; Phase 5 will add internal points).
  Map our field names to the snapshot schema in ONE function so the
  comparison is field-for-field.
- `compare(lua_trace, engine_trace, fields=...) -> list[Diff]` with a
  readable diff.

## 3. Tests: `tests/engine/test_transition_oracle.py`

- Harness smoke tests (must PASS): a plain High Card play on a fresh
  Small Blind; a discard of 2 cards; `end_round` after a winning play.
  Each asserts the Lua trace is internally sane (chips > 0, cards moved to
  the right areas, events drained) — these test the harness, not our
  engine.
- Event-timing sanity (must PASS, proves the harness models queued money):
  the audit's C05 scenario — $4, Bull, play a single Gold-sealed 2.
  Expected from the source: Bull reads `dollars + dollar_buffer` and the
  play scores **21** chips (7 + 2x7); at observation point (a)
  `dollar_buffer == 3` and `dollars == 4`; after drain `dollars == 7`,
  `dollar_buffer == 0`. If the real code disagrees with these numbers,
  trust the code, and report it.
- Differential tests vs OUR engine, one per Phase 5 finding where a
  scenario can express it: C01, C03, C05, C10, C11, D12, D18, D22, D28,
  D29, D30, D31, D32, plus C09 (House/Mark facing) via `set_blind`. Each
  marked `xfail(strict=True, reason="<finding>")` — they are the red tests
  Phase 5 turns green. Use the audit's "Reproduced" numbers as the
  scenarios where given. Before marking one xfail, confirm the Lua side
  produces the source-faithful value the audit claims; if the audit is
  wrong, say so in the report and encode what Lua actually does.
- Also add a handful (>=5) of differential tests on scenarios we expect to
  ALREADY match (plain pair, flush with a +mult joker, a discard with no
  jokers, a Small Blind end_round with interest) — non-xfail — so the
  comparison itself is validated.

## Rules

- No changes to `jackdaw/engine/*` or existing tests. New files only.
- `uv run pytest tests/engine -q` green (new xfails strict). `uv run ruff
  check` clean on new Python files.
- Runtime: the whole new test file under ~30 s.
- Do not commit. Final message: files added, how the Lua environment is
  stubbed (list every stub that touches gameplay state, with why), the
  event-loop drain design, test counts, every place the audit's claimed
  source behavior turned out wrong, and anything you could not load from
  the real source (with the reason).
