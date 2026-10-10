# Engine Phase 5, ticket P5-1: oracle source path + the S4-a event-timing sweep

Branch `engine-phase1-state`, base `2bdbcc2` (Phase 4 exit). Read first:
`docs/engine_fixing_plan_2026-10-06.md` (S4, Phase 0 item 4, Phase 4 STATUS
and its "Residuals, owned by Phase 5", Phase 5), sections 5, 8 and 9 of
`docs/engine_class_design_2026-10-06.md`, and the audit entries C01, C03, C05,
C09, C10, C11, D12, D14, D18, D22, D28, D29, D30, D31, D32 in
`docs/engine_audit.md`.

Lua source (1.0.1o, read-only, gitignored, NOT in the repo):
`~/Code/Code/balatro-strategy/balatro_source/Balatro` (has `functions/`,
`card.lua`, `blind.lua`, `game.lua`, `engine/event.lua`, ...).

This ticket changes NO engine behavior. It is the instrument the rest of
Phase 5 is specified against.

## 1. Oracle source path (Phase 0 item 1, still owed)

The Lua oracles (`scripts/lua_scoring_oracle.lua`, `lua_hand_eval_oracle.lua`,
`extract_prototypes.lua`) read `project_root .. "balatro_source/..."`, which
does not exist in this checkout, so `tests/engine/test_hand_eval_oracle.py::TestLiveOracle`
ERRORS and other oracle tests skip.

- One resolution rule, shared: env var `BALATRO_SOURCE` if set, else
  `<project_root>/balatro_source` if it exists (old layout, keep working),
  else `~/Code/Code/balatro-strategy/balatro_source/Balatro`. Implement it
  once in Lua (a small `scripts/lua_source_path.lua` the oracles `dofile`/
  `require`) and once in Python (`tests/_lua_source.py` or a conftest
  fixture) — the Python side passes the resolved path to the Lua scripts
  (env var or argv), so they agree. `tests/engine/test_state_map.py:48`
  uses the same helper.
- Tests that need the source or LuaJIT/lupa SKIP with a clear reason when
  either is missing (never error). Here both exist: after this change,
  `uv run pytest tests/engine -q` must show the previously-erroring/skipped
  oracle tests PASSING. Report the before/after skip+error counts. If a
  formerly-skipped oracle test now FAILS on a real mismatch, do not fix the
  engine: mark it `xfail(strict=True, reason="<finding id or NEW: ...>")`
  and list it in the report.

## 2. The S4-a sweep (Phase 0 item 4) -> `docs/engine_s4a_event_timing.md`

This is the spec for Phase 5's money ledger and the observation points of
the transition oracle. Read the Lua source, not our engine, as the
authority; cite `file:line` for every row. Be exhaustive within scope;
"not found" is a valid cell, a guess is not.

Scope: everything that runs inside these Lua functions, including the
joker/seal/enhancement/blind code they call:
`G.FUNCS.play_cards_from_highlighted`, `G.FUNCS.evaluate_play`,
`G.FUNCS.discard_cards_from_highlighted` (incl. the Hook path),
`G.FUNCS.draw_from_deck_to_hand` / `draw_card`, `new_round`,
`Blind:set_blind`, `Blind:press_play`, `Blind:debuff_hand`,
`Blind:drawn_to_hand`, `Blind:stay_flipped`, `Blind:defeat`, `end_round`,
`G.FUNCS.evaluate_round`, `G.FUNCS.cash_out`.

### 2a. Event model

First, document precisely how `G.E_MANAGER` / `Event` (`engine/event.lua`)
behave as far as ordering is concerned: queues (`base`, `other`...),
`trigger` kinds (`immediate`, `after`, `before`, `ease`, `condition`),
`blocking` / `blockable`, and what order events queued from inside another
event's `func` run in relative to events queued earlier. State the
consequence as a rule we can implement synchronously (e.g. "all events
queued during evaluate_play run, FIFO, after it returns and before X").
Note every place where nesting changes that order (an event queuing
events).

### 2b. Money table

One row per money READ and per money WRITE in scope:

| Site (Lua file:line) | Who (joker/seal/blind/back/system) | Read or write | View read: `dollars` (committed) or `dollars + dollar_buffer` | Write kind: `ease_dollars(x)` queued / `ease_dollars(x, true)` instant / direct assignment | Touches `dollar_buffer`? (+ where it is zeroed) | When the queued write commits (which flush point) |

Must include at least: Gold seal, Gold card (held, end of round), Lucky
card money, Golden Ticket, Business Card, Rough Gem, Reserved Parking,
Faceless, Mail-In Rebate, Trading Card, To Do List, Matador, Vagabond, Bull,
Bootstraps, Satellite/Rocket/Golden/Cloud 9/Delayed Gratification
(`calc_dollar_bonus`), Egg/Gift Card (sell value), The Ox, The Tooth,
Golden Needle / discard cost, interest, blind reward, hands/discards
bonus, rental, Credit Card floor (`bankrupt_at`) checks, and any
`G.GAME.dollars` read used for legality (`can_play`, buying is out of scope).
Confirm or correct the Vagabond analysis in plan S4 / design section 5
(Vagabond reads committed `dollars`, so it does NOT see a queued Gold seal
payout but DOES see The Ox's instant drain).

### 2c. Counter / flag table

Same columns adapted (read/write, which value view, instant vs inside a
queued event, flush point) for: `G.GAME.hands_played`,
`current_round.hands_played`, `hands_left`, `discards_left`,
`discards_used`, `current_round.most_played_poker_hand` (when written),
`G.GAME.last_hand_played`, hand `played` / `played_this_round` /
`visible`, `G.GAME.chips` (when the round total is committed vs read by the
win check), `blind.triggered`, `blind.prepped`, `blind.hands` /
`hands_used`, `card.ability.played_this_ante` (and where Lua clears it),
`card.facing` (stay_flipped), `joker_buffer` / `consumeable_buffer`
(where incremented and zeroed), `G.GAME.round_scores`, `unused_discards`,
perishable/rental tallies, and Mr. Bones' `saved` / game-over check.

### 2d. Ordered step lists

For each scoped Lua function, a numbered list of its steps in execution
order, with the synchronous part and each queued event marked, `file:line`
per step. This is what P5-2..P5-4 port line by line, so the granularity is
"one observable state change or one dispatch per step". In particular pin:
where `hands_played` is incremented relative to `evaluate_play`; where the
played cards move to discard; where the redraw happens; where `end_round`
is reached from play; the order of the game-over check, the end_of_round
joker pass (incl. held-card repetitions for Gold card / Blue seal and the
Red seal / Mime retriggers), rental, perishable, `calc_dollar_bonus`,
`Blind:defeat`, Pillar marker reset, `most_played_poker_hand`, and
interest.

### 2e. Python mapping

For each step and each table row, the current Python site
(`jackdaw/engine/...:line`) and whether it matches, is out of order (which
finding), or is missing. Close with a short "Ledger spec" section: the
minimal `MoneyLedger` API and the exact flush points Phase 5 must
implement, derived from 2a-2d (design section 5 is the draft; correct it
where the source disagrees).

## Rules

- Do not change engine behavior. Only the path helper, the oracle scripts'
  path resolution, tests' skip logic, and the new doc.
- `uv run pytest tests/engine -q` stays green apart from new strict xfails.
  `uv run ruff check` clean on touched files.
- Do not commit. Final message: what changed (files), oracle test counts
  before/after, any new mismatches found, and the top 10 surprises from the
  sweep (places where the Python engine differs from the source in ways NOT
  already listed in the audit findings above), each with `file:line` on
  both sides.
