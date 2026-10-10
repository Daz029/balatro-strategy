# Engine Phase 5, ticket P5-2: money ledger + line-faithful play / evaluate_play

Branch `engine-phase1-state`, after P5-1. Read first:
`docs/engine_s4a_event_timing.md` (P5-1's sweep — THE spec for this ticket:
its ordered step lists 2d for `play_cards_from_highlighted` and
`evaluate_play`, its money table 2b and its "Ledger spec"),
`docs/engine_fixing_plan_2026-10-06.md` (S4, Phase 4 STATUS incl.
"Residuals, owned by Phase 5", Phase 5), design doc sections 5 and 8, and
audit entries C01, C03, C05, D12, D29. Lua source:
`~/Code/Code/balatro-strategy/balatro_source/Balatro` (read-only).
Where this ticket and the sweep disagree, the Lua source wins; say so in the
report.

## Scope

`game.py::_handle_play_hand` (+ `_press_play`) and `scoring.py::score_hand`
become line-faithful ports of `G.FUNCS.play_cards_from_highlighted` and
`G.FUNCS.evaluate_play`, with a `# L: state_events.lua:NNN` citation per
step. OUT of scope here (later tickets, do not touch beyond what the play
path needs to keep working): the redraw/draw function (P5-3), discard and
The Hook (P5-3), `end_round` / Mr. Bones / `Blind.defeat` (P5-4). Keep
`score_hand`'s `saved` / Mr. Bones block exactly as it is (C11 is P5-4).

## 1. Money ledger (pass-local, the Phase 4 precedent)

Lua's `dollar_buffer` is zero at every stable decision point, exactly like
`joker_buffer` / `consumeable_buffer`, which Phase 4 deliberately kept on
the pass's `EffectQueue` (NOT in `gs`) so the solver — which scores cloned
cards against the LIVE `gs` and never applies — cannot leak a pending value.
Do the same for money:

- `EffectQueue` gains the pass's money view: `pending_dollars` (Lua's
  `dollar_buffer` contribution of this pass) and the instant changes made
  during the pass (The Ox's `ease_dollars(-dollars, true)`). Two reads:
  `committed()` = live `gs["dollars"]` with this pass's INSTANT changes;
  `with_buffer()` = `committed() + gs["dollar_buffer"] + pending_dollars`.
  Follow the sweep's section 6 "Ledger specification" exactly — in
  particular: an ordinary `ease` does NOT imply a buffer; only the producers
  that call `G.GAME.dollar_buffer = ... + ret` in Lua (Gold seal / Lucky /
  Golden Ticket / Business / Rough Gem / Reserved Parking — `get_p_dollars`
  and friends; To Do List / Matador per their lines) reserve; Mail, Trading,
  Faceless, Tooth, discard cost, rental and held Gold do not. `EaseDollars`
  gains `instant: bool = False` and `buffered: bool = False` (or two effect
  classes — your call, documented); a buffered effect updates the queue's
  view at EMISSION, its `apply` commits; an instant one changes the
  committed view at emission. Preserve the FIFO order rules in section 6
  (clear-before-commit for `get_p_dollars`, commit-before-clear for To Do
  List / Matador); coalescing is allowed only where you show no money
  observer runs in between.
- Flush points: implement F1 and F2 from the sweep's section 1 table for
  the play path (the `EffectQueue.apply()` calls sit exactly there, with
  F2's internal order: play cards to discard, hand counters, then the
  events emitted inside scoring).
- `read.money_committed` / `read.money_with_buffer` / `StateView.money` /
  `StateView.money_with_buffer` read THROUGH the pass's queue when there is
  one (the context carries `queue`). Every money consumer in scope reads
  the view its Lua line reads (sweep table 2b): Bull and Bootstraps
  `with_buffer`, Vagabond `committed`, and the rest per the table. A test
  per consumer.
- Every handler money payout in a scoring pass (Gold seal `p_dollars`,
  Lucky money, Golden Ticket, Business Card, Rough Gem, Reserved Parking,
  To Do List, Matador, ...) becomes an `EaseDollars` emitted at the
  point Lua emits it, so a later consumer in the same pass sees it in
  `with_buffer`. `ScoreResult.dollars_earned` survives as a derived
  property (sum of the pass's non-instant `EaseDollars`) for existing
  readers (grep `dollars_earned` in `jackdaw/` and `scripts/`); the play
  handler stops adding it separately — `apply_effects` commits.
- `gs["dollar_buffer"]` stays 0 at every stable decision point. Add a test
  that asserts it after every `step()` in the seeded rollout
  (`tests/engine/_rollout.py`).

## 2. `evaluate_play` order (fixes C01, C03, C05, D12, D29)

Port the step list from the sweep. At minimum:

- C01: hand stats (`played`, `played_this_round`, `visible`,
  `last_hand_played`) once, before the debuff branch (already done in
  Phase 4 — keep, and move `gs["last_hand_played"]` into the pipeline if
  Lua writes it there). `hands_played` (round and run) after
  `evaluate_play`, at the queued-event point the sweep pins.
- D29: The Ox fires INSIDE `Blind.debuff_hand` (Lua `blind.lua:562`),
  before card scoring: compares the played hand to
  `current_round.most_played_poker_hand` (the stored value, NOT
  `hand_levels.most_played()`), sets `triggered`, and drains money
  INSTANTLY (`EaseDollars` with `instant=True`, amount = -committed). Delete
  the post-scoring Ox block in `_handle_play_hand`. Money earned by the
  triggering hand must survive.
- C03: delete the `individual_hand_end` context everywhere (scoring phase
  8d, the handler branch, any dispatcher/test references, the
  producer/consumer allowlist if it is listed). Vampire strips
  enhancements and scales in the `before` pass, before the cards score,
  and applies its XMult once in the main pass; clear its temporary guard
  the way Lua does. Audit repro: fresh Vampire + a single Mult-enhanced 2
  scores **7**, not 42.
- D12: the `after` pass runs for blocked (debuffed) hands too; a blocked
  hand returns its result after it. Ice Cream / Seltzer decay on a blocked
  play.
- DNA residual: Lua emplaces DNA's copy into `G.hand` during the before
  pass, so the held-card loop scores it (Phase 4 residual). In the
  pipeline, the copy must join the HELD cards scored in phase 8 for this
  hand — on the live path as a real card in `gs["hand"]` (via lifecycle),
  on the solver path without touching live `gs` (the clone / queue rule
  in Phase 4's solver-isolation test must keep passing). If Lua's timing
  says otherwise, follow Lua and cite it.
- Also from the Phase 4 residuals: The Ox now drains before scoring, so
  verify Matador's main pass sees an Ox trigger (Lua: `blind.triggered`).
- Keep the scoring arithmetic order (design section 8) exactly as it is
  unless the sweep shows a deviation — the scoring oracle
  (`tests/engine/test_scoring_oracle.py`) must stay green.

### Explicitly NOT in this ticket (await a user decision)

- NEW-P5-1-05: Lua sorts the played cards by hand position before scoring
  (`state_events.lua:459-483`); our engine keeps SELECTION order on
  purpose (`best_play_order` / the solver choose scoring order through it).
  Do not change it; leave a `# NEW-P5-1-05` comment at the site.
- NEW-P5-1-06 (HAND_PLAYED phase): no new observable phase is needed in a
  synchronous engine; do not add one.

## 3. `play_cards_from_highlighted` order

Port the sweep's step list: `hands_left` decrement point, per-card stats
(`times_played`, `played_this_ante`), `Blind:press_play` (Hook, Tooth,
Fish, Crimson Heart prep — leave The Hook's discard mechanics for P5-3 but
put the call at the right point), then `evaluate_play`, then the queued
events (played cards to discard, `hands_played += 1`, money commit), then
the win / redraw / game-over branch (keep calling the existing redraw code
for now; P5-3 replaces it).

## 4. Tests (red first)

New `tests/engine/test_play_sequencing.py`, everything through `step()`
(or `score_hand` for pipeline-internal ordering). Each test must FAIL on
the parent commit — check it and say so per test in the report. At least:
C01 (one High Card play records `played == 1`; DNA / Sixth Sense see the
first hand as hand 0 through `step()`), C03 (the 7-chip repro), C05 (the
Bull + Gold seal repro: **21** chips, $7 after; Vagabond committed-view
case incl. the Ox interaction from the plan's S4 note), D12, D29 (Ox with a
hand that BECOMES most-played this play does not trigger; triggering hand's
Gold seal money survives the drain), DNA copy scored as held this hand,
`dollar_buffer == 0` at stable points, and solver isolation with money
consumers (Bull/Bootstraps/Vagabond scored against live `gs` leave
`gs["dollars"]`/`gs["dollar_buffer"]` untouched).

If the transition oracle (`tests/engine/test_transition_oracle.py`, ticket
P5-O, may or may not be merged yet) exists, flip its xfails for the
findings fixed here and keep them strict.

## 5. Performance gate

Phase 4 measured the solver's per-decision wall time on 20 fixed seeds per
stage (stage2, stage3), parent vs child back to back (see Phase 4 STATUS
"Performance" in the plan; find the exact command in git history / docs —
`git log -p --all -S "63.4"` — and reuse it). This ticket's pipeline
changes sit on the solver hot path: report the same numbers for `2bdbcc2`
(or P5-1's commit) vs this change. Gate: within ~10%. If over, profile and
fix before reporting (the EffectQueue money view must not allocate per
read).

## Rules

- `uv run pytest tests/engine tests/env tests/scripts -q -x` green apart
  from strict xfails (report any test you changed and why; changing an
  expected number requires citing the Lua line that justifies the new
  value). `uv run ruff check` clean on touched files.
- No new direct `gs[...]` reads in handlers (the Phase 2 lint).
- Update `docs/engine_class_design_2026-10-06.md` section 5 / 8 only where
  the built code differs from the draft (pass-local money view instead of
  `MoneyLedger` in gs, etc.). Short.
- Do not commit. Write the report to
  `docs/implementation-tickets/engine-p5-2-REPORT.md` (inside the repo) as
  your last step, and repeat it as your final message: files changed, each finding fixed with the
  test that proves it (and that it failed on the parent), every changed
  pre-existing expectation with its Lua citation, perf numbers, and any NEW
  divergence found (do not fix out-of-scope ones; list them).
