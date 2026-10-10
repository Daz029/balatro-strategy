# Engine Phase 5, ticket P5-4: end_round, Blind.defeat, evaluate_round, cash_out

Branch `engine-phase1-state`, after P5-3. Read first:
`docs/engine_s4a_event_timing.md` section 1 (F6, F7, F8) and section 4
(`end_round`, `Blind:defeat`, `G.FUNCS.evaluate_round`, `G.FUNCS.cash_out`);
plan Phase 5 and Part 6; audit C10, C11, D18, D22, D30, D32 (defeat half);
the P5-2 and P5-3 reports in `docs/implementation-tickets/`. Lua:
`~/Code/Code/balatro-strategy/balatro_source/Balatro`. Lua wins; cite
`# L: file:line` per ported step. Port Lua's behavior faithfully even where
it looks like a bug (e.g. NEW-P5-1-03), with a comment saying so.

## 1. `end_round` (replaces `_round_won` + the play handler's loss branch)

Every play that ends the round (blind beaten OR no hands left) and the
empty-hand/deck loss go through ONE `end_round(gs)` port:

- C11: `game_over` is decided from COMMITTED accumulated `gs["chips"]` vs
  the blind target (`state_events.lua:87-98`). Mr. Bones is an
  `end_of_round` handler with `game_over=True` and Lua's 25% threshold
  (`card.lua:3047-3062`); `saved` comes from that pass. Delete the
  `saved` / Mr. Bones block in `score_hand` and `ScoreResult.saved` (check
  readers in `jackdaw/` and `scripts/`). The structural test from Phase 4
  must not need an allowlist entry for this.
- D18 / D22: the per-joker interleave — for each joker: `end_of_round`
  dispatch, consume `saved`, queue rental, perishable tally/debuff — then
  the rest, matching `state_events.lua:99-110`. A rental joker that
  destroys itself still pays rent. On an ordinary LOSS, Lua still runs this
  maintenance before GAME_OVER: do the same (rent charged, Gros Michel
  etc. evaluated) and then set GAME_OVER.
- Won/saved branch: unused discards; on Boss, `most_played_poker_hand`
  snapshot with Lua's exact tie code (NEW-P5-1-03 — Lua never updates
  `_order`; replicate and comment).
- C10: held-card end-of-round effects with repetitions: for each held
  card and each repeat, the card's EOR effect (Gold card `h_dollars`, Blue
  seal Planet) plus individual joker effects; Red seal and joker
  repetitions (Mime) collected on the first pass (`state_events.lua:171-233`).
  Blue seal Planet = LAST played hand (`G.GAME.last_hand_played`), created
  through the normal creation path (proper ability, cost, pool
  registration), respecting consumable room per creation. Audit repros:
  held Gold+Red seal + one Mime pays **$9**; Blue seal after a High Card
  win with Pair most-played gives a properly initialised **Pluto**.
- Queued moves/reset event (`state_events.lua:234-283`): hand to discard,
  Boss ante change, discard to deck (keep today's verified deck order),
  enter ROUND_EVAL, blind state Defeated, **clear `played_this_ante` on
  every playing card on Boss defeat (D30)**, revert temporary modifiers
  (Juggle tag hand size), reset target cards Idol/Mail/Ancient/Castle HERE
  (NEW-P5-1-09 — move them out of cash-out), clear discard/forced flags.
- Flush F6 at the end.

## 2. `Blind.defeat` (D32 second half)

Add `Blind.defeat(gs)` porting `blind.lua:276-344`: flip face-down jokers
back, Manacle hand-size restore (delete the inline restore in
`_round_won`), and the final `set_blind(None, ..., reset=True)` semantics
(clear boss debuffs on cards AND jokers through the setter — respect
permanent perishable debuffs, D25 rule). Called from `evaluate_round` at
Lua's point (F7), not from `end_round`.

## 3. `evaluate_round` (F7) and `cash_out` (F8)

- Blind reward row only if committed chips beat the blind; a Mr.
  Bones-saved loss gets $0 (NEW-P5-1-10, `state_events.lua:1135-1146`).
- `Blind.defeat`, Back `eval`, then hands remaining, Green Deck discards,
  each joker's `calc_dollar_bonus` computed NOW (after the EOR pass, D18:
  a fresh Rocket on a defeated boss pays **$3**; a Golden Joker debuffed by
  perishing pays nothing), eval tags, interest from committed dollars.
  Keep `RoundEarnings` as the record of the rows; `current_round.dollars`
  written at the bottom-row point.
- `cash_out`: shuffle, reset `jokers_purchased` and hands/discards before
  entering SHOP (NEW-P5-1-02 — check what shop obs / env code reads
  `hands_left` / `discards_left` in SHOP and report any visible change),
  commit `current_round.dollars` through the ledger, then snapshot
  `previous_round.dollars`, then chips to zero / post-Boss tags / blind
  choices. `tests/engine/test_cashout_ordering.py` pins today's interest
  ordering: it must stay green or each change must be justified by a Lua
  line.

## 4. Tests (red first)

New `tests/engine/test_end_round_sequencing.py`, through `step()`; each
must FAIL on the parent (check and say so): C10 ($9, Pluto), C11 (7 chips
vs 300 with no hands left: Bones NOT consumed, GAME_OVER; 80 of 300:
saved, $0 blind row), D18 (Rocket $3; perished Golden pays 0), D22 (self-
destroying rental pays rent; rent on a losing round), D30 (Pillar after a
new ante does not debuff a card played in an earlier ante), D32 defeat
cleanup, NEW-P5-1-02/03/09/10. Flip the matching transition-oracle xfails
(keep strict). Then run the seeded rollout (`tests/engine/_rollout.py`) and
the `tests/env` suite.

## 5. Data-impact note

Add a short "Phase 5 data impact" paragraph to the report (shop obs values
in SHOP after NEW-P5-1-02, economy changes, label semantics) for the plan's
Part 6.

## Rules

As P5-2. Do not commit. Report to
`docs/implementation-tickets/engine-p5-4-REPORT.md` and repeat it as the
final message.
