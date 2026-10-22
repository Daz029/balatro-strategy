# Engine fidelity assessment — 2026-10-21

Audited engine: `engine-legality` at `df0519ff7467f67cedee95f76fac9f496c890dc7`.
Source: local `~/Code/Code/balatro-strategy/balatro_source/Balatro`, whose
`version.jkr` identifies **1.0.1o-FULL / 1.0.1o / PROD_PC_Console**.
Lua citations below are relative to that directory. Source hashes and measured
results are saved in [the evidence file](engine_fidelity_2026-10-21_evidence.json).
This assessment adds documentation and diagnostics; it does not change engine behavior.

## Assessment

The engine has substantial regression coverage and source-based implementations
of scoring, lifecycle, money sequencing, and legality. It is a useful headless
model of ordinary, all-unlocked vanilla runs. **It is not yet justified to call
it fully faithful to the source, or bit-exact across full seeded runs.** Two
reproduced gameplay differences remain, and the differential tests have gaps
that explain why the existing suite stays green.

The October 12 audit closed a particular list of findings. That is meaningful
progress, but neither the number of closed findings nor the passing-test count
is a percentage of game fidelity. This review reopens the boss-disable completion
case and identifies a counterexample to the live-grabber design's frame equivalence.

## Confirmed findings

### F1 — P1: frame-cached tallies change scoring when recomputed mid-hand

Play a Steel-enhanced Two of Hearts with Vampire and Steel Joker. The source
scores **9 chips**, while this engine scores **7**. Both Lua 5.1 and native
LuaJIT reproduce the source result.

In Lua, `Card:update` records `steel_tally=1` before the play
(`card.lua:4185`). Vampire removes the Steel enhancement in the synchronous
before-scoring pass (`card.lua:3459`), but Steel Joker still reads the stored
tally later in that same `evaluate_play` call (`card.lua:3929`). Its x1.2
therefore combines with Vampire's x1.1: `floor(7 * 1.1 * 1.2) = 9`.
`Game:update` handles events before updating cards (`game.lua:2509,2631`);
there is no intervening card update inside the synchronous scoring function.

Python's [StateView.steel_tally](../jackdaw/engine/read.py:443) lazily counts
the current card enhancements when Steel Joker first requests the value.
That is after Vampire removed Steel, so it reads zero and scores
`floor(7 * 1.1) = 7`. Caching on first access does not establish the same
observation time as a frame snapshot.

**Impact:** scores, win thresholds, and training labels for interacting builds.
Steel Joker is confirmed; Stone Joker, Driver's License, and other per-frame
values need interaction checks rather than assuming every tally is affected
identically. The general rule “replace Card:update fields with live getters”
is not source-equivalent.

**Recommended correction:** preserve gameplay-relevant frame snapshots at
explicit transition boundaries. A complete graphical frame loop is unnecessary,
but the simulator must distinguish values cached before a synchronous pass
from values that Lua actually reads live during that pass.

### F2 — P1: disabling an already-beaten boss does not finish the round

With 700 accumulated chips against a 1,200-chip Wall, sell Luchador.
Python disables the Wall and lowers its target to 600, but returns in
`selecting_hand`. Lua's `Blind:disable` queues `NEW_ROUND` as soon as current
chips meet the reduced boss requirement (`blind.lua:399-408`), starting automatic
round completion without another play. Native LuaJIT and Lua 5.1 both show this
transition. The same missing path applies to a sufficiently scored Violet Vessel.

The omission is explicit in [Blind.disable](../jackdaw/engine/blind.py:510):
the old Phase 5 TODO is still present. Selling follows
`game._handle_sell_card -> shop.sell_card -> DisableBlind -> Blind.disable`
and returns without checking for completion. Phase 6 made this reachable through
a legal sale during a blind.

**Impact:** the simulator requires an unnecessary additional hand, changing
remaining-hand payouts and triggering effects that the real game would not run.
This is a residual C08 lifecycle gap despite that row's earlier “Fixed” status.

**Recommended correction:** settle the source's post-disable completion event
at the action boundary and enter the canonical end-round path, preserving sale
effect ordering. Add integration cases for both threshold-lowering bosses.

The diagnostic compares Python's actual Luchador sale with Lua's actual
`Blind:disable` primitive; it does not claim to run the entire Lua sale UI.
The missing transition is independently visible in the source.

### F3 — P1 verification gap: the transition oracle bypasses pre-play timing

The stock [bootstrap](../scripts/lua_transition_oracle/bootstrap.lua:491)
reconstructs play manually: it queues `ease_hands_played(-1)`, calls
`Blind:press_play`, and invokes `evaluate_play` immediately, before draining
those events. The real `play_cards_from_highlighted` queues scoring behind the
hand decrement, card movement, and blind pre-play work
(`functions/state_events.lua:450-537`).

| Scenario | Stock oracle | Source callback diagnostic | Python engine |
|---|---:|---:|---:|
| Last hand, Two of Hearts, Dusk | 7 chips | 9 chips | 9 chips |
| $4, Two of Hearts, The Tooth, Bull | 15 chips | 13 chips | 13 chips |
| Steel Two, Vampire, Steel Joker | missing `steel_tally` error | 9 chips | 7 chips |

The first two are **oracle defects, not engine scoring defects**. The third
also exposes an oracle omission: its clock drains events without calling
`Card:update`, so a normally initialized frame tally can be nil. The existing
29 transition cases do not cover these combinations.

The diagnostic calls the real play callback and real `Card:update` before the
action and after event-manager updates. Both Lua runtimes agree on every result
shown. UI alerts are stubbed, as are the bootstrap's existing rendering objects.
This is a focused improvement in evidence, not a validated replacement for the
whole production oracle or `Game:update`.

### F4 — P2 verification gap: shop fixtures are not an independent shop oracle

[run_shop_oracle.py](../scripts/generate_fixtures/run_shop_oracle.py:1) generates
expected results with the same Python `populate_shop` implementation that
[test_shop_oracle.py](../tests/engine/test_shop_oracle.py:1) tests. The checked-in
fixture explicitly records this Python provenance. The generator's explanation
that matching RNG primitives establishes equivalent shop output is insufficient:
pool membership, iteration order, exclusions, call counts, stickers, and timing
can differ even when individual random numbers agree.

These fixtures remain useful regression tests. They do not independently prove
Lua shop parity. The referenced `tests/engine/test_rng_sequence.py` is absent
from this branch; separate RNG tests and LuaJIT pipeline fixtures do exist.

**Recommended correction:** collect shop expectations from the actual Lua
shop/pool routines or the live game, including multi-shop sequences, ownership
changes, packs, tags, debt, and vouchers. Preserve source version and provenance.

### F5 — training boundary: face-down Joker identity is observable

The source flips and shuffles Jokers for Amber Acorn (`blind.lua:190-204`).
The engine represents their facing, but
[encode_joker](../jackdaw/env/observation.py:404) and
[encode_jokers_batch](../jackdaw/env/observation.py:1204) encode exact Joker identity
and position without checking facing. A focused check returned identical feature
vectors before and after flipping a Joker face-down. Playing-card encoders do
hide face-down ranks/suits, so the treatment is inconsistent across card types.

This is outside core transition fidelity, but matters for claiming a policy plays
the same information game as a human. An omniscient engine state is appropriate;
the agent observation needs an explicit visibility contract. If privileged
information is intentional, label that evaluation setting accordingly. This
review has not audited every downstream encoder or inferred-feature leak.

## Judgment calls in translating the real game into a simulator

| Translation choice | Current treatment / evidence | Fidelity requirement or explicit tradeoff |
|---|---|---|
| Frames and derived counters | Live getters plus a lazily cached scoring view replace `Card:update`. F1 disproves equivalence in at least one case. | Preserve snapshot time for gameplay fields. Do not equate “latest value” with “value Lua reads.” Frame count itself can disappear when it has no gameplay observer. |
| Animation and event scheduling | Synchronous transitions, effect ordering, and F1–F8 barriers replace wall-clock animation. The oracle uses a forced 60 Hz event drain. | Remove elapsed time only after preserving emission order, deferred writes, nested events, blocking rules, and the next input boundary. A fixed 60 Hz drain is a test choice, not proof across frame rates or game speeds. |
| Money and creation buffers | `EffectQueue` separates committed, instant, and buffered dollars, plus slot reservations. | Keep Bull/Bootstraps' buffered view distinct from Vagabond's committed view. Capacity reservations and room-before-RNG affect gameplay, even if final buffers are zero. Do not collapse all payouts into one late sum. |
| Intermediate game states | Public `step` exposes stable decision points, omitting animation states such as `HAND_PLAYED`. | Settle automatic effects before offering another action. F2 shows why this is a semantic contract, not just a performance optimization. Excluding animation-window exploits should be an explicit scope decision. |
| Card ordering and UI geometry | Ordered PlayHand indices specify physical left-to-right scoring order; swaps expose rearrangement. | This is a reasonable compound action because the real player can rearrange cards before playing. The adapter must enact that arrangement; click order alone is not source scoring order. Preserve held-card order and Death's positional direction too. The stock Lua oracle sorts by its existing `T.x`, so arbitrary index permutations need an adapter-aware comparison. |
| Randomness | Named streams emulate LuaJIT; three pack-variant sites choose variant 1 instead of consuming raw `math.random`; Stone `get_id` returns -1 instead of a random negative ID. | Distinguish strategic equivalence, seeded decision equivalence, and bit-exact execution. Cosmetic raw random calls and frame activity can affect the shared global RNG. A seed alone does not establish all raw-RNG state. The variant simplification is documented, but is still a deviation. |
| Lua tables and object identity | Python lists/dicts replace Lua areas/tables; the oracle substitutes CardArea geometry and uses Lua 5.1. | Preserve ordering where gameplay observes it and identity where copies/removals matter. Lua `pairs` order is not a portable contract. Native LuaJIT checks matter for order-sensitive cases; the four diagnostics here agree across runtimes, not every possible case. |
| Numeric representation | Python integers/floats stand in for LuaJIT doubles; scoring rounds at explicit points. Endless is excluded. | Preserve each rounding/casting point and bound supported magnitudes. Python's unbounded integers are not a substitute for Lua double arithmetic. Do not extrapolate ordinary-run results to extreme scaling builds or endless. |
| Hidden information | Engine state contains true card identities; player-card observations hide some fields, Joker rows do not (F5). | Keep full simulator state separate from what a policy may see. Automatic sorting/ordering, feature aggregates, and solver access can reveal hidden information even if visible rows are masked. |
| Legality versus training actions | Engine legality is centralized; training masks deliberately retain earlier restrictions. | Report engine capability and agent capability separately. Spectral picks, debt-aware purchases, boss rerolls, selling outside shops, and buy-and-use are not uniformly exposed to training policies. Cerulean Bell has an interim action-repair shim; changing the submitted action changes the policy contract. |
| Policy convenience versus game rules | Solver/environment helpers can auto-order Jokers and start from synthetic states; `play_ordering.py` documents heuristic approximations. | A legal auto-order can still outsource a strategic choice. Distinguish engine scoring correctness from solver optimality and agent decision quality. Synthetic states also need reachability checks when used as fidelity evidence. |
| Profiles, challenges, and run horizon | All-unlocked/all-discovered profile; challenges, non-full profiles, and endless explicitly raise (`scope.py`). | This is a defensible supported subset, not all of vanilla. State those exclusions when reporting fidelity or win rates. UI, achievements, and persistence can be omitted only when their gameplay effects are fixed by the chosen profile. |

The recommended contract is **source-equivalent gameplay at stable decision
boundaries for the declared run/profile scope**, with explicit frame snapshots
inside transitions where Lua reads them. Full graphical timing is unnecessary;
discarding timing-sensitive state is not.

## Evidence and limits

Executed on this checkout:

- `tests/engine tests/env tests/bridge`: **2,301 passed, 1 skipped, 2 deselected**
  (9.74 seconds). The skip is `test_maskable_guard.py` because torch is absent;
  the default test configuration excludes benchmark/live tests.
- Focused transition, hand-evaluation, RNG, and source-state-map files:
  **244 passed**, including all **29 transition-oracle cases**. This is a subset
  of the broad run, not 244 additional independent scenarios.
- The four audit diagnostics: run with `lupa.lua51` (lupa 2.6) and native
  **LuaJIT 2.1.1788856981**. Native diagnostics use native random functions,
  avoiding the stock transition oracle's reuse of Python's RNG implementation.
- Ruff passes for the diagnostic script.

Reproduce the diagnostic from the repository root:

```sh
.venv/bin/python -m scripts.audit_engine_fidelity
```

Set `BALATRO_SOURCE` to use another source directory. The command emits source
hashes and results; it reports an unavailable native LuaJIT explicitly. It runs
in-memory harness adjustments and leaves the source, engine, and stock oracle
files untouched. Nonzero/incorrect gameplay results are recorded for review;
its successful exit means the diagnostic ran, not that fidelity passed.

This review inspected transition, frame, legality, lifecycle, RNG, shop-test
provenance, and observation seams. It did not exhaustively compare every
Joker/consumable/boss combination or run the installed graphical game/BalatroBot
end to end. The existing live-harness and Windows verification follow-ups remain
open. Source-based checks and targeted native execution support the findings
above, but do not establish a global parity guarantee.

## Recommended next work

1. Repair the transition oracle's play entry and frame updates, then add the
   Dusk/Tooth controls so it cannot falsely blame correct engine behavior.
2. Fix F1's observation-time semantics and F2's post-disable round completion;
   retain the reproductions as differential regression cases and expand the
   affected interaction families.
3. Build independent Lua shop/pack/action sequences. Compare state and RNG
   streams across multiple decisions, not just isolated final scores.
4. Define the visibility and action contracts for training adapters; address
   face-down Joker information and the already-deferred mask rebuild.
5. Run live BalatroBot sequences on the same source/build before asserting
   full gameplay parity. Regenerate affected labels after scoring fixes, and
   version the engine semantics with harvested data and checkpoints.
