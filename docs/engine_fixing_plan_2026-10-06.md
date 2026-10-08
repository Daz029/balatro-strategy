# Engine fixing plan

Date: **2026-10-06**. Input: [`engine_audit.md`](engine_audit.md) (audited 2026-09-05 at
`d459179`; 15 critical findings C01–C15, 59 direct deviations D01–D59, plus 8
"functionally equivalent" notes E01–E08) and the user annotations in it and in
`jackdaw/engine/state.py`. Branch at time of writing: `engine-fixes` @ `5317b71`.

This is a plan, not a fix log. Nothing in the engine was changed to write it.

---

## 0. Summary

The audit's 74 findings are not 74 separate bugs. About 50 of them come from
**six design problems** in how the engine is built. The other ~24 are ordinary
local bugs (a wrong comparison, a wrong RNG key, a missing guard). Fixing the
local bugs one at a time is cheap, but it is the method that produced the
CLAUDE.md "integration-seam joker bug" list (Throwback, Idol, blueprint_compat,
Marble, Riff-raff, jokers=None, Campfire/Rocket, Gros Michel, To Do List…). That
list keeps growing because the design makes new bugs of the same kind easy to
write and hard to see. The overhaul fixes the six design problems first, so
that the local fixes land on a structure that cannot quietly drop them.

| ID | Systemic problem | Findings it causes |
|---|---|---|
| **S1** | Effects are returned as descriptors and applied by hand-written, per-call-site appliers that each know only a subset of fields. Handlers read a frozen, caller-assembled snapshot. | C02, C05\*, D07, D11, D15, D19\*, D23\*, D35\*, D46, (C14 Anaglyph) |
| **S2** | Game state is an untyped dict with no owner per key: aliased keys, writers and readers using different names, derived values cached in flags that nothing maintains. | C04, C07, C13, C14 (Green), D03, D04, D19, D26, D40\*, D42\* |
| **S3** | No single lifecycle path for creating, copying, mutating or removing cards (and blinds). Call sites assign fields directly and skip the bookkeeping Lua bundles into `Card:*` methods. | C06, C08, C10\*, C12, C13 (Magic Trick), D16, D17, D20, D24, D25, D33\*, D35, D36, D37, D41, D42, D47, D53\* |
| **S4** | Top-level phases (play, discard, draw, set blind, end round, defeat) are hand-written procedures that reorder Lua's steps, and Lua's event-queue timing (immediate vs deferred) is not modeled at all. | C01, C03, C05, C09, C10, C11, D12, D14, D18, D22, D28, D29, D30, D31, D32 |
| **S5** | Action legality is split between `actions.py` enumeration and `step()` validation, both incomplete, and agent-side restrictions are built into the engine. | D23\*, D27, D50, D52\*, D56, D57 |
| **S6** | Verification checks each handler on its own. There is no integrated differential oracle and no structural check that every produced field has a consumer and every state key read has a writer. | Every class above. This is why they were missed. |

\* = the finding also has a local component, listed in Part 2.

**Decisions I need from you** are collected in Part 7. The biggest ones: S1's
effect representation, S2's state representation, S4's money/event-time model
(this is where your Vagabond note applies), and S5's engine-vs-env legality
boundary.

**State first (added 2026-10-06):** reads go through small read functions
("grabbers") over live state, and before those are written the state must be
able to hold every value the game uses. Part 1b inventories that against the
real Lua source. It cannot yet: about 15 values have no home, 12 concepts are
stored under several names, and `Card` lacks about 10 fields. Phase 1 fixes
that before anything else.

**Blast radius:** this is a label-semantics change everywhere. Every solver
label, harvested blob, BC shard and checkpoint was produced on the current
engine. See Part 6.

**Current training is Red Deck, White Stake, no challenge** (`BACK_KEY="b_red"`,
`STAKE=1` in `shop_gym.py`, `hand_play_gym.py`, `generate_hand_demos.py`). That
splits the findings by training impact. The priority tags below use it:

- **P0**: affects current training states (Red/White, no challenge).
- **P1**: affects a reachable vanilla state that we do not train on yet (other
  decks or stakes, sell-during-blind, and so on).
- **P2**: API hygiene, extreme numeric cases, or out of scope.

---

## Part 1. Systemic / design issues

### S1. Snapshot-and-descriptor effect model with per-site appliers

**Architecture now.** Joker handlers are pure functions
`(Card, JokerContext) -> JokerResult | None` (`jokers.py:276`). A `JokerContext`
carries a frozen `GameSnapshot` (`jokers.py:35`): about 20 scalar fields (money,
hands_left, tallies, probabilities…) that the **caller** fills from `gs`.
`JokerResult` has typed scoring fields (`chips`, `mult_mod`, `Xmult_mod`,
`dollars`, `remove`, `saved`, …) plus an untyped `extra: dict` for everything
else (`create`, `destroy_joker`, `destroy_random_joker`, `disable_blind`,
`pool_flag`, `duplicate_random_joker`, …). Consumables mirror this design with
`ConsumableResult` (`consumables.py:70`), which has about 14 optional mutation
fields.

At least six independent call sites build their own snapshot and their own
applier:

| Call site | Snapshot built from | Applier |
|---|---|---|
| `score_hand` (`scoring.py:398`) | `gs` top-level mirrors that `_handle_play_hand` writes first (`gs["money"]`, `gs["hands_left"]`, `gs["stone_tally"]`, …, `game.py:612`) | inline per phase; phase 5 kept only `level_up` until recently, phase 9 only looks at `extra["create"]` |
| `_fire_setting_blind` (`game.py:1783`) | 3 fields | `_apply_setting_blind_mutations`: knows `disable_blind`, `destroy_random_joker`, `set_hands`, `set_discards`, `create` |
| `on_end_of_round` (`jokers.py:322`) | 4 fields (`game.py:1579`) | `_round_won`: knows `pool_flag` only (added 2026-09-04) |
| `_fire_shop_joker_context` (`game.py:2354`) | own | `_apply_shop_mutations` |
| `_use_consumable_card` (`game.py:1970`) | own | `_apply_consumable_result`, ~14 field branches in a fixed order |
| discard (`_build_discard_snapshot`, `game.py:980`) | own | inline |

**The problem.** Three failure modes follow from this design, and the audit
finds every one of them:

1. **Dropped effects.** A handler that returns a field its call site's applier
   does not know about has no effect. Nothing raises. DNA's `extra.create` in
   `before`, Sixth Sense's create in the destroy pass, 8 Ball's create in
   `individual`, Hallucination's on booster open (C02), Ceremonial Dagger's
   `destroy_joker` at setting_blind (D15), `selling_self` never dispatched
   (D23). Before 2026-09-04 the same was true of `pool_flag` and To Do List's
   dollars. The handler's unit test passes because it checks the return value,
   not the effect.
2. **Starved snapshots.** Each call site fills only the snapshot fields its
   author thought of. The end-of-round snapshot has 4 of the ~20 fields, so
   Delayed Gratification reads `discards_used=0` (D19). Loyalty reads the
   round-local `hands_played` (D07). Every default value in `GameSnapshot` is a
   silent wrong answer.
3. **Wrong application order and capacity.** Each applier runs its fields in its
   own fixed order. Ankh resolves the copy before it destroys the other jokers,
   so the copy fails the room check (D35). Generation creates the card and
   registers it in the pool, which advances RNG, and only then checks room
   (D46).

**Proposed direction.**

- One **effect vocabulary** shared by jokers, consumables, tags, blinds and
  seals: `Create(set, key?, area, edition?, …)`, `Destroy(card)`,
  `DisableBlind()`, `SetMoney`/`EaseMoney`, `LevelUp(hand, n)`,
  `SetPoolFlag(flag)`, `AddPlayingCard(...)`, `Copy(src, dst)`, … Scoring
  deltas (`chips`/`mult`/`x_mult`) stay as plain numeric fields because they are
  the hot path.
- One **applier**, `apply_effects(gs, effects, source, phase)`, used by every
  call site. Two rules: an unknown effect **raises**, and the applier owns room
  and capacity reservation, so it checks room before any RNG draw.
- One **context builder**, `build_context(gs, phase, **per_call)`. It derives
  every snapshot field from live state in one place, so no call site assembles
  its own. `GameSnapshot` can stay frozen for speed, but it has exactly one
  constructor, and that constructor fills every field.
- Delete the `JokerContext` InitVar backward-compat path (`jokers.py:140–240`).
  It is a second way to build a snapshot with defaults.

**Solver constraint (hard).** `scripts/hand_solver.py` and `play_ordering.py`
call `score_hand` counterfactually on `fast_clone`d cards, jokers, hand levels,
blind and RNG (`hand_solver.py:538, 689`). Whatever replaces S1 must keep
"score a hypothetical play without committing it" cheap. Concretely:
`score_hand` operates on a cloneable scoring state and returns effects, and the
real play path commits them through `apply_effects`. The solver never commits.

**Decision needed (D-S1):** choose the effect representation. **Recommended:**
typed effect dataclasses with exhaustive dispatch. Alternative: keep dict
`extra` plus a registry of known keys that raises on any unknown key. That is
cheaper and covers failure mode 1, but not 2 or 3.

---

### S2. Untyped state dict with aliased, unowned and cached-derived keys

**Architecture now.** `gs` is a plain `dict[str, Any]`. `state.py` documents it
and says so itself: *"documentation only — no runtime enforcement"*. Writers and
readers agree on key names only by convention. The engine has about 576 `gs` /
`game_state` accesses, `jackdaw/env` about 232 and `scripts/` about 203.

**The problem.** Three kinds of key drift, all present:

| Kind | Instances |
|---|---|
| **Aliases**: same concept, two keys, writer and reader on different ones | Oops writes flat `probabilities_normal` (`card.py:772`); scoring reads `gs["probabilities"]["normal"]` (`game.py:641`); Wheel reads the flat one (`consumables.py:535`): **C04**. Stakes write `gs["modifiers"]["enable_*_in_shop"]`; the factory reads flat `gs["enable_*_in_shop"]` (`card_factory.py:401`): **C07**. Green Deck writes flat, economy reads `modifiers`: **C14**. Omen Globe writes `omen_globe`, packs read `has_omen_globe`: **C13**. Chaos writes flat `free_rerolls`, the shop reads `current_round.free_rerolls`: **D26**. Lucky writes `ability["lucky_trigger"]`, Lucky Cat reads top-level: **D04**. |
| **Read with no writer** | `has_telescope`, `has_astronomer` (C13), `deck_enhancements` (D42), `ability.nine_tally`, `ability.planet_types_used` (D19), the Tarot usage tally (D40). |
| **Mirrors**: copies that must be refreshed by hand | `gs["money"]` mirrors `dollars`; `gs["hands_left"]`, `gs["discards_used"]` mirror `current_round`; `stone_tally`, `steel_tally` (`game.py:612–634`). These are refreshed only on the play path, so every other path reads stale values. |

A fourth, related form: **global modifiers threaded by argument.** Lua's
`Card:is_face()` / `Card:is_suit()` query the active jokers (Pareidolia,
Smeared) themselves. Python passes `pareidolia=` / `smeared=` booleans, so every
call site that forgets one is wrong: Ride the Bus, Caino, discard context,
Flower Pot, Seeing Double, suit-debuff blinds (**D03**). The same pattern
produced the `jokers=None` bug (Four Fingers, Shortcut and Smeared all inert)
fixed in K3.

**Proposed direction.** Keep the dict. Replacing it with a dataclass means
~1,000 call-site edits and breaks every pickled harvest blob. Instead:

1. **A canonical key registry** in `state.py`: one name per concept, a type, and
   an owner module. Aliases are removed. A migration helper maps old keys to
   canonical ones on blob restore (see Part 5).
2. **Derive, don't cache ("grabbers").** Every "has X" or "count of X" value
   becomes a small read function over live state: `has_voucher(gs, "v_telescope")`,
   `find_joker(gs, "Astronomer")`, `deck_enhancements(gs)`, `nine_tally(gs)`,
   `probability_normal(gs)`. Lua mostly computes these on demand (`find_joker`,
   iteration over `G.playing_cards`). Where Lua does cache (for example
   `G.GAME.probabilities`), the cache gets exactly one writer.
3. **A rules view** (`rules(gs)` → `pareidolia`, `smeared`, `four_fingers`,
   `shortcut`, `splash`, …) computed in one place. `Card.is_face` / `is_suit`
   take that view, never loose booleans.
4. **A static check (S6):** an AST scan that lists every `gs[...]` /
   `gs.get(...)` key read and fails when a read key has no writer anywhere.
   This would have caught C04, C07, C13, C14, D19, D26 and D42 mechanically.

**Decision needed (D-S2):** keep the dict plus registry, derived accessors and
lint (recommended), or migrate to a typed `GameState`.

Grabbers can only read what the state holds. Before writing them, Part 1b
checks that the state has a home for every value Lua reads or writes. It does
not yet.

---

### S3. No single card (or blind) lifecycle path

**Architecture now.** Lua concentrates bookkeeping in methods:
`Card:set_edition` (scoring values, negative slot, cost), `Card:set_ability` /
`set_base` (blind debuff re-evaluation), `Card:add_to_deck` /
`remove_from_deck` (passives, pool registration), `Card:set_debuff` (passive
remove/re-add, perishable guard), `Card:remove` (pool exclusion cleanup),
`copy_card`, `CardArea:emplace` (area routing, flip), `Blind:disable` /
`Blind:defeat`, and the `playing_card_joker_effects` notifications. Python has
some of these methods (`card.py:209–420`, `746–825`), but call sites routinely
skip them:

- `target.edition = edition` (`game.py:2166`) instead of `set_edition` → **C12**
- `planet = Card(center_key=…); planet.ability = {...}` (`game.py:1657`) → **C10**
- `c = Card(); c.set_ability(...)` with no front (`game.py:1863`) → **D16**
- `copy.copy` / `deepcopy` + sort_id patch (`game.py:2139`, Perkeo `2417`) → **D24, D35**
- `blind.disabled = True` (`game.py:1830`) instead of `Blind.disable()` → **C08**
- `card.debuff = False` loops (`game.py:1696`, `1833`) → **D25**
- `set_debuff` is a bare assignment (`card.py:418`) → **D25**
- `set_eternal` / `set_perishable` skip the compatibility rules → **D47**
- `used_jokers` is registered at creation and never released (`card_factory.py:389`) → **C06**
- Death's manual partial copy (`game.py:2060`) → **D33**
- Suit/rank/enhancement changes skip the debuff refresh → **D36**; `set_base` loses `suit_nominal_original` → **D41**
- Playing-card creation routes by `phase == SELECTING_HAND` only, so cards created while a pack is open go to the deck → **D37**
- Created / destroyed notifications fire on some paths only → **D17**
- Turtle Bean decay edits `extra.h_size` without touching live `hand_size` → **D20**
- Coupon state → **D53**

**The problem.** Every new mutation path has to re-derive five or so pieces of
bookkeeping by hand, and the audit shows most do not. This is the same shape as
S1, applied to objects instead of effects.

**Proposed direction.** A `lifecycle` module that is the **only** way to:
create a card (factory, always with a front for playing cards, rolling edition
whatever the area), copy one (`copy_card` semantics, fresh identity), place it
(`emplace(gs, card, area)`, which handles area routing during
`PACK_OPENING` / `SELECTING_HAND`, notifications, flips and debuff evaluation),
remove it (`destroy`, which handles pool release, `remove_from_deck` and
notifications), and change it (`set_edition`, `set_ability`, `set_base`,
`set_debuff`, `set_sticker`). It also owns `Blind.disable` and `Blind.defeat`.
Enforce it with a lint (S6) that bans direct assignment of `.edition`,
`.debuff`, `.seal`, `.ability =` and bare `Card(` outside `card.py`,
`card_factory.py` and `lifecycle.py`.

C06 needs its own sub-decision. Lua's pool exclusion is "**currently exists**"
(cleared on `Card:remove` when no copy remains). The Python
"historically generated" rule was a deliberate choice: the code says it never
unregisters sold jokers. Matching Lua changes seeded shop sequences, which
breaks shop/RNG oracle fixtures and every harvested shop state.
**Recommended: match Lua.** This is a decision, not just a bug fix
(**D-S3a**).

---

### S4. Phase sequencing does not follow Lua, and Lua event timing is not modeled

**Architecture now.** Each top-level transition is one long procedure with
numbered steps: `_handle_play_hand` (11 steps, `game.py:529`), `_round_won`
(11 steps, `game.py:1550`), `_handle_select_blind`, `_handle_discard`,
`_draw_hand`. Each cites Lua line numbers in its docstring, but the order is the
author's paraphrase. Subroutines that Lua shares are written once per call site
in Python: draw-to-hand (with `stay_flipped`, `prepped`, re-debuff) is
duplicated between `_draw_hand`, the post-play redraw (`game.py:713–758`) and
Serpent's special path. Discard is separate from Hook's forced discard.

E02 in the audit says synchronous phases are equivalent **if** "the caller
applies it exactly once at the same logical point". In practice the
procedures do not hold to that rule.

**The problem has two layers.**

1. **Ordering drift.** Hand counters are incremented before scoring and
   `record_play` is called twice (C01). Vampire's invented
   `individual_hand_end` phase double-applies its multiplier (C03). The Ox runs
   after scoring and wipes earnings (D29). Blocked hands return before the
   `after` pass (D12). Mr. Bones is evaluated inside `score_hand` against
   **this hand's** score (C11). End-of-round dollar bonuses are computed before
   end-of-round mutations, so Rocket pays stale (D18). Burglar runs before
   `start_round` overwrites it (D14). Rental and loss timing differ (D22).
   End-of-round held cards get no Red seal / Mime retriggers (C10). Pillar
   markers are never reset (D30). The draw path never calls `stay_flipped`
   (C09). Crimson Heart ignores `prepped` (D28). Hook bypasses the discard
   pipeline (D31). There is no `Blind:defeat` equivalent (D32).
2. **Event time is not modeled.** In Lua, `evaluate_play` runs synchronously,
   but most of its money changes are `ease_dollars(..)` calls that **queue an
   event** and update `G.GAME.dollars` later. Lua bridges the gap with
   `G.GAME.dollar_buffer` for the consumers that must see pending money (Bull,
   Bootstraps). `ease_dollars(x, true)` (instant) is used where the change must
   be visible at once (The Ox's drain in `debuff_hand`). So during scoring
   there are **two money views**: committed `dollars`, and `dollars + buffer`.
   Each consumer reads the specific view its Lua line reads. Python has one
   frozen snapshot value (`money`) plus one post-hoc application. That gives
   the right answer for some consumers and the wrong one for others.

**Your Vagabond note (in C05) is this second layer, and I think it is
correct.** The audit models C05 as "Lua updates money live, Python freezes it".
That is not accurate either. Vagabond reads committed `G.GAME.dollars`, which a
non-instant gold-seal or Matador payout has **not** reached yet but The Ox's
**instant** drain has. That is the "flag remembered after Ox, before
gold/Matador" behavior you describe. Bull and Bootstraps read
`dollars + dollar_buffer` and **do** see the pending payouts. So the fix for C05
is not "make money live". It is "model committed money and the pending buffer,
and point each consumer at the view its Lua line uses." *Caveat: this is from my
reading of 1.0.1o. `balatro_source/` is gitignored and not on this machine, so
every read site (Bull, Bootstraps, Vagabond, Ox, To Do List, gold seal, Lucky)
must be checked against the source before implementation. That check is task
S4-a below.*

**Proposed direction.**

- Rewrite each top-level Lua function as a **line-faithful port with a
  `# L:` citation per step**: `evaluate_play`, `play_cards_from_highlighted`,
  `discard_cards_from_highlighted` (with the Hook flag), `draw_card` /
  `G.FUNCS.draw_from_deck_to_hand` (one shared draw with `stay_flipped`,
  `prepped`, debuff), `new_round` / `set_blind`, `end_round`, `Blind:defeat`,
  `evaluate_round`, `cash_out`. Shared Lua subroutines become shared Python
  functions. Keep them **synchronous**.
- Model event timing explicitly, and only where it is observable: a
  `MoneyLedger` with `committed` / `pending` (`dollar_buffer`) and
  `ease(instant=bool)`. Pending flushes at the point Lua's queued events would
  have run. Add any other committed-vs-pending state the S4-a sweep finds.
- Per-step **observation-point tests** in the oracle (S6): state after each Lua
  step, not just the final total.

**Decision needed (D-S4):** port line-faithfully with a minimal event model
(recommended), or build a general deferred-event queue. A general queue is
heavier and slower on the solver hot path, and the only gameplay-visible timing
appears to be money plus a few flags.

---

### S5. Action legality split across two places, with agent restrictions inside the engine

**Architecture now.** `actions.py` enumerates legal actions; `step()` handlers
validate separately. Both are partial (D56: duplicate play indices, negative
targeting indices, overfilling slots, `SellCard.area` unrestricted). Some
restrictions in the engine are there for **the bot or agent**, not because Lua
has them: sell is allowed only in `SHOP` (D23); all Spectral-pack picks are
omitted from legal actions "due to a bot-interface limitation" (D57); there is
no boss-reroll action, so Retcon and Director's Cut are dead (D52); packs forbid
using, selling or arranging owned consumables (D57). Cerulean Bell's forced card
is not enforced (D27). Credit Card debt and zero-cost affordability are wrong
(D50).

**The problem.** The engine's job is to be the game. Agent-convenience
restrictions belong in the env's masks. They are already there: `shop_gym`
masks its Discrete(686), and the hand env masks its own space. Putting them in
the engine makes the **simulator** unfaithful even for consumers that want the
full game (the solver, oracle comparisons, future agents).

**Proposed direction.** One `legality.py` with
`is_legal(gs, action) -> bool | reason`. `step()` raises on illegal actions and
`get_legal_actions()` filters by the same predicate, so the two cannot drift.
The engine exposes the **full vanilla action set**: sell in blinds and packs,
boss reroll, buy-and-use, consumable use and sell in packs, Spectral picks.
Each env keeps or extends its own mask. Our current agents never send the new
actions until we choose to append them, so **no checkpoint breaks**.

**Decision needed (D-S5):** confirm "engine = full vanilla legality, env =
restrictions" (recommended). This is also where we decide whether
**sell-during-blind** joins the agent action spaces later. It is strategically
real (Luchador / Invisible Joker / Diet Cola timing, selling to clear Verdant
Leaf), but it is an env and action-space decision, so it is out of scope here.

---

### S6. Verification measures handlers, not the integrated game

**Architecture now.** About 1,092 engine tests. Most build a `JokerContext` by
hand and assert the handler's return value. Some integration tests
(`test_jokers_integration.py`, `test_integration_seam_fixes.py`,
`test_cashout_ordering.py`) drive `step()`. Oracles: `lua_scoring_oracle.lua`
(a stubbed LuaJIT scoring pipeline), an RNG oracle and a hand-eval oracle
(LuaJIT via `lupa`; **14 tests skip without it**), and the BalatroBot live
harness (`docs/validation.md`, ~250 scenarios, needs a running game).

**The problem.** A handler test cannot fail on a call-site bug, by
construction. CLAUDE.md has written down this lesson about ten times
("assert through the integration path") and it is still enforced only by
memory. The oracles cover scoring and RNG primitives, not phase sequencing,
lifecycle or effect application. That is exactly where S1–S4 live.

**Proposed direction (Phase 0 below):**

1. **Structural tests that fail mechanically:**
   - *Producer/consumer:* every `JokerResult` / `ConsumableResult` /
     `TagResult` field and every effect type a handler can return has a
     consumer at every call site where that handler can fire. Implement by
     AST-scanning the handler for returned fields and checking the call-site
     applier's handled set.
   - *Reader/writer:* the S2 key lint.
   - *Lifecycle:* the S3 direct-assignment lint.
   - *Registry coverage:* every registered joker, consumable, tag, voucher, boss
     and deck has at least one test that goes through `step()`.
2. **A headless Lua oracle for whole transitions.** Extend
   `lua_scoring_oracle.lua` into a harness that loads the real
   `state_events.lua`, `card.lua` and `blind.lua` with stubbed UI, and a
   synchronous `G.E_MANAGER` that drains events in order and keeps
   instant/queued semantics. Feed the same controlled state to both engines,
   run `evaluate_play` / `end_round` / `set_blind` / discard / draw, and compare
   state at each logical observation point (money and buffer, counters, card
   identity and order, abilities, RNG stream state). **Prerequisite:** the
   `balatro_source/` tree and LuaJIT on the development machine. Neither is on
   this Mac. The audit ran on the Windows machine.
3. **Every audit finding becomes a failing test first** (`xfail(strict=True)`
   tagged with its finding ID) before its fix lands. Then the fix flips it, and
   "verified FAILING on the parent" (the project's existing rule) becomes
   automatic.

---

## Part 1b. State completeness inventory (added 2026-10-06)

**Question:** before building grabbers, does the game state have somewhere to
store every value the real game reads or writes? **Answer: not yet.** About 15
gameplay-relevant values have no home or no writer, 12 concepts are stored
under two or more names, 14 keys are hand-refreshed copies of live values, and
the `Card` object is missing about 10 fields Lua relies on.

### Method

`scripts/engine_state_inventory.py --lua-src <balatro_source/Balatro>` scans
the vanilla 1.0.1o source for every `G.GAME.<path>` read and write (176 paths at
depth 2). It also AST-scans `jackdaw/engine` for every game-state key read and
write (172 keys) plus the keys `init_game_object` creates, and flags each Lua
path as MISSING (the engine never touches it) or NO-WRITER (Lua writes it at
runtime, the engine only initialises or reads it). The raw run flags 107 rows.
The tables below are the hand-curated result: renames, UI-only fields and
scanner false positives are removed. Per-card and `Blind` fields were compared
separately (`self.*` writes in `card.lua` / `blind.lua`, `.ability.*` across all
Lua files, against the `Card` and `Blind` dataclasses).

The source tree is at `~/Code/Code/balatro-strategy/balatro_source/Balatro`
on this machine. It is gitignored, so the script takes its path as an argument.

Known scanner limits: it does not follow local aliases beyond `cr` / `rr`
(`shop = gs["shop"]; shop["joker_max"] = …` in `vouchers.py:174` shows up as
NO-WRITER), and the RNG's private `state` dict (`rng.py:433`) shows up as
`seed` / `hashed_seed` reads.

### 1b-1. Missing: gameplay values with no home, or initialised but never written

| Lua field | Read by (Lua) | Engine today | Finding |
|---|---|---|---|
| `dollar_buffer` | Bull (`card.lua:3936`), Bootstraps; incremented by every deferred payout, reset by a queued event (`card.lua:1085–1086`) | absent | C05 |
| `joker_buffer`, `consumeable_buffer` | room checks for deferred creations: Riff-raff (`card.lua:2529`), Vagabond (`3743`), 8 Ball, Cartomancer, … | initialised, never read or written | C02, D46 |
| `last_hand_played` | Blue seal Planet (`card.lua:1047`); set at `state_events.lua:576` | absent | C10b |
| `current_round.most_played_poker_hand` | The Ox (`blind.lua:562`); written at `state_events.lua:137` | initialised, never written (Ox recomputes live instead) | D29 |
| `consumeable_usage[key]` (`{count, order, set}`) | Satellite (distinct Planets used), usage-ordered effects; `misc_functions.lua:1191` | initialised, never written | D19, D40 |
| `consumeable_usage_total.{tarot, planet, spectral, tarot_planet, all}` | Fortune Teller and others | stored as `consumable_usage_total` (different spelling), partly written | D40 |
| `ecto_minus` | Ectoplasm's growing hand-size cost | initialised, never used | D34 |
| `orbital_choices[ante][blind_type]` + tag `ability.orbital_hand` / `blind_type` | Orbital Tag (`tag.lua:107–109`) | absent | D54 |
| `facing_blind` | gates `first_hand_drawn` (`game.lua:3226`) | absent | C02 (Certificate) |
| `shop_free`, `shop_d6ed` | Coupon / D6 Tag one-shop flags (`tag.lua:383, 448`; cleared `button_callbacks.lua:2932`) | handled by `TagResult.coupon` / `temp_reroll_cost`; **verify equivalence** | — |
| `last_blind.{boss, name}` | Investment Tag (`tag.lua:119`) | derived from `gs["blind"]` at cash-out (`game.py:1036`); equivalent only while that slot still holds the defeated blind. **Verify, then make it a grabber.** | — |
| `modifiers.money_per_hand`, `money_per_discard`, `no_interest` | Green Deck cash-out | written flat, read under `modifiers` | C14 |

### 1b-2. Aliases: one concept, several names

Each needs one canonical key, and a load-time migration for old blobs.

| Concept | Names in use | Finding |
|---|---|---|
| probability multiplier | `probabilities.normal` (scoring) vs `probabilities_normal` (Oops, Wheel) | C04 |
| shop sticker flags | `modifiers.enable_*_in_shop` (stakes) vs flat `enable_*_in_shop` (factory) | C07 |
| Green Deck economy | `modifiers.*` vs flat | C14 |
| voucher-owned flags | `used_vouchers[v_*]` vs `omen_globe` vs `has_omen_globe`, `has_telescope` | C13 |
| free rerolls | `current_round.free_rerolls` vs flat `free_rerolls` | D26 |
| playing-card count | `playing_card_count` (factory, `card_factory.py:379`) vs `playing_cards_count` (play path). **New: found by the scan.** The factory always reads its default of 52. | new |
| consumable usage | `consumeable_usage_total` (Lua) vs `consumable_usage_total` vs `consumable_usage_tarot` | D40 |
| most-played hand | `current_round.most_played_poker_hand` vs `most_played_hand` (read, no writer) vs `hand_levels.most_played()` | D29 |
| booster ante scaling | `modifiers.booster_ante_scaling` (Lua) vs flat `booster_ante_scaling` (read, no writer) | C15 |
| inflation | `modifiers.inflation` (Lua challenge flag) vs `inflation_modifier` (read, no writer) vs `inflation` (amount) | C15 |
| The Eye history | `Blind.hands_used` (read by `debuff_hand`) vs `blind.hands` (dead write, `game.py:1938`). Latent: had the write hit `hands_used`, its all-False dict would block every hand under membership semantics. | — |
| Lucky trigger | `card.ability["lucky_trigger"]` vs top-level `card.lucky_trigger` | D04 |

### 1b-3. Mirrors: hand-refreshed copies of live values

`_handle_play_hand` copies these into `gs` right before scoring
(`game.py:612–634`) and nowhere else, so every other path reads stale values:
`money`, `hands_left`, `discards_left`, `discards_used`,
`current_round_hands_played`, `deck_cards_remaining`, `playing_cards_count`,
`stone_tally`, `steel_tally`, `enhanced_card_count`, `mail_card_id`,
`idol_card`, `ancient_suit`, `consumable_usage_tarot`. **Delete them all.**
Each becomes a grabber.

### 1b-4. Per-card fields Lua uses that `Card` lacks

| Field | Purpose in Lua | Finding |
|---|---|---|
| `unique_val` | nominal tie-break in sorting | D01 |
| `getting_sliced` | marks a joker being destroyed this pass (Madness, Dagger, Riff-raff guards) | D15 |
| `added_to_deck` | makes `add_to_deck` / `remove_from_deck` idempotent | D25 |
| `shattered`, `destroyed`, `removed` | destruction state inside a pass (Glass Joker counts shattered cards) | D17 |
| `lucky_trigger` (top level) | Lucky Cat | D04 |
| `ability.perma_debuff` | permanent debuff (perishable, `debuff_played_cards`); re-applied in `Card:update` | D25, C15 |
| `ability.couponed` | free-item status that survives repricing | D53 |
| `ability.wheel_flipped` | The Wheel face-down marker | C09 |
| `ability.discarded` | discard marker (`cardarea.lua:71`, cleared `state_events.lua:278`) | D31 |
| `ability.queue_negative_removal` | defers the Negative slot loss on removal (`card.lua:632, 689, 4733`) | C12, D24 |

`Card.facing` and `vampired` already exist. `vampired` is never cleared (C03).
`pinned` is challenge-only (scope).

### 1b-5. Per-frame derived values: these are the grabber list

Lua recomputes these every frame in `Card:update` (`card.lua:4126–4260`), so
in Lua they are always current. The engine either caches them in a snapshot or
never computes them. Each becomes a grabber computed at read time:

| Lua value (`card.lua` update) | Grabber | Finding |
|---|---|---|
| Temperance `ability.money` (sum of joker sell costs, capped) | `temperance_money(gs, card)` | — |
| Throwback `x_mult` (from `skips`) | `throwback_xmult(gs, card)` | (fixed once already) |
| Driver's License `driver_tally` | `enhanced_count(gs)` | — |
| Steel Joker `steel_tally`, Stone Joker `stone_tally` | `count_enhancement(gs, "m_steel")` | — |
| Cloud 9 `nine_tally` | `rank_count(gs, 9)` | D19 |
| Joker Stencil `x_mult` (empty slots + Stencils, **debuffed included**) | `stencil_xmult(gs, card)` | D10 |
| Swashbuckler `mult` (sell costs, **debuffed included**) | `swashbuckler_mult(gs, card)` | D10 |
| Wheel `eligible_strength_jokers`, Ectoplasm/Hex `eligible_editionless_jokers` (edition filter only, **no debuff filter**) | `editionless_jokers(gs)` | D39 |
| Blueprint/Brainstorm `blueprint_compat` | already `resolve_copy_targets` | — |
| `perma_debuff` → `debuff` | part of the debuff grabber | D25 |

These follow Lua's update loop, which iterates all cards without debuff
filtering. That is exactly why D10 and D39 exist: the Python versions added
filters that Lua's loop does not have.

### 1b-6. `Blind` and hand levels

- `Blind`: declare `prepped` as a field (Fish and Crimson Heart set it with a
  bare `setattr`, `game.py:2507`; D28, C09). Delete the dead `blind.hands`
  write. Other Lua `Blind` fields missing from Python are UI (`chip_text`,
  `loc_debuff_*`, `pos`, `children`, …).
- `HandLevels` already stores `level`, `chips`, `mult`, `played`,
  `played_this_round` and `visible`. Lua's `s_*`, `l_*` and `example` fields
  are static data or UI. The secret-hand bug (D13) is about *when* `visible`
  is set, not about storage.

### 1b-7. Not needed (UI, profile, animation)

`STOP_USE`, `PACK_INTERRUPT`, `chips_text`, `win_notified`, `subhash`,
`viewed_back`, `round_resets.loc_blind_states`, `current_round.round_text`,
`current_round.dollars_to_be_earned`, `current_round.current_hand.*_text`,
`previous_round.dollars`, `round_scores.*`, `hand_usage`, `max_jokers`,
`current_boss_streak`, `cards_played`, `starting_voucher_count`,
`first_used_hand_level` (no writer in vanilla either), `perscribed_bosses`
(debug/challenge), `challenge_tab`. `selected_back` is a `Back` object that the
engine builds from `selected_back_key` on demand. That is a representation
choice, not missing state.

### 1b-8. Design consequence: keep Lua's buffers as real state

The three buffers (`dollar_buffer`, `joker_buffer`, `consumeable_buffer`) exist
in Lua because effects are queued events. A synchronous engine might seem not to
need them. **Keep them anyway, as real state with Lua's exact semantics.** Lua
checks `#cards + buffer` in most places but not all, and Bull reads
`dollars + buffer` while Vagabond reads `dollars` alone. Reproducing the
buffers line for line is simpler and safer than proving, check by check, that
immediate application is equivalent. They get one writer each (the money ledger
and the effect applier from S1/S4), and a reset at the point where Lua's queued
reset event runs.

**Exit criterion for "state is rich enough":** a checked-in mapping assigns
every Lua `G.GAME` path, `Card` field and `Blind` field to exactly one of:
*stored at key K*, *grabber G*, or *out of scope (reason)*. A test runs
`engine_state_inventory.py` when the source tree is present and fails on any
unmapped path. It skips without the source, the same way the LuaJIT oracle
tests do.

---

## Part 2. Implementation issues (local bugs)

These are wrong in a single place and survive any refactor unchanged. "Lands"
says where the fix belongs: **now** = in a handler or helper body that the
overhaul does not rewrite, so it can go in immediately; **with Sx** = touches
code that is being restructured, so fix it inside that phase.

| ID | Issue | Fix | Lands | Pri |
|---|---|---|---|---|
| C01b | `record_play` called twice on successful hands (`scoring.py:565` and `game.py:699`) | Single call inside evaluation, before the blind-debuff branch | with S4 | P0 |
| C10b | Blue seal makes the Planet for the **most-played** hand; Lua uses **last played** | Use the last-played hand | with S4 | P0 |
| C11b | Mr. Bones never tests the 25% threshold; compares this hand rather than accumulated chips | Lua game-over check: `chips >= 0.25 * blind` on accumulated chips, in `end_round` | with S4 | P0 |
| C13b | Observatory: no held-consumable X1.5 pass in scoring | Add the joker-area/consumable pass Lua runs | with S4 | P1 |
| C14b | Magic/Ghost starting consumables never instantiated; Anaglyph `trigger_effect('eval')` never dispatched | Instantiate in `initialize_run`; dispatch on boss defeat | now | P1 |
| D01 | `_card_nominal` duplicated 3× (`hand_eval`, `card_area`, `card.py`), all differing from Lua `get_nominal` (Stone suit scaling, `unique_val` tiebreak) | One `get_nominal`; delete the copies | now | P0 (High Card component and ordering) |
| D02 | `is_face` checks `base.id`, not `get_id()`, so Stone faces count | Use `get_id()` | now | P0 (rare: needs a Stone face card) |
| D05 | Brainstorm in slot 0 copies slot 1 | Target `jokers[0]`; no-op when that is itself | now | P0 |
| D06 | Spurious `not ctx.blueprint` guards on Hiker, DNA, Burnt Joker, Perkeo | Remove (Sixth Sense keeps its guard) | now | P0 |
| D08 | Raised Fist picks the first tied minimum; Lua picks the last (`>=`) | `<=` scan | now | P0 |
| D09 | Blackboard fails on an empty held hand | `black == total` (0 == 0 passes) | now | P0 |
| D10 | Joker-on-joker pass skips debuffed **targets** (Baseball Card); Stencil and Swashbuckler exclude debuffed jokers | Iterate targets regardless of debuff (now); the tallies become grabbers (1b-5) | now / with grabbers | P0 |
| D11b | Matador main-pass branch (`blind.triggered`) missing | Add the branch; keep `triggered` assignments (Arm, Flint, card debuffs) | with S1/S4 | P0 |
| D13 | Secret hands become visible on level-up, not on play | Reveal on play in evaluation; `level_up_hand` does not reveal | now | P1 |
| D15b | Madness excludes `jokers[0]` instead of itself; no Eternal / being-sliced filter | Exclude self; apply Lua filters | now | P0 |
| D21 | Gift Card skips consumables | Include consumables | now | P0 |
| D23b | `selling_self` never dispatched (Luchador, Diet Cola, Invisible Joker); Verdant Leaf sale-disable missing | Dispatch through the S1 applier | with S1 | P0 (shop sells) |
| D32b | Amber Acorn shuffles once; Lua shuffles 3× (`aajk`) | 3 shuffles (RNG stream count) | now | P0 |
| D33b | Death picks highest `sort_id`; Lua picks the **rightmost highlighted card by hand position** | Pick by position. **Note:** the locked In-blind-merge design (CLAUDE.md) plans to bypass this rule with env-side direct construction. Fixing the engine rule does not conflict with that design, but its rationale paragraph ("sort_id … possible faithfulness gap") is now confirmed and should be updated. | with S3 | P1 (pack Death in s-track) |
| D34 | Ectoplasm always costs 1 hand size; Lua cost grows 1, 2, 3, … | Run-wide counter | now | P1 |
| D35b | Ankh copy keeps Negative and the original identity | Strip Negative; fresh identity (via `copy_card`) | with S3 | P1 |
| D38 | Grim draws a rank even though its list is Ace-only | Assign Ace; draw suit only (`grim_create`) | now | P0 (RNG stream) |
| D39 | Wheel, Ectoplasm and Hex exclude debuffed jokers from candidates | Filter by edition only, via the `editionless_jokers` grabber (1b-5) | with grabbers | P1 |
| D40 | Tarot use never counted (Fortune Teller); Black Hole recorded as a Planet / `last_tarot_planet` | Port `set_consumeable_usage` per set | now | P0 |
| D45 | Shop Tarot/Planet slots may roll Soul or Black Hole | `soulable=False` for shop slots | now | P0 |
| D48 | Hallucination key `hallucination` (Lua: `halu`+ante); rental stream drawn even when rentals are off | Fix the key; short-circuit the draw | now | P0 (RNG stream) |
| D51 | Overstock / Clearance redemption does not resize or reprice the current shop or sell values | Run Lua's shop-size and cost updates on redeem | now | P0 |
| D52b | Retcon recorded as free; Lua charges the normal $10 | Fix the price (the action itself is S5) | with S5 | P1 |
| D53 | Uncommon/Rare tag jokers not free; edition-tag coupon state incomplete | Set `couponed`; reprice through `set_cost` | with S3 | P1 |
| D54 | Orbital Tag picks its hand at redemption; Lua fixes it when the tag is offered and copies it for Double Tag | Store `orbital_hand` on the offered tag | now | P1 (needs SkipBlind, which s1 exposes) |
| D55 | Redeeming the Voucher Tag's extra voucher clears the ante voucher | Clear only for the non-extra voucher | now | P1 |
| — | Unseeded `math.random` choices (Charm/Meteor Mega variant, etc.) are deterministic in Jackdaw | Already in `real-balatro-compatibility-audit.md`; keep as documented equivalence or seed explicitly | — | P2 |

---

## Part 3. Scope decisions (not bugs until we decide they are in scope)

| Area | Findings | Recommendation |
|---|---|---|
| Challenges | C15, D43, D44 (second half) | **Out of scope for now.** Make `initialize_run(challenge=…)` **raise** for any challenge whose rules include an unimplemented modifier, instead of silently running a different game. Implement later only if challenges enter training. |
| Profile progression | D44 | **Out of scope.** Keep E07 (fully-unlocked profile) as a documented precondition. Make `apply_profile_to_game_state` raise on any non-fully-unlocked profile. |
| Endless numerics | D49 | **Out of scope** (win_ante=8). Add an assert at ante > 16 so nobody trusts it unknowingly. |
| `CardArea` helpers | D58 | **Delete** if the run loop does not use them (it uses raw lists). Otherwise mark private. A "source-mirroring" helper that does not mirror the source is a trap. |
| Statistics / telemetry | D59 | Fix the one gameplay-relevant piece (`jokers_purchased` counts non-jokers). Leave the rest of career and round_scores out of scope. |
| Other stakes / decks | C07, C14 | **In scope, P1.** They fall out of S2 almost for free, and s1/h2 may vary stake. |

---

## Part 4. Your inline annotations

- **C05 / Vagabond:** agreed. See S4 layer 2. The audit's "make money live"
  framing would break Vagabond. The fix is the committed/pending money model.
- **C08 / "ante-1 Wall should be impossible":** checked. The natural path is
  fine. `get_new_boss` (`blind.py:503`) filters by `boss.min <= max(1, ante)`,
  and The Wall has `min: 2` (`data/blinds.json`). `HandPlayAdapter` samples
  bosses through the same `get_new_boss` with the sampled ante
  (`hand_play_adapter.py:447`). The audit's ante-1 Wall was a forced fixture.
  To make it a guarantee rather than an accident, add (a) an assert in blind
  setting that a boss's `min` is at most the current ante (and showdown bosses
  only at showdown antes), unless an explicit `allow_forced_boss` test flag is
  set, and (b) a test that samples many antes through both `initialize_run` and
  `HandPlayAdapter` and checks every boss's min/showdown constraint. The C08
  bug itself (Chicot not calling `Blind.disable`) is real at ante ≥ 2 and stays
  in S3.
- **`state.py` "rental_rate — THIS IS NEVER CALLED, IN VANILLA OR HERE":**
  checked against the source (2026-10-06). Both sides **read** it: vanilla in
  `Card:calculate_rental` (`card.lua:2273`, `ease_dollars(-G.GAME.rental_rate)`),
  the engine at `economy.py:175`. What is true is that nothing **writes** it
  after init (`game.lua:1915` sets 3), in vanilla or here, so it is effectively
  a constant. Suggest rewording the note to "never mutated after init". The
  2026-07-18 rental fixes stand.
- **Vagabond (C05), confirmed against the source:** Vagabond reads committed
  `G.GAME.dollars` only (`card.lua:3744`). Bull reads
  `G.GAME.dollars + G.GAME.dollar_buffer` (`card.lua:3936`), and so does
  Bootstraps. Deferred payouts add to `dollar_buffer` and queue a reset event
  (`card.lua:1085–1086`). The Ox drains instantly with
  `ease_dollars(-G.GAME.dollars, true)` (`blind.lua:565`).

---

## Part 5. Phased plan

The order is driven by dependencies: build the measuring instrument first, then
the structures, then the procedures that use them, then the local cleanup that
depends on them.

### Phase 0. Instrument (no behavior change)

1. `balatro_source/` (1.0.1o) is at `~/Code/Code/balatro-strategy/balatro_source`
   on this Mac. Point the oracle tests at it, install LuaJIT, and make the 14
   skipped oracle tests run (they skip silently today).
2. Build the S6 structural tests: producer/consumer, reader/writer key lint,
   lifecycle-assignment lint, registry-through-`step()` coverage. They will fail
   loudly on the current code. Commit them as `xfail(strict)` with finding IDs.
3. Convert every C/D finding with a "Reproduced" example into an integrated
   `xfail(strict)` regression test through `step()` / `score_hand`.
4. **S4-a source sweep:** for every money, counter and flag consumer, record in
   a table which Lua view it reads (committed vs buffered, instant vs queued
   writer). This table is the spec for the money ledger. It is also where your
   Vagabond analysis gets confirmed or corrected against the source.
5. Build the headless Lua transition oracle (S6 item 2), starting with
   `evaluate_play` and `end_round`.

*Exit: every finding has a red test; the oracle runs `evaluate_play` on a
controlled state.*

### Quick wins (parallel track, Q)

The Part 2 rows marked **now**. They are in handler or helper bodies the
overhaul does not touch. Each gets a test that is red first. Batch them by
area (scoring handlers, consumables, RNG keys, shop/voucher/tag) so that each
batch's downstream blast radius can be written up once in
`engine_changes.md`. Runs alongside Phases 0–2.

### Phase 1. State completeness (Part 1b)

Make the state able to hold every value the game reads or writes, before
anything reads it through grabbers:

- Add the missing stored fields (1b-1): `dollar_buffer`, `joker_buffer`,
  `consumeable_buffer` (kept with Lua semantics per 1b-8), `last_hand_played`,
  a written `current_round.most_played_poker_hand`, `consumeable_usage` and
  `consumeable_usage_total` with all their sub-keys, `ecto_minus`,
  `orbital_choices`, `facing_blind`.
- Collapse every alias in 1b-2 to one canonical key, and write the load-time
  migration so old blobs restore into the canonical keys.
- Declare the missing `Card` fields (1b-4) and `Blind.prepped` (1b-6); delete
  the dead `blind.hands` write.
- Resolve the two "verify" rows (`shop_free` / `shop_d6ed`, `last_blind`).
- Check in the path → {key, grabber, out of scope} mapping and its test.

Adding a field changes no behaviour by itself: the writers and readers come in
the later phases. Unwired new fields are expected and allowed until Phase 2.

*Exit: the inventory test passes, with zero unmapped Lua paths.*

**STATUS 2026-10-06: EXIT MET** (branch `engine-phase1-state`). Implementation
tickets ran through `codex exec` and were reviewed here.

- **Mapping:** `jackdaw/engine/state_map.py` assigns every scanned Lua
  `G.GAME` path (176), `Card` field, `Blind` field and `ability` key (49) to
  `Stored` / `Grabber` / `OutOfScope`. `tests/engine/test_state_map.py`
  checks that every non-lazy stored path resolves on a fresh run. When the
  source is present (`BALATRO_LUA_SRC`), it also checks that nothing is
  unmapped. Zero unmapped at time of writing.
- **Added storage** (1b-1, 1b-4, 1b-6): `dollar_buffer`, `last_hand_played`,
  `consumeable_usage_total`, `orbital_choices`, `facing_blind`, `shop_free`,
  `shop_d6ed`; `Card.added_to_deck/getting_sliced/shattered/destroyed/removed/lucky_trigger`
  (plain defaults, so old pickled cards load); `Card.unique_val` as an
  order-preserving property over `sort_id` (Lua's node ID cannot be
  reproduced, and only its order is observable); `Blind.prepped`. The dead
  The Eye `blind.hands` write and the dead `tags` init key are deleted.
- **Aliases collapsed** (1b-2). Each fix has a regression test that was seen
  failing first (`tests/engine/test_state_aliases.py`). This pulls these fixes
  forward from Phase 2: C04 (Oops/probabilities; Oops now scales every entry,
  as in Lua), C07 (stake sticker flags), C14 Green half, D26 (Chaos also
  recalculates reroll cost, as in Lua), D04 (Lucky trigger, cleared per
  repetition), D40 spelling only, challenge `booster_ante_scaling` /
  `inflation`, and Omen Globe's flat flag write.
- **Writers added:** `last_hand_played` (start of play) and
  `current_round.most_played_poker_hand`. The second ports Lua's boss-defeat
  loop. It does NOT reuse `HandLevels.most_played`, whose default differs. The
  Ox still reads live (D29, Phase 5).
- **`migrate_state(gs)`** in `state.py` maps every removed alias on restore.
  It is called by both adapters' `restore_state` and by
  `harvest_restore.restore_state`. D-data still says re-harvest; migration is
  only the safety net.
- **Verify rows resolved:**
  - `last_blind` is **equivalent** and becomes a grabber. Lua's eval tags
    (Investment, Anaglyph) run synchronously in `evaluate_round`, before the
    queued `Blind:defeat` event resets `last_blind`, so it always equals the
    just-defeated blind (`gs["blind"]`).
  - `shop_free` / `shop_d6ed` were **NOT equivalent** (new finding).
    `fire_tag_context` consumed every matching tag, so a second D6 or Coupon
    tag was wasted in the same shop. Lua applies one per shop visit and keeps
    the second for the next shop. The flags are now stored, gate the two
    handlers, and are cleared at cash-out.
- **Deliberately deferred to Phase 2 (grabbers, not storage):**
  - the `packs.py` `has_omen_globe` / `has_telescope` readers (TODO comments
    in place; Omen Globe and Telescope stay inert until then);
  - `has_astronomer`, `deck_enhancements`, `most_played_hand` (Telescope's
    live pick);
  - `playing_card_count` vs `playing_cards_count`, and the 1b-3 mirrors.
- **Naming exceptions recorded in the map, not renamed:**
  `pack_choices`→`pack_choices_remaining` (env obs reads it),
  `G.GAME.tags`→`awarded_tags`, `Blind.hands`→`hands_used`.

### Phase 2. Grabbers (S2)

Every read goes through a grabber. Canonical key registry in `state.py`;
grabbers for every derived value in 1b-5 plus `has_voucher`, `find_joker`,
`deck_enhancements`, `probability_normal`, the `rules(gs)` view feeding
`is_face` / `is_suit`, and the two money views (`money_committed`,
`money_with_pending`); delete the 1b-3 mirror keys and read through grabbers
instead; per-scoring-call memoisation where the solver's inner loop needs it
(computed once per `score_hand` call, then discarded, so it can never go
stale); lint that rejects direct `gs[...]` reads in handlers.
*Fixes C04, C07, C13 (flags), C14 (Green), D03, D04, D10 (tallies), D19, D26,
D39, D40 (tally), D42 (`deck_enhancements`). Reader/writer lint goes green.*

**STATUS 2026-10-06: EXIT MET** (branch `engine-phase1-state`, commits
`ff8091e`..`1e4662e`). P2-1 to P2-3 ran through `codex exec` and were reviewed
here. Codex then hit its usage cap mid-P2-3, so P2-3 was verified and finished
here, and P2-4 and P2-5 were implemented here.

- **`jackdaw/engine/read.py`**: the getters and `Rules`, plus `StateView`.
  `StateView` is a lazy view over live state, cached for one scoring call,
  and it exposes every old `GameSnapshot` attribute name.
  - Every production snapshot site now passes a `StateView`, so no call site
    assembles its own values any more. That fixes the starved snapshots: D19
    Delayed Gratification, and Cloud 9's never-written `nine_tally`.
  - The 1b-3 mirrors are deleted. Played cards now sit in
    `played_cards_area` while scoring, as in `G.play`.
- **Fixed, each with a test seen failing first.**
  - Mirrors: the solver scored every hypothetical play against the previous
    real play's leftover mirror values (dollars, tallies, deck counts).
  - C13: Omen Globe, Telescope and Astronomer.
  - D42: the enhancement pool gate.
  - D10: Stencil and Swashbuckler count debuffed jokers. Stencil fires only
    with an empty slot (`card.lua:3967`), which the review caught.
  - D39: Wheel, Ectoplasm and Hex candidates include debuffed jokers.
  - D03 / D02: `is_face` and `is_suit` take a REQUIRED `Rules`. Fixes Ride the
    Bus, Faceless Joker, Flower Pot and Seeing Double under Smeared, the
    suit-debuff bosses under Smeared, and stops a Stone King counting as a
    face.
  - D40: Lua's `set_consumeable_usage` is ported. Tarots are now counted, and
    Black Hole counts as a Spectral (not a Planet).
  - D19 Satellite: counts distinct Planet keys.
- **Registry and lints:**
  - `state.STATE_KEYS` and `GROUP_KEYS` register every key.
  - `tests/engine/test_state_registry.py` fails on: an unregistered key (a new
    alias), a key that is read but never written, a dead registry entry, or a
    handler reading raw state.
- **The reader/writer scan found three more dead keys:**
  - **`observation.py` read the flat `four_fingers` / `shortcut` / `smeared` /
    `splash` keys, which nothing ever wrote.** Global-context index 29 and the
    Splash term of card feature 9 were constant for every state. They now read
    `read.rules(gs)`. **This is an obs VALUE change for every existing
    checkpoint** (Part 6: warm-start only anyway).
  - `played_hand_types` was a mirror re-synced at three call sites. It is now a
    getter.
  - The challenge Inflation pass looped over `all_shop_cards`, which was never
    written. It now uses `read.all_cards`.
- **Performance:**
  - `score_hand`, worst case (5 tally jokers, 52-card state): 113 → 118 µs.
    Plain boards are unchanged.
  - 45 fixed solver decisions: 2.02 s → 1.97 s.
- **Residuals, owned by later phases:**
  - Hypothetical solver plays see pre-play counters (`hands_left` not yet
    decremented). This is what each handler sees during `evaluate_play`, which
    is Phase 5 / S4-a.
  - Each money consumer still reads committed dollars. The Bull vs Vagabond
    split (C05) is Phase 5.
  - Hand-DETECTION flags (`get_flush(smeared=)`, solver templates) still travel
    as booleans derived once per joker list and bridged to `Rules`.
  - The non-handler procedural code (`game.py` flows) still reads `gs`
    directly. Phases 4–5 rewrite it.
  - `shop.buy_card` is an unused duplicate of `_handle_buy_card` (Phase 7
    helper deletion).

### Phase 3. Lifecycle (S3)

The `lifecycle` module; route every creation, copy, removal, edition, ability,
base, debuff and sticker change through it; `Blind.disable` / `Blind.defeat`;
pool exclusion by current existence (per D-S3a); area routing with pack-open
hands; notifications on every path.
*Fixes C06, C08, C10 (Planet init), C12, C13 (Magic Trick/Illusion base),
D16, D17, D20, D24, D25, D33, D35, D36, D37, D41, D42 (editions on created
jokers), D47, D53. Lifecycle lint goes green.*

**STATUS 2026-10-08: EXIT MET** (branch `engine-phase1-state`, P3-1 `dc6b7b3`,
P3-2 `9822d36`, P3-3 after it). Three tickets ran through `codex exec` and
were reviewed and tightened here. Every listed finding has a regression test
that was seen failing first, except where an earlier ticket in the phase had
already fixed it (noted per ticket in the commit messages).

- **Setters** (`card.py`, `blind.py`): `set_edition`, `set_cost`, `set_seal`,
  `set_rental`, `set_perishable`, `set_debuff` take `gs` as a REQUIRED
  argument. There is no fallback to an empty state, because a debuff toggled
  without `gs` would silently skip the passive removal (D25).
  `add_to_deck` / `remove_from_deck` port Lua 564–704: an `added_to_deck`
  guard, `from_debuff`, `queue_negative_removal`, `ease_discard`, Chicot
  disabling an active boss, Astronomer repricing, and a blind debuff refresh.
  `Blind.disable(gs)` ports `blind.lua:356`, including The Manacle's one-card
  draw. Sticker storage is the `Card` fields only. `set_base` resets
  `times_played` (Lua) and keeps `suit_nominal_original`.
- **`lifecycle.py`**: `PoolTracker`, `copy_card`, `emplace`, `remove`,
  `add_playing_cards`, `destroy_playing_cards`, `fire_joker_context`.
  - `emplace` inserts deck cards at index 0 (Lua index 1, the bottom of the
    draw pile; the position feeds later shuffles). It does not sort the hand.
  - Pool exclusion: release on `Card:remove` unless an owned copy exists.
    This is NOT an exact "currently exists" equality: removing one of two
    displayed same-named cards (two Arcana packs) clears the key while the
    sibling is still shown, exactly as Lua does. The tested invariant is
    `owned keys ⊆ used_jokers ⊆ existing keys`. Playing-card centers are not
    tracked, because no pool reads them.
- **Notifications** (D17) fire once per batch, as Lua's
  `playing_card_joker_effects` does. Glass Joker counts only `shattered`
  cards. Scoring marks destroyed cards and notifies; the removal itself
  happens in the play handler, never inside `score_hand`. The solver calls
  `score_hand` on cloned cards with the LIVE `gs`.
- **Lint**: `tests/engine/test_engine_lint.py` (AST, `jackdaw/engine` +
  `jackdaw/env`) bans card-field assignment, bare `Card(`, `copy`/`deepcopy`
  (import aliases included), and gs-less `set_ability`/`set_base`/… outside
  `card.py`, `card_factory.py` and `lifecycle.py`. The allowlist has 3
  entries, keyed by function name rather than line number, and a stale entry
  fails the test.
- **Structural rollouts**: `tests/engine/_rollout.py` is a greedy-hand,
  shop-heavy seeded driver (20 runs, >1000 steps, hundreds of buys, sells,
  rerolls and packs). Random play dies in the first blind and never reaches
  the shop. The driver checks the pool bounds, `added_to_deck` on owned cards,
  no card in two areas, and the slot identities after every step.
- **Boss guard** (Part 4 C08 note): the selected boss must satisfy
  `min <= max(1, ante)` and the showdown rule, unless `allow_forced_boss` is
  set. The guard is skipped after Hieroglyph or Petroglyph, which lower the
  ante after the boss was chosen. The shop-heavy driver found this crash at
  ante 0.
- **Deferred:**
  - `Blind.defeat` (D30, D32) and the "disable beats the boss → NEW_ROUND"
    event go to Phase 5.
  - DNA, Sixth Sense, Certificate and 8 Ball creation, and the
    `first_hand_drawn` dispatch (C02), go to Phase 4.
  - Held-card retriggers (C10) go to Phase 5.
  - Room checks still run AFTER the RNG draw for some creations (D46) and
    move to Phase 4. The phase's own change: a no-room card is now `remove`d,
    so it releases its pool key.
- **Data impact (Part 6)**: shop sequences change (C06 release, D42 edition
  rolls, the deck insert position), as do the obs `times_played` feature and
  the economy. No checked-in fixture changed. Re-harvest as planned.

### Phase 4. Effect pipeline (S1)

Effect vocabulary; single applier with room reservation and raise-on-unknown;
single context builder; delete the InitVar compatibility path; every call site
(scoring phases, setting_blind, end_of_round, shop contexts, consumables,
discard, selling, booster open, tags) goes through it.
*Fixes C02, D07, D11 (Matador main), D15 (Dagger), D23 (selling_self,
Verdant), D35 (ordering), D46, C14 (Anaglyph dispatch). Producer/consumer test
goes green.* Performance gate: the solver's per-decision wall time is within
about 10% of today's. Measure it with the existing validation harnesses' timing
output.

### Phase 5. Sequencing (S4)

Line-faithful ports of `evaluate_play`, `play_cards_from_highlighted`, discard
(with Hook), shared draw, `set_blind` / `new_round`, `end_round`,
`Blind:defeat`, `evaluate_round` / cash-out; the money ledger from the S4-a
table. Remove `individual_hand_end`.
*Fixes C01, C03, C05, C09, C10 (retriggers), C11, D12, D14, D18, D22, D28,
D29, D30, D31, D32. Oracle observation-point tests go green.*

### Phase 6. Legality (S5)

`legality.py`; full vanilla action set in the engine (sell anywhere legal, boss
reroll with the Retcon / Director's Cut pricing, Spectral picks, consumable use
in packs, buy-and-use); remove agent-convenience restrictions from the engine;
confirm each env's masks still produce exactly the same legal sets as before
(pin this with a test per env, so checkpoints are unaffected).
*Fixes D23 (phases), D27, D50, D52, D56, D57.*

### Phase 7. Scope items and close-out

The Part 3 raises and asserts; CardArea deletion; the remaining Part 2 rows;
rerun the BalatroBot live harness end to end; a final audit diff (re-run the
audit's probe, `.scratch/engine_audit_probe.py`, on the Windows machine).

### Parallelism

Phase 1 comes first because Phases 2–5 all assume the fields exist. After
it, Phases 2 and 3 can run in parallel worktrees. They touch different code,
with one seam: `set_edition` writing a negative slot through the S2 key. Phase
4 needs both. Phase 5 needs Phase 4 because the sequenced procedures call the
applier. Phase 6 is independent of 4 and 5 except D23's `selling_self`, so it
can start after Phase 2. The quick-win track (Q) runs throughout.

---

## Part 6. Downstream blast radius (read before merging any phase)

Rules already established in this project apply. Computation fixes are
inherited by re-scoring. **Stored-state** fixes are **not**: a pickled blob
keeps its stale cache (the C2 capture-skew rule). Labels are never re-scored.

| Artifact | Effect of the overhaul |
|---|---|
| Hand-solver labels (all stages, v3 shards, stage5 harvested) | **Invalid.** C01 (Supernova, Obelisk, DNA, Sixth Sense counters), C03 (Vampire), C04 (Oops: probabilistic jokers, Lucky, Glass), C05 (Bull), D05, D06, D08, D09, D10 and more change scores or legal outcomes directly. Full regen. |
| Harvest corpus (`data/harvest_s0`, blobs) | Stale **stored state**: `used_jokers` contains sold and seen jokers (C06), no `deck_enhancements`, probability keys aliased, stale Turtle Bean `hand_size`, Pillar markers, and so on. **Recommended: re-harvest** after the overhaul. It is cheap (seconds per run, no solving) and avoids writing a lossy migration. If the old corpus must be reused, the Phase 2 key migration plus a per-field repair audit is required. |
| Shop reservoir (`reservoir.pkl`) | Same as harvest. Rebuild. |
| s0 / s1 checkpoints | Trained against a different game (shop pools via C06/D45/D51, economy via D18/D21, Rocket, Gift Card, and so on). Usable as `--init-from` warm starts. The obs schema is unaffected unless S2 renames something the encoder reads (check `shop_obs.py` and `observation.py` for every key they read). Values and policies need retraining. |
| h0.5 / h1 BC + PPO | Same: warm-start only. The h1 pipeline (regen, then BC, then PPO) has to run after the overhaul anyway. |
| `V_curve(ante, $)`, cashout mirror | `V_curve` is derived from the s0 critic, so recompute it. The cashout mirror replays the engine's own `CashOut`, so it tracks automatically (re-verify after D18/D22). |
| Oracle fixtures (`tests/fixtures/*`) | Shop and RNG fixtures change under C06 and D48. Regenerate them **from the Lua oracle**, never from our own engine (that would be self-referential). |

**Sequencing against the training roadmap.** CLAUDE.md marks this work CRITICAL
(9/4/2026). Labels and harvests are worthless until Phases 2–5 land. So:
**freeze new label generation and harvesting until Phase 5 exits**. Training
runs already in flight can finish as diagnostics but should not be promoted.

---

## Part 7. Decisions needed

Status as of 2026-10-06: every decision below was agreed by the user in the planning session. Target shape in the user's words: "setters, getters, a clean scoring pipeline and correct buffers; tests on the OOP". Class design: `docs/engine_class_design_2026-10-06.md`.

| # | Decision | Recommendation | Status |
|---|---|---|---|
| D-S1 | Effect representation | Typed effect dataclasses, one applier, raise on unknown | **DECIDED 2026-10-06**: an `Effect` base class with one subclass per effect kind |
| D-S2 | State representation | Keep dict + canonical registry + derived accessors + lint (not a typed `GameState` migration) | **DECIDED 2026-10-06**: keep the dict; all reads through grabbers |
| D-S3a | Pool exclusion lifetime (C06) | Match Lua ("currently exists"), accepting shop-sequence and fixture churn | **DECIDED 2026-10-06**: match Lua (the current engine is bugged) |
| D-S4 | Event-time model | Line-faithful synchronous ports + minimal committed/pending money ledger, scoped by the S4-a source sweep | **DECIDED 2026-10-06**: clean scoring pipeline + correct buffers |
| D-S5 | Legality boundary | Engine = full vanilla action set; env masks hold agent restrictions | **DECIDED 2026-10-06** |
| D-scope | Challenges / profiles / endless / CardArea / stats | Out of scope; raise or assert instead of silently diverging (Part 3) | **DECIDED 2026-10-06** |
| D-data | Downstream data | Freeze label and harvest generation until Phase 5; re-harvest rather than migrate | **DECIDED 2026-10-06** |
| D-rental | `state.py` rental_rate note | Resolved: read in vanilla and here, never written after init; reword the note (Part 4) | resolved |
| D-buffers | Keep Lua's `dollar_buffer` / `joker_buffer` / `consumeable_buffer` as real state | Yes, with Lua semantics (Part 1b-8) | **DECIDED 2026-10-06** |
| D-fields | Add the missing state and `Card` fields before grabbers (Phase 1) | Yes | **DECIDED 2026-10-06**: state completeness is Phase 1, before grabbers |

---

## Appendix. Full finding → category map

| Finding | Category | Phase | Pri |
|---|---|---|---|
| C01 hand counters / double record | S4 + local C01b | 5 | P0 |
| C02 DNA / Sixth Sense / Certificate / 8 Ball / Hallucination creations | S1 (Certificate: S4 `first_hand_drawn` dispatch) | 4 | P0 |
| C03 Vampire timing and double XMult | S4 | 5 | P0 |
| C04 Oops probabilities | S2 | 2 | P0 |
| C05 money during scoring (Vagabond note) | S4 (money ledger) | 5 | P0 |
| C06 pool exclusion lifetime | S3 (+ D-S3a) | 3 | P0 |
| C07 stake sticker flags | S2 | 2 | P1 |
| C08 Chicot disable | S3 | 3 | P0 |
| C09 face-down bosses | S4 (shared draw) | 5 | P0 |
| C10 EOR held-card retriggers / Blue-seal Planet | S4 + S3 + local C10b | 5 | P0 |
| C11 Mr. Bones | S4 + local C11b | 5 | P0 |
| C12 consumable editions | S3 | 3 | P0 |
| C13 Telescope / Omen / Observatory / Astronomer / Magic Trick | S2 + S3 + local C13b | 2/3/5 | P0 (Astronomer), P1 (rest) |
| C14 Magic / Ghost / Green / Anaglyph | S2 + S1 + local C14b | Q/2/4 | P1 |
| C15 challenges | scope | 7 | P2 |
| D01 nominal | local | Q | P0 |
| D02 Stone face | local | Q | P0 |
| D03 Pareidolia / Smeared threading | S2 (`rules` view) | 2 | P0 |
| D04 Lucky Cat field | S2 | 2 | P0 |
| D05 Brainstorm leftmost | local | Q | P0 |
| D06 blueprint guards | local | Q | P0 |
| D07 Loyalty | S1 (context builder) | 4 | P0 |
| D08 Raised Fist ties | local | Q | P0 |
| D09 Blackboard empty | local | Q | P0 |
| D10 debuffed joker targets / tallies | local + grabbers (tallies) | Q/2 | P0 |
| D11 Matador | S1 + local | 4/5 | P0 |
| D12 blocked-hand after pass | S4 | 5 | P0 |
| D13 secret-hand visibility | local | Q | P1 |
| D14 Burglar | S4 | 5 | P0 |
| D15 Dagger / Madness | S1 + local | Q/4 | P0 |
| D16 Marble front | S3 | 3 | P0 |
| D17 created/destroyed notifications | S3 | 3 | P0 |
| D18 EOR dollar-bonus timing | S4 | 5 | P0 |
| D19 Cloud 9 / Satellite / Delayed Gratification | S2 + S1 | 2/4 | P0 |
| D20 Turtle Bean decay | S3 | 3 | P0 |
| D21 Gift Card | local | Q | P0 |
| D22 rental / loss lifecycle | S4 | 5 | P1 (needs rentals) |
| D23 selling_self / sell phases | S1 + S5 | 4/6 | P0 / P1 |
| D24 Perkeo copy | S3 | 3 | P0 |
| D25 debuff passives / perishable | S3 | 3 | P0 |
| D26 Chaos / Drunkard current-round | S2 | 2 | P0 |
| D27 Cerulean Bell | S5 | 6 | P0 |
| D28 Crimson Heart prepped | S4 | 5 | P0 |
| D29 The Ox | S4 | 5 | P0 |
| D30 Pillar reset | S4 | 5 | P0 |
| D31 Hook discard pipeline | S4 | 5 | P0 |
| D32 Amber Acorn / defeat cleanup | S4 + local | Q/5 | P0 |
| D33 Death | S3 + local | 3 | P1 |
| D34 Ectoplasm | local | Q | P1 |
| D35 Ankh | S1 + S3 | 3/4 | P1 |
| D36 consumable debuff refresh | S3 | 3 | P0 |
| D37 pack-created card placement | S3 | 3 | P0 |
| D38 Grim RNG | local | Q | P0 |
| D39 edition consumables + debuff | grabber (editionless jokers) | 2 | P1 |
| D40 Tarot usage / Black Hole | local + S2 | Q/2 | P0 |
| D41 suit change nominal / Checkered key | S3 | 3 | P1 |
| D42 created-joker editions / deck_enhancements | S3 + S2 | 2/3 | P0 |
| D43 bans | scope | 7 | P2 |
| D44 profile / challenge vouchers | scope | 7 | P2 |
| D45 shop soulable | local | Q | P0 |
| D46 generate before room check | S1 | 4 | P0 |
| D47 sticker setters | S3 | 3 | P1 |
| D48 RNG keys | local | Q | P0 |
| D49 endless numerics | scope | 7 | P2 |
| D50 Credit Card affordability | S5 | 6 | P0 |
| D51 voucher shop resize/reprice | local | Q | P0 |
| D52 Retcon / Director's Cut | S5 + local | 6 | P1 |
| D53 coupon tags | S3 + local | 3 | P1 |
| D54 Orbital Tag | local | Q | P1 |
| D55 Voucher Tag | local | Q | P1 |
| D56 step accepts illegal actions | S5 | 6 | P2 (agents mask) |
| D57 pack picks / pack actions | S5 | 6 | P0 (Spectral picks) / P1 |
| D58 CardArea helpers | scope | 7 | P2 |
| D59 statistics | scope + local | 7 | P2 |
