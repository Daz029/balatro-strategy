# S4-a event and state timing sweep

Source authority: Balatro 1.0.1o (`version.jkr`: `1.0.1o-FULL`,
`PROD_PC_Console`).  Audited against branch `engine-phase1-state` at
`2bdbcc2`.  Lua paths below are relative to the Balatro source root; Python
paths are relative to this repository.  `NEW-P5-1-*` labels are observations
from this sweep, not engine changes.

This document covers gameplay-visible state in and below
`play_cards_from_highlighted`, `evaluate_play`,
`discard_cards_from_highlighted`, `draw_from_deck_to_hand` / `draw_card`,
`new_round`, the listed `Blind` methods, `end_round`, `evaluate_round`, and
`cash_out`.  Animation-only writes are collapsed into ordering barriers, but
every gameplay dispatch and gameplay-state mutation is listed.

## 1. Event model

### Queues and insertion

`EventManager` owns five independent arrays: `unlock`, `base`, `tutorial`,
`achievement`, and `other` (`engine/event.lua:109-116`).  `add_event` defaults
to `base`; normal insertion appends and `front=true` inserts at index 1
(`engine/event.lua:122-129`).  The manager traverses each live array from
index 1 upward and removes completed events in place (`engine/event.lua:171-193`).
Lua `pairs` supplies no cross-queue ordering guarantee, so only order within
one queue is usable by the engine port.

An event appended by another event's `func` is appended behind every event
already in that queue.  Because traversal uses the live array, it can be
visited in the same update only if no earlier blocking event prevents it.
In practice the parent is normally blocking, so its children wait until the
next manager update.  `front=true` is the sole preemption mechanism; none of
the scoped gameplay paths uses it.

### Triggers and blocking

| Trigger | Completion rule | Source |
|---|---|---|
| `immediate` (default) | Calls `func()` on its first eligible update; time is immediately done. | `engine/event.lua:5-20,98-102` |
| `after` | Waits until `created/start time + delay`, then calls `func()`. | `engine/event.lua:51-57` |
| `before` | Calls `func()` on every eligible update until it returns true, but cannot be removed until its delay has elapsed. | `engine/event.lua:92-97` |
| `ease` | Captures the referenced value at first handling, interpolates it until the delay ends, then writes the exact endpoint. | `engine/event.lua:25-36,58-86` |
| `condition` | Calls `func()` until true; its time requirement is always satisfied. | `engine/event.lua:37-44,88-91` |

`blocking` and `blockable` both default to true (`engine/event.lua:7-15`).
The first handled blocking event marks the queue blocked for the rest of that
manager update—even if it completed and was removed.  Later `blockable=false`
events may still run; ordinary events wait for the next update
(`engine/event.lua:175-190`).  Paused events are skipped as specified at
`engine/event.lua:50-51,183-185`.

### Synchronous-port rule and flush points

The synchronous engine may replace animation time with the following ordering
rule:

> Execute a scoped Lua function synchronously to emit state changes and FIFO
> events.  Preserve emission-time writes immediately.  At each flush point,
> execute the queued gameplay events in base-queue FIFO order; events emitted
> by a callback go behind all already-emitted events.  Do not enter the next
> gameplay state until the events that precede its state-transition event have
> flushed.

The required Phase 5 flush points are:

| ID | Flush point | Exact consequence | Source |
|---|---|---|---|
| F1 | Before `evaluate_play` | Commit the queued hand decrement, move selected cards to play, run Hook's discard dispatch and its queued moves, and commit Tooth costs. | `state_events.lua:465-515`; `blind.lua:464-504` |
| F2 | After `evaluate_play`, before the HAND_PLAYED win/continue decision | First run `draw_from_play_to_discard` and increment both hand counters; then run events emitted inside scoring, including dollar commits/buffer clears, creations/destructions, chip easing, and the post-play permanent-debuff event. | `state_events.lua:509-534,657-662,719-730,836-839,985-995,1029-1084`; `game.lua:3187-3204` |
| F3 | End of a voluntary discard, before redraw | Run per-card Purple/Joker events, queued card moves, money/cost events, discard decrement, and the DRAW_TO_HAND transition.  A Faceless payout is nested and therefore lands after first-level discard events but before the draw callback. | `state_events.lua:379-446`; `card.lua:2858-2869`; `game.lua:3208-3244` |
| F4 | Each draw batch | Run each `draw_card` event FIFO, then the queued SELECTING_HAND / `Blind:drawn_to_hand` event.  `first_hand_drawn` is dispatched synchronously after draw events are *queued*, but before they execute. | `common_events.lua:386-423`; `game.lua:3219-3241` |
| F5 | End of `new_round`, before the first draw | Resolve `Blind:set_blind` events and setting-blind effects (including nested Chicot/Cartomancer), then the shuffle/state event; nested effects already in the queue still precede the subsequently queued draw callback. | `state_events.lua:290-352`; `blind.lua:119-150,190-204`; `card.lua:2491-2601` |
| F6 | End of `end_round`, before round evaluation | Commit EOR Joker effects, rental, held-card repeats, card movements, ante/reset events, and nested removals.  The final first-level event enters ROUND_EVAL; nested removals already queued run before the round-evaluation UI callback. | `state_events.lua:87-287` |
| F7 | End of `evaluate_round`, before cash-out is enabled | Build/animate the reward rows, run the queued `Blind:defeat`, and finally write `current_round.dollars` in the bottom-row event. | `state_events.lua:1135-1208`; `common_events.lua:927-1093` |
| F8 | Cash-out | Run the shop/counter reset event, commit the one queued `ease_dollars(current_round.dollars)`, then snapshot `previous_round.dollars`. | `button_callbacks.lua:2912-2955` |

The Python engine has no general event manager.  `EffectQueue.apply()` applies
effects immediately and resets only its local slot reservations
(`jackdaw/engine/effects.py:54-101`); state transitions are hand-written in
`jackdaw/engine/game.py`.  Phase 5 therefore needs the flush points above, not
animation delays or a frame scheduler.

## 2. Money reads and writes

`ease_dollars(x)` queues an immediate event; only its callback writes
`G.GAME.dollars`.  `ease_dollars(x, true)` writes in the caller immediately
(`functions/common_events.lua:68-108`).  `ease_dollars` never touches
`dollar_buffer`: the producer must reserve and clear it explicitly.

“Commit” below means the applicable F1-F8 flush.  Cash-out row values are
pending earnings, not `dollar_buffer`.

| Lua site | Who | Access and view | Write / buffer / commit | Python site and verdict |
|---|---|---|---|---|
| `card.lua:1068-1088`; `state_events.lua:719-724` | Gold seal; Lucky card money; enhancement `p_dollars` | No balance read. | Producer adds payout to `dollar_buffer` and queues a zero; scorer queues `ease_dollars(ret)`.  Both settle at F2. | `card.py:807-829`, `scoring.py:617-659`, `game.py:650-652`: payout is aggregated and directly committed after scoring; buffer is never written. **Out of order: C05.** |
| `card.lua:3150-3157`; `state_events.lua:726-730` | Golden Ticket | No balance read. | Adds configured dollars to buffer, queues its clear, returns dollars; scorer queues commit at F2. | `jokers.py:1211-1217`, `scoring.py:625-659`, `game.py:650-652`: amount matches; buffer/timing missing (**C05**). |
| `card.lua:3175-3183`; `state_events.lua:726-730` | Business Card | No balance read. | On successful roll adds $2 to buffer, queues clear, returns $2; commit F2. | `jokers.py:1220-1233`, `scoring.py:625-659`, `game.py:650-652`: amount/RNG match; buffer/timing missing (**C05**). |
| `card.lua:3224-3231`; `state_events.lua:726-730` | Rough Gem | No balance read. | Adds configured dollars to buffer, queues clear, returns dollars; commit F2. | `jokers.py:678-684`, `scoring.py:625-659`, `game.py:650-652`: amount matches; buffer/timing missing (**C05**). |
| `card.lua:3302-3317`; `state_events.lua:844-848` | Reserved Parking | No balance read. | Held-card success adds dollars to buffer and queues clear; returned dollars queue at F2. | `jokers.py:1236-1251`, `scoring.py:697-723`, `game.py:650-652`: amount matches; buffer/timing missing (**C05**). |
| `card.lua:3491-3499` | To Do List | No balance read. | Calls queued `ease_dollars` itself, then adds the same amount to buffer and queues clear.  Commit F2; the returned `dollars` is display data, not reapplied by `evaluate_play`'s before pass (`state_events.lua:627-638`). | `jokers.py:1308-1346`, `scoring.py:562-578`, `game.py:650-652`: paid once, but deferred only as a result aggregate and never buffered (**C05**). |
| `card.lua:2735-2745,3719-3728` | Matador | Reads `blind.triggered`, not money. | Both debuffed and main branches queue `ease_dollars`, then add buffer and queue clear; commit F2. | `jokers.py:1354-1363`, `scoring.py:482-500,735-757`: both branches exist, but money is aggregated without buffer (**C05**); Ox cannot trigger it in time (**D29**). |
| `card.lua:3743-3755` | Vagabond | Reads committed `G.GAME.dollars` only. | No money write; reserves a consumable and queues creation. | `jokers.py:2072-2078` uses `StateView.money`; the read kind is correct (`read.py:392-398`), but Python has not committed Ox at this point and has no Lua buffer timing (**C05/D29**). **Confirmed:** queued Gold/Matador money is invisible; Ox's instant drain is visible. |
| `card.lua:3936-3940` | Bull | Reads `dollars + dollar_buffer`. | Read only. | `jokers.py:1008-1014` reads `ctx.game.money` (committed), despite `money_with_buffer` existing at `read.py:27-29,396-398`. **C05, concrete mismatch.** |
| `card.lua:4046-4050` | Bootstraps | Reads `dollars + dollar_buffer`. | Read only. | `jokers.py:1116-1124` also reads committed `ctx.game.money`. **C05, concrete mismatch.** |
| `blind.lua:560-567` | The Ox | Reads committed dollars. | `ease_dollars(-G.GAME.dollars, true)` drains immediately inside `debuff_hand`, before before/card/Joker scoring.  No buffer mutation. | `game.py:663-679` drains after `score_hand`, after scoring dollars, and targets dynamically recomputed history. **D29.** |
| `blind.lua:497-504` | The Tooth | No balance read; $1 per played card. | Outer `after` event emits one queued `ease_dollars(-1)` per played card; all commit at F1 before actual `evaluate_play`.  No buffer. | `game.py:2075-2077` subtracts the batch synchronously before score. Final pre-score value matches; intermediate per-card commits are collapsed. |
| `card.lua:2802-2811` | Trading Card | No balance read. | Queues `ease_dollars(extra)` synchronously during each discard dispatch; no buffer; commit F3. | `jokers.py:1287-1300`, `game.py:825-872` aggregates and commits before queued-effect application. Final value matches; event timing is collapsed. |
| `card.lua:2825-2833` | Mail-In Rebate | No balance read. | Queues `ease_dollars(extra)` per matching discarded card; no buffer; commit F3. | `jokers.py:1274-1284`, `game.py:825-872`: same final amount, collapsed timing. |
| `card.lua:2858-2869` | Faceless Joker | No balance read. | Queues an event whose callback queues `ease_dollars`; nested payout commits late in F3, after already queued first-level discard events.  No buffer. | `jokers.py:1259-1271`, `game.py:825-872`: committed together with other discard money. **NEW-P5-1-01: nested position lost.** |
| `state_events.lua:432-437` | System / Golden Needle discard cost | No balance or `bankrupt_at` legality read. | Queues `ease_dollars(-discard_cost)` only for a voluntary discard; Hook skips cost and discard counters. Commit F3; no buffer. | `game.py:874-885` subtracts directly and is bypassed by Hook's separate path. Cost final value matches; Hook pipeline is **D31**. |
| `card.lua:1033-1064`; `state_events.lua:171-233` | Held Gold card at EOR | No balance read. | Each original/Red-seal/Mime repetition calls queued `ease_dollars(h_dollars)`; no dollar buffer; commit F6. | `game.py:1545-1586` pays once per held card via `EaseDollars`; Red/Mime repeats are missing (**C10**). |
| `card.lua:2271-2275`; `state_events.lua:99-110` | Rental sticker | No balance read. | Each Joker queues `ease_dollars(-rental_rate)` immediately after its own EOR calculation and before its perishable calculation; commit F6. | `round_lifecycle.py:100-126` tallies but does not write money; `economy.py:174-182,235-242` postpones the deduction to cash-out while adjusting interest. **D22.** |
| `card.lua:1655-1679`; `state_events.lua:1175-1182` | Golden Joker, Cloud 9, Rocket, Satellite, Delayed Gratification | These read tallies/counters, not the balance. | Return pending cash-out row amounts after EOR/rental/perishable have run; no balance/buffer write here. | `jokers.py:2286-2324`; calculated at `game.py:1525-1529`, before EOR mutation/perishing. **D18.** |
| `card.lua:2896-2901` | Rocket EOR growth | No balance read. | Directly increments its future/current `extra.dollars` during F6, before `calculate_dollar_bonus`. | `jokers.py:2333-2342` mutates correctly, but bonus was already snapshotted (**D18**). |
| `card.lua:2985-3009`; `card.lua:375-384` | Egg / Gift Card | Reads card `cost`/`extra_value`, not balance. | Directly increases sell-value inputs and calls `set_cost`; Gift Card visits Jokers then consumables during EOR. | `jokers.py:2345-2368`, `card.py:543-568`: Egg matches. Gift Card receives only `ctx.jokers` and therefore omits consumables (**D21**, outside the ticket's requested audit subset but pre-existing). |
| `state_events.lua:1139-1189` | Blind reward, remaining hands/discards, Joker bonuses, eval tags | No balance read. | Sums into local `dollars`; later bottom-row event writes `current_round.dollars` at F7 (`common_events.lua:1064-1088`). | `economy.py:185-220,235-251`, `game.py:1697-1710`: ordinary components match; Investment Tag is applied separately at cash-out (`game.py:982-989`) rather than included in `round_earnings`. |
| `state_events.lua:1191-1203` | Interest | Reads committed dollars only, after F6 money (including held Gold and rental) and before all cash-out rows. | Adds interest only to local cash-out total; no buffer. | `economy.py:181-182,223-233` reconstructs post-rental committed money. It matches ordinary wins, but relies on compensation rather than the actual committed balance. |
| `button_callbacks.lua:2938-2944` | System cash-out | Reads `current_round.dollars`, then later committed dollars. | Queues the total with `ease_dollars`; the following event snapshots `previous_round.dollars`. Commit F8. | `game.py:978-991` directly adds earnings, then eval-tag dollars, then snapshots. Final ordinary balance matches; event/state order is collapsed. |
| `card.lua:593-595,655-657` | Credit Card | No dollar write. | Directly lowers/raises `bankrupt_at` when added/removed. | `card.py:868-870,935-937`: matches. |
| `button_callbacks.lua:2048-2056` | Play legality | **No money read and no `bankrupt_at` check found.** | None. | `_handle_play_hand` (`game.py:539-576`) likewise has no money check: matches. Buying/reroll legality is out of scope. |

## 3. Counters, flags, and reservations

| Lua site | State | Read/write and timing | Python site and verdict |
|---|---|---|---|
| `state_events.lua:523-524` | `G.GAME.hands_played`; `current_round.hands_played` | Direct increment in the queued post-evaluate move event, before F2 scoring events commit. | `game.py:638-641`: after `score_hand`, before chips/money/effects. Matches this observation point. |
| `state_events.lua:296-305` | round counters; hand `played_this_round` | Reset in `new_round`'s immediate event. | `run_init.py:394-403,435-438`: matches values; Python additionally resets `jokers_purchased` here, which Lua resets at cash-out (**NEW-P5-1-02**). |
| `common_events.lua:152-188`; `state_events.lua:475` | `hands_left` | Play queues `-1`; F1 makes it visible to scoring. Needle/Burglar use the same queued helper. | `game.py:590-592`, `game.py:1766-1777`, `effects.py:473-490`: same stable values, timing collapsed. |
| `common_events.lua:111-149`; `state_events.lua:432-437` | `discards_left` | Voluntary discard queues `-1`; `discards_used` is incremented synchronously. Hook changes neither. | `game.py:874-885`; Hook bypass at `game.py:2064-2073`. Voluntary stable value matches; Hook dispatch is **D31**. |
| `state_events.lua:574-578` | hand `played`, `played_this_round`, `visible`; `last_hand_played` | All written synchronously at the start of `evaluate_play`, even for a blocked hand. | `scoring.py:473-480` records counts, `game.py:619-623` sets last hand. `HandLevels.record_play` does not set visible (`hand_levels.py:83-87`): pre-existing **D13**. |
| `state_events.lua:129-138` | `most_played_poker_hand` | Written only after a won/saved Boss. Source compares `v.order` to `_order` but never updates `_order`, so ties follow `pairs` traversal rather than a maintained best order. | `game.py:1654-1670` updates the best order and is deterministic. **NEW-P5-1-03.** Ox also uses the wrong live target (**D29**). |
| `state_events.lua:1044-1061`; `game.lua:3193-3204` | `G.GAME.chips` | Nonblocking ease event commits the round total; the following blocking ease completes before the HAND_PLAYED state decision reads chips. | `game.py:646-648,692-701` commits instantly before decision. Same decision value; intermediate timing collapsed. |
| `blind.lua:519-569`; `state_events.lua:455,654-662` | `blind.triggered` | Cleared before play; debuff-hand, Arm/Ox, Flint, and debuffed scoring cards set it before Matador's applicable pass. | `game.py:560-563`; `blind.py:203-311`; `scoring.py:587-591`. Ox is too late, so Matador misses it (**D29**); other Phase 4 paths match. |
| `blind.lua:78-95,176-177,488-496,572-603` | `blind.prepped` | Set true for a newly set blind except Fish; Heart/Fish set it on press; `drawn_to_hand` always clears it, even if disabled. | `blind.py:50-54,355-411`, `game.py:2079-2088`: create defaults false, `drawn_to_hand` neither gates Heart on `prepped` nor clears it. **D28/C09.** |
| `blind.lua:157-175,535-548` | Eye `hands`; Mouth `only_hand` | Reset at set-blind; `debuff_hand(check=false)` writes the accepted type. | `blind.py:255-269`; fresh Blind holds empty `hands_used` and `only_hand=None` (`blind.py:56-64`). Semantically matches. |
| `state_events.lua:459-482,265-267`; `blind.lua:634-636` | `played_this_ante` | Cleared from every selected play first, then set on played cards; all playing-card markers clear only on Boss defeat. | Set at `game.py:595-603`; no clear in `_advance_ante` (`game.py:1720-1735`). **D30.** |
| `blind.lua:605-622`; `cardarea.lua:32-43,592-607`; `card.lua:4113-4122`; `state_events.lua:307-309` | `card.facing`; `wheel_flipped` | Every actual draw asks `stay_flipped`; emplacement records `wheel_flipped`. Flipping front clears it; new round also clears all markers. | `_draw_hand` never calls `stay_flipped` (`game.py:1460-1481`) (**C09**) and `start_round` never clears `wheel_flipped` (**NEW-P5-1-04**). |
| `card.lua:2529-2540,2561-2576` | `joker_buffer` | Riff-raff reserves `+N` synchronously and zeroes in its creation event. Dagger reserves `-1` synchronously and zeroes in its slice event, allowing a later creator to see room. | Pass-local reservations in `effects.py:54-85`; no `gs.joker_buffer` writes. Stable capacity is modeled, but oracle observation state differs. |
| `card.lua:1040-1060,2254-2265,2545-2559,2604-2616,3106-3120,3743-3799` | `consumeable_buffer` | Blue/Purple seals, Cartomancer, Sixth Sense, 8 Ball, Vagabond, Superposition, and Séance increment synchronously; their creation event zeroes it. | Local `EffectQueue.reserved` (`effects.py:54-85,151-179`), never the mapped game-state field. Stable room mostly matches; timing/state view is absent. |
| `state_events.lua:430,482`; `game.lua:1869-1876` | `round_scores` | Cards played/discarded increment at action time; other counters live elsewhere in source. | Discards at `game.py:894-898`; cards played has no writer in the play path. Pre-existing **D59**. |
| `state_events.lua:124`; `tag.lua:158-173` | `unused_discards`; total hands | Unused discards accumulate only on non-game-over (including saved) rounds; run-wide hands increment after each evaluated play. Tags read these committed totals later. | `game.py:638-641,1625-1631`, `tags.py:214-220`: matches successful/saved paths; ordinary loss skips all EOR maintenance (**D22**). |
| `card.lua:2271-2288`; `state_events.lua:99-110` | rental/perishable tallies | For each Joker in order: EOR callback, queue rental, then decrement/perish immediately. A queued self-removal occurs later. | `game.py:1528-1543`, `round_lifecycle.py:100-126`: all EOR callbacks/effects are completed before a second all-Joker maintenance pass; rental is only tallied. **D18/D22.** |
| `state_events.lua:92-115`; `card.lua:3047-3062` | `game_over`; Mr. Bones `saved` | Reads committed accumulated chips; Bones saves only at `chips/blind >= .25`, queues its destruction, and flips the local `game_over` false before later Jokers. | `scoring.py:850-860`, `jokers.py:2532-2540`, `game.py:692-701`: uses this hand's pre-commit score, lacks threshold, and occurs before `end_round`. **C11.** |
| `state_events.lua:111-170,251-283` | win/round flags and Pillar reset | Win/game-over decision precedes held EOR; won path eventually enters ROUND_EVAL, marks blind state, and clears Pillar markers on Boss. | `_round_won` starts only after Python's earlier decision (`game.py:692-701,1497-1717`); defeat cleanup and Pillar reset are missing (**C11/D30/D32**). |
| `button_callbacks.lua:2928-2934` | cash-out round resources | Before entering SHOP, resets `jokers_purchased`, `discards_left`, and `hands_left`, then clears shop flags. | `game.py:978-1013` clears shop flags but leaves all three resources until `start_round` (`run_init.py:394-403`). **NEW-P5-1-02.** |

## 4. Ordered transition specifications

Verdicts refer to the current Python implementation, not the intended Phase 5
port.

### `G.FUNCS.play_cards_from_highlighted`

1. **Sync:** reject if play already contains a card; stop input, clear
   `blind.triggered`, clear forced selections, and sort highlighted cards by
   physical x (`state_events.lua:450-463`). Python validates phase/count and
   clears triggered, but preserves caller index order (`game.py:539-587`):
   **NEW-P5-1-05**, because card/Joker scoring is order-sensitive.
2. **Queued immediate:** enter HAND_PLAYED (`state_events.lua:465-472`). Python
   has no intermediate phase (`game.py:539-747`): **NEW-P5-1-06**.
3. **Queued:** decrement hands (`state_events.lua:475`; helper
   `common_events.lua:152-188`), then the delay barrier (`state_events.lua:476`).
   Python decrements immediately at `game.py:590-592`.
4. **Sync + one queued draw per card:** update `times_played`,
   `played_this_ante`, and cards-played stats, then queue hand-to-play movement
   (`state_events.lua:478-484`). Python mutates/moves at `game.py:577-603`;
   `round_scores.cards_played` is missing (**D59**).
5. **Dispatch:** `Blind:press_play`; if it returns true, queue feedback and a
   delay (`state_events.lua:488-502`). Python calls `_press_play` at
   `game.py:605-609,2049-2088`; Hook is **D31** and Tooth is collapsed.
6. **Queued immediate parent:** dispatch unlock, then enqueue in order:
   `evaluate_play`, an `after` event that moves play to discard and increments
   both hand counters, and an immediate `STATE_COMPLETE=false` event
   (`state_events.lua:504-537`). Children of `evaluate_play` append behind the
   already queued move/counter and state-complete events. Python calls scoring
   directly, increments counters, commits state/effects, and moves cards
   (`game.py:619-689`); stable counter order matches, event observation does not.
7. **F2 / state controller:** only after the preceding score/chip events does
   `update_hand_played` read committed chips and `hands_left` and choose
   NEW_ROUND or DRAW_TO_HAND (`game.lua:3187-3204`). Python decides at
   `game.py:692-745`.

### `G.FUNCS.evaluate_play`

1. **Sync:** evaluate the poker hand; increment hand `played` and
   `played_this_round`, set `last_hand_played`, usage, and `visible`
   (`state_events.lua:571-578`). Python: `game.py:619-623` and
   `scoring.py:455-480`; visibility missing (**D13**).
2. **Sync:** add Splash/all Stone cards, sort the scoring hand, and queue only
   highlight/UI delays (`state_events.lua:580-612`). Python hand evaluation and
   Splash are `scoring.py:446-458,522-536`; equivalent scoring set.
3. **Dispatch:** `Blind:debuff_hand` (`state_events.lua:614`). This is where Ox
   drains instantly and Arm/Eye/Mouth mutate (`blind.lua:519-569`). Python calls
   at `scoring.py:476-520`, but Ox is postponed to `game.py:663-679` (**D29**).
4. **Allowed branch, sync:** read base hand values, apply first-use level,
   dispatch every Joker `before`, apply level-ups, then reread base values
   (`state_events.lua:615-642`). Python `scoring.py:538-578` orders base before
   the before pass but refreshes after a level-up; equivalent for current
   handlers except DNA's queued/placement residual documented by Phase 4.
5. **Dispatch:** blind `modify_hand` (`state_events.lua:643-647`). Python
   `scoring.py:580-585`: matches.
6. **For each scored card:** update rank/suit history; a debuffed card sets
   `blind.triggered`; otherwise collect Red-seal then Joker repetitions
   (`state_events.lua:648-684`). Python `scoring.py:587-613`; history is missing
   (**D59**).
7. **For each repetition:** dispatch card evaluation, then individual Jokers;
   clear `lucky_trigger`; apply chips, mult, play dollars, Joker dollars,
   extras, xMult, then edition (`state_events.lua:685-778`). Python
   `scoring.py:614-659`; numeric order matches, money is aggregated (**C05**).
8. **For each held card:** dispatch card and individual-Joker effects; only on
   the first pass collect Red-seal and Joker/Mime repetitions; apply dollars,
   held mult, xMult (`state_events.lua:782-872`). Python `scoring.py:661-723`:
   matches scoring-pass repeats/order.
9. **Dispatch over Jokers then consumables:** edition chips/mult, main result,
   every Joker's `other_joker`, then edition xMult (`state_events.lua:873-944`).
   Python only iterates Jokers (`scoring.py:735-775`), so the Observatory
   consumable pass remains missing (pre-existing **C13b** in the plan).
10. **Dispatch:** deck-back final scoring (`state_events.lua:946-948`). Python
    `scoring.py:777-791`: matches.
11. **Sync dispatch + queued removal:** run `destroying_card`, Glass chance,
    notify `remove_playing_cards`, then queue shatter/dissolve
    (`state_events.lua:950-996`). Python `scoring.py:793-837` and
    `game.py:654-661`: same broad phase, with removal committed by caller.
12. **Blocked branch:** zero score and dispatch every Joker `debuffed_hand`
    (`state_events.lua:997-1028`). Python returns immediately at
    `scoring.py:480-511`, so step 16 below is skipped (**D12**).
13. **Queued:** score display, nonblocking ease of `G.GAME.chips`, blocking
    ease of hand chip total, and clearing displayed hand name
    (`state_events.lua:1029-1066`). Python computes at `scoring.py:839-841`
    and commits in `game.py:646-648`.
14. **Dispatch always:** every Joker `after` (`state_events.lua:1068-1075`).
    Python `scoring.py:843-848`; blocked branch omission is **D12**.
15. **Queued immediate:** permanently debuff scoring cards if configured
    (`state_events.lua:1077-1084`). Python has only a comment at
    `scoring.py:862-864`: **C15, missing write.**
16. **Return:** caller's already queued event moves surviving played cards and
    increments counters (`state_events.lua:517-527`), then F2 completes before
    the win check. Python `game.py:638-689`.

### `G.FUNCS.discard_cards_from_highlighted` (voluntary and Hook)

1. **Sync:** stop input, clear forced selections, cap by discard-area room,
   sort left-to-right (`state_events.lua:379-393`). Python validates at
   `game.py:750-787`; Hook bypasses it (**D31**).
2. **Dispatch:** every Joker `pre_discard`, with `hook` propagated
   (`state_events.lua:394-396`). Python voluntary path `game.py:789-820` does
   not carry a hook flag; Hook has no dispatch (**D31**).
3. **Per card dispatch:** seal first, then every Joker `discard`; collect
   removal, then either destroy or queue movement to discard
   (`state_events.lua:397-422`). Python voluntary path `game.py:823-872`
   broadly matches; Hook bypasses all of it (**D31**).
4. **Dispatch:** if any destroyed, notify every Joker `remove_playing_cards`
   (`state_events.lua:424-428`). Python lifecycle notification at
   `game.py:871`; matches voluntary path.
5. **Sync:** increment cards-discarded statistics (`state_events.lua:430-431`).
   Python `game.py:894-898`: matches count.
6. **Voluntary only:** queue discard cost, queue `discards_left - 1`, directly
   increment `discards_used`, set DRAW_TO_HAND, and queue
   `STATE_COMPLETE=false` (`state_events.lua:432-446`). Python performs all
   directly at `game.py:874-920`.
7. **F3 then F4:** queued card/money/counter events settle, then the state
   controller draws and calls `drawn_to_hand` (`game.lua:3208-3244`). Python
   redraw/debuff/handler is inline at `game.py:900-944`.

### `draw_from_deck_to_hand`, `draw_card`, and the draw state

1. `draw_from_deck_to_hand` detects a zero-capacity empty-hand loss outside
   Tarot/Spectral packs (`state_events.lua:355-360`). `_draw_hand` lacks this
   transition (`game.py:1460-1481`); only post-action callers separately test
   empty hand (`game.py:1484-1494`). **NEW-P5-1-07: initial draw can enter an
   empty selecting-hand state.**
2. It computes fill-space, or exactly up to three for Serpent after any prior
   play/discard (`state_events.lua:362-368`), then queues one sorted draw per
   card (`state_events.lua:369-376`). Python duplicates Serpent loops at
   `game.py:706-718,903-920`; shared draw is missing as described by S4/C09.
3. Each `draw_card` is a `before` event: remove from source, ask
   `Blind:stay_flipped`, apply challenge flip, emplace, and sort
   (`common_events.lua:386-423`; `cardarea.lua:592-607`). Python batch-pops and
   sorts once (`game.py:1460-1481`) and never calls `stay_flipped` (**C09**).
4. In DRAW_TO_HAND, `first_hand_drawn` is dispatched immediately after the
   draw events are emitted but before they run; then SELECTING_HAND /
   `Blind:drawn_to_hand` is queued behind the draw/Certificate events
   (`game.lua:3219-3241`; Certificate `card.lua:2462-2481`). Python dispatches
   it after cards are already in hand and debuffed (`game.py:267-297`).
   **NEW-P5-1-08: observation point differs**, though Certificate's current
   result usually converges.

### `new_round`

1. **Queued immediate:** initialize discards/hands and reset round counters,
   per-hand `played_this_round`, and every `wheel_flipped` marker
   (`state_events.lua:290-309`). Python `start_round` at
   `run_init.py:384-438` omits `wheel_flipped` (**NEW-P5-1-04**).
2. **Sync:** count Chaos, recalculate reroll cost, consume round bonuses, and
   mark the blind state/subhash (`state_events.lua:311-331`). Python
   `run_init.py:405-420` and `game.py:215-217`: stable values match.
3. **Dispatch:** `Blind:set_blind`, then every Joker `setting_blind`
   (`state_events.lua:333-337`). Python creates/applies blind then fires Jokers
   at `game.py:235-247`; D14 is fixed, nested ordering remains collapsed.
4. **Queued immediate:** set DRAW_TO_HAND, shuffle with `nr..ante`, clear state
   complete (`state_events.lua:338-349`). Python shuffles then draws inline at
   `game.py:256-307`; stable deck/draw order matches.

### `Blind:set_blind`

1. **Sync:** replace blind identity/reward/debuff/mult, clear subtractions,
   disabled/triggered, set `prepped=true`, last-blind metadata, target chips,
   and reward display (`blind.lua:78-118`). Python construction is
   `blind.py:72-121` and `game.py:202-217`; `prepped` begins false
   (**C09/D28**).
2. **Queued nested UI barrier:** blind reveal and `blind_set=true`
   (`blind.lua:119-150`). No gameplay Python counterpart; safe to omit.
3. **Sync boss state:** Eye/Mouth reset; Fish clears prepped; Water/Needle queue
   resource subtraction; Manacle changes hand size; Acorn flips immediately
   and queues three shuffles (`blind.lua:157-205`). Python
   `game.py:1757-1802` applies resources immediately and performs one shuffle
   (**D32**).
4. **Sync dispatch:** redebuff every playing card and every Joker
   (`blind.lua:207-215`). Python debuffs deck and later hand at
   `game.py:249-272`; Joker-area pass is not performed here.

### `Blind:press_play`

1. Disabled returns with no mutation (`blind.lua:464-465`; Python
   `game.py:2059-2060`).
2. Hook queues random selection and the *same discard function* with
   `hook=true`, sets triggered, then inserts a delay (`blind.lua:466-487`).
   Python directly removes cards (`game.py:2064-2073`): **D31**.
3. Crimson Heart sets triggered/prepped only with a Joker; Fish sets prepped
   (`blind.lua:488-496`). Python `game.py:2079-2088`: matches writes.
4. Tooth queues an outer delayed event that emits per-card juice, dollar, and
   delay events; sets triggered (`blind.lua:497-507`). Python batch-subtracts
   at `game.py:2075-2077`; final F1 value matches.

### `Blind:debuff_hand`

1. Disabled returns; configured hand/size tests clear then set triggered and
   may block (`blind.lua:519-534`). Python `blind.py:228-253`: matches.
2. Eye/Mouth read and, unless previewing, mutate their remembered hand
   (`blind.lua:535-548`). Python `blind.py:255-269`: matches.
3. Arm tests level and mutates level immediately; Ox compares the previously
   snapshotted `most_played_poker_hand` and instantly drains committed dollars
   (`blind.lua:550-569`). Python splits both into scoring/caller code
   (`scoring.py:513-520`, `game.py:663-679`); Ox is **D29**.

### `Blind:drawn_to_hand` and `stay_flipped`

1. Cerulean Bell preserves or chooses a forced card (`blind.lua:572-587`).
   Python returns an index then caller writes the flag (`blind.py:386-395`,
   `game.py:298-302`): state matches; action enforcement is pre-existing D27.
2. Crimson Heart runs only when prepped, clears existing Joker debuffs through
   the setter, then debuffs one eligible Joker (`blind.lua:588-600`). Python
   omits the prepped gate (`blind.py:396-409`): **D28**.
3. `prepped=nil` executes even for a disabled blind (`blind.lua:601-603`).
   Python never clears it (`blind.py:381-411`): **C09/D28**.
4. `stay_flipped` tests Wheel, first-action House, face-card Mark, then prepped
   Fish (`blind.lua:605-622`). Python method has the first three and leaves Fish
   as a no-op (`blind.py:413-448`), and no draw caller invokes it (**C09**).

### `end_round`

1. **Queued after event:** read committed accumulated chips and blind target to
   initialize `game_over` (`state_events.lua:87-98`). Python decides in the play
   handler (`game.py:692-701`): **C11**.
2. **Per Joker, interleaved dispatch:** `end_of_round(game_over)`, consume
   `saved`, queue rental, then mutate perishable tally/debuff
   (`state_events.lua:99-110`; `card.lua:2271-2288,3047-3062`). Python does all
   Joker effects, applies them, then all maintenance
   (`game.py:1522-1543`): **D18/D22**.
3. **Sync:** set win flag, branch to GAME_OVER or won-round flow
   (`state_events.lua:111-124`). Python never calls this path on an ordinary
   loss (**D22**).
4. **Won/saved sync:** accumulate unused discards; on Boss snapshot most-played
   hand; update streak/unlock state; possibly queue `win_game`
   (`state_events.lua:124-170`). Python `game.py:1625-1684`; tie behavior differs
   (**NEW-P5-1-03**).
5. **For each held card and every repeat:** compute the card EOR effect plus
   every individual Joker effect; on first pass collect Red-seal and Joker/Mime
   repetitions; queue held dollars and process extras
   (`state_events.lua:171-233`). Python `game.py:1545-1586` omits all repeats
   (**C10**).
6. **Queued moves/reset:** hand to discard; Boss ante changes; discard to deck;
   then an after event enters ROUND_EVAL, marks blind state, clears Pillar
   markers on Boss, reverts temporary modifiers, resets target cards, and
   clears discard/forced flags (`state_events.lua:234-283`). Python performs
   card return/advance inline (`game.py:1588-1695`), omits Pillar reset
   (**D30**), and delays target-card resets until cash-out
   (`game.py:960-968`: **NEW-P5-1-09**).

### `Blind:defeat`

1. Queue dissolve events; immediately flip all face-down Jokers and restore
   Manacle hand size (`blind.lua:276-344`). Python has no defeat method; it
   manually restores Manacle at `game.py:1686-1689` and omits full cleanup
   (**D32**).
2. The final queued event calls `set_blind(nil, nil, true)`, which re-runs
   debuff clearing (`blind.lua:330-337,78-215,624-652`). Python keeps the old
   blind object and clears only playing-card debuffs at `game.py:1616-1623`
   (**D32**).

### `G.FUNCS.evaluate_round`

1. **Sync:** if committed chips beat the blind, add its reward row; otherwise
   add a saved/$0 row (`state_events.lua:1135-1146`). Python always sets
   `blind_reward=blind.dollars` when `_round_won` is entered
   (`economy.py:185-188`): **NEW-P5-1-10—Mr. Bones-saved losses incorrectly
   receive the blind reward.**
2. **Queued before:** call `Blind:defeat`; queue background event; synchronously
   dispatch back `eval` (`state_events.lua:1148-1163`). Python dispatches Back
   at `game.py:1641-1652` but never defeat (**D32**).
3. **Sync calculations / queued rows:** remaining hands, Green Deck discards,
   each Joker's *post-EOR* dollar bonus, each eval tag, then interest from
   committed dollars (`state_events.lua:1165-1203`). Python
   `economy.py:190-233`; bonus timing is **D18**, tag representation differs.
4. **Queued bottom row:** write `current_round.dollars` and expose cash-out
   (`state_events.lua:1205-1208`; `common_events.lua:1064-1093`). Python stores a
   `RoundEarnings` immediately at `game.py:1697-1710`.

### `G.FUNCS.cash_out`

1. **Sync/queued:** disable the button, shuffle deck, then queue removal of the
   evaluation UI; reset `jokers_purchased`, hands/discards, enter SHOP, clear
   shop flags (`button_callbacks.lua:2912-2937`). Python shuffles then skips the
   resource resets (`game.py:949-978,993-1013`): **NEW-P5-1-02**.
2. Queue `ease_dollars(current_round.dollars)`, then queue a snapshot of
   committed `previous_round.dollars` (`button_callbacks.lua:2938-2944`).
   Python adds earnings and eval-tag money directly, then snapshots
   (`game.py:978-991`).
3. Queue chips-to-zero, update post-Boss tags, reset blind choices, delay
   (`button_callbacks.lua:2948-2955`). Python's blind progression was already
   performed in `_round_won` (`game.py:1672-1684`); shop population follows at
   `game.py:998-1013`.

## 5. New sweep findings / top surprises

These are not C01/C03/C05/C09/C10/C11/D12/D14/D18/D22/D28-D32.

1. **NEW-P5-1-01:** Faceless payout is a nested event, unlike Mail/Trading
   (`card.lua:2858-2869`); Python batches all discard dollars
   (`jackdaw/engine/game.py:825-870`).
2. **NEW-P5-1-02:** Lua resets hands, discards, and `jokers_purchased` before
   entering the shop (`button_callbacks.lua:2921-2934`); Python leaves them
   until the next blind (`game.py:978-1013`; `run_init.py:394-403`).
3. **NEW-P5-1-03:** Lua's Boss most-played tie code never updates `_order`
   (`state_events.lua:129-138`); Python does (`game.py:1654-1670`).
4. **NEW-P5-1-04:** Lua clears every `wheel_flipped` marker at new round
   (`state_events.lua:307-309`); Python `start_round` does not
   (`run_init.py:384-444`).
5. **NEW-P5-1-05:** Lua sorts played cards by physical hand position before
   scoring (`state_events.lua:459-483`); Python explicitly preserves caller
   selection order (`game.py:577-587`).
6. **NEW-P5-1-06:** Lua enters HAND_PLAYED in a queued event before moving the
   selected cards and dispatching `Blind:press_play` (`state_events.lua:465-488`);
   Python remains in its current phase throughout the direct handler and only
   chooses the next phase after scoring (`game.py:539-747`).
7. **NEW-P5-1-07:** Lua turns a zero-capacity empty initial hand into GAME_OVER
   (`state_events.lua:355-360`); Python's initial `_draw_hand` caller enters
   SELECTING_HAND anyway (`game.py:267-307,1460-1494`).
8. **NEW-P5-1-08:** Lua dispatches `first_hand_drawn` after draw events are
   queued but before the cards arrive (`game.lua:3219-3241`); Python dispatches
   after drawing/debuffing (`game.py:267-286`).
9. **NEW-P5-1-09:** Lua resets Idol/Mail/Ancient/Castle targets before entering
    ROUND_EVAL (`state_events.lua:251-280`); Python waits for the cash-out action
    (`game.py:960-968`).
10. **NEW-P5-1-10 (highest gameplay impact):** a Mr. Bones-saved loss gets a
    $0 blind row in Lua (`state_events.lua:1139-1145`), while Python always
    includes `blind.dollars` (`economy.py:185-188`).

No engine behavior was changed by this sweep.

## 6. Ledger specification for Phase 5

The draft in class-design section 5 is directionally correct but must not make
`ease()` itself imply a buffer.  Lua producers explicitly choose whether to
reserve pending money.

Minimal API:

```python
class MoneyLedger:
    def committed(self) -> int: ...
    def with_buffer(self) -> int: ...
    def reserve(self, amount: int) -> None: ...       # dollar_buffer += amount
    def clear_buffer_later(self) -> None: ...          # enqueue dollar_buffer = 0
    def ease(self, amount: int, *, instant: bool = False) -> None: ...
    def flush(self) -> None: ...                       # FIFO commits/clears
```

Required rules:

- `ease(..., instant=True)` mutates committed dollars immediately and never
  changes the buffer.  Ox uses this before scoring (`blind.lua:560-567`).
- Ordinary `ease` only queues a committed delta.  Gold/Lucky/Golden Ticket/
  Business/Rough Gem/Reserved Parking explicitly call `reserve` and
  `clear_buffer_later`; Mail, Trading, Faceless, Tooth, discard cost, rentals,
  and held Gold do not.
- Bull and Bootstraps use `with_buffer`; Vagabond, Ox, interest, Economy Tag,
  and legality reads use `committed` (`card.lua:3744,3936-3939,4046-4049`;
  `state_events.lua:1191-1203`; `tag.lua:176-185`).
- Preserve FIFO within a flush.  For `get_p_dollars` producers, buffer-clear
  events are emitted before the scorer emits the dollar commit
  (`card.lua:1084-1087`; `state_events.lua:719-724`).  To Do List/Matador emit
  commit before clear (`card.lua:2738-2740,3492-3494,3721-3723`).  A synchronous
  implementation may coalesce this only after proving there is no intervening
  money observer.
- Flush the ledger at F1, F2, F3, F6, and F8.  F4/F5/F7 contain no direct
  committed-dollar write except events inherited from an earlier nested
  callback; those inherited events must retain FIFO position.
- Cash-out earnings are a separate `current_round.dollars` accumulator.  They
  are not `dollar_buffer` and do not affect interest until F8
  (`state_events.lua:1135-1208`; `button_callbacks.lua:2938-2944`).

No second committed-vs-pending gameplay scalar was found.  Slot buffers are
emission-time reservations, already conceptually represented by
`EffectQueue.reserved`, but the transition oracle must expose their increments
and zeroes at the cited Lua points.
