# Balatro engine correctness audit

Audit date: 2026-09-05.

## Verdict and scope

**The engine does not currently match the supplied Balatro source end to end.** Its extracted gameplay definitions and tested seeded RNG primitives match, and many individual effect handlers are faithful. However, the integrated run loop omits effects, supplies incomplete state to handlers, applies effects in the wrong order, and permits or excludes different actions. Passing the existing engine suite is not evidence of whole-game equivalence.

This is a correctness comparison, not a style, architecture, performance, or changes-since-commit review. It compares the current working tree of `jackdaw/engine` with the supplied `balatro_source/Balatro`, whose `version.jkr` identifies **1.0.1o-FULL / PROD_PC_Console**. Repository HEAD was `d45917983c50bde72245c9f9bd1bac573b79842e`. The pre-existing working-tree modification to `jackdaw/engine/state.py` was included and left untouched. No engine fixes were made.

The sections below distinguish:

- **Critical/direct deviations:** different scores, resources, card availability, legal decisions, run outcomes, seeded behavior, or externally observable engine results. Critical findings are listed first; other direct differences follow by subsystem.
- **Functionally equivalent representation differences:** different implementation or storage with the same specified gameplay effect, subject to the explicit qualifications given.

All deviations identified in this audit are recorded below, including incomplete systems and lower-priority API differences. This is **not a proof that no other deviations exist**: the complete LÖVE game was not executed through every possible run or combination. Static findings follow both the Lua effect and the Python caller; executable examples are marked **Reproduced**. A helper that implements an effect but is never called does not count as a matching system.

Paths in finding evidence use `E/` for [`jackdaw/engine`](../jackdaw/engine/) and `L/` for [`balatro_source/Balatro`](../balatro_source/Balatro/). Line numbers refer to the audited working tree, not a future fixed version.

## Verification performed

| Check | Result and limitation |
| --- | --- |
| Existing engine suite | `.venv\Scripts\python.exe -m pytest tests/engine -q --disable-warnings`: **1,092 passed, 14 skipped, 1 warning**. The skipped external-Lua checks do not establish parity. |
| Extracted definitions versus evaluated Lua tables | No differences in compared fields across **299 centers, 30 blinds, 24 tags, 8 stakes, 52 card fronts, 4 seals**. Lua empty tables and JSON empty arrays normalized; metadata `key`, `pos`, `boss_colour`, `vars`, `discovered`, `unlocked`, `alerted`, `start_alerted` excluded. This result concerns definitions, not runtime effects or profile gating. |
| Challenge definitions | All **20** Python challenge definitions matched evaluated `L/challenges.lua`, with empty-container normalization. Their application is incomplete: see C15 and D44. |
| Actual LuaJIT RNG comparison | Loaded the supplied `misc_functions.lua` using `lupa.luajit21`; **5,000 draws**, five seeds and four interleaved named streams, **zero mismatches** against `PseudoRandom.random`. This does not validate every game's call site or call count. |
| Actual Lua hand-component comparison | **10,000** reproducibly generated hands of 1–5 cards, including Stone/Wild/debuffed cards and Four Fingers, Shortcut, Smeared combinations. **261 component differences, all `High Card` card selection**; no other returned component differences in this sample. Nominal values were deliberately shared between implementations to isolate evaluator behavior; this is not an independent verification of `Card.get_nominal`. See D01. |
| Actual Lua blind-amount comparison | Scaling 1–3, antes 0–39: exact through ante 16; integer/double differences from ante 17; Python raises `OverflowError` at ante 39 where the supplied Lua function produces NaN. See D49. |
| Integrated Python examples | Controlled states passed through `initialize_run`, `step`, card creation, and scoring; representative results appear below. Oracle expectations come from the supplied source, not descriptions or wiki text. |

The local, read-only diagnostic harness is `.scratch/engine_audit_probe.py` (ignored scratch artifact, not a committed test). It was run with `.venv\Scripts\python.exe .scratch/engine_audit_probe.py`. These checks do not require a network service or a running Balatro instance. The essential reproduction pattern is also included at the end of this report.

## Critical / direct deviations

### Critical: core run and scoring behavior

### C01 — Hand counters advance at the wrong time and successful hands are counted twice

Lua increments the run/round hand counter **after** `evaluate_play`, while updating the selected poker-hand history once inside that evaluation, before the blind-debuff branch. Python increments run/round counters before scoring, then calls `HandLevels.record_play` in both `score_hand` and `_handle_play_hand` for successful hands. Blocked hands instead receive their history update only after scoring returns.

**Reproduced:** one High Card play records `played == 2`. DNA and Sixth Sense see the first round hand as hand 1 instead of 0. Supernova, Obelisk, Card Sharp/history consumers and most-played-hand logic receive different histories. This is not a zero-based/one-based representation change because the handler comparisons were not translated with it.

Evidence: `E/game.py:579, 699`; `E/scoring.py:565`; `E/hand_levels.py:83`; `L/functions/state_events.lua:523–524, 574–578`; `L/card.lua:2604, 3501`.

### C02 — Several card-creation callbacks never produce their effect

The following are independent integration omissions, in addition to C01:

| Effect | Source behavior | Engine deviation |
| --- | --- | --- |
| DNA | First single-card hand creates a copy in hand. | The `before` pass discards `extra.create`; the `playing_card_copy` descriptor is also unsupported by the general resolver. |
| Sixth Sense | Destroys the first-hand single 6 and creates a Spectral if there is room. | The destruction pass does not preserve the create result. Correcting only the counter still would not create the Spectral. |
| Certificate | `first_hand_drawn` adds a random front with a randomly selected seal, drawn into hand. | No production dispatch of `first_hand_drawn` was found. The alternate setting-blind creation branch has no front and hardcodes a Gold seal. |
| 8 Ball | Each successful eligible scoring trigger creates a Tarot, respecting capacity/buffering. | Individual-Joker application ignores its `extra.create` result. |
| Hallucination | Opening an eligible booster can create a Tarot. | Shop booster opening discards returned mutations; tag pack opening does not dispatch the opening callback. Its probability RNG key also differs, D48. |

**Reproduced:** first-hand single 6 with DNA or Sixth Sense creates nothing; Sixth Sense leaves the 6 alive. Certificate leaves the standard 52-card deck at 52.

Evidence: `E/scoring.py:352, 588, 811`; `E/card_factory.py:456`; `E/game.py:337, 1218, 1817, 2354`; `E/jokers.py:1979–2167`; `L/game.lua` `Game:update_draw_to_hand`; `L/card.lua:2336, 2462, 2604, 3106, 3501`.

### C03 — Vampire strips enhancements too late and applies its multiplier twice

Lua strips/upgrades Vampire in the `before` pass, before the enhanced cards score, then applies the accumulated XMult in the main Joker pass. Python introduces an `individual_hand_end` pass after scored and held cards, applies Vampire XMult there, and applies it again in the main pass. It also leaves its `vampired` marker set rather than clearing the temporary guard, so a subsequently re-enhanced card can be skipped on later plays.

**Reproduced:** a fresh Vampire and a single Mult-enhanced 2 produce 7 chips × 6.05 Mult = **42** in Python. The source strips the +4 enhancement first and applies X1.1 once, yielding floor(7 × 1.1) = **7**.

Evidence: `E/scoring.py:733, 749`; `E/jokers.py:1768`; `L/card.lua:3465` and Vampire's main-scoring branch.

### C04 — Oops! All 6s does not change most gameplay probabilities

Adding/removing Oops changes flat `probabilities_normal`. Scoring reads `gs['probabilities']['normal']`; end-of-round snapshots omit probability state and retain their default of 1. Wheel of Fortune reads the flat field, so different effects disagree about the probability in the same run. Lua updates every entry in `G.GAME.probabilities` and all these consumers use that shared state.

**Reproduced:** adding one Oops gives flat probability 2 but nested normal probability 1. Lucky/Glass and probabilistic scoring Jokers consequently use the unmodified chance; end-of-round extinction probabilities do too.

Evidence: `E/card.py:772, 810`; `E/game.py:641, 1580`; `E/scoring.py:456`; `E/consumables.py:535`; `L/card.lua:612, 673, 988, 1076, 3020`.

### C05 — Scoring snapshots freeze money that Lua changes during scoring

Python snapshots money before scoring and accumulates dollar effects for the caller to apply afterward. Lua's scoring money effects update/buffer money as effects execute. Later money-dependent effects therefore see different balances: Bull, Bootstraps, Vagabond, and interactions with Gold seals, Lucky money, To Do List, and dollar-paying Jokers are affected. Aggregating the final dollars is equivalent only if no intervening effect reads money.

**Reproduced:** at $4, playing a Gold-sealed 2 with Bull gives **15 chips** in Python (7 + 2×4), with dollars becoming $7 afterward. Lua's Bull sees the $3 earned and gives **21 chips** (7 + 2×7).

Evidence: `E/game.py:614, 655`; `E/scoring.py:398, 749`; `E/jokers.py:1058, 1166, 2089`; `L/functions/state_events.lua:706–753`; `L/card.lua` Bull/Bootstraps/Vagabond and dollar-buffer branches.

Note from user: Vagabond, I believe, would behave correctly using a frozen snapshot of money. I believe it pays out depending on pre-play state. It's a little weird than that, even: having the ox set money to 0 makes it pay out too, evne if you had money beforehand. There's a point before hand is scored and after The Ox activates where Vagabond remembers its flag, even if gold seals, matador, etc. trigger. 

### C06 — Pool exclusion tracks historical creation instead of currently existing cards

Python registers every generated center in `used_jokers` and deliberately never unregisters a sold Joker. Rerolling/leaving shops and consuming/removing cards likewise lack Lua's general removal bookkeeping. Lua `Card:remove` clears the center when no other applicable card of that name remains. Thus Python permanently excludes previously seen shop cards and spent consumables, even when the player never bought them. This changes future shops, packs, generated cards, fallback frequency, and seeded sequences.

Pack generation separately clears temporary exclusions before the open pack is actually removed, allowing effects such as Emperor to generate identities still displayed in the pack. Some directly constructed/copied consumables never register at all. These are inconsistent lifetime rules, not merely a differently named set.

**Reproduced:** selling a Joker leaves its center excluded in `used_jokers`.

Evidence: `E/card_factory.py:389`; `E/game.py:121, 142, 2323, 2427`; `E/packs.py:28`; `L/card.lua:349–354, 4727–4755`; `L/functions/common_events.lua:1987`.

### C07 — Higher-stake Eternal, Perishable, and Rental generation flags are disconnected

Stake initialization stores flags in `gs['modifiers']`; the card factory reads flat top-level names. Standard higher-stake runs therefore do not apply these stickers to shop/pack Jokers. Correct constants in `stakes.json` do not repair this integration.

**Reproduced:** a Gold Stake run has all three flags true under `modifiers`, with all corresponding flat fields absent.

Evidence: `E/stakes.py` `apply_stake_modifiers`; `E/card_factory.py:399`; `L/functions/common_events.lua:2133–2154`.

### C08 — Disabling a blind is not the source's disable operation

Chicot's setting-blind result only assigns `blind.disabled = True`; it does not call the existing `Blind.disable` method or process its restoration effects. Wall/Violet Vessel targets retain their increased size. Chicot's source-side immediate effect when acquired during an active boss is also absent from `Card.add_to_deck`. Luchador has an additional unreachable selling callback, D23.

**Reproduced:** ante-1 Wall with Chicot is marked disabled but still requires **1,200** chips; Lua disables Wall's extra ×2 and requires **600**.
NOTE: ante-1 wall should be impossible; make sure to check the engine doesn't allow it and correctly batches by ante, as the real game does. 

Evidence: `E/game.py:1817`; `E/blind.py:432`; `E/card.py:745`; `L/blind.lua:356–415`; `L/card.lua:594, 2355`.

### C09 — Face-down boss behavior is missing or applied to the wrong cards

`Blind.stay_flipped` exists, but the real draw path never calls it. House's first draw, Wheel's probabilistic draws, and Mark's face-card draws remain face-up. Fish's play-redraw branch flips **all** held cards rather than just cards being drawn; the Python drawn-to-hand handler also does not clear `prepped` as Lua does. This changes available information and card-facing state, not just animation.

**Reproduced:** House and Mark deal face-up cards, including the initial face cards. Wheel's missing call is established statically; one all-face-up Wheel draw alone would not prove a probability defect.

Evidence: `E/game.py:738, 1515`; `E/blind.py:350, 398`; `L/blind.lua:572–623`; `L/functions/common_events.lua:386`.

### C10 — End-of-round held-card processing omits retriggers and chooses the wrong Planet

Python pays each held Gold enhancement once and creates at most one Planet per held Blue seal. Lua evaluates held end-of-round effects with Red-seal and Mime repetition. Blue seals create the Planet for the **last played hand**, not the most-played hand. Python also constructs these Planets as raw `Card` objects with minimal abilities and zero cost instead of using normal card initialization.

**Reproduced:** a held Gold card with Red seal and one Mime pays **$3**, versus **$9** in source. Winning with High Card after making Pair the most played creates **Mercury**, cost 0, rather than a properly initialized **Pluto**.

Evidence: `E/game.py:1620–1668`; `L/card.lua:1033` `get_end_of_round_effect`; `L/functions/state_events.lua` `end_round` held-card repetition loop.

### C11 — Mr. Bones can save below its threshold and be consumed on a winning round

Its Python handler saves unconditionally on `game_over`. The scoring caller decides to invoke it by comparing **this hand's** score against the target, not accumulated round chips, and never tests the 25% threshold. Thus it can save at almost zero chips or disappear even when the cumulative score actually wins. The empty-hand/deck loss path and blocked-hand early return do not run the proper end-of-round save check.

**Reproduced:** 7 accumulated chips against a 300-chip blind with no hands left consumes Bones and enters `ROUND_EVAL`, despite being below 75 chips.

Evidence: `E/jokers.py:2542`; `E/scoring.py:865`; `E/game.py:704, 1537`; `L/card.lua:3047`; `L/functions/state_events.lua` `end_round` game-over calculation.

### C12 — Consumable-applied editions lack scoring values and slot/cost maintenance

Aura, Wheel of Fortune, Ectoplasm, and Hex assign raw edition flags instead of using the edition lifecycle. `get_edition` reads populated scoring fields, so a resulting Foil/Holographic/Polychrome flag need not grant its chips/Mult/XMult. Applying Negative to an owned Joker does not add its slot; later removing that Joker still subtracts a slot. Costs/sell values are not refreshed. Perkeo's separately implemented copy path also omits negative-consumable slot maintenance, D24.

**Reproduced:** Hex leaves a Polychrome flag but `get_edition()` returns no XMult contribution. Applying Ectoplasm to two owned Jokers leaves `joker_slots == 5`, not 7.

Evidence: `E/game.py:2160`; `E/card.py:376, 780, 824`; `E/consumables.py:524, 935–989`; `L/card.lua:387–525, 1434–1510`.

### C13 — Several major vouchers and Astronomer have no effective consumer wiring

| System | Deviation |
| --- | --- |
| Telescope | Pack generation reads `has_telescope` and a supplied most-played value; voucher redemption/normal run state never supplies the required flag. |
| Omen Globe | Voucher application writes `omen_globe`; Arcana generation reads `has_omen_globe`. Spectral replacement never activates normally. |
| Observatory | No scoring pass evaluates held consumables for the matching-Planet X1.5 effect. Recording the redeemed voucher is insufficient. |
| Astronomer | Pricing checks `has_astronomer`, but acquiring/removing the Joker does not maintain that flag or reprice cards. |
| Magic Trick / Illusion | Shop playing-card generation creates cards without a base/front. The existing `roll_illusion_modifiers` helper is not called by the live shop path. |

**Reproduced:** redeemed Telescope/Observatory have no relevant active flag, Omen has only the wrong flag, and `create_card('Base', ...)` returns `base is None`.

Evidence: `E/packs.py:138–178`; `E/vouchers.py:225–251`; `E/card_factory.py:385, 433`; `E/shop.py:152, 333`; `E/scoring.py:749`; `L/card.lua:1717–1755`; `L/functions/state_events.lua:878–938`; `L/functions/UI_definitions.lua:742` `create_card_for_shop`.

### C14 — Magic, Ghost, Green, and Anaglyph Decks do not receive all their effects

Magic/Ghost starting consumable keys are stored but never instantiated by `initialize_run` or its standard runner. Green's bonus/interest settings are written at the top level while earnings reads `modifiers`, producing ordinary cash-out rules. Anaglyph's `Back.trigger_effect('eval')` implementation is never dispatched on boss defeat.

**Reproduced:** Magic and Ghost start with empty consumable areas. A Green Deck win with $10, three unused hands and three discards pays ordinary hands $3/discards $0/interest $2, instead of hands $6/discards $3/interest $0. The ordinary $3 blind reward is separate in both cases.

Evidence: `E/run_init.py:245–268`; `E/back.py:108–130, 181`; `E/economy.py:190–243`; `E/game.py:1550`; `E/runner.py:84`; `L/back.lua` `apply_to_run` and `trigger_effect`.

### C15 — Challenge configurations match, but substantial challenge rules are not implemented

Starting challenge Jokers remain descriptors in `challenge_jokers`, and starting consumables remain keys; neither is materialized by the standard run path. Custom rules are frequently stored without a runtime consumer. In particular:

- `chips_dollar_cap`: scoring does not clamp chip changes to the money limit through Lua's `mod_chips`.
- `minus_hand_size_per_X_dollar`: no dynamic money-based hand-size adjustment.
- `flipped_cards`: no corresponding draw-time probability path.
- `all_eternal`: the factory never applies the challenge's universal Eternal rule.
- `set_eternal_ante`, `set_joker_slots_ante`: boss/ante transition omits these rule changes.
- `debuff_played_cards`: scoring ends with a comment describing the modifier but does not perform it.
- `inflation`: live purchases do not increment inflation and reprice all affected cards.

Challenge bans are additionally lost in several generators, D43. This does **not** mean every challenge modifier is broken: ordinary starting-parameter overrides and reward/interest switches read from `modifiers` do have implementations.

Evidence: `E/challenges.py:571–618`; `E/run_init.py:270–362`; `E/scoring.py:881`; `E/game.py:1061, 1191, 1218, 1760`; `L/game.lua:2063–2148`; `L/functions/misc_functions.lua:684`; `L/functions/common_events.lua:403, 2133`; `L/functions/state_events.lua:240–248, 1080`; `L/card.lua:1800, 1858`.

### Other direct deviations: evaluation and scoring

### D01 — Nominal sorting and High Card component selection differ

`hand_eval._card_nominal` and `card_area._card_nominal` use a flat Stone penalty and an increasing `sort_id` contribution, unlike source `Card:get_nominal`, which scales suit nominal for Stone cards and uses `unique_val`. They also disagree with Python's own `Card.get_nominal`, whose synthetic micro-tiebreaker decreases with sort ID. The differential sample found 261 different High Card component selections, including all-Stone hands and duplicated rank/suit cards.

Do not interpret that as 261 different final poker-hand classifications: duplicate ranks often make Pair or better the selected hand, and all Stone cards are added to scoring anyway. The confirmed differences are the primitive result and sorting/tie behavior. Card ordering can matter to first-card effects and user selection; replacing source's unseeded unique tie with a deterministic synthetic tie is not guaranteed physically identical ordering.

Evidence: `E/hand_eval.py` `_card_nominal`, `get_highest`; `E/card_area.py:154, 172`; `E/card.py:584`; `L/card.lua:950`; `L/functions/misc_functions.lua` `get_highest`.

### D02 — Stone face-card classification is wrong

Python `Card.is_face` checks `base.id` rather than `get_id()`. A Stone card with an underlying J/Q/K therefore counts as a face without Pareidolia, whereas Lua's negative Stone ID does not. This affects face Jokers, Plant/Mark, and destruction/discard face tests. A baseless card also returns false before Pareidolia is considered; proper source playing cards retain a base.

Evidence: `E/card.py:486`; `L/card.lua:964`.

### D03 — Pareidolia and Smeared state is omitted from some effect contexts

Ride the Bus and Caino use face checks without the active Pareidolia flag. The discard snapshot/context omits the global face/suit modifiers needed by Faceless Joker and Castle. Flower Pot/Seeing Double aggregate suit checks omit Smeared. Suit-debuffing blinds likewise call suit checks without Smeared. Lua's card methods query active Jokers directly, so their behavior follows the global effects in all these contexts.

Evidence: `E/jokers.py:1111, 1136, 1306, 1519, 1748, 2476`; `E/game.py:766, 980`; `E/blind.py` `debuff_card`; `L/card.lua:964, 4064` and the corresponding Joker branches.

### D04 — Lucky Cat reads a different trigger field from the one Lucky cards set

Lucky card methods write `ability['lucky_trigger']`; Lucky Cat reads the card's top-level `lucky_trigger`. Lua both writes and reads the top-level field and clears it for each individual scoring iteration. Python does not perform that matching clear. Lucky Cat consequently misses normal Lucky-card growth; manually supplying the top-level flag risks stale repeated growth.

Evidence: `E/card.py:614–638, 701–727`; `E/jokers.py:1656`; `L/card.lua:988, 1076` and Lucky Cat's individual branch; `L/functions/state_events.lua:700`.

### D05 — Brainstorm in the leftmost position copies the next Joker instead of itself/no effect

Python `_find_leftmost` excludes the copying card and returns the next card. Lua targets `G.jokers.cards[1]`; when that is Brainstorm itself, it does not recurse/copy another Joker. Positioning Brainstorm first should not act as a second-position copier.

Evidence: `E/jokers.py:1424, 1470`; `L/card.lua:2321`.

### D06 — Blueprint/Brainstorm incorrectly suppress some copyable effects

Hiker, DNA, Burnt Joker, and Perkeo contain `not ctx.blueprint` guards that the corresponding source branches do not have. The copy dispatcher already checks prototype compatibility; these extra guards prevent legitimate additional upgrades/copies/level-ups. Sixth Sense is not included here: its source has a non-Blueprint guard.

Evidence: `E/jokers.py:2009, 2290, 2553, 2584`; `L/card.lua:2413, 2749, 3067, 3501`.

### D07 — Loyalty Card uses round-local time and wrong creation time

The snapshot handed to Loyalty is round-local hands played, and generic card creation supplies the default creation hand count of zero. Source Loyalty tracks against the run-wide `G.GAME.hands_played` at creation. Charge progress can reset every round and newly acquired copies can start at the wrong point. C01 also shifts when the comparison is evaluated.

Evidence: `E/jokers.py:1186`; `E/scoring.py:398` snapshot construction; `E/card_factory.py:385`; `E/card.py:293`; `L/card.lua:330, 3632`.

### D08 — Raised Fist selects the first tied lowest card, source selects the last

Python replaces the lowest held card only on `<`; Lua uses `>=` when scanning and therefore replaces on ties. With a Red seal or debuffed card on only one tied lowest card, this changes which individual held-card iteration receives the Mult and how often it repeats.

Evidence: `E/jokers.py:1208`; `L/card.lua:3324`.

### D09 — Blackboard fails with an empty held hand

Python requires a truthy held-card list. Source tests whether black-card count equals total held-card count; zero equals zero, so an empty held hand satisfies Blackboard. Playing the entire hand loses its X3 in Python.

Evidence: `E/jokers.py:1076`; `L/card.lua` Blackboard main-scoring branch.

### D10 — Joker-to-Joker effects incorrectly omit debuffed target Jokers

The main loop skips a debuffed Joker completely before evaluating other Jokers against it. Source still runs the other-Joker context for that target. Baseball Card can therefore fail to give X1.5 for a debuffed Uncommon Joker. Independently, Stencil counts only non-debuffed Stencils and Swashbuckler excludes debuffed neighbors' sell values, while source counts those occupying cards/values too.

Evidence: `E/scoring.py:749–785`; `E/jokers.py:1092, 1946, 1959`; `L/functions/state_events.lua:878–938`; `L/card.lua` Baseball Card branch and `update` Stencil/Swashbuckler tally branches.

### D11 — Matador only pays on completely blocked hands

Python only implements the `debuffed_hand` branch. Lua also pays in the normal main-Joker pass when `blind.triggered` is true, including relevant scoring-card debuffs and triggering Arm/Flint cases. Python neither preserves all those trigger assignments nor handles Matador's corresponding main branch.

Evidence: `E/jokers.py:1401`; `E/scoring.py:497, 749`; `L/card.lua:3719`; `L/functions/state_events.lua:651–667`.

### D12 — Blocked hands skip after-hand effects

`score_hand` returns immediately after the debuffed-hand callbacks. Lua's `after` Joker pass occurs outside the success/debuff conditional. Ice Cream and Seltzer, for example, should still decay after a blocked play. The early return also bypasses the proper Bones check, already covered by C11.

Evidence: `E/scoring.py:497–520, 854`; `L/functions/state_events.lua:998, 1065`.

### D13 — Secret-hand visibility is updated on level-up instead of play

Python makes a hand visible when it reaches level >1, but `record_play` never makes it visible. Lua marks the played hand visible during evaluation, and `level_up_hand` does not reveal every secret hand it levels. A first Five of a Kind/Flush House/Flush Five can remain invisible, while Black Hole reveals unplayed secret hands. This affects visible-hand selection such as To Do List, not only the display.

Evidence: `E/hand_levels.py:63–88`; `E/jokers.py:1355`; `L/functions/common_events.lua:464`; `L/functions/state_events.lua:578`.

### Other direct deviations: Joker and round lifecycle

### D14 — Burglar's round setup is overwritten

The setting-blind mutation adds hands and sets discards to zero before `start_round` replaces those counters from round-reset values. Source executes Burglar after initializing round resources.

**Reproduced:** Red Deck with Burglar starts at four hands/four discards, not seven hands/zero discards.

Evidence: `E/game.py:158–286, 1847`; `E/run_init.py:366`; `E/jokers.py:2254`; `L/card.lua` Burglar setting-blind branch; `L/functions/state_events.lua:290–353`.

### D15 — Ceremonial Dagger never removes its victim; Madness can destroy itself or an Eternal

Dagger returns `destroy_joker`, but setting-blind mutation application has no consumer for it. The victim remains, and can keep contributing/being valued on later rounds. Source marks it for slicing and destroys it, preventing its own later setting-blind effect. Madness's resolver excludes `jokers[0]` rather than the actual Madness source, and does not apply source's Eternal/getting-sliced eligibility filters. Ordering Madness anywhere but first exposes self-destruction and the wrong protected target.

Evidence: `E/jokers.py:1828, 1901`; `E/game.py:1817–1897`; `L/card.lua:2490–2525, 2561`.

### D16 — Marble creates a Stone card without an underlying front

The setting-blind branch constructs `Card()` then applies Stone, omitting source's `marb_fr` draw and playing-card front. A Stone still scores its 50 bonus, but its underlying rank/suit, future conversion by Vampire/Strength/Death, identity and RNG stream do not match. The missing card-added notification is separately covered by D17.

Evidence: `E/game.py:1854–1888`; `L/card.lua:2580–2601`.

### D17 — Card-created/card-destroyed notifications do not cover all mutation paths

Hologram is notified for shop/Standard-pack additions but not all Marble, DNA, Cryptid, or Spectral-created cards. Caino and Glass Joker get scoring-destruction notifications but not consumable destruction or Trading Card's discard destruction. Lua explicitly invokes `playing_card_joker_effects` and removal contexts in these paths. Correct local scoring growth does not imply correct growth over an entire run.

Evidence: `E/game.py:1061, 1288, 1854, 2031, 2180`; `E/scoring.py:833`; `E/game.py:766` discard removal; `L/card.lua:1206–1213, 1336–1338, 1570–1586, 2580`; `L/functions/state_events.lua` discard removal notification.

### D18 — End-of-round dollar bonuses are calculated before mutations/perishing

Python calculates all `calc_dollar_bonus` values before end-of-round Joker effects and perishable maintenance. Source executes end-of-round effects/rental/perishable first and calculates the cash-out bonuses later. Rocket should increase its payout on the boss just defeated, not only next round. A Golden Joker becoming debuffed by perishing should not still supply the precomputed bonus.

**Reproduced:** a fresh Rocket pays $1 on a defeated boss while its stored payout becomes $3; source pays $3 at that cash-out.

Evidence: `E/jokers.py:352–409`; `E/game.py:1580–1619`; `L/functions/state_events.lua:99–109, 1175`; `L/card.lua` Rocket end-of-round branch and `calc_dollar_bonus`.

### D19 — Cloud 9, Satellite, and Delayed Gratification read missing state

Cloud 9 reads `ability.nine_tally`, but no engine updater populates it. Satellite reads `ability.planet_types_used`, while tracked usage is stored elsewhere and never writes that field. Delayed Gratification's end-of-round snapshot omits `discards_used`, leaving zero even after discarding. Source recomputes/counts the appropriate actual deck, Planet identities, and round discard counter.

**Reproduced:** Cloud 9 + Delayed Gratification + fresh Rocket with four deck 9s, a used discard, and two discards left pays $5: Cloud $0, incorrect Delayed $4, Rocket $1. Source would pay Cloud $4, Delayed $0, Rocket $1. The equal total in this particular combination is coincidental; each independent contributor is wrong.

Evidence: `E/jokers.py:2317–2348`; `E/game.py:1580`; `E/consumables.py:616`; `L/card.lua:1661–1683, 4191`.

### D20 — Turtle Bean does not reduce the actual hand size as it decays

Python reduces `extra.h_size` but leaves the live `gs['hand_size']` unchanged. Removing the card later reverses only the remaining smaller bonus, leaving permanent extra hand capacity. Source changes the hand-area limit at each decay.

Evidence: `E/jokers.py:2566`; `E/card.py:769, 807`; `L/card.lua:2903`.

### D21 — Gift Card excludes owned consumables

Python increases sell value only for the Joker list. Lua applies the increase to both owned Jokers and consumables. This changes Temperance's surrounding economy and the sale value of saved consumables.

Evidence: `E/jokers.py:2382`; `L/card.lua` Gift Card end-of-round branch.

### D22 — Rental and losing-round lifecycle timing differs

Python removes self-destructing Jokers before rental tally, so a rental Joker that expires/destroys itself avoids its final rent. Lua's per-Joker end-of-round loop calculates rental before delayed removal completes. Python also postpones the cash deduction to `CashOut`; consumables are allowed in its `ROUND_EVAL`, making the larger pre-rent balance observable to Hermit/Temperance-like decisions. The interest calculator's effective-money subtraction repairs some totals but not that intermediate state.

On ordinary loss Python goes directly to `GAME_OVER`, omitting the source end-of-round context/maintenance sequence. Persistent statistics, final money, card mutation and save effects need not match even when both runs lose.

Evidence: `E/game.py:704–718, 998, 1136, 1598`; `E/round_lifecycle.py:65`; `E/economy.py:117`; `L/functions/state_events.lua:87–120`; `L/card.lua:2271–2289`.

### D23 — Selling lacks selling-self effects and is restricted to the shop

`_handle_sell_card` documents but never dispatches `selling_self`. Luchador, Diet Cola, and a charged Invisible Joker therefore have no sale effect. It also never applies Verdant Leaf's disable-on-Joker-sale behavior. Separately, selling is permitted only in `SHOP`, whereas source permits sales at other stable decision phases, including an active hand and booster selection. This prevents important source strategies even if the callbacks are fixed.

Evidence: `E/game.py:1107`; `E/actions.py:325–504`; `L/card.lua:1590–1653, 2354`; `L/blind.lua` Verdant Leaf behavior.

### D24 — Perkeo copies alias state and do not expand consumable capacity

The shop mutation resolver uses `copy.copy`, so the copy shares mutable ability data with its original and retains identity/sort fields. It assigns a Negative flag without normal edition, cost, ownership, and consumable-slot handling. Source creates a distinct card using `copy_card`, sets Negative and calls `add_to_deck`. The copy must supply its extra consumable slot; simply allowing an overfull list leaves subsequent capacity checks wrong. Blueprint suppression is D06.

Evidence: `E/game.py:2394–2424`; `E/jokers.py:2584`; `L/card.lua:2413`; `L/functions/common_events.lua:2158` `copy_card`.

### D25 — Debuffing does not remove/reapply Joker passives or preserve permanent perishing

`Card.set_debuff` only changes a Boolean; round maintenance sometimes assigns it directly. Lua removes/re-adds applicable passive effects on debuff changes, and refuses to clear an expired Perishable. Hand size, discard modifiers, probability, credit and interest effects therefore remain active while their source Joker is debuffed. Crimson Heart's clear-all pass can revive expired Python Perishables; the source setter prevents this.

Evidence: `E/card.py:418, 745`; `E/round_lifecycle.py:119`; `E/blind.py:379`; `L/card.lua:526–538, 564–704`.

### D26 — Some passive acquisition/removal changes do not reach current-round state

Chaos the Clown writes a flat `free_rerolls`; shop reroll cost uses `current_round.free_rerolls`. Acquiring/selling it in the current shop does not immediately update the proper free-reroll balance or cost. The next round's reset partially hides this by recounting Chaos cards, without filtering debuffed copies. Drunkard/Merry Andy change future `round_resets.discards`, but not the immediate discard counter when gained or removed mid-round. Lua applies both where appropriate.

Evidence: `E/card.py:757–768, 795–806`; `E/run_init.py:389`; `E/shop.py:512`; `L/card.lua:588–605, 651–668`.

### Other direct deviations: blinds and run transitions

### D27 — Cerulean Bell's forced selection is not enforced

The engine marks a card `forced_selection` but neither legal-action construction nor play/discard validation requires it in the selected set. Players can ignore the forced card. The flag also lacks the full source selection/unselection lifecycle. The source ties it into hand highlighting and prevents manually unselecting the forced card.

Evidence: `E/blind.py:365`; `E/game.py:529, 766`; `E/actions.py` hand action construction; `L/blind.lua:575–586`; `L/cardarea.lua` highlight handling.

### D28 — Crimson Heart rerolls on discards, not just prepared draws

Python executes the random debuff on every `drawn_to_hand` invocation, without testing `prepped`. Lua only changes its Joker when prepared and clears the preparation afterward. A discard therefore permits an extra debuff reroll in Python and advances the stream incorrectly. The permanent-perishable interaction is D25.

Evidence: `E/blind.py:378`; `E/game.py:766` redraw path; `L/blind.lua:488, 588–603`.

### D29 — Ox uses a changing hand target and wipes newly earned money

Lua compares against `current_round.most_played_poker_hand`, fixed by the run's prior update, and drains money in `debuff_hand` before card/Joker scoring. Python recomputes `hand_levels.most_played()` after this hand's history/scoring and sets money to zero after scoring dollars have been awarded. A hand can become newly most-played and incorrectly trigger Ox; valid earnings from the triggering hand are also erased.

Evidence: `E/game.py:672–690`; `L/blind.lua:562`; `L/functions/state_events.lua:128–137`.

### D30 — Pillar's played-this-ante markers never reset at the new ante

Python marks played cards but `_advance_ante` does not clear those markers. Lua clears them on boss defeat. A card played several antes earlier can still be debuffed by a later Pillar.

Evidence: `E/game.py:589, 1760`; `L/functions/state_events.lua:266`.

### D31 — Hook bypasses normal discard effects

The engine's Hook implementation directly removes/moves random held cards. Lua calls the discard pipeline with the Hook flag, including applicable per-card discard callbacks and Purple seals. Mail-In Rebate, discard scaling/destruction and seal effects consequently differ. Source's Hook-specific exclusions, such as Burnt Joker's guard, must be preserved too; treating Hook exactly like a voluntary discard would also be incorrect.

Evidence: `E/game.py:2475`; `L/blind.lua` `press_play`; `L/functions/state_events.lua` `discard_cards_from_highlighted`.

### D32 — Amber Acorn performs one shuffle instead of three, and defeat cleanup is incomplete

Source shuffles the Joker area three times on `aajk`; Python does it once. The order and stream state differ even though the animation is omitted. Source defeat also performs boss cleanup such as returning flipped Jokers/cards to their proper facing and clearing boss debuffs. Python manually clears playing-card debuffs and restores Manacle/Juggle hand size, but does not call an equivalent full defeat operation; Acorn-facing and boss-applied Joker state can persist.

Evidence: `E/game.py:1668–1740, 1927`; `L/blind.lua:190–209` and `Blind:defeat`.

### Other direct deviations: consumables and card mutation

### D33 — Death copies by creation order, not hand position, and copies only part of the card

The handler chooses the highest `sort_id` rather than the rightmost highlighted card in hand order. The fallback copy only replaces base/enhancement/edition/seal, leaves the target's other ability data in place and aliases the edition object. Lua copies the selected right-hand card's full relevant ability state.

**Reproduced:** arrange a newer 2 to the left of an older Ace carrying +25 permanent bonus; Death produces two 2s, retaining +25 on the old Ace's object. Source produces two Aces with the copied Ace's bonus.

Evidence: `E/consumables.py:432`; `E/game.py:2060`; `L/card.lua` Death branch; `L/functions/common_events.lua:2158` `copy_card`.

### D34 — Ectoplasm always costs one hand size

Source increments its run-wide usage penalty: successive uses cost 1, 2, 3, … hand size. Python always returns -1 and maintains no increasing counter.

**Reproduced:** two uses reduce base hand size 8 to **6**, rather than **5**. Missing Negative slots are separately C12.

Evidence: `E/consumables.py:948`; `L/card.lua` Ectoplasm application around `1470–1510`.

### D35 — Ankh checks capacity before destroying the other Jokers and retains Negative

The general result applier resolves Ankh's copy before removing its other Jokers. At a full row, the copy is created then discarded for lack of space, leaving just the survivor after destruction. Source makes room and produces the copy. The Python deepcopy also preserves a Negative edition on the copy, while source explicitly strips it; it retains the original sort identity instead of creating a fresh card identity.

**Reproduced:** Ankh with five ordinary Jokers leaves **one**, not two.

Evidence: `E/consumables.py:1003`; `E/game.py:2094, 2168, 2180`; `E/card_factory.py:541`; `L/card.lua:1434`; `L/functions/common_events.lua` `copy_card`.

### D36 — Consumable changes do not immediately refresh boss debuffs

Source `set_base`/`set_ability` reevaluates the blind after a card is changed. Python enhancement/suit/rank/copy application does not do so. Changing away from a debuffed suit or converting a card into/out of Stone/Wild can leave the old debuff until an unrelated later draw refresh, potentially through the very next play.

Evidence: `E/card.py:209, 214`; `E/game.py:2048–2080`; `L/card.lua:97–147, 223–365`.

### D37 — Playing cards created in an open pack go to the deck instead of its dealt hand

The general descriptor and Cryptid paths use `SELECTING_HAND` as their only hand-placement condition; `PACK_OPENING` routes copies/new cards to the deck. Source places Cryptid/Familiar/Grim/Incantation creations into `G.hand`, including while an Arcana/Spectral pack is open. Subsequent picks in the same pack therefore have a different set of available targets. Some `add_to_deck` descriptors also always append to deck, regardless of phase.

Evidence: `E/game.py:2128–2158, 2218`; `E/consumables.py:718, 810–858`; `L/card.lua:1206–1213, 1336`.

### D38 — Grim consumes an extra rank RNG draw

The shared `_roll_card_spec` draws from the rank list even when Grim's list contains only Ace. Source assigns Ace directly, then draws only the suit from `grim_create`. This changes Grim's suits and later values of that stream. Familiar and Incantation's rank-before-suit ordering agrees with source and is not a finding.

Evidence: `E/consumables.py:780, 828`; `L/card.lua:1319–1327`.

### D39 — Edition consumables reject debuffed editionless Jokers that source allows

Wheel of Fortune, Ectoplasm, and Hex filter out debuffed candidate Jokers. Source's corresponding eligible-Joker caches filter edition status, not debuff status. These effects can therefore become unusable or select a different Joker while a boss/perishing debuffs candidates.

Evidence: `E/consumables.py:524, 948, 967` and `can_use_consumable`; `L/card.lua:1470` and `update` eligible-edition pools around `4210`.

### D40 — Tarot usage is not counted, while Black Hole is incorrectly recorded as Planet usage

Normal Tarot use never increments the tally that Fortune Teller reads. `_track_planet_usage` is only called for Planets and Black Hole; applying it to Black Hole incorrectly records a Spectral as a Planet and writes it as `last_tarot_planet`. Consequently Fool can copy Black Hole instead of preserving the previous Tarot/Planet identity. Source's usage bookkeeping distinguishes the sets.

Evidence: `E/consumables.py:616–673`; `E/game.py:2039`; `E/jokers.py:1178`; `L/functions/common_events.lua` `set_consumeable_usage`; `L/card.lua` Fool/Black Hole use branches.

### D41 — Changing suits loses original-suit tie information; Checkered card keys remain stale

`Card.set_base` reconstructs the base and loses `suit_nominal_original`; Lua retains that original value across base changes. Checkered's private `_change_suit` similarly overwrites the original-suit nominal, and does not update `card_key`. Converted Clubs can remain keyed as Clubs while their suit is Spades. This changes identity-based observations and some tied-card sorting; it is not a change to Checkered's basic 26-Spades/26-Hearts counts.

Evidence: `E/card.py:209, 320`; `E/deck_builder.py:149–183`; `L/card.lua:109, 138–144`; `L/back.lua:239–253`.

### Other direct deviations: generation, shop, tags, and profiles

### D42 — Created-Joker editions and enhancement-gated pools are disconnected

Generic created Jokers outside `area in ('shop', 'pack')` never receive an edition roll, unlike Lua, which rolls Joker editions regardless of these sticker-only area conditions. Judgement, Wraith, Soul, Riff-raff, and tag-generated Jokers are affected where their path uses blank-area descriptors. Separately, the factory reads `deck_enhancements`, but normal runs never maintain it; enhancement-gated Jokers such as Stone Joker, Steel Joker, Glass Joker, and Lucky Cat remain unavailable despite eligible cards being present.

Evidence: `E/card_factory.py:378, 398–425, 555–578`; `E/pools.py:454`; `L/functions/common_events.lua:1956–2053, 2133–2155`.

### D43 — Bans/profile information are lost in ante generators and hidden-card checks

`assign_ante_blinds` does not forward banned keys or discovery data to its tag draws, banned keys to voucher selection, or banned bosses/custom `win_ante` to boss selection. The standalone tag helper forwards discovery but also omits bans. Challenge restrictions therefore do not constrain all generated offers/blinds. Hidden Soul/Black Hole chance handling also lacks the source's existing-card/Showman and ban checks; forced keys bypass source-style banned-key fallback.

Evidence: `E/tags.py:439, 479–572`; `E/vouchers.py:89–137`; `E/card_factory.py:353–369`; `E/pools.py` `check_soul_chance`; `L/functions/common_events.lua:1901–2053, 2093–2115, 2338–2383`.

### D44 — Profile progression is not implemented; custom challenge voucher timing is unsafe

`fresh_profile` and `apply_profile_to_game_state` expose unlock/discovery state, but pool filtering deliberately skips the unlock test, and normal run callbacks do not implement source career/unlock/discovery progression. A fresh profile is therefore not simulated faithfully. Fully unlocked play is a legitimate narrower assumption, E07, but not an implementation of profile parity.

For custom challenge definitions, challenge vouchers are applied before `initialize_run` overwrites hand/Joker/consumable limits and round resets from starting parameters. A custom challenge starting with Crystal Ball/Antimatter/Grabber can lose the effect despite marking the voucher used. The supplied 20 definitions do not use those particular starting vouchers, so this second issue is an exposed-API edge case, not an additional failure of those 20 built-in definitions.

Evidence: `E/profile.py:94–128`; `E/pools.py:376, 440`; `E/challenges.py:582`; `E/run_init.py:272–295`; `L/functions/common_events.lua:1988` and unlock/discovery functions; `L/game.lua` `start_run` voucher setup.

### D45 — Shop Tarot/Planet generation permits hidden pack-only replacements

The general factory defaults `soulable=True`, and the normal shop slot path does not override it. Source's `create_card_for_shop` passes no soulable flag; booster generation opts into it explicitly. Shop Tarot/Planet slots can therefore roll Soul/Black Hole in Python when the corresponding source shop slots cannot.

Evidence: `E/card_factory.py:263–277, 363`; `E/shop.py:383`; `L/functions/UI_definitions.lua:742` `create_card_for_shop`; `L/card.lua:1731–1755`; `L/functions/common_events.lua:2093`.

### D46 — Generation advances RNG and excludes cards even when there is no room

`_resolve_create_descriptors` creates/registers a card before checking destination capacity. Effects like Cartomancer/Vagabond/Superposition can draw a card, mutate pool exclusion, and then silently discard it when full. Source checks room/consumeable buffers before creating. Multiple deferred creators likewise do not share equivalent reservation state. A visually empty effect can still alter the seeded future in Python.

Evidence: `E/game.py:2180–2227`; `E/jokers.py:2055, 2089, 2102, 2121`; `L/card.lua:2545, 3743, 3762, 3787`.

### D47 — Eternal/Perishable compatibility is not enforced by setters

`Card.set_eternal` and `set_perishable` set flags directly, without source's center compatibility constraints and mutual-exclusion behavior. Once callers provide the stake flags or explicit setters are used, incompatible self-destructing/scaling Jokers can receive stickers they should not. This is distinct from the normal-run flag-wiring failure C07.

Evidence: `E/card.py:407–414`; `E/card_factory.py:406–414`; `L/card.lua` `set_eternal`, `set_perishable`.

### D48 — Some RNG call-site keys/counts differ even though the RNG primitive matches

Hallucination uses `hallucination`, while source uses `halu` plus the ante. The factory always consumes the rental stream even when Rentals are disabled; Lua short-circuits that draw when the flag is off. The latter is a stream-state difference, normally isolated from gameplay until that stream is subsequently used under changed flags, not proof of changed White Stake shop contents by itself. Other confirmed gameplay-relevant call-count omissions are C02, D16, D32, D38 and D46.

Evidence: `E/jokers.py:2167`; `E/card_factory.py:416`; `L/card.lua:2337`; `L/functions/common_events.lua:2145`.

### D49 — Endless numeric behavior differs from Lua doubles

Blind amounts become Python arbitrary-precision integers after decimal rounding, whereas source remains double precision. Example: scaling 1/ante 17 gives Python `42000000000000000000000000`, versus Lua's double `4.2e25`, which is not exactly that integer. Overflow is also handled differently: the tested ante-39 Python calculation raises, while the supplied Lua function yields NaN. Scoring `math.floor`/integer conversions have analogous non-finite risks. These are direct extreme/endless numeric differences, not defects established for ordinary ante-1–8 thresholds.

Evidence: `E/data/blind_scaling.py` `get_blind_amount`; `E/scoring.py:847`; `L/functions/misc_functions.lua` `get_blind_amount`; `L/functions/state_events.lua:1031`.

### D50 — Credit Card does not enable debt purchases; free items can be blocked while in debt

Live purchase/voucher/booster/reroll handlers and legal-action construction compare cost against raw dollars, ignoring `bankrupt_at`. Source affordability allows the Credit Card's debt margin and explicitly permits zero-cost actions. Python also rejects a zero-cost card when dollars are negative. Recording the credit limit as a passive is insufficient.

Evidence: `E/game.py:1077, 1202, 1218, 1364`; `E/actions.py:423–464`; `E/card.py:764`; `L/functions/button_callbacks.lua` affordability callbacks and `buy_from_shop`.

### D51 — Voucher redemption does not immediately resize/reprice the current shop

Overstock/Overstock Plus update `shop.joker_max` but do not create the newly available current-shop card. Clearance Sale/Liquidation set a discount percentage but do not recompute existing cards/vouchers/packs or owned sell values. Source calls shop-size and card-cost updates during voucher application. Future generation partially benefits, while current offers remain wrong.

Evidence: `E/vouchers.py:185, 253`; `E/game.py:1191`; `L/card.lua:1886, 1914`.

### D52 — Retcon's recorded price and boss-reroll action surface are wrong

Python describes/records unlimited **free** boss rerolls for Retcon; source Retcon allows unlimited rerolls at the normal $10 price. Neither Retcon nor Director's Cut is connected to a player boss-reroll action in the engine, so both also lack their intended decision entirely. A Boss Tag's automatic reroll is a different operation and does not implement these vouchers.

Evidence: `E/vouchers.py:318–340`; `E/actions.py` action union; `L/functions/button_callbacks.lua` `reroll_boss` / `reroll_boss_button`.

### D53 — Uncommon/Rare tags do not make their created Joker free; coupon state is incomplete

The forced-rarity branch creates a priced Joker and never applies the source coupon. Edition tags do set `cost = 0`, but do not preserve `ability.couponed` or run the matching price recalculation; a later price recomputation can lose the free status, and edition sell value can remain stale. Coupon handling must cover both generated-rarity and modified-edition paths.

Evidence: `E/shop.py:357–401`; `E/game.py` coupon tag application; `L/tag.lua:350–459`; `L/card.lua:369–384`.

### D54 — Orbital Tag chooses a new random hand at redemption rather than its advertised target

Python draws from the hand list when applying the tag. Source stores an `orbital_hand` associated with the blind's offered tag, chooses among the appropriate visible hands when building the offer, and preserves that target for Double Tag copies. The engine omits that advance information and can level a different/unrevealed hand; duplicated tags can choose independently.

Evidence: `E/tags.py:228`; `E/game.py:2522`; `L/tag.lua:105–111, 193–197, 324–328`; `L/functions/UI_definitions.lua` orbital-choice setup.

### D55 — Buying an extra Voucher Tag voucher clears the ordinary ante voucher

All redemptions set `current_round.voucher = None`. Source only clears it for the non-`extra` voucher; the extra tagged voucher should not erase the ordinary outstanding voucher for later shops. Python does not preserve the distinction through redemption.

Evidence: `E/game.py:1191–1215`; `E/shop.py:404`; `L/card.lua:1850`.

### Other direct deviations: actions and public helpers

### D56 — `step` accepts actions that source cannot perform

Capacity is checked in parts of legal-action construction but not enforced by live `BuyCard`/Buffoon pick handlers or `_gain_joker`. Direct actions can overfill ordinary Joker/consumable slots. `SellCard.area` is not restricted to owned Jokers/consumables, so other state lists can be sold. Play/discard indices are not required to be unique, permitting the same physical card to be processed more than once. Consumable targeting accepts negative indices and silently filters some out-of-range indices rather than rejecting the action. These are correctness defects of the exposed transition API even if a particular agent never sends such actions.

Evidence: `E/game.py:121, 529, 766, 1061, 1107, 1163, 1288, 1990`; `L/cardarea.lua` highlighting constraints; `L/functions/button_callbacks.lua` buy/sell/play eligibility.

### D57 — Pack picks bypass consumable eligibility and omit valid player decisions

`PickPackCard` pops/spends a choice and calls the handler without `can_use_consumable`. Unusable cards or invalid target sets can consume a pick as a no-op. Conversely, legal actions explicitly omit **all Spectral-pack picks** due to a bot-interface limitation, despite the engine exposing such picks. Packs also disallow owned-consumable use, selling, and hand/Joker arrangement that source permits; the shop lacks a buy-and-use consumable action when storing it is impossible. These are available-strategy differences, not UI representation alone.

Evidence: `E/game.py:1288, 1136, 1441, 1463, 1480`; `E/actions.py:486–504`; `L/functions/button_callbacks.lua:2155–2247`; `L/card.lua` `can_use_consumeable`, `can_sell_card`.

### D58 — Exported CardArea helpers only implement a subset of source area semantics

`CardArea.draw_card` always pops the end, appends at the end, and enforces a simple destination limit regardless of source/destination type. Source draw/remove/emplace behavior depends on area type, supports hand overfill for certain effects, and invokes flip/debuff behavior. `add_to_highlighted` also omits source highlighting limits and forced-selection rules. The main game loop uses raw lists and implements some operations separately, so these helper differences must not all be counted as additional observed run-loop defects. They are direct differences if these documented source-mirroring helpers are used as the public primitives.

Evidence: `E/card_area.py:63–112, 195`; `L/cardarea.lua` `emplace`, `remove_card`, `add_to_highlighted`; `L/functions/common_events.lua:386`.

### D59 — Statistics/profile-facing state is incomplete or differently counted

The engine increments `current_round.jokers_purchased` for every shop card, not only Joker purchases, and only maintains a subset of source `round_scores`, rank/suit play history, career and discovery counters. The source maintains these at specific play/purchase/use events. Some missing fields already cause effects above; the remaining ones mean run telemetry/save/profile outputs cannot be treated as an equivalent source snapshot. This is lower priority if callers explicitly consume only the supported gameplay fields.

Evidence: `E/game.py:580–590, 924, 1082, 1396`; `E/run_init.py:59`; `E/profile.py`; `L/functions/state_events.lua:473–482, 651–652`; `L/functions/button_callbacks.lua` shop purchase bookkeeping.

## Functionally equivalent, but different representation

These are not remediation findings by themselves. Equivalence is limited to the stated behavior; related integration defects above remain defects.

### E01 — Python records/enums and dictionaries instead of Lua objects/globals

Dataclasses, center keys, enum hand/rank/suit identifiers, and explicit dictionaries can represent the same state as Lua tables and `G.*` objects. Likewise, zero-based Python action indices are equivalent to one-based Lua positions when translated consistently. C01 is not such a translation because it changes the time at which the counter is observed.

Evidence: `E/card.py`, `E/state.py`, `E/actions.py`; `L/card.lua`, `L/game.lua` initialization.

### E02 — Synchronous phases instead of rendering/events, where causal ordering is preserved

Skipping animation, sounds, sprites, easing and status text is appropriate for a headless engine. Returning a mutation descriptor instead of scheduling a Lua event is equivalent when the caller applies it exactly once at the same logical point, with the same capacity and ordering constraints. The lost/deferred effects catalogued above do not qualify. Face-down information, sales during packs, Acorn shuffles, and pre-scoring money changes are gameplay, not removable presentation.

Evidence: `E/game.py`, `E/scoring.py`; `L/functions/state_events.lua` and `L/engine/event.lua`.

### E03 — Explicit seeded RNG state and Python implementation of the LuaJIT generator

The tested pseudohash/stream/random implementation is functionally equivalent for the sampled seeded draws, despite storing state explicitly rather than using `G.GAME.pseudorandom` and LuaJIT's generator. This is supported by 5,000 exact comparisons, not inferred from similar probability distributions. Different call sites/keys/counts, such as D48, remain direct deviations.

Evidence: `E/rng.py`; `L/functions/misc_functions.lua` `pseudohash`, `pseudoseed`, `pseudorandom`.

### E04 — Grouped hand evaluation rather than the source's search structure

Python's rank-grouping and straight/flush search structure can return the same poker-hand components without reproducing Lua's loops. The sample found no non-High-Card component differences, including Four Fingers/Shortcut/Smeared cases. That supports the sampled component equivalence, not all scoring integration or every possible hand. D01 explicitly records the component that did differ.

Evidence: `E/hand_eval.py`; `L/functions/misc_functions.lua` hand-evaluation functions.

### E05 — Fixed negative Stone rank for rank matching

Lua returns a random negative value for a Stone's effective rank; Python returns -1. Both are equivalent for rank predicates/grouping that exclude nonpositive ranks. This is not a declaration that all Stone behavior matches: face classification, nominal sorting, and baseless generated Stones differ as recorded above. The random source-side negative number is not itself a requirement to add gameplay RNG consumption to Python's named seeded streams.

Evidence: `E/card.py:565`; `L/card.lua` `get_id`; both hand-evaluation implementations.

### E06 — Cosmetic booster variant identifiers can be collapsed

The supplied source has multiple art variants with identical pack kind, size, choices and other gameplay config. Selecting a fixed equivalent Buffoon/Charm/Meteor variant where source selects between those identical variants changes the center/art identity but not pack options under the same gameplay generation state. The pack-generation streams use type/context/ante rather than the cosmetic variant's identity. This conclusion applies only to genuinely identical configs, not Normal/Jumbo/Mega substitutions, missing effects on opening, or different RNG calls in the generation itself.

Evidence: `E/shop.py:241`, `E/tags.py`; matching `P_CENTERS` booster configs in `L/game.lua`; `L/card.lua:1691–1780` pack generation.

### E07 — Fully unlocked profile as an explicit simulation precondition

Using an all-unlocked/all-discovered profile is equivalent to that particular source save and reasonable for a run simulator. It is different from implementing fresh-save progression or accepting and respecting arbitrary `Profile` values; those unsupported behaviors are D44. This audit does not mark the all-unlocked default itself as a gameplay bug under that precondition.

Evidence: `E/profile.py:72–92`; source pool unlock/discovery filters.

### E08 — Pure hand-level and Plasma arithmetic

Recomputing level chips/Mult from initial values and increments is equivalent to source `level_up_hand`'s formula for supported finite levels. Plasma's final `floor((chips + mult)/2)` applied to both values is likewise the same gameplay transformation. Returning a result object instead of updating display fields is representational. Secret visibility (D13), missing level-up triggers, and extreme numeric behavior (D49) are not covered by this equivalence.

Evidence: `E/hand_levels.py:63–77`; `E/back.py:173–179`; `L/functions/common_events.lua:464`; `L/back.lua:108–172`.

## Coverage boundaries and rejected suspicions

The audit traced the main engine action loop, scoring and Joker dispatch, playing-card effects, consumables, blinds, shops/packs/pools, tags/vouchers, deck/stake/challenge initialization, round economy and profile-facing state. It compared definition data separately from runtime wiring. Bot protocols, observation encoders, policy strength, play-order search optimality, training environments and transport correctness were not validated as equivalent implementations of the source UI. Restrictions embedded in the **engine's** legal actions were included because they directly change the allowed game.

No full graphical-source replay, exhaustive Joker-pair test matrix, all-seeds pool comparison, persistence round-trip proof, or non-finite endless-game compatibility test was completed. The numeric test described above is a bounded probe. Consequently “no deviation found” for a sampled component must not be read as certification of the whole system.

Some plausible suspicions were checked and **not** promoted to findings:

- Replacing a normal enhancement preserves Python's permanent Hiker bonus; the +25 probe remained +25. Death's incomplete copy is a separate issue.
- Familiar/Incantation draw rank before suit in both versions; only Grim's extra singleton-rank draw is a confirmed difference.
- The 20 challenge **definitions** match; failures are application/consumer behavior.
- Several existing fixes do correctly wire Idol/Castle targets, ordinary Plasma final scoring, initial back vouchers, and portions of rental-adjusted interest. Their presence does not repair the independently missing context/lifecycle paths.
- Most-played/hand-group discrepancies were not inferred from Joker descriptions; findings use the supplied source conditions, which can be more specific than the displayed text.

## Minimal integrated reproduction pattern

Run from the repository root using the existing virtual environment. This creates in-memory run state only and demonstrates C01 without patching the engine:

```python
from jackdaw.engine.actions import SelectBlind, PlayHand
from jackdaw.engine.card_factory import card_from_control
from jackdaw.engine.game import step
from jackdaw.engine.run_init import initialize_run

gs = initialize_run('b_red', 1, 'AUDIT')
step(gs, SelectBlind())
gs['hand'] = [card_from_control({'r': '2', 's': 'H'})]
step(gs, PlayHand((0,)))
print(gs['hand_levels']['High Card'].played)  # engine: 2; source: 1
```

For isolated Joker probes, add a correctly initialized Joker through `_gain_joker` before blind selection, then control the hand/blind/money for the stated case. This private helper is used only to construct fixtures; it is not a recommended external integration API. Integrated regression tests should compare state **at the source's logical observation points**, including card identity/order, counters, money, capacity, effects, and RNG stream state—not merely the final hand total or presence of a handler.
