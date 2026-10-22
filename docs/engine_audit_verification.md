# Engine audit verification

**Follow-up, 2026-10-21:** the [source fidelity assessment](engine_fidelity_2026-10-21.md)
reproduces a remaining C08 boss-disable completion gap and a frame-cached tally
scoring difference, and identifies limitations in the transition/shop oracles.
The closure table below records the October 12 audit, not a current guarantee
of complete source parity.

Date: **2026-10-12**. Branches `engine-closeout` (off `engine-phase1-state`)
and `engine-legality` (Phase 6, off `engine-closeout`).
Input: the 74 findings in [`engine_audit.md`](engine_audit.md) (C01–C15,
D01–D59). Plan: [`engine_fixing_plan_2026-10-06.md`](engine_fixing_plan_2026-10-06.md).

Every finding has one of these statuses:

- **Fixed**: an integration test drives the real path (`step()`,
  `score_hand`, `initialize_run` or the Lua transition oracle) and was seen
  FAILING on the code before the fix.
- **Raises**: out of scope by user decision (2026-10-06, confirmed 2026-10-12).
  The engine raises `UnsupportedGameModeError` instead of silently running a
  different game.
- **Not a bug**: re-checked against the 1.0.1o Lua source; the engine already
  matched Lua and the audit was wrong.
Legality (Phase 6) landed on `engine-legality` the same day: the engine is
fully vanilla and the environments hold the constraints. Env masks are
reconstructed later (CLAUDE.md TODO 4); until then
`tests/env/test_mask_pin.py` pins the training envs' masks byte-identical.

Summary over the 74 findings: **69 fixed, 3 raise (C15, D44, D49), 1 not a
bug (D55), 1 deleted (D58)**. All 71 in-scope findings are closed. Two
"fixed" rows are partly elsewhere: D43's ban half raises with challenges, and
D59's telemetry half is out of scope. Test paths are under `tests/engine/`.

| ID | Finding (short) | Status | Proving tests |
|---|---|---|---|
| C01 | Hand counters timing / double record | Fixed (P4+P5) | `test_play_sequencing::test_c01_*`, transition oracle |
| C02 | DNA / Sixth Sense / 8 Ball / Hallucination / Certificate creations dropped | Fixed (P4) | `test_effect_pipeline::TestScoringEffects`, `TestMissingDispatches` |
| C03 | Vampire timing, double XMult | Fixed (P5) | `test_play_sequencing::test_c03_*`, oracle |
| C04 | Oops! probabilities alias | Fixed (P1) | `test_state_aliases::TestC04Probabilities` |
| C05 | Money during scoring (Bull vs Vagabond, Ox) | Fixed (P5 ledger) | `test_play_sequencing::test_c05_*`, oracle |
| C06 | Pool exclusion = currently exists | Fixed (P3) | `test_lifecycle::TestC06CurrentlyExists` |
| C07 | Stake sticker flags alias | Fixed (P1) | `test_state_aliases::TestC07ShopStickerFlags` |
| C08 | Chicot / Blind.disable | Fixed (P3) | `test_blind_lifecycle::TestC08ChicotDisable`, `TestC08BossAnteGuard` |
| C09 | Face-down bosses on real draws | Fixed (P5) | `test_draw_discard_sequencing::test_c09_*`, oracle |
| C10 | EOR held-card retriggers; Blue seal last-played Planet | Fixed (P3+P5) | `test_end_round_sequencing::test_c10_*`, `test_lifecycle_wiring::TestC10BlueSealInitialization` |
| C11 | Mr. Bones threshold | Fixed (P5) | `test_end_round_sequencing::test_c11_*`, oracle |
| C12 | Consumable editions via setter | Fixed (P3) | `test_card_setters::TestC12EditionSetter`, `test_lifecycle_wiring::TestC12EditionSetters` |
| C13 | Telescope / Omen / Astronomer / Magic Trick | Fixed (P1–P3) | `test_grabber_wiring::TestC13VoucherAndAstronomerFlags`, `test_lifecycle::TestC13MagicTrickIllusion`, `test_card_setters::TestC13AstronomerCost` |
| C13b | Observatory held-Planet X1.5 | **Fixed (close-out)**: never implemented before | `test_closeout_local::TestC13bObservatory` |
| C14 | Green / Anaglyph / Magic / Ghost decks | Fixed (P1, P4, CO-2) | `test_state_aliases::TestC14GreenDeckEconomy`, `test_effect_pipeline::...anaglyph...`, `test_closeout_shop::TestC14StartingDeckEffects` |
| C15 | Challenge rules | **Raises** | `test_run_init::TestChallengesRaise` |
| D01 | Nominal / unique_val tiebreak | Fixed (CO-1) | `test_closeout_local::TestD01Nominal` |
| D02 | Stone face classification | Fixed (P2) | `test_rules_view::test_stone_king_is_not_a_face_without_pareidolia` |
| D03 | Pareidolia / Smeared threading | Fixed (P2 `Rules`) | `test_rules_view` (Ride the Bus, Flower Pot, Seeing Double, Goad, Faceless) |
| D04 | Lucky Cat trigger field | Fixed (P1) | `test_state_aliases::TestD04LuckyTrigger` |
| D05 | Brainstorm in slot one | Fixed (CO-1) | `test_closeout_local::TestD05Brainstorm` |
| D06 | Spurious Blueprint guards | Fixed (CO-1) | `test_closeout_local::TestD06BlueprintGuards` |
| D07 | Loyalty run-wide count | Fixed (P4) | `test_effect_pipeline::...loyalty_card_fires_on_sixth_run_wide_hand...` |
| D08 | Raised Fist last tied card | Fixed (CO-1) | `test_closeout_local::TestD08RaisedFistTie` |
| D09 | Blackboard on empty held hand | Fixed (CO-1) | `test_closeout_local::TestD09BlackboardEmptyHand` |
| D10 | Debuffed joker targets / tallies | Fixed (P2 tallies; **close-out** joker-on-joker) | `test_grabber_wiring::TestD10*`, `test_closeout_local::TestD10JokerOnJokerDebuffedTarget` |
| D11 | Matador main pass | Fixed (P4) | `test_effect_pipeline::...matador...`, `test_play_sequencing::test_d29_matador_main_pass_sees_ox_trigger` |
| D12 | Blocked hand still runs after pass | Fixed (P5) | `test_play_sequencing::test_d12_*`, oracle |
| D13 | Secret-hand visibility on play | Fixed (CO-1) | `test_closeout_local::TestD13SecretHandVisibility` |
| D14 | Burglar order | Fixed (P4) | `test_effect_pipeline::...burglar_applies_after_round_reset` |
| D15 | Ceremonial Dagger / Madness | Fixed (P4) | `test_effect_pipeline::...dagger...`, `...madness...` |
| D16 | Marble front | Fixed (P3) | `test_lifecycle::TestD16MarbleFront` |
| D17 | Created/destroyed notifications | Fixed (P3) | `test_lifecycle_wiring::TestD17Notifications` |
| D18 | EOR dollar-bonus timing | Fixed (P5) | `test_end_round_sequencing::test_d18_*`, oracle |
| D19 | Cloud 9 / Satellite / Delayed Gratification | Fixed (P2) | `test_grabber_wiring::TestD19*` |
| D20 | Turtle Bean live hand size | Fixed (P3) | `test_lifecycle_wiring::TestD20TurtleBeanDecay` |
| D21 | Gift Card consumables | Fixed (CO-1) | `test_closeout_local::TestD21GiftCardConsumables` |
| D22 | Rental / losing-round lifecycle | Fixed (P5) | `test_end_round_sequencing::test_d22_*`, oracle |
| D23 | selling_self + Verdant / sell phases | Fixed (P4: selling_self, Verdant; **P6**: sell in every stable phase) | `test_effect_pipeline::TestQueuedSales`, `test_legality::TestD23SellOutsideTheShop` |
| D24 | Perkeo copy | Fixed (P3) | `test_lifecycle_wiring::TestD24PerkeoCopy` |
| D25 | Debuff passives / perishable | Fixed (P3) | `test_card_setters::TestD25DebuffSetter` |
| D26 | Chaos / Drunkard current round | Fixed (P1) | `test_state_aliases::TestD26ChaosFreeRerolls` |
| D27 | Cerulean Bell enforcement | Fixed (P6); interim env shim `env/interim_constraints.with_cerulean_bell` | `test_legality::TestD27CeruleanBell`, `tests/env/test_interim_constraints.py` |
| D28 | Crimson Heart prepped | Fixed (P5) | `test_draw_discard_sequencing::test_d28_*`, oracle |
| D29 | The Ox | Fixed (P5) | `test_play_sequencing::test_d29_*`, oracle |
| D30 | Pillar reset | Fixed (P5) | `test_end_round_sequencing::test_d30_*`, oracle |
| D31 | Hook through discard pipeline | Fixed (P5) | `test_draw_discard_sequencing::test_d31_*`, oracle |
| D32 | Amber Acorn 3 shuffles / defeat cleanup | Fixed (P5) | `test_draw_discard_sequencing::test_d32_*`, `test_end_round_sequencing::test_d32_*` |
| D33 | Death copy by position | Fixed (P3) | `test_lifecycle_wiring::TestD33DeathCopy` |
| D34 | Ectoplasm growing cost | Fixed (CO-1) | `test_closeout_local::TestD34EctoplasmPenalty` |
| D35 | Ankh order / Negative | Fixed (P3+P4) | `test_lifecycle_wiring::TestD35AnkhCopy`, `test_effect_pipeline::...ankh...` |
| D36 | Debuff refresh after consumable change | Fixed (P3) | `test_card_setters::TestD36*`, `test_lifecycle_wiring::TestD36*` |
| D37 | Pack-created card placement | Fixed (P3) | `test_lifecycle_wiring::TestD37CreatedPlayingCardAreas` |
| D38 | Grim RNG draw | Fixed (CO-1) | `test_closeout_local::TestD38GrimRng` |
| D39 | Edition consumables + debuffed jokers | Fixed (P2 candidates; **P6** usability check) | `test_grabber_wiring::TestD39EditionlessCandidates`, `test_legality::TestD39UsabilityCountsDebuffedJokers` |
| D40 | Tarot usage / Black Hole | Fixed (P2; verified CO-1) | `test_closeout_local::TestD40ConsumableUsage`, `test_state_aliases::TestD40*` |
| D41 | Suit-change nominal / Checkered keys | Fixed (P3) | `test_card_setters::TestD41BaseSetter` |
| D42 | Created-joker editions / deck_enhancements | Fixed (P2+P3) | `test_lifecycle::TestD42CreatedJokerEditions`, `test_grabber_wiring::TestD42DeckEnhancementGate` |
| D43 | Bans lost / hidden Soul-Black Hole checks | Fixed (CO-2, vanilla part); **Raises** for bans (challenge-only) | `test_closeout_shop::TestD43HiddenCardChecks` (incl. Spectral both-rolls fix) |
| D44 | Profile progression | **Raises** (all-unlocked profile only) | `test_run_init::TestProfileScope` |
| D45 | Shop slots not soulable | Fixed (CO-2) | `test_closeout_shop::TestD45ShopHiddenCards` |
| D46 | Room check before RNG | Fixed (P4) | `test_effect_pipeline::...eight_ball_full_consumables_draws_no_rng`, `...hallucination...skips_rng_when_full` |
| D47 | Sticker setters | Fixed (P3) | `test_card_setters::TestD47StickerSetters` |
| D48 | RNG keys (Hallucination, rental stream) | Fixed (P4 + CO-2) | `test_closeout_shop::TestD48RngCallSites` |
| D49 | Endless numerics | **Raises** | `test_scope_guards::TestEndlessRaises` |
| D50 | Credit Card debt purchases | Fixed (P6) | `test_legality::TestD50CreditCardAffordability` |
| D51 | Voucher resize/reprice current shop | Fixed (CO-2) | `test_closeout_shop::TestD51VoucherShopRefresh` |
| D52 | Retcon price / boss reroll | Fixed (P6): `RerollBoss` action, $10, Boss Tag shares the primitive | `test_legality::TestD52BossReroll` |
| D53 | Coupon tags | Fixed (P3) | `test_card_setters::TestD53CostSetter`, `test_lifecycle_wiring::TestD53TagPricing` |
| D54 | Orbital Tag fixed at offer | Fixed (CO-2) | `test_closeout_shop::TestD54OrbitalTagOffer` |
| D55 | Voucher Tag extra voucher | **Not a bug**: Lua `Card:redeem` clears `current_round.voucher` unconditionally (`card.lua:1850`) | `test_closeout_shop::TestD55VoucherTagRedemption` |
| D56 | step accepts illegal actions | Fixed (P6 `legality.py`); selling from a non-owned area was already rejected | `test_legality::TestD56StepRejectsImpossibleActions` |
| D57 | Pack picks / Spectral picks / pack actions | Fixed (P6 engine); Spectral picks still masked out by the ENV until the mask rebuild | `test_legality::TestD57PackDecisions` |
| D58 | CardArea helpers | **Deleted** (`card_area.py` removed) | — |
| D59 | Statistics | Fixed (CO-2, gameplay part: `jokers_purchased`); telemetry out of scope | `test_closeout_shop::TestD59JokerPurchaseCounter` |

## Found during verification (not in the original audit list)

- **NEW-CO1-01**: To Do List could re-roll to its current target (Lua excludes
  it, `card.lua:2978`). Fixed; `TestD13...::test_to_do_list_reroll_excludes_current_target`.
- **D43 Spectral roll order**: `check_soul_chance` returned on a Soul hit, so a
  Spectral never drew its Black Hole roll (stream desync) and a double hit gave
  Soul. Fixed; `TestD43HiddenCardChecks::test_spectral_rolls_both_streams_and_black_hole_overrides_soul`.
- **C13b, D10 joker-on-joker**: listed as Part 2 rows in the plan but never
  fixed by any phase. Fixed in the close-out (above).
- **NEW-P5-1-05 resolved** (user decision): the bot's index order is the
  left-to-right play order; `test_play_order_contract.py`.

## Data impact

Everything above changes label semantics or shop/RNG streams (D01 ordering,
D06 copied effects, D08/D09/D13, D34, D38 and D45/D48 RNG streams, D51 shop
contents, C13b/D10 scores, Observatory). Per Part 6 of the plan: regenerate
labels and re-harvest against this engine. Checkpoints are warm starts only.
