# Engine CO-2 report: shop, tags, run init, and RNG close-out

## Result

Implemented the in-scope CO-2 findings against Balatro 1.0.1o. The new
integration suite had 10 failures and 4 passes on the unmodified parent; it now
has 14 passes. No legality, Spectral-pack-pick, or boss-reroll work was touched,
no checked-in fixture changed, and no commit was made.

## Findings

- **C14 (remaining half) — failed on parent: yes.** Magic Deck now starts with
  two real Fool cards and Ghost Deck with a real Hex, created through
  `create_card` and `lifecycle.emplace`. Forced keys consume no pool, hidden,
  edition, or sticker RNG (`back.lua:184-195`, `common_events.lua:2087-2150`).
  Green Deck and Anaglyph Deck were already fixed by Phase 1 and Phase 4,
  respectively. Proving tests: the four methods in
  `TestC14StartingDeckEffects`.

- **D45 — failed on parent: yes.** The ordinary shop-slot factory now passes
  `soulable=False`, matching `create_card_for_shop`; Tarot and Planet shop
  slots therefore cannot become The Soul or Black Hole, while boosters still
  opt in (`UI_definitions.lua:742-776`, `card.lua:1731-1757`). The regression
  sweeps seeds until both hidden streams exceed 0.997; the buggy Planet case
  produced Black Hole on the parent. Proving test:
  `TestD45ShopHiddenCards::test_seed_sweep_shop_tarot_and_planet_slots_never_become_hidden_cards`.

- **D48 (remaining half) — failed on parent: yes.** Rental RNG now
  short-circuits when Rentals are disabled and still advances at Gold Stake
  (`common_events.lua:2137-2146`). This changes White-Stake shop RNG/sequence
  parity: the `ssjr` stream is no longer advanced or serialized; because RNG
  streams are independently named, current card identities remain unchanged
  until a flow later enables/uses Rentals. No fixture was regenerated.
  Hallucination's `halu..ante` key was already fixed by Phase 4. Proving tests:
  both methods in `TestD48RngCallSites`.

- **D51 — failed on parent: yes.** Redeeming Overstock or Overstock Plus in an
  open shop now fills the added slot through the real shop factory
  (`common_events.lua:1097-1115`). Clearance Sale and Liquidation immediately
  call `set_cost` over all live owned and shop cards, repricing shop cards,
  vouchers, boosters, and owned sell values (`card.lua:1917-1923`). Proving
  tests: both methods in `TestD51VoucherShopRefresh`.

- **D54 — failed on parent: yes.** Orbital Tag now chooses only among visible
  hands when the tag is offered, stores the hand under
  `orbital_choices[ante][blind_type]`, carries it on the awarded tag, and
  redeems without a second RNG draw. Double Tag copies the same stored hand
  (`UI_definitions.lua:1506-1515`; `tag.lua:103-111,191-198,320-328`). Proving
  test: `TestD54OrbitalTagOffer::test_offer_fixes_visible_hand_and_double_tag_copies_it`.

- **D55 — failed on parent: no; ticket prose rejected in favor of Lua.** The
  supplied 1.0.1o source unconditionally assigns
  `G.GAME.current_round.voucher = nil` in `Card:redeem` at `card.lua:1850`.
  Therefore a Voucher Tag's extra voucher does clear the ordinary voucher in
  this source, contrary to the finding text. The engine already matched Lua
  and was not changed. Proving test:
  `TestD55VoucherTagRedemption::test_extra_voucher_clears_current_voucher_like_lua`.

- **D43 (vanilla part) — failed on parent: yes.** Hidden-card handling now
  follows the source's eligible sets (Soul for Tarot/Spectral/Tarot_Planet,
  Black Hole for Planet/Spectral) and suppresses the corresponding draw when
  that card already exists unless Showman is active
  (`common_events.lua:2087-2100`). Challenge ban forwarding was not added.
  `banned_keys` is initialized empty in vanilla, and the only engine writer
  that populates it is `apply_challenge`; no non-challenge path populates
  banned keys. Proving tests: both methods in `TestD43HiddenCardChecks`.

- **D59 (gameplay part) — failed on parent: yes.**
  `current_round.jokers_purchased` now increments only after confirming the
  bought shop card is a Joker. Consumable and playing-card purchases do not
  increment it (`button_callbacks.lua:2440-2449`). Career and `round_scores`
  telemetry remain out of scope. Proving test:
  `TestD59JokerPurchaseCounter::test_counter_ignores_consumables_and_counts_jokers`.

## Changed pre-existing expectations

- The hidden-card boundary unit now tests Tarot, not Joker: Lua permits Soul
  for Tarot/Spectral/Tarot_Planet and never for a generic Joker create
  (`common_events.lua:2088-2093`).
- Orbital unit coverage now supplies and asserts the hand fixed at offer time;
  the Double Tag unit fixture carries the same `orbital_hand` and `blind_type`
  fields as a real awarded tag (`tag.lua:103-111,320-328`).
- No checked-in shop-oracle expectation or fixture changed.

## New divergence found (not fixed)

No new engine divergence was found. The D55 ticket/audit claim conflicts with
the supplied Lua source and was resolved in Lua's favor as described above.
(Review re-checked: `Card:redeem` clears `current_round.voucher` at
`card.lua:1819` when `shop_voucher`, then again unconditionally at `1850`.)

## Review fix (2026-10-12)

- **D43, Spectral roll order — fixed in review.** `check_soul_chance` returned
  as soon as the Soul roll hit, so for Spectral the Black Hole roll was never
  drawn (stream desync) and a double hit gave Soul. Lua
  (`common_events.lua:2088-2099`) has no early exit: both rolls draw and a
  Black Hole hit reassigns `forced_key`. Now accumulates `forced` and returns
  once. Proving test
  `TestD43HiddenCardChecks::test_spectral_rolls_both_streams_and_black_hole_overrides_soul`
  (scripted RNG; a natural double hit is ~1 in 1.1e5 seeds), seen failing on
  the pre-fix `pools.py` (`'c_soul' == 'c_black_hole'`). The early return
  pre-dated CO-2.

## Verification

- Parent close-out baseline: **10 failed, 4 passed**.
- Current close-out file: **14 passed**.
- `BALATRO_SOURCE=... uv run pytest tests/engine tests/env -q`:
  **2122 passed, 1 skipped**; all **29/29** transition-oracle tests ran and
  passed.
- Script tests not requiring unavailable optional ML/analysis packages:
  **313 passed, 8 skipped**.
- Focused shop/tag/structure gate: **170 passed**.
- Ruff on every changed Python file: **clean**. `git diff --check`: **clean**.
- `tests/env` legal-action masks are unchanged: no action/enumeration or env
  file was modified, and the full env suite passed.

The exact mandated combined command could not collect because this checkout
lacks optional `torch` and `scipy` packages (the same six pre-existing script
modules reported by CO-1). Engine/env/oracle and every unaffected script test
are green as reported above.
