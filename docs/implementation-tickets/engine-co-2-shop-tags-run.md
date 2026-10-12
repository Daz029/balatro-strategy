# Engine close-out, ticket CO-2: local findings in shop, vouchers, tags, run init, RNG

Branch `engine-closeout`, after CO-1 (read `docs/implementation-tickets/engine-co-1-REPORT.md`).
Same reading list, source path, citation rule and structural rules as CO-1
(`docs/implementation-tickets/engine-co-1-local-scoring.md`, "Rules" section) —
they all apply here verbatim. Lua wins over this ticket and the audit text.

Same per-finding method as CO-1: integration test through `step()` /
`initialize_run` / the real shop path; run it on the unmodified code and record
whether it fails; "already fixed" is a valid outcome (keep the test, no code
change); otherwise fix and see it pass.

## Findings

- **C14 (remaining half)** — Magic Deck and Ghost Deck starting consumables are
  stored as keys but never instantiated (`back.lua` `apply_to_run`). After
  `initialize_run` the consumable area must hold real cards created through
  `lifecycle` / the factory exactly as Lua creates them (check Lua for the
  creation path, edition/sticker rolls and RNG keys — if Lua consumes no RNG,
  neither may we). Anaglyph and Green were fixed earlier — confirm with a
  test each, no code change expected.
- **D45** — shop Tarot/Planet slots must not roll The Soul / Black Hole:
  Lua's `create_card_for_shop` passes no soulable flag; only boosters opt in
  (`UI_definitions.lua:742`, `card.lua:1731-1755`). Test with a seed sweep that
  would hit a Soul under the bug.
- **D48 (remaining half)** — the factory consumes the rental RNG stream even
  when Rentals are disabled; Lua short-circuits that draw when the flag is off
  (`common_events.lua:2145`). Hallucination's key half is already fixed —
  confirm with a test. This changes shop sequences at White Stake: say so in
  the report, and regenerate any checked-in fixture ONLY from the Lua oracle,
  never from our own engine.
- **D51** — redeeming Overstock / Overstock Plus mid-shop adds the newly
  available current-shop card; Clearance Sale / Liquidation reprice the existing
  shop cards, vouchers, boosters and owned sell values immediately
  (`card.lua:1886, 1914`; follow what Lua calls — shop-size update and
  `set_cost` over the relevant areas).
- **D54** — Orbital Tag's hand is fixed when the tag is OFFERED
  (`tag.lua:105-111`, `orbital_choices[ante][blind_type]` storage added in Phase
  1) and copied by Double Tag (`tag.lua:193-197`); redemption levels that stored
  hand (`tag.lua:324-328`). Only visible hands are eligible.
- **D55** — redeeming the Voucher Tag's EXTRA voucher must not clear
  `current_round.voucher`; only the ordinary ante voucher clears it
  (`card.lua:1850`).
- **D43 (vanilla part only)** — Soul / Black Hole hidden-chance handling must
  apply Lua's existing-card / Showman checks (`common_events.lua` around
  2093-2115; read it). Ban-list forwarding is CHALLENGE-only and challenges are
  being made to raise in CO-3 — do not implement ban forwarding; do confirm in
  the report that no vanilla (non-challenge) path populates banned keys.
- **D59 (gameplay part only)** — `current_round.jokers_purchased` counts only
  JOKER purchases (find its Lua writer and readers). The rest of D59 (career,
  round_scores telemetry) is out of scope — do not implement.

## Rules

Tests in `tests/engine/test_closeout_shop.py`, one class per finding ID. Full
suite green with `BALATRO_SOURCE` set; `tests/env` legal-action masks unchanged
(report). Do not commit. Do not touch legality, Spectral pack picks or boss
reroll (deferred by the user). Report to
`docs/implementation-tickets/engine-co-2-REPORT.md` and repeat it as the final
message (same sections as CO-1).
