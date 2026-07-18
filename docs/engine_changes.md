# Engine changes vs the inherited upstream

The `jackdaw` engine was forked from TylerFlar's `jackdaw-balatro` (126 upstream
commits, ending around the gymnasium-wrapper work, 2026-03). Everything after is
this project. This doc is the running catalog of **engine behavior changes** made
post-fork — every entry changes what the engine computes, so every entry has a
data blast radius (labels, checkpoints, harvest corpora were produced under the
old behavior).

Standing lesson behind most entries (CLAUDE.md "Integration-seam joker bugs"):
a handler whose unit tests are green can still be dead or double-firing at its
real call site. Assert through the integration path (`step()` / `score_hand`),
never the handler alone.

## 2026-07-18 — cash-out economy audit (4 fixes)

Found while verifying the interest ordering for the h1 Terminal-$ term
(`docs/post-regen-training-plan.md` section 3). Ordering suite:
`tests/engine/test_cashout_ordering.py` (drives the real
`SelectBlind → PlayHand → CashOut` state machine). The cashout mirror
(`jackdaw/env/cashout_mirror.py`) replays the engine's own `CashOut` on an
RNG-exact clone, so it tracks all of these by construction.

### 1. Joker end-of-round payouts: double-counted + interest leak (`bd33d4d`)

`_round_won` applied `on_end_of_round`'s `dollars_earned` (Golden Joker, Cloud 9,
Rocket, Satellite, Delayed Gratification) directly to `gs["dollars"]` AND the
same amount flowed through `calculate_round_earnings(joker_dollars=...)` into
`earnings.total` at `CashOut`. Net: paid twice ($1 start + Golden Joker ended
at $16 instead of $9) and the payout crossed $5 brackets to earn interest it
shouldn't (vanilla computes interest on the pre-payout balance; payouts are
cash-out rows, `state_events.lua:1175` vs `:1191`).

Provenance: introduced upstream in `2db4d20` ("Implement full round end");
`a9b62c3` noticed the double call but fixed only the RNG-desync half (the
`joker_dollars=` pass-through), leaving the money double-count. Fix: delete the
direct application; payout flows once through `earnings.total`.

### 2. Rental: double-charged, interest on the doubly-reduced balance (`c07cfff`)

`process_round_end_cards` deducted `rental_rate` directly from `gs["dollars"]`,
then `calculate_round_earnings` — called with that already-post-rental balance —
subtracted `rental_cost` again both for interest (`effective_money`) and in
`total`. Measured: $20 + one rental joker → $22 instead of vanilla's $26
(rental charged twice; interest bracket 14//5=2 instead of 17//5=3). Fix: the
lifecycle pass only tallies; the single charge lives in
`calculate_round_earnings`, keeping vanilla's "rental deducts before interest"
via `effective_money`.

Rider: `economy.py`'s own rental predicate wrongly skipped debuffed rentals —
vanilla `calculate_rental` (card.lua:2271) has no debuff gate (rental is a
sticker, not an ability) — and missed the top-level `card.rental` field. Both
sites now share `round_lifecycle.is_rental`.

### 3 + 4. Held-card round-end money: seal/enhancement mix-up (`86bb8a6`)

One mis-keyed predicate in `_round_won` step 3 produced two bugs:

- **Held Gold-SEAL cards wrongly earned $3** at round end. Vanilla Gold Seal
  pays only when the card is **scored** (already correct in `get_p_dollars`;
  pinned both directions — scored pays in-blind and counts toward interest,
  played-but-unscored pays nothing — in `e15b682`).
- **Gold CARD (`m_gold`) money was completely dead**: `h_dollars: 3` was copied
  into `ability` at creation and never read anywhere. Vanilla pays $3 per held
  gold card at end of round.

Fix: the held payment sums `ability["h_dollars"]` over held non-debuffed cards
(config-driven, no name set — the K1 rot lesson), landing before
`calculate_round_earnings` so it counts toward interest (in-blind money class,
per the verified plan ordering).

### Blast radius (all four)

Economy-wide, live since 2026-03-17: every s0 training run, the harvest corpus,
and all shop evals happened in the old economy (payout-joker builds ~2x as
lucrative as vanilla; rentals overcharged; gold cards worthless held). Hand-demo
**labels are unaffected** — the solver is money-blind (P(clear) only); dollars
feed obs features and sampling marginals, not label targets. The V_curve
(extracted from the s0 critic) inherits the old economy's valuations — recorded
as a rider at the plan's section 3; s1's critic retrains in the fixed economy.

## Earlier engine fixes (pre-dating this doc; details in CLAUDE.md at the cited items)

| Fix | Nature | Where recorded |
|---|---|---|
| `score_hand` passed `jokers=None` to `evaluate_hand` — Four Fingers, Shortcut, Smeared Joker completely inert in-game and in every label | detection flags never derived | K3 block (`03e288d`) |
| Blueprint/Brainstorm ignored `blueprint_compat` — copies fired the 29 incompatible jokers | copy resolution | B2 status (branch `worktree-pre-regen-b2-hand-potential`) |
| The Idol could never fire — `idol_card` cached without the `id` the handler compares | stale round-target cache | B2 status; C1 capture-skew repair |
| Throwback could never fire through real scoring — `GameSnapshot` never received `skips` | integration seam | stage-preset item |
| Marble Joker stone cards scored as normal cards and crashed `reset_round_targets` — enhancement key stored where effect name belongs | `set_ability` misuse | h0.5 item |
| Riff-raff created jokers with no room check — 6 jokers in 5 slots, corrupt scoring + obs crash | missing capacity check | h0 BC-run item |
| Tag wiring: D6 reroll base ignored (`or` vs Lua falsy), reroll cost recomputed losing Surplus/Glut discount + escalating during Chaos free rerolls, Double Tag read a never-written key and dropped non-dollar dups | three dormant bugs | tag-wiring item |
| Shared-`Blind` mutation: hypothetical scoring corrupted live boss state (solver + `best_play_order`) | missing clone | stage-4 item (`fast_clone_blind`) |

Post-fork engine changes that are **deliberate non-vanilla behavior** (not bug
fixes): env-side auto-ordering (`best_play_order` / `best_joker_order` — vanilla
lets the player reorder freely, the env computes the optimum), and the planned
Death direct-construction hook (in-blind merge section of CLAUDE.md).
