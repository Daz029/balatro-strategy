# Post-regen training plan — h1 → s1 → merge (GRILLED AND LOCKED 2026-07-17)

Decision record + ordered implementation plan for everything after the h1 hand-demo
regeneration. Companion to the CLAUDE.md sections "h1 / s1 seam", "h1 architecture —
Candidate B COMMITTED", "In-blind merge", and the s1 blocks. Scope: the nine roadmap
items below, each resolved in a grilling session 2026-07-17; the wave plan at the end
is the build order.

Timing anchor: 5 stage3_full examples, single worker, laptop (i5-1340P) = 237.68s
≈ **47.5s/example** — ~4x the historical ~12s figure, driven by the B-phase changes
(v3 encode, trigger matrix, prescreen, B7 gate). RIDER: gate K changed per-example
cost at n=8 (prescreen now runs at every hand size) — **re-time after K merges**
before trusting any wall-clock plan for the regen (~42k examples, 9600X).
RE-TIMED POST-K (2026-07-18, same machine/config + `--dollar-marginals`): 119.15s
for 5 = **23.8s/example, 5/5 generated, 0 failures** — K roughly HALVED per-example
cost (the 218→~15 prescreen cut at n=8 outweighs the kicker-variant fan-out). The
wall-clock plan for the remaining regen stages can be built on ~24s/example
laptop-single-worker scaling.

## The nine items, resolved

### 1. Candidate B BC vs flat-head control — the gate

- **Primary gate: joint NLL of the labeled action + exact-set-match accuracy**,
  head-to-head on the **shared ≤8-position support only**. Comparability rests on the
  monotone mask: every legal set has exactly one reachable pick sequence, so B's
  summed per-step log-probs are the log-probability of the *identical discrete event*
  as the flat head's single softmax row. Same event, same scale, and it is exactly
  what BC training minimizes — instant over the ~3.3k val examples.
- **Secondary readouts**: B's absolute NLL on the ≥8-position stratum (B alone — flat
  cannot be CE-scored there); the flat control's dropped-label fraction (the recorded
  rider); p_clear-head MSE for both models; a memorization canary (sequence CE → ~0
  on ~50 examples) as the smoke check.
- **Sequence-length stratification is part of the gate, not a diagnostic appendix.**
  Every gate run must emit tables by true emitted sequence length and the
  corresponding true set-size stratum; these must not be silently pooled into one
  aggregate. Required metrics are: type-token accuracy; stop-token accuracy;
  per-pick-position NLL; predicted versus true set-size distributions;
  invalid/forced-termination rate under free-running decoding; and entropy by
  decoding step. Pick and entropy tables retain step indices, and free-running
  tables are stratified by the true length of the decoded example.
- **Gate verdict rule:** aggregate NLL and exact-set match remain required, but
  they cannot pass the gate by themselves. The gate is incomplete/fails if any
  required length stratum or metric is missing or only reported in aggregate.
  Candidate B is judged against pre-registered per-stratum thresholds (and against
  the flat control where comparable), with teacher-forced and free-running
  semantics labelled explicitly.
- **Offline decode check:** on a representative held-out subset, compare the locked
  greedy decoder against beam search. Record sequence validity, predicted set-size
  distribution, sequence NLL, and action/set disagreement by true sequence-length
  stratum. This is a gate review input and risk check; it does not silently replace
  the pinned greedy deployment convention with beam search.
- **Winrate = reference readout only**, never the gate. Rejected as primary on two
  grounds: statistical power (BC clear rates run 4–20% with median label p_clear 0.0
  — resolving a 2-point difference needs thousands of episodes per arm) and
  build-order inversion (winrate requires env-side B decoding = item 5, which should
  not sit ahead of the BC gate meant to de-risk B). It IS uniquely informative on
  >8-card states, where flat can still *play* (positions 0–7 always legal) but can't
  be scored — keep it as the qualitative check there.

### 2. KLToBCMaskablePPO rewrite

As specced at the Candidate B section: compound autoregressive distribution,
teacher-forced per-step KL sum for the leash, sequence log-probs for the PPO ratio,
sequence-CE BC. Sequencing fact from the grill: the rewrite **consumes B's
distribution class**, so it cannot lead — it is built in the shadow of the BC runs
(critical path: v3 encoder → B head/distribution → seq-CE BC trainer → BC runs ∥ KL
rewrite).

### 3. Terminal $ term — V_curve verification before wiring

- **Interest ordering, verified two-sided**: FIRST verify the engine — mid-blind
  earnings (Business Card, Rough Gem, gold cards) land BEFORE interest and may cross
  thresholds; end-of-round payouts (Golden Joker, Rocket) land AFTER and must not
  affect it (same rule class as the verified "Investment pays after interest") —
  THEN unit-test our cashout mirror against the engine's ordering.
  **DONE 2026-07-18 — and the verification found FOUR inherited economy bugs**
  (the expected ordering was violated in the live engine): joker end-of-round
  payouts double-counted and leaking into interest; rental double-charged with
  interest on the doubly-reduced balance; held Gold-Seal cards wrongly paid $3;
  Gold Card `h_dollars` stored but never read (held gold cards paid $0). All four
  fixed as separate commits; full record in `docs/engine_changes.md`. Ordering is
  pinned through the real `step()` path in `tests/engine/test_cashout_ordering.py`;
  the mirror (`jackdaw/env/cashout_mirror.py::dollars_after_cashout`) replays the
  engine's own `CashOut` on an RNG-exact clone, so it tracks any future economy
  change by construction. RIDER: s0 (and the harvest, and therefore the V_curve)
  was trained/captured in the PRE-fix economy — payout-joker builds were ~2x as
  lucrative and rentals cheaper-then-costlier than vanilla — so the V_curve is
  mildly optimistic about payout-build money states; s1's own critic retrains in
  the fixed economy, same second-order class as the other partner-era distortions.
- **Sweeps edit engine state, never observation vectors.** Dollars is derived into
  ≥5 obs feature families (`shop_obs.py:207/213/235` — voucher affordability, raw
  shop_context dollars; `observation.py:544/735/887-924` — per-item affordability,
  GC log-dollars, interest + `spendable_above_interest`). Editing the obs vector
  alone produces internally contradictory observations — OOD for the critic, a
  quietly-garbage V_curve that PPO would happily optimize. Rule: restore the
  harvested engine blob → `gs["dollars"] = k` → **re-encode the FULL observation** →
  forward the frozen s0 critic. "Counterfactuals edit engine state, never obs
  vectors."
- **Curve gut-checks**: HARD — weakly monotone nondecreasing in $ at fixed ante
  (a decreasing cell is noise or a bug), values in [0,1] (s0 reward is 1{win}),
  per-cell sample counts reported with a fallback for sparse cells (negative
  dollars, ante 7). SOFT — interest kinks at $5 multiples are a *diagnostic only*,
  downgraded 2026-07-17: this cashout's interest is realized money living inside
  the `dollars_after_cashout` argument (verified by the mirror test, and it moves
  the agent rightward *along* the curve at any $), while future rounds' interest is
  the only interest that appears as shape *in* the curve — and it is smeared by
  whatever the s0 policy does with money before the next cashout. Sharp kinks
  confirm the critic learned interest; their absence is ambiguous and must not fail
  the gate alone.
- Wiring itself is as locked at the h1/s1 seam: clear-gated `1 + V_curve(ante,
  dollars_after_cashout)`, NO decay (deliberate objective change), both recorded
  riders (build-blind ante-average; engine-ordering mirror) stand.

### 4. Mixed start_state_sampler — restored-snapshot validity

The rule **inverts** between injection and restore. Synthetic injection must apply
acquisition passives faithfully (the B1 `add_to_deck` lesson). Restore must apply
**nothing**: the blob already contains every applied effect (hand size, scaling
accumulation, boss history); re-applying a passive double-applies it. Fidelity to
the capture is the point. The real requirements:

1. **Stored-state staleness grep** (the C2 capture-skew rule): an engine fix that
   changes *computation* is inherited free (blobs re-score under current code); a
   fix that changes *stored state* is not (the Idol `id` repair precedent). Any
   engine fix landing between harvest capture and h1 PPO gets the same grep; repair
   on restore if needed.
2. **Consumable tolerance**: `HandPlayAdapter` never injects consumables, so
   `HandPlayGymEnv`'s consumable block has been always-masked forever; a full-run
   snapshot carries real owned consumables. The obs schema already handles it (the
   2026-07-13 shard rider), but the env encode path has never seen one — the
   recorded post-regen test "HandPlayGymEnv consumable-tolerance for restored
   full-run snapshots" covers exactly this. Tolerance only: consumables appear in
   obs; `UseConsumable` stays out of the hand action space until h2.
3. **Terminal-restore guard**: raise on restoring a terminal snapshot (shop-env
   precedent). Harvest captured at decision points so it should never fire — keep
   it loud if it does.
4. **Config anchor stays nonzero** — stage1-4 config sampling never leaves the
   mixture (same coverage-bias logic as the shop reservoir).

Reward semantics need no change: P(clear this blind) from a mid-round state is
honest — banked score and spent discards just make it a harder or easier instance.

### 5. h1 partner wrapper (B-decoding HandCheckpointPolicy)

Contract is minimal — `ShopGymEnv`'s `hand_policy` slot is `game_state → one engine
action`, played out against the real engine (`shop_run_adapter.py:149`); nothing
consumes partner log-probs or entropy (those live in item 2, hand-training side).

- **Greedy per-step argmax under the monotone mask, TYPE TOKEN INCLUDED, pinned as
  THE deterministic decode convention** everywhere the policy deploys (partner
  slot, `eval_hand_policy`, any future harvest). Greedy sequence decode is not the
  joint mode (that needs beam), and it doesn't matter: the partner needs a
  *consistent deployed policy*, not the mode — s1's values are "given how the
  partner plays". Beam rejected: no principled stopping point (width-2 at the type
  token invites the same argument at pick 1), deploy-time decode the training never
  used, compute on a hot path, and a divergence surface between call sites.
- Riders: the wrapper is **parity call site #4** for the byte-identical monotone
  mask (BC / PPO / eval / partner); it encodes the v3 obs and emits the engine
  action directly from picks (type + ascending indices, plays routed through
  `best_play_order`) — it never touches Discrete(436).

### 6. SkipBlind at s1 — offered-tag observability

Action side already specced (one row at canonical 686, cold head, boss = SelectBlind
only). Obs side resolved:

- **24-dim offered-tag one-hot appended to `shop_context`**, indexed by the
  existing `observation.py::_TAG_IDX` (the `_TAG_KEYS` list, `NUM_TAGS=24`);
  all-zero when no skip is offered (boss blinds).
- Why not the embedding route: the **vocabulary freeze forbids it** (tags are not
  in centers.json; adding keys reorders ids and corrupts every shop checkpoint),
  and a second embedding table for 24 items with no pooled sharing and one decision
  site is machinery without payoff. Precedent seals it: held tags are ALREADY
  one-hot in GC `[135:159]` (`awarded_tags` binary) — which also means the Double
  Tag subtlety (skip value depends on held tags) is already observable; only the
  *offered* tag was missing.
- Widening wrinkle: shop_context is a flat MLP input, so this widening is NOT
  covered by the masked-pool argument — it is still no-retrain via
  **zero-initialized new first-layer columns** in the load shim (byte-identical
  outputs on old states; same verification gate as the joker widening).

### 7. Shop joker rows 8 → 15 + SellJoker [687,694)

Decided 2026-07-16 — see the `MAX_JOKER_ROWS` item in CLAUDE.md. Nothing new from
this grill; build the code + shims + split-mapping pins in wave 0, fire at s1.

### 8. s1 script swaps — Φ shaping, floor, chaining

- **Φ = frozen s0 critic fed TRUNCATED observations**: slice the widened obs back
  to s0's schema (joker rows 15→8, drop the tag one-hot) before the critic sees
  it. Policy-invariance never depended on this choice — Ng's guarantee holds for
  ANY state-only Φ; bad Φ costs convergence speed, never optimality. Truncation's
  actual virtues: the critic is evaluated **exactly in-distribution** (truncation
  reproduces s0's own encoding bit-for-bit, since widening is a pure superset),
  and zero new encode paths. The shimmed-widened-copy alternative is the one
  option that is actually OOD (the shared per-row joker encoder would pool rows
  9–15 with real contributions the critic never trained on). PIN the inverse
  property: `truncate(widened_obs(s)) == old_obs(s)`.
- **Φ(terminal) ≡ 0 — the one line where invariance genuinely breaks.** With γ=1
  the shaping sum telescopes to `Φ(s_T) − Φ(s_0)`; `Φ(s_0)` is a per-episode
  constant, but evaluating the critic on terminal states makes `Φ(s_T)` differ
  between win and loss — a real objective change. Shaping on terminal transitions
  is `−Φ(s)`, full stop. Coefficient still decays to zero per project standard.
- **Nextround floor re-baselined against h1** (the partner s1 actually trains
  with), and chaining as documented: `--init-from s0_a4_v4` + `--init-reservoir`
  (reservoir migrates free — snapshots are engine blobs re-encoded at load).

### 9. In-blind merge

Locked 2026-07-16 (CLAUDE.md "In-blind merge" section): pointer subsumes
`SelectTarget`, Death direct-construction + 16-rung ladder, owned-consumable
targeting goes live, tarot-targeting policy learned at the merge. Out of near-term
scope; nothing changed by this grill.

## Ablation checkpoints

Record these as named, resumable checkpoints on the same held-out evaluation
seeds. Each checkpoint should isolate the newly added component from the prior
one, while preserving the BC-gate tables and the standard eval readouts:

1. **B BC versus flat BC** — the Wave 1 architecture/control comparison and BC
   gate baseline.
2. **B PPO without the terminal-dollar term** — the h1 PPO baseline.
3. **Add the terminal-dollar term** — isolate the value of the V_curve-based
   terminal contribution.
4. **Add the restored-state mixture** — isolate restored-state coverage from the
   fresh-start sampler.
5. **Replace the old partner with h1** — isolate the partner-policy change.
6. **Enable the full s1 schema and shaping** — evaluate the widened observation,
   SkipBlind, and Φ-shaping package as the final s1 step.

Do not collapse these into one before/after result: retain the checkpoint,
configuration, seed split, and metric delta for every transition.
Short test runs are sufficient for these ablations; they are implementation and
direction checks, not claims of final training performance. Reserve full-scale
training for the checkpoint that survives the relevant gate.

## Ordered build plan

**Wave 0 — now, during the regen window (all parallel, no gates):**
- Interest-ordering verification (engine first, then the cashout mirror test).
  DONE 2026-07-18 — four inherited economy bugs found and fixed en route
  (see section 3 and `docs/engine_changes.md`).
- V_curve extraction harness (blob restore → dollars edit → full re-encode →
  frozen s0 critic) + curve artifact + gut-checks (hard/soft split above).
  DONE 2026-07-18 (`scripts/extract_v_curve.py`, lookup in
  `jackdaw/agents/v_curve.py`): full det-corpus run — 17,967/17,967 shop
  states, 1.46M critic forwards, 0 failures, ~12.5 min → `data/v_curve.json`
  (gitignored; copy to the training machine with the checkpoint). Gut-checks:
  interest-bracket structure clearly captured (sawtooth dips of ≤0.009
  immediately after each $5 multiple — the 102 "monotonicity violations" ARE
  the bracket nonlinearity, not defects; curve flattens past ~$25-30,
  matching the engine's $25 interest cap); ante 7 flagged sparse (18 states —
  consumers should clamp/smooth there); 5.3% of raw critic outputs fall
  outside [0,1] (unclamped regression head, means stored as-is). The
  pre-fix-economy rider in section 3 applies to this artifact.
- v3 smoke pass: local low-depth shards incl. a few C1-manifest records through
  the C2 front-end and back through `train_bc.py`'s loader — closes the last
  C-phase gate. Re-time per-example cost after K merges.
  DONE 2026-07-18 (laptop, branch `wave-1`): 11 C1-manifest records (8 head-of-
  manifest 6 det + 2 sampled, plus 3 deliberately consumable-carrying found by
  scanning restored blobs) labeled via `--manifest` + `--allow-sha-mismatch`,
  11/11 written, 0 failures. All shards load through `train_bc.py::load_dataset`
  with every validation passing: schema v3, 11/11 flat-compatible, index labels
  legal under reconstructed masks, hand block up-padded to width 40, and — the
  2026-07-13 rider verified end-to-end — the real consumable block comes through
  non-empty (1/2/1 rows, nonzero features) on the consumable-carrying records.
  Pooling with the 4,000-row stage2_k3_relabel brute corpus works (4,011
  examples, per-stage weights applied). Sample rider: all 11 hands were width 8
  (early-ante s0 states); the >8-width loader path rests on the B4 test suite,
  not this sample. Re-time was already done post-K (23.8s/example, header note).
- Build (not fire) the s1 code: SkipBlind row 686, offered-tag one-hot, joker
  rows 8→15 + SellJoker [687,694), load shims, byte-identical-on-old-states
  tests, split k→action-index mapping pins.
  DONE 2026-07-18 behind opt-in `ShopRunConfig.s1_schema` (default OFF =
  byte-identical s0, pinned incl. exact `torch.equal` on the real a4_v4
  checkpoint through `checkpoint_migration.widen_s0_checkpoint`); blind-select
  "proceed" reuses the NextRound row; the tag one-hot rides additively through
  a zero-init Linear (concat into the shared LayerNorm would break
  byte-identity). Not yet wired into `train_shop_ppo.py` — that is the
  s1-kickoff step.

**Wave 1 — the B stack (critical path):**
1. v3-consuming encoder (embedding-gather card encoder, trigger-match
   fixed-weight cross-attention).
2. B head + compound autoregressive distribution + monotone-mask machinery.
   The Wave 1 gate must record, by true sequence-length/set-size stratum:
   type-token accuracy, stop-token accuracy, per-pick-position NLL, predicted
   versus true set-size distribution, invalid/forced-termination rate under
   free-running decoding, and entropy by step. These are gate inputs, not optional
   diagnostics; no verdict may be recorded from aggregate NLL alone.
3. Sequence-CE BC trainer; mask-parity harness (call sites #1–3: BC/PPO/eval).
4. BC gate (item 1 above): NLL + exact-set-match vs flat control on ≤8 support;
   ≥8 stratum + dropped fraction reported; winrate reference only.
5. `KLToBCMaskablePPO` rewrite — in the shadow of the BC runs.

**Wave 2 — h1 PPO (gated on wave 1 + regen done):**
6. Terminal $ term wired into the `HandPlayGymEnv` hook (V_curve lookup;
   clear-gated; no decay).
7. Mixed `start_state_sampler` (item 4 above: nonzero config anchor, consumable
   tolerance, terminal guard, staleness grep).
8. h1 PPO on the 9600X; evals include the discard-bias fingerprint re-run and
   archetype decomposition (calibration, not gates).

**Wave 3 — s1 (gated on h1):**
9. B-decoding `HandCheckpointPolicy` (item 5 above).
10. s1 kickoff: widened obs + SkipBlind live; Φ shaping via truncation +
    Φ(terminal)=0 + decay, replacing `c_ante`; floor re-baselined vs h1;
    `--init-from` + `--init-reservoir`.

**Wave 4 — in-blind merge** (post-s1, as locked).
