# Pre-Regeneration Handoff

**Audience:** any agent implementing the pre-regen work. This document assumes you
have NOT read the grilling transcript. It tells you the sequence, the exact work,
and — most importantly — the specific ways this work goes wrong. Read the whole
Pitfalls section before writing any code. The decision record (the *why* behind
every choice here) is CLAUDE.md → "Pre-regeneration build plan — GRILLED AND LOCKED
(2026-07-12)". If this document and CLAUDE.md ever disagree, CLAUDE.md wins; fix the
drift.

## Context in three paragraphs

The hand agent (h1) is about to be retrained from scratch: new BC demonstration
data, new observation schema, new action head (Candidate B, an autoregressive
pointer). All demonstration shards get regenerated ONCE, on the 9600X, at
~12s/example × ~42k examples. Because that regen is expensive and happens exactly
once, **every change that affects label semantics or shard schema must land before
it**. That set of changes is this handoff.

The new data has two sources: the existing domain-randomized stages 1-4 (regenerated
with the new schema and fixes), and a NEW "harvested" stage (~8k examples) — real
mid-run states captured from rollouts of the final shop agent (s0 = `s0_a4_v4`,
deterministic) paired with the current hand agent (h0.5 =
`runs/hand_ppo/hand_ppo_2000000_steps.zip`), then labeled by the exact solver.

Work splits into three phases. **A** (harvest + evidence) is schema-independent and
runnable immediately on this machine. **B** (schema bump + solver changes) is where
all label-semantics changes live. **C** (selection + labeling front-end) is gated on
B. The regen itself is out of scope.

## Sequence and gating

```
A1 harvest script ──> A2 run passes + reductions ──> (corpus banked)
                                └─> A3 fingerprint eval ──> gates B6

B1 add_to_deck fix ─> B2 feature bump ─> B3 best_joker_order ─> B4 labels/width/tier-1
                                                                    └─> B5 prescreen+validation
A3 result ──────────────────────────────────────────────────────────> B6 (conditional)

[all of B done] ──> C1 selection manifest ──> C2 snapshot-fed labeling front-end
                                                   └──> regen (9600X, not this scope)
```

Hard rules:
- A1/A2 can run in parallel with all of B (blobs are engine state, not encoded obs).
- Nothing in C runs until ALL of B is merged (B6 only if A3 triggered it).
- A3 must be resolved (triggered or cleared) before B6's go/no-go, and B6 —
  if it happens — must finish before C2.

## Task specifications

### A1 — `scripts/harvest_s0_rollouts.py` (new)

Drive full runs and capture every hand-turn decision state.

- Construct `ShopGymEnv(win_ante=8)` with `hand_policy` = h0.5 via
  `HandCheckpointPolicy`, shop policy = PPO checkpoint `s0_a4_v4`, deterministic
  (argmax) by default; `--sample-shop` switches the shop policy to sampling
  (h0.5 stays deterministic in BOTH modes — it is the partner being targeted).
- Capture mechanism: wrap the hand policy in `HarvestingHandPolicy(inner, sink)`.
  The interception point already exists — `shop_run_adapter.py` `_advance()` calls
  `self._hand_policy(self._gs)` on every SELECTING_HAND step. The wrapper pickles
  `self._gs` (same mechanism as `ShopRunAdapter.snapshot_state`: full state,
  RNG included) BEFORE delegating to the inner policy, then returns the inner
  policy's action unchanged.
- A record = engine blob + metadata row: `record_id` (= `{run_seed}_{turn_idx}`),
  `run_seed`, `turn_idx`, `ante`, `blind_type`/`boss_key`, `hand_size`, `dollars`,
  `hands_left`, `discards_left`, `source` (`det`/`sampled`), `git_sha`,
  `schema_note` (free text, e.g. engine test count at capture time).
- **Capture EVERYTHING** — no subsampling, no per-run caps at capture time.
  Blobs go in per-worker/per-run shard files (append-friendly); metadata goes in ONE
  flat table (JSONL is fine) so later queries never unpickle a blob.
- Also capture SHOP-state snapshots (a second sink at shop decision points, same
  runs) tagged `kind=shop` — these feed V_curve later; bank them now, one pass.
- Seeds: `HARVEST_{i:08d}` for the deterministic pass, `HARVEST_S_{i:08d}` for the
  sampled pass. NEVER any seed with the `EVAL_` prefix.
- Tests: blob restores byte-identically and continues (reuse the RNG round-trip test
  pattern from `tests/env/test_shop_run_adapter.py`); metadata row matches its blob
  (unpickle a sample and cross-check ante/dollars/hand_size); wrapper is transparent
  (inner policy's actions unchanged, run outcome identical with/without wrapper).

### A2 — run passes + reductions/readout

- ~1200 deterministic runs + ~500 sampled runs. Seconds per run; runs locally.
- Emit free reductions as group-bys over the metadata table, **deterministic corpus
  only**: per-ante `$` marginals at hand-turn entry (feeds stage 1-4 money regen)
  and the hand-size histogram (informs B's max decode length).
- Emit the coverage readout (both corpora, split by source): ante marginal
  (state counts at ante ≥2, ≥3), distinct jokers owned, distinct owned-joker
  key-sets (build families), hand-size tail. This readout has NO pass/fail
  threshold — it tunes the manifest's det:sampled ratio in C1 (default 75:25).

### A3 — discard-bias fingerprint eval (+ archetype decomposition)

Decides whether B6 (the banked-discard solver fix) happens. Two signals; the
trigger is their **conjunction**:

1. **Locate:** h0.5 deterministic clear rate ÷ shard label-mean `p_clear`
   (the free solver-ceiling trick), bucketed by starting
   `(hands_left, discards_left)`. Symptom = recovery in discards≥2 buckets
   materially below discards=0 buckets (>5 recovery points, outside a bootstrap CI).
2. **Attribute:** h0.5's discard *frequency* per bucket vs the solver labels'
   discard frequency (from shards, free). The label bias has a known sign — it can
   only push toward discarding TOO EAGERLY. Bias survived = h0.5 discards at or
   above the teacher's rate in the deficit buckets. Directionless error or
   UNDER-discarding = NOT the label bias; do not build B6.

Greedy (`GreedyHandPolicy`) as a control on stage 1 only. The flush/straight/pair
archetype decomposition of eval recovery runs in this same pass (calibrates how much
recovery to expect from the B2 features; it gates nothing).

### B1 — `add_to_deck` at injection (`jackdaw/env/hand_play_adapter.py`)

`reset()` builds jokers with bare `create_joker(key)` and never applies acquisition
passives — injected Juggler/Troubadour/Turtle Bean/Stuntman states carry a hand size
the real engine would never produce. Fix:

- After `_apply_scaling_state` has run for all jokers, call each injected joker's
  `add_to_deck(gs)` exactly once. Full passive — do NOT cherry-pick `h_size`.
- Add a flat hand-size tail knob to `HandPlayConfig` (e.g.
  `hand_size_delta_range` + a tail probability, mirroring the 20% flat money-tail
  pattern) for coverage beyond what sampled builds produce.
- Tests: Juggler injection deals 9 cards; decayed Turtle Bean deals base + decayed
  h_size (NOT +5); Stuntman deals 6; passives applied exactly once (no
  double-application on any path); Drunkard bumps discards on top of the sampled
  value; deterministic per seed.

### B2 — observation feature bump (`jackdaw/env/observation.py`, demo writer)

- Per-card +3: suit-count-of-my-suit /5; rank-count-of-my-rank /4; best
  straight-window occupancy among 5-rank windows containing my rank (occupancy =
  distinct ranks present ÷ ranks needed). Wheel window (A-2-3-4-5) included. Under
  Four Fingers, window length 4 (flushes also count at 4 — reflect in the suit-count
  normalization ONLY if the engine flag path confirms it; verify, don't assume).
  Under Shortcut, windows tolerate single-rank gaps. Flags via
  `get_hand_eval_flags(jokers)`.
- GC +21: per-suit counts /hand_size (4); per-rank counts /4 (13); max suit count
  /5 (1); best window occupancy (1); Four Fingers bit, Shortcut bit (2).
- Trigger-match matrix `trigger_match[card, joker_slot, {scored, held}]` (bool,
  written to shards) + `joker_center_key_id` int array (frozen vocab ids — from the
  same sorted-centers.json mapping the shop obs uses; the VOCABULARY FREEZE test
  applies).
- Trigger predicate: `(card, joker, gs) -> (scored: bool, held: bool)`. Implement a
  4-class taxonomy with a build-time coverage check that CLASSIFIES EVERY joker key
  in the vocabulary (unclassified = hard error at import/test time, not silent
  zeros):
  - Class 1 static per-card (suit/rank/face/parity conditions).
  - Class 2 state-dependent per-card (Ancient, Idol, Castle, Mail...) — reads gs.
  - Class 3 set-level (Flower Pot, Seeing Double, Blackboard, hand-type-conditional
    jokers) — all-zero rows BY DESIGN.
  - Class 4 non-card (economy etc.) — all-zero rows.
- Blueprint/Brainstorm: resolve the copy chain via the ENGINE's own resolution code
  path; joker rows gain resolved-target `center_key_id`, target descriptor, and an
  active-copy bit (inactive when pointing at nothing/incompatible → zeroed target
  fields); match rows inherit from the resolved target.
- `schema_version` bump; `train_bc.py` loader updated (up-pads old widths, hard-fails
  on unknown schema).
- Encode budget: everything above must stay O(n) single-pass; the <500µs budget from
  the obs-limitation decision record still applies.
- Tests: known-hand fixtures for each feature (4-flush draw, open-ended straight
  draw, wheel draw, Four Fingers/Shortcut variants); taxonomy coverage (every vocab
  key classified); Ancient predicate tracks the rotating suit across states;
  Photograph marks ALL faces; Flower Pot rows all-zero; Blueprint inherits
  Photograph's face matches; inactive Blueprint zeroed.

### B3 — `best_joker_order` (`jackdaw/engine/play_ordering.py`)

- One function used by BOTH the env execution path and the solver. Signature ~
  `best_joker_order(gs, played_subset) -> ordering`.
- Algorithm: stable sort putting additive-mult jokers before x-mult jokers
  (closed-form optimal for the independent-joker chain); if Blueprint/Brainstorm
  owned, argmax over ≤4 copy-target placements, each scored by one cheap evaluation
  of the candidate play.
- Env (`HandPlayGymEnv` and the shop env's auto-resolved hand phase): compute once
  per COMMITTED play, apply as a persistent mutation of `gs["jokers"]`, then execute
  through `best_play_order` as before.
- Solver: apply the sort once per hand-turn (it is subset-independent); on the exact
  current-hand path, re-run the copy-target argmax per candidate subset; on the MC
  future-hand path, use the hand-turn-entry target (fixed).
- Solver hypotheticals must CLONE the joker list before mutating (extend the
  `fast_clone_*` family). This is the stage-4 shared-blind bug class — assume any
  in-place mutation of shared state during hypothetical evaluation is a bug until
  proven otherwise.
- Tests: brute-force 120-permutation ground truth on constructed boards — pure
  additive, pure x-mult, mixed, with/without Blueprint, with per-card-phase jokers
  (Greedy) present. The closed form is only PROVEN for the independent phase;
  the per-card-phase interleaving is verified empirically here. Also: solver/env
  consistency test (the committed action's label value equals env execution value
  under the same subset).

### B4 — index-set labels, width 40, tier-1 rework (`scripts/generate_hand_demos.py`)

- Labels: `(5,)` int array, ascending, -1 pad; delete `card_target_mask` and the
  `idx >= MAX_HAND_CARDS` GenerationError.
- `MAX_HAND_CARDS_OBS` 12 → 40 in `hand_play_gym.py`; demo writer writes ACTUAL hand
  width; loader up-pads (existing pattern from the 12-width fix).
- Rewrite `validate_label_executability` tier 1 schema-native: indices unique,
  sorted, within actual hand length; 1-5 cards; PlayHand requires hands_left ≥ 1,
  Discard requires discards_left ≥ 1. Tier 2 (engine execution) unchanged.
- Tests: a synthetic big-hand state (hand > 8) with a label touching position ≥ 8
  passes tier 1 and executes through tier 2 (this exact case silently vanished
  under the old tier 1 — pin it).

### B5 — solver prescreen + validation harness (`scripts/hand_solver.py`, new script)

- Prescreen: for hands with n > 8, `best_immediate_play` evaluates only top-k
  template-derived subsets from `rank_templates_cheaply`, selected FAMILY-DIVERSE:
  best candidate per template family first (each flush suit, each straight window,
  rank-line groups), then fill remaining k by rank.
- Validation harness (one-time script): ~50 hands flat over sizes 9-12 (dealt via
  B1's hand-size knob), run FULL brute force once per hand, score prescreen cuts at
  k = 3, 5, 8 (say) from the same evaluations. Metric = regret:
  `p_clear(brute best) − p_clear(prescreen choice)`, both valued by brute force.
  Noise floor: re-solve a sample of n=8 states with different `mc_seed`s, compute
  the same regret between seeds. Accept the smallest k with mean regret ≤ ~1.33×
  the floor. Record the chosen k in CLAUDE.md.
- If no k passes: raise k / widen families; if still failing, the prescreen design
  returns to review — do NOT ship a failing k because the schedule wants it.

### B6 (CONDITIONAL on A3) — banked-discard credit in `estimate_future_hand_distribution`

Do not design this until A3 triggers. If it triggers: the fix must go through B5's
regret harness (prescreen-style validation vs brute force on discard-rich states)
before any label is generated with it. Budget expectation: it will slow labeling;
that cost lands on all ~42k examples — keep the implementation's cost profile
explicit in the PR.

### C1 — selection script → manifest (new, small)

- Input: the metadata table. Output: a MANIFEST file — ordered list of `record_id`s
  (~8k) — plus the parameters used (seed, strata, ratios) embedded in a header.
- Selection rules: ante-stratified; ≤8 records per run with ≤2-3 per ante bucket
  within a run; det:sampled ratio default 75:25 (tune from A2's readout); fixed RNG
  seed so the same inputs always emit the identical manifest.
- The manifest is a versioned artifact — check it in (it's a few hundred KB).

### C2 — snapshot-fed labeling front-end (`scripts/generate_hand_demos.py`)

- New mode: consume a manifest + blob store instead of `HandPlayAdapter` sampling.
  Restore blob → `gs` → the EXISTING solve/encode body (`generate_one_example`'s
  logic from `gs = adapter.raw_state` down runs unchanged on a restored state) →
  `write_shard` into a new stage dir (e.g. `stage5_harvested`).
- `mc_seed` = the `record_id`. Worker partitioning: fixed ranges over the manifest
  order, `--num-workers` EXPLICIT and recorded (see Pitfalls #5).
- On load, compare each record's `git_sha` stamp to the current checkout; mismatch =
  hard error with an override flag (`--allow-sha-mismatch`) that logs loudly.
- Failures (solver exceptions, tier-2 rejects) go to `worker_N_failures.jsonl` and
  skip, per the existing design — but ALSO emit a summary count at end; if >2-3% of
  the manifest fails, stop and investigate rather than shipping a silently-thinned
  stage.

## Pitfalls — read before coding

1. **Tier-1 validation silently deletes the point of the harvest.** The old tier 1
   maps labels into the Discrete(436) mask; any label touching hand position ≥8
   raises and the example is *skipped, not failed*. If B4 ships without the rework,
   phase 2 completes "successfully" and every big-hand example is quietly missing.
   The failure summary in C2 is the backstop; the pinned test in B4 is the fix.
2. **`add_to_deck` ordering and idempotence.** Apply passives AFTER
   `_apply_scaling_state` (decayed Turtle Bean), exactly ONCE per joker (a second
   call doubles hand size), and BEFORE the deal (injection already precedes
   `SelectBlind`, keep it that way). Do not cherry-pick fields out of the passive.
3. **Solver/env divergence on joker order is a label-poisoning bug.** One shared
   function. If the solver values a play under an order the env won't reproduce (or
   vice versa), labels systematically overpay — the 5-card-discard-cap class of bug,
   which wasn't caught until BC consumed the data. The consistency test in B3 is
   non-negotiable.
4. **Hypothetical evaluation must not mutate shared state.** The joker list joins
   the blind/hand-levels/RNG in the `fast_clone` discipline. The stage-4 bug
   (shared `Blind` mutated by hypothetical `score_hand` calls) was invisible for
   months because earlier stages never exercised the mutating branch. Same shape
   here: order mutations during candidate ranking must happen on clones.
5. **Resume partitioning is fixed by (total, num_workers).** 2026-07-05 incident:
   resuming a transferred partial run with a different default worker count
   silently re-partitioned — 925 duplicated + 321 missing examples. ALWAYS pass
   `--num-workers` explicitly; C2 partitions over the manifest, which makes the
   record list stable, but the worker count must still be pinned and recorded.
6. **Seed discipline.** `HARVEST_`/`HARVEST_S_` prefixes are reserved; `EVAL_` seeds
   must NEVER be harvested (leaks the held-out eval suite into training data).
   `mc_seed` for a harvested example is its `record_id`, not the run seed alone
   (two records from one run must not share MC draws).
7. **Blobs are pickles of live engine objects.** An engine class change between
   capture and labeling can break unpickling — or worse, load fine while behavior
   has drifted. That's what the `git_sha` stamp is for: check it loudly in C2.
   If an engine bug fix lands mid-stream (history says it will), decide explicitly:
   re-harvest, or accept and document the skew. Never silently proceed.
8. **Class-2 trigger predicates must read live state.** Ancient Joker's suit
   rotates every round; The Idol's card changes; Castle's suit changes. A
   config-derived static table gives confidently wrong match bits that no test on
   fresh states will catch — test against mid-run states with rotated values.
9. **Class-3 all-zero rows are a decision, not a gap.** Flower Pot / Seeing Double /
   Blackboard / hand-type jokers have NO honest per-card bit ("which card triggers
   Flower Pot?" has no answer). Their signal is the GC set-structure features. Do
   not "helpfully" mark all cards, the scarcest suit, or anything else — fabricated
   per-card signal was explicitly rejected.
10. **Match bits mean "candidate", not "will fire".** Photograph marks every face
    card; only the first scored face gets the ×2. The bit is class membership;
    whether it fires depends on the chosen set/order and is the policy's job.
11. **Copy resolution must reuse the engine's path.** Reimplementing
    Blueprint/Brainstorm chain rules in feature code WILL drift from the engine
    (compatibility rules, chain termination). Resolve via the engine, inherit match
    rows from the resolved target, zero the fields when the copy is inactive.
12. **Regret, not disagreement.** In B5 and B6 validation, two near-tied plays
    "disagree" harmlessly. The metric is the p_clear cost of the choice under
    brute-force valuation. Calibrate acceptance against the n=8 MC-reseed noise
    floor, not against zero.
13. **Family-diverse top-k.** Naive top-k off `rank_templates_cheaply` can return k
    variants of one flush line and starve discard-toward-straight candidates whose
    cheap rank is systematically lower than their exact value. Best-per-family
    first, then fill.
14. **Deterministic-only reductions.** The `$` marginals and hand-size histogram
    describe the DEPLOYED policy's distribution; including sampled-pass records
    smears a deliberately-worse policy into the money prior. Filter on the source
    tag.
15. **The fingerprint trigger is a conjunction.** A recovery deficit in
    discard-rich buckets alone does NOT justify B6 — it's equally consistent with
    plain learning weakness. Only deficit + directional teacher-mirroring (h0.5
    discarding at/above the teacher's known-inflated rate) indicates the solver.
    Under-discarding can NOT be caused by this bias; if you see it, the solver is
    exonerated.
16. **Width 40 is obs-only.** No parameter scales with it, shards write actual
    width, the loader up-pads. Do not regenerate anything for width; the FEATURE
    changes are what force regen. Truncation beyond 40 stays lowest-first (no
    finite width is provably safe — Serpent compounding).
17. **B's decode grammar (post-regen, recorded here so it isn't lost):** picks in
    strictly ascending index order enforced by the per-step mask; explicit stop
    token as a fixed extra logit slot (not positioned at hand size); stop illegal
    at zero picks, forced at five. The mask constraint must be identical at BC,
    PPO, and eval — pin with a parity test when B is built.
18. **Coverage criteria that can never pass are not criteria.** Do not gate the
    harvest on things like "every joker seen N times" — an argmax policy will never
    buy all 150 jokers, and vocabulary breadth is stages 1-4's job. The readout
    tunes the manifest mix; it doesn't pass or fail.
19. **Where things run.** Harvest, reductions, fingerprint eval, all tests: this
    machine. Brute-force prescreen validation (n=11-12) and the regen itself: 9600X
    if local runtime is painful. `runs/` is gitignored — checkpoints move by manual
    transfer; verify `s0_a4_v4` and the h0.5 zip exist locally before A2.

## Definition of done (per phase)

- **A:** corpus banked (both passes), reductions + readout emitted, fingerprint
  verdict written down (triggered / cleared, with the bucket tables), all A1 tests
  green.
- **B:** all feature/label/solver tests green including the brute-force ordering
  test and the big-hand tier-1 pin; prescreen k chosen and recorded; B6 resolved
  (built+validated, or explicitly skipped citing A3).
- **C:** manifest checked in; a smoke labeling run (a few dozen manifest records
  end-to-end into a shard, loaded back by `train_bc.py`'s loader) passes before
  the full 9600X job is queued.
