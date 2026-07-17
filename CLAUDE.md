# Balatro RL Project — Context Handoff

Simulator: `jackdaw-balatro` (Python, faithful seeded reimplementation of Balatro's engine).
Overall goal: train RL agents to play Balatro, split into two sub-problems: **hand/discard
play within a blind** ("ante-play") and **shop decisions**.

## Key architectural decision: two separate agents, not one

Hand-play and shop are structurally different problems and are being developed on separate
tracks. This split was deliberate, not incidental — see "Why raw RL for shop" below for why
the two tracks ended up with different training strategies.

## Ante-play (hand/discard) track

**Status: exact solver built and working; pivoting to distillation into an RL policy.**

- `hand_solver.py` (already built, in this project) implements an exact-where-possible solver:
  - Template enumeration (flush-by-suit, straight-by-window, rank-count groups) — fixed,
    joker-agnostic; joker awareness comes from calling the real engine's `score_hand`, not
    from hardcoded joker logic.
  - Exact hypergeometric / multivariate-coverage reachability math (the multivariate version
    is required for straights specifically — a flat hypergeometric threshold overcounts when
    a rank is heavily duplicated, since a straight needs ONE card per missing rank, not a
    count threshold across interchangeable successes).
  - Recursive discard-chain solver within one hand-turn (exact, since discards are
    "conditioned" — you pick a target, so the outcome is a clean hit/miss).
  - Monte Carlo estimation ONLY at the hand-to-hand boundary (unconditioned — no target
    exists yet for a hand you haven't seen). This is the one place approximation is
    structurally necessary, not just for convenience.
  - Objective is **P(clear the blind)**, not raw EV — maximizing EV is a different (wrong)
    objective once there's a hard chip threshold: safe/low-variance play is correct when
    comfortably ahead, high-variance play is correct when behind with few hands left.
  - Known limitation found during testing: recursion is correct but SLOW (~40s/decision
    with top_k=3, depth 2) — too slow for use inside an RL loop. Root cause: `best_immediate_play`
    brute-forces C(8,5) subsets at every recursion node, each going through a permutation
    search in `evaluate_value` (needed for order-sensitive jokers like Photograph, which reads
    the first scoring card — order genuinely matters for some jokers, confirmed in source).
    Next speed fix identified but not yet implemented: skip the permutation search entirely
    when no order-sensitive jokers are present in the current joker list.

- **Decision: don't try to make the solver fast enough for real-time RL use.** Instead,
  distill it: use the solver offline (slow is fine — it only runs once per labeled example)
  to generate `state -> action` demonstrations, train a fast policy via behavior cloning,
  then fine-tune with PPO (likely MaskablePPO — discard/play legality varies by state,
  a natural fit for action masking) against the real P(clear)-style terminal reward.

- **BC-to-RL exploration collapse is a known risk, mitigations decided:**
  1. Don't train BC to full convergence (preserves policy entropy for RL to work with).
  2. Pretrain the critic via regression on solver-computed values before RL fine-tuning
     (uncalibrated critic -> noisy advantage estimates -> can reinforce BC behavior for
     wrong reasons, or fail to escape it).
  3. Adaptive-KL penalty toward the BC policy during fine-tuning (AlphaStar-style), decayed
     over training, rather than relying on PPO's clip alone.
  4. Run multiple seeds/settings and compare — local minima here are seed-sensitive.

- **Training the hand-agent in isolation from shop** (so shop doesn't need to be trained/exist
  yet): write a custom `GameAdapter` (it's a `Protocol` in `jackdaw/env/game_interface.py`,
  confirmed swappable) that starts episodes directly in the hand-play phase with an injected,
  domain-randomized joker set / deck / ante / money / hands-discards-left, bypassing the
  normal full-run shop-included reset. Plug into the existing `BalatroEnvironment` unchanged.
  - Curriculum: start with no jokers (validate pipeline) -> small random joker subsets ->
    full random coverage across ante range and hands/discards-remaining distributions.
  - Known distributional-shift risk: random joker subsets won't match what an actual
    optimizing shop policy produces later (a good shop agent concentrates on synergistic
    builds, not uniform-random jokers). Planned fix: two-stage training — broad/generalist
    under randomization first, then a second fine-tuning pass once a shop policy (scripted
    or learned) exists, training against the distribution IT actually induces.

## Shop track

**Status: raw RL from scratch, starting now. No external demonstration data available (see
"Data sourcing" below) — decided NOT to block training on data acquisition.**

- **Bootstrap / iterated best-response loop** (agreed design, not yet implemented):
  1. h0: baseline hand-agent trained under domain randomization (above).
  2. s0: shop-agent trained from scratch, using h0 as its reward oracle for "was this
     purchase good" (this is why hand-play needed to be fast/learned, not just exact-but-slow
     — the shop agent needs to query it constantly during its own training).
  3. Self-play rollout: run s0+h0 together, record the actual hand-phase state distribution
     this induces.
  4. h1: fine-tune hand-agent against that induced distribution (closes the distributional-
     shift gap flagged above).
  5. s1: re-train/fine-tune shop using improved h1. Iterate — should converge in a handful
     of rounds since this is cooperative iterated best-response, not adversarial self-play.

- **Exploration mitigations for the shop agent (150 jokers, combinatorial synergy space),
  none requiring external data** (original list — PARTLY SUPERSEDED by the grilled design
  below: pool curriculum replaced by horizon curriculum, synergy-heuristic shaping demoted
  to evidence-gated; count-based bonus survives as-is):
  - Curriculum on the joker pool itself: small curated synergistic subset first, widen to
    all 150 once the agent reliably finds value in the small pool.
  - Cheap synergy heuristic as potential-based reward shaping, computed directly from
    `jackdaw-balatro`'s own joker effect/trigger-condition data (no external data needed).
  - Intrinsic/count-based exploration bonus for under-visited joker combinations.

- **Potential-based reward shaping is the agreed mechanism for injecting ANY soft-preference
  signal (heuristic-derived or, later, log-derived) into RL training** without risking
  changing what's actually optimal:
  `F(s,a,s') = γΦ(s') − Φ(s)` for potential function `Φ(s)` (state-only, NOT state+action —
  including the action breaks the policy-invariance guarantee). Reference: Ng, Harada &
  Russell 1999, "Policy Invariance Under Reward Transformations." Only speeds up/slows down
  convergence, provably cannot change which policy is optimal. Decay the shaping coefficient
  to zero over training regardless of source.

### Shop-agent design — GRILLED AND LOCKED (2026-07-05); build COMPLETE (2026-07-06)

- **Decision surface (s0)**: SHOP + PACK_OPENING + UseConsumable-in-shop. Blind select is
  auto-resolved to SelectBlind — SkipBlind deferred to s1: with tag call-sites unwired (see
  open item) skipping has cost but no benefit, so exposing it now teaches a false
  "never skip" prior; even wired, skip's value is drowned in h0's hand-play noise at s0.
  CashOut auto-resolved (only legal action, not a decision). Joker reordering excluded
  (same argument as the hand agent's rejected reorder actions: multi-step chains under
  sparse reward are an exploration trap); appendable later if eval shows order matters.
- **Episodes: full runs, pluggable start-state sampler.** Shop-visit-scoped episodes
  REJECTED: a shop visit has no honest local reward, and scoring it with h0's
  P(clear next blind) is systematically myopic (zeroes economy, scaling jokers, consumable
  banking — the point of shop play). Sampler mixture {fresh_run, reservoir_shop,
  reservoir_pack_pending} with an always-nonzero fresh-run anchor (~50%). Reservoir =
  engine-state snapshots harvested at decision points from rollouts of current AND past
  checkpoints (snapshot diversity vs distribution collapse), stratified by ante + coarse
  build features. Restart-distribution changes can't corrupt the objective (reward stays
  honest run outcome) — the only risk is coverage bias, handled by the anchor fraction.
  Cold start self-anneals (early reservoir has garbage builds exactly when policy can't
  exploit realism anyway). Hard requirement: snapshot serialize/restore with exact
  RNG-state round-trip (verify identical continuations).
- **Reward**: `r = 1{run won} + beta(progress) * c_ante * 1{blind cleared}`, gamma=1.0.
  c linear-increasing in ante (normalized so a full clear sums ~1) — a crude sketch of
  true-V increments, so less objective distortion per unit of density than a uniform
  per-blind bonus (which overpays early survival -> "grind safely" bias). beta decays to
  zero (project-standard), so the final optimized objective is exactly P(win) regardless of
  how wrong c is. Raw per-blind reward is NOT potential-based shaping under gamma=1
  (doesn't telescope; it's a real objective change — hence the decay). Env emits components
  in `info`; blending is a training-loop hyperparameter. s1 upgrade: replace c with
  Phi = s0-critic values (true potential-based shaping from a learned V).
- **Action space** (`jackdaw/agents/shop_action_space.py`, canonical append at 436):
  BuyCard x4 (Overstock Plus), RedeemVoucher x4 (tag-stacking headroom), OpenBooster x2
  (engine-fixed), SellJoker x8 (negative-edition headroom; shop obs joker rows widen to 8
  to match — hand demo schema stays 5, pooled encoders are row-count-agnostic in
  parameters so widening at merge is a schema bump not a retrain), SellConsumable x3 +
  UseConsumable x3 (Crystal Ball), Reroll, NextRound, PickPackCard x5, SkipPack = 32
  actions, plus SelectTarget x218 (reuses the hand block's COMBOS enumeration verbatim —
  only sizes 1-3 are ever legal in vanilla, but one combo table everywhere + 4-5-target
  future-proofing is worth 126 permanently-masked rows) = 250 appended; canonical space
  is now Discrete(686). s0's policy head is FULL canonical width with the hand block
  permanently masked: dead params, but merge becomes "canonical index == head row".
- **Targeting is LEARNED via a two-step pending state** (env-side auto-target heuristics
  REJECTED — like tarot usage, targeting has to be learned): carrier action
  (PickPackCard k / UseConsumable j) resolves immediately if no targets needed (planets,
  jokers, Standard-pack playing cards — covered for free); else env enters a
  pending-target state where ONLY legal SelectTarget combos are unmasked; the combo
  completes the engine action. The dealt `pack_hand` occupies the hand-card obs rows —
  invariant: "targetable cards live in the hand rows", making in-blind consumable
  targeting at merge pure reuse of the same family. Pending state must be OBSERVABLE
  (flag in shop_context + "selected" bit on the carrier row), not mask-only — identical
  obs with different legal sets confuses the critic. No cancel action (exploration trap).
  Targeting-event sparsity (few per run, conditioned on dealt layout) is attacked by
  reservoir_pack_pending oversampling — importance-shifting where experience is collected,
  not what's optimal. Evidence-gated fallback if targeting stays noise: potential-based
  shaping from a cheap target prior (policy-invariant, decays) — never action override.
- **Obs**: reuse `encode_global_context` FROZEN as-is + a new small `shop_context` vector
  (reroll cost, free rerolls, pending-target flag, pack_choices_remaining, blinds
  cleared). Entity blocks: hand/pack_hand 8x15 (reused), owned jokers 8xD, owned
  consumables 3xD, shop slots 4xD, pack contents 5xD, vouchers 4xD, boosters 2xD. Shop
  slots / pack contents are mixed-type -> union encoding (type one-hot + cost + the
  type's feature layout, absent features zeroed).
  **Identity = learned embeddings + effect descriptors, concatenated**:
  IMPLEMENTED AS one unified nn.Embedding(NUM_CENTER_KEYS+1=300, 16, padding_idx=0)
  over the whole centers.json vocabulary (`observation.center_key_id` — the mapping
  already existed "for embedding table sizing"), instead of the originally-sketched
  per-type tables — same param budget, sharing property (same item = same vector in
  owned/shop/pack rows) holds trivially, and mixed-type shop slots need no routing.
  VOCABULARY FREEZE: ids come from sorted centers.json keys; changing that file
  reorders ids and corrupts every shop checkpoint (pinned in
  `tests/agents/test_shop_policy.py::TestVocabularyFreeze`). Rationale: a scalar
  ordinal ID has false geometry (numeric neighbors are unrelated jokers; every gradient
  corrupts neighbors) and makes synergy a 150x150 pointwise lookup; embeddings give
  learnable geometry, descriptors (trigger type / effect family / scaling rate, from the
  engine's own joker config data) give day-one generalization and pool-transfer; each
  covers the other's failure mode. Embedding table is inspectable mid-training (t-SNE =
  is synergy learning happening at all).
- **Hand phases resolved by a `hand_policy` callable** on the env (game_state -> engine
  action; same pattern as `adapter_factory`): `GreedyHandPolicy`
  (rank_templates_cheaply-based, ms-fast, deterministic — the test fixture, since tests
  can't depend on a checkpoint that changes every retrain; kept forever as the ablation
  baseline isolating shop-value-vs-hand-skill) now; h0/h1 checkpoint wrapper when
  trained. Partner swaps every bootstrap iteration by design.
- **Curriculum: horizon, not pool** (supersedes the pool curriculum): full 150-joker pool
  from step one, run horizon capped — win = clear ante 2, then 4, then full 8. Pool
  restriction fights the embedding design (129 embeddings cold while trunk geometry
  crystallizes on 21, then distribution shock — self-inflicting the CLAUDE.md
  distributional-shift failure); horizon stages are prefixes of the true objective (no
  unlearning at transitions), early episodes are short (short credit chains when the
  critic is coldest, and cheap). Pool restriction is emergency-fallback only. Count-based
  bonus stays: beta/sqrt(N) on (sorted owned-joker key-set) and (pending-effect,
  target-pattern) pairs, decayed to zero.
- **File layout / build order** (dependency-driven):
  1. engine tag-context wiring (standalone; see open item),
  2. `jackdaw/agents/shop_action_space.py`,
  3. `jackdaw/agents/greedy_hand_policy.py`,
  4. `jackdaw/env/shop_run_adapter.py` (+ snapshot RNG round-trip tests),
  5. `jackdaw/agents/joker_descriptors.py` + `jackdaw/agents/shop_policy.py`,
  6. `jackdaw/env/shop_gym.py` (pending-target state machine lives here),
  7. `scripts/train_shop_ppo.py` + `scripts/eval_shop_policy.py` (smoke on ante-2 horizon
     with greedy partner while h0 finishes).

### Data sourcing investigation — CONCLUDED, don't re-pursue without new data

- Video-based BC: rejected. Requires building a full CV extraction pipeline (state parsing +
  action inference from frame diffs) essentially from scratch, comparable in scope to the RL
  work itself, plus real version-mismatch risk against patched Balatro rulesets. Not worth it
  unless a ready-made extraction pipeline surfaces.
- Player log files: TWO structurally different formats were found in practice, only one of
  which is useful, and neither supports full BC:
  - **Client-side ("Lovely" mod) logs**: DO carry per-item purchase/sale identity
    (`action:boughtCardFromShop,card:X`) and, in one variant, even per-card play/discard
    detail (`MP_RLOG: 4 play 1.2.3.5.7`). BUT never record what else was OFFERED at a shop
    visit — only the action taken, not the rejected alternatives. This means true
    `state -> action` supervised BC isn't possible from this format; usable only as
    **aggregate priors** (contextual purchase-frequency statistics, weighted by session
    outcome/furthest-ante-reached as a cheap skill proxy) fed in as potential-based shaping
    or as a policy-network input feature, never as a hard rule.
  - **Server-side logs** (the ones actually available in bulk — 2GB, confirmed 100% of this
    corpus): only carry high-level sync data (score, location, ante, blind-ready, PvP status,
    `spentLastShop` dollar totals) for opponent-view purposes. NO purchase/play-level detail
    at all — this is a protocol-level limitation (the server doesn't need card-level detail
    to sync PvP state), not a sampling gap, so more of this data won't fix it.
  - Conclusion: shop-side log-informed shaping is DEAD given only server logs are available.
    Server logs are still useful for: (a) calibrating the hand-agent's domain-randomization
    money/ante/hands-left distributions to match real play, since `spentLastShop` and
    `setAnte`/`failRound` patterns are real signal for that. (b) Seeds present in server logs
    could seed real blind sequences/decks for hand-agent training episodes (jokers still
    randomized on top) — not yet implemented.
  - If client-format logs ever turn up in bulk: the potential-based shaping mechanism above
    was specifically chosen so this could be added later without redoing any training —
    bolt it on, no risk of corrupting whatever's already been learned.

## h1 / s1 seam — GRILLED AND LOCKED (2026-07-07)

Decision record from the pre-h1/s1 grilling session. h1 = ONE schema bump + full demo
regen + fresh BC + PPO (a feature-layout change breaks the encoder input widths, so
h0.5 weights can't carry over — every layout change batches into this single bump).
s0 must FINISH before h1 data work starts (the harvested stage, money calibration, and
$-term all consume s0 artifacts). Sequence: reservoir persistence -> s0 (9600X) ->
[while it trains: h0.5 fingerprint evals, engine interest-ordering verification, B head
build — all s0-independent] -> harvest pass -> labels -> regen -> BC -> PPO -> h1 -> s1.

### h1 data

- **Harvested BC stage** (new, ~8k examples): solver-labeled states from a DEDICATED
  rollout pass with FINAL s0 + h0.5 — not a training-time hook (h1 targets the final
  policy's induced distribution, not a mixture over training-time checkpoints; a
  separate pass is also zero risk to the s0 run). Snapshot EVERY hand-turn decision
  point, subsample ~6-8 kept per run, ante-stratified (within-run correlation control;
  mid-round states — discards spent, score banked, real boss history — are exactly what
  single-snapshot generation structurally cannot produce). Harvested states carry hand
  levels, editions, stickers, and scaling accumulation for free — those generation gaps
  dissolve for this stage; stages 1-4 stay base-edition / run-start-levels deliberately
  (coverage breadth; realism lives here).
- **Labels = raw card-index sets, NO positional max** (removing that cap is what this
  step exists for). The shard schema stops encoding an action-space assumption: a
  combo-lookup head consumes them as subset lookups, a pointer head as sorted pick
  sequences — the head choice never forces relabeling. Obs width becomes the only cap:
  `MAX_HAND_CARDS_OBS` 12 -> 40 (user call: with B committed, width is nearly free —
  no parameter scales with it, masked padding contributes nothing, shards write actual
  width and the loader up-pads, and B's decode length is the number of PICKS (<=5) not
  the width — so max it out past any plausible hand: 16 still truncates in degenerate
  builds, a 5-slot hand-size build starts ~22 before Serpent compounding). Truncation
  stays lowest-first as the safety valve regardless — no finite width is provably safe
  (documented at the Serpent item). Masked-block widening is free; the FEATURE bump
  below is what forces regen.
- **Solver big-hand cost** (`best_immediate_play` is C(n,5) per recursion node: 56 at
  n=8, ~4.4k at n=16): template-prescreen for n>8 — rank lines via
  `rank_templates_cheaply`, run the full exact evaluation only on the top-k
  template-derived subsets. The label becomes "exact among prescreened candidates";
  acceptable because PPO-against-the-real-game is already the documented label-bias
  corrector. VALIDATE once before trusting at 11+: ~50 hands of size 9-10, prescreened
  label vs full brute force, measure the p_clear gap.
- **Schema bump contents** (the ONE bump): flush/straight hand-potential features (per
  the obs-limitation item; full spec v4.1 in the pre-regen build plan section), width
  40, index-set labels. Verified 2026-07-07: hand levels (GC [30:90]), editions, and
  stickers are ALREADY in the obs — nothing else rides. EXPANDED 2026-07-12: the bump
  also carries the trigger-match matrix, copy-resolution fields, and joker
  center_key_ids (see pre-regen section). EXPANDED 2026-07-13: shards also STORE the
  real consumable block (harvested states carry genuinely owned consumables; stages
  1-4 write it empty) instead of the loader synthesizing zeros. Labels stay
  consumable-blind (the solver ignores consumables — document at the writer); BC just
  learns an input that is inert for hand-play. What it buys: h2's BC pool won't show
  zeros where live play shows real consumables, removing the one plausible pressure
  to re-regenerate shards at the in-blind merge. In-blind consumable SELECTION itself
  stays at h2 — P(clear-this-blind) is a dishonest objective for consumables
  (Planet/banking value is cross-blind; the shop-visit-episode myopia argument), so
  teaching it in the isolated env would train greedy consumable burning.
- **Stages 1-4 regen config**: money sampled from HARVESTED per-ante dollar marginals
  (dollars at hand-turn entry, stratified by ante), with a ~20% flat tail so BC still
  sees off-distribution money states — retires the flat/uniform placeholder AND the
  server-log calibration dependency.
  WIRED 2026-07-16 (was designed-but-unbuilt — the reductions artifact existed with no
  consumer): `dollar_marginals` + `dollar_flat_tail_prob` (default 0.2) in
  `HandPlayConfig`, sampled per-ante with nearest-harvested-ante fallback (the harvest
  covers antes 1-7 only; marginals include NEGATIVE dollars — Credit Card debt states);
  `--dollar-marginals data/harvest_s0/reductions.json` on `generate_hand_demos.py`
  (data/ is gitignored — copy reductions.json to the regen machine by hand). The B1
  flat hand-size tail is user-locked (2026-07-16) at prob 0.1, delta uniform +1..+4
  (additive after add_to_deck passives, so equivalent to bumping base hand size) and
  baked into ALL FOUR stage presets. `--dollar-tail-prob` / `--hand-size-tail-prob` /
  `--hand-size-delta-range` CLI overrides exist. Regen output goes to a FRESH dir
  (`data/hand_agent_demos_v3/`), keeping the h0-era pool intact per the
  pre_discard_cap backup pattern.

### Rollout / harvest pass — LAID OUT (2026-07-12), not yet built

The dedicated pass that produces the harvested BC stage above, plus three free
byproducts. FINAL s0 for this bootstrap iteration = `s0_a4_v4` (a8 skipped — a4
plateaued at the early-game/hand-partner bottleneck, not the shop; grinding a8 would
consolidate the same ~1% win signal). Partner = h0.5
(`runs/hand_ppo/hand_ppo_2000000_steps.zip`).

**Fan-out — one rollout feeds four consumers:**
- Harvested BC stage (~8k): hand-turn snapshots -> solver-labeled.
- Stages 1-4 money regen: per-ante `$` marginals at hand-turn entry (free reduction).
- Candidate B decode length: hand-size histogram (free reduction).
- `V_curve(ante,$)`: shop-state snapshots + the s0 critic (money sweeps).

**Structural key — the harvest splits into two phases that gate differently:**
- **Phase 1 — rollout + capture (NEW, light, schema-INDEPENDENT, runnable now):**
  `scripts/harvest_s0_rollouts.py`. Drives `ShopGymEnv(win_ante=8, hand_policy=h0.5)`
  with s0 = deterministic `PPOPolicy(a4_v4)` — `win_ante=8` so runs reach natural
  death/win, deterministic to match the DEPLOYED (argmax) induced distribution
  (`--sample-shop` escape hatch if coverage is thin; h0.5 stays deterministic — it's
  the partner being targeted). Interception point already exists at
  `shop_run_adapter.py:149` `engine_step(self._gs, self._hand_policy(self._gs))`: wrap
  the partner in `HarvestingHandPolicy(inner, sink)` that pickles `self._gs`
  (RNG-exact, self-contained — same mechanism as `snapshot_state`) then delegates.
  A record = `{engine_blob, ante, blind_type/boss_key, hand_size, dollars, hands_left,
  discards_left, run_seed, turn_idx}` (metadata cached so stratification/stats need no
  unpickle). Subsampling: SUPERSEDED 2026-07-12 — capture EVERYTHING; the ~8k
  ante-stratified / <=8-per-run cut is a separate seeded selection script emitting a
  manifest (see the pre-regen build plan section below). Also capture SHOP-state
  snapshots (for V_curve) at shop decision points in the SAME pass (one extra sink,
  saves a second rollout). Seeds `HARVEST_{i:08d}` — reserved prefix, DISJOINT from
  `EVAL_` (harvesting eval seeds leaks the held-out suite into training). No solving ->
  seconds/run -> runs on THIS machine. A `gs` blob is engine state not encoded obs, so
  phase 1 has ZERO dependency on the schema bump — bank the corpus against a4_v4 NOW,
  in parallel with the schema-bump / prescreen work (its hand-size histogram also tunes
  Candidate B's max decode length).
- **Phase 2 — label (REUSE, heavy, GATED on the schema bump):** extend
  `generate_hand_demos.py` with a snapshot-fed front-end — restore blob -> `gs` ->
  the EXISTING `generate_one_example` solver+encode body (everything from ~L341 works
  off `gs = adapter.raw_state`, so a restored blob flows through identically) ->
  `write_shard`. What changes is exactly the h1 schema bump, NOT harvest-specific:
  index-set labels (remove the `MAX_HAND_CARDS` positional cap), width-40 obs +
  flush/straight features, and the solver big-hand prescreen for n>8 (validate at 11+
  first). 9600X job (~12s/example; ~8k / 12 workers ~ a couple hours); reuses the
  existing multiprocessing/shard/resume machinery, partitioned over the manifest.

**Free reductions (no extra pass):** per-ante `$` marginals (money regen) and the
hand-size histogram (Candidate B) are group-bys over phase-1 metadata — emit at end of
phase 1.

**Ordering:** phase 1 (any time, bank now) -> [schema bump lands + prescreen validated]
-> phase 2 -> regen -> BC -> PPO. V_curve additionally gates on the s0 critic (a4_v4
already has it). Deep-ante coverage for h1 stays with the retained domain-randomized
stages 1-4; the harvest adds REALISM for the early antes s0 actually reaches.

### Pre-regeneration build plan — GRILLED AND LOCKED (2026-07-12)

Decision record from the pre-regen grilling session. Execution handoff (task specs,
sequence, pitfalls, written for any implementing agent) lives in
`docs/pre-regen-handoff.md` — keep the two in sync. Scope = everything that must land
BEFORE the h1 label regeneration; anything touching label semantics lands here or
forces a second regen.

- **Harvest capture policy: capture EVERYTHING, thin later** (supersedes the
  subsample-at-capture sketch above): every hand-turn blob + metadata row hits disk
  (~10-30k records, 1-3GB); the ~8k selection (ante-stratified, <=8/run) is a SEPARATE
  seeded script over metadata emitting a MANIFEST (ordered record IDs, versioned
  artifact). Phase 2 consumes the manifest only, never the raw corpus. Rationale:
  subsampling parameters become re-runnable queries instead of irreversible
  capture-time commitments; free reductions run over the full induced distribution.
- **Dual harvest pass, banked up front**: ~1200 deterministic runs (`HARVEST_{i:08d}`)
  + ~500 sampled (`--sample-shop`, `HARVEST_S_{i:08d}`); every record carries a source
  tag + git SHA stamp (engine-version-skew check, loud at phase 2). Coverage "check"
  is a READOUT tuning the manifest's det:sampled ratio (default 75:25 det-heavy —
  deployed-distribution anchor logic), not a pass/fail gate ("every joker >=10x"-class
  criteria can never pass, and vocabulary breadth is stages 1-4's job, not the
  harvest's). Free reductions ($ marginals, hand-size histogram) compute over the
  DETERMINISTIC corpus only (sampled runs would smear a worse policy's money behavior
  into the money prior).
- **Record identity** = `{run_seed}_{turn_idx}`: keys `mc_seed` (label
  reproducibility) and phase-2 resume/partitioning (manifest-ordered; explicit
  `--num-workers` per the 2026-07-05 operational lesson).
- **Label encoding**: `(5,)` int array, canonical ASCENDING index order, -1 pad
  (replaces `card_target_mask`). Shape decoupled from obs width permanently;
  structurally caps at the engine's 5-card limit; B consumes it directly as the
  teacher-forced pick sequence. schema_version bump; loader hard-fails on
  unsorted/duplicate/out-of-bounds-vs-hand_mask/empty.
- **Candidate B pick order: canonical-order-as-mask-constraint.** B's per-step decode
  mask additionally requires each pick index > the last (monotone). Every set has
  exactly one reachable sequence; off-canonical prefixes are unreachable (no OOD
  prefixes), there is no order entropy for the KL leash to defend, and sequence CE
  stays comparable to the flat-head control. REJECTED: random-order training (order
  entropy is null-direction exploration — all orderings execute identically),
  marginal-contribution ordering (degenerate on joint-value hands — every flush card
  has identical LOO delta — and undefined for discards). Chain rule over the sorted
  encoding is fully general, so no set distribution is lost. Parity test required:
  the mask constraint must be byte-identical at BC, PPO, and eval. Nothing
  order-related rides the regen.
- **Feature spec v4.1** (the flush/straight bump, fully specced; motivating rubric:
  pooling destroys counts and joint structure, so cross-card / cross-entity facts
  must be injected per-card or into the pooling-immune GC vector):
  - Per-card +3 static: suit-count-of-my-suit /5; rank-count-of-my-rank /4; best
    straight-window occupancy among windows containing my rank (wheel window
    included; window length 4 under Four Fingers; gap-tolerant under Shortcut —
    flags via `get_hand_eval_flags`).
  - GC +21: per-suit counts /hand_size (4), per-rank counts /4 (13), max suit count
    (1), best window occupancy over all windows (1), Four Fingers + Shortcut bits (2).
  - **Trigger-match matrix** `trigger_match[card, joker_slot, {scored, held}]`
    (bools, stored in shards) + `joker_center_key_id` array. Consumed at BC-time as
    fixed-weight cross-attention: card encoder input = own features ⊕ sum over
    matched jokers of (learned embedding over the FROZEN center-key vocab ⊕ 24-dim
    descriptor), scored and held summed separately. REJECTED: scalar trigger counts
    (collapse joker identity), 150-dim per-joker one-hot (sparse, cold weights, no
    geometry), descriptor-sum alone (collapses within-class identity — the
    Hack-vs-descriptor-twin objection).
  - **Trigger taxonomy** (build-time coverage check must classify EVERY vocab
    joker): class 1 per-card static (match = trigger-CLASS membership, "candidate"
    semantics — Photograph marks all faces, not just the first); class 2 per-card
    state-dependent (Ancient/Idol/Castle/Mail — predicate signature
    `(card, joker, gs) -> bool` reading LIVE state; a static config table silently
    mismarks Ancient every round); class 3 set-level (Flower Pot/Seeing Double/
    Blackboard/hand-type-conditional jokers — all-zero rows DELIBERATELY, no honest
    per-card bit exists; the GC set-structure features are their signal); class 4
    non-card (economy etc. — all-zero).
  - **Copy resolution (Blueprint/Brainstorm)**: joker rows gain resolved-target
    center_key_id + target descriptor + active-copy bit; match rows inherit through
    the resolved chain; resolution MUST reuse the engine's own copy-resolution path
    (no reimplementation). Rationale: pooling destroys adjacency, so "has a copy
    effect" alone is structurally uninterpretable. Known-deferred: joker
    scoring-order sequencing (+mult/xmult interleave) stays invisible in obs —
    consistent, engine-scored in labels, second-order.
  - Attention DEFERRED to h2, evidence-gated on the post-bump archetype eval: the
    match matrix absorbs its main motivation, and stacking a second architecture
    change into the B-validation seam makes regressions unattributable.
- **Joker auto-ordering — engine-side `best_joker_order` in `play_ordering.py`**
  (the `best_play_order` precedent; vanilla joker reorder is a free unrestricted
  action, so auto-ordering is vanilla-faithful; agent-visible reorder actions stay
  rejected): closed-form additive-mult-BEFORE-x-mult sort (provably optimal for the
  independent-joker mult chain: (m0+m)*x >= m0*x+m for x>=1) + copy-target argmax
  (<=4 candidates, only when Blueprint/Brainstorm owned). Env: computed once per
  COMMITTED play, persistent vanilla-legal mutation. Solver: sort once per hand-turn
  (subset-independent); per-candidate copy-target on the exact current-hand path
  ONLY; fixed target on the MC future-hand path (matches that path's existing
  approximation tier). ONE shared implementation — solver/env divergence here is the
  discard-cap bug class. `fast_clone` discipline extends to the joker list. Validate
  the sort against brute-force 120-perm ground truth on constructed boards (the
  closed form covers the independent phase; per-card-phase trigger interleaving
  needs the empirical check).
- **`add_to_deck` injection fix**: `HandPlayAdapter.reset` injects with bare
  `create_joker` and NEVER applies acquisition passives (`card.py::add_to_deck`) —
  every stage-3/4 demo that sampled Juggler/Troubadour/Turtle Bean/Stuntman carries
  a FALSE hand size the engine would never deal with that build. Fix: apply the FULL
  passive (no cherry-picking h_size — cherry-picking is a new drift surface, and
  Drunkard's +1 discard on top of sampled ranges is true-conditional-on-build;
  document the small discard-marginal tilt), applied AFTER `_apply_scaling_state`
  (a decayed Turtle Bean must apply its decayed h_size, not the fresh +5). Plus a
  small flat hand-size tail knob in `HandPlayConfig` (same pattern/rationale as the
  20% flat money tail). Voucher-odds-by-ante weighting REJECTED (it models shop
  behavior — the same circularity that killed the evidence gate); big-hand realism
  comes from the mechanism, coverage from the flat tail.
- **Prescreen** (supersedes "~50 hands of size 9-10" above): family-DIVERSE top-k —
  best-per-template-family first, then fill by rank (naive top-k can be k variants
  of one flush line, starving discard-toward-straight candidates). k chosen
  empirically: ONE brute-force validation pass scores all k-cuts simultaneously
  (top-3/5/8...), pick the smallest passing k. Validation set: ~50 hands FLAT over
  sizes 9-12 (11-12 included — that is where extrapolation was being trusted), dealt
  via the new hand-size knob. Metric = REGRET: p_clear(brute-force best) -
  p_clear(prescreen's choice), both valued by brute force (disagreement-rate
  over-counts harmless near-ties). Accept within ~1.33x the n=8 noise floor,
  measured by MC-reseeding the same states. Heavy compute (n=12 is ~800 subsets/
  hand): 9600X if slow locally.
- **Discard-bias fingerprint — two-signal CONJUNCTION gates the solver fix**
  (fix-first rejected: the banked-discard fix multiplies MC-boundary label cost
  across all ~42k labels and is the exact solver-change class that produced the
  discard-cap / shared-blind / id() bugs, against a near-free eval): signal 1
  LOCATES — per-bucket ceiling recovery (h0.5 deterministic clear rate / shard
  label-mean p_clear, bucketed by starting (hands_left, discards_left)); deficit in
  discards>=2 buckets is necessary but NOT sufficient (also consistent with plain
  learning weakness). Signal 2 ATTRIBUTES via the bias's known sign (play-only
  future valuation can only push toward discarding TOO EAGERLY): h0.5 discard rate
  at-or-above the teacher's inflated rate in those buckets = bias survived;
  directionless error or UNDER-discarding = solver not indicated (remedy is
  training, not labels). Greedy control on stage 1 only (greedy discards chaff on
  weak hands and is joker-blind — not a clean no-discard floor). The flush/straight
  archetype decomposition rides the same eval pass. If triggered, the fix (B6) goes
  through the prescreen's regret harness before any label is trusted.
- **Tier-1 executability rework**: the current tier 1 maps labels through
  `combo_to_action` into the Discrete(436) mask — any >8-position label raises ->
  GenerationError -> silently skipped to failures.jsonl, i.e. phase 2 would QUIETLY
  DROP exactly the big-hand examples the bump exists to capture. Rewrite
  schema-native: indices unique/sorted/within actual hand bounds, 1-5 cards,
  play/discard legality vs hands_left/discards_left. Tier 2 (real engine execution)
  survives unchanged — the engine takes raw indices at any position.
- **Build order**: A1 `harvest_s0_rollouts.py` -> A2 run both passes + reductions/
  coverage readout -> A3 fingerprint eval || B1 add_to_deck fix -> B2 feature bump
  -> B3 best_joker_order -> B4 labels/width-40/tier-1 -> B5 prescreen + validation
  -> B6 (conditional on A3) -> B7 discard-ranking fidelity (added 2026-07-14)
  -> C1 selection/manifest -> C2 snapshot-fed front-end
  -> regen (9600X, out of scope). A1 is schema-independent (bank the corpus NOW);
  ALL label-semantics items (B1-B7) must land before C2 runs. Post-regen scope,
  recorded not built: B's monotone-mask parity test, the embedding-gather card
  encoder, `HandPlayGymEnv` consumable-tolerance test for restored full-run
  snapshots.
  STATUS 2026-07-13: A1/A2/B1 done (main); B2 slices 1-3 done (branch
  `worktree-pre-regen-b2-hand-potential`): hand-potential encoders
  (D_HAND_CARD=18 / D_HAND_GLOBAL=256, shared shop constants untouched),
  `jackdaw/env/trigger_match.py` (match matrix + full-vocab 4-class taxonomy
  with import-time coverage hard-fail), copy resolution via the engine's own
  path. TWO engine bugs found+fixed en route, both the Throwback class:
  The Idol could never fire (reset_round_targets stored idol_card without the
  "id" the handler compares against), and Blueprint/Brainstorm ignored
  blueprint_compat (copies could fire the 29 incompatible jokers). B2 slice 4
  done 2026-07-13 (branch `worktree-pre-regen-b2-slice4-schema-v2`):
  SCHEMA_VERSION=2 writer (trigger_match + id arrays + real consumable
  block), `build_observation_v2`/`observation_space_v2` as a VERSIONED SEAM
  (v1 stays byte-identical and the env default per the h0.5 sequencing flag;
  flips only at h1 BC/PPO), v2-only BC loader with width-generic up-pad.
  Slice-4 decisions: v2 NOT frozen until B4 (B4 promotes the completed
  schema to v3 — no v2 datasets in the gap); consumable block = 8 PER-INSTANCE rows in
  engine slot order, tail-truncating (Crystal Ball=3 slots, Perkeo
  negatives unbounded; stacked type+count rows REJECTED — row index must
  stay engine slot index for h2 UseConsumable addressing, and stacking
  would force the re-regen the rider prevents); copy-target fields store
  the frozen-vocab key id, never the descriptor vector. Two post-regen
  riders recorded at handoff pitfall 17: the flat-head control DROPS
  labels touching position >=8 (report the dropped fraction), and B's env
  interface is the label encoding itself (type + ascending picks vector;
  masks built policy-side; v1 Discrete(436) path survives for h0.5).
  STATUS 2026-07-14 (same branch): B5 DONE — `prescreen_play_candidates` +
  prescreen path in `best_immediate_play` (n>8): family = the candidate's
  REALIZED scoring line (template-keyed families are gameable — kicker
  padding lets every weak template piggyback the dominant line); ranking
  is JOKER- and HELD-AWARE (user call: jokerless ranking filters
  joker-favored lines before the exact pass can see them); pair pin at
  index 1 of every k>=2 cut (user call: a pair's cheap rank is weak but
  its value is consistent — no draw, no luck). Validation
  (`scripts/validate_prescreen.py`, 48 hands sizes 9-12): sampled regret
  0.0 at k=3/5/8 vs noise floor 0.022 with IDENTICAL best-in-cut rate
  0.646 across k => misses are candidate-GENERATOR-side, raising k buys
  nothing (that's also the lever if the boundary-stress exposure — mean
  0.12 p_clear at a blind placed exactly at the best play's total — ever
  needs shrinking); minimal passing k=3, PRESCREEN_TOP_K=4 (user margin
  call; SUPERSEDED — the constant is 5 in code as of `718a284`, bumped for
  extra margin, and k has been capture-INVARIANT in every arm measured
  since, so the value buys margin, never capture). Big-hand labels now
  0.7-10s at hand sizes 12-17 (in budget).
  A3 DONE, verdict CLEARED — B6 SKIPPED: discards>=2 recovery deficits
  are large (stage2/3/4 +30/+79/+105 pts, CIs exclude 0) BUT h0.5
  UNDER-discards vs the teacher in exactly those buckets (-0.11/-0.08/
  -0.07, CIs below zero) — conjunction fails on attribution; training
  problem, not labels. Report: `data/fingerprint_a3.json`. Archetype
  calibration for the B2 features: pair recovery beats flush/straight in
  every stage (stage3 0.675 vs 0.467/0.410). NEW pre-regen item B7
  (user-locked 2026-07-14): `rank_templates_cheaply`'s discard-branch
  ranking joker/held-aware too (B5's ranking precedent; label-semantics
  change at every hand size => gates the regen). B7 code + tests BUILT
  same session (`_ranking_score` shared with the prescreen; per-branch
  held = kept + hold/completion overflow; `joker_aware=False` escape
  hatch); its old-vs-new validation run
  (`scripts/validate_discard_ranking.py`) must ACCEPT before any label
  generation — record the result at the B7 spec in the handoff doc.
  B4 DONE (2026-07-15): v3 shards write actual-width hand blocks and
  `(5,)` ascending `card_indices` labels padded with `-1`; loader up-pads
  cards/trigger matches to width 40 and hard-validates label canonicality,
  hand-mask bounds, and action budgets. The legacy flat BC trainer rejects
  wide labels explicitly pending Candidate B, rather than dropping them.
  Verification: Ruff + 64 generator/solver/env B4 tests pass in the native
  WSL environment; the torch-dependent `test_train_bc.py` loader suite
  passed natively 2026-07-16 (13 tests, Torch 2.10.0+cpu) — B4 verification
  fully closed.
  ### B7 discard-shortlist DEPTH-GATED WIDENING (validated + locked 2026-07-16,
  branch `b7-topk-depth-gate` off main)
  The faithful-MC old-vs-new gate (`scripts/validate_discard_ranking.py`, 200
  stage3 states, n_samples=80, `data/discard_ranking_validation.json`) came back
  NET-POSITIVE but not clean: 25 helped / 9 regressed at the production discard
  `top_k=4`. The follow-up top_k SWEEP (`scripts/validate_discard_ranking_sweep.py`,
  k=4/6/8/12/64, `data/discard_ranking_sweep.json`) localized EVERY regression to
  the rank-k truncation boundary: widening the shortlist to 6 heals ~all the hard
  cases (regressions 9->6; 3 of the 4 uniform-worse seeds clear), and k=64 =>
  0 disagreements (B7 is a provable no-op once the box covers all templates).
  DECISION: KEEP B7 and DEPTH-GATE the shortlist width rather than flat-bump —
  `top_k=6` at `discards_left <= 2`, `top_k=4` at `discards_left >= 3`
  (`_discard_shortlist_k` in hand_solver.py). `solve_hand_turn` /
  `solve_hand_for_ante_clear` take `top_k=None` = the gate; an explicit int is a
  fixed box for the validation harness / sweep / existence-proof tests;
  `generate_hand_demos` uses the default, so the regen inherits the gate. This is
  a LABEL-SEMANTICS change (a bigger box can move the argmax at shallow-discard
  nodes) => pre-regen lock, lands before C2.
  - ERROR MODE it targets: B7 ranks discard branches by `p_reach x cheap_value`,
    and `cheap_value` scores ONE idealized completion — an EV of the PEAK hit,
    NOT P(clear). It is threshold-blind, so it over-ranks a high-ceiling
    completion and can drop a higher-CONVERSION discard past the top-4 cut, where
    the exact valuation never sees it. Same single-idealized-hit limitation the
    harness docstring flags; it bites only on the DISCARD side (a redraw
    distribution exists) — the play-side prescreen B5 is exact because the cards
    are already in hand, so B5's clean result is NOT evidence for B7.
  - EXAMPLE (seed DISCARD_RANK_VAL_00000247): hand `Ad Qh Qd Js 9c 9d 5s 2s`,
    jokers incl. Shoot the Moon (+mult per Queen HELD). At top_k=4 B7 dropped
    `discard{Js 9c 9d 5s 2s}` (keeps BOTH Queens — faithfully best at every goal,
    P(clear) 0.45/0.34/0.21/0.075) for a one-Queen line with a flashier single
    completion (uniform -0.175). The two-Queen discard sits at rank 5, so widening
    to 6 re-includes it and the loss vanishes.
  - WHY GATED, not flat-6: solve cost scales ~`(k/4)^discards_left` (measured:
    d_left=1 x1.3@6/x2.0@8; d_left=2 x2.2@6/x4.0@8; d_left=3 model ~x3.4@6/x8@8;
    d_left=0 unaffected). discards_left is ~uniform over {0,1,2,3} in the regen,
    so a FLAT 6 ~doubles the whole regen wall (flat 8 ~4x) with multi-minute
    stragglers on deep big-hand states (one measured state 61s -> 257s at k=8).
    The regressors ALL live at shallow depth (d_left 1-2; seed 247 is d_left=1)
    where the wide box is cheap, so the gate keeps 4 at d_left>=3 to cap the
    `(k/4)^3` tail: blended ~1.7x the regen (vs flat-6 ~2.0x) and a bounded
    per-example worst case.
  - STAKES: low. The residual boundary regressions are the PPO-correctable class
    (real reward is P(clear); the mis-ranked discard is a legal single-step
    action; the A3 verdict "training problem not labels" stands) — PPO makes
    gated-B7 and jokerless converge, so this is a cleaner BC STARTING prior at
    negligible tail cost, not a correctness gate. Verification: ruff-clean (no
    new errors), 168 solver/generation tests pass incl. `generate_hand_demos` on
    the gated default.
  ### C1 + C2 BUILT (2026-07-16, branch `pre-regen-c1-c2-manifest-labeling`)
  All B gates complete, so the C phase ran. C1
  (`scripts/select_harvest_manifest.py`) selects which harvested records get
  labeled and emits the checked-in `manifests/h1_harvested.json`; C2
  (`--manifest` mode in `generate_hand_demos.py` +
  `scripts/harvest_restore.py`) labels them into `stage5_harvested` v3 shards.
  Full records in `docs/pre-regen-handoff.md`. The load-bearing findings:
  - **C1 realized 7,891 records, and the CAPS — not the 8k target — bind.**
    Every (run,ante) bucket has >=3 records so `per_ante_cap=3` always binds;
    det supply is 5,891 (109 SHORT of its 6,000 target), sampled thins
    2,467 -> 2,000. Realized det_frac 0.7465 ~ the 75:25 anchor anyway. Targets
    are CEILINGS: a short source is NOT backfilled from the other (that would
    silently shift the anchor). The ante marginal is deliberately NOT flattened
    — deep-ante coverage is stages 1-4's job, the harvest's is realism at the
    antes s0 reaches, and ante 7 has 20 records total. Manifest lives in
    `manifests/` because `data/` is gitignored.
  - **CAPTURE SKEW IS REAL — and the rule generalizes.** The corpus was
    captured at sha `57f1088` on the 9600X (a commit not in this repo, so the
    check can only report "differs", never diff it). **An engine fix that
    changes COMPUTATION is inherited by the harvest for free — a blob is
    re-scored by current code (blueprint_compat, B3 ordering, B5 prescreen, B7
    ranking all apply automatically). An engine fix that changes STORED STATE
    is NOT: the blob's stale cache wins, because fidelity to the capture is
    what preserves the old bug.** B2's Idol fix is the latter — it added
    `idol_card["id"]` to the round-start cache that the handler matches on, so
    every pre-fix blob would label The Idol as DEAD (~0.53% of hand records,
    silently). Repaired EXACTLY on restore (`id` is a pure function of the
    stored `rank` via the engine's own `_RANK_ID`, so nothing was lost, only
    uncached); verified on real blobs at exactly 2.00x, `_RANK_ID` pinned
    against `CardBase.from_card_key`. Grep for this class whenever a fix lands
    between capture and labeling. A key-set diff CANNOT catch it (the field
    hides inside a cached dict value) — the shape guard is what covers it.
  - **The h1 regen must pass `--allow-sha-mismatch`** (fatal by default;
    pitfall #7's "never silently proceed"). Every failure is tagged by
    exception type, counted and logged (a big count under ONE tag = a
    systematic fault eating a whole CLASS of states); >3% failure stops the run
    rather than shipping a thinned stage.
  Next: the wider smoke run + loading a harvested shard back through
  `train_bc.py`'s loader (the one C-phase gate not yet covered end-to-end),
  then the regen itself on the 9600X.
  ### Kicker variants + prescreen-at-n=8 — GRILLED AND LOCKED (2026-07-16)
  Decision record for `docs/bruteforce_speedup_and_kicker_design.md` (findings
  doc; measurements live there — the 17x budget overrun, the 0.845 capture /
  90% max-regret n=8 verdict, the 27/27 kicker attribution). This is a NEW
  pre-regen item ("K"): a LABEL-SEMANTICS change that gates the remaining
  regen stages. Not yet built.
  - **One path, gated**: kicker fix lands -> gate passes -> delete
    `PRESCREEN_HAND_LIMIT` entirely (every hand size screened uniformly —
    kills the n<=8/n>8 seam that let B5's residual hide). Gate fails ->
    rescan/classify surviving misses (the stage2 oracle below supplies the
    table), add a hypothesis, re-measure; endgame if classifiable structure
    runs out = keep brute force at n=8 (eat the multi-day run) + accept the
    n>8 tail bias.
  - **The gate, TWO arms, both required**: (1) root arm — score-capture-by-
    VALUE >= 95% and regret within ~1.33x the MC-reseed noise floor, on
    stage2 density, at n=8 AND the 9-12 tail (B1 knob), plus a stage3/4
    sample for copy-joker coverage; completed brute-force stage2 folds in as
    a ~4k-state FREE oracle (a stored PlayHand label IS the 218-way argmax —
    §3's free-oracle note — so capture costs k+1 evals/state). (2) full-solve
    arm (BUILD IT — currently sketched only): wrap `best_immediate_play`
    inside ~10 real solves so every node at true depth is measured (~5k
    nodes, ~25 min); same node-level bar, depth-stratified; root-action
    agreement as smoke readout only. Root-only gating REJECTED (measures the
    prescreen where it fires least — 1 of ~488 nodes, none post-discard; and
    deep misses are DIRECTIONAL: undervalued future hands tilt labels toward
    playing now, compounding the documented play-only MC bias). Passing both
    arms is also what licenses B7's validated sweep to carry over unchanged
    (its numbers were measured with brute force inside the recursion).
  - **The fix is GENERATION-ONLY — `_ranking_score` untouched** ("change the
    ranking criteria" REJECTED: `_ranking_score` is a real joker/held-aware
    `score_hand` call that already values kickers correctly when given them;
    27/27 misses had the right line — generation starved the candidate set,
    ranking judged what it was given; and touching it forces B7 revalidation
    for zero benefit). `_kicker_pad` becomes a hypothesis-gated variant
    emitter: a variant = the GREEDY argmax completion of a line under ONE
    hypothesis about where kicker value lives — (1) inert/nominal-best
    (current behavior, kept), (2) scored-value (chips + enhancement +
    EDITION + scored-channel candidacy bits), emitted ONLY when the Splash
    flag is set
    (Splash is class-3 all-zero in the trigger matrix BY DESIGN — the flag
    must come from `get_hand_eval_flags`, and without Splash kickers never
    score so the variant is pure waste), (3) held-value (retain held-channel
    matches — Baron/Shoot the Moon/Mime — and held enhancements like steel;
    pad with the cards least valuable held), (4) play-away-lowest (provably
    exact for Raised Fist's min term; the matrix's raised-fist bit marks the
    CURRENT minimum — wrong rule for the counterfactual choice). NO
    magnitudes anywhere: the key's only job is presence-in-set; the exact
    evaluator arbitrates (that is why "raw derivation" is safe). Presence
    gates read RESOLVED joker identities only (engine copy-resolution path,
    B2) — `blueprint_compat` (Splash is on the 29-incompat list) comes free;
    the emitter never inspects raw keys. Dedupe collapses variants on plain
    boards, so the budget is adaptive, not 3x flat. Accepted-and-measured
    residuals: mixed-hypothesis optima (Raised Fist + a steel 2) and class-3
    set-level jokers (Blackboard held-purity, Flower Pot under Splash) — they
    earn a hypothesis only if the rescan shows them.
  - **Family key REDEFINED to `(hand_type, line-card identity set)`** via a
    splash-agnostic hand-type scan of the played 5 (no joker scoring), NOT
    `score_hand`'s `scoring_cards`: under Splash every played card scores, so
    kicker variants of one line get distinct scoring sets = distinct families
    and would crowd `family_best` — pitfall #13 recreated by the fix on
    exactly the boards it targets. Key must keep the card SET (hand type
    alone would collapse a pair of Kings with a pair of 3s) and must not
    choke on base-less Stone cards (existing fallback path).
  - **Variants RIDE**: `top_k` counts LINES; every surviving line carries all
    its variants into the exact pass (~15 candidates vs 218 — the doc's
    budget math), which arbitrates with the full ordering search.
    Cheap-arbitration-is-final REJECTED: it silently demotes kicker choice to
    the fixed-order ranking tier (Photograph/Hanging Chad class), the same
    contract erosion that hid B5's residual.
  - **Discard side: measure-first, NO code change** (its completions are
    idealized draws — kicker variants there are hypotheses about hypothetical
    cards). Rerun `validate_discard_ranking_sweep.py` on STAGE2-CONFIG
    density (B7 validated on stage3 states — the same density lesson).
    TRIPWIRE: regret above the same noise-floor-scaled bar, or the
    directional signature (dropping discards whose value lives in
    kickers/held cards), pulls variant extension into discard completions
    forward WITH B7 revalidation costed in. Narrowing the B7 depth-gate k to
    fund variants REJECTED (re-opens the measured rank-truncation boundary —
    un-pays for B7).
  - **Data disposition**: the in-flight brute-force stage2 (~10h wall,
    validating the doc's median-not-mean health warning almost exactly) is
    KEPT — brute labels are strictly better; provenance (stage2 =
    brute-exact, other stages = prescreened) recorded at the writer. Its
    ~10% hand-size-tail examples (~400, identifiable by shard hand width >8)
    carry the LIVE n>8 kicker bug -> relabel post-fix. No other stage starts
    before the fix + gate. Amdahl honesty: the 218->~15 cut is ~5-6x overall
    (node count is discard-driven, untouched), not 17x — it hits the
    100-200 CPU-h budget only because the live run shows the mean projections
    were outlier-dominated. Fallbacks (cut stage2's 4k, re-query C1's d=4
    stratum) OFF THE TABLE pending the timing histogram harvested from the
    live run's worker logs (n=4k, free — replaces the ~100-example timing
    study). `--shard-size 25` on every relaunch (doc §7).
  - **Build order**: K1 emitter + family key + riders -> K2 harness arms
    (full-solve arm + stage2-oracle fold-in) -> K3 gate runs (n=8 + tail +
    stage3/4 copy sample) -> delete the limit -> K4 stage2 tail relabel +
    discard-side density measurement -> regen (stages 1,3,4 + C2 stage5).
    AMENDED 2026-07-17: K3's first arm found four bugs (see the K3 block),
    so K3 is now [fix -> re-run arms -> B7 revalidation] before any verdict;
    the delete-the-limit step still gates on both arms passing, and K4's
    "tail relabel" has grown into a full stage2 regen (the corpus is
    ~29% corrupt on the inert jokers alone).
  - **K1 BUILT 2026-07-17** (branch `kicker-variants-k1`), spec followed with
    TWO user-caught corrections recorded below. `hand_solver.py`:
    `_resolved_joker_views` (gates read the engine's `resolve_copy_targets`),
    `_card_channel_counts` (candidacy tallies via a new PUBLIC
    `trigger_match.trigger_predicate` accessor — the solver reads the SAME B2
    taxonomy the obs does rather than a second joker list that would rot),
    `_KickerGates` + `_kicker_variants` (4 gated hypotheses, deduped),
    `_line_family` (splash-agnostic `get_best_hand` scan), and the body
    regrouped so `top_k` counts FAMILIES with variants riding. Prefix
    stability survives at LINE granularity but is NO LONGER INDEXABLE by k
    (entry j != line j) — **both validation harnesses slice `[:k]` from one
    max-k call and must switch to per-k calls at K2**
    (`validate_prescreen.py:204`, `validate_prescreen_n8.py:187`).
    - CORRECTION 1 (editions): the spec's hypothesis-2 key omitted EDITIONS.
      They are absent from `trigger_match` by design (it is a card x JOKER
      matrix) but fire on the scored channel, so under Splash a Polychrome
      kicker is x1.5 — a Poly 2 would have ranked below a plain King, the
      exact miss class K1 exists to kill. Added as a presence bit
      (`_scored_kicker_key`); no magnitudes, so the "raw derivation is safe"
      argument is untouched.
    - CORRECTION 2 (held enhancements): a hardcoded {"Steel Card",
      "Gold Card"} set was replaced by `_has_held_enhancement`, reading the
      engine's own config (`get_chip_h_x_mult` / `get_chip_h_mult` /
      `ability["h_dollars"]` — Gold has no accessor). Correct on today's
      content either way, but the name set is the rot pattern §6 q4 warns
      about; it also gets debuff-correctness free.
    - VERIFIED against brute force on a purpose-built fixture that
      reproduces the miss (trip Kings + kicker bait; the sibling suite's
      dominant-flush hands pass pre-K1 and prove NOTHING): Splash+Lusty
      444 -> 585 (regret 141, 24.1% -> 0), Raised Fist 540 -> 660 (regret
      120, 18.2% -> 0), plain board unchanged. Both regression tests
      confirmed FAILING on pre-K1. Measured fan-out 14-23 candidates vs
      brute's 218, matching the doc's ~15 budget.
    - SUITE VERIFIED 2026-07-17 (commit `b2ec3ff`, clean tree): the 49
      targeted tests (22 kicker-variant incl. both regressions above + 27
      prescreen) plus the full `tests/scripts` + `tests/env` sweep — 821
      passed, 0 failures, ~22min — covering the generation and
      `trigger_match` suites. RUFF: the 3 errors ruff reports on
      K1-touched files (UP035 `Callable` in trigger_match; I001 + F401
      `COPY_JOKER_KEYS` in hand_solver) ALL PRE-DATE this commit —
      verified by running ruff against the PARENT (2 errors already) and
      by the diff touching none of those lines. K1 adds zero new lint;
      repo-wide 16 is likewise pre-existing. `COPY_JOKER_KEYS` IS dead
      (gates read resolved identities via `resolve_copy_targets`) but it
      is old debt, not K1's — deleting it is safe and unrelated.
    - ACCEPTED RESIDUAL (new, beyond the spec's list): SEALS are not
      consulted by the held-value hypothesis. A Blue seal is genuinely
      held-value (Planet at end of round), so a held variant can pad one
      away. Earns a term only if the K3 rescan shows it.
  - **K2 BUILT 2026-07-17** (same branch): the gate's measurement machinery —
    no gate RUN claimed (that is K3). Full record in the design doc's K2
    block. (1) Per-k fix: `validate_prescreen.py` now calls
    `prescreen_play_candidates` once PER k instead of slicing `[:k]` from a
    max-k call (K1 made output non-indexable by k; the K1 note also fingered
    `validate_prescreen_n8.py:187` — WRONG, its `_box_at_k` was always
    per-k, no change needed). (2) Root harness generalized for the K3 arms:
    brute-arm truth = explicit `_brute_argmax` enumeration (a
    `best_immediate_play` call above the limit PRESCREENS, so tail "truth"
    would be the box itself, regret always 0); `--hand-sizes`,
    `--force-tail` (B1 knob; skips shard arm — modified config can't
    reproduce shard states), `--stage-preset`, `--require-jokers`
    (Blueprint/Brainstorm hit ~5% naturally on stage3); shard arm stays
    n==8-only (n>8 labels already prescreened = not an oracle); stage2
    oracle fold-in = `--n-shard 0` on the transferred brute corpus.
    (3) Full-solve arm BUILT: `scripts/validate_prescreen_fullsolve.py` —
    `PrescreenNodeProbe` patches `best_immediate_play`/`solve_hand_turn`/
    `estimate_future_hand_distribution` as module attributes; every node of
    a real solve measured at true depth (discards_left stack, "mc" stratum
    for future-hand samples, box valued under the node's own
    search_orderings tier); wrapper returns the original's result unchanged
    (instrumented solve byte-identical — pinned by test; at n<=8 the
    original result IS the free brute truth). Root-action agreement =
    smoke readout only: re-solve with PRESCREEN_HAND_LIMIT=0 +
    PRESCREEN_TOP_K=k patched (the exact post-K3 production config, MC
    sampler included). 7 new tests in
    `tests/scripts/test_validate_prescreen_fullsolve.py`; prescreen +
    kicker suites still green (56 total); ruff clean on all touched files.
  - **K3 IN PROGRESS 2026-07-17** (branch `engine-hand-eval-flags-fix`, off
    `kicker-variants-k1`@593f57c — the K3 fixes are NOT on the K1/K2 branch).
    The gate
    RAN, and its first arm found FOUR bugs — three of them label-corrupting
    and one of them engine-wide. No gate verdict is claimed yet: every arm
    measured before `7ce81e7` is STALE and re-running. Numbers below are
    pre-fix and kept only as the trail.
    - **THE BIG ONE — `score_hand` passed `jokers=None` to `evaluate_hand`
      (fixed `03e288d`).** `evaluate_hand` derives every hand-DETECTION
      flag from that list, so **Four Fingers, Shortcut and Smeared Joker
      were COMPLETELY INERT** — in-game, in the env, and in EVERY solver
      label ever generated. Splash survived only because `score_hand`
      re-derives it independently at Phase 3c (now a documented idempotent
      restatement, pinned). Their `hand_eval` unit tests passed the whole
      time: the defect was in the integration seam — the Throwback / Idol /
      blueprint_compat / Marble / Riff-raff class, now five for five.
      Traced to `7633d34` copy-pasting the line from `09105e3`'s
      `score_hand_base`, which is jokerless BY DESIGN, so the None was
      carryover and never load-bearing. `score_hand_base`'s `joker_flags`
      arg was also accepted-and-discarded (`_ = joker_flags`); it now
      reaches `evaluate_hand(flags=...)`, because detection and effects are
      SEPARATE AXES (a Four Fingers owner's 4-card straight is a Straight
      even when priced jokerlessly).
      LESSON, generalized: a joker whose tests all pass can still be dead.
      Assert through the INTEGRATION path (`score_hand`), never the handler
      — a test that builds the flags by hand cannot fail on this bug class.
    - **Bug A — solver, `build_templates` (fixed `a60dbbf`).** Four Fingers
      REPLACED the 5-length straight windows with 4-length ones instead of
      ADDING them, so a natural 5-card straight was structurally
      unproposable (a window is an explicit rank set; the 5th rank fails a
      4-window's predicate). Both `needed` values now emitted. A 5-window
      with `needed=4` would be a NEW reachability bug — "4 of these 5
      ranks" does not imply 4 CONSECUTIVE ranks ({7,8,9,J} is not a
      straight). Pinned engine-side too (`fd0d876`): the engine was already
      correct, but nothing stopped the same mistake being ported into it.
    - **Bug C — solver flush predicate got THREE rules wrong (fixed
      `7ce81e7`).** It was a bare `_card_suit(c) == s`, re-deriving suit
      matching instead of asking the engine; `Card.is_suit(flush_calc=True)`
      owns all three and is now delegated to: SMEARED (H=D, S=C — miss
      `stage3_full_00003496`, `QC 9S 8S 3C 2C` is a Flush, regret 200),
      **WILD counts as EVERY suit** (NOT flag-gated — mislabeling boards
      since long before the jokers=None bug, 60 vs brute 284), and STONE
      never counts. Under smeared it emits TWO colour-group templates
      (`flush_black`/`flush_red`), not four: four is the same two predicates
      twice, doubling the box and crowding `family_best` = pitfall #13
      recreated by the fix. `_FakeCard` (DeckComposition probe) keeps suit
      identity — composition tracks (rank, suit) only, so a wild in the
      DECK is unrepresentable either way; the fallback undercounts wild
      draws, the safe direction.
    - **Hypothesis 5 — type-upgrade kicker pad (added `7ce81e7`).** The
      first hypothesis that asks what the pad makes the HAND, not what it
      is worth as a CARD. Miss `stage3_full_00001545` (Four Fingers):
      padding the 4-card heart flush `QH 7H 5H 4H` with the 6 of CLUBS is a
      STRAIGHT FLUSH (FF wants 4 hearts for the flush and 4 of 7-6-5-4 for
      the straight, and vanilla lets those be different cards) — regret
      892, 73.4%, the arm's largest. The winning set is the UNION of two
      templates' cards, so no single predicate proposes it and no per-card
      key ranks the club 6 above the King. UNGATED (trips + pair -> Full
      House is the same shape); dedupe collapses it when it changes
      nothing. `_kicker_variants` takes `flags` as a REQUIRED positional —
      an empty default would let a call site silently claim "no FF /
      Shortcut / Smeared".
    - **What arm C actually showed** (stage3 copy-jokers, PRE-Bug-C-fix):
      brute capture DROPPED 1.000 -> 0.960 after the engine fix. That is
      the HONEST direction — fixing the engine CREATES better truth lines,
      so the generator's own gaps surface. Expect every re-run arm to be
      harder than its pre-fix number; a capture that IMPROVES after an
      engine fix deserves suspicion, not relief.
    - **Method notes worth keeping**: the per-miss board dump (`0cad469`,
      `--dump-miss-k`) is what converted "27/27 misses have true_size=5"
      into four named bugs — aggregate capture/regret located nothing.
      Every new test was checked against the PARENT commit; two drafts of
      the wild-card fixture PASSED on broken code because `_keep_priority`
      is `(enhanced, nominal)` and pads an enhanced wild in for the wrong
      reason (only a Glass Ace outranking it on that key discriminates).
      A green test on a known-broken solver is worthless.
    - **Consequences / still open**:
      1. Every K3 arm must be re-run against `7ce81e7` (in flight).
      2. **B7 revalidation is MANDATORY** — Bug A and Bug C both change
         `build_templates`, which feeds the discard side
         (`rank_templates_cheaply`, `solve_hand_turn`), so the locked
         depth-gate sweep (25 helped / 9 regressed at k=4) was measured on
         templates that could not propose 5-card straights, smeared flushes
         or wild flushes. Cost note: the harness is ~25 min on a
         DISAGREEING state and ~0.05s on an agreeing one, so 200 states is
         a 9600X job, not a local one.
      3. **The in-flight stage2 brute corpus is CORRUPT and must not be
         kept**: ~29% of it owns j_four_fingers / j_shortcut / j_smeared
         (measured via shard `joker_ids`), and those labels were produced
         with the jokers inert. Selective relabel is sound — the engine fix
         is provably a no-op without those three (flags all-false == the
         old `jokers=None` path) — but Bug C's wild-card half is NOT so
         gated, so a wild-card flush board is mislabeled with no joker
         present at all. Wholesale regen is the safe call.
      4. The obs features (`hand_potential_features`) take four_fingers /
         shortcut but NOT smeared — an informativeness gap on the same
         axis, not label-corrupting. Unfixed.
  - **K3 CLOSED 2026-07-17 — GATE PASSED, LIMIT DELETED** (`cb9eeb0`).
    Two MORE bugs fell after the four above (six total, five of which
    silently corrupted labels), then both arms cleared the >=0.95
    capture-by-value bar:
    | arm | capture |
    |---|---|
    | root, n=8 stage2 brute | 0.980 (k-invariant) |
    | root, stage3 copy-jokers | 0.980 |
    | root, 9-12 tail | 0.950 |
    | full-solve, node-level at true depth | 0.9808 (d0 .981 / d1 .997 / d2 1.0 / d3 1.0) |
    `PRESCREEN_HAND_LIMIT` is GONE: one screened path at every hand size,
    labels uniformly "exact among prescreened candidates". The n<=8 seam is
    exactly where B5's residual hid — brute-forcing there meant the box was
    never MEASURED there.
    - **Bug D — the HARNESS screened with `smeared=False`** (`a4a0053`).
      Both validation harnesses passed four_fingers/shortcut but not
      smeared (default False), so Smeared boards were SCREENED with raw-suit
      templates and SCORED against a smeared engine — the box built for a
      different board than it was measured on. This is why arm C still
      reported the smeared miss at an unchanged regret of 200 AFTER the Bug
      C fix while a unit test on the same board passed. **A contradiction
      between two measurements is a bug in one of them — chase it.**
      `prescreen_play_candidates` took the detection flags TWICE (booleans
      -> templates, `eval_flags` -> kicker gates) with nothing forcing
      agreement; it now RAISES on contradiction.
    - **Bug E — the cheap rank was an emission-order LOTTERY**
      (`caf3394`). `_ranking_score` is fixed-order by design, and dedupe
      was by card-identity SET, first-wins — so a family's rank was decided
      by `itertools.permutations` order. Measured
      (stage2_curated_00002797, Photograph + Hanging Chad): the full house
      was first emitted threes-first -> 1408, the kings-first emission of
      the SAME FIVE CARDS -> 4080 was skipped as a duplicate, so at 1408 it
      ranked ~7th, was cut at top_k=5, and the exact pass never saw a FULL
      HOUSE at all — it lost to a 3008 straight. Fix: MAX over emitted
      orders (the generator already emits both; the good one was free
      information). A canonical sort was REJECTED (user call): any fixed
      order is a guess. GATED on `_needs_permutation_search` — not an
      optimization but the difference between viable and not (`raw` is
      dense with duplicates; ungated it cost 7x at n=8, 16.9 -> 117.4
      ms/state, which would leave the prescreen barely beating the brute it
      replaces — and n=8 takes this path now).
    - **Seating fix (`a02e47b`)**: Shortcut WIDENS a straight window and
      Four Fingers adds shorter ones, so a hold OVERFLOWS and five seats
      must be chosen — by `sorted(hold, key=_keep_priority)[:5]`, i.e.
      joker-blind (measured: hold `7S 6S 5S 4S 3C 2D` seated `7-6-5-4-3`
      when Wee Joker wanted the 2). K1's hypotheses cannot reach it: they
      choose KICKERS and only run when the line is SHORT of 5. Fix: emit
      the seatings, let `_ranking_score` rank them. NO hypothesis, NO type
      filter, NO family dedupe — all three were tried and all three discard
      a candidate on a PROXY for value before the scorer sees it, which is
      the bug itself. This killed the hand-size gradient: n=12 went 0.896
      -> 0.958 -> 0.979 (vs n=8's 0.980).
    - **THE SILENT TRAP, and it generalizes** (`cb9eeb0`): the full-solve
      arm reused `best_immediate_play`'s own result as truth (the
      "free oracle"), correct ONLY while n<=8 brute-forced. With the limit
      gone that call PRESCREENS, so its result IS the box under test —
      reusing it scores the box against itself and reports **regret
      identically 0 at every n**, i.e. the arm silently becomes a no-op
      that always passes. Truth is now always an explicit C(n,1..5)
      enumeration. K2 flagged this exact hazard for the tail arm; it
      applies to EVERY harness the moment its oracle and its subject become
      the same code. **When deleting a fast path, re-check every measurement
      that was licensed by it.**
    - **THE TIE ASSUMPTION**: `_brute_argmax` vs `best_immediate_play` used
      to pin SUBSET identity because both sides were the same enumeration.
      They are now different searches that TIE — the fixture is High Card,
      where only the top card scores, so `[A]` and `[A,x,x,x,x]` are worth
      exactly the same; brute enumerates sizes ascending and takes the
      1-card play, the box returns a 5-card one. **The argmax is NOT
      unique, which is why capture is measured BY VALUE everywhere.** Any
      new comparison of two searches must compare values, never sets.
    - **ACCEPTED RESIDUALS** (user call): ~5% of states, mean regret ~22
      chips on the tail, ONE coherent family — class-3 SET-LEVEL jokers
      (Jolly/Droll/Blackboard/Square), where no honest per-card bit exists
      by taxonomy design, plus Four Fingers. The tail's 0.950 is carried by
      n=9/n=12 (n=10 0.939, n=11 0.935 individually below). The full-solve
      `mc` stratum sits at 0.925 (160 nodes, mean regret 44 — future-hand
      samples at the coarse `search_orderings=False` tier). Rationale: all
      PPO-correctable in the documented sense (real reward is P(clear), a
      mis-ranked play is a legal single-step action — the A3 "training
      problem, not labels" precedent). CAVEATS RECORDED, not waved: "PPO
      fixes it" is a claim about the END of fine-tuning (BC teaches the
      prior, the KL leash holds the policy near it early, so a line BC never
      proposes may go unsampled until the leash decays); and the `mc`
      stratum feeds p_clear VALUES — the critic's warm start — not just
      actions.
    - **B7 REVALIDATED post-fix** and its verdict SURVIVES: 200 states,
      k=4, n_samples=80 — 32 helped / 13 regressed (was 28/11),
      frac_helped 0.333 (was 0.308), worst_paired_diff -0.20 (unchanged),
      disagreement 0.495 (was 0.470 — expected, the templates changed).
      8 of the 11 original regressors still regress INCLUDING seed 247
      (the Shoot-the-Moon keep-both-Queens case), so the depth gate's
      rationale carries over intact. NOT re-run: the k=4/6/8/12/64 SWEEP
      that localized regressions to the rank-k boundary — deferred as
      low-risk on the qualitative match.
    - The DISCARD side has the identical seat-blindness (`eval_cards =
      hold[:5]`, not even keep-priority) — deliberately UNFIXED per the K
      spec's measure-first rule (its completions are idealized draws, so a
      seating hypothesis there is a hypothesis about hypothetical cards),
      documented at the site with the sweep as its tripwire.

### h1 architecture — Candidate B COMMITTED (autoregressive pointer head)

- Hand agent ONLY: the shop's flat Discrete(686) head survives s1 untouched
  (`--init-from` requires it); B subsumes the shop's SelectTarget combo block at the
  in-blind merge, not before.
- Why committed without the evidence gate: the gate is CIRCULAR — its counter is
  conditioned on current policy behavior, so if s0 rarely buys +hand-size jokers
  BECAUSE h0.5 can't exploit them, the histogram reads low without proving A@12
  suffices — plus a strong prior that >12-card hands are common in +hand-size builds
  (one Turtle Bean = 13 immediately, for multiple turns). The shop_gym instrumentation
  counter is NOT built at all: the hand-size histogram falls out of the harvest corpus
  for free, and it only tunes B's max decode length, which isn't needed until after s0
  anyway.
- Change-compounding mitigation (B lands at the same seam as the bump + new BC pool +
  new reward term): validate B BC-ONLY first — sequence CE on the same pool
  (canonicalize set -> sorted picks), compare val metrics against a flat-head control
  before any PPO. Known costs accepted: custom compound distribution (exits vanilla
  MaskablePPO), `KLToBCMaskablePPO` rewrite (KL leash becomes a teacher-forced per-step
  sum), sequence-CE BC, and the flat-head shop-merge plan unwinds (B was already its
  admitted end-state).

### h1 objective & training

- **Terminal $ term** (fills the marked hook in `HandPlayGymEnv`): on CLEAR only,
  reward = 1 + `V_curve(ante, dollars_after_cashout)`; a loss pays nothing (run over,
  money worthless — "clearing dominates" preserved). `V_curve(ante, $)` is extracted
  OFFLINE once: counterfactual money-sweeps of the s0 critic over harvested shop states
  (edit dollars in the obs, forward the critic), averaged per (ante, $) cell —
  per-dollar sweeping captures interest-threshold nonlinearity for free, and the values
  are already in P(win) units because s0's reward is 1{run won} (no scale
  hyperparameter). NO decay (deliberate objective change, not shaping — as already
  documented at the hook). Start-of-episode dollars are a per-episode constant, so the
  absolute lookup doesn't shift the optimum. TWO RIDERS (user-added): (a) the
  ante-average ERASES build-specific money valuations (a To the Moon or Bull build
  values held dollars very differently) — note it in code; the contextual-critic query
  is the named upgrade path if h1's money play looks wrong. (b) The cashout math must
  mirror engine interest ordering — in-blind money earnings (Business Card, Rough Gem,
  gold cards) land BEFORE interest and can cross thresholds; end-of-round payout jokers
  (Golden Joker, Rocket) pay AFTER interest and must not affect it (same rule class as
  the verified "Investment pays after interest") — VERIFY the engine orders these
  correctly before mirroring it.
- **PPO starts from a mixed sampler**: `start_state_sampler` hook on `HandPlayGymEnv`
  mirroring the shop env's (`() -> snapshot | None`, None = config-sample); mixture =
  stage1-4 configs + harvested snapshots, config anchor always nonzero (same
  anchor/coverage-bias logic as the shop reservoir — reward stays honest, so coverage
  bias is the only risk). Reuses the existing RNG-exact engine pickle round-trip. Also
  the path that exercises B's big-hand decoding under real training, not just BC —
  without it, the KL leash decays and late PPO drifts back toward config-distribution
  play, evaporating half the induced-distribution fix.
- **Discard-bias fingerprint GATES the solver fix**: run the documented
  hands_left x discards_left bucketed eval on h0.5 NOW (cheap, this machine); only if
  the play-only label bias survived PPO does `estimate_future_hand_distribution` learn
  to credit banked discards (every solver change risks a discard-cap-class label bug,
  so don't touch it if PPO already corrects). The flush/straight ARCHETYPE
  decomposition runs in the same eval pass — it no longer gates the fix (locked into
  the bump) but calibrates how much recovery to expect from it.

### s1

- **SkipBlind exposed**: ONE action appended at canonical index 686 (append-only
  contract; fresh cold head row, s0 loads verbatim). Blind-select stops being
  auto-resolved on non-boss blinds (SelectBlind + SkipBlind both unmask; boss =
  SelectBlind only, vanilla-consistent). The offered TAG's identity must become
  observable (shop_context / entity row) — skipping without seeing the tag is
  uninformed. One decision point per blind, no chain — the exploration-trap argument
  doesn't apply. The original deferral rationale is fully expired (tags wired and
  in-game verified; the partner is h1, not noisy h0).
- **Joker rows -> 15 at s1 kickoff** (DECIDED 2026-07-16; full record at the
  `MAX_JOKER_ROWS` open item): obs joker block widens 8 -> 15 (weight-preserving —
  masked-pool trunk is row-count-agnostic; needs only an `--init-from` load shim plus
  the byte-identical-outputs check on <=8-joker states) and SellJoker slots 8-14 append
  as seven cold head rows at [687,694), right after SkipBlind — same append-only
  mechanism. 15 matches `MAX_JOKERS_V2`. The positional invariant "SellJoker slot k
  targets obs joker row k" now spans the split blocks — pin the k -> action-index
  mapping in tests.
- **Reservoir persistence** (BUILD BEFORE s0 KICKOFF — the one pre-s0 code change):
  DONE 2026-07-07 (branch `worktree-shop-reservoir-persistence`). Root cause confirmed:
  `train_shop_ppo.py` built the reservoir fresh per invocation and only the model saved,
  so the a2->a4->a8 chain discarded snapshots between stages and s1 would start empty —
  the "current AND past checkpoints" diversity rule didn't survive invocations. Fix:
  `ShopReservoir.save()`/`.load()` pickle strata + config + RNG bit-generator state (so
  a resumed run's sampling stream continues, not restarts); `main()` saves to
  `{log_dir}/reservoir.pkl` at final-save AND on the checkpoint cadence
  (`ReservoirCheckpointCallback`, so a killed run resumes with a matching reservoir, not
  an empty one); `--init-reservoir <path>` loads a prior stage's reservoir (omit = fresh,
  seeded once at a2). Round-trip tests in
  `tests/scripts/test_train_shop_ppo.py::TestReservoir` (strata/config/RNG-stream
  identity + post-reload capacity enforcement); end-to-end verified a2->a4: stage a4
  loaded the a2 reservoir and grew it (diversity carried across invocations). Wire it
  into the a4/a8 stages via `--init-reservoir` (see the s0 kickoff command below).
- Phi = s0-critic potential-based shaping replaces the crude `c_ante` blind bonus
  (state-only potential, decays — pure training-script change), and the nextround
  floor re-baselines against s1's actual partner (h1). Both as already documented.

## In-blind merge — shop targeting → pointer + Death direction — GRILLED AND LOCKED (2026-07-16)

Decision record for retiring the shop's combinatorial `SelectTarget` block in favor
of Candidate B's autoregressive pointer, plus how tarot targeting (esp. Death) is
resolved. Ties into "h1 architecture — Candidate B COMMITTED" and the action-space
ceiling open item.

### Grounding findings (engine-verified this session)

- **Targeting is ORDER-INSENSITIVE at the engine for every consumable.** Death copies
  the **rightmost-by-`sort_id`** highlighted card onto the other (`consumables.py:431`
  `_death`: `rightmost = max(highlighted, key=lambda c: c.sort_id)`); every other
  targeted consumable iterates `ctx.highlighted` symmetrically or is single-target.
  No handler reads selection ORDER. => Candidate B's **monotone-ascending pointer mask**
  (each pick index > last) works VERBATIM for targeting; a target set is just an
  ascending index set. No per-tarot order convention needed. REJECTED: order-carrying
  pointer / Death-specific reorder action for *selection order* — the engine ignores
  order, so those add null-direction entropy (the exact thing B's monotone mask
  rejects). Vanilla-only claim; pin with a test in case a future consumable is
  genuinely order-directed.
- **Death direction is NOT an agent choice in the current engine, and a reorder action
  can't create one.** `sort_id` is an immutable creation counter (`card.py:26`).
  `SwapHand` (`game.py:1412`) reorders the hand LIST but not `sort_id`, and Death reads
  `sort_id` — so swapping doesn't move Death's direction. The vanilla constraint the
  user identified is real ("you can only duplicate a card leftward; you cannot duplicate
  the leftmost card"), but the choice lives two layers down (engine `sort_id` semantics,
  possibly a faithfulness gap vs `card.lua:1111` — the author's own comment calls
  `sort_id` a "position proxy"), NOT in the action-space head.

### Death direction — env-side auto-direction via DIRECT CONSTRUCTION (option ii)

Agent picks the 2-card SET via the monotone pointer (unchanged); the ENV computes which
card is the **survivor** (template, duplicated) vs the **override** (overwritten) and
hands that direction to the handler directly (small `death_source` hint on
`ConsumableContext`, or build `copy_card` in the caller) — **bypassing the `sort_id`
rule entirely**. No `SwapHand`, no dependency on the `sort_id`-vs-vanilla faithfulness
question. Mirrors the `best_play_order` / `best_joker_order` precedent: ordering that
matters but is engine-computed so the agent never spends an action on it (direction is
ordering-within-a-chosen-set, the play-ordering analog — NOT the target-SELECTION
analog that CLAUDE.md says must be learned). REJECTED alt (i): make `_death` read hand
list-index + internal `SwapHand` — works but changes engine semantics and leans on the
swap machinery; only preferred if the engine must stay the source of truth for Death
direction (e.g. a future human-play mode), which we don't need.

**Survivor heuristic — 16-rung lexicographic ladder** (higher rung wins the moment
exactly one card has that attribute; the winner is the survivor/duplicated):
1 Red seal · 2 Purple seal · 3 Blue seal · 4 Gold seal · 5 Gold Card enhancement
(`m_gold`) · 6 Glass · 7 Polychrome · 8 Steel · 9 Wild · 10 Lucky · 11 Holographic ·
12 Foil · 13 Mult · 14 Bonus · 15 Rank · 16 Stone.
- Rank rung uses a **bespoke `K>Q>J>A>10>9>…>2` key — NOT `get_nominal()`** (which is
  Ace-high; easy trap since it's right there). Gold appears twice on purpose (seal @4,
  `m_gold` enhancement @5). Negative dropped (not reachable on playing cards). Stone has
  no rank so it loses rung 15 to any ranked card; rung 16 is only a two-Stones tiebreak.
  Wild/Lucky sit ABOVE the Holo/Foil editions. Lexicographic => a single high rung
  outranks any stack below it (a Red-seal deuce beats a Glass+Poly King) — intended.
- **Known limitation**: the ladder is **build-blind** ("best card" is build-dependent —
  flush build wants Wild, mult build wants Glass/Steel). Accepted for v1.

### Pointer replaces combinatorial — WHEN, and why not earlier

Candidate B (hand pointer, committed at the h1 seam) subsumes the shop's `SelectTarget`
combo block **at the in-blind merge**. What transfers at the merge: the **trunk**
(card/joker encoders + embeddings) and the **pointer machinery** (autoregressive decode,
per-step masking, sequence-CE, KL leash). What does NOT transfer: the **tarot-targeting
POLICY** — a play-selector is not a Death-targeter; that behavior is learned at the merge.
Crucially, that learning is unavoidable with ANY head (the flat combinatorial block is
equally cold on targeting), so the pointer does not ADD retrain cost — it REDUCES it
(warm card encoder, no C(n,k) blowup, one mechanism instead of two). "Reuse of the same
family" = same action family + mechanism, NOT a transferred skill.

**CORRECTION (this session): targeting is LIVE in s0/s1 via packs — NOT inert.** Opening
an Arcana/Spectral pack in the shop DEALS a hand from the deck for targeting
(`game.py:327-342`, `gs["hand"] = combined_hand`), so `PickPackCard` on a pack tarot
enters the pending state and the flat `SelectTarget` block IS exercised in the s-track.
(Only `UseConsumable`-from-owned targeting is inert in s0 — empty shop hand + engine
forbids `UseConsumable` during `PACK_OPENING`.) Consequence: the flat block carries its
two weaknesses onto the pack path — the 8-position ceiling and C(n,k) growth — so
**pack-tarot targeting for >8-card hand-size builds is undervalued in s0/s1** (real,
bounded, second-order, same class as the hand-play ceiling; self-corrects at the merge).
Bounded because s0 reaches early antes where the dealt pack hand is ~8 cards, fully
coverable by the flat block (C(8,3)=56); the ceiling only bites for hand-size builds,
rare that early.

The pointer **still can't come earlier — for sequencing reasons, not inertness** (this
corrects an earlier "defer because inert" rationale): s0 already committed to the flat
`Discrete(686)` head (`s0_a4_v4` exists), the pointer doesn't exist until h1 (which runs
AFTER s0), and s1 must `--init-from` s0's flat head. So s0/s1 are stuck with flat
combinatorial pack-targeting regardless; the merge is the first point the pointer CAN
replace it.

### Timeline

- **s0** (done / in progress): flat combinatorial pack-targeting, accepted limitations
  above.
- **h1**: Candidate B pointer built + validated BC-only against a flat-head control
  (hand side; per the Candidate B section).
- **s1**: flat head RETAINED (`--init-from` from s0); pack-targeting stays flat.
- **In-blind merge**: pointer replaces the shop's `SelectTarget` block; Death
  direct-construction + heuristic land here (targeting also goes live for owned
  consumables); the tarot-targeting policy is learned here.
- The Death engine piece (direct-construction hook + heuristic) is engine-side and
  touches neither the head nor the obs, so it can be BUILT/tested in isolation any time;
  it only FIRES once targeting is live, i.e. functionally at the merge.

### Escape hatches

- **Learned Death direction (v2)** — if the fixed ladder underperforms: replace it with
  a LEARNED survivor pick via the autoregressive pointer, **never combinatorial exposure**
  (C(n,k) explodes at large hand sizes — the same reason we retire `SelectTarget`).
  Because option (ii) already has the env honoring whatever direction it's handed, v2 is
  ADDITIVE — no rework. This is the one place "pointer carries order" becomes REAL
  (non-null) entropy, because the survivor choice actually changes the outcome (unlike
  selection order, which the engine discards).
- **Earlier pointer (s1 instead of merge)** — if pack targeting proves to matter more
  than the 8-card regime suggests: do the pointer surgery at s1 (load s0's trunk, swap
  the target block for a cold pointer, fine-tune). Front-loads the pointer's complexity
  ahead of the hand-side BC validation meant to de-risk it, so **emergency-only**.

## Open items / not yet implemented

- [x] Speed fix for `hand_solver.py`: `_needs_permutation_search` already skips the search
      entirely when no order-sensitive effect is present (tested in
      `tests/scripts/test_hand_solver_order_sensitivity.py`). Found and fixed a correctness bug
      while grilling this: for 5-card hands still needing the search, the old
      `all_perms[:MAX_PERMUTATIONS]` slice (`MAX_PERMUTATIONS = 24 == 4!`) was, due to how
      `itertools.permutations` orders its output, every permutation sharing the *original* first
      card and nothing else — zero exploration of alternative first-scored cards, exactly the
      property Photograph/Hanging Chad depend on. Replaced with
      `_first_last_covering_permutations` (20 orderings, exact for a single order-sensitive
      contributor) plus `_count_order_sensitive_sources` to detect when 2+ contributors require
      falling back to full 120-permutation enumeration. Tested in
      `tests/scripts/test_hand_solver_permutation_coverage.py` against brute-force ground truth.
- [x] Custom `GameAdapter` for isolated hand-turn episode injection — `HandPlayAdapter`
      (`jackdaw/env/hand_play_adapter.py`), tested in `tests/env/test_hand_play_adapter.py`.
      Plugs into `BalatroEnvironment` unchanged via `adapter_factory=lambda: HandPlayAdapter(cfg)`
      (already generic over `GameAdapter`), but no training script wires it in yet. Boss-blind
      selection is ante-correct if `blind_stages` is widened to include `"Boss"`, but the default
      curriculum config still excludes it deliberately (out of scope until a later stage). Money
      (`dollars_range`) is sampled flat/uniform regardless of ante — placeholder until the
      shop-agent's marginal-value-of-$1 curve exists. Injected jokers are always base/no-edition,
      no-sticker — deferred until curriculum targets full shop-purchase-realistic coverage.
- [x] Offline demonstration-generation pipeline — `scripts/generate_hand_demos.py`, tested in
      `tests/scripts/test_generate_hand_demos.py`. Samples via `HandPlayAdapter`, labels via
      `solve_hand_for_ante_clear` (the P(clear)-not-EV oracle — measured ~12s/example, so
      designed for multiprocessing from the start: workers write their own `.npz` shards
      independently, seeded `f"{stage_name}_{global_index:08d}"` partitioned into fixed
      per-worker ranges so the dataset is reproducible regardless of `--num-workers`). Run once
      per curriculum stage (pass a `HandPlayConfig` + `--stage-name`) into
      `{output_dir}/{stage_name}/`, not one mixed pool. Card-selection labels are a multi-hot
      mask over the padded hand width (matches `ActionMask.card_mask`), not raw index tuples.
      A solver exception on one sampled state is logged to `worker_N_failures.jsonl` and
      skipped, not fatal to the run. Found and fixed a second pre-existing bug while building
      this: `best_immediate_play`'s `held = [c for c in hand if c not in combo]` used
      value-equality, so an Erratic deck's duplicate-valued cards would both get excluded from
      `held` when only one was actually selected, undercounting held-card-count-based joker
      scoring — fixed to filter by `id()`, tested in
      `tests/scripts/test_hand_solver_duplicate_cards.py`.
      Later hardening (post-discard-cap): the future-hand MC estimator is seeded
      per-example (`mc_seed` threads into `estimate_future_hand_distribution`, and the
      `prob_clear_given_future` LCG is reset per solve — labels are now byte-reproducible
      across machines, verified against a cross-machine overlap), uses 16 samples (was 40)
      with `search_orderings=False` on hypothetical future hands (measured 1.8-11x
      speedup; the exact current-hand path keeps the full ordering search — prescreen
      lever "C" explicitly rejected). Known documented label bias: future hands are
      valued play-only, so banked discards are credited at zero -> labels tilt toward
      spending discards early; PPO-against-the-real-game is the intended corrector, and
      eval bucketing by (hands_left, discards_left) is the diagnostic fingerprint.
      OPERATIONAL LESSON (2026-07-05 incident): resume partitioning is fixed by
      (total_examples, num_workers) — ALWAYS pass `--num-workers` explicitly. Copying a
      partial run to a machine with a different CPU count silently re-partitions on
      resume (the default is cpu_count-1): produced 925 duplicated + 321 never-generated
      seeds in stage 2. Dedup/verify with a unique-seed count before consuming any
      transferred dataset.
- [x] Curriculum stage presets + realistic state injection (grilled and decided; supersedes
      "pool sizes/counts not yet decided"):
      - Named presets in `generate_hand_demos.py` (`--stage stage1_no_jokers|stage2_curated|
        stage3_full` = frozen config + example count: 2k / 4k / 20k; flags override). Stage 2
        is a 21-joker archetype-spanning pool (jokers that *change the optimal action*, incl.
        The Ancient); stage 3 is all 150 (legendaries and economy jokers deliberately included
        — "irrelevant to hand-play" is itself a label worth learning).
      - `JokerCountBand` in `HandPlayConfig`: count-first weighted sampling (30% five-joker /
        20/20/10/10/10 down to zero), 0-1-joker states confined to antes 1-2, 5-joker to ante
        3+. Count marginals are honored exactly; the ante marginal absorbs the tilt (deliberate).
        Raise-don't-clamp when a band exceeds pool/joker_slots.
      - Scaling-joker accumulated-state randomization (`_SCALING_SPECS`, 25 entries): without
        it, Ride the Bus/Obelisk/etc. always appear at zero accumulation and BC learns "dead
        slot" — a systematically false prior. cap = trigger_opportunities(ante) x difficulty
        fraction (0.70 common / 0.60 uncommon / 0.50 rare; Yorick 0.70, Caino 0.50) x
        per-trigger gain; sampled uniform-quantized in [0, cap]. Decay jokers (Ice Cream/
        Popcorn/Ramen/Seltzer) sample only alive states; Campfire flat [x1.0, x2.5] (boss
        reset); Loyalty Card gets a uniform charge position. Run-stat priors (skips, tarot
        usage) seeded for formula-based jokers (Throwback, Fortune Teller). Known deferred
        gap: hand LEVELS (Planet upgrades) and per-hand-type usage counts (Supernova) are
        still always run-start values.
      - Engine bug found & fixed while wiring this: `score_hand`'s `GameSnapshot` never
        received `skips`, so Throwback could never fire through the real scoring pipeline
        (its handler unit tests passed — the gap was integration). One-line fix in
        `scoring.py`, integration-tested in `tests/engine/test_scoring.py`.
- [x] BC + PPO fine-tune training loop for hand-agent (built and tested end-to-end; grilled
      design decisions below supersede any earlier sketch):
      - **Canonical `Discrete(436)` action space** (`jackdaw/agents/hand_action_space.py`):
        {play, discard} x all 1-5-card subsets of 8 hand positions, size-lexicographic order,
        APPEND-ONLY forever (BC labels and checkpoint action-head rows depend on index
        stability; the full-run wrapper's per-step resampled Discrete(500) table is unusable
        for BC). Shop-merge action families later append at 436+ with fresh head rows.
      - **Env-side optimal ordering, no reorder actions**: the agent picks a card *subset*;
        `HandPlayGymEnv` plays it in engine-optimal order via
        `jackdaw/engine/play_ordering.py::best_play_order` when an order-sensitive joker/card
        is present (helpers moved there from `hand_solver.py`, re-imported under old names;
        ~7ms worst case, only on order-sensitive boards). Decided over swap/sort actions:
        with the KL-to-BC leash active, PPO discovering multi-step reorder chains under
        sparse reward is exactly the exploration-collapse failure mode, spent on something
        the engine can compute exactly.
      - **`HandPlayGymEnv`** (`jackdaw/env/hand_play_gym.py`): Dict obs == BC demo-shard
        schema exactly, PLUS an always-masked consumable block (2x7) — with masked pooling
        an absent entity type contributes exactly nothing (no false-zero signal), and the
        obs space/checkpoint format survives the shop merge unchanged. Reward: terminal 1/0
        = P(clear), gamma=1.0, NO shaping; docstring marks the h1-stage hook where the shop
        critic's marginal-value-of-$1 adds a terminal `f(hands_left, dollars)` term (unused
        hands aren't free forever — deliberate objective change at h1, not shaping).
      - **Shared net** (`jackdaw/agents/hand_policy.py`): pooled per-entity-type MLP encoders
        (no attention at 8+5+2 entities), whole trunk in a custom SB3 features extractor,
        `net_arch=[]` so BC->PPO transfer is a plain per-module `load_state_dict`
        (`load_bc_weights_into_policy`); tested to reproduce identical masked distributions
        AND values (`tests/agents/test_hand_policy.py`) — the value head regresses solver
        `p_clear`, which with 1/0 reward + gamma=1 IS the PPO critic target (calibrated warm
        start).
      - **`scripts/train_bc.py`**: pools stage dirs (BC is supervised; sequential stages just
        forget), CRC32-of-seed val split, masked CE with smoothing 0.05 confined to the legal
        set, +0.5*MSE on p_clear, early-stop patience 2 / max 10 epochs / best-epoch
        checkpoint, per-epoch val entropy stored in checkpoint metadata (diagnose
        over-sharpened BC after a bad PPO run). Loader hard-fails on schema drift or labels
        illegal under their own reconstructed masks.
      - **`scripts/train_hand_ppo.py`**: `KLToBCMaskablePPO` — train() copied from sb3-contrib
        2.7.1 (version-pinned by a test; re-diff before bumping) + reverse KL(pi||pi_BC)
        against the frozen BC net, `beta_eff = beta0 * progress_remaining * m`, m adaptive
        x/1.5 toward KL target, so the leash provably reaches zero. ALL hyperparameters are
        provisional — retune from checkpointed output (tensorboard + eval callback).
      - **`scripts/eval_hand_policy.py`**: fixed suite on reserved `EVAL_` seed prefix (never
        train on it), deterministic clear rate, `--solver-ceiling` caches mean solver p_clear
        over the same seeds as the exact-play reference. NOTE: for a *ballpark* per-stage
        ceiling, don't pay `--solver-ceiling` (~12s/seed; 30min+ for joker stages under core
        contention) — take the mean of the `p_clear` labels already stored in that stage's
        demo shards (same solver output, thousands of samples, instant; validated 2026-07-06,
        stage3 label-mean 0.307 vs 50-eval-seed 0.295 agree). `--solver-ceiling` is only worth
        it for a paired comparison on the exact EVAL_ seeds.
- [x] Solver 5-card discard cap (found when BC first consumed real stage-1 data: 8.3% of
      labels were 6-8-card discards — unexecutable, the engine caps a discard at 5, and
      reachability had assumed 6-8 replacement draws). `cap_discard` in `hand_solver.py`
      splits template non-matches into (<=5 discarded, kept-in-hand — enhanced/high-nominal
      kept preferentially); `hold` stays matches-only for still_needed/eval math, `kept`
      rides along for hand reconstruction. Tested in
      `tests/scripts/test_hand_solver_discard_cap.py` incl. the original failing seed.
      Pre-fix datasets backed up to `data/hand_agent_demos_pre_discard_cap/`; stages 1-2
      regenerated with the fix. Hardening added so this class of bug (solver models an
      action the engine can't execute) dies at the source next time:
      `validate_label_executability` in `generate_hand_demos.py` — tier 1 checks the label
      against the canonical Discrete(436) mask, tier 2 executes it through the real engine
      (destructive, runs after obs encoding on the throwaway state; ~0.5ms vs 2-12s/solve).
      Failures land in `worker_N_failures.jsonl` per the existing skip-not-fatal design.
- [x] h0 first real BC run + eval (2026-07-06). `scripts/train_bc.py` on all four stages
      pooled (stage1+2+3+4 = 33,977 examples, 30,668 train / 3,309 val), max-epochs 25 /
      patience 2 -> early-stopped at epoch 20, best epoch 18 (val_ce 2.119, val_acc 0.563,
      val_entropy 2.322, val p_clear MSE 0.057). Checkpoint `runs/bc/h0_s1234_25ep/` (runs/ is
      gitignored — lives only in the main checkout). CPU-only torch; dataset load over ~3.6k
      shards is the slow part (~3-4 min). Eval vs solver ceiling (ceiling = mean p_clear from
      demo labels): stage1 3.7%/0.141 (~26% recovery), stage2 17.3%/0.390 (~44%), stage3
      16%/0.307 (~52%). Median p_clear is 0.0 in EVERY stage — most domain-randomized states
      are unwinnable by design, so absolute clear-rate is dominated by the winnable fraction
      and the ceiling ratio is the real signal. Recovery is a sane pre-PPO BC baseline (weakest
      on barren no-joker stage1, best on joker stage3), entropy 2.32 preserved for PPO. stage4
      policy eval was blocked on the hand-obs-width limitation; after that fix (see the
      MAX_HAND_CARDS_OBS item below) it ran clean over all 300 eval seeds incl. the
      previously-crashing EVAL_00000236: 8.7%/0.209 (~41% recovery) — in line with the other
      joker stages, no boss-specific collapse (stage4 demos were already in the BC pool).
      - Engine bug found + fixed during this eval (branch `worktree-riffraff-room-check`):
        Riff-raff created Common Jokers at blind start with NO room check
        (`game.py::_apply_setting_blind_mutations`, the `ctype=="Joker"` branch), unlike the
        other create path (`create_jokers`, ~L363). A `GameSnapshot` joker_count taken once up
        front went stale, so the handler over-returned count=2 and the applier produced 6
        jokers in a 5-slot game — corrupts real scoring AND crashes the fixed-width obs
        encoders (MAX_JOKERS=5), which blocked stage3/4 eval. Fixed by capping creation at the
        actual `len(jokers)` at apply time (true vanilla "if you have room" semantics).
        Regression tests in `tests/engine/test_jokers_integration.py::TestRiffRaff`; 210 engine
        tests pass.
- [x] Shop-agent BUILD (design grilled and locked — full decision record in "Shop-agent
      design" section above; follow its build order 1-7). DONE: (1) tag wiring [see own
      item], (2) `jackdaw/agents/shop_action_space.py` — Discrete(686), offsets pinned in
      `tests/agents/test_shop_action_space.py`, (3) `jackdaw/agents/greedy_hand_policy.py`
      — subset-search greedy baseline (NOTE: `get_best_hand` follows Lua played-selection
      semantics, flush/straight never detect on >5 cards — hence the C(8,5) search; wins
      ~4/6 ante-1 runs at stake 1), (4) `jackdaw/env/shop_run_adapter.py` — full-run
      episodes, hand_policy auto-resolve, `win_ante` horizon knob (engine advances ante
      before the won flag halts, so a win at N leaves ante N+1), pickle snapshot/restore
      with byte-identical RNG continuation verified in
      `tests/env/test_shop_run_adapter.py`, (5) `jackdaw/agents/joker_descriptors.py`
      (300x24 engine-derived descriptor matrix, row 0 = pad) + `jackdaw/env/shop_obs.py`
      (23-key Dict obs, union item rows, `*_ids` via `center_key_id`, PendingTarget
      observable bits) + `jackdaw/agents/shop_policy.py` (unified embedding + frozen
      descriptor buffer + masked-pool trunk; VOCABULARY FREEZE pinned in
      `tests/agents/test_shop_policy.py`), (6) `jackdaw/env/shop_gym.py` — `ShopGymEnv`,
      Discrete(686), tested in `tests/env/test_shop_gym.py` (19 tests). Key decisions
      made while building (6):
      - Env reward = `1{run won}` ONLY; the per-blind density term is emitted every step
        as `info["reward_components"]` (`win`, `blinds_cleared`, `blind_bonus` =
        `blind_clear_bonus(ante_before)` per clear, ante/108 so a full no-skip clear sums
        to exactly 1) — the beta blend lives in the training script via a wrapper, never
        in the env. Blind-cleared detection = diff of `gs["round"]` (the engine increments
        it exactly at blind defeat). The ante-1 Small blind is auto-cleared during
        `reset()` (no decision precedes it) so episodes see at most 23 credited clears.
      - Pending-target state machine: carrier (PickPackCard/UseConsumable) with
        `needs_targets` does NOT step the engine — env stores `PendingTarget`, masks only
        legal SelectTarget combos (`select_target_mask` over `len(gs["hand"])`), the combo
        completes as `target_indices` (hand indices; pack_hand lives in `gs["hand"]`).
        Carrier step returns reward 0/non-terminal. No cancel action. `c_aura` is
        special-cased to (1,1,True) — its centers.json config is empty (its 1-target rule
        lives in `can_use_consumable`), so `get_consumable_target_info` alone would treat
        it as untargeted and use it as a no-op. Per-card target constraints live in
        `legal_target` (currently: Aura requires an editionless target, matching vanilla's
        disabled-button rule) and apply in TWO places by necessity: filtering the
        pending-state combo mask, AND gating carrier legality via eligible-target counts
        (`pack_row_legal` / pending entry) — a carrier entering a pending state with zero
        legal targets would deadlock the episode, since there is no cancel action.
      - PACK_OPENING PickPackCard rows are rebuilt ENV-SIDE (`pack_row_legal`): the
        engine's `_handle_pick_pack_card` applies picks with NO can_use validation — an
        unmasked Buffoon pick at full joker slots would overfill past `joker_slots`
        (Riff-raff-class corruption; TODO: engine-side validation is the real fix). Rules:
        joker picks need a free slot (negative editions exempt, same as the BuyCard mask);
        untargeted consumables gate on `can_use_consumable`; targeted ones on
        `len(hand) >= min_cards`. This also LIFTS `get_action_mask`'s blanket
        Spectral-pack skip-only restriction (a balatrobot RPC limitation — the engine
        handles Spectral picks natively and two-step targeting supplies the targets).
      - `env.snapshot()` bundles the adapter's engine blob + the pending state (restore
        via `reset(options={"snapshot": blob})` or the `start_state_sampler` hook —
        `() -> bytes | None`, None = fresh run; the reservoir mixture policy belongs to
        the training script). Restoring a terminal snapshot raises. Fresh resets retry
        past (rare) seeds where the hand policy loses the auto-resolved first blind.
      - s0 fact confirmed while building: targeted consumables are unusable from the
        owned rows all through s0 — in SHOP the hand is empty (vanilla-consistent), and
        the engine forbids UseConsumable during PACK_OPENING. The pending-"consumable"
        path is implemented and unit-tested anyway (the in-blind merge activates it, and
        `get_action_mask`'s carrier legality — evaluated with an empty highlight set —
        must be upgraded then).
      (7) `scripts/train_shop_ppo.py` + `scripts/eval_shop_policy.py`, tested in
      `tests/scripts/test_train_shop_ppo.py` (16 tests) and smoke-verified end-to-end
      (2048 steps, win_ante=2, greedy partner, ~31 fps on this CPU; schedules decayed,
      reservoir harvested, checkpoint saved+reloaded through eval):
      - Training-loop side of the split: `ShopRewardWrapper` blends
        `r + blend_beta * blind_bonus + count_beta * novelty` from
        `info["reward_components"]`; both coefficients decay linearly to zero via a
        shared mutable `TrainingSchedules` updated by `ScheduleCallback` from PPO
        progress (SB3 updates progress at the START of each rollout collection, so a
        run whose total_timesteps == n_steps never advances it — test gotcha).
      - `CountBonus`: 1/sqrt(N) on the sorted owned-joker key-set, awarded ONLY when
        the set changes (per-step awarding would reward loitering in the shop), and on
        (carrier key, combo size) at pending-target completion.
      - `ShopReservoir`: strata = (ante, pack_pending) bounded deques; sample() = fresh
        anchor prob (must be nonzero, enforced) -> pack stratum with prob `pack_frac`
        (targeting oversampler) -> uniform ante stratum. Harvesting is env-side in the
        wrapper (`harvest_prob` per non-terminal step). Sharing schedules/counts/
        reservoir across envs REQUIRES DummyVecEnv (single process; documented).
      - Horizon curriculum = one invocation per stage; `--init-from prev.zip`
        (canonical action space + obs schema frozen, so weights load verbatim).
      - `eval_shop_policy.py`: reserved `EVAL_` seed suite; metrics win_rate,
        mean_final_ante, mean_rounds_cleared, mean_steps; seeds where the hand policy
        loses the auto-resolved FIRST blind are excluded and reported as
        `n_dead_at_reset` (no shop decision influenced them). `--policy nextround` =
        do-nothing baseline (same partner) isolating shop value from hand skill.
        BASELINE NUMBER (2026-07-06): nextround @ win_ante=2, greedy partner, 50 eval
        seeds -> win_rate 0.0 (0/49 played, 1 dead at reset), mean final ante 1.53 —
        a never-buys shop cannot clear ante 2 with greedy hands, so headroom above the
        floor is wide open for s0.
- [x] Engine tag-context wiring (all 7 previously-unwired contexts now fire; tested in
      `tests/engine/test_tag_wiring.py`, 19 tests):
      - `fire_tag_context(gs, context, first_only=..., **kwargs)` in `tags.py` is the
        single poll: FIFO over `awarded_tags`, consumes entries whose handler fires
        (`consumed`/`consumed_context` flags), leaves conditional non-fires (Investment
        off-boss) un-consumed. Immediate tags now also marked consumed at skip-award.
      - D6: handler CHANGED from `free_rerolls=1` (wrong) to vanilla "rerolls start at
        $0": sets `round_resets.temp_reroll_cost=0` at `shop_start` (cash-out), cost
        climbs $1/reroll, cleared by next `start_round` (receiving end already existed).
      - Rare/Uncommon + editions: `shop.create_shop_slot_card` (new, used by BOTH
        populate and reroll paths) — `store_joker_create` REPLACES the slot's type poll
        with a forced-rarity Joker; `store_joker_modify` then makes the first
        base-edition Joker free + Foil/Holo/Poly/Negative. One tag per card, FIFO.
      - Voucher Tag: extra voucher per tag in `_populate_shop` via
        `get_next_voucher_key(from_tag=True)` ('Voucher_fromtag' pool key pre-existed).
      - Coupon: initial shop cards + boosters cost 0 (vouchers full price; rerolls NOT
        free — fires on populate only). Investment: +$25 at cash-out only when the
        beaten blind was a boss. Juggle: `rr["temp_handsize"]` request before
        `start_round`, applied amount recorded in `cr["temp_handsize_applied"]`,
        REVERTED at end-of-round (the apply existed; the revert didn't — hand size
        would have grown permanently).
      - THREE dormant bugs fixed en route: (1) `shop.calculate_reroll_cost` used
        `or` on `temp_reroll_cost` — Lua `0 or x`→0, Python falls through, so D6's $0
        base was ignored; (2) `_handle_reroll` recomputed from `base_reroll_cost`,
        losing the Reroll Surplus/Glut discount (mutates `round_resets.reroll_cost`)
        after the first reroll AND escalating price while Chaos free rerolls remained
        (vanilla doesn't); (3) `_check_double_tag` read `gs["tags"]` which NOTHING ever
        wrote (Double Tag could never fire) and applied only the dollars field of the
        dup (Orbital/Top-up dups silently dropped) — now scans `awarded_tags`, applies
        the full effect, dup entry behaves like a fresh award.
      - IN-GAME VERIFICATION COMPLETE (user tested in real Balatro, 2026-07-06):
        (a) the tag-forced shop joker is generated ON THE SPOT on its own RNG stream —
        it does NOT grab the next joker from the shop pool sequence. Code CHANGED to
        match: `create_shop_slot_card` now uses vanilla's tag append keys
        (`_TAG_APPEND_BY_RARITY` = {2:'uta', 3:'rta'}) instead of 'sho' for the forced
        create, so the normal shop sequence is untouched (the card that would have
        filled the slot appears in the next slot); regression-pinned in
        `test_tag_wiring.py::test_tag_create_does_not_consume_shop_stream`.
        (b)-(e) all confirmed as implemented, no changes: D6 climb is exactly $0 then
        $1, $2, ...; Coupon leaves vouchers full price and does NOT apply to rerolled
        cards; a rerolled card IS eligible for a pending Rare/edition tag; Investment
        pays after earnings at cash-out, so it does not affect that round's interest.
- [x] Stage 4 hand-demo preset (boss blinds) — GRILLED AND LOCKED (2026-07-06); built and
      tested, generation not yet run. h0 had never seen a Boss blind (all three earlier
      stages exclude "Boss" from `blind_stages`), but full-run shop episodes hit one every
      third blind — an h0 that folds at bosses distorts every s0 shop value toward "we die
      anyway".
      - **Preset** (`generate_hand_demos.py::stage_presets()["stage4_boss"]`):
        `blind_stages=("Boss",)` only (stages 1-3 already cover Small/Big broadly, so this
        stage exists purely for boss exposure), full 150-joker pool + `DEFAULT_COUNT_BANDS`
        (boss awareness and joker coverage compound rather than trading off), 8000 examples
        (provisional, same "marginal exposure not combinatorial coverage" framing as
        stage3's 20000/150).
      - **Key insight that shaped the design**: BC demo generation is single-snapshot —
        `generate_one_example` calls `reset()` once, solves, labels, and does exactly one
        `adapter.step()` purely to validate executability before discarding the state. The
        engine is never stepped forward through a sequence of real hand-turns before
        labeling. So a history-dependent boss's round state (`Blind.hands_used`/
        `only_hand`) can only ever be genuinely set by a real decision *within the same
        trajectory* — which never exists at generation time. Fabricating history is
        therefore only an honest thing to do for bosses whose constraint lives directly on
        the `Blind` instance; anything requiring true round history is out of scope by
        construction, not an oversight (mid-turn PPO rollouts and full runs — which DO call
        `step()` repeatedly across turns within one episode — are where these debuffs get
        exercised for real, with no injection needed).
      - **The Eye** (`HandPlayConfig.randomize_boss_history`, `boss_history_hands_played_range`
        default `(0,3)`, `boss_history_best_hand_weight` default `0.05`): when sampled
        hands-played-so-far > 0, marks that many distinct hand-types "used"
        (`blind.hands_used`), with 5% probability forcing the *current* hand's own
        best-detectable line (`greedy_hand_policy.estimate_best_hand_type`, promoted to
        public) to be one of them. Deliberately adversarial: a build that's only good at one
        hand type should get punished by The Eye more often than chance, since that's
        exactly the resulting-state distribution a build/shop-value signal needs to see to
        learn to value flexibility.
      - **The Mouth**: locks `only_hand` to a plain uniform-random hand type, NOT weighted
        toward the current hand's best line. Its "first hand of the round" is a genuinely
        different, unseen hand this adapter has no way to reconstruct — correlating the lock
        with *this* hand's best type would just be wrong, not adversarial (unlike The Eye,
        where "already blocked" and "current hand's best type" are the same axis by
        construction).
      - **The Ox excluded**: its debuff reads `HandLevels.most_played()`, a run-cumulative
        per-hand-type play count — the same hand-levels/usage-count gap already documented
        as deferred elsewhere in this file (grouped with Supernova). Fixing it here would
        duplicate that fix in a second place; it rides on the existing gap instead.
      - **Bug found and fixed while building this**: `hand_solver.py` never cloned the
        `Blind` object — every hypothetical `score_hand`/`score_hand_base` call (template
        ranking, permutation search, discard-chain recursion, MC future-hand sampling)
        shared and mutated the SAME blind, since `debuff_hand` has no "preview, don't
        mutate" mode `score_hand` ever uses. Under The Eye/Mouth this meant two purely
        hypothetical evaluations of the same hand type — neither ever actually played —
        would corrupt each other (confirmed via repro: second eval incorrectly read as
        already-blocked). The exact same bug existed in `play_ordering.py::best_play_order`,
        which is worse — it drives *real, committed* plays in `HandPlayGymEnv`, so a
        discarded ordering candidate could corrupt the live game's boss state before the
        chosen order is even executed. Fixed with `fast_clone_blind` (mirrors the existing
        `fast_clone_hand_levels`/`fast_clone_rng`/`fast_clone_card` pattern in
        `play_ordering.py`), applied at every hypothetical call site in both files. This was
        invisible before because stage1-3 never set `blind.boss=True`, so `debuff_hand`'s
        history-mutating branches never ran.
      - Tests: `tests/scripts/test_hand_solver_boss_debuffs.py` (Psychic min-card block,
        Flint halving, Eye/Mouth blocking via the solver's `evaluate_value`, shared-blind
        non-corruption through `rank_templates_cheaply`), two regressions added to
        `tests/scripts/test_hand_solver_mutation.py` for the clone-safety bug, and boss-
        history sampling coverage in `tests/env/test_hand_play_adapter.py` (weight-1.0/0.0
        determinism, Mouth's uniformity, the `randomize_boss_history=False` escape hatch,
        The Ox and non-history bosses left untouched).
- [ ] KNOWN OBS LIMITATION — flush/straight structure invisible to the "best hand"
      features (decided: fix at the h1 regeneration seam, NOT now):
      `observation.py::_compute_hand_analysis` calls `get_best_hand` on the FULL 8-card
      hand, but that function ports Lua's played-selection semantics — flush/straight
      predicates only fire within a <=5-card selection, so on 8 cards they NEVER
      detect (verified empirically: a hand containing a complete heart flush reports
      "High Card"; rank-multiple types like Pair still detect fine). Consequence:
      `is_best_hand_card` (card feature 13) and the GC hand_type_vec are
      systematically blind to suit/sequence structure — exactly the cases hardest to
      reconstruct from raw features (suit is a scalar ordinal; mean/max pooling
      destroys counts). It's CONSISTENT (same at gen + inference, nothing false), so
      it's an informativeness gap, not correctness. Expected effect: h-agent weaker on
      flush/straight boards; second-order, s0 will undervalue flush-shaped jokers
      (its values are "given how h0 plays"). Why the self-lock doesn't close: h1's BC
      pool stays domain-randomized (stage 2 over-represents suit jokers), the obs fix
      lands before s1 re-values, and the count-based bonus keeps s-agents sampling
      underused joker sets. Fix plan: O(n) "hand potential" features (max same-suit
      count, straight-window occupancy, rank-multiplicity profile — conveys draws
      too, and respects the <500us encode budget; do NOT subset-search like
      `greedy_hand_policy._best_selection`, 56 evals ~ 3ms blows it), schema-version
      bump, regenerate at h1 (all shards store encoded arrays). DIAGNOSTIC GATE: when
      h0 lands, decompose eval clear-rate vs solver ceiling by board archetype
      (flush-relevant / straight-relevant / pair-family); a large flush-bucket gap
      pulls the fix forward to the stage-4 regeneration seam instead.
      LOCKED 2026-07-07 (h1/s1 grill): the fix lands in the h1 schema bump (see the
      "h1 / s1 seam" section). The archetype decomposition still runs — on h0.5, in
      the same eval pass as the discard-bias fingerprint — but now only calibrates
      expected recovery; it no longer gates timing. One detail superseded: h1's BC
      pool is no longer purely domain-randomized (a harvested stage joins it), but
      stages 1-4 remain, so the stage-2 suit-joker coverage argument stands.
- [x] Hand-card obs width too small for The Serpent (fixed 2026-07-06; SUPERSEDES the
      earlier "regenerate at the h1 seam" plan — the fix turned out not to need
      regeneration at all): The Serpent draws exactly 3 cards after every play/discard
      with no hand-size cap, so hands legitimately grow past 8 mid-round — and growth
      COMPOUNDS (each 1-card action nets +2; worst case with default hands/discards
      budgets is ~20), so no fixed width alone is crash-proof. +Hand-size effects (Turtle
      Bean, Troubadour, Juggler, Juggle tag, vouchers) push full-run hands to 10-13 too.
      Fix (obs-only; action space and demo schema untouched):
      - `hand_play_gym.MAX_HAND_CARDS_OBS = 12`, decoupled from the action space's
        `MAX_HAND_CARDS = 8` (frozen positions). Rows beyond 12 TRUNCATE, never raise:
        the hand is engine-sorted descending and the action space can only address
        positions 0-7, so dropped rows are the lowest cards and unplayable anyway.
        Positions 8-11 are visible-but-unplayable (see the action-space ceiling item
        below). Joker/consumable blocks keep the strict raise (a Riff-raff-class engine
        bug should stay loud).
      - KEY INSIGHT that killed the regeneration plan: widening a masked padded block is
        semantically exact — zero rows beyond the mask are literally what
        `build_observation` produces — so `train_bc.py`'s loader just zero-pads the
        shards' 8-wide hand blocks to 12 at load time (hard-fails if a shard is ever
        WIDER than the obs). No schema bump (feature layout unchanged; width is
        shape-inferable), no shard regeneration, no forced retrain: the pooled encoders
        are row-count-agnostic in parameters, so the existing h0 checkpoint loads into
        the widened model unchanged (verified: real stage1 shards through the new loader
        + h0 checkpoint forward pass; Serpent over-draw episode encodes in-space —
        regression tests in `tests/env/test_hand_play_gym.py::TestHandOverflow`).
      - `generate_hand_demos.py` keeps writing 8-wide blocks (generation is
        single-snapshot; reset hands never exceed 8, and labels must fit the 8-position
        action space regardless). Invariant is now "write width <= MAX_HAND_CARDS_OBS",
        enforced in the loader.
      - The flush/straight fix above is now DECOUPLED from this one: it still needs its
        own schema bump + regeneration at the h1 seam (it changes feature layout).
      - Stage4 eval unblocked and RUN (300 episodes, h0): clear_rate 8.7% vs label-mean
        ceiling 0.209 (~41% recovery, consistent with stages 2-3) — see the h0 BC run
        item above; `runs/bc/h0_s1234_25ep/eval_stage4_boss.json`.
- [x] Joker obs width too small in full runs (fixed 2026-07-07; PARTIALLY SUPERSEDES the
      Serpent item's "joker block keeps the strict raise" line): `build_observation`'s
      joker block raised `entity count 6 exceeds max 5` and killed the first s0 training
      run. Root cause is NOT an engine bug — >5 physical jokers is a LEGITIMATE full-run
      state two ways: (1) Negative-edition jokers don't consume a joker slot (engine
      buy-legality at `actions.py:429` is `len(jokers) >= joker_slots and not is_negative`),
      (2) slot-expanding vouchers raise `joker_slots` above 5. The hand agent never saw
      this because `HandPlayAdapter` injects only base/no-edition jokers within count bands
      (max 5); the shop agent buying a Negative joker is the first path that produces it,
      and the auto-resolved hand phase calls `build_observation` via `HandCheckpointPolicy`.
      Fix (obs-only, NO checkpoint/schema change — the width-5 block is FROZEN until the
      shop merge widens it to 8 at the h1 seam, so h0.5's `.zip` obs space is untouched and
      `predict()` sees the exact width it trained on): encode every joker, then TRUNCATE to
      `MAX_JOKERS=5` (jokers aren't positionally addressed by any hand action, so this is a
      pure informativeness gap like the >12 hand-card tail — the engine still scores all
      jokers). "Bugs stay loud" preserved by raising ONLY on genuine overfill: non-negative
      jokers exceeding `joker_slots` (true Riff-raff-class state). Regression tests in
      `tests/env/test_hand_play_gym.py::TestJokerOverflow` (negative excess truncates,
      voucher-expanded slots truncate, non-negative overfill raises); full h0.5 partner
      path verified on a live 6-joker SELECTING_HAND state (returns a legal PlayHand, no
      crash). Second-order distortion (h0.5 plays 6+-joker states blind to the truncated
      rows -> s0 slightly undervalues Negative/wide-joker builds) is the same class as the
      flush/straight obs gap and self-corrects at h1 when the v2 block widens (to 15, see
      the next item — SUPERSEDES the earlier "widens to 8 at the h1 seam" plan on this and
      the line above: the hand v2 block is `MAX_JOKERS_V2=15`, not 8; 8 is the SHOP row count).
- [ ] KNOWN OBS/ACTION LIMITATION — shop joker cap `MAX_JOKER_ROWS=8` (decision record
      2026-07-15; DECIDED 2026-07-16: fix BOTH halves at s1 KICKOFF, and neither is a
      retrain — see the DECIDED block below; the hand-side counterpart was FIXED — v1 `MAX_JOKERS=5` frozen, v2
      `MAX_JOKERS_V2=15`, expand-not-truncate + dual counter, branch
      `worktree-joker-cap-15`): the shop obs joker block clips physical jokers past 8
      (`shop_obs.py:177` `gs["jokers"][:MAX_JOKER_ROWS]`), so s0 is blind to jokers 9+ on
      a wide/negative build. Structurally DIFFERENT from the hand fix, three ways, so it is
      NOT a mechanical apply:
      1. **Jokers are positionally addressed** — `MAX_JOKER_ROWS=8` is defined in BOTH
         `shop_obs.py:52` and `shop_action_space.py:51` and MUST stay equal, because
         SellJoker slot k targets obs joker row k (header invariant: "the mask for 'sell
         joker 5' must have an obs row 5 to look at"). The hand could widen obs alone
         precisely because jokers there are NOT positionally addressed; the shop cannot.
      2. **8 is already negative-aware headroom, and it only CLIPS (no bug)** — SellJoker=8
         was sized "5 base + Antimatter + negative-edition headroom" (`shop_action_space.py:12`),
         and `build_shop_observation` clips rather than raises, so there is NO silent-drop /
         harvest-labeling bug forcing the issue (unlike the generation raise that forced the
         hand fix). Pure informativeness gap for the rare >8-joker state.
      3. **A live/in-progress s0 checkpoint bakes both spaces** — single obs schema (no
         v1/v2 seam) + `Discrete(686)` head. SellJoker sits at `[446,454)` mid-canonical, so
         widening it IN PLACE shifts PickPackCard/SkipPack/SelectTarget and every pinned
         offset — a direct append-only-contract violation; the only append-safe add is
         non-contiguous SellJoker rows at 686+.
      THREE CHOICES:
      - **A. Leave at 8** (recommended): 8 already = 5 base + Antimatter + 2 negative
        headroom; obs clips safely; >8 jokers is rare; widening breaks a mid-training s0 run
        and the append-only contract for a marginal gain. No forcing function (unlike the
        hand's generation drop-bug).
      - **B. Widen obs only -> 15, keep SellJoker=8**: agent SEES all jokers for
        scoring/synergy but jokers 9-15 stay unsellable. Breaks s0's obs space (retrain), NOT
        the action head. Coherent middle, only worth it if s0 is being retrained anyway.
      - **C. Widen obs + SellJoker -> 15**: full parity, but breaks s0 obs AND action head,
        plus the append-only contract (in-place shift) or a non-contiguous SellJoker append;
        full s0 retrain. Most invasive.
      DECIDED 2026-07-16 (supersedes the "Gating fact" framing, A's recommendation, and
      the B/C cost labels above, both of which were WRONG about "retrain"): do BOTH
      halves at s1 kickoff — full C-level parity at B-level cost, no retrain:
      - **Obs widen 8 -> 15 is WEIGHT-PRESERVING, not a retrain**: the shop trunk is
        `masked_pool` over shared per-row encoders (verified in `shop_policy.py` — no
        flatten, no positional parameters), so widening a masked padded block is
        semantically exact (the `MAX_HAND_CARDS_OBS` 8->12 argument verbatim: zero rows
        past the mask contribute exactly nothing). What breaks is only SB3's obs-space
        equality check at `--init-from` — fixed by a load shim (rebuild the policy with
        the widened space, `load_state_dict` the a4_v4 params; the
        `load_bc_weights_into_policy` precedent), not by retraining. Verification gate:
        old vs widened model must produce byte-identical outputs on <=8-joker states.
      - **SellJoker rows 9-15 = SEVEN non-contiguous cold head rows appended at
        [687,694)** (right after SkipBlind at 686) — exactly the SkipBlind mechanism:
        append-only contract preserved, pinned offsets untouched, cold rows learned
        during s1's PPO (which runs anyway — new partner, new shaping). Cost = the
        split SellJoker layout ([446,454) + [687,694)) plus a mapping shim in the mask
        builder and action decode. The split is TEMPORARY by the project's own plan:
        consolidate when Candidate B subsumes the flat head at the in-blind merge.
        Invariant #1 above generalizes, not breaks: "SellJoker slot k targets obs joker
        row k" now spans both blocks (k<8 -> 446+k, k>=8 -> 687+(k-8)) — pin the full
        mapping in tests, since the two constants stop being equal-by-inspection.
      - **Why s1 exactly**: >8-joker states only become strategy-relevant when the hand
        partner can exploit wide/negative builds — which is exactly what h1 + Candidate
        B deliver. Pre-s1 the clip is second-order (h0.5 can't cash those builds in;
        partner noise dominates their value estimates). Later than s1 means s1 trains
        blind to rows 9+ under the first partner that makes them worth buying, and
        s1's values are what h2/s2 inherit. a4_v4 stays final and untouched; nothing in
        the C1/C2 -> regen -> h1 chain reads the shop obs schema.
      - **Riders**: (a) 15 is chosen to MATCH `MAX_JOKERS_V2=15` — hand and shop must
        agree on which states are fully observable (divergence here is the solver/env-
        divergence bug class applied to obs). (b) The reservoir survives untouched:
        snapshots are engine blobs re-encoded at load, zero migration — one more reason
        the s1 seam is cheap.
- [ ] KNOWN ACTION-SPACE CEILING — 8-position combo enumeration vs big hands (decision
      record 2026-07-06; DECIDED 2026-07-07: Candidate B COMMITTED, build at the h1
      seam — see the "h1 / s1 seam" section and the RESOLVED note below): the canonical
      Discrete(436) can only select among hand positions 0-7, but >8-card hands are
      SYSTEMIC, not a Serpent tail — +hand-size builds a competent shop agent should buy
      (Turtle Bean, Troubadour, Juggler, Juggle tag, vouchers) mean 10-13-card hands at
      every hand-turn, where the agent can never play a flush completion sitting at
      position 9 or discard below position 7 (discarding LOW cards is exactly what you
      do). Downstream distortion: s0's values are "given how h0 plays", so the cap
      systematically undervalues the whole +hand-size joker family — same distortion
      class as the flush/straight obs gap, but on the action side.
      - RETRACTION: the earlier blanket "multi-step chains under sparse reward are an
        exploration trap" argument does NOT apply to autoregressive action *decoding*
        (grilled 2026-07-06): the rejected reorder/cancel actions were extra ENV steps
        competing with direct actions; autoregressive selection is a factorization of
        one decision within a single env step — no wasted turns, joint log-prob = sum of
        sub-pick log-probs, per-sub-step masking is natural (AlphaStar-style compound
        distribution). Per-card INDEPENDENT scoring (multi-binary "pick your top k")
        stays rejected on expressiveness: subset value is joint (a 6H is only worth
        picking with four other hearts), which independent Bernoullis structurally
        cannot represent.
      - Candidate A — widened enumeration with computed logits: extend positions to 12
        (append the new combos after the shop block per the append-only contract,
        ~3,170 hand actions), and compute each combo's logit from its cards' embeddings
        (pool + MLP) instead of a dense head row. Keeps MaskablePPO, masked-CE BC, the
        KL leash, and the shop-merge "canonical index == head row" plan; BC data reuses
        as-is (labels are sets). Costs: ugly split index layout (hand combos in [0,436)
        AND post-686), O(n^5) growth caps it at ~12 positions for good.
      - Candidate B — autoregressive pointer head: action = (type, pick, ..., done)
        over card embeddings. Unbounded width, linear compute, identity-conditioned by
        construction, would also subsume the shop's SelectTarget combo block later.
        Costs: exits vanilla MaskablePPO (custom compound distribution;
        `KLToBCMaskablePPO` rewrite; KL leash becomes teacher-forced per-step KL sum),
        BC becomes sequence CE (easy — canonicalize set -> sorted picks), and it
        unwinds the locked shop-merge flat-head design. Probably the right end-state,
        but wrong to put ahead of s0 on the critical path (its distortion at s0 is
        second-order by the same argument that deferred SkipBlind).
      - EVIDENCE GATE (cheap, no training needed): when `shop_gym.py` is built,
        instrument hand-turns with a counter for (a) fraction of turns with hand > 8,
        and (b) fraction where `GreedyHandPolicy`'s UNRESTRICTED C(n,5) choice touches a
        position >= 8 (greedy searches the full hand, so it's a free oracle for what
        the cap forbids). If 12 positions covers ~all of it -> A; if the distribution
        runs past 12 routinely -> B. Also note: the trunk is permutation-invariant
        (masked mean/max pooling) while the action space is positional — the model
        binds identity to position only via the engine's descending sort; both A and B
        would make that binding direct, which is an independent argument for doing ONE
        of them eventually even if the counter reads low.
      - RESOLVED 2026-07-07: the evidence gate is DEAD as a decider — it's circular
        (the counter is conditioned on current policy behavior: if s0 rarely buys
        +hand-size jokers because h0.5 can't exploit them, it reads low without proving
        A@12 suffices) and the >12 prior is strong (one Turtle Bean = 13-card hands for
        multiple turns). Candidate B committed at the h1 seam, hand agent only; the
        shop_gym counter is never built (the hand-size histogram falls out of the
        harvest corpus and only tunes B's max decode length). Full decision record in
        the "h1 / s1 seam" section.
      - SHOP-SIDE (2026-07-16): the same combinatorial `SelectTarget` block also caps
        pack-tarot targeting, which is LIVE in s0/s1 (Arcana/Spectral pack open deals a
        hand, `game.py:327-342`) — the pointer replaces it at the in-blind merge. Full
        record + timeline + escape hatches in the "In-blind merge — shop targeting →
        pointer + Death direction" section.
- [x] h0-checkpoint hand-policy wrapper for the shop env's `hand_policy` slot —
      `jackdaw/agents/hand_checkpoint_policy.py::HandCheckpointPolicy`, tested in
      `tests/agents/test_hand_checkpoint_policy.py` (10 tests, both BC .pt and PPO .zip
      kinds). Deterministic masked-argmax, drop-in for `GreedyHandPolicy`. Refactored the
      shared decode path out of `HandPlayGymEnv` into module-level
      `hand_play_gym.hand_action_mask` + `action_to_engine_action` (the env methods now
      delegate; the play path still routes through `best_play_order`), so the wrapper
      feeds h0 byte-identical inputs to `eval_hand_policy`'s `_BCPolicy` and the >8-card
      Serpent over-draw degrades identically (obs truncates to 12, Discrete(436) mask
      spans positions 0-7 = always a legal play; no greedy fallback needed).
      - EMPIRICAL FINDING (2026-07-07, real h0 = `runs/bc/h0_s1234_25ep`): h0-as-partner
        is currently WEAKER than the greedy baseline on easy blinds. On EVAL_0..19 at
        win_ante=1, greedy clears the auto-resolved ante-1 Small blind 20/20; h0 clears
        it ~13/20 (loses ~35%). Action traces confirm the wrapper is faithful (legal,
        coherent plays, correct scoring) — h0 simply misplays: undersized plays (e.g. a
        1-card PlayHand for 16 chips when it needed to score), over-discarding (the
        documented play-only future-hand label bias), poor budget management. This is
        expected pre-PPO BC behavior (h0 recovers only ~26-52% of solver ceiling and was
        deliberately under-trained to preserve entropy) and is exactly what the
        PPO-fine-tune / bootstrap loop corrects. IMPLICATION for s0: an h0 that folds
        35% of ante-1 Smalls injects large variance into shop-value estimates (a good
        purchase looks bad when the partner randomly loses the next blind). Open choice
        for the bootstrap kickoff — train s0 against greedy first (reliable, low-noise,
        but the ablation baseline not the real partner) vs h0 (real distribution, noisy),
        or fine-tune h0 with PPO before wiring it in. Not yet decided.
- [x] h0 -> h0.5 PPO fine-tune (DECIDED 2026-07-07: fine-tune h0 with PPO BEFORE wiring
      it into s0, per the h0-wrapper finding that raw h0 folds 35% of ante-1 Smalls and
      would inject huge variance into shop-value estimates). RUN COMPLETE 2026-07-07 (2M
      steps on the 9600X, mixture of all four stages, checkpoint transferred back to
      `runs/hand_ppo/hand_ppo_2000000_steps.zip`):
      - RESULT — the fine-tune fixed exactly the partner-reliability problem it targeted.
        h0.5-vs-greedy-vs-h0 on the auto-resolved ante-1 Small (50 EVAL_ seeds, greedy
        partner semantics): greedy 49/50, h0(BC) 32/50 (the ~35% fold confirmed), **h0.5
        47/50** — fold rate 36% -> 6%, on par with greedy (94% vs 98%) while keeping real
        hand skill greedy lacks. Domain-randomized per-stage clear rates moved only
        modestly (those distributions are mostly-unwinnable by design, so absolute
        clear-rate is dominated by the unwinnable fraction — see the median-p_clear=0 note
        on the h0 BC item): stage1 3.7%->3.7% (flat, barren no-joker), stage2 17.3%->20.0%
        (44%->51% recovery), stage3 16%->17% (52%->55%), stage4 8.7%->9.0% (41%->43%).
        Evals in `runs/hand_ppo/eval_stage{1,2,3,4}.json`; comparison via the throwaway
        `compare_partners.py` (survival test = phase != GAME_OVER after `ShopRunAdapter.reset`,
        NOT a RuntimeError catch — reset returns a terminal GAME_OVER snapshot on a lost
        blind, it does not raise; the earlier RuntimeError-catch version falsely read 20/20).
      - DECISION for s0 kickoff: use h0.5 as the partner (reliable early blinds now, so
        shop-value estimates aren't drowned in partner-loss variance). Greedy stays the
        ablation baseline.
      Setup notes from the build (kept for reference):
      - `train_hand_ppo.py` extended to a STAGE MIXTURE (`--stages` comma-list, default =
        the four stages BC pooled; `--stage` still forces one). Round-robin across envs.
        RATIONALE: bosses live ONLY in stage4 (stages 1-3 are Small/Big), so a single-stage
        fine-tune would leave boss play to drift as the KL leash decays — a real regression
        for a full-run partner. `make_vec_env` now takes a config OR a list (single-config
        callers unchanged). Tested in `test_train_hand_ppo.py::TestStageMixture`.
      - ENGINE BUG found + fixed while shaking this out (blocked the run entirely, and was
        a latent scoring bug too): Marble Joker's setting-blind create-applier
        (`game.py::_apply_setting_blind_mutations`, `ctype=="playing_card"`) hand-built
        `{"effect": enhancement}` storing the enhancement CENTER KEY ("m_stone") where the
        effect NAME ("Stone Card") belongs. Consequences: (1) every Marble stone card
        scored as a NORMAL card everywhere (all Stone logic checks `effect == "Stone Card"`
        — no +50 chips, not stone in flushes, not counted by Stone Joker), (2) it crashed
        `round_lifecycle.reset_round_targets` (leaked the Stone filter, then hit the
        base=None a stone card carries via an unguarded `mail.base.id`). Fixed by using
        `set_ability(enhancement)` (matches the deck-init path ~L2058), and hardened
        `reset_round_targets`'s valid-card filter to also exclude base-less cards
        (defense: they can never be the idol/mail/castle card). Regression tests in
        `test_jokers_integration.py::TestMarbleJoker`; 996 engine tests pass. NOTE: some
        stage3/stage4 demo seeds with Marble Joker were silently skipped during generation
        (the crash was caught per-example) — not worth regenerating (rare, and PPO against
        the fixed engine is the corrector).
      - RUN COMMAND (on the 9600X, needs the h0 checkpoint transferred to
        `runs/bc/h0_s1234_25ep/` since runs/ is gitignored, plus this branch's code):
        `uv run python scripts/train_hand_ppo.py --bc-checkpoint
        runs/bc/h0_s1234_25ep/bc_checkpoint.pt --total-timesteps 2000000 --n-envs 8
        --log-dir runs/hand_ppo/h0_finetune_s0 --seed 0`. Shakeout verified end-to-end on
        this machine (kl_bc≈0.008 at warm start, leash decaying, model saved).
- [ ] Bootstrap loop orchestration (h0.5 -> s0 -> rollout -> h1 -> s1 -> ...). Partner
      wrapper (HandCheckpointPolicy) and both training scripts now exist; h0.5 fine-tune is
      DONE and chosen as the s0 partner (above). WIRING DONE 2026-07-07: both
      `train_shop_ppo.py` and `eval_shop_policy.py` take `--hand-policy <ckpt>` (omit =
      greedy baseline). Threads a SINGLE shared HandCheckpointPolicy instance through
      make_train_env/build_model into every ShopGymEnv AND the eval env (DummyVecEnv is
      single-process + the policy is deterministic/stateless, so one instance is correct
      and avoids N torch copies). `load_hand_policy(path)` helper; smoke-verified
      end-to-end with the h0.5 zip (train 512 steps @ ~28 fps + nextround eval, partner
      loads, reservoir harvests, checkpoint round-trips). NOTE: with h0.5 as partner,
      `n_dead_at_reset` drops toward 0 (vs greedy's occasional loss), so re-baseline the
      nextround floor against h0.5 before reading s0's win rate.
      NEXT ACTION — reservoir persistence is DONE (see the s1-section item), so s0 is
      unblocked. Kick off s0 on the 9600X (horizon
      curriculum, one invocation per stage, `--init-from` chains the model and
      `--init-reservoir` chains the reservoir; runs/ is
      gitignored so transfer the h0.5 zip to
      `runs/hand_ppo/hand_ppo_2000000_steps.zip` first). Stage a2 (single line):
      `uv run python scripts/train_shop_ppo.py --win-ante 2 --total-timesteps 2000000
      --n-envs 8 --hand-policy runs/hand_ppo/hand_ppo_2000000_steps.zip
      --log-dir runs/shop_ppo/s0_a2 --seed 0`
      then a4 with `--win-ante 4 --init-from runs/shop_ppo/s0_a2/shop_ppo_final.zip
      --init-reservoir runs/shop_ppo/s0_a2/reservoir.pkl --log-dir runs/shop_ppo/s0_a4`
      (keep --hand-policy/--total-timesteps/--n-envs), then a8 likewise
      (--init-from + --init-reservoir from s0_a4). Eval each with
      `eval_shop_policy.py --policy <best_model.zip> --win-ante N --hand-policy <h0.5 zip>`
      (MATCH the partner to what s0 trained against) and the `--policy nextround` floor.
- [ ] Server-log parser for money/ante/failure calibration statistics (not started; only
      manual `grep` exploration done so far on two 1000-2000 line samples). DEMOTED
      2026-07-07: money calibration now comes from harvested per-ante marginals (see the
      "h1 / s1 seam" section — strictly closer to the distribution h1 actually faces, at
      zero build cost); this item's remaining value is only seeding real blind
      sequences/decks for training episodes.

## Money/dollar handling (DESIGNED 2026-07-07, awaiting s0 critic)

Marginal value of a dollar is context-dependent (interest thresholds, reroll cost scaling,
whether you've already cleared the current blind's score requirement). Planned approach:
derive a marginal-value-of-$1 curve from the shop-agent's own trained critic
(`V(state, money=k) - V(state, money=k-1)`) once it exists, feed that down into the hand-
agent's score/cash tradeoff. Concrete form is now locked (see "h1 / s1 seam" — "Terminal
$ term"): offline `V_curve(ante, $)` lookup from counterfactual money-sweeps of the s0
critic over harvested shop states, added clear-gated to the hand env's terminal reward at
h1, with the build-specific-valuation caveat and the interest-ordering rider recorded
there. Still blocked on the s0 critic existing; the `hand_solver.py` placeholder stands.

## Agent skills

### Issue tracker

Issues tracked in GitHub Issues on Daz029/balatro-strategy via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) — no repo-specific overrides. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root (neither exists yet — created lazily by `/domain-modeling`). See `docs/agents/domain.md`.
