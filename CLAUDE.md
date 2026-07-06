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

### Shop-agent design — GRILLED AND LOCKED (2026-07-05); build not started

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
      policy eval is BLOCKED — see the MAX_HAND_CARDS known-obs-limitation item below.
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
- [ ] Shop-agent BUILD (design grilled and locked — full decision record in "Shop-agent
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
        it as untargeted and use it as a no-op.
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
      REMAINING: (7) `scripts/train_shop_ppo.py` (MaskablePPO, beta/c blend wrapper,
      count-based bonus, reservoir sampler) + `scripts/eval_shop_policy.py`; smoke on
      ante-2 horizon with greedy partner.
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
      - IN-GAME VERIFICATION PENDING (user offered to test in real Balatro; assumptions
        marked TODO in code): (a) does a tag-forced shop joker still burn the
        type-selection RNG poll, and does its pool append key differ from 'sho'?
        (b) exact D6 cost climb ($0 then $1,$2... assumed); (c) Coupon skips vouchers
        and skips rerolled cards (assumed yes/yes); (d) a rerolled card is eligible for
        a pending Rare/edition tag (assumed yes); (e) Investment pays after earnings at
        cash-out, so it does NOT affect that round's interest (assumed).
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
- [ ] KNOWN OBS LIMITATION — hand-card obs width (`MAX_HAND_CARDS=8`) too small for the
      boss The Serpent (decided: fix at the h1 regeneration seam TOGETHER WITH the
      flush/straight fix above, NOT now): The Serpent always draws 3 cards after a play or
      discard, so discarding fewer than 3 from a full hand legitimately grows the hand to
      9-10 cards. The hand obs block is fixed at 8 rows (demo schema is 8x15), so
      `hand_play_gym.py::build_observation` raises `entity count 9 exceeds max 8` mid-round on
      `step` (verified 2026-07-06, stage4 seed `EVAL_00000236`, blind The Serpent, hand grew
      to 9). This is NOT a bug — it's a faithful engine state the obs simply can't represent.
      At reset hand_size is only 7-8; the overflow appears only after a play/discard under The
      Serpent. Demo generation is single-snapshot (never rolls forward past reset), so it
      never hit this — training data has zero >8-hand states — but eval AND PPO rollouts on
      stage4_boss DO. Consequence: no policy clear-rate is obtainable on stage4 until fixed
      (its solver ceiling, 0.209, is still available from demo labels, since generation only
      solves the reset state). Fix: widen `MAX_HAND_CARDS` to ~10-12 — but the obs width in
      `hand_play_gym.py` and the demo write width in `generate_hand_demos.py` must move
      together — schema-version bump, regenerate all shards + retrain. Same regeneration seam
      as the flush/straight fix; do them in one pass. Interim option rejected: skipping
      un-encodable episodes in eval biases stage4 down by dropping The Serpent (a real boss),
      so it's not a clean number. Distinct from the Riff-raff MAX_JOKERS overfill (that was an
      engine bug, now fixed; this is a genuine obs-width shortfall, no engine bug involved).
- [ ] Bootstrap loop orchestration (h0 -> s0 -> rollout -> h1 -> s1 -> ...).
- [ ] Server-log parser for money/ante/failure calibration statistics (not started; only
      manual `grep` exploration done so far on two 1000-2000 line samples).

## Money/dollar handling (deferred, not solved)

Marginal value of a dollar is context-dependent (interest thresholds, reroll cost scaling,
whether you've already cleared the current blind's score requirement). Planned approach:
derive a marginal-value-of-$1 curve from the shop-agent's own trained critic
(`V(state, money=k) - V(state, money=k-1)`) once it exists, feed that down into the hand-
agent's score/cash tradeoff. Not implementable until a shop critic exists — currently a
placeholder in `hand_solver.py`'s design notes.

## Agent skills

### Issue tracker

Issues tracked in GitHub Issues on Daz029/balatro-strategy via the `gh` CLI; external PRs are not a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) — no repo-specific overrides. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root (neither exists yet — created lazily by `/domain-modeling`). See `docs/agents/domain.md`.
