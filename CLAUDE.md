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
  none requiring external data:**
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
      `tests/scripts/test_hand_solver_duplicate_cards.py`. Actual joker-pool sizes/target
      per-stage example counts are left as CLI parameters, not hardcoded — not yet decided.
- [ ] BC + PPO fine-tune training loop for hand-agent (design decided: partial-convergence BC,
      critic warm-start via regression on solver values, adaptive-KL-to-BC-policy fine-tune).
- [ ] Shop-agent environment/action-space design (not yet started — buy/sell/reroll/skip,
      combinatorial per-shop-slot choices).
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
