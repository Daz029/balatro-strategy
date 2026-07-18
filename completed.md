# Completed work

Completed implementation items moved from `CLAUDE.md`. Design decisions and active plans remain in `CLAUDE.md` for context.

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

