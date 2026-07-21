# s1 training hiccups — diagnoses, measurements, and two disproven theories

Investigation record for the `s1_a4_v3` run (2026-07-20). Three hypotheses were
raised to explain why the a4 stage plateaued; **two were disproven by
measurement and one was confirmed.** The disproofs are the load-bearing part of
this document — both wrong theories were plausible, both drove real config
changes, and both cost a training run before the measurement that killed them.

Read the "Future worries" section even if nothing here looks relevant: the
checkpoint-selection finding contaminates the whole bootstrap chain, not just
this run.

> **PROVENANCE WARNING (added 2026-07-21).** The numbers in Issues 2 and 4 below
> were read off a TensorBoard event file that **may not be this run's.** The
> local `runs/shop_ppo/s1_a4_v3/` holds a loose top-level
> `events.out.tfevents.1784539401.*.20988 (1).0` that is **byte-identical** to
> the one in `runs/shop_ppo/s1_a4_WARM/` — same timestamp, same PID, same
> length, verified identical contents. `train_shop_ppo.py` calls `model.learn()`
> with no `tb_log_name` (line 976), so SB3 writes to a **subdirectory**
> (`s1_a4_v3/MaskablePPO_1/`), exactly as `s1_v2/` and `s0_a4_v4/` show locally.
> A loose file at the top level of a run dir was therefore **not written by this
> script** and is a hand-copy stray. Until `s1_a4_v3/MaskablePPO_1/` is
> transferred, treat Issue 2's 0.200 peak / 0.108 / 0.026 and Issue 4's collapse
> as **possibly describing `s1_a4_WARM`**. See "Reading the wrong file" in the
> a3 section for how this was caught and the one-line check that catches it.

## The run

```
--win-ante 4 --s1-schema
--init-from     runs/shop_ppo/s1_a2/best_model/best_model.zip
--init-reservoir runs/shop_ppo/s1_a2/reservoir.pkl
--phi-checkpoint runs/shop_ppo/s0_a4_v4/best_model/best_model.zip --blend-beta0 0
--hand-policy   runs/hand_ppo_b/h1/best_model/best_model.zip --partner-money-ordering
--init-temperature 5.0 --ent-coef 0.01
--total-timesteps 1000000 --n-envs 8
--eval-freq 25000 --eval-episodes 200 --checkpoint-freq 100000
```

The predecessor `s1_a2` stage used the same shape, warm-started from
`s0_a4_v4/best_model` for **both** `--init-from` and `--phi-checkpoint`, with
`--eval-episodes 50`.

Note the eval-episode discrepancy: every logged eval value in `s1_a4_v3` is a
multiple of 0.02, so the callback ran **50** episodes regardless of what the CLI
said. That is itself worth chasing (see Future worries #1).

## Ground truth — honest 200-episode evals

All against the h1 partner with `--partner-money-ordering`, `EVAL_*` seeds.

| policy | horizon | win rate | mean final ante | mean rounds | mean steps |
|---|---|---|---|---|---|
| `s1_a4_v3/best_model` | ante 2 | **0.565** | 2.36 | 4.10 | 16.4 |
| `s1_a4_v3/best_model` | ante 3 | **0.315** | 2.68 | 5.09 | 25.6 |
| `s1_a4_v3/best_model` | ante 4 | **0.090** | 2.785 | 5.45 | 29.9 |
| `nextround` (floor) | ante 4 | **0.000** | 1.38 | 2.39 | 4.85 |

Per-ante conditional survival:

| transition | P(clear \| reached) |
|---|---|
| → ante 3 | 0.558 |
| → ante 4 | 0.286 |

**s1 is the strongest result the project has produced**: 0.565 at ante 2 against
s0's recorded 0.268. Caveat that must travel with that number — s0's 26.8% was
measured with the **h0.5** partner and this is **h1**, so the doubling is the
*pair* improving. It cannot be attributed to the shop policy alone.

The decay is smooth and *accelerating* (0.558 → 0.286), not a cliff at a
specific ante. That signature is builds failing to compound against a
superlinear difficulty curve — a joint shop/hand failure, not "the partner hits
a wall at blind X."

## Issue 1 — the entropy "collapse" — DISPROVEN

**Hypothesis.** `train/entropy_loss` sat at −0.05 and reached it within a few
thousand steps. Read as: a converged stage hands the next horizon a
near-deterministic policy, so PPO cannot explore. Neither `--ent-coef` nor
`--learning-rate` addresses an *inherited* collapse (the bonus claws back
against a saturated softmax; a smaller step size holds the policy at the bad
init more faithfully). Proposed fix: divide the action head's weight and bias by
a temperature `T > 1`, which scales all logits uniformly — flattening the
softmax while preserving argsort exactly, so the learned ranking survives and
only the confidence is undone.

**Built.** `soften_action_logits` + `--init-temperature` in
`scripts/train_shop_ppo.py`, with `test_init_temperature_raises_entropy_and_keeps_the_ranking`
pinning the argsort-preservation property. Committed `1ab6b79`.

**Result.** At `--init-temperature 5.0`: entropy went 0.05 → 0.1 and stayed
there for ~500k steps, drifting to 0.14 by 800k. Final `entropy_loss` −0.1785.

**Disproof.** Measured the action head against a freshly initialized one:

| | std | absmax |
|---|---|---|
| fresh SB3 init, shape (694, 256) | 0.00038 | 0.0020 |
| `s1_a2/best_model` | 0.0204 | 0.3376 |

54× the init std, 170× the absmax. Two conclusions, in tension:

1. An earlier claim in this investigation that the head was "near fresh-init
   scale, therefore barely trained" was **wrong by two orders of magnitude** —
   asserted from memory of SB3's `gain=0.01` without computing what that
   produces on a (694, 256) matrix.
2. But temperature still failed empirically. Dividing by 5 put the weights
   *below* fresh-init scale and entropy still only doubled, which means the
   large logit spread comes from **feature magnitude**, not head weights.

**And the whole framing was wrong anyway.** Entropy was *rising* (0.1 → 0.14
over 300k steps), not collapsing; the reference ceiling of 2.3 nats assumed ~10
legal actions and was never measured (many shop states have 2–4 legal moves,
ceiling 0.7–1.4 nats); and the agent scores 0.565 at ante 2 against a 0.0 floor,
which is not what an unable-to-explore policy looks like.

**Disposition.** `--init-temperature` is retained in the codebase (harmless,
tested, argsort-preserving) but **should not be used**. It addressed a
non-problem.

## Issue 2 — checkpoint selection is a lottery — CONFIRMED

**Finding.** `s1_a4_v3`'s logged eval curve peaked at 0.200 at 460k steps. The
same checkpoint measured honestly on 200 episodes scores **0.090**.

**Proof.** Eval ran 50 episodes (all values are multiples of 0.02). At a true
win rate near 0.108, binomial σ ≈ 0.042. The 0.200 peak is **2.2σ** above the
curve's own mean — and the expected maximum of 50 independent standard-normal
draws is ≈ 2.25σ. The peak is exactly what a flat policy with this noise level
produces by chance; no signal is required to explain it. `MaskableEvalCallback`
saves `best_model` on `mean_reward` improvement, so **it selects the luckiest
measurement, not the best policy**, and honest re-evaluation regresses it to the
mean. The 0.20 and the 0.09 do not disagree; the 0.09 is correct.

The eval env is confirmed unwrapped (`train_shop_ppo.py` ~line 937: plain
`ShopGymEnv`, no Φ, no blend), so `eval/mean_reward` *is* an honest win rate —
it is just an extremely noisy one.

**Fix.** `--eval-episodes 200` (σ ≈ 0.021) on every subsequent run, or select on
the final model. At 200 episodes, differences under ~0.04 between checkpoints
remain noise.

## Issue 3 — Φ dominating the critic — DISPROVEN

**Hypothesis.** `train/explained_variance` = 0.999 and `train/value_loss` =
0.002. Proposed mechanism: with `gamma=1.0` and `blend_beta=0`, real reward is
`1{won}` (sparse, terminal, binary), while potential shaping telescopes so the
return from `s` is `R − βΦ(s)`. Since `Φ` is a deterministic function of state —
and the critic shares its architecture, so can represent it exactly — the critic
memorizes `−βΦ(s)` and the genuine win/loss signal becomes a small residual.

**Disproof.** EV plotted against `phi_beta` as the schedule decays:

| step | phi_beta | explained_variance |
|---|---|---|
| 987k | 0.0149 | 0.9997 |
| 991k | 0.0108 | 0.9990 |
| 995k | 0.0067 | 1.0000 |
| 999k | 0.0026 | 1.0000 |
| 1000k | 0.0006 | 0.9989 |

Across all 25 measurements where `phi_beta` < 0.05: **mean EV 0.9985, min
0.9821**. With the shaping term essentially removed, the critic still explains
99.85% of return variance. It was never fitting Φ.

**Second, independent error in the same hypothesis.** EV ≈ 1.0 was treated as a
smoking gun at all. In PPO a high explained variance normally indicates a
*healthy* critic. The stated mechanism ("advantages collapse to zero") is also
wrong on its own terms: GAE advantages derive from TD errors and from how
actions differ at a state, not from `1 − EV`. A perfectly fit value function
still yields meaningful advantages.

**What the number actually means** (the surviving explanation, proposed by the
user): the return is **largely determined by the state the shop finds itself
in, not by the marginal decision taken there**. By the time the agent chooses
buys, the build's trajectory is mostly set. The critic learned to read that.
This is directly consistent with the 0.286 conditional at ante 4 — most runs are
visibly doomed well before they end. It is the shop learning how the hand fails.

**Disposition.** Φ is not harmful in the way claimed, but it is also not earning
its keep: it is a potential fitted to s0's partner (h0.5) and horizon, and it
demonstrably is not what drives the critic's accuracy. `--phi-beta0 0.1` is
justified as a simplification, not a rescue.

## Issue 4 — the late collapse — REAL, UNEXPLAINED

**Finding.** The eval curve's first 40 measurements average **0.108** with no
trend. The last 10 average **0.026**. Ten evals is 500 episodes — far too much
to be noise. `mean_ep_length` falls with it (35 → 17.8), so late episodes die
earlier.

This is the one confirmed problem that has no confirmed cause.

**Coincidence to note, not a conclusion.** Both dense terms decay to zero on the
same linear schedule and both arrive there in that window: `phi_beta` (1.0 →
0.0006) and `count_beta` (peak 0.045 → 0). An earlier draft attributed the
collapse to Φ's removal; with Φ exonerated as the critic's driver, the vanishing
**count-based exploration bonus** is the better remaining suspect — a policy
that gets actively *worse* rather than merely stopping improving is more
consistent with losing exploration pressure than with losing a potential.

Treat that as a hypothesis. Two have already died here.

**Mitigation available now.** Take a checkpoint from ~700k (before the collapse
window) rather than `best_model` or the final model.

## Future worries

1. **Checkpoint-selection noise contaminates the bootstrap chain, not just one
   run.** `s1_a2` also ran `--eval-episodes 50`, so *its* `best_model` is also a
   lottery pick — and `s1_a4_v3` was warm-started from it. Every stage of the
   horizon curriculum hands the next one a checkpoint selected partly on a
   favorable seed draw. This compounds silently across the whole h/s bootstrap.
   Separately: the run was launched with `--eval-episodes 200` but the callback
   evidently used 50. **Verify the CLI argument actually reaches
   `MaskableEvalCallback`** — if it does not, every logged eval in the project's
   history is noisier than assumed.

   **AMENDED 2026-07-21 — the plumbing is fine.** `s1_a3` logged eval values in
   multiples of 0.005 (= 200 episodes) from the same code path, so
   `--eval-episodes` does reach the callback. The 50-episode granularity was a
   property of whichever run that event file describes — which, per the
   provenance warning above, may be `s1_a4_WARM` rather than `s1_a4_v3`. The
   *first* half of this worry (lottery selection compounding across the
   bootstrap chain) stands and was independently reconfirmed: see the a3
   section's checkpoint sweep.

   Also found: the in-training eval env **never resets its episode counter**
   (`shop_gym.py:323`), so each eval draws a fresh 200 `EVAL_` seeds and the a3
   run consumed `EVAL_0..7999`. This is unbiased (seeds are iid, so binomial
   remains the correct null) but it means the callback does **not** have the
   fixed-suite property `eval_shop_policy.py`'s docstring describes — only the
   CLI evaluator does. Consequence: training-curve values and CLI evals are
   never directly comparable, and it consumes reserved eval seed space at
   200 seeds per eval.

2. **a8 is unreachable and should not be attempted.** At a 0.286 conditional on
   ante 4, compounding through ante 8 is effectively zero, and PPO would spend a
   run doing credit assignment against almost no terminal reward. This repeats
   the documented s0 decision to skip a8 ("a4 plateaued at the early-game/
   hand-partner bottleneck, not the shop").

3. **Sparsity is not the a4 problem, so finer horizons will not fix it by that
   route.** 1M steps at ~30 steps/episode is ~33k episodes; 9% of those is
   ~3,000 winning episodes. That is not a sparse signal. An a3 stage is still
   worth doing — it keeps the policy in the 30–55% band where the advantage
   signal is richest, and stages are prefixes of the true objective so nothing
   is unlearned at the transition — but expect conditioning gains, not a
   breakthrough.

4. **The EV finding implies low action leverage, which caps what more shop
   training can buy.** If outcomes are mostly determined by state rather than by
   the marginal shop decision, the shop agent may be near its ceiling given this
   partner. The leverage is then in the hand agent, which points at the
   **harvest → h2** bootstrap step: fine-tuning h1 against the state
   distribution s1 actually induces is the mechanism designed to attack exactly
   the deep-ante conversion gap the survival curve shows.

5. **Cross-iteration comparisons conflate shop and partner improvements.** The
   headline 0.565-vs-0.268 uses different hand partners. Any future claim that
   "sN is better than sN−1" needs the partner held fixed, or it is not a
   statement about the shop agent.

6. **Φ's provenance will keep drifting.** `--phi-checkpoint` currently points at
   `s0_a4_v4`, a critic fitted to a different partner and a different horizon
   than the run consuming it. This gets worse each bootstrap iteration. Either
   re-derive the potential per iteration or accept a small fixed coefficient.

7. **Feature magnitude is unmeasured.** Issue 1's disproof implies large trunk
   output norms (tiny head weights producing sharp logits). Nothing here
   establishes that is a problem, but it was inferred rather than measured, and
   it is the kind of thing that silently affects optimization. A probe over real
   observations — logit spread, legal-action count, feature norm — would close
   it cheaply.

## Process lessons

- **Two of three hypotheses died to a measurement that could have been taken
  first.** The Φ theory was killed by correlating two scalars already present in
  the event file. The entropy theory was killed by instantiating a fresh model
  and reading `.std()`. Both took under a minute; both came after config changes
  and a 1M-step run.
- **A "peak" in a noisy eval curve is the maximum of N draws, not a maximum of
  performance.** Always convert to σ before reading a trend. Selecting a
  checkpoint on that curve institutionalizes the error.
- **Do not assert an initialization scale from memory.** `gain=0.01` says
  nothing about the resulting std without the layer shape.
- **A healthy diagnostic can be misread as a pathology.** High explained
  variance was treated as evidence of a bug when it is normally evidence of a
  working critic. Check what the metric means *when things are fine* before
  building a mechanism around an anomalous-looking value.

---

# The a3 stage (`s1_a3`) — 2026-07-21

Everything below is labelled **[MEASURED]** or **[SPECULATION]**. The split is
the point: this investigation produced four solid measurements, three
withdrawn claims, and three live theories, and conflating those categories is
what cost the a4_v3 investigation two config changes and a run.

## The run

```
--win-ante 3 --s1-schema
--init-from      runs/shop_ppo/s1_a4_v3/best_model/best_model.zip
--init-reservoir runs/shop_ppo/s1_a4_v3/reservoir.pkl
--phi-checkpoint runs/shop_ppo/s0_a4_v4/best_model/best_model.zip
--blend-beta0 0 --phi-beta0 0.1
--hand-policy runs/hand_ppo_b/h1/best_model/best_model.zip --partner-money-ordering
--ent-coef 0.01 --total-timesteps 1000000 --n-envs 8
--eval-freq 25000 --eval-episodes 200 --checkpoint-freq 100000 --seed 0
```

Event file: `s1_a3/…21268.0`, identified as genuine by `shop/phi_beta` maxing
at **0.0998** (matching `--phi-beta0 0.1`).

## Reading the wrong file — how it happened and the check that stops it

**[MEASURED]** The first pass of this analysis was run against
`s1_a3/…21164 (1).0`, a stray with `phi_beta` max **0.998**. Since
`phi_beta = phi_beta0 * progress_remaining` and is logged raw
(`train_shop_ppo.py:141,155`), a run launched with `--phi-beta0 0.1` **cannot**
log 0.998. That single scalar disqualifies the file.

It produced a confident, entirely fictitious analysis: a "0.125 peak vs 0.415
CLI anomaly", a "late collapse reproduced at z = −8.7", and a "1.5× binomial
variance" finding. All three belonged to an unrelated run. The user caught it
by noticing TensorBoard disagreed.

**Verify before analysing any transferred event file:**

```
uv run --with tensorboard python -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E; a=E('PATH'); a.Reload(); print(max(x.value for x in a.Scalars('shop/phi_beta')))"
```

The value must equal the run's `--phi-beta0`. More generally: **pick one logged
scalar that the launch command pins, and check it first.** `phi_beta` is the
natural one here because it is a direct function of a CLI argument.

Correct paths are always `runs/shop_ppo/<stage>/MaskablePPO_1/`. Anything loose
at a run dir's top level was hand-copied and its identity is unknown. The three
stray files (`s1_a3`, `s1_a4_v3`, `s1_a4_WARM`) should be deleted once the real
ones are in place — they are live traps.

## Measured findings

**[MEASURED] The stage learns.** First 10 evals mean **0.284**, last 10
**0.378** (2,000 episodes a side): **6.4σ**. The final block is also the
best block — no terminal collapse.

**[MEASURED] The dip is transient and mid-run, and it damages the count-bonus
hypothesis.** Evals at 625–700k read 0.170 / 0.230 / 0.205 / 0.280 — mean 0.221
over 800 episodes, ~7σ below the run mean of 0.341, with `mean_ep_length`
falling 28.0 → 19.4 (the same short-episode signature as Issue 4's collapse).
It then recovers fully to 0.42+ and stays there. Meanwhile `count_beta` decays
monotonically (0.0499 → 0.00003) and `phi_beta` (0.0998 → 0.00006): both are
still nonzero *during* the dip and keep vanishing through 700k→1M, which is
exactly the window where the policy recovers and posts its best numbers.
**Losing exploration pressure cannot explain a dip that heals while the pressure
keeps falling.** This is the third measurement against the Issue 4 suspect.

**[MEASURED] `--eval-episodes` reaches the callback correctly.** All a3 eval
values are multiples of 0.005 → 200 episodes, as launched. Future-worry #1's
plumbing concern is closed (see the amendment there).

**[MEASURED] The selection lottery reproduces, in the expected direction.**
`best_model` was saved at 975k on a logged **0.490**; honest 200-episode
re-eval of that same checkpoint gives **0.415**. 0.490 sits ~2σ above the
local mean of ~0.38. Issue 2 holds.

**[MEASURED] Eval variance is ~1.9× binomial, and it is policy movement, not
measurement noise.** Binomial sd falls below the χ² 95% CI on the observed sd
in both flat blocks (first10: 0.0319 vs CI [0.0413, 0.1097]; last10: 0.0343 vs
[0.0455, 0.1208]). Since eval is argmax (`deterministic=True`,
`train_shop_ppo.py:953`) and seeds are iid, the excess can only be the policy's
*true* win rate moving: sd ≈ √(0.066² − 0.034²) ≈ **5.7 percentage points per
25k steps**.

**[MEASURED] The variance is NOT exploration-driven.** Correlations of
\|Δeval\| against per-interval training metrics (n=39; \|r\| > 0.32 needed for
significance):

| metric | r |
|---|---|
| `train/approx_kl` | −0.093 |
| `train/clip_fraction` | +0.119 |
| `train/entropy_loss` | −0.170 |
| `shop/count_beta` | +0.027 |

All flat. And detrended excess variance **grows** as the bonus vanishes: evals
0–19 ratio **1.74×**, evals 20–39 ratio **2.62×** — the opposite of what an
exploration-bonus mechanism predicts. Also ruled out: the shared partner
instance (one `HandCheckpointPolicy` serves all 8 training envs *and* the eval
env) holds no mutable per-episode state, so it cannot correlate outcomes.

**[MEASURED] Checkpoint choice is worth as much as hundreds of thousands of
steps.** Six evals of the same policy family at ante 4 on the *identical*
`EVAL_0..199` suite — so zero seed-draw noise between rows:

| checkpoint | ante-4 win | entropy (nats) | mean steps |
|---|---|---|---|
| 600k | 0.210 | 0.0974 | 35.4 |
| 700k | 0.115 | 0.0483 | 28.1 |
| 800k | 0.115 | 0.0099 | 36.7 |
| 900k | 0.145 | 0.0348 | 26.4 |
| 975k (`best_model`) | 0.200 | 0.0816 | 25.7 |
| 1M | 0.160 | 0.1188 | 26.5 |

A 0.095 spread is 19 net episodes; under McNemar that clears significance
unless neighbours disagree on >~90 of 200 seeds, which is implausible 100k
steps apart.

**[MEASURED] `s1_a3`'s honest ante-4 ability is ~0.158, not 0.200.** The six
rows average 0.158; the 0.210 is the max of six draws. **The headline 0.200
from `best_model` was the same lottery statistic one level up.** Any
cross-stage comparison quoting single checkpoints (including this document's
own 0.415/0.200-vs-0.315/0.090) is max-vs-max and should be read as such.

**[MEASURED] Ante-3 CLI evals of `s1_a3/best_model`:** 0.415 at ante 3, 0.200
at ante 4, `n_dead_at_reset` 0 in both.

## Live theories

**[SPECULATION] Pivotal-decision argmax flips.** The surviving explanation for
policy wander with no correlation to update magnitude: shop leverage is
concentrated in a few high-leverage decisions (which joker at ante 1–2, whether
to commit to a risky build) rather than spread evenly. Flipping the argmax at
one such node is a negligible *average* KL over the state distribution — which
is why the correlations are flat — but moves the win rate several points at
once. Consistent with Issue 3's surviving reading (outcomes largely determined
by state) and with the near-deterministic policy (0.01–0.12 nats), which leaves
no stochasticity to average a pivotal flip away. **Untested.** The test:
diff which actions two adjacent checkpoints take on the same states, and check
whether the disagreements concentrate in a small set of high-leverage nodes.

**[SPECULATION] Entropy tracks ante-4 competence.** Across the six sweep rows,
entropy and ante-4 win rate correlate at **r = 0.71** — but n=6, **p ≈ 0.11**,
not significant, and confounded (both could be driven by the same underlying
wander rather than one causing the other). The eye-catcher is 800k: entropy
0.0099, an essentially collapsed policy, tied for worst. Note this is *not*
Issue 1 resurrected — that concerned an inherited collapse at init, this is
oscillation during training. Test: an a4 run at `--ent-coef 0.02` against the
0.01 baseline. **That is a second run, not a config change to slip into the
first.**

**[SPECULATION] The Φ / explained-variance oddity.** a3 (`--phi-beta0 0.1`)
settles at EV **0.823** (min 0.505); the a4_v3-family file with `phi_beta0`
1.0 sits at **0.9985**. Issue 3's disproof rested on a *within-run* correlation
late in that run; this *across-run* comparison points the other way. Confounded
three ways (different horizon, different init, different outcome variance) —
and note that at p ≈ 0.35 return variance is *higher* than at p ≈ 0.10, which
should push EV *up*, not down. **An observation the Issue 3 disproof does not
cover, not a reversal of it.**

## Claims made and withdrawn in this session

Recorded because the failure modes are the same ones the Process Lessons
already name, and they recurred anyway.

1. **"A 0.125-vs-0.415 anomaly in the wrong direction for lottery selection."**
   Withdrawn — read off a stray event file. The lesson isn't "check your files";
   it's that a *coherent, mechanism-laden story* was constructed on top of the
   bad data before anyone checked whether the data was the right data.
2. **"Ante-4 declines while ante-3 stays flat → horizon-3 specialization."**
   Withdrawn before publication. 975k returns to 0.200 and 600k is 0.210 — a
   trough in the middle, not a trend. Six noisy points do not carry a trend, and
   there was a ready-made theory (episodes terminate on clearing ante 3, so
   nothing rewards being well-positioned *at* ante 3) waiting to absorb it.
   **A mechanism that would explain the pattern is not evidence the pattern is
   real.**
3. **"Init a4 from 600k — more headroom for the new horizon."** Withdrawn on
   measurement: entropy at 600k (0.0974) and `best_model` (0.0816) is
   equivalent, and 1M is higher than both. The plasticity premise was simply
   false, and it had been flagged as a judgement call rather than measured
   before being used to make a recommendation.

## Next step (as of 2026-07-21)

Run a4 with `--init-from` **`s1_a3/best_model`**: tied at ante 4 within noise,
holder of the only honest fixed-suite ante-3 number (0.415), and 375k more
training on antes 1–3 — which must be cleared to reach ante 4, so shared-prefix
competence transfers directly. Change only `--win-ante`.

**Pre-registered bar: ≥0.23 at ante 4** (honest baseline 0.158 + 2σ at n=200).
Setting the bar against the 0.210 max would institutionalise the lottery.
Re-verify any headline checkpoint on a second disjoint seed set
(`EVAL_200..399`) before believing it.

If a4 misses that bar, **stop shop training and go to harvest → h2.** Three
independent results now point there: Issue 3's surviving reading, Future-worry
#4, and the pivotal-flip theory. Per Future-worry #2, do not attempt a8.

Optional and cheap: sweep ante **3** over the same five checkpoints. It settles
the init choice on the metric that actually transfers, and `best_model`'s
claim to be best at ante 3 currently rests on one seed set.

Tooling gap: `eval_shop_policy.py` does not dump per-seed outcomes, so paired
(McNemar) tests across checkpoints are impossible and every comparison above is
an eyeball on two rates. A `--dump-episodes` flag writing per-seed win/loss
would fix this permanently.

## Additional process lessons

- **Verify data identity before analysing it, using a scalar the launch command
  pins.** One number (`phi_beta` max) disqualified the file; it was checked
  only after a full analysis had been built and delivered.
- **"Max of N draws" is fractal.** Issue 2 caught it in the eval curve. It
  recurred one level up when a checkpoint sweep's *best row* was quoted as the
  policy's ability. If you selected on it, it is biased — at every level.
- **State whether a claim is measured before using it to recommend anything.**
  The 600k recommendation was built on an explicitly-flagged unproven premise,
  and flagging it did not stop it from driving a decision.

# The fixed-engine retrain — `s1_a2_pr2` (2026-07-21)

**Context.** Everything above (`s1_a2`, `s1_a3`, `s1_a4_v3`/`WARM`) trained on
the **pre-K3 engine** — Four Fingers / Shortcut / Smeared inert, the
`jokers=None` bug, etc. `s1_a2_pr2` is the **first shop stage on the fixed
engine.** Treat the flawed-engine stages as non-baselines: the old `s1_a3`'s
~0.37–0.42 (warm-started from `s1_a4_v3`, flawed engine) is **not** a rung for
this chain and must not be reused. Its ~30–40% coincidentally overlaps the pr2
zero-shot a3 number below — do not conflate them.

The run:

```
--win-ante 2 --s1-schema
--init-from      runs/shop_ppo/s0_a4_v4/best_model/best_model.zip
--phi-checkpoint runs/shop_ppo/s0_a4_v4/best_model/best_model.zip
--blend-beta0 0 --phi-beta0 0.1 --init-temperature 5.0
--hand-policy    runs/hand_ppo_b/h1/best_model/best_model.zip --partner-money-ordering
--ent-coef 0.01 --total-timesteps 1000000 --n-envs 8
--eval-freq 25000 --eval-episodes 200 --checkpoint-freq 100000
--log-dir runs/shop_ppo/s1_a2_pr2 --seed 0
```

No `--init-reservoir` — pr2 seeds a fresh reservoir off the s0 warm start.

> **PROVENANCE (consistent, not confirmed — flagged per this doc's own rule).**
> The pr2 numbers were read from a **loose top-level** event file `…4184.0`
> (originally in a mis-named `s1_a2_sr2/` dir; the local dir was corrected to
> `s1_a2_pr2/` on 2026-07-21), with **no `MaskablePPO_1/` subdir** beside it —
> the hand-copied-stray pattern the a4_v3/a3 sections flag. The pinning scalar
> is consistent: `shop/phi_beta` maxes at 0.100, matching `--phi-beta0 0.1`. The
> genuine file is `runs/shop_ppo/s1_a2_pr2/MaskablePPO_1/` on the training
> machine — locate it and re-check `phi_beta` before treating anything below as
> settled.

## Measured

**[MEASURED] a2 saturated early; the back half only sharpened.** Warm-started
from `s0_a4_v4`, win rate plateaued by ~75k and never moved for the remaining
~900k (wandered 0.57–0.72, no trend). `entropy_loss` fell −0.27 → −0.019 over
the run. This is Issue 1 restated with the real finding foregrounded: not a
collapse but a **converged policy sharpening on a saturated objective.** The
`eval/mean_reward` trough at 800k (0.555) is small-sample wobble — the user's
200-ep checkpoint sweep (300k–1M) holds 0.57–0.65 throughout, no catastrophic
break.

**[MEASURED] best_model = step 575,000.** Unique peak of `eval/mean_reward`
(0.72; EvalCallback saves on new-max, so the step is read off the curve, not
the file). `entropy_loss` there ≈ **−0.08** vs −0.019 at 1M — ~4× the entropy
of the endpoint, and it predates the whole back-half sharpening. 200-ep eval
**0.65**. It is the better warm-start on both axes (win rate *and* remaining
entropy); the 1M final is a hair worse (over-sharpened, within noise).

**[MEASURED] The 702k KL spike is the reporting artifact, not a real step.**
`approx_kl` = 0.63 at 702,464 (~15× the ~0.04 baseline), no numeric blowup,
`explained_variance` recovers to 0.90+ within ~10k steps. Same stale-probs-cache
signature already documented (`shop-ppo-nan-grad-guard`). It is at 702k, not
800k — the eval trough lags it by ~100k.

**[MEASURED] Shop is the entire lever at a2.** `nextround` floor (passive shop,
always advance) with the h1 partner = **0.000** win rate, dying at mean ante
1.38 — the base deck + h1 cannot clear ante 1 unaided. Trained-0.65 is *all*
shop value. Kills any "shop-irrelevant at a2" reading.

**[MEASURED] Zero-shot horizon density of the a2 best_model** (200 ep each,
`--win-ante N`, h1 partner, `--partner-money-ordering`; the a2 policy evaluated
with the finish line moved deeper, no extra training):

| finish line | win rate |
|---|---|
| ante 2 (trained) | 0.65 |
| ante 3 | 0.37 |
| ante 4 | 0.09 |

a2→a3 is a **learnable step** — 0.37 is dense enough to bootstrap PPO
immediately. a2→a4 is a **cliff**: 0.09 is below the density where PPO explores
effectively, so training a4 straight from a2 fights sparse-reward exploration
from a peaked prior. That is the a4=0.09 failure of the earlier sections, seen
here at its source (the policy simply never learned the economy/scaling ante 4
demands).

## The plan it implies

**[SPECULATION, but measurement-gated] Rebuild the curriculum chain on
best_model links, softened, with an inserted a3 rung.**

1. Chain `best_model → best_model`, never off a collapsed final. Soften at each
   hop (pr2 used `--init-temperature 5.0` from s0; keep it high to err toward
   exploration into the new deeper states rather than inheriting a3/a4's
   collapse-chaining).
2. Insert an **a3 rung** between a2 and a4 (the density gradient says a2→a4 is
   too big for this partner). Re-run it **on the fixed engine** from
   `s1_a2_pr2/best_model` — the flawed-engine `s1_a3` cannot serve.
3. **zero-shot-density-gates-the-next-rung.** After a rung trains, eval its
   best_model zero-shot at the next horizon *before* launching there:
   **≥ ~0.2 → trainable; ~0.1 → insert another rung** instead of throwing PPO at
   a sparse objective. This is the nextround-floor discriminator applied to
   curriculum spacing — measure the density each rung hands the next, don't
   guess it.

**Mirror pr2's full flag set; change only `--win-ante`, `--init-from`,
`--init-reservoir`, `--log-dir`.** A stripped command silently flips the
objective and the eval quality: dropping `--phi-checkpoint`/`--blend-beta0 0`
reverts to the c_ante blend bonus (blend-beta0 defaults to 1 without Φ), and
dropping `--eval-episodes` reverts to 50-episode evals (default) — 4× noisier
best_model selection, the exact lottery this doc keeps hitting. a3 command:

```
uv run python scripts/train_shop_ppo.py --win-ante 3 --s1-schema \
  --init-from      runs/shop_ppo/s1_a2_pr2/best_model/best_model.zip \
  --init-reservoir runs/shop_ppo/s1_a2_pr2/reservoir.pkl \
  --phi-checkpoint runs/shop_ppo/s0_a4_v4/best_model/best_model.zip \
  --blend-beta0 0 --phi-beta0 0.1 --init-temperature 5.0 \
  --hand-policy runs/hand_ppo_b/h1/best_model/best_model.zip --partner-money-ordering \
  --ent-coef 0.01 --total-timesteps 1000000 --n-envs 8 \
  --eval-freq 25000 --eval-episodes 200 --checkpoint-freq 100000 \
  --log-dir runs/shop_ppo/s1_a3_pr2 --seed 0
```

(`--init-reservoir` is the one addition vs pr2, carrying pr2's snapshot
diversity forward — confirm `s1_a2_pr2/reservoir.pkl` exists, else omit for a
fresh one.) Then eval `s1_a3_pr2/best_model` at `--win-ante 4`; that number
decides whether a4 launches from it or earns another rung first.

# The a3 rung ran — `s1_a3_pr2` (2026-07-21)

The rung above was run. **It was launched with `--init-temperature 3.0`, not
the 5.0 the command block specifies** — note the discrepancy before reusing any
of this to reason about the softening.

> **PROVENANCE.** Read from the loose top-level `…24624.0` (no `MaskablePPO_1/`
> subdir — the hand-copy pattern), but **verified genuine**: `shop/phi_beta`
> maxes at 0.0998, matching `--phi-beta0 0.1`. The CLI evals below are on
> `EVAL_0..199`, disjoint from the callback's consumed range.

## Verdict — the rung works; launch a4, but confirm the gate first

**[MEASURED] The a3 rung converted the a2→a4 cliff into a trainable step.**
Zero-shot ante-4 density (200 ep, h1 partner, `--partner-money-ordering`):
`s1_a2_pr2/best_model` = **0.09** (the cliff — sparse, PPO can't explore off it);
`s1_a3_pr2/best_model` = **0.22**. The a2_pr2 density gate (≥~0.2 trainable,
~0.1 insert-a-rung) is cleared, and 0.22 beats even the flawed-engine a3's
honest ~0.158. This is the density gradient working exactly as the a2_pr2 plan
designed it.

**Caveat that must travel with 0.22:** it is only **~0.7σ over the 0.2 line**
(n=200 sd ≈ 0.029) and is read off `best_model` — the lottery-selected 950k
peak. Before burning a 1M-step a4 run, re-verify on `EVAL_200..399` **and** eval
900k (a non-lottery neighbor) at ante 4; launch a4 only if both hold ≥~0.2. If
they sag to ~0.15 the cliff is not really cleared and harvest→h2 is the better
call (the EV/low-leverage findings already lean there).

## Measured findings

**[MEASURED] The stage learns, then plateaus flat — more steps won't help.**
`eval/mean_reward` (40 evals): front half (<500k) mean **0.346**, back half
(≥600k) mean **0.422**, but the back-half trend slope is **−0.002 per 100k
steps** (n=17) — statistically zero. Logged peak **0.500 at 950k**
(`best_model`); the very next points fall back (975k 0.435, **1M 0.370**). 950k
is the top of a ~0.37–0.50 oscillation, not the tip of a rising curve. Extending
a3 buys more wander and a fresh lottery peak, not a higher plateau.

**[MEASURED] Honest ante-3 ability ≈ 0.46–0.47, not the 0.525 headline.**
`best_model` CLI re-eval: **0.525** at ante 3 (`n_dead_at_reset` 0). Notably it
did **not** regress down this time (logged 0.500 → honest 0.525) — the opposite
of a4_v3's 0.200→0.090. But 0.525 is only ~1.5σ over the back-half cluster
(sweep 800k/900k/1M = 0.47/0.465/0.41), so it is consistent with a ~0.47 truth,
not a separate tier. Treat honest a3 ante-3 ability as **~0.46–0.47**; 0.525 is
still a single-checkpoint headline (max-of-N is fractal — Issue 2's lesson).
Ante-3 checkpoint sweep (300k→1M): 0.35 / 0.36 / 0.39 / 0.37 / 0.335 / 0.47 /
0.465 / 0.41 — flat first half, real +0.10 back-half lift.

**[MEASURED] The early sag is the softening tax, and it cost ~300–500k steps.**
a3 was handed a policy already at **0.37 zero-shot at ante 3**, yet dipped to
**0.295 / 0.295 / 0.290** at 100–275k — *below its own warm-start density* —
before climbing past it by ~700k. `--init-temperature 3.0` raised init
`norm_entropy` to **0.44 at step 2048** (so temperature does move entropy at
T=3, contra one Issue-1 worry), but PPO consumed it within ~50k steps. Half the
1M budget went to undoing the flattening. Confounded with plain ante-3 horizon
adaptation — can't pin the whole sag on temperature — but the softening's
mechanical cost is visible and its benefit is not (see next).

## Live theories re-examined against this run

**[MEASURED, against Φ/exploration being the culprit] The wander is not
exploration-driven.** `|Δeval|` vs interval metrics (n=39): `count_beta`
+0.090, `phi_beta` +0.090, `entropy_loss` −0.082, `clip_fraction` +0.036 — all
flat; only `approx_kl` is non-flat at **−0.352** (bigger eval swings on
*smaller* KL updates). That negative-KL signature is the **pivotal-flip**
theory, not lost exploration pressure. And performance is *best* in the back
half where both bonuses have decayed to ~0 (at 950k `count_beta` 0.003,
`phi_beta` 0.005) — if shaping were harmful, removing it would hurt; it doesn't.
Fourth independent measurement against the Issue-4 count-bonus suspect.

**[MEASURED, against raising `--ent-coef`] Entropy and competence are decoupled
within this run.** eval level vs `norm_entropy`: r = **−0.50** over all 40 evals
(confounded by the softening recovery), r = **−0.10** in the clean back half
(≥500k, n=21). Higher entropy buys no win rate here — the opposite sign to the
a3-flawed sweep's confounded `r=+0.71` (n=6, p≈0.11), and on far more data. Do
**not** raise `ent-coef` as a default and do not slip it into the a4 launch (it
confounds the launch/no-launch read). If the entropy-competence question is
worth settling it is a dedicated 0.01-vs-0.02 A/B run, ranked below launching a4
and below harvest→h2.

**[SPECULATION → now better-supported] Temperature is not needed; drop it going
forward.** Its injected exploration (0.44 entropy early) coincided with the
*worst* eval (the sag); gains came after re-sharpening (best = lowest-entropy
950k). Measured recovery tax, unmeasured/absent benefit, and you already handle
collapse-chaining by chaining off `best_model` (softer) rather than the
over-sharpened final. Issue 1's "addressed a non-problem" disposition stands.
Escape hatch: read the first ~20k steps of the a4 run — only if `norm_entropy`
is pinned near a3's ending ~0.19 **and won't move** under `ent-coef` is
temperature (argsort-preserving) the right cheap fix.

## Next step

Launch a4 from `s1_a3_pr2/best_model`, **dropping `--init-temperature`**,
mirroring the full pr2 flag set (change only `--win-ante 4`, `--init-from`,
`--init-reservoir`, `--log-dir runs/shop_ppo/s1_a4_pr2`) — **after** the two
ante-4 confirmation evals above clear ≥~0.2. Keep `--phi-checkpoint` /
`--blend-beta0 0` / `--phi-beta0 0.1` / `--eval-episodes 200`; dropping any
silently reverts the objective or re-noises best_model selection. If the gate
fails or a4 plateaus flat like a3, stop shop training and go to harvest→h2 — the
EV/low-leverage finding, Future-worry #4, and the pivotal-flip signature all
point there.

# The a4 rung ran — `s1_a4_pr2` (2026-07-21)

The a4 rung was launched from `s1_a3_pr2/best_model` per the plan above
(dropping `--init-temperature`). **It failed — the a2→a4 cliff the a3_pr2 gate
pre-registered is real, and this is the harvest→h2 trigger firing.**

> **PROVENANCE — genuine, not a stray.** Read from
> `runs/shop_ppo/s1_a4_pr2/MaskablePPO_1/…7100.0` (a proper subdir, not a
> hand-copied top-level file). `shop/phi_beta` maxes at **0.0998**, matching
> `--phi-beta0 0.1`. Ran 1.2M steps (checkpoints to 1.2M).

## Measured

**[MEASURED] a4 PPO walked the policy *downhill* from its warm start and never
recovered.** `eval/mean_reward` (48 evals, 200 ep each — values are multiples of
0.005):

| step | eval win |
|---|---|
| 25k | **0.225** ← `best_model`, and the max of all 48 evals |
| 100k | 0.140 |
| 200k | 0.070 |
| ~flat rest | ~0.12–0.13 |
| mean after 100k | **0.126** |

`best_model` is the **25k checkpoint** — i.e. the warm start with almost no a4
training on it. The 0.225 first eval is just the inherited a3_pr2 zero-shot ante-4
density (0.22) showing through before training corrupts it. Every one of the
1.2M steps made the policy worse. Extending past 1M (to 1.2M) bought nothing.

**[MEASURED] The mechanistic tell — entropy *inflates*, the opposite of a3.**
`rollout/normalized_entropy` **0.19 → 0.44** over the run (a3_pr2 *consumed*
entropy and sharpened; this diffuses); `entropy_loss` −0.166 → −0.183 (more
entropy); `mean_ep_length` 32 → 26 (dying earlier); `explained_variance` dipped
to 0.05 before recovering to 0.69. Init `normalized_entropy` was 0.187 (≈ a3's
ending ~0.19), confirming `--init-temperature` was correctly dropped — the
diffusion is NOT inherited softening.

The signature is **reward-signal starvation, not exploration starvation.** At
ante 4 with terminal-only `1{won}` reward at ~0.1 density there is no consistent
gradient for the policy to become confident about, so the `ent-coef 0.01` bonus
dominates a near-flat reward landscape and pushes the policy toward uniform; as
it drifts off the good warm-start argmax, the win rate erodes toward the floor.
This is entropy *inflation* on a rewardless plateau — not Issue 1's inherited
collapse, and not a knob the temperature could have fixed (more entropy is the
disease here, not the cure).

**[MEASURED] The collapsed policy churns.** First trace (5 ep, new tooling
below): **46 BuyCard vs 37 SellJoker** — buy-then-dump joker churn — 0 wins.
Behaviorally consistent with a diffused policy taking near-random legal actions.

**[MEASURED, separate finding — DIAGNOSED] The eval Simplex crash is a
library false-positive, NOT policy degeneracy.** `policy.predict` raised
`ValueError: ... probs ... Simplex() ... invalid values` on some `EVAL_` seeds.
Root cause is the sb3-contrib `MaskableCategorical` stale-probs-cache bug (see
`shop-ppo-nan-grad`, layer 4), and `eval_shop_policy.py` simply never installed
the guard that `train_shop_ppo.py` does — so the in-training eval callback
survived while the standalone CLI eval crashed. PROVEN: on the failing step the
raw `action_net` logits were finite (min −10.4, max 16.7), and with the guard
installed the seed replays to completion and in fact **wins**, with the guard's
genuine-non-finite catch counter at **0**. So this crash is orthogonal to the a4
collapse and is NOT evidence of numerical degeneracy — the policy is healthy on
that state. Fixed by extracting the guard into `jackdaw/env/maskable_guard.py`
and installing it on the eval checkpoint path.

## Tooling added this session

`scripts/eval_shop_policy.py --dump-decisions <path>` writes a full per-decision
JSONL trace (state at the decision, decoded action family/slot/label, legal set,
terminal/won). Opt-in; aggregate metrics unchanged when absent. This is how the
buy/sell churn above was read, and it is the instrument for actually *seeing*
what a collapsed policy does rather than inferring it from scalar curves.

## Verdict / next step

The a2→a4 cliff was not cleared: the 0.22 zero-shot the a3_pr2 gate passed (only
~0.7σ over the 0.2 line, read off a lottery `best_model`) does not survive honest
a4 training. Three independent results now point the same way — Issue 3's
low-leverage EV reading, Future-worry #4, and the pivotal-flip signature — so
**stop shop rungs and go to harvest → h2.** Per Future-worry #2, do not attempt
a8. A temperature-3.0 a4 rerun was started to sanity-check the softening angle,
but the diagnosis says temperature is orthogonal to a reward-density failure;
expect it to diffuse the same way (watch whether its eval climbs *above* the
~0.22 warm-start line or decays while `normalized_entropy` inflates past ~0.3).
