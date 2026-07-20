# s1 training hiccups — diagnoses, measurements, and two disproven theories

Investigation record for the `s1_a4_v3` run (2026-07-20). Three hypotheses were
raised to explain why the a4 stage plateaued; **two were disproven by
measurement and one was confirmed.** The disproofs are the load-bearing part of
this document — both wrong theories were plausible, both drove real config
changes, and both cost a training run before the measurement that killed them.

Read the "Future worries" section even if nothing here looks relevant: the
checkpoint-selection finding contaminates the whole bootstrap chain, not just
this run.

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
