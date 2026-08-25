# pass@k frontier

Can RL raise **pass@1 and pass@k together**, and beat best-of-n distillation at a **matched
rollout budget**? ~70 training runs on MATH-500 with Qwen2.5-1.5B-Instruct, evaluated with
an unbiased pass@k estimator and paired bootstrap CIs.

## Headline

**Gating out already-solved prompts is the only method that provably improves the whole
curve.** Zero the gradient on any prompt the model already solves ≥50% of the time,
resample the ones it never solves, cap the run at the same sample budget everyone else got.

Paired bootstrap vs base, n=256 samples/problem, MATH-500 level ≥4 (262 problems).
`*` = 95% CI excludes zero.

| arm | rollouts | Δp@1 | Δp@16 | Δp@64 | Δp@128 | Δp@256 |
|---|---|---|---|---|---|---|
| **gate (solved-prompt gating)** | 15.4k | **+7.0*** | **+4.7*** | **+3.5*** | **+2.8*** | +0.8 |
| MaxRL, lr 1e-5 | 15.4k | **+7.6*** | +4.2* | +1.9 | +1.4 | +0.8 |
| GRPO, 300 steps | 38.4k | +5.9* | +4.2* | +2.7* | +1.8 | +0.4 |

Base: p@1 24.9, p@16 63.4, p@64 77.4, p@256 87.8.

MaxRL edges pass@1 (overlapping CIs); the gate is the only arm still significant at k=128,
and it does it on 40% of the 300-step GRPO budget.

## What did not reproduce

**The pass@1-up / pass@k-down tradeoff.** It appeared at 120 steps and vanished with longer
training and adequate eval samples. Across ~70 runs, every statistically significant pass@k
effect was *positive*. At n=64 nothing was significant at k=64 at all — that was an eval
resolution limit, not a null result, and it took n=256 to see the real effects.

## Three things that cost real compute to learn

**1. The MaxRL advantage must be baselined.** `A_i = r_i·G/K` (un-baselined) gives pass@1
0.013; `A_i = (r_i − mean)/(mean + ε)` gives 0.325. Identical in expectation — `E[S] = 0`,
so the baseline does not change `∇log p` — but only the baselined form puts a negative
gradient on failures, making the group sum to zero and the update self-anchoring. 18x.

**2. Learning rate is not shared across objectives.** GRPO peaks at 3e-6, MaxRL at 1e-5,
BonBon at 3e-6. Holding lr fixed across arms feels like the controlled comparison and is
actually a measurement of lr sensitivity. At a shared 1e-6, MaxRL looks broken.

**3. Evaluate the best checkpoint, not the last.** 27 of 56 runs peaked >1.5 points before
their final step. Only saving `final/` silently reports the post-peak state. `--save_every`
now tracks a `best/` checkpoint by probe pass@1.

## Method comparison

| family | best result | why |
|---|---|---|
| solved-prompt gating | **+7.0 p@1, +2.8 p@128** | gradient only from unsolved prompts |
| MaxRL (baselined) | +7.6 p@1, p@64 n.s. | `1/p` hard-prompt upweighting |
| α-fair `(r−p)/p^λ` | +6.6 (λ=1.5) | λ=1 (MaxRL) is near-optimal; λ=2 is worse |
| geometric pass@k | +6.5 | bounded weight `c/(c+gp)²`, no `1/p` blowup |
| GRPO | +2.8 at 15.4k, +5.9 at 38.4k | |
| REINFORCE + baseline | +4.1 | λ=0 of the α-fair family |
| BonBon (from the paper) | +2.5 | offline; ceiling is best-of-8 of a frozen base |
| RAFT / REINFORCE no baseline | never beats base | positive-only, no anchoring term |
| BOND / J-BOND | flat or collapse | fixed anchor; binary reward fires the n=2 signal <6% |

Why best-of-n distillation underperforms here: BoN's optimality assumes a reward with a
strict ordering and a meaningful tail. Binary correctness gives neither — best-of-8 is a tie
among all correct samples. BoN also sits a bounded `log n − (n−1)/n` ≈ 1.2 nats from the
reference at n=8, while online RL has no such bound. And Qwen2.5-**Instruct** was already
rejection-sampled on math, so there is little left to distill.

## Layout

```
train.py       one rollout loop, 15 objectives; arms differ ONLY in advantage()
train_bon.py   BonBon (arXiv 2406.00832) and BOND / J-BOND (arXiv 2407.14622) from the papers
data.py        MATH loading, boxed extraction, math_verify reward
log.py         rollout dump, fixed probe set, per-step diversity metrics
evalk.py       vLLM pass@k eval        aggregate.py  unbiased pass@k, level-stratified
paired.py      paired bootstrap CIs vs base   plot_*.py  figures
runs/<arm>/    args.json, rollouts.jsonl (every sampled completion), diversity.jsonl,
               passk_probe.jsonl, probe/step*.jsonl (fixed probe, same seed across arms)
evals/         n=64 counts      evals256/  n=256 counts      results/  aggregates + figures
```

Checkpoints excluded (regenerable); rollouts in Git LFS. Cluster paths and hostnames are
`PF_*` environment variables.

## Objectives

Group of G=8 rollouts, binary `r_i`, `p = mean(r)`, `K = Σr_i`:

| name | advantage |
|---|---|
| `grpo` | `(r_i − p)/(std + ε)` |
| `maxrl` | `(r_i − p)/(p + ε)` — ascends `J = log p` |
| `maxrl_nb` | `r_i · G/K` — un-baselined ablation |
| `powmaxrl` | `(r_i − p)/p^λ` — α-fair, `J_λ = p^(1−λ)/(1−λ)` |
| `geopassk` | `(r_i − p)·c/(c + gp)²`, `c = 1/k_eff` — geometric weights over k |
| `gatehard` | grpo, zero gradient if `K/G ≥ 0.5`; K=0 groups resampled |
| `raft` | `r_i` — SFT on correct = vanilla REINFORCE, no baseline |
| `drgrpo` | `r_i − p` — REINFORCE with baseline (λ=0) |
| `entropic` | `e^{βr_i} − mean` |
| `hientropy` | grpo on the top-20% entropy tokens |

`--gate_mode {mask,drop}` decides whether gated groups stay in the loss denominator (an
implicit lr decay) or are removed. `--max_rollouts` caps the budget so gating's resampling
does not buy extra samples. `--flush_every` turns `raft` into on-policy RAFT (1), iterated
ReST (30), or offline BoN distillation (120).

## Reproducing

```bash
export PF_HF_HOME=... PF_ROOT=...
python train.py --objective gatehard --gate_mode drop --lr 1e-5 --steps 200 \
                --max_rollouts 15360 --save_every 20 --out_dir runs/gate
python evalk.py --model runs/gate/final --out evals/gate.jsonl --n 256
python paired.py
```

Every non-gated arm sees exactly 15,360 rollouts and 120 optimizer steps of batch 128,
asserted at exit.
