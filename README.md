# pass@k frontier

Can RL raise **pass@1 and pass@k together**, and beat best-of-n distillation at a
**matched rollout budget**? ~60 training runs on MATH-500 with Qwen2.5-1.5B-Instruct.

## Headline

Corrected **MaxRL at lr 1e-5** is the only arm that is genuinely Pareto: it beats the base
model at every k from 1 to 32 and **holds pass@64 at base level**, where GRPO and BonBon
both pay ~1 point at k=64 for their pass@1 gain.

| arm | pass@1 | pass@64 |
|---|---|---|
| **fix_maxrl_1e-5** | **0.502** | **0.882** |
| hi_grpo (best GRPO) | 0.446 | 0.872 |
| bonbon_e6_3e-6 (best BonBon) | 0.445 | 0.872 |
| base | 0.415 | 0.882 |
| hi_maxrl (un-baselined MaxRL) | 0.028 | 0.482 |

n=64 samples, all 500 MATH-500 problems, unbiased pass@k estimator (Chen et al. 2021).

## Three findings worth the compute

**1. The MaxRL advantage must be baselined.** `A_i = r_i·G/K` (un-baselined) gives pass@1
0.028; `A_i = (r_i − mean)/(mean + ε)` gives 0.502. Same objective in expectation — since
`E[S] = 0`, subtracting the baseline does not change `∇log p` — but only the baselined form
has a negative gradient on failures, so the group sums to zero and the update is
self-anchoring. An 18x difference from one term.

**2. Learning rate is not shared across objectives.** GRPO peaks at 3e-6, MaxRL at 1e-5,
BonBon at 3e-6. At a single shared lr (the natural "controlled" choice) MaxRL looks broken.
Every method needs its own sweep or the comparison measures lr sensitivity.

**3. Best-of-n distillation is data-limited, not compute-limited.** BonBon at 6 epochs
(≈isoflop with GRPO) scores 0.445 vs 0.437 at 1 epoch: **+0.8 points for 6x the training
FLOPs**. Its ceiling is best-of-8 of the frozen base, and Qwen2.5-Instruct was already
rejection-sampled on math during post-training.

## Layout

```
train.py       one rollout loop, 15 objectives; arms differ ONLY in advantage()
train_bon.py   BonBon (arXiv 2406.00832) and BOND / J-BOND (arXiv 2407.14622) from the papers
data.py        MATH loading, boxed extraction, math_verify reward
log.py         rollout dump, fixed probe set, per-step diversity metrics
evalk.py       vLLM pass@k eval, n=64 on MATH-500
aggregate.py   unbiased pass@k, level-stratified
plot_all.py    every figure;  build_report.py  self-contained HTML report
runs/<arm>/    args.json, rollouts.jsonl (every sampled completion), diversity.jsonl,
               passk_probe.jsonl, probe/step*.jsonl (fixed probe, same seed across arms)
evals/         {idx, level, n, c} per problem, per arm
results/       aggregated pass@k JSON + figures
```

Model checkpoints (`runs/*/final/`) are excluded — ~65 x 3 GB, regenerable from the code.
Rollouts and probe dumps are in Git LFS.

## Objectives

For a group of G=8 rollouts on one prompt, binary rewards `r_i`, `p = mean(r)`, `K = Σr_i`:

| name | advantage | note |
|---|---|---|
| `grpo` | `(r_i − p)/(std + ε)` | standard RLVR |
| `maxrl` | `(r_i − p)/(p + ε)` | J = log p; GRPO with mean, not std, in the denominator |
| `maxrl_nb` | `r_i · G/K` | un-baselined; ablation showing what the baseline buys |
| `raft` | `r_i` | SFT on correct = vanilla REINFORCE, no baseline |
| `drgrpo` | `r_i − p` | REINFORCE with baseline; λ=0 of the family below |
| `powmaxrl` | `(r_i − p)/p^λ` | α-fair family, `J_λ = p^(1−λ)/(1−λ)` |
| `geopassk` | `(r_i − p)·c/(c + g·p)²` | geometric weights over k; bounded at p=0 |
| `difkl` | `p^κ·A_grpo + (1−p^κ)·A_fwd` | sharpen easy prompts, cover hard ones |
| `gatehard` | grpo, zero gradient if `K/G ≥ 0.5` | hard prompts only; K=0 groups resampled |
| `entropic` | `e^{βr_i} − mean(e^{βr})` | risk-seeking |
| `hientropy` | grpo on top-20% entropy tokens | "forking tokens" |

The SFT ladder is one flag: `--flush_every` with `--objective raft` gives fully on-policy
RAFT (1), iterated ReST (30), or offline BoN distillation (120, policy frozen throughout).

## Reproducing

```bash
python train.py --objective maxrl --lr 1e-5 --steps 120 --out_dir runs/maxrl_1e-5
python evalk.py --model runs/maxrl_1e-5/final --out evals/maxrl_1e-5.jsonl --n 64 --greedy
python aggregate.py --glob "evals/*.jsonl" --min_level 4
```

Every arm sees exactly 15,360 rollouts and 120 optimizer steps of batch 128, asserted at
exit — matched on both axes.
