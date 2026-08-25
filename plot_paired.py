"""The money plot: paired improvement over base with bootstrap CIs, at n=256."""
import json, math, os, random
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
random.seed(0); B = 4000

def pak(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

def pp(arm, lvl=4):
    p = f"evals256/{arm}.jsonl"
    if not os.path.exists(p): return None
    r = [json.loads(l) for l in open(p) if l.strip()]
    r = [x for x in r if x.get("level", 0) >= lvl]
    if not r: return None
    return {x["idx"]: {k: pak(x["n"], x["c"], k) for k in KS} for x in r}

base = pp("base")
ARMS = [("gatecap_1e-5", "GATE (solved-prompt gating)", "tab:blue"),
        ("fix_maxrl_1e-5", "MaxRL lr 1e-5", "tab:purple"),
        ("long_grpo", "GRPO 300 steps (2.5x budget)", "tab:green")]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
# left: absolute curves
xs = KS
b_abs = [sum(base[i][k] for i in base) / len(base) for k in KS]
ax[0].plot(xs, b_abs, color="#444", lw=3, marker="o", ms=4, label="base (no training)")
for arm, lab, c in ARMS:
    a = pp(arm)
    if not a: continue
    m = [sum(a[i][k] for i in a) / len(a) for k in KS]
    ax[0].plot(xs, m, color=c, lw=2, marker="o", ms=4, label=lab)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k")
ax[0].set_title("MATH-500 level>=4, n=256 samples/problem"); ax[0].grid(alpha=.3)
ax[0].legend(fontsize=8.5, loc="upper left")

# right: paired delta with CI
for arm, lab, c in ARMS:
    a = pp(arm)
    if not a: continue
    common = [i for i in base if i in a]
    mus, los, his = [], [], []
    for k in KS:
        d = [a[i][k] - base[i][k] for i in common]
        mu = sum(d) / len(d)
        bo = sorted(sum(d[random.randrange(len(d))] for _ in range(len(d))) / len(d)
                    for _ in range(B))
        mus.append(mu * 100); los.append(bo[int(.025 * B)] * 100); his.append(bo[int(.975 * B)] * 100)
    ax[1].plot(xs, mus, color=c, lw=2, marker="o", ms=4, label=lab)
    ax[1].fill_between(xs, los, his, color=c, alpha=.15)
ax[1].axhline(0, color="#444", lw=1.5, ls="-")
ax[1].set_xscale("log", base=2); ax[1].set_xlabel("k")
ax[1].set_ylabel("pass@k improvement over base (points)")
ax[1].set_title("paired bootstrap, 95% CI — shading above zero = real")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8.5)
fig.suptitle("Only solved-prompt gating improves the curve significantly past k=64", fontsize=12.5)
fig.tight_layout()
fig.savefig("results/figE_paired.png", dpi=140)
print("wrote results/figE_paired.png")
