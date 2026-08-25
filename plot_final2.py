"""Final frontier figure with the full arm set, including the gate arms."""
import json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [1, 2, 4, 8, 16, 32, 64]

def pak(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

def curve(arm, lvl=4):
    p = f"evals/{arm}.jsonl"
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    if lvl:
        recs = [r for r in recs if r.get("level", 0) >= lvl]
    if not recs:
        return None
    m, e = [], []
    for k in KS:
        v = [pak(r["n"], r["c"], k) for r in recs]
        mu = sum(v) / len(v)
        var = sum((x - mu) ** 2 for x in v) / max(len(v) - 1, 1)
        m.append(mu); e.append((var / len(v)) ** 0.5)
    return m, e

HEAD = [
    ("base",            "base (no training)",        "#444",       "-",  3.0),
    ("hi_grpo",         "GRPO  15.4k",               "tab:green",  "-",  1.5),
    ("long_grpo",       "GRPO  38.4k (300 steps)",   "tab:green",  "--", 2.0),
    ("fix_maxrl_1e-5",  "MaxRL lr 1e-5  15.4k",      "tab:purple", "-",  2.0),
    ("gatecap_1e-5",    "GATE sample-matched  15.4k", "tab:blue",  "-",  3.0),
    ("gatehard_1e-5",   "gate  18k",                 "tab:blue",   "--", 1.5),
    ("pow15_1e-5",      "alpha-fair lam=1.5",        "tab:brown",  "-",  1.4),
    ("geo32_1e-5",      "geometric pass@k k_eff=32", "tab:cyan",   "-",  1.4),
    ("reinforce_b_1e-5","REINFORCE + baseline",      "tab:olive",  "-",  1.2),
    ("bonbon_e6_3e-6",  "BonBon best",               "tab:orange", "-",  1.4),
    ("fix_raft_kl",     "RAFT + KL anchor",          "tab:pink",   "-",  1.2),
]

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
b = curve("base")
for arm, lab, c, ls, lw in HEAD:
    r = curve(arm)
    if not r:
        continue
    m, e = r
    ax[0].errorbar(KS, m, yerr=e, marker="o", ms=3, lw=lw, ls=ls, color=c, label=lab, capsize=1.5)
    ax[1].errorbar(m[0], m[-1], xerr=e[0], yerr=e[-1], marker="o", ms=10 if lw >= 3 else 7,
                   color=c, capsize=2, zorder=5 if lw >= 3 else 3)
    ax[1].annotate(lab.split("  ")[0], (m[0], m[-1]), fontsize=7.5, xytext=(6, 4),
                   textcoords="offset points", color=c)
ax[1].axvline(b[0][0], color="#444", ls=":", lw=1, alpha=.6)
ax[1].axhline(b[0][-1], color="#444", ls=":", lw=1, alpha=.6)
ax[1].fill_between([b[0][0], 0.34], b[0][-1], 0.83, color="tab:green", alpha=.06)
ax[1].text(0.328, 0.826, "Pareto-positive\n(beats base at k=1 and k=64)", fontsize=7,
           color="tab:green", ha="right", va="top")
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k")
ax[0].set_title("MATH-500 level>=4 (262 problems, n=64)"); ax[0].grid(alpha=.3)
ax[0].legend(fontsize=7, loc="upper left")
ax[1].set_xlabel("pass@1"); ax[1].set_ylabel("pass@64")
ax[1].set_title("the frontier: only the gate and long-GRPO clear base on both axes")
ax[1].grid(alpha=.3)
fig.tight_layout()
fig.savefig("results/figD_final.png", dpi=140)
print("wrote results/figD_final.png")
