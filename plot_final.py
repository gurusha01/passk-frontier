"""Final figures: the frontier, the lr response, the long-horizon result."""
import glob, json, math, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "."; OUT = "results"; KS = [1, 2, 4, 8, 16, 32, 64]

def pak(n, c, k): return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

def curve(arm, lvl=4):
    p = f"{ROOT}/evals/{arm}.jsonl"
    if not os.path.exists(p): return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    if lvl: recs = [r for r in recs if r.get("level", 0) >= lvl]
    if not recs: return None
    m, e = [], []
    for k in KS:
        v = [pak(r["n"], r["c"], k) for r in recs]
        mu = sum(v)/len(v); var = sum((x-mu)**2 for x in v)/max(len(v)-1, 1)
        m.append(mu); e.append((var/len(v))**0.5)
    return m, e

# ---- Fig A: the frontier, headline arms only -------------------------------------
HEAD = [("base","base (no training)","#444","-",3.0),
        ("hi_grpo","GRPO 120 steps","tab:green","-",1.8),
        ("long_grpo","GRPO 300 steps","tab:green","--",2.4),
        ("fix_maxrl_1e-5","MaxRL (baselined) lr 1e-5","tab:purple","-",2.4),
        ("hi_maxrl","MaxRL un-baselined","tab:purple",":",1.6),
        ("bonbon_e6_3e-6","BonBon best (6 ep)","tab:orange","-",1.8),
        ("long_entropic","entropic 300 steps","tab:red","--",1.8),
        ("hi_sft-offline","BoN distillation (my RAFT)","tab:gray",":",1.6)]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
for arm, lab, c, ls, lw in HEAD:
    r = curve(arm)
    if not r: continue
    m, e = r
    ax[0].errorbar(KS, m, yerr=e, marker="o", ms=3.5, lw=lw, ls=ls, color=c, label=lab, capsize=2)
    ax[1].errorbar(m[0], m[-1], xerr=e[0], yerr=e[-1], marker="o", ms=9, color=c, capsize=2)
    ax[1].annotate(lab.split(" (")[0], (m[0], m[-1]), fontsize=7.5, xytext=(5,4),
                   textcoords="offset points", color=c)
b = curve("base")
ax[1].axvline(b[0][0], color="#444", ls=":", lw=1, alpha=.6)
ax[1].axhline(b[0][-1], color="#444", ls=":", lw=1, alpha=.6)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k")
ax[0].set_title("MATH-500 level>=4 (262 problems, n=64)"); ax[0].grid(alpha=.3)
ax[0].legend(fontsize=7.5, loc="upper left")
ax[1].set_xlabel("pass@1"); ax[1].set_ylabel("pass@64")
ax[1].set_title("the frontier: up and right wins (dotted = base)"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/figA_frontier.png", dpi=140); plt.close(fig)

# ---- Fig B: lr response, per objective -------------------------------------------
SER = {"GRPO": [("lr_grpo_3e-7",3e-7),("hi_grpo",3e-6),("lr_grpo_3e-6",3e-6)],
       "MaxRL (baselined)": [("fix_maxrl_1e-6",1e-6),("fix_maxrl_3e-6",3e-6),("fix_maxrl_1e-5",1e-5)],
       "MaxRL (un-baselined)": [("lr_maxrl_1e-7",1e-7),("lr_maxrl_3e-7",3e-7),("maxrl",1e-6),("hi_maxrl",3e-6)],
       "RAFT / vanilla REINFORCE": [("lr_sft-online_1e-7",1e-7),("lr_sft-online_3e-7",3e-7),
                                     ("sft-online",1e-6),("hi_sft-online",3e-6)],
       "BonBon": [("bonbon_e6_3e-7",3e-7),("bonbon_e6_1e-6",1e-6),("bonbon_e6_3e-6",3e-6),("bonbon_e6_1e-5",1e-5)]}
COL = {"GRPO":"tab:green","MaxRL (baselined)":"tab:purple","MaxRL (un-baselined)":"tab:pink",
       "RAFT / vanilla REINFORCE":"tab:gray","BonBon":"tab:orange"}
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for name, pts in SER.items():
    xs, y1, y64 = [], [], []
    for arm, lr in pts:
        r = curve(arm)
        if not r: continue
        xs.append(lr); y1.append(r[0][0]); y64.append(r[0][-1])
    if not xs: continue
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    ax[0].plot([xs[i] for i in o], [y1[i] for i in o], marker="o", color=COL[name], label=name)
    ax[1].plot([xs[i] for i in o], [y64[i] for i in o], marker="o", color=COL[name], label=name)
for a, t, bl in ((ax[0], "pass@1 vs learning rate", b[0][0]), (ax[1], "pass@64 vs learning rate", b[0][-1])):
    a.axhline(bl, color="#444", ls=":", lw=1, label="base")
    a.set_xscale("log"); a.set_xlabel("learning rate"); a.set_title(t); a.grid(alpha=.3)
    a.legend(fontsize=7)
fig.suptitle("Each objective has a different optimum: GRPO 3e-6, MaxRL 1e-5", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/figB_lr.png", dpi=140); plt.close(fig)

# ---- Fig C: 120 vs 300 steps -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for short, lng, lab, c in (("hi_grpo","long_grpo","GRPO","tab:green"),
                           ("hi_entropic","long_entropic","entropic","tab:red"),
                           ("fix_maxrl_3e-6","long_maxrl","MaxRL","tab:purple")):
    for arm, ls, tag in ((short,"-","120 steps"), (lng,"--","300 steps")):
        r = curve(arm)
        if r: ax[0].plot(KS, r[0], marker="o", ms=3, ls=ls, color=c, lw=2 if ls=="--" else 1.3,
                         label=f"{lab} {tag}")
    rs, rl = curve(short), curve(lng)
    if rs and rl:
        ax[1].plot([rs[0][0], rl[0][0]], [rs[0][-1], rl[0][-1]], marker="o", color=c, lw=2)
        ax[1].annotate(lab, (rl[0][0], rl[0][-1]), fontsize=8, xytext=(5,4), textcoords="offset points")
r = curve("base")
ax[0].plot(KS, r[0], color="#444", lw=3, label="base")
ax[1].scatter([r[0][0]], [r[0][-1]], color="#444", s=70, zorder=5)
ax[1].annotate("base", (r[0][0], r[0][-1]), fontsize=8, xytext=(5,-10), textcoords="offset points")
ax[1].axhline(r[0][-1], color="#444", ls=":", lw=1); ax[1].axvline(r[0][0], color="#444", ls=":", lw=1)
ax[0].set_xscale("log", base=2); ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k")
ax[0].set_title("120 vs 300 steps"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
ax[1].set_xlabel("pass@1"); ax[1].set_ylabel("pass@64")
ax[1].set_title("arrows: where 2.5x more training moves you"); ax[1].grid(alpha=.3)
fig.suptitle("Longer training lifts BOTH metrics -- the tradeoff is a short-horizon artifact", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/figC_long.png", dpi=140); plt.close(fig)
print("wrote figA_frontier, figB_lr, figC_long")
