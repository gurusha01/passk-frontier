"""Every figure for the full report. Reads runs/ + evals/ directly."""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "${PF_ROOT}"
OUT = f"{ROOT}/results"
OBJ_COLOR = {"grpo": "tab:green", "maxrl": "tab:purple", "raft": "tab:gray",
             "entropic": "tab:orange", "hientropy": "tab:red"}
DIV_KEYS = [("reward_mean", "train reward"), ("token_entropy", "policy token entropy"),
            ("logp_mean", "mean token logprob"), ("distinct_answers", "distinct answers/group"),
            ("distinct_correct", "distinct CORRECT answers/group"),
            ("frac_k0", "frac groups K=0 (no signal)"),
            ("frac_kG", "frac groups K=G (saturated)"),
            ("self_bleu", "self-BLEU (lower = more diverse)"),
            ("distinct_4", "distinct-4gram ratio"), ("len_mean", "completion length (tokens)"),
            ("len_std", "completion length std")]


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def load():
    runs = {}
    for d in sorted(glob.glob(f"{ROOT}/runs/*")):
        a = os.path.basename(d)
        if a.startswith("_") or not os.path.exists(f"{d}/args.json"):
            continue
        runs[a] = {"args": json.load(open(f"{d}/args.json")),
                   "div": jl(f"{d}/diversity.jsonl"), "probe": jl(f"{d}/passk_probe.jsonl")}
    return runs


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def eval_curve(path, ks, min_level=0):
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if r.get("level", 0) >= min_level] if min_level else recs
    if not recs:
        return None, None, 0
    nmin = min(r["n"] for r in recs)
    ks = [k for k in ks if k <= nmin]
    m, e = [], []
    for k in ks:
        v = [pass_at_k(r["n"], r["c"], k) for r in recs]
        mu = sum(v) / len(v)
        var = sum((x - mu) ** 2 for x in v) / max(len(v) - 1, 1)
        m.append(mu)
        e.append((var / len(v)) ** 0.5)
    return ks, (m, e), len(recs)


# ---------------------------------------------------------------- Fig 1: probe trajectories
def fig_probes(runs):
    groups = [("lr 1e-7", ["lr_maxrl_1e-7", "lr_sft-online_1e-7"]),
              ("lr 3e-7", ["lr_grpo_3e-7", "lr_maxrl_3e-7", "lr_sft-online_3e-7"]),
              ("lr 1e-6 (wave 1)", ["grpo", "maxrl", "sft-online", "sft-iter", "sft-offline",
                                    "entropic", "hientropy"]),
              ("lr 3e-6 (wave 1c)", ["hi_grpo", "hi_maxrl", "hi_sft-online", "hi_sft-iter",
                                     "hi_sft-offline", "hi_entropic", "hi_hientropy"])]
    metrics = [("1", "probe pass@1"), ("4", "probe pass@4"), ("16", "probe pass@16")]
    fig, axes = plt.subplots(3, 4, figsize=(21, 11), sharey="row")
    for col, (title, arms) in enumerate(groups):
        for row, (kk, mname) in enumerate(metrics):
            ax = axes[row][col]
            for a in arms:
                r = runs.get(a)
                if not r or not r["probe"]:
                    continue
                x = [p["step"] for p in r["probe"]]
                y = [p["curve"].get(kk, p["curve"].get(int(kk))) for p in r["probe"]]
                obj = r["args"]["objective"]
                ls = "-"
                if "sft-iter" in a:
                    ls = "--"
                elif "sft-offline" in a:
                    ls = ":"
                ax.plot(x, y, marker="o", ms=3, lw=1.6, ls=ls,
                        color=OBJ_COLOR.get(obj, "k"), label=a.replace("hi_", "").replace("lr_", ""))
            ax.grid(alpha=.3)
            if row == 0:
                ax.set_title(title, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(mname, fontsize=11)
            if row == 2:
                ax.set_xlabel("training step")
            ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Probe pass@k during training (50 MATH-500 problems, n=16) — grouped by learning rate",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_probe_trajectories.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------- Fig 2: full diversity metric grid
def fig_diversity(runs, arms, tag, title):
    fig, axes = plt.subplots(3, 4, figsize=(20, 11))
    for ax, (key, lab) in zip(axes.flat, DIV_KEYS):
        for a in arms:
            r = runs.get(a)
            if not r:
                continue
            pts = [(d["n_rollouts"], d[key]) for d in r["div"]
                   if key in d and d[key] == d[key]]
            if not pts:
                continue
            ls = "--" if "sft-iter" in a else (":" if "sft-offline" in a else "-")
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.4, ls=ls,
                    color=OBJ_COLOR.get(r["args"]["objective"], "k"),
                    label=a.replace("hi_", "").replace("lr_", ""))
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("rollouts")
        ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=7, ncol=2)
    for ax in axes.flat[len(DIV_KEYS):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- Fig 4: lr response
def fig_lr_response(runs):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    series = {}
    for a, r in runs.items():
        if not r["probe"]:
            continue
        obj = r["args"]["objective"]
        name = obj if obj != "raft" else f"raft/{'online' if r['args']['flush_every']==1 else ('iter' if r['args']['flush_every']==30 else 'offline')}"
        lr = float(r["args"]["lr"])
        last, first = r["probe"][-1], r["probe"][0]
        series.setdefault(name, []).append(
            (lr, last["curve"].get("1", last["curve"].get(1)),
             last["curve"].get("16", last["curve"].get(16)),
             first["curve"].get("1", first["curve"].get(1))))
    for name, pts in sorted(series.items()):
        pts.sort()
        c = OBJ_COLOR.get(name.split("/")[0], "k")
        ls = "--" if "iter" in name else (":" if "offline" in name else "-")
        ax[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", color=c, ls=ls, label=name)
        ax[1].plot([p[0] for p in pts], [p[2] for p in pts], marker="o", color=c, ls=ls, label=name)
        ax[2].plot([p[0] for p in pts], [p[1] - p[3] for p in pts], marker="o", color=c, ls=ls, label=name)
    for a_, t in zip(ax, ["final probe pass@1", "final probe pass@16",
                          "change in probe pass@1 (final - step 0)"]):
        a_.set_xscale("log")
        a_.set_xlabel("learning rate")
        a_.set_title(t, fontsize=11)
        a_.grid(alpha=.3)
        a_.legend(fontsize=7)
    ax[2].axhline(0, color="k", lw=1, ls="-", alpha=.6)
    fig.suptitle("Learning-rate response per objective (each point is a full 120-step run)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_lr_response.png", dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------- Fig 5: frontier(s)
def fig_frontier(tag, pattern, title):
    ks = [1, 2, 4, 8, 16, 32, 64]
    paths = sorted(p for p in glob.glob(f"{ROOT}/evals/*.jsonl")
                   if pattern(os.path.basename(p)[:-6]))
    if len(paths) < 2:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for col, (lvl, lname) in enumerate([(0, "all 500"), (4, "level>=4"), (5, "level 5")]):
        ax = axes[col]
        nprob = 0
        for p in paths:
            arm = os.path.basename(p)[:-6]
            kk, mv, n = eval_curve(p, ks, lvl)
            if not kk:
                continue
            nprob = max(nprob, n)      # n from the last path can be 0 if that file is empty
            m, e = mv
            lw = 3.0 if arm == "base" else 1.4
            ax.errorbar(kk, m, yerr=e, marker="o", ms=3, lw=lw, capsize=2,
                        label=f"{arm}", zorder=5 if arm == "base" else 2)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k")
        ax.set_title(f"{lname}  (n={nprob} problems)", fontsize=11)
        ax.grid(alpha=.3)
        if col == 0:
            ax.set_ylabel("pass@k")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{tag}.png", dpi=130)
    plt.close(fig)
    return True


def main():
    os.makedirs(OUT, exist_ok=True)
    runs = load()
    fig_probes(runs)
    fig_diversity(runs, ["grpo", "maxrl", "sft-online", "sft-iter", "sft-offline",
                         "entropic", "hientropy"], "fig2_diversity_lr1e-6",
                  "Training diversity metrics — wave 1, lr 1e-6 (all 120 steps)")
    fig_diversity(runs, ["hi_grpo", "hi_maxrl", "hi_sft-online", "hi_sft-iter",
                         "hi_sft-offline", "hi_entropic", "hi_hientropy"],
                  "fig3_diversity_lr3e-6",
                  "Training diversity metrics — wave 1c, lr 3e-6 (all 120 steps)")
    fig_lr_response(runs)
    fig_frontier("fig5_frontier_lr1e-6", lambda a: a == "base" or ("_" not in a),
                 "pass@k frontier — wave 1 (lr 1e-6), n=64, unbiased estimator")
    fig_frontier("fig6_frontier_lr3e-6", lambda a: a == "base" or a.startswith("hi_"),
                 "pass@k frontier — wave 1c (lr 3e-6), n=64, unbiased estimator")
    print("figures written to", OUT)
    for f in sorted(glob.glob(f"{OUT}/fig*.png")):
        print("  ", os.path.basename(f), os.path.getsize(f) // 1024, "KB")


if __name__ == "__main__":
    main()
