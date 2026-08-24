"""Result figures: the pass@k frontier (the answer) and the training trajectories (the why)."""
import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAJ = [("reward_mean", "train reward"), ("token_entropy", "policy entropy"),
        ("distinct_correct", "distinct correct answers / group"),
        ("frac_k0", "frac groups with K=0"), ("self_bleu", "self-BLEU (lower=diverse)"),
        ("len_mean", "completion length")]


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passk", default="results/passk.json")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = json.load(open(args.passk))

    # ---- Figure 1: the frontier -------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    kmax = max(int(k) for k in res[list(res)[0]]["curve"])   # int, not string, sort
    bx = by = None
    for arm, r in res.items():
        ks = sorted(int(k) for k in r["curve"])
        v = [r["curve"][str(k)] if str(k) in r["curve"] else r["curve"][k] for k in ks]
        e = [r["stderr"][str(k)] if str(k) in r["stderr"] else r["stderr"][k] for k in ks]
        lw = 2.5 if arm in ("base", "maxrl") else 1.3
        ax[0].errorbar(ks, v, yerr=e, marker="o", ms=3, lw=lw, label=arm, capsize=2)
        ax[1].errorbar(v[0], v[-1], xerr=e[0], yerr=e[-1], marker="o", ms=8, capsize=2)
        ax[1].annotate(arm, (v[0], v[-1]), fontsize=8,
                       xytext=(4, 4), textcoords="offset points")
        if arm == "base":
            bx, by = v[0], v[-1]
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("k"); ax[0].set_ylabel("pass@k"); ax[0].set_title("MATH-500 pass@k")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)
    if bx is not None:      # base crosshair: anything below/left of these lines lost to base
        ax[1].axvline(bx, color="k", ls=":", lw=1, alpha=.5)
        ax[1].axhline(by, color="k", ls=":", lw=1, alpha=.5)
    ax[1].set_xlabel("pass@1"); ax[1].set_ylabel(f"pass@{kmax}")
    ax[1].set_title("the frontier (up and right wins; dotted = base)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{args.out}/frontier.png", dpi=150)

    # ---- Figure 2: training trajectories ----------------------------------------------
    arms = sorted(d for d in os.listdir(args.runs)
                  if os.path.exists(f"{args.runs}/{d}/diversity.jsonl"))
    if not arms:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for arm in arms:
        rows = load_jsonl(f"{args.runs}/{arm}/diversity.jsonl")
        for a, (key, title) in zip(axes.flat, TRAJ):
            xs = [r["n_rollouts"] for r in rows if r.get(key) == r.get(key)]
            ys = [r[key] for r in rows if r.get(key) == r.get(key)]
            a.plot(xs, ys, lw=1.3, label=arm)
            a.set_title(title, fontsize=10); a.set_xlabel("rollouts"); a.grid(alpha=.3)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{args.out}/trajectories.png", dpi=150)

    # ---- Figure 3: mid-training pass@k probe ------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in arms:
        p = f"{args.runs}/{arm}/passk_probe.jsonl"
        if not os.path.exists(p):
            continue
        rows = load_jsonl(p)
        x = [r["n_rollouts"] for r in rows]
        ax[0].plot(x, [r["curve"].get("1", r["curve"].get(1)) for r in rows], marker="o",
                   ms=3, label=arm)
        ax[1].plot(x, [r["curve"].get("16", r["curve"].get(16)) for r in rows], marker="o",
                   ms=3, label=arm)
    ax[0].set_title("probe pass@1 vs rollouts"); ax[1].set_title("probe pass@16 vs rollouts")
    for a in ax:
        a.set_xlabel("rollouts"); a.grid(alpha=.3); a.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(f"{args.out}/probe_trajectory.png", dpi=150)
    print("wrote", args.out + "/{frontier,trajectories,probe_trajectory}.png")


if __name__ == "__main__":
    main()
