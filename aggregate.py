"""Unbiased pass@k curves per arm. Estimator (Chen et al. 2021):
pass@k = 1 - C(n-c, k) / C(n, k), averaged over problems.

Taken from passk_experiment/aggregate_passk.py; extended to sweep a directory of arms and
to report a standard error, since the effect sizes here are expected to be small.
"""
import argparse
import glob
import json
import math
import os


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curve_for(path, ks, min_level=0):
    with open(path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    if min_level:
        recs = [r for r in recs if r.get("level", 0) >= min_level]
    if not recs:
        return None                     # eval still writing, or no problems at this level
    n_min = min(r["n"] for r in recs)
    out = {"n_problems": len(recs), "n_samples": n_min, "curve": {}, "stderr": {}}
    for k in [k for k in ks if k <= n_min]:
        v = [pass_at_k(r["n"], r["c"], k) for r in recs]
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)
        out["curve"][k] = m
        out["stderr"][k] = (var / len(v)) ** 0.5
    if "greedy" in recs[0]:
        out["greedy_pass1"] = sum(r["greedy"] for r in recs) / len(recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="evals/*.jsonl", help="one jsonl per arm, named <arm>.jsonl")
    ap.add_argument("--ks", default="1,2,4,8,16,32,64")
    ap.add_argument("--out", default="results/passk.json")
    ap.add_argument("--min_level", type=int, default=0,
                    help="restrict to MATH-500 problems at or above this level. Training is "
                         "on levels 1-3, so --min_level 4 is both the harder (unsaturated) "
                         "slice and an out-of-distribution difficulty readout")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]
    res = {}
    for p in sorted(glob.glob(args.glob)):
        c = curve_for(p, ks, args.min_level)
        if c is None:
            print(f"  (skipping {os.path.basename(p)}: no records yet)")
        else:
            res[os.path.basename(p)[:-6]] = c
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    n = res[list(res)[0]]["n_problems"] if res else 0
    print(f"min_level={args.min_level}  n_problems={n}")
    print(f"{'arm':<18}" + "".join(f"p@{k:<7}" for k in ks) + "greedy")
    for arm, r in res.items():
        print(f"{arm:<18}" + "".join(
            f"{r['curve'].get(k, float('nan')):<9.3f}" for k in ks)
            + f"{r.get('greedy_pass1', float('nan')):.3f}")


if __name__ == "__main__":
    main()
