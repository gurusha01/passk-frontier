"""Dump every run's config, full diversity trajectory, probe pass@k trajectory and eval
curves into one JSON, so the report can be built without touching the cluster again."""
import glob
import json
import math
import os


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curves(path, ks=(1, 2, 4, 8, 16, 32, 64)):
    recs = [json.loads(l) for l in open(path)]
    if not recs:
        return None
    out = {}
    for name, sub in (("all", recs),
                      ("L4", [r for r in recs if r.get("level", 0) >= 4]),
                      ("L5", [r for r in recs if r.get("level", 0) >= 5])):
        if not sub:
            continue
        nmin = min(r["n"] for r in sub)
        d = {"n_problems": len(sub), "curve": {}, "stderr": {}}
        for k in [k for k in ks if k <= nmin]:
            v = [pass_at_k(r["n"], r["c"], k) for r in sub]
            m = sum(v) / len(v)
            var = sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)
            d["curve"][k] = m
            d["stderr"][k] = (var / len(v)) ** 0.5
        if "greedy" in sub[0]:
            d["greedy"] = sum(r["greedy"] for r in sub) / len(sub)
        out[name] = d
    return out


def main():
    root = "${PF_ROOT}"
    res = {}
    for d in sorted(glob.glob(f"{root}/runs/*")):
        arm = os.path.basename(d)
        if arm.startswith("_"):
            continue
        e = {}
        if os.path.exists(f"{d}/args.json"):
            e["args"] = json.load(open(f"{d}/args.json"))
        for key, f in (("diversity", "diversity.jsonl"), ("probe", "passk_probe.jsonl")):
            p = f"{d}/{f}"
            if os.path.exists(p):
                e[key] = [json.loads(l) for l in open(p)]
        e["done"] = os.path.isdir(f"{d}/final")
        res[arm] = e
    for p in sorted(glob.glob(f"{root}/evals/*.jsonl")):
        arm = os.path.basename(p)[:-6]
        c = curves(p)
        if c:
            res.setdefault(arm, {})["eval"] = c
    print(json.dumps(res))


if __name__ == "__main__":
    main()
