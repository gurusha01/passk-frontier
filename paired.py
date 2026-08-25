"""Paired bootstrap vs base. Every arm is evaluated on the SAME problems, so the paired
difference has far less variance than the marginal error bars imply."""
import json, math, os, random

KS = [1, 16, 64]
random.seed(0)

def pak(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)

def per_problem(arm, lvl=4):
    p = f"evals/{arm}.jsonl"
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("level", 0) >= lvl]
    if not recs:
        return None
    return {r["idx"]: {k: pak(r["n"], r["c"], k) for k in KS} for r in recs}

base = per_problem("base")
idxs = sorted(base)

ARMS = ["fix_maxrl_1e-5", "gatecap_1e-5", "gatecapmask_1e-5", "gatehard_1e-5",
        "gatedrop_1e-5", "pow15_1e-5", "pow20_1e-5", "geo8_1e-5", "geo32_1e-5",
        "long_grpo", "hi_grpo", "reinforce_b_1e-5", "fix_raft_kl", "bonbon_e6_3e-6"]

B = 5000
print(f"{'arm':<20}" + "".join(f"{'d@'+str(k):>22}" for k in KS))
print(f"{'':<20}" + "".join(f"{'(95% CI, paired)':>22}" for k in KS))
for arm in ARMS:
    a = per_problem(arm)
    if not a:
        continue
    common = [i for i in idxs if i in a]
    cells = []
    for k in KS:
        d = [a[i][k] - base[i][k] for i in common]
        mu = sum(d) / len(d)
        boots = []
        for _ in range(B):
            s = sum(d[random.randrange(len(d))] for _ in range(len(d)))
            boots.append(s / len(d))
        boots.sort()
        lo, hi = boots[int(.025 * B)], boots[int(.975 * B)]
        sig = "*" if (lo > 0 or hi < 0) else " "
        cells.append(f"{mu*100:+6.1f} [{lo*100:+5.1f},{hi*100:+5.1f}]{sig}")
    print(f"{arm:<20}" + "".join(f"{c:>22}" for c in cells))
print("\n* = 95% paired bootstrap CI excludes zero.  n =", len(idxs), "problems, level>=4")
