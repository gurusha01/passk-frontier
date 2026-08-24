import numpy as np
from train import advantage

G = 8
print(f"{'K/G':>5} {'p':>6} | {'grpo':>18} | {'maxrl':>18} | {'pow1.5':>18} | {'pow2':>18} | {'geo(k=8)':>18} | {'difkl':>18}")
for K in range(0, G + 1):
    rg = np.array([1.0] * K + [0.0] * (G - K))
    grp = np.zeros(G, dtype=int)
    row = []
    for obj, kw in (("grpo", {}), ("maxrl", {}), ("powmaxrl", dict(lam=1.5)),
                    ("powmaxrl", dict(lam=2.0)), ("geopassk", dict(k_eff=8.0)),
                    ("difkl", dict(kappa=1.0))):
        w = advantage(rg, grp, 1, G, obj, 2.0, 0.5, None, **{**dict(lam=1.5, k_eff=8.0, kappa=1.0, pfloor=1/16), **kw})
        pos = w[0] if K > 0 else 0.0
        neg = w[-1] if K < G else 0.0
        row.append(f"{pos:+7.2f}/{neg:+7.2f} s{w.sum():+.0e}")
    print(f"{K}/{G:>2} {K/G:6.3f} | " + " | ".join(row))
