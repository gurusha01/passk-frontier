import json, math, os, random
KS=[1,4,16,64,128,256]; random.seed(0); B=5000
def pak(n,c,k): return 1.0 if n-c<k else 1.0-math.comb(n-c,k)/math.comb(n,k)
def pp(arm,lvl=4):
    p=f"evals256/{arm}.jsonl"
    if not os.path.exists(p): return None
    r=[json.loads(l) for l in open(p) if l.strip()]
    r=[x for x in r if x.get("level",0)>=lvl]
    if not r: return None
    nmin=min(x["n"] for x in r)
    return {x["idx"]:{k:pak(x["n"],x["c"],k) for k in KS if k<=nmin} for x in r}, nmin
b,nmin=pp("base")
ks=[k for k in KS if k<=nmin]
print(f"n={nmin} samples/problem, {len(b)} problems (level>=4)\n")
print(f"{'arm':<18}"+"".join(f"{'p@'+str(k):>9}" for k in ks))
print(f"{'base':<18}"+"".join(f"{sum(b[i][k] for i in b)/len(b)*100:>9.1f}" for k in ks))
print()
print(f"{'arm':<18}"+"".join(f"{'d@'+str(k):>20}" for k in ks))
for arm in ["gatecap_1e-5","fix_maxrl_1e-5","long_grpo"]:
    r=pp(arm)
    if not r: print(f"{arm:<18} (pending)"); continue
    a,_=r; common=[i for i in b if i in a]; cells=[]
    for k in ks:
        d=[a[i][k]-b[i][k] for i in common]; mu=sum(d)/len(d)
        bo=sorted(sum(d[random.randrange(len(d))] for _ in range(len(d)))/len(d) for _ in range(B))
        lo,hi=bo[int(.025*B)],bo[int(.975*B)]
        cells.append(f"{mu*100:+5.1f}[{lo*100:+5.1f},{hi*100:+5.1f}]{'*' if (lo>0 or hi<0) else ' '}")
    print(f"{arm:<18}"+"".join(f"{c:>20}" for c in cells))
print("\n* = 95% paired bootstrap CI excludes zero")
