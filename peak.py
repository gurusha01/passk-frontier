import glob, json, os
rows=[]
for d in sorted(glob.glob("runs/*/passk_probe.jsonl")):
    a=os.path.basename(os.path.dirname(d))
    r=[json.loads(l) for l in open(d) if l.strip()]
    if len(r)<3: continue
    p1=[(x["step"], x["curve"].get("1", x["curve"].get(1))) for x in r]
    bs,bv=max(p1,key=lambda t:t[1]); fs,fv=p1[-1]
    rows.append((bv-fv, a, bs, bv, fs, fv))
rows.sort(reverse=True)
print(f"{'arm':<22}{'best_step':>10}{'best_p@1':>10}{'final_step':>11}{'final_p@1':>10}{'gap':>8}")
for gap,a,bs,bv,fs,fv in rows:
    if gap>0.015:
        print(f"{a:<22}{bs:>10}{bv:>10.3f}{fs:>11}{fv:>10.3f}{gap:>+8.3f}")
print(f"\n{len([r for r in rows if r[0]>0.015])} of {len(rows)} runs peak >1.5pt before the end")
