import json, os, sys
for a in sys.argv[1:]:
    p = f"runs/{a}/passk_probe.jsonl"
    if not os.path.exists(p):
        print(f"{a:22s} (none)"); continue
    rs = [json.loads(l) for l in open(p)]
    cells = []
    for r in rs:
        c = {str(k): v for k, v in r["curve"].items()}
        cells.append(f"s{r['step']}:{c['1']:.3f}/{c['16']:.2f}")
    print(f"{a:22s} " + "  ".join(cells))
