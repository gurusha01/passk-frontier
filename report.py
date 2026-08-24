"""Console readout of the training trajectories: probe pass@k, diversity, and probe text."""
import argparse
import json
import os

ARMS = ["sft-offline", "sft-iter", "sft-online", "grpo", "maxrl", "entropic", "hientropy"]


def rows(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--arms", default="")
    ap.add_argument("--text", type=int, default=0, help="show N probe completions per arm")
    ap.add_argument("--steps", default="0,119")
    args = ap.parse_args()
    arms = args.arms.split(",") if args.arms else [
        a for a in ARMS if os.path.isdir(f"{args.runs}/{a}")]

    print("\n== probe pass@k trajectory (50 MATH-500 problems, n=16) ==")
    for a in arms:
        rs = rows(f"{args.runs}/{a}/passk_probe.jsonl")
        if not rs:
            continue
        print(f"\n  {a}")
        for d in rs:
            c = {int(k): v for k, v in d["curve"].items()}
            print(f"    step {d['step']:>4}  p@1={c[1]:.3f}  p@4={c[4]:.3f}  p@16={c[16]:.3f}")

    print("\n== diversity: step 0 -> final ==")
    ks = ["reward_mean", "token_entropy", "distinct_correct", "frac_k0", "len_mean"]
    print("  " + "arm".ljust(13) + "".join(k[:16].ljust(20) for k in ks))
    for a in arms:
        rs = rows(f"{args.runs}/{a}/diversity.jsonl")
        if not rs:
            continue
        line = "  " + a.ljust(13)
        for k in ks:
            v = [r[k] for r in rs if r.get(k) == r.get(k)]
            line += f"{v[0]:.3f} -> {v[-1]:.3f}".ljust(20) if v else "n/a".ljust(20)
        print(line)

    if args.text:
        for st in [int(s) for s in args.steps.split(",")]:
            print(f"\n{'='*90}\n== probe completions @ step {st} (problem 0) ==\n{'='*90}")
            for a in arms:
                rs = rows(f"{args.runs}/{a}/probe/step{st}.jsonl")
                if not rs:
                    continue
                r = rs[0]
                print(f"\n--- {a}  c={r['c']}/{r['n']}")
                for t in r["texts"][:args.text]:
                    print("    " + t[:300].replace("\n", " ") + " ...")


if __name__ == "__main__":
    main()
