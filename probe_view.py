"""Read the fixed probe set: same problem, same seed, every arm, every step, side by side.

  python inspect.py --step 0 --step 60 --step 119 --problem 0
  python inspect.py --arms base,grpo,maxrl --problem 2 --step 119 --full
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--arms", default="", help="comma list; default all")
    ap.add_argument("--step", action="append", type=int, default=[])
    ap.add_argument("--problem", type=int, default=0)
    ap.add_argument("--n", type=int, default=3, help="completions to show per cell")
    ap.add_argument("--chars", type=int, default=400)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    arms = args.arms.split(",") if args.arms else sorted(
        d for d in os.listdir(args.runs) if os.path.isdir(f"{args.runs}/{d}/probe"))
    steps = args.step or [0]

    for step in steps:
        print("=" * 100)
        print(f"STEP {step}   problem #{args.problem}")
        print("=" * 100)
        for arm in arms:
            p = f"{args.runs}/{arm}/probe/step{step}.jsonl"
            if not os.path.exists(p):
                print(f"[{arm}] (no probe at this step)")
                continue
            with open(p) as f:
                recs = [json.loads(l) for l in f]
            if args.problem >= len(recs):
                continue
            r = recs[args.problem]
            if arm == arms[0] and step == steps[0]:
                print(f"\nPROBLEM: {r['problem'][:500]}\nGOLD: {r['gold']}\n")
            print(f"\n--- {arm}   c={r['c']}/{r['n']}   distinct answers="
                  f"{len({t.rfind(chr(92)+'boxed') for t in r['texts']})}")
            for i, (t, rew) in enumerate(zip(r["texts"][:args.n], r["rewards"][:args.n])):
                body = t if args.full else t[:args.chars].replace("\n", " ")
                print(f"  [{i}] r={rew:.0f}  {body}{'' if args.full else ' ...'}")


if __name__ == "__main__":
    main()
