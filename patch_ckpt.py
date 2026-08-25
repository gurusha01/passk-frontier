import ast
s = open("train.py").read()

s = s.replace(
'''    ap.add_argument("--max_rollouts", type=int, default=0,''',
'''    ap.add_argument("--save_every", type=int, default=0,
                    help="also save a checkpoint every N steps, and keep a `best` symlink "
                         "pointing at the highest probe pass@1 seen. Without this only the "
                         "FINAL checkpoint exists, which silently reports the post-peak "
                         "state for any run that overfits -- 27 of 56 runs in this study "
                         "peaked more than 1.5 points before their last step")
    ap.add_argument("--max_rollouts", type=int, default=0,''')

# track best probe pass@1 and checkpoint on it
s = s.replace(
'''            run.log_passk_probe(step, curve)
            print("PASSK_PROBE", step, json.dumps(curve), flush=True)''',
'''            run.log_passk_probe(step, curve)
            print("PASSK_PROBE", step, json.dumps(curve), flush=True)
            if args.save_every:
                p1 = curve.get(1, 0.0)
                if p1 > best["p1"]:
                    best["p1"], best["step"] = p1, step
                    d = f"{args.out_dir}/best"
                    os.makedirs(d, exist_ok=True)
                    model.save_pretrained(d)
                    tok.save_pretrained(d)
                    json.dump({"step": step, "probe_pass1": p1, "curve": curve},
                              open(d + "/best.json", "w"), indent=2)
                    print(f"BEST step={step} probe_pass@1={p1:.4f} -> {d}", flush=True)''')

s = s.replace(
'''        if step % args.probe_every == 0 or step == args.steps - 1:
            run.log_probe(step, sample_probe(model, tok, probe_set, 8, args, pad))''',
'''        if args.save_every and step and step % args.save_every == 0:
            d = f"{args.out_dir}/ckpt_{step}"
            model.save_pretrained(d)
            tok.save_pretrained(d)
            print("CKPT", d, flush=True)

        if step % args.probe_every == 0 or step == args.steps - 1:
            run.log_probe(step, sample_probe(model, tok, probe_set, 8, args, pad))''')

s = s.replace(
'''    buf = {"fulls": [], "glens": [], "rewards": [], "group": [], "sizes": []}
    gbase = 0
    ptr = 0''',
'''    buf = {"fulls": [], "glens": [], "rewards": [], "group": [], "sizes": []}
    best = {"p1": -1.0, "step": -1}
    gbase = 0
    ptr = 0''')

s = s.replace(
'''    print("SAVED", d, "rollouts", run.n_rollouts, flush=True)''',
'''    print("SAVED", d, "rollouts", run.n_rollouts, flush=True)
    if args.save_every:
        print(f"BEST_CKPT step={best['step']} probe_pass@1={best['p1']:.4f}", flush=True)''')

open("train.py", "w").write(s)
ast.parse(s)
print("ok:", s.count("save_every"), "save_every refs,", s.count('best["p1"]'), "best refs")
