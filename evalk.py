"""pass@k eval with vLLM: n samples per MATH-500 problem, write {idx, n, c} per problem.

Record schema matches passk_experiment/passk_eval.py so aggregate.py is unchanged.
Sampling is untruncated (top_p=1.0, top_k=-1) and identical for every arm, since pass@k is
only comparable across models at a fixed sampling configuration.
"""
import argparse
import json
import os

os.environ.setdefault("HF_HOME", "${PF_HF_HOME}")

from data import EVAL_INSTR, load_math500, reward  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--n_probs", type=int, default=500)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--gpu_frac", type=float, default=0.85)
    ap.add_argument("--greedy", action="store_true", help="also record greedy pass@1")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    probs = load_math500(args.n_probs)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p + "\n\n" + EVAL_INSTR}],
                                       tokenize=False, add_generation_prompt=True)
               for p, _, _ in probs]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_frac,
              max_model_len=2048, trust_remote_code=True)
    sp = SamplingParams(n=args.n, temperature=args.temp, top_p=1.0, top_k=-1,
                        max_tokens=args.max_new, seed=1234)
    outs = llm.generate(prompts, sp)

    greedy = None
    if args.greedy:
        g = llm.generate(prompts, SamplingParams(n=1, temperature=0.0, max_tokens=args.max_new))
        greedy = [reward(o.outputs[0].text, probs[i][1]) for i, o in enumerate(g)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for i, o in enumerate(outs):
            texts = [c.text for c in o.outputs]
            c = int(sum(reward(t, probs[i][1]) for t in texts))
            rec = {"idx": i, "level": probs[i][2], "n": len(texts), "c": c}
            if greedy is not None:
                rec["greedy"] = greedy[i]
            f.write(json.dumps(rec) + "\n")
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
