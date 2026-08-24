"""MATH data + binary verifiable reward, shared by training and eval.

Train prompts come from hendrycks_math (levels 1-3); pass@k eval uses MATH-500.
Reward is math_verify on the last \\boxed{} of the completion, so it is the same
function everywhere and no arm can differ by grading.
"""
import os
import re

os.environ.setdefault("HF_HOME", "${PF_HF_HOME}")

EVAL_INSTR = r"Please reason step by step, and put your final answer within \boxed{}."
MATH_CONFIGS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
LEVEL_RE = re.compile(r"Level (\d)")


def last_boxed(s):
    """Content of the final \\boxed{...} in s, or None. Brace-balanced."""
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        d += (s[k] == "{") - (s[k] == "}")
        if d == 0:
            return s[j + 1:k]
    return None


def reward(text, gold):
    """1.0 iff the completion's final boxed answer verifies against gold."""
    from math_verify import parse, verify
    try:
        g = parse("\\boxed{" + gold + "}")
        p = parse(text)
        return 1.0 if (p and verify(g, p)) else 0.0
    except Exception:
        return 0.0


def norm_answer(text):
    """Normalized boxed string, for counting distinct answers within a group."""
    b = last_boxed(text)
    if b is None:
        return None
    return re.sub(r"\s+|\\left|\\right|\\!|\\,", "", b)


def load_train(n=2000, max_level=3, seed=7291):
    """[(problem, gold, level)] from hendrycks_math train, filtered to <= max_level."""
    from datasets import load_dataset, concatenate_datasets
    ds = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", c, split="train")
                               for c in MATH_CONFIGS]).shuffle(seed=seed)
    out = []
    for r in ds:
        m = LEVEL_RE.search(r.get("level", ""))
        lvl = int(m.group(1)) if m else 5
        if lvl > max_level:
            continue
        g = last_boxed(r["solution"])
        if g:
            out.append((r["problem"], g, lvl))
        if len(out) >= n:
            break
    return out


def load_math500(n=None):
    """[(problem, gold, level)] from MATH-500 test."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if n:
        ds = ds.select(range(min(n, len(ds))))
    return [(r["problem"], r["answer"], int(r.get("level", 0))) for r in ds]


def build_prompt(tok, problem):
    return tok.apply_chat_template(
        [{"role": "user", "content": problem + "\n\n" + EVAL_INSTR}],
        tokenize=False, add_generation_prompt=True)
