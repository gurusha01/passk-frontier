"""Trajectory logging: raw rollouts, a fixed probe set, and per-step diversity metrics.

Kept out of train.py so metrics can be recomputed offline from rollouts.jsonl without
rerunning training. Every arm writes the same schema, so runs are directly diffable.
"""
import json
import os
from collections import Counter

from data import norm_answer


class Run:
    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir + "/probe", exist_ok=True)
        self.rollouts = open(out_dir + "/rollouts.jsonl", "a")
        self.div = open(out_dir + "/diversity.jsonl", "a")
        self.n_rollouts = 0        # cumulative budget counter; must match across arms

    def log_rollouts(self, step, groups):
        """groups: [{prompt_idx, level, texts[G], rewards[G], logps[G], ntoks[G]}]"""
        for g in groups:
            for i, t in enumerate(g["texts"]):
                self.rollouts.write(json.dumps({
                    "step": step, "prompt_idx": g["prompt_idx"], "level": g["level"],
                    "gen_idx": i, "text": t, "reward": g["rewards"][i],
                    "logp_mean": g["logps"][i], "n_tokens": g["ntoks"][i]}) + "\n")
            self.n_rollouts += len(g["texts"])
        self.rollouts.flush()

    def log_diversity(self, step, groups, token_entropy):
        self.div.write(json.dumps(
            {"step": step, "n_rollouts": self.n_rollouts,
             "token_entropy": token_entropy, **diversity_metrics(groups)}) + "\n")
        self.div.flush()

    def log_probe(self, step, records):
        with open(f"{self.dir}/probe/step{step}.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def log_passk_probe(self, step, curve):
        with open(self.dir + "/passk_probe.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "n_rollouts": self.n_rollouts,
                                "curve": curve}) + "\n")


def distinct_n(toks_list, n=4):
    """Fraction of distinct n-grams across a group's completions."""
    grams = Counter()
    for toks in toks_list:
        for i in range(len(toks) - n + 1):
            grams[tuple(toks[i:i + n])] += 1
    tot = sum(grams.values())
    return len(grams) / tot if tot else 0.0


def self_bleu_proxy(toks_list, n=4):
    """Mean pairwise n-gram Jaccard overlap. Higher = less diverse. Cheap self-BLEU stand-in."""
    sets = [set(tuple(t[i:i + n]) for i in range(len(t) - n + 1)) for t in toks_list]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = len(sets[i] | sets[j])
            tot += len(sets[i] & sets[j]) / u if u else 0.0
            cnt += 1
    return tot / cnt


def diversity_metrics(groups):
    """Per-step aggregates over the training rollouts of one step."""
    G = len(groups[0]["texts"])
    ks = [int(sum(g["rewards"])) for g in groups]
    khist = [0.0] * (G + 1)
    for k in ks:
        khist[k] += 1.0 / len(ks)

    da, dc, sb, dn, lens, logps = [], [], [], [], [], []
    for g in groups:
        answers = [norm_answer(t) for t in g["texts"]]
        da.append(len({a for a in answers if a is not None}))
        dc.append(len({a for a, r in zip(answers, g["rewards"]) if a is not None and r > 0}))
        toks = [t.split() for t in g["texts"]]
        sb.append(self_bleu_proxy(toks))
        dn.append(distinct_n(toks))
        lens += g["ntoks"]
        logps += g["logps"]

    def mean(x):
        return sum(x) / len(x) if x else 0.0

    var = mean([(l - mean(lens)) ** 2 for l in lens])
    return {
        "reward_mean": mean([r for g in groups for r in g["rewards"]]),
        "k_hist": khist,
        "frac_k0": khist[0],
        "frac_kG": khist[G],
        "distinct_answers": mean(da),
        "distinct_correct": mean(dc),
        "self_bleu": mean(sb),
        "distinct_4": mean(dn),
        "logp_mean": mean(logps),
        "len_mean": mean(lens),
        "len_std": var ** 0.5,
    }
