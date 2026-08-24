"""One rollout loop, N objectives. Every arm in the study differs ONLY in `advantage()`
(and, for the SFT ladder, in `--flush_every`), so all arms see an identical rollout budget
by construction: steps * n_prompts * num_gen.

The SFT ladder falls out of --flush_every with --objective raft:
  flush_every=1      -> fully on-policy RAFT (update every step)
  flush_every=30     -> iterated rejection sampling / ReST (4 rounds over 120 steps)
  flush_every=steps  -> offline BoN distillation (policy is frozen during all collection,
                        because no update happens until the single flush at the end)

Sampling for rollouts is deliberately untruncated (top_k=0, top_p=1.0, temp=1.0). Truncated
sampling would suppress the tail this experiment is trying to measure.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "${PF_HF_HOME}")

from data import build_prompt, load_math500, load_train, reward  # noqa: E402
from log import Run  # noqa: E402

OBJECTIVES = ["grpo", "drgrpo", "maxrl", "maxrl_nb", "raft", "rwr", "entropic",
              "entbonus", "hientropy", "cliphigher", "klanchor", "gatehard",
              "powmaxrl", "geopassk", "difkl"]
GRPO_LIKE = {"grpo", "entbonus", "hientropy", "cliphigher", "klanchor"}


def advantage(rewards, group, P, G, objective, beta, tau, gate=None,
              lam=1.5, k_eff=8.0, kappa=1.0, pfloor=1.0 / 16):
    """Per-rollout weight A_i. This function IS the experiment.

    `gate`: (thresh, sizes) for the hardness-gated arm. sizes[gi] is that prompt's actual
    group size (>G for K=0 prompts that got a resample). A group is zeroed out -- no
    gradient -- once its solve rate K/size >= thresh: it's "solved enough", so it stops
    contributing. This is curriculum-by-masking rather than curriculum-by-resampling; the
    resampling itself happens in rollout_gated() below, before advantage() is ever called.
    """
    w = np.zeros(len(rewards), dtype=np.float64)
    for gi in range(P):
        m = group == gi
        rg = rewards[m]
        K = rg.sum()
        if gate is not None:
            thresh, sizes = gate
            if len(rg) and K / len(rg) >= thresh:
                continue                        # solved: zero gradient, skip this group
        if objective in GRPO_LIKE:
            if rg.std() > 1e-6:
                w[m] = (rg - rg.mean()) / (rg.std() + 1e-6)
        elif objective == "drgrpo":
            w[m] = rg - rg.mean()
        elif objective == "maxrl":
            # MaxRL maximizes J = log p, so grad J = E[r S]/p. Since E[S] = 0, subtracting
            # the baseline b = mean(r) leaves the expectation unchanged but cuts variance
            # and -- the part that matters -- restores a negative gradient on failures:
            # A_i = -1 whenever r_i = 0, and A_i = (1-p)/p when r_i = 1, so the group sums
            # to zero and the update is self-anchoring like GRPO. This is GRPO with the
            # MEAN in the denominator instead of the std, i.e. hard prompts are up-weighted
            # as 1/p rather than 1/sqrt(p).
            mu = rg.mean()
            w[m] = (rg - mu) / (mu + 1e-6)
        elif objective == "maxrl_nb":
            # The un-baselined estimator (1/K) sum r_i S_i. Same expectation as `maxrl`,
            # but strictly non-negative, so there is no anchoring term. Kept only as an
            # ablation: it isolates exactly what the baseline subtraction buys.
            if K > 0:
                w[m] = rg * (G / K)
        elif objective == "raft":
            if K > 0:
                w[m] = rg
        elif objective == "rwr":
            if K > 0:
                ew = np.exp((rg - rg.mean()) / tau)
                w[m] = ew / ew.mean()
        elif objective == "powmaxrl":
            # (a) alpha-fair family. A_i = (r_i - p)/p^lam ascends J_lam = p^(1-lam)/(1-lam)
            # for lam != 1, and log p at lam == 1. lam=0 is expected reward, lam=1 is MaxRL,
            # lam>1 weights hard prompts harder. Group sum stays zero for every lam because
            # sum(r_i - p) = 0, so the update stays self-anchoring.
            # p=0 makes the numerator zero too, so such a group contributes nothing (there
            # is no signal in it); the floor only guards small-but-nonzero p, where the
            # weight ~ p^-lam would otherwise explode and one group would own the batch.
            mu = rg.mean()
            if mu > 0:
                w[m] = (rg - mu) / max(mu, pfloor) ** lam
        elif objective == "geopassk":
            # (b) geometric weights over k: w_k = (1-g) g^(k-1) with pass@k = 1-(1-p)^k
            #   J = 1 - (1-g)(1-p)/(1-g(1-p)),   dJ/dp = (1-g)/(1-g(1-p))^2
            # so A_i = (r_i - p) * c/(c + g*p)^2 with c = 1-g = 1/k_eff. Unlike MaxRL's 1/p
            # this is BOUNDED: at p=0 it equals k_eff instead of diverging. It is the lam=2
            # power rule with a floor that falls out of the objective instead of being set
            # by hand -- so p=0 is the max-weight point, not a failure point.
            mu = rg.mean()
            c = 1.0 / k_eff
            g = 1.0 - c
            w[m] = (rg - mu) * c / (c + g * mu) ** 2
        elif objective == "difkl":
            # (c) difficulty-dependent forward/reverse KL mix. Reverse KL (z-scored, pushes
            # wrong samples down) is mode-seeking: buys pass@1, costs pass@k. Forward KL
            # (positive-only, never pushes anything down) is mass-covering. Weight them by
            # difficulty so easy prompts sharpen while hard prompts keep their coverage.
            mu = rg.mean()
            b = mu ** kappa
            sd = rg.std()
            a_rev = (rg - mu) / (sd + 1e-6) if sd > 1e-6 else rg * 0.0
            a_fwd = rg.astype(np.float64)      # forward-KL estimator ~ r_i * grad log pi
            w[m] = b * a_rev + (1.0 - b) * a_fwd
        elif objective == "entropic":
            e = np.exp(beta * rg)
            w[m] = e - e.mean()
        else:
            raise ValueError(objective)
    return w


def forward_stats(model, chunk, glens, pad, want_base=False, ref=None):
    """Per-token logp, entropy and gen-mask for a micro-batch of left-padded sequences."""
    m = max(x.numel() for x in chunk)
    inp = torch.full((len(chunk), m), pad, dtype=torch.long, device="cuda")
    gmask = torch.zeros((len(chunk), m), dtype=torch.bool, device="cuda")
    for r, (x, L) in enumerate(zip(chunk, glens)):
        inp[r, m - x.numel():] = x
        gmask[r, m - L:] = True
    attn = (inp != pad).long()
    tgt = inp[:, 1:]

    logits = model(input_ids=inp, attention_mask=attn).logits[:, :-1, :].float()
    lp = -F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                          reduction="none").view(tgt.shape)
    logZ = torch.logsumexp(logits, dim=-1)
    ent = logZ - (logits * logits.softmax(-1)).sum(-1)      # H = logZ - E[logit]
    blp = None
    if want_base:
        # Reference logprobs for the KL anchor. Under LoRA the base policy is recovered by
        # disabling the adapter; under a full finetune there is no adapter, so `ref` must be
        # a separately loaded frozen copy of the base model.
        with torch.no_grad():
            if ref is not None:
                bl = ref(input_ids=inp, attention_mask=attn).logits[:, :-1, :].float()
            else:
                with model.disable_adapter():
                    bl = model(input_ids=inp, attention_mask=attn).logits[:, :-1, :].float()
            blp = -F.cross_entropy(bl.reshape(-1, bl.size(-1)), tgt.reshape(-1),
                                   reduction="none").view(tgt.shape)
    return lp, ent, gmask[:, 1:].float(), blp


@torch.no_grad()
def generate_batched(model, tok, problems, n, args, pad, temp, micro):
    """n samples for each problem, batched ACROSS prompts (left-padded).

    Generating one prompt at a time leaves the GPU at ~35% utilization; batching prompts
    together is the single biggest speedup available here and needs no separate engine.
    Returns [[(prompt_ids, gen_ids, text)] * n] per problem, in problem order.
    """
    model.eval()
    model.config.use_cache = True
    texts_all = [[] for _ in problems]
    ids_all = [[] for _ in problems]
    prompt_ids = [tok(build_prompt(tok, p), return_tensors="pt",
                      add_special_tokens=False)["input_ids"][0] for p in problems]
    per = max(1, micro // n)                       # prompts per generate call
    for s in range(0, len(problems), per):
        chunk = prompt_ids[s:s + per]
        L = max(x.numel() for x in chunk)
        inp = torch.full((len(chunk), L), pad, dtype=torch.long, device="cuda")
        for r, x in enumerate(chunk):
            inp[r, L - x.numel():] = x             # left pad
        attn = (inp != pad).long()
        out = model.generate(input_ids=inp, attention_mask=attn, do_sample=True,
                             temperature=temp, top_p=args.top_p, top_k=0,
                             max_new_tokens=args.max_new, num_return_sequences=n,
                             pad_token_id=pad)
        for j, row in enumerate(out):              # ordered prompt-major: p0 x n, p1 x n, ...
            pi = s + j // n
            gen = row[L:]
            gen = gen[gen != pad]
            if gen.numel() == 0:
                gen = row[L:L + 1]
            ids_all[pi].append(gen)
            texts_all[pi].append(tok.decode(gen, skip_special_tokens=True))
    return prompt_ids, ids_all, texts_all


def rollout(model, tok, batch, G, args, pad, gate_resample=False):
    """Generate G completions for each (problem, gold, level). Returns groups + flat tensors.

    gate_resample: for the `gatehard` arm. Any prompt whose first G samples are ALL wrong
    (K=0, no gradient signal at all) gets G more samples appended, doubling that group's
    size to 2G. Solved groups (K/size >= gate_thresh) are masked to zero gradient in
    advantage(), not dropped here -- dropping would break the token-count denominator's
    bookkeeping, masking keeps it simple. Net effect: this arm does NOT hit the same
    generated-sequence budget as the matched arms -- it spends extra samples on hard
    prompts instead, which is the point. Total is logged and reported, not asserted.
    """
    pids, ids_all, texts_all = generate_batched(
        model, tok, [b[0] for b in batch], G, args, pad, args.temp, args.gen_micro)
    if gate_resample:
        zero = [gi for gi in range(len(batch)) if sum(reward(t, batch[gi][1])
                for t in texts_all[gi]) == 0]
        if zero:
            probs2 = [batch[gi][0] for gi in zero]
            _, ids2, texts2 = generate_batched(model, tok, probs2, G, args, pad,
                                               args.temp, args.gen_micro)
            for j, gi in enumerate(zero):
                ids_all[gi] += ids2[j]
                texts_all[gi] += texts2[j]
    fulls, glens, rewards, group, groups, sizes = [], [], [], [], [], []
    for gi, (prob, gold, lvl) in enumerate(batch):
        rs, ntoks = [], []
        for gen, t in zip(ids_all[gi], texts_all[gi]):
            r = reward(t, gold)
            fulls.append(torch.cat([pids[gi].to(gen.device), gen]))
            glens.append(int(gen.numel()))
            rewards.append(r)
            group.append(gi)
            rs.append(r)
            ntoks.append(int(gen.numel()))
        sizes.append(len(rs))
        groups.append({"prompt_idx": gi, "level": lvl, "texts": texts_all[gi], "rewards": rs,
                       "logps": [0.0] * len(rs), "ntoks": ntoks, "gold": gold})
    return fulls, glens, np.array(rewards), np.array(group), groups, sizes


def update(model, opt, buf, args, pad, P_eff, ref=None):
    """Optimizer pass over a buffer of rollouts. Returns (loss, entropy, gradnorm, n_steps).

    `--opt_bs` sequences per optimizer step. Default 0 means "the whole buffer", which is
    the standard RL behaviour (one step per rollout batch). The SFT ladder sets it small so
    that a flush is a real SFT pass with many gradient steps over the collected data, which
    is what BoN distillation actually does; a single averaged step over 15k sequences at
    lr 1e-6 would barely move the model and would not be the baseline we mean to test.
    """
    fulls, glens, rewards, group = buf["fulls"], buf["glens"], buf["rewards"], buf["group"]
    gate = (args.gate_thresh, buf.get("sizes")) if args.objective == "gatehard" else None
    w = advantage(rewards, group, P_eff, args.num_gen, "grpo" if args.objective == "gatehard"
                  else args.objective, args.beta, args.tau, gate,
                  args.pow_lambda, args.k_eff, args.kappa, args.pfloor)
    w_t = torch.tensor(w, dtype=torch.float32, device="cuda")
    want_base = args.objective == "klanchor" and args.kl_beta > 0

    if args.gate_mode == "drop":
        # Drop vs mask: a zeroed group contributes nothing to the gradient either way, but
        # under `mask` its tokens stay in `denom`, so the step shrinks as more groups get
        # gated -- an implicit lr decay tied to competence, which confounds the curriculum.
        # `drop` removes them before denom is computed, keeping the step size constant.
        # (With binary rewards and z-scored advantages, w==0 for a whole group exactly when
        # that group is gated or degenerate -- K=0 or K=G -- so this filter is precise.)
        keep = [i for i in range(len(fulls)) if w[i] != 0.0]
        if not keep:
            return 0.0, float("nan"), 0.0, 0
        fulls = [fulls[i] for i in keep]
        glens = [glens[i] for i in keep]
        w_t = w_t[keep]
        w = w[keep]

    N = len(fulls)
    opt_bs = args.opt_bs if args.opt_bs > 0 else N
    order = list(range(N))
    if args.opt_bs > 0:
        random.shuffle(order)                       # SFT pass: shuffle, RL: keep group order

    model.train()
    model.config.use_cache = False
    # For clipping to do anything the ratio must be against the policy as it was BEFORE the
    # first inner epoch. Cached per micro-batch slice on epoch 0; the slices are identical
    # across epochs because `order` is shuffled once, outside the mu loop.
    clipping = args.objective == "cliphigher" and args.mu > 1
    old_cache = {}
    ent_sum, tok_sum, lv, nsteps, gn = 0.0, 0.0, 0.0, 0, 0.0
    for ep in range(max(args.mu, 1)):
        for bs in range(0, N, opt_bs):
            idx = order[bs:bs + opt_bs]
            denom = float(sum(glens[i] for i in idx))
            opt.zero_grad()
            for s in range(0, len(idx), args.loss_micro):
                sub = idx[s:s + args.loss_micro]
                chunk = [fulls[i] for i in sub]
                gl = [glens[i] for i in sub]
                a = w_t[sub].unsqueeze(1)
                if float(a.abs().sum()) == 0.0 and args.objective != "entbonus":
                    continue                        # no signal in this micro-batch
                lp, ent, vm, blp = forward_stats(model, chunk, gl, pad, want_base, ref)

                m = vm
                if args.objective == "hientropy":
                    # 80/20 rule: keep only the top-q entropy ("forking") tokens.
                    v = ent[vm > 0]
                    if v.numel() > 0:
                        thr = torch.quantile(v.detach().float(), 1.0 - args.ent_frac)
                        m = vm * (ent.detach() >= thr).float()

                if clipping:
                    key = tuple(sub)
                    if ep == 0:
                        old_cache[key] = lp.detach().clone()
                    ratio = torch.exp(lp - old_cache[key])
                    unc = ratio * a
                    clipped = torch.clamp(ratio, 1 - args.eps_lo, 1 + args.eps_hi) * a
                    pol = -(torch.min(unc, clipped) * m).sum() / denom
                else:
                    pol = -(a * lp * m).sum() / denom

                loss = pol
                if args.objective == "entbonus":
                    loss = loss - args.ent_coef * (ent * vm).sum() / denom
                if want_base:
                    loss = loss + args.kl_beta * ((lp - blp) * vm).sum() / denom

                loss.backward()
                lv += float(loss.detach())
                ent_sum += float((ent.detach() * vm).sum())
                tok_sum += float(vm.sum())
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            nsteps += 1
    return lv, ent_sum / max(tok_sum, 1), gn, nsteps


def sample_probe(model, tok, probs, n, args, pad, temp=1.0):
    """n samples for each problem, used for both the fixed probe dump and the pass@k probe."""
    _, _, texts_all = generate_batched(model, tok, [p[0] for p in probs], n, args, pad,
                                       temp, args.gen_micro)
    out = []
    for (prob, gold, lvl), texts in zip(probs, texts_all):
        rs = [reward(t, gold) for t in texts]
        out.append({"problem": prob, "gold": gold, "level": lvl,
                    "texts": texts, "rewards": rs, "n": len(rs), "c": int(sum(rs))})
    return out


def pass_at_k(n, c, k):
    import math
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--objective", default="grpo", choices=OBJECTIVES)
    ap.add_argument("--flush_every", type=int, default=1, help="1=online, steps=offline BoN")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--n_prompts", type=int, default=16)
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--gen_micro", type=int, default=64, help="sequences per generate call")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_level", type=int, default=3)
    ap.add_argument("--loss_micro", type=int, default=2)
    ap.add_argument("--opt_bs", type=int, default=0,
                    help="sequences per optimizer step; 0 = whole buffer (RL). Set ~32 for "
                         "the SFT ladder so a flush is a real SFT pass, not one giant step")
    ap.add_argument("--lora", action="store_true", help="fallback if full FT OOMs")
    ap.add_argument("--beta", type=float, default=2.0, help="entropic risk-seeking beta")
    ap.add_argument("--tau", type=float, default=0.5, help="rwr temperature")
    ap.add_argument("--ent_coef", type=float, default=1e-3)
    ap.add_argument("--ent_frac", type=float, default=0.2, help="hientropy top-q fraction")
    ap.add_argument("--kl_beta", type=float, default=1e-3)
    ap.add_argument("--mu", type=int, default=1, help="inner epochs; >1 activates clipping")
    ap.add_argument("--pow_lambda", type=float, default=1.5,
                    help="powmaxrl: exponent lam in A=(r-p)/p^lam (lam=1 is MaxRL)")
    ap.add_argument("--pfloor", type=float, default=1.0 / 16,
                    help="powmaxrl: floor on p, stops the p^-lam blowup at small p")
    ap.add_argument("--k_eff", type=float, default=8.0,
                    help="geopassk: effective k; geometric weight g = 1 - 1/k_eff")
    ap.add_argument("--kappa", type=float, default=1.0,
                    help="difkl: mixing exponent, beta(p) = p^kappa")
    ap.add_argument("--max_rollouts", type=int, default=0,
                    help="stop once this many rollouts are generated (0 = no cap). Needed "
                         "for gatehard, whose K=0 resampling makes its sample count exceed "
                         "steps*P*G -- capping charges the resamples against the budget so "
                         "it is sample-matched with the other arms rather than 17%% richer")
    ap.add_argument("--gate_thresh", type=float, default=0.5,
                    help="gatehard: zero gradient for any group with K/size >= this")
    ap.add_argument("--gate_mode", default="mask", choices=["mask", "drop"],
                    help="mask keeps gated groups in the loss denominator (implicit lr "
                         "decay); drop removes them, holding step size constant")
    ap.add_argument("--eps_lo", type=float, default=0.2)
    ap.add_argument("--eps_hi", type=float, default=0.28)
    ap.add_argument("--probe_every", type=int, default=10)
    ap.add_argument("--passk_every", type=int, default=20)
    ap.add_argument("--passk_probs", type=int, default=50)
    ap.add_argument("--passk_n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7291)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.out_dir + "/args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        tok.pad_token = "<|endoftext|>"
    tok.padding_side = "left"
    pad = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=64, lora_dropout=0.0, task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
    ref = None
    if args.objective == "klanchor" and args.kl_beta > 0 and not args.lora:
        ref = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                   device_map="cuda", trust_remote_code=True)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, betas=(0.9, 0.95))

    train = load_train(4000, args.max_level, args.seed)
    probe_set = train[-8:]                       # fixed, never trained on (ptr never reaches it)
    passk_set = load_math500(args.passk_probs)
    run = Run(args.out_dir)
    print(f"[{args.objective}] {len(train)} prompts, P={args.n_prompts} G={args.num_gen} "
          f"flush_every={args.flush_every} budget={args.steps*args.n_prompts*args.num_gen}",
          flush=True)

    buf = {"fulls": [], "glens": [], "rewards": [], "group": [], "sizes": []}
    gbase = 0
    ptr = 0
    for step in range(args.steps):
        if args.max_rollouts and run.n_rollouts >= args.max_rollouts:
            print(f"[budget] stopping at step {step}: {run.n_rollouts} rollouts "
                  f">= cap {args.max_rollouts}", flush=True)
            break
        batch = [train[(ptr + i) % (len(train) - 8)] for i in range(args.n_prompts)]
        ptr += args.n_prompts
        fulls, glens, rewards, group, groups, sizes = rollout(
            model, tok, batch, args.num_gen, args, pad,
            gate_resample=(args.objective == "gatehard"))
        run.log_rollouts(step, groups)

        buf["fulls"] += fulls
        buf["glens"] += glens
        buf["rewards"].append(rewards)
        buf["group"].append(group + gbase)
        buf["sizes"] += sizes
        gbase += args.n_prompts

        ent = float("nan")
        if (step + 1) % args.flush_every == 0 or step == args.steps - 1:
            b = {"fulls": buf["fulls"], "glens": buf["glens"],
                 "rewards": np.concatenate(buf["rewards"]),
                 "group": np.concatenate(buf["group"]), "sizes": buf["sizes"]}
            lv, ent, gn, nsteps = update(model, opt, b, args, pad, gbase, ref)
            buf = {"fulls": [], "glens": [], "rewards": [], "group": [], "sizes": []}
            gbase = 0
            print(json.dumps({"step": step, "obj": args.objective, "loss": round(lv, 4),
                              "mean_r": round(float(rewards.mean()), 4),
                              "entropy": round(ent, 4), "gradnorm": round(gn, 3),
                              "opt_steps": nsteps, "n_rollouts": run.n_rollouts}), flush=True)
        else:
            print(json.dumps({"step": step, "obj": args.objective, "buffering": len(buf["fulls"]),
                              "mean_r": round(float(rewards.mean()), 4),
                              "n_rollouts": run.n_rollouts}), flush=True)

        run.log_diversity(step, groups, ent)

        if step % args.probe_every == 0 or step == args.steps - 1:
            run.log_probe(step, sample_probe(model, tok, probe_set, 8, args, pad))
        if step % args.passk_every == 0 or step == args.steps - 1:
            recs = sample_probe(model, tok, passk_set, args.passk_n, args, pad)
            curve = {k: sum(pass_at_k(r["n"], r["c"], k) for r in recs) / len(recs)
                     for k in (1, 2, 4, 8, 16) if k <= args.passk_n}
            run.log_passk_probe(step, curve)
            print("PASSK_PROBE", step, json.dumps(curve), flush=True)

    if args.objective != "gatehard" and not args.max_rollouts:
        assert run.n_rollouts == args.steps * args.n_prompts * args.num_gen, \
            f"budget mismatch: {run.n_rollouts}"
    else:
        print(f"[gatehard] final n_rollouts={run.n_rollouts} "
              f"(nominal {args.steps*args.n_prompts*args.num_gen}, extra spent on K=0 prompts)")
    d = args.out_dir + "/final"
    if args.lora:
        model = model.merge_and_unload()
    model.save_pretrained(d)
    tok.save_pretrained(d)
    print("SAVED", d, "rollouts", run.n_rollouts, flush=True)


if __name__ == "__main__":
    main()
