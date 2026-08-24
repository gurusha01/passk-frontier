"""BonBon and BOND / J-BOND, implemented from the papers.

BonBon  -- Gui, Garbacea, Veitch, NeurIPS 2024 (arXiv 2406.00832)
  L_SFT-BoN  = -E[log pi_th(y_(n))]                                            (4.1)
  L_IPO-BoN  = E[( log(pi_th(y_(n))/pi_th(y_(1)))
                 - log(pi_0(y_(n))/pi_0(y_(1))) - 1/(2 beta*_n) )^2]           (4.3)
  beta*_n    = 1 / (2 (n-1) sum_{k=1}^{n-1} 1/k)   ->  1/(2 beta*_n) = 18.15 at n=8   (4.2)
  L_BonBon   = alpha L_SFT-BoN + (1-alpha) L_IPO-BoN,  alpha = 0.005           (4.4)
  y_(n) = best of n, y_(1) = worst of n, both drawn from the reference pi_0. Offline.

BOND -- Sessa et al., 2024 (arXiv 2407.14622)
  Jeffreys:   J^beta = (1-beta) KL(pi_BoN || pi) + beta KL(pi || pi_BoN), beta=0.5  (11)
  forward :   grad KL(pi_BoN||pi) = -E_{y~pi_BoN} grad log pi(y)  -> SFT on best-of-N   (13)
  backward:   grad KL(pi||pi_BoN) = -(N-1) E_{y~pi}[ grad log pi(y)
                  ( r_BOND(y) - beta_BOND (log pi(y) - log pi_ref(y)) ) ]            (15)
  r_BOND(y) = log p_le(y),  beta_BOND = 1/(N-1),  p_le by Monte-Carlo             (8,10)

J-BOND -- Algorithm 2. n=2, EMA anchor, and a crude 2-sample reward:
  r_J-BOND(y) = -log(16) if r(y) < min(r(y'_1), r(y'_2)) else 0                     (17)
  theta_anchor <- (1-eta) theta_anchor + eta theta                                  (18)
  update: (1-beta) G_FW + beta G_BW + gamma G_Reg

DEVIATION, stated because it matters: both papers assume the reward induces a strict
ordering. Ours is binary (math_verify), so best/worst-of-n are ties. We break ties by
shorter completion, which is a defensible secondary criterion and also pushes against the
length inflation the earlier runs showed. BOND's backward KL is unaffected: with binary
reward p_le(y) = 1 when y is correct and 1 - p_correct when it is not.
"""
import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HOME", "${PF_HF_HOME}")

from data import build_prompt, load_math500, load_train, reward  # noqa: E402
from log import Run  # noqa: E402
from train import generate_batched, pass_at_k  # noqa: E402


def harmonic(m):
    return sum(1.0 / k for k in range(1, m + 1))


def ipo_margin(n):
    """1/(2 beta*_n) = (n-1) * H_{n-1}  -- BonBon eq (4.2). 18.15 at n=8."""
    return (n - 1) * harmonic(n - 1)


def rank_key(text, r):
    """(reward, -length) -- reward first, shorter wins ties. See DEVIATION in docstring."""
    return (r, -len(text))


def seq_logp(model, fulls, glens, pad, micro, grad=True):
    """Sum of token logprobs per sequence: log pi(y|x). Returns a 1-D tensor."""
    out = []
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for s in range(0, len(fulls), micro):
            chunk, gl = fulls[s:s + micro], glens[s:s + micro]
            m = max(x.numel() for x in chunk)
            inp = torch.full((len(chunk), m), pad, dtype=torch.long, device="cuda")
            gm = torch.zeros((len(chunk), m), dtype=torch.bool, device="cuda")
            for r, (x, L) in enumerate(zip(chunk, gl)):
                inp[r, m - x.numel():] = x
                gm[r, m - L:] = True
            lg = model(input_ids=inp, attention_mask=(inp != pad).long()).logits[:, :-1, :].float()
            lp = -F.cross_entropy(lg.reshape(-1, lg.size(-1)), inp[:, 1:].reshape(-1),
                                  reduction="none").view(inp[:, 1:].shape)
            out.append((lp * gm[:, 1:].float()).sum(-1))
    return torch.cat(out)


def lnorm(lp, lens, mode):
    """Sequence log-prob, optionally divided by length.

    The papers' gradients are sequence-level sums (mode="seq"), which is the faithful
    choice and what we default to. mode="token" divides by completion length; it is the
    common modern practice and it removes the length bias that inflated completions in the
    earlier RAFT runs, so it is offered as an explicit, swept alternative rather than a
    silent default.
    """
    return lp / lens if mode == "token" else lp


def cat_full(pid, gen):
    return torch.cat([pid.to(gen.device), gen])


# --------------------------------------------------------------------------- BonBon
def run_bonbon(model, ref, tok, opt, train, args, pad, run):
    """Offline: sample n from pi_0 once per prompt, keep (best, worst), then train."""
    n = args.num_gen
    margin = ipo_margin(n)
    nprompt = args.budget // n
    print(f"[bonbon] n={n} margin=1/(2b*)={margin:.3f} prompts={nprompt} alpha={args.alpha}",
          flush=True)

    pairs = []
    for s in range(0, nprompt, args.gen_prompts):
        batch = train[s:s + args.gen_prompts]
        if not batch:
            break
        pids, ids, texts = generate_batched(ref, tok, [b[0] for b in batch], n, args,
                                            pad, args.temp, args.gen_micro)
        for gi, (prob, gold, lvl) in enumerate(batch):
            rs = [reward(t, gold) for t in texts[gi]]
            order = sorted(range(n), key=lambda i: rank_key(texts[gi][i], rs[i]))
            lo, hi = order[0], order[-1]
            if rs[hi] == rs[lo]:
                continue                      # no preference signal in this group
            pairs.append({"best": cat_full(pids[gi], ids[gi][hi]),
                          "worst": cat_full(pids[gi], ids[gi][lo]),
                          "bl": int(ids[gi][hi].numel()), "wl": int(ids[gi][lo].numel())})
            run.n_rollouts += 0
        run.log_rollouts(s, [{"prompt_idx": gi, "level": b[2], "texts": texts[gi],
                              "rewards": [reward(t, b[1]) for t in texts[gi]],
                              "logps": [0.0] * n, "ntoks": [int(x.numel()) for x in ids[gi]]}
                             for gi, b in enumerate(batch)])
        print(json.dumps({"phase": "collect", "prompts": s + len(batch),
                          "pairs": len(pairs), "n_rollouts": run.n_rollouts}), flush=True)
    print(f"[bonbon] {len(pairs)} usable pairs from {nprompt} prompts", flush=True)

    # reference log-ratios are fixed: precompute once
    with torch.no_grad():
        for s in range(0, len(pairs), args.loss_micro):
            ch = pairs[s:s + args.loss_micro]
            b = seq_logp(ref, [c["best"] for c in ch], [c["bl"] for c in ch], pad,
                         args.loss_micro, grad=False)
            w = seq_logp(ref, [c["worst"] for c in ch], [c["wl"] for c in ch], pad,
                         args.loss_micro, grad=False)
            for c, x in zip(ch, (b - w).tolist()):
                c["ref_ratio"] = x

    order = list(range(len(pairs)))
    random.shuffle(order)
    step = 0
    for ep in range(args.epochs):
        for s in range(0, len(order), args.opt_bs):
            idx = order[s:s + args.opt_bs]
            opt.zero_grad()
            lsft = lipo = 0.0
            for t in range(0, len(idx), args.loss_micro):
                ch = [pairs[i] for i in idx[t:t + args.loss_micro]]
                lb = seq_logp(model, [c["best"] for c in ch], [c["bl"] for c in ch],
                              pad, args.loss_micro)
                lw = seq_logp(model, [c["worst"] for c in ch], [c["wl"] for c in ch],
                              pad, args.loss_micro)
                rr = torch.tensor([c["ref_ratio"] for c in ch], device="cuda")
                # eq (4.1) is the plain log-likelihood of the best sample, i.e. the token
                # SUM, and eq (4.3) is a squared difference of sequence log-ratios. Both
                # are sequence-level, so keep them that way and let alpha do the balancing.
                bl = torch.tensor([c["bl"] for c in ch], device="cuda").float()
                sft = -(lb / bl).mean() if args.len_norm == "token" else -lb.mean()
                ipo = (((lb - lw) - rr - margin) ** 2).mean()
                loss = (args.alpha * sft + (1 - args.alpha) * ipo) / max(
                    1, (len(idx) + args.loss_micro - 1) // args.loss_micro)
                loss.backward()
                lsft += float(sft.detach()); lipo += float(ipo.detach())
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            opt.step()
            print(json.dumps({"step": step, "ep": ep, "sft": round(lsft, 3),
                              "ipo": round(lipo, 3), "gradnorm": round(gn, 3)}), flush=True)
            step += 1
            maybe_probe(model, tok, args, pad, run, step)
    return step


# ----------------------------------------------------------------------------- BOND
def run_bond(model, ref, tok, opt, train, args, pad, run):
    """Non-iterative BOND: Jeffreys(pi || BoN(pi_ref)) with MC quantiles."""
    N, beta = args.num_gen, args.jeffreys
    bB = 1.0 / (N - 1)
    per = N + 1                                  # N ref samples (BoN + quantile) + 1 policy
    steps = args.budget // (args.gen_prompts * per)
    print(f"[bond] N={N} beta={beta} beta_BOND={bB:.4f} steps={steps}", flush=True)

    ptr = 0
    for step in range(steps):
        batch = [train[(ptr + i) % len(train)] for i in range(args.gen_prompts)]
        ptr += args.gen_prompts
        probs = [b[0] for b in batch]
        rp, ri, rt = generate_batched(ref, tok, probs, N, args, pad, args.temp, args.gen_micro)
        pp, pi_, pt = generate_batched(model, tok, probs, 1, args, pad, args.temp, args.gen_micro)

        bon_f, bon_l, pol_f, pol_l, rbond, groups = [], [], [], [], [], []
        for gi, (prob, gold, lvl) in enumerate(batch):
            rr = [reward(t, gold) for t in rt[gi]]
            hi = max(range(N), key=lambda i: rank_key(rt[gi][i], rr[i]))
            bon_f.append(cat_full(rp[gi], ri[gi][hi])); bon_l.append(int(ri[gi][hi].numel()))
            pr = reward(pt[gi][0], gold)
            pol_f.append(cat_full(pp[gi], pi_[gi][0])); pol_l.append(int(pi_[gi][0].numel()))
            # p_le(y) by Monte-Carlo over the N reference samples -- eq (10)
            ple = sum(1 for x in rr if x <= pr) / N
            rbond.append(math_log(max(ple, 1.0 / (2 * N))))
            groups.append({"prompt_idx": gi, "level": lvl, "texts": rt[gi],
                           "rewards": rr, "logps": [0.0] * N,
                           "ntoks": [int(x.numel()) for x in ri[gi]]})
        run.log_rollouts(step, groups)
        run.n_rollouts += args.gen_prompts       # count the policy samples too

        opt.zero_grad()
        # forward KL: SFT on the best-of-N reference samples (eq 13)
        nmb = max(1, (len(bon_f) + args.loss_micro - 1) // args.loss_micro)
        for s in range(0, len(bon_f), args.loss_micro):
            lp = seq_logp(model, bon_f[s:s + args.loss_micro], bon_l[s:s + args.loss_micro],
                          pad, args.loss_micro)
            tk = torch.tensor(bon_l[s:s + args.loss_micro], device="cuda").float()
            ((1 - beta) * (-lnorm(lp, tk, args.len_norm).mean()) / nmb).backward()
        # backward KL: policy gradient with the BOND reward (eq 15)
        with torch.no_grad():
            ref_lp = seq_logp(ref, pol_f, pol_l, pad, args.loss_micro, grad=False)
        R = []
        for s in range(0, len(pol_f), args.loss_micro):
            with torch.no_grad():
                cur = seq_logp(model, pol_f[s:s + args.loss_micro],
                               pol_l[s:s + args.loss_micro], pad, args.loss_micro, grad=False)
            rb = torch.tensor(rbond[s:s + args.loss_micro], device="cuda")
            R.append((N - 1) * rb - (cur - ref_lp[s:s + args.loss_micro]))
        R = torch.cat(R)
        B = R.mean()                              # batch baseline
        for s in range(0, len(pol_f), args.loss_micro):
            lp = seq_logp(model, pol_f[s:s + args.loss_micro], pol_l[s:s + args.loss_micro],
                          pad, args.loss_micro)
            tk = torch.tensor(pol_l[s:s + args.loss_micro], device="cuda").float()
            adv = (R[s:s + args.loss_micro] - B).detach()
            (beta * (-(lnorm(lp, tk, args.len_norm) * adv).mean()) / nmb).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        print(json.dumps({"step": step, "mean_rbond": round(float(np.mean(rbond)), 4),
                          "mean_R": round(float(R.mean()), 3), "gradnorm": round(gn, 3),
                          "n_rollouts": run.n_rollouts}), flush=True)
        maybe_probe(model, tok, args, pad, run, step)
    return steps


def math_log(x):
    import math
    return math.log(x)


# --------------------------------------------------------------------------- J-BOND
def run_jbond(model, anchor, tok, opt, train, args, pad, run):
    """Algorithm 2: 1 policy sample + 2 anchor samples per prompt, EMA anchor."""
    beta, gamma, eta = args.jeffreys, args.gamma, args.eta
    NEG = math_log(16.0)
    per = 3
    steps = args.budget // (args.gen_prompts * per)
    print(f"[jbond] beta={beta} gamma={gamma} eta={eta} steps={steps}", flush=True)

    ptr = 0
    for step in range(steps):
        batch = [train[(ptr + i) % len(train)] for i in range(args.gen_prompts)]
        ptr += args.gen_prompts
        probs = [b[0] for b in batch]
        ap, ai, at = generate_batched(anchor, tok, probs, 2, args, pad, args.temp, args.gen_micro)
        pp, pi_, pt = generate_batched(model, tok, probs, 1, args, pad, args.temp, args.gen_micro)

        bo2_f, bo2_l, pol_f, pol_l, rj, groups = [], [], [], [], [], []
        for gi, (prob, gold, lvl) in enumerate(batch):
            ar = [reward(t, gold) for t in at[gi]]
            hi = max(range(2), key=lambda i: rank_key(at[gi][i], ar[i]))
            bo2_f.append(cat_full(ap[gi], ai[gi][hi])); bo2_l.append(int(ai[gi][hi].numel()))
            pr = reward(pt[gi][0], gold)
            pol_f.append(cat_full(pp[gi], pi_[gi][0])); pol_l.append(int(pi_[gi][0].numel()))
            rj.append(-NEG if pr < min(ar) else 0.0)          # eq (17)
            groups.append({"prompt_idx": gi, "level": lvl, "texts": at[gi] + pt[gi],
                           "rewards": ar + [pr], "logps": [0.0] * 3,
                           "ntoks": [int(x.numel()) for x in ai[gi]] + [pol_l[-1]]})
        run.log_rollouts(step, groups)

        opt.zero_grad()
        nmb = max(1, (len(bo2_f) + args.loss_micro - 1) // args.loss_micro)
        # forward KL: SFT on the best-of-2 anchor sample
        for s in range(0, len(bo2_f), args.loss_micro):
            lp = seq_logp(model, bo2_f[s:s + args.loss_micro], bo2_l[s:s + args.loss_micro],
                          pad, args.loss_micro)
            tk = torch.tensor(bo2_l[s:s + args.loss_micro], device="cuda").float()
            ((1 - beta) * (-lnorm(lp, tk, args.len_norm).mean()) / nmb).backward()
        # backward KL + extra KL regularisation, both against the moving anchor
        with torch.no_grad():
            anc_lp = seq_logp(anchor, pol_f, pol_l, pad, args.loss_micro, grad=False)
            cur_lp = seq_logp(model, pol_f, pol_l, pad, args.loss_micro, grad=False)
        R = torch.tensor(rj, device="cuda") - (cur_lp - anc_lp)
        B = R.mean()
        klv = (cur_lp - anc_lp)
        for s in range(0, len(pol_f), args.loss_micro):
            lp = seq_logp(model, pol_f[s:s + args.loss_micro], pol_l[s:s + args.loss_micro],
                          pad, args.loss_micro)
            tk = torch.tensor(pol_l[s:s + args.loss_micro], device="cuda").float()
            adv = (R[s:s + args.loss_micro] - B).detach()
            reg = klv[s:s + args.loss_micro].detach()
            ln = lnorm(lp, tk, args.len_norm)
            ((beta * (-(ln * adv).mean())
              + gamma * (-(ln * reg).mean())) / nmb).backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        with torch.no_grad():                       # EMA anchor, eq (18)
            for pa, pm in zip(anchor.parameters(), model.parameters()):
                pa.mul_(1 - eta).add_(pm.detach(), alpha=eta)
        print(json.dumps({"step": step, "frac_penalised": round(float(np.mean([x < 0 for x in rj])), 3),
                          "mean_kl": round(float(klv.mean()), 3), "gradnorm": round(gn, 3),
                          "n_rollouts": run.n_rollouts}), flush=True)
        maybe_probe(model, tok, args, pad, run, step)
    return steps


# ---------------------------------------------------------------------------- probes
_PROBE = {}


def maybe_probe(model, tok, args, pad, run, step):
    from train import sample_probe
    if args.probe_every and step % args.probe_every == 0:
        run.log_probe(step, sample_probe(model, tok, _PROBE["fixed"], 8, args, pad))
    if args.passk_every and step % args.passk_every == 0:
        recs = sample_probe(model, tok, _PROBE["passk"], args.passk_n, args, pad)
        curve = {k: sum(pass_at_k(r["n"], r["c"], k) for r in recs) / len(recs)
                 for k in (1, 2, 4, 8, 16) if k <= args.passk_n}
        run.log_passk_probe(step, curve)
        print("PASSK_PROBE", step, json.dumps(curve), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["bonbon", "bond", "jbond"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lr", type=float, default=3e-6)
    ap.add_argument("--budget", type=int, default=15360, help="generated sequences, all methods")
    ap.add_argument("--num_gen", type=int, default=8, help="n for BonBon / N for BOND")
    ap.add_argument("--gen_prompts", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.005, help="BonBon eq (4.4)")
    ap.add_argument("--jeffreys", type=float, default=0.5, help="beta in eq (11)")
    ap.add_argument("--gamma", type=float, default=0.1, help="J-BOND extra KL reg, eq (19)")
    ap.add_argument("--eta", type=float, default=0.02, help="EMA anchor rate, eq (18)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--opt_bs", type=int, default=16)
    ap.add_argument("--loss_micro", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=768)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--gen_micro", type=int, default=64)
    ap.add_argument("--max_level", type=int, default=3)
    ap.add_argument("--probe_every", type=int, default=20)
    ap.add_argument("--passk_every", type=int, default=40)
    ap.add_argument("--passk_probs", type=int, default=50)
    ap.add_argument("--passk_n", type=int, default=16)
    ap.add_argument("--len_norm", default="seq", choices=["seq", "token"],
                    help="seq = faithful sequence-level gradient; token = length-normalised")
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7291)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(args.out_dir + "/args.json", "w"), indent=2)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None or tok.pad_token_id == tok.eos_token_id:
        tok.pad_token = "<|endoftext|>"
    tok.padding_side = "left"
    pad = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="cuda", trust_remote_code=True)
    frozen = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                  device_map="cuda", trust_remote_code=True)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    train = load_train(4000, args.max_level, args.seed)
    _PROBE["fixed"] = train[-8:]
    _PROBE["passk"] = load_math500(args.passk_probs)
    run = Run(args.out_dir)

    if args.method == "bonbon":
        run_bonbon(model, frozen, tok, opt, train[:-8], args, pad, run)
    elif args.method == "bond":
        run_bond(model, frozen, tok, opt, train[:-8], args, pad, run)
    else:
        run_jbond(model, frozen, tok, opt, train[:-8], args, pad, run)

    d = args.out_dir + "/final"
    model.save_pretrained(d)
    tok.save_pretrained(d)
    print("SAVED", d, "rollouts", run.n_rollouts, flush=True)


if __name__ == "__main__":
    main()
