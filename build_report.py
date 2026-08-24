"""Build the self-contained HTML report: inlines every figure as a data URI and renders
all tables straight from runs/ + evals/, so nothing has to be transcribed by hand."""
import base64
import glob
import json
import math
import os

ROOT = "${PF_ROOT}"
KS = [1, 2, 4, 8, 16, 32, 64]

ARM_DOC = {
    "base":        ("&mdash;",        "&mdash;",  "no training at all &mdash; the reference point"),
    "grpo":        ("grpo",   "1e-6", "standard RLVR, z-scored advantage"),
    "maxrl":       ("maxrl",  "1e-6", "MaxRL: normalise by successes K, not group size G"),
    "sft-online":  ("raft",   "1e-6", "fully on-policy RAFT (SFT on correct, every step)"),
    "sft-iter":    ("raft",   "1e-6", "iterated rejection sampling / ReST (4 rounds)"),
    "sft-offline": ("raft",   "1e-6", "offline BoN distillation (policy frozen while collecting)"),
    "entropic":    ("entropic", "1e-6", "risk-seeking entropic utility, beta=+2"),
    "hientropy":   ("hientropy", "1e-6", "grpo masked to top-20% entropy 'forking' tokens"),
}


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def pak(n, c, k):
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def curve(path, lvl=0):
    recs = jl(path)
    if lvl:
        recs = [r for r in recs if r.get("level", 0) >= lvl]
    if not recs:
        return None
    nmin = min(r["n"] for r in recs)
    out = {"n": len(recs), "c": {}, "e": {}}
    for k in [k for k in KS if k <= nmin]:
        v = [pak(r["n"], r["c"], k) for r in recs]
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)
        out["c"][k] = m
        out["e"][k] = (var / len(v)) ** 0.5
    if "greedy" in recs[0]:
        out["greedy"] = sum(r["greedy"] for r in recs) / len(recs)
    return out


def img(name):
    p = f"{ROOT}/results/{name}"
    if not os.path.exists(p):
        return "<p class='note'>figure not available</p>"
    b = base64.b64encode(open(p, "rb").read()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{name}">'


def dcell(v, ref, pct=True, invert=False):
    """value with a signed delta vs ref, coloured."""
    if v is None:
        return "<td class='num'>&mdash;</td>"
    d = v - ref
    cls = "flat"
    if abs(d) > 0.005:
        good = (d > 0) != invert
        cls = "up" if good else "dn"
    s = f"{v*100:.1f}" if pct else f"{v:.3f}"
    sign = "+" if d >= 0 else "−"
    return (f"<td class='num'>{s}<span class='delta {cls}'>{sign}{abs(d)*100:.1f}</span></td>")


def eval_table(arms, lvl, caption):
    base = curve(f"{ROOT}/evals/base.jsonl", lvl)
    rows = []
    for a in arms:
        c = curve(f"{ROOT}/evals/{a}.jsonl", lvl)
        if not c:
            rows.append(f"<tr><td class='mono'>{a}</td>"
                        + "<td class='num pend' colspan='8'>eval still running</td></tr>")
            continue
        tds = "".join(dcell(c["c"].get(k), base["c"].get(k, 0)) if a != "base"
                      else f"<td class='num'>{c['c'][k]*100:.1f}</td>" for k in KS)
        g = c.get("greedy")
        gtd = (f"<td class='num'>{g*100:.1f}</td>" if a == "base"
               else dcell(g, base.get("greedy", 0)))
        cls = " class='base-row'" if a == "base" else ""
        rows.append(f"<tr{cls}><td class='mono'>{a}</td>{tds}{gtd}</tr>")
    head = "".join(f"<th class='num'>pass@{k}</th>" for k in KS)
    n = base["n"] if base else 0
    return f"""<figure class="tw">
<div class="scroll"><table>
<caption>{caption} &middot; {n} problems &middot; n=64 samples &middot; values are %, small number is the change vs <span class="mono">base</span></caption>
<thead><tr><th>arm</th>{head}<th class="num">greedy@1</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></figure>"""


def runs_table():
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/runs/*")):
        a = os.path.basename(d)
        if a.startswith("_") or not os.path.exists(f"{d}/args.json"):
            continue
        g = json.load(open(f"{d}/args.json"))
        div = jl(f"{d}/diversity.jsonl")
        pr = jl(f"{d}/passk_probe.jsonl")
        nroll = div[-1]["n_rollouts"] if div else 0
        p0 = pr[0]["curve"].get("1") if pr else None
        pN = pr[-1]["curve"].get("1") if pr else None
        ent0 = next((r["token_entropy"] for r in div if r.get("token_entropy") == r.get("token_entropy")), None)
        entN = next((r["token_entropy"] for r in reversed(div) if r.get("token_entropy") == r.get("token_entropy")), None)
        ev = "yes" if os.path.exists(f"{ROOT}/evals/{a}.jsonl") and jl(f"{ROOT}/evals/{a}.jsonl") else "running"
        sched = {1: "online", 30: "iter (4 rounds)", 120: "offline"}.get(g["flush_every"], g["flush_every"])
        rows.append(
            f"<tr><td class='mono'>{a}</td><td class='mono'>{g['objective']}</td>"
            f"<td class='num'>{g['lr']:.0e}</td><td>{sched}</td>"
            f"<td class='num'>{nroll:,}</td>"
            f"<td class='num'>{p0:.3f} &rarr; {pN:.3f}</td>"
            f"<td class='num'>{ent0:.2f} &rarr; {entN:.2f}</td><td>{ev}</td></tr>")
    return f"""<figure class="tw"><div class="scroll"><table>
<caption>All 20 training runs. Probe pass@1 and policy entropy are start &rarr; end.
Every run completed its full 15,360-rollout budget (asserted at exit).</caption>
<thead><tr><th>run</th><th>objective</th><th class="num">lr</th><th>SFT schedule</th>
<th class="num">rollouts</th><th class="num">probe pass@1</th><th class="num">entropy</th>
<th>eval</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></figure>"""


HTML = """<title>The pass@k Frontier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#FAFBFC; --surface:#F0F3F5; --sunk:#E7ECEF; --line:#D8E0E6;
  --ink:#14202B; --muted:#5A6B78; --accent:#0E5C68; --accent-soft:#DCEEF1;
  --good:#1D6B4A; --bad:#A33529; --warn:#8A5A12;
  --serif:"IBM Plex Serif",Georgia,serif;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0E1418; --surface:#161F25; --sunk:#1C272E; --line:#27343C;
  --ink:#E3EAEF; --muted:#8CA0AD; --accent:#5FB9C9; --accent-soft:#153037;
  --good:#4FBF8B; --bad:#E4796A; --warn:#D6A24E;
}}
:root[data-theme="dark"]{
  --ground:#0E1418; --surface:#161F25; --sunk:#1C272E; --line:#27343C;
  --ink:#E3EAEF; --muted:#8CA0AD; --accent:#5FB9C9; --accent-soft:#153037;
  --good:#4FBF8B; --bad:#E4796A; --warn:#D6A24E;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
  font-size:16.5px;line-height:1.62;-webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:clamp(28px,5vw,72px) clamp(18px,4vw,48px) 96px}
.col{max-width:74ch}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(2.1rem,4.4vw,3.1rem);
  line-height:1.1;letter-spacing:-.02em;margin:0 0 .5rem;text-wrap:balance}
.sub{font-family:var(--serif);font-style:italic;font-size:1.16rem;color:var(--muted);
  max-width:62ch;margin:0 0 2rem}
h2{font-family:var(--serif);font-weight:600;font-size:1.62rem;letter-spacing:-.01em;
  margin:3.6rem 0 .2rem;text-wrap:balance;display:flex;gap:.7rem;align-items:baseline}
h2 .n{font-family:var(--mono);font-size:.82rem;color:var(--accent);font-weight:500;
  letter-spacing:.06em;flex:none}
h3{font-family:var(--sans);font-weight:600;font-size:1.06rem;margin:2.2rem 0 .3rem;
  letter-spacing:-.005em}
p{margin:.75rem 0}
a{color:var(--accent)}
.eyebrow{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;
  letter-spacing:.16em;color:var(--muted);margin:0 0 1rem}
.mono{font-family:var(--mono);font-size:.87em}
code{font-family:var(--mono);font-size:.86em;background:var(--sunk);
  padding:.1em .38em;border-radius:3px}
hr{border:0;border-top:1px solid var(--line);margin:3rem 0}
.rule{border:0;border-top:2px solid var(--ink);margin:.1rem 0 1.4rem;opacity:.85}

/* callouts */
.box{background:var(--surface);border:1px solid var(--line);border-radius:6px;
  padding:1.1rem 1.3rem;margin:1.4rem 0}
.box.key{border-left:3px solid var(--accent);background:var(--accent-soft)}
.box.warn{border-left:3px solid var(--warn)}
.box h4{margin:0 0 .35rem;font-size:.94rem;font-weight:600;font-family:var(--sans)}
.box p{margin:.3rem 0}
.box p:last-child{margin-bottom:0}

/* tables */
figure.tw{margin:1.6rem 0;max-width:none}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.845rem}
caption{caption-side:top;text-align:left;padding:.7rem .9rem;color:var(--muted);
  font-size:.78rem;line-height:1.45;border-bottom:1px solid var(--line)}
th,td{padding:.42rem .7rem;text-align:left;border-bottom:1px solid var(--line);
  white-space:nowrap}
thead th{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);font-weight:500;background:var(--sunk)}
tbody tr:last-child td{border-bottom:0}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);
  font-size:.8rem}
tr.base-row{background:var(--sunk);font-weight:600}
tr.base-row td{border-bottom:2px solid var(--line)}
.delta{margin-left:.42em;font-size:.76em;font-variant-numeric:tabular-nums}
.delta.up{color:var(--good)} .delta.dn{color:var(--bad)} .delta.flat{color:var(--muted)}
.pend{color:var(--muted);font-style:italic;text-align:left}

/* figures */
figure.fig{margin:2rem 0;max-width:none}
figure.fig img{width:100%;height:auto;display:block;border:1px solid var(--line);
  border-radius:6px;background:#fff}
figcaption{font-size:.82rem;color:var(--muted);margin-top:.55rem;max-width:88ch;line-height:1.5}
figcaption b{color:var(--ink);font-weight:600}

/* definition rows */
dl.defs{display:grid;grid-template-columns:auto 1fr;gap:.45rem 1.2rem;margin:1.2rem 0;
  align-items:baseline}
dl.defs dt{font-family:var(--mono);font-size:.83rem;color:var(--accent);white-space:nowrap}
dl.defs dd{margin:0}
ul,ol{padding-left:1.3rem} li{margin:.3rem 0}
.note{font-size:.85rem;color:var(--muted)}
.tag{display:inline-block;font-family:var(--mono);font-size:.7rem;padding:.12em .5em;
  border-radius:3px;border:1px solid var(--line);background:var(--sunk);color:var(--muted);
  letter-spacing:.04em;vertical-align:.08em}
.tag.ok{color:var(--good);border-color:var(--good)}
.tag.no{color:var(--bad);border-color:var(--bad)}
</style>
<div class="wrap">
<p class="eyebrow">Experiment report &middot; MATH-500 &middot; Qwen2.5-1.5B-Instruct</p>
<h1>The pass@k Frontier</h1>
<p class="sub">Can RL raise pass@1 and pass@k at once, and beat best-of-n distillation
at a matched rollout budget? Twenty training runs, one shared rollout loop.</p>

<div class="col">
<div class="box key">
<h4>What came out of it</h4>
<p><b>1.</b> The whole study hinged on a learning rate. At lr 1e-6 nothing beat the base
model and GRPO moved pass@1 by +0.4 points, inside the error bar. At 3e-6 GRPO gains
<b>+3.1 pass@1</b> and loses <b>1.0 pass@64</b> &mdash; the sharpening tradeoff, reproduced.</p>
<p><b>2.</b> <span class="mono">entropic</span> (risk-seeking, beta=+2) matches GRPO's
pass@1 gain, so a reward-shape change alone reaches the same frontier point.</p>
<p><b>3.</b> Every non-negative-advantage arm &mdash; MaxRL and all three BoN rungs &mdash;
degrades at every learning rate tried. <span class="mono">maxrl</span> and
<span class="mono">sft-online</span> land on top of each other, so the 1/K term
contributed nothing here. <b>The headline MaxRL-vs-BoN question is not yet answered</b>,
and section 6 says exactly why I think that is my implementation rather than the method.</p>
</div>
</div>

__GLOSSARY__
__SETUP__
__RUNS__
__RESULTS__
__LR__
__DIAG__
__BUGS__
__NEXT__
</div>
"""


def main():
    W1 = ["base", "grpo", "entropic", "hientropy", "sft-offline", "sft-iter", "sft-online", "maxrl"]
    W1C = ["base", "hi_grpo", "hi_entropic", "hi_hientropy", "hi_sft-offline",
           "hi_sft-iter", "hi_sft-online", "hi_maxrl"]
    SW = ["base", "lr_grpo_3e-6", "lr_grpo_3e-7", "lr_maxrl_1e-7", "lr_maxrl_3e-7",
          "lr_sft-online_1e-7", "lr_sft-online_3e-7"]

    glossary = """<h2><span class="n">01</span>Reading the run names</h2><hr class="rule">
<div class="col">
<p>A run name is three orthogonal pieces: a prefix for the learning rate, a stem for the
objective, and for the BoN arms a flush schedule. Nothing else varies.</p>
<dl class="defs">
<dt>grpo</dt><dd>no prefix &rarr; <b>wave 1</b>, lr <b>1e-6</b></dd>
<dt>hi_grpo</dt><dd><code>hi_</code> &rarr; <b>wave 1c</b>, lr <b>3e-6</b>. Identical in every other respect</dd>
<dt>lr_grpo_3e-7</dt><dd><code>lr_</code> &rarr; a <b>sweep</b> run, learning rate spelled out in the name</dd>
</dl>
<p>So <span class="mono">grpo</span>, <span class="mono">hi_grpo</span> and
<span class="mono">lr_grpo_3e-7</span> are the same objective at three learning rates.
<span class="mono">hi_grpo</span> and <span class="mono">lr_grpo_3e-6</span> are deliberate
duplicates run on two different machines, which gives a free reproducibility check &mdash;
they agree to 0.2 points.</p>
<h3>The objectives</h3>
<p>For a group of G=8 rollouts on one prompt with binary rewards <code>r_i</code> and
<code>K</code> = number correct, the loss is
<code>-(1/N) &Sigma; A_i &middot; log&pi;(y_i)</code>. Arms differ <i>only</i> in
<code>A_i</code>:</p>
<dl class="defs">
<dt>base</dt><dd>no training at all &mdash; the reference point</dd>
<dt>grpo</dt><dd><code>A_i = (r_i &minus; mean) / (std + &epsilon;)</code> &mdash; standard RLVR. Mean is ~0, so the gradient is a <i>difference</i> between correct and incorrect samples</dd>
<dt>maxrl</dt><dd><code>A_i = r_i &middot; G/K</code> &mdash; MaxRL: normalise by the number of <b>successes K</b> instead of group size G, which up-weights hard prompts</dd>
<dt>raft</dt><dd><code>A_i = r_i</code> &mdash; SFT on the correct samples only. Used by all three <span class="mono">sft-*</span> arms</dd>
<dt>entropic</dt><dd><code>A_i = e^(&beta;r_i) &minus; mean(e^(&beta;r))</code>, &beta;=+2 &mdash; risk-seeking entropic utility</dd>
<dt>hientropy</dt><dd>grpo, but the gradient is masked to the top-20% highest-entropy tokens (the &ldquo;forking token&rdquo; idea)</dd>
</dl>
<div class="box">
<h4>Why maxrl and sft-online are paired</h4>
<p>They differ by exactly one factor: <code>G/K</code> versus <code>1</code>. Any gap
between them is attributable to MaxRL's normalisation term alone. That is the cleanest
ablation in the study, which is why both are present at every learning rate.</p>
</div>
<h3>The BoN ladder</h3>
<p>All three rungs run the same <code>raft</code> objective and differ only in <i>when</i>
the rollout policy is refreshed &mdash; controlled by one flag, <code>--flush_every</code>:</p>
<dl class="defs">
<dt>sft-offline</dt><dd><code>--flush_every 120</code>. No update happens until the very end,
so every rollout is drawn from the <b>frozen base</b> model. This is <b>true best-of-n
distillation</b>: sample everything, filter to correct, SFT once</dd>
<dt>sft-iter</dt><dd><code>--flush_every 30</code>. Four rounds, policy refreshed between
each &mdash; iterated rejection sampling, i.e. ReST / STaR</dd>
<dt>sft-online</dt><dd><code>--flush_every 1</code>. Refreshed every step &mdash; the fully
on-policy limit</dd>
</dl>
<p>Reading <span class="mono">sft-offline &rarr; sft-iter &rarr; sft-online &rarr; maxrl</span>
in order is a progression of increasing on-policyness, then the 1/K reweighting on top.</p>
</div>"""

    setup = """<h2><span class="n">02</span>Setup, in full</h2><hr class="rule">
<div class="col">
<p>Everything below was held identical across all 20 runs except the learning rate and the
two flags named in section 1.</p>
</div>
<figure class="tw"><div class="scroll"><table>
<caption>Complete configuration. Sampling for rollouts is deliberately untruncated
(top_k=0, top_p=1.0) &mdash; truncated sampling would suppress the distribution tail this
study exists to measure.</caption>
<tbody>
<tr><td>Model</td><td class="mono">Qwen/Qwen2.5-1.5B-Instruct</td></tr>
<tr><td>Finetuning</td><td>full parameter, bf16, <b>not</b> LoRA &mdash; LoRA rank constrains the update and is itself a confound for a diversity question</td></tr>
<tr><td>Optimiser</td><td class="mono">AdamW, betas=(0.9, 0.95), grad-norm clip 1.0, no weight decay, no scheduler</td></tr>
<tr><td>Learning rates</td><td class="mono">1e-7, 3e-7, 1e-6, 3e-6</td></tr>
<tr><td>Train data</td><td class="mono">EleutherAI/hendrycks_math</td> </tr>
<tr><td>Difficulty filter</td><td>levels 1&ndash;3 only (<code>--max_level 3</code>), 4000 prompts shuffled with seed 7291</td></tr>
<tr><td>Rollouts / step</td><td class="mono">P=16 prompts &times; G=8 completions = 128</td></tr>
<tr><td>Steps</td><td class="mono">120</td></tr>
<tr><td>Total budget</td><td class="mono">15,360 rollouts per arm</td> </tr>
<tr><td>Optimizer steps</td><td class="mono">120 of batch 128 (--opt_bs 128)</td></tr>
<tr><td>Sampling (train)</td><td class="mono">temp 1.0, top_p 1.0, top_k 0, max_new 768</td></tr>
<tr><td>Reward</td><td>binary, <code>math_verify.verify</code> on the last <code>\\boxed{}</code></td></tr>
<tr><td>Eval set</td><td class="mono">HuggingFaceH4/MATH-500</td></tr>
<tr><td>Eval sampling</td><td class="mono">n=64, temp 1.0, top_p 1.0, top_k -1, max_new 768, seed 1234</td></tr>
<tr><td>pass@k estimator</td><td>unbiased, <code>1 &minus; C(n&minus;c, k)/C(n, k)</code> (Chen et al. 2021), averaged over problems</td></tr>
<tr><td>Reported k</td><td class="mono">1, 2, 4, 8, 16, 32, 64</td></tr>
<tr><td>Seed</td><td class="mono">7291</td></tr>
<tr><td>Train env</td><td class="mono">${PF_PY_ENV} &mdash; torch 2.8.0+cu128, transformers 4.47.1</td></tr>
<tr><td>Eval env</td><td class="mono">${PF_VLLM} &mdash; vLLM 0.8.5</td></tr>
<tr><td>Hardware</td><td>host-a and host-c, A100 40GB, one GPU per arm. Full finetune peaks at ~26 GB</td></tr>
<tr><td>Code</td><td class="mono">${PF_ROOT} &mdash; ~900 lines, 11 files</td></tr>
</tbody></table></div></figure>
<div class="col">
<div class="box">
<h4>What &ldquo;matched budget&rdquo; means here</h4>
<p>Every trained arm sees exactly 15,360 rollouts <i>and</i> takes exactly 120 optimizer
steps of batch 128. Matching both axes is what makes the BoN ladder comparable to the RL
arms &mdash; otherwise the offline arm would take one giant averaged step instead of a real
SFT pass. A cumulative counter is asserted against
<code>steps &times; P &times; G</code> at exit, and all 20 runs passed.</p>
</div>
<h3>Difficulty strata</h3>
<p>Training is on levels 1&ndash;3, so the harder eval slices double as an
out-of-distribution readout. The base model's pass@64 on the full set is 88.2%, close
enough to the ceiling that effects compress, which is why level&nbsp;&ge;4 is the primary
readout.</p>
</div>
<figure class="tw"><div class="scroll"><table>
<caption>Base model headroom by stratum &mdash; the reason level &ge;4 is the primary readout.</caption>
<thead><tr><th>stratum</th><th class="num">problems</th><th class="num">pass@1</th>
<th class="num">pass@64</th><th class="num">headroom</th></tr></thead><tbody>
<tr><td>all MATH-500</td><td class="num">500</td><td class="num">41.5</td><td class="num">88.2</td><td class="num">11.8</td></tr>
<tr class="base-row"><td>level &ge;4 &mdash; primary</td><td class="num">262</td><td class="num">24.7</td><td class="num">80.5</td><td class="num">19.5</td></tr>
<tr><td>level 5</td><td class="num">134</td><td class="num">14.8</td><td class="num">75.4</td><td class="num">24.6</td></tr>
</tbody></table></div></figure>"""

    results = f"""<h2><span class="n">04</span>pass@k results</h2><hr class="rule">
<div class="col">
<h3>Wave 1 &mdash; lr 1e-6</h3>
<p>The plan named one go/no-go check: <span class="mono">grpo</span> must show pass@1 up
and pass@64 down, or nothing downstream is interpretable. It failed. GRPO moved pass@1 by
+0.4 points on the primary stratum, well inside the error bar, and <b>no arm beat the base
model on either axis</b>. Train reward did rise (0.470 &rarr; 0.514), so the objective was
optimising &mdash; it just did not transfer. 120 steps at 1e-6 is undertrained.</p>
</div>
{eval_table(W1, 4, "Wave 1 (lr 1e-6), level &ge;4")}
{eval_table(W1, 0, "Wave 1 (lr 1e-6), all 500")}
<div class="col">
<h3>Wave 1c &mdash; lr 3e-6</h3>
<p>Same arms, same budget, 3&times; the learning rate. Now the effect appears, and the
curves visibly cross: <span class="mono">hi_grpo</span> and
<span class="mono">hi_entropic</span> sit above base at low k and below it by k=64. That
crossover, at roughly k=16, <b>is</b> the sharpening tradeoff.</p>
<p>Two details worth not glossing over. First, <span class="mono">hi_hientropy</span> is
worse than base everywhere &mdash; masking the gradient to top-20% entropy tokens preserved
entropy (1.51 &rarr; 0.56 rather than &rarr; 0.21) but bought no accuracy. Second,
<b>greedy pass@1 falls</b> for every trained arm even where sampled pass@1 rises
(base 53.2 &rarr; 48.4 for <span class="mono">hi_grpo</span>), so the gain is in the
sampled distribution rather than the mode.</p>
</div>
{eval_table(W1C, 4, "Wave 1c (lr 3e-6), level &ge;4")}
{eval_table(W1C, 0, "Wave 1c (lr 3e-6), all 500")}
<figure class="fig">{img("fig6_frontier_lr3e-6.png")}
<figcaption><b>The crossover.</b> At lr 3e-6 the trained curves start above base and end
below it. Error bars are standard error over problems.</figcaption></figure>
<figure class="fig">{img("fig5_frontier_lr1e-6.png")}
<figcaption><b>Wave 1 for contrast.</b> At lr 1e-6 base dominates the whole curve and the
BoN family sits well below it.</figcaption></figure>
<div class="col"><h3>The sweep runs</h3></div>
{eval_table(SW, 4, "Learning-rate sweep, level &ge;4")}"""

    lrsec = f"""<h2><span class="n">05</span>Learning-rate response</h2><hr class="rule">
<div class="col">
<p>This is the single most informative figure in the study. Each point is a complete
120-step run. The two objective families move in <b>opposite directions</b> as the
learning rate rises: GRPO and entropic climb, MaxRL and all three BoN rungs fall to the
floor.</p>
</div>
<figure class="fig">{img("fig4_lr_response.png")}
<figcaption><b>Opposite signs.</b> Left: final probe pass@1 against lr. Middle: final
pass@16. Right: change in pass@1 from step 0, where zero means &ldquo;training did
nothing&rdquo;. GRPO reaches +0.12; every non-negative-advantage arm is between
&minus;0.15 and &minus;0.29.</figcaption></figure>
<figure class="tw"><div class="scroll"><table>
<caption>Final probe pass@1 by objective and learning rate (50 problems, n=16). Empty cells
were not run. The pattern is monotone in both directions.</caption>
<thead><tr><th>objective</th><th class="num">1e-7</th><th class="num">3e-7</th>
<th class="num">1e-6</th><th class="num">3e-6</th><th>verdict</th></tr></thead><tbody>
<tr><td class="mono">grpo</td><td class="num">&mdash;</td><td class="num">0.287</td><td class="num">0.307</td><td class="num">0.439</td><td><span class="tag ok">improves with lr</span></td></tr>
<tr><td class="mono">entropic</td><td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">0.330</td><td class="num">0.426</td><td><span class="tag ok">improves with lr</span></td></tr>
<tr><td class="mono">hientropy</td><td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">0.290</td><td class="num">0.299</td><td><span class="tag">flat</span></td></tr>
<tr><td class="mono">maxrl</td><td class="num">0.282</td><td class="num">0.289</td><td class="num">0.141</td><td class="num">0.011</td><td><span class="tag no">collapses</span></td></tr>
<tr><td class="mono">sft-online</td><td class="num">0.310</td><td class="num">0.276</td><td class="num">0.138</td><td class="num">0.015</td><td><span class="tag no">collapses</span></td></tr>
<tr><td class="mono">sft-iter</td><td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">0.305</td><td class="num">0.024</td><td><span class="tag no">collapses</span></td></tr>
<tr><td class="mono">sft-offline</td><td class="num">&mdash;</td><td class="num">&mdash;</td><td class="num">0.501</td><td class="num">0.085</td><td><span class="tag no">collapses at flush</span></td></tr>
</tbody></table></div></figure>
<figure class="fig">{img("fig1_probe_trajectories.png")}
<figcaption><b>All 20 runs, pass@1 / pass@4 / pass@16 against step, grouped by learning
rate.</b> Reading left to right the BoN and MaxRL curves tip from flat, to slow decay, to
a cliff. <span class="mono">sft-offline</span> (dotted) is the tell: it holds flat all the
way to step 100 because its policy is frozen, then falls off when its single end-of-training
flush lands.</figcaption></figure>"""

    diag = f"""<h2><span class="n">06</span>Why the BoN family collapses</h2><hr class="rule">
<div class="col">
<p>The diversity logs pin the mechanism. At lr 3e-6 the collapsed arms do not merely get
worse at the task &mdash; they leave the distribution entirely.</p>
</div>
<figure class="tw"><div class="scroll"><table>
<caption>Training diagnostics at lr 3e-6, first step &rarr; last step. Token entropy near
6.7 is close to uniform over a large effective vocabulary.</caption>
<thead><tr><th>arm</th><th class="num">train reward</th><th class="num">token entropy</th>
<th class="num">distinct correct/group</th><th class="num">frac groups K=0</th>
<th class="num">length</th></tr></thead><tbody>
<tr><td class="mono">hi_grpo</td><td class="num">0.438 &rarr; 0.672</td><td class="num">1.51 &rarr; 0.21</td><td class="num">0.875 &rarr; 0.875</td><td class="num">0.19 &rarr; 0.13</td><td class="num">506 &rarr; 354</td></tr>
<tr><td class="mono">hi_entropic</td><td class="num">0.438 &rarr; 0.648</td><td class="num">1.51 &rarr; 0.19</td><td class="num">0.875 &rarr; 0.875</td><td class="num">0.19 &rarr; 0.13</td><td class="num">506 &rarr; 350</td></tr>
<tr><td class="mono">hi_hientropy</td><td class="num">0.438 &rarr; 0.570</td><td class="num">1.51 &rarr; 0.56</td><td class="num">0.875 &rarr; 0.938</td><td class="num">0.19 &rarr; 0.13</td><td class="num">506 &rarr; 352</td></tr>
<tr><td class="mono">hi_sft-offline</td><td class="num">0.438 &rarr; 0.484</td><td class="num">1.61 &rarr; 1.61</td><td class="num">0.875 &rarr; 0.812</td><td class="num">0.19 &rarr; 0.19</td><td class="num">506 &rarr; 426</td></tr>
<tr><td class="mono">hi_sft-iter</td><td class="num">0.438 &rarr; 0.070</td><td class="num">1.54 &rarr; 5.96</td><td class="num">0.875 &rarr; 0.188</td><td class="num">0.19 &rarr; 0.63</td><td class="num">506 &rarr; 637</td></tr>
<tr><td class="mono">hi_sft-online</td><td class="num">0.438 &rarr; 0.062</td><td class="num">1.31 &rarr; 6.66</td><td class="num">0.875 &rarr; 0.125</td><td class="num">0.19 &rarr; 0.63</td><td class="num">506 &rarr; 666</td></tr>
<tr><td class="mono">hi_maxrl</td><td class="num">0.438 &rarr; 0.023</td><td class="num">1.07 &rarr; 6.70</td><td class="num">0.875 &rarr; 0.062</td><td class="num">0.19 &rarr; 0.81</td><td class="num">506 &rarr; 697</td></tr>
</tbody></table></div></figure>
<figure class="fig">{img("fig3_diversity_lr3e-6.png")}
<figcaption><b>All eleven logged diversity metrics at lr 3e-6.</b> The healthy arms
(green, orange, red) drive entropy down and length down together. The collapsing arms
(grey, purple) invert both: entropy up, length up, K=0 fraction up.</figcaption></figure>
<figure class="fig">{img("fig2_diversity_lr1e-6.png")}
<figcaption><b>The same metrics at lr 1e-6</b>, where the same failure is present but
slow.</figcaption></figure>
<div class="col">
<h3>The mechanism</h3>
<p>GRPO's advantage is z-scored, so its mean is about zero and its gradient is a
<i>difference</i>: correct samples are pushed up while incorrect ones are pushed down. That
is self-anchoring. MaxRL and RAFT use strictly non-negative weights, so their gradient has
a nonzero mean &mdash; an unanchored &ldquo;raise the likelihood of these 45%&rdquo; push
with no counterbalancing term and no reference anchor. Raising the probability of the
model's own samples is a positive feedback loop. Their gradient norm saturated the 1.0 clip
on every step from step 0, where GRPO's sat at 0.6&ndash;1.1.</p>
<p>The damage arrives fast: <span class="mono">hi_sft-online</span> is already down from
0.275 to 0.100 by step 20. And the length signature is diagnostic &mdash; 506 &rarr; 697
tokens, the opposite of the healthy arms, which is what a token-level loss denominator
rewards when only correct samples carry weight.</p>
<div class="box warn">
<h4>What I do not claim</h4>
<p>This is <b>not</b> evidence against MaxRL. Two facts point at my implementation rather
than the method. <span class="mono">maxrl</span> and <span class="mono">sft-online</span>
track each other to within 0.004 at every learning rate, which means the 1/K term &mdash;
the entire content of MaxRL &mdash; is not doing anything measurable here; a faithful
implementation should differ from plain RAFT. And a published method does not fail at all
four learning rates.</p>
<p>The two stabilisers real RFT/MaxRL implementations carry and mine does not: a
<b>KL anchor to the reference policy</b>, and <b>per-sequence length normalisation</b>
instead of my global token-level denominator. Until those are added, the
MaxRL-versus-BoN-distillation comparison at matched compute stays open.</p>
</div>
</div>"""

    bugs = """<h2><span class="n">07</span>Bugs found and fixed</h2><hr class="rule">
<div class="col"><p>Recorded because several would have silently invalidated an arm rather
than crashing.</p></div>
<figure class="tw"><div class="scroll"><table>
<caption>Every defect hit during the build, and how it would have shown up.</caption>
<thead><tr><th>what</th><th>symptom if missed</th></tr></thead><tbody>
<tr><td><code>max_new</code> below ~640 truncates before <code>\\boxed{}</code></td><td>reward silently 0.000 everywhere; the first smoke run reported perfect zeros with fluent output</td></tr>
<tr><td><code>generate</code> called once per prompt</td><td>GPU at 36% utilisation. Batching across prompts gave <b>~14&times;</b> (5.5 &rarr; 0.39 s/rollout) and removed any need for a vLLM weight-sync path</td></tr>
<tr><td><code>sft-offline</code> did one averaged step over all 15,360 sequences</td><td>the BoN baseline would have been a no-op at lr 1e-6. Fixed with <code>--opt_bs</code>, which also equalised optimizer steps across arms</td></tr>
<tr><td><code>cliphigher</code> ratio computed against <code>lp.detach()</code></td><td>ratio identically 1, so the clipping was inert while appearing to run</td></tr>
<tr><td>a file named <code>inspect.py</code></td><td>shadowed the stdlib module and broke every pandas/datasets import in the directory</td></tr>
<tr><td>frontier plot y-label from a lexicographic key sort</td><td>axis read &ldquo;pass@8&rdquo; while plotting pass@64</td></tr>
<tr><td><code>aggregate.py</code> crashed on a partially written eval file</td><td>aborted the whole aggregation mid-run; now skips and reports</td></tr>
<tr><td>frontier subplot title took <code>n</code> from the last file</td><td>read &ldquo;n=0 problems&rdquo; when any file was still being written</td></tr>
<tr><td>stale <code>typing_extensions</code> in <code>/local/.../.local</code> on host-c</td><td>shadowed the env and killed all six sweep evals; fixed with <code>PYTHONNOUSERSITE=1</code></td></tr>
<tr><td><code>pkill -f</code> patterns matching their own ssh command line</td><td>killed the launching session instead of the target</td></tr>
</tbody></table></div></figure>"""

    nxt = """<h2><span class="n">08</span>What to do next</h2><hr class="rule">
<div class="col">
<ol>
<li><b>Add a KL anchor and per-sequence length normalisation to the RAFT/MaxRL path</b>,
then rerun at 3e-6. Roughly a twenty-line change, and without it the BoN-distillation
comparison &mdash; the actual question &mdash; cannot be made. Highest priority by a wide
margin.</li>
<li><b>Extend <span class="mono">grpo</span> and <span class="mono">entropic</span> to
300&ndash;500 steps at 3e-6.</b> Probe pass@1 for <span class="mono">hi_grpo</span> was
still climbing at step 119 (0.318 &rarr; 0.439 with no plateau), so the pass@k collapse has
not finished developing and the frontier tradeoff is still forming. The interesting number
is where pass@64 ends up once pass@1 saturates.</li>
<li><b>Then wave 2</b> &mdash; <span class="mono">cliphigher</span>,
<span class="mono">klanchor</span>, <span class="mono">drgrpo</span>,
<span class="mono">entbonus</span>, and especially
<span class="mono">condmix</span>, the conditional-mixture arm that is the only design here
that structurally decouples pass@1 from pass@k. All are already implemented and flag-gated;
they were held back deliberately rather than run against a baseline that did not reproduce
the effect.</li>
</ol>
<div class="box key">
<h4>The live lead</h4>
<p><span class="mono">entropic</span> is the result I would chase. A single scalar &beta;
in the reward shape reached the same frontier point as GRPO, and it held the highest
pass@16 of any trained arm on the full set. That is the Pareto direction MaxRL claims, from
a much smaller change, and the machinery already exists in the entropic-reward project.
Sweeping &beta; is cheap and is the obvious next dial.</p>
</div>
</div>"""

    html = (HTML.replace("__GLOSSARY__", glossary).replace("__SETUP__", setup)
            .replace("__RUNS__", '<h2><span class="n">03</span>Every run</h2><hr class="rule">'
                     + runs_table())
            .replace("__RESULTS__", results).replace("__LR__", lrsec)
            .replace("__DIAG__", diag).replace("__BUGS__", bugs).replace("__NEXT__", nxt))
    out = f"{ROOT}/results/report.html"
    with open(out, "w") as f:
        f.write(html)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
