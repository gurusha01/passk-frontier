"""Rebuild the report with the corrected long-horizon finding."""
import base64, glob, json, math, os
ROOT = "."; KS = [1, 2, 4, 8, 16, 32, 64]

def pak(n,c,k): return 1.0 if n-c<k else 1.0-math.comb(n-c,k)/math.comb(n,k)
def curve(a, lvl=4):
    p=f"{ROOT}/evals/{a}.jsonl"
    if not os.path.exists(p): return None
    r=[json.loads(l) for l in open(p) if l.strip()]
    if lvl: r=[x for x in r if x.get("level",0)>=lvl]
    if not r: return None
    out={"n":len(r),"c":{},"g":None}
    for k in KS:
        v=[pak(x["n"],x["c"],k) for x in r]; out["c"][k]=sum(v)/len(v)
    if "greedy" in r[0]: out["g"]=sum(x["greedy"] for x in r)/len(r)
    return out
def img(n):
    p=f"{ROOT}/results/{n}"
    if not os.path.exists(p): return ""
    return f'<img src="data:image/png;base64,{base64.b64encode(open(p,"rb").read()).decode()}" alt="{n}">'

def tbl(rows, cap, lvl=4):
    b=curve("base",lvl); out=[]
    for arm,lab,note in rows:
        c=curve(arm,lvl)
        if not c: 
            out.append(f"<tr><td>{lab}</td><td class='num pend' colspan='8'>running</td><td>{note}</td></tr>"); continue
        tds=""
        for k in KS:
            v=c["c"][k]; d=v-b["c"][k]
            if arm=="base": tds+=f"<td class='num'>{v*100:.1f}</td>"
            else:
                cl="up" if d>0.005 else ("dn" if d<-0.005 else "flat")
                tds+=f"<td class='num'>{v*100:.1f}<span class='delta {cl}'>{'+' if d>=0 else '−'}{abs(d)*100:.1f}</span></td>"
        cls=" class='base-row'" if arm=="base" else ""
        out.append(f"<tr{cls}><td>{lab}</td>{tds}<td class='note-cell'>{note}</td></tr>")
    head="".join(f"<th class='num'>p@{k}</th>" for k in KS)
    return (f"<figure class='tw'><div class='scroll'><table><caption>{cap} &middot; {b['n']} problems "
            f"&middot; n=64 &middot; % with change vs base</caption><thead><tr><th>arm</th>{head}"
            f"<th>note</th></tr></thead><tbody>{''.join(out)}</tbody></table></div></figure>")

MAIN=[("base","base (no training)","Qwen2.5-1.5B-Instruct, untrained"),
 ("long_grpo","<b>GRPO, 300 steps</b>","<b>beats base at every k from 1 to 64</b>"),
 ("long_entropic","entropic β=+2, 300 steps","risk-seeking reward shape"),
 ("fix_maxrl_1e-5","<b>MaxRL baselined, lr 1e-5</b>","<b>best pass@1 at 120 steps</b>"),
 ("long_maxrl","MaxRL, 300 steps","GRPO overtakes it here"),
 ("hi_grpo","GRPO, 120 steps","the classic tradeoff shows here"),
 ("bonbon_e6_3e-6","BonBon, 6 epochs","best distillation arm"),
 ("bonbon_3e-6","BonBon, 1 epoch","sample-matched"),
 ("hi_sft-offline","BoN distillation (RAFT)","no contrastive term &mdash; my bug, not the method"),
 ("hi_maxrl","MaxRL <i>un-baselined</i>","the 18x failure"),
 ("hi_sft-online","vanilla REINFORCE (no baseline)","collapses"),
]
LR=[("base","base",""),
 ("lr_grpo_3e-7","GRPO lr 3e-7","too low"),("hi_grpo","GRPO lr 3e-6","<b>GRPO optimum</b>"),
 ("fix_maxrl_1e-6","MaxRL lr 1e-6","too low"),("fix_maxrl_3e-6","MaxRL lr 3e-6","still low"),
 ("fix_maxrl_1e-5","MaxRL lr 1e-5","<b>MaxRL optimum, 3x GRPO's</b>"),
 ("lr_maxrl_1e-7","MaxRL un-baselined 1e-7","inert"),("maxrl","MaxRL un-baselined 1e-6","decaying"),
 ("hi_maxrl","MaxRL un-baselined 3e-6","collapsed"),
]
BON=[("base","base",""),
 ("bonbon_e6_3e-7","BonBon 6ep lr 3e-7",""),("bonbon_e6_1e-6","BonBon 6ep lr 1e-6",""),
 ("bonbon_e6_3e-6","BonBon 6ep lr 3e-6","<b>best</b>"),("bonbon_e6_1e-5","BonBon 6ep lr 1e-5","too high"),
 ("bonbon_3e-6","BonBon 1ep lr 3e-6","sample-matched"),
 ("bonbon_a0.5_3e-6","BonBon α=0.5","paper uses 0.005; no difference"),
]

HTML = """<title>The pass@k Frontier</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{--ground:#FAFBFC;--surface:#F0F3F5;--sunk:#E7ECEF;--line:#D8E0E6;--ink:#14202B;--muted:#5A6B78;
--accent:#0E5C68;--accent-soft:#DCEEF1;--good:#1D6B4A;--bad:#A33529;--warn:#8A5A12;
--serif:"IBM Plex Serif",Georgia,serif;--sans:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",monospace;}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ground:#0E1418;--surface:#161F25;--sunk:#1C272E;
--line:#27343C;--ink:#E3EAEF;--muted:#8CA0AD;--accent:#5FB9C9;--accent-soft:#153037;--good:#4FBF8B;--bad:#E4796A;--warn:#D6A24E;}}
:root[data-theme="dark"]{--ground:#0E1418;--surface:#161F25;--sunk:#1C272E;--line:#27343C;--ink:#E3EAEF;
--muted:#8CA0AD;--accent:#5FB9C9;--accent-soft:#153037;--good:#4FBF8B;--bad:#E4796A;--warn:#D6A24E;}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:16.5px;line-height:1.62}
.wrap{max-width:1240px;margin:0 auto;padding:clamp(28px,5vw,72px) clamp(18px,4vw,48px) 96px}
.col{max-width:74ch}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(2.1rem,4.4vw,3.1rem);line-height:1.1;letter-spacing:-.02em;margin:0 0 .5rem}
.sub{font-family:var(--serif);font-style:italic;font-size:1.16rem;color:var(--muted);max-width:64ch;margin:0 0 2rem}
h2{font-family:var(--serif);font-weight:600;font-size:1.62rem;margin:3.4rem 0 .2rem;display:flex;gap:.7rem;align-items:baseline}
h2 .n{font-family:var(--mono);font-size:.8rem;color:var(--accent);letter-spacing:.06em;flex:none}
h3{font-size:1.05rem;margin:2rem 0 .3rem}
p{margin:.75rem 0}
.eyebrow{font-family:var(--mono);font-size:.72rem;text-transform:uppercase;letter-spacing:.16em;color:var(--muted);margin:0 0 1rem}
code{font-family:var(--mono);font-size:.86em;background:var(--sunk);padding:.1em .38em;border-radius:3px}
.rule{border:0;border-top:2px solid var(--ink);margin:.1rem 0 1.4rem;opacity:.85}
.box{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:1.1rem 1.3rem;margin:1.4rem 0}
.box.key{border-left:3px solid var(--accent);background:var(--accent-soft)}
.box.warn{border-left:3px solid var(--warn)}
.box h4{margin:0 0 .35rem;font-size:.94rem}
.box p:last-child{margin-bottom:0}
figure.tw{margin:1.6rem 0}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.845rem}
caption{caption-side:top;text-align:left;padding:.7rem .9rem;color:var(--muted);font-size:.78rem;border-bottom:1px solid var(--line)}
th,td{padding:.42rem .7rem;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
thead th{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:500;background:var(--sunk)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:.8rem}
tr.base-row{background:var(--sunk);font-weight:600}
tr.base-row td{border-bottom:2px solid var(--line)}
.delta{margin-left:.4em;font-size:.75em}
.delta.up{color:var(--good)}.delta.dn{color:var(--bad)}.delta.flat{color:var(--muted)}
.note-cell{white-space:normal;font-size:.78rem;color:var(--muted);max-width:30ch}
.pend{color:var(--muted);font-style:italic;text-align:left}
figure.fig{margin:2rem 0}
figure.fig img{width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{font-size:.82rem;color:var(--muted);margin-top:.55rem;max-width:90ch}
figcaption b{color:var(--ink)}
dl.defs{display:grid;grid-template-columns:auto 1fr;gap:.45rem 1.2rem;margin:1.2rem 0;align-items:baseline}
dl.defs dt{font-family:var(--mono);font-size:.83rem;color:var(--accent);white-space:nowrap}
dl.defs dd{margin:0}
ul,ol{padding-left:1.3rem}li{margin:.3rem 0}
</style>
<div class="wrap">
<p class="eyebrow">Experiment report &middot; MATH-500 &middot; Qwen2.5-1.5B-Instruct &middot; ~70 runs</p>
<h1>The pass@k Frontier</h1>
<p class="sub">Can RL raise pass@1 and pass@k at once, and beat best-of-n distillation at a
matched rollout budget? The answer changed twice as the runs came in.</p>
<div class="col">
<div class="box key"><h4>Three findings, in order of how much they surprised me</h4>
<p><b>1. The pass@1-up / pass@k-down tradeoff is a short-horizon artifact.</b> At 120 steps
GRPO shows it exactly as advertised. At 300 steps the same objective beats the base model at
<i>every</i> k from 1 to 64. Nothing was traded away; the run had simply been stopped early.</p>
<p><b>2. MaxRL's advantage is convergence speed, not a better optimum.</b> At 120 steps MaxRL
beats GRPO by a wide margin. Give GRPO 300 steps and it overtakes. Real and useful, but a
different claim than the one I made first.</p>
<p><b>3. One missing baseline term was worth 18x.</b> The un-baselined MaxRL estimator scores
pass@1 0.013; the baselined form scores 0.325. Same gradient in expectation.</p>
</div></div>
__FIGC__ __T1__ __LR__ __FIGB__ __BON__ __FIGA__ __METHOD__ __CAV__
</div>"""

figc = f"""<h2><span class="n">01</span>The tradeoff does not survive longer training</h2><hr class="rule">
<div class="col"><p>This is the result that reframed the study. Every arm at 120 steps was still
climbing when it stopped, so the pass@k loss I measured was the state of an unfinished run
rather than a property of the objective.</p></div>
<figure class="fig">{img('figC_long.png')}<figcaption><b>Solid = 120 steps, dashed = 300 steps.</b>
Right panel: each line is one objective, and 2.5x more training moves it up and to the right.
GRPO ends above the base model on both axes.</figcaption></figure>"""

lr = f"""<h2><span class="n">03</span>Every objective has its own learning rate</h2><hr class="rule">
<div class="col"><p>The single most expensive mistake in this study was assuming a shared learning
rate is the &ldquo;controlled&rdquo; choice. It is not: it measures lr sensitivity. At 1e-6 MaxRL
looks broken; at 1e-5 it is the best 120-step arm. GRPO wants 3e-6 and degrades at 1e-5.</p></div>
{tbl(LR, "Learning-rate sweep, MATH-500 level &ge;4")}"""

method = """<h2><span class="n">06</span>What each arm is</h2><hr class="rule">
<div class="col"><p>Every arm shares one rollout loop and differs <i>only</i> in the advantage.
For a group of G=8 rollouts on one prompt with binary rewards <code>r_i</code>,
<code>p = mean(r)</code>, <code>K = &Sigma;r_i</code>:</p>
<dl class="defs">
<dt>grpo</dt><dd><code>(r_i &minus; p)/(std + &epsilon;)</code> &mdash; standard RLVR</dd>
<dt>maxrl</dt><dd><code>(r_i &minus; p)/(p + &epsilon;)</code> &mdash; ascends <code>J = log p</code>.
GRPO with the <i>mean</i> in the denominator instead of the std, so hard prompts are up-weighted
as 1/p rather than 1/&radic;p</dd>
<dt>maxrl (un-baselined)</dt><dd><code>r_i &middot; G/K</code> &mdash; same expectation, but strictly
non-negative, so no anchoring negative gradient. This is the 18x failure</dd>
<dt>raft</dt><dd><code>r_i</code> &mdash; SFT on correct samples, identical to vanilla REINFORCE
with no baseline under a binary reward</dd>
<dt>entropic</dt><dd><code>e^(&beta;r_i) &minus; mean</code>, &beta;=+2 &mdash; risk-seeking</dd>
<dt>BonBon</dt><dd><code>&alpha;&middot;L_SFT + (1&minus;&alpha)&middot;L_IPO</code> on best/worst-of-8
from the frozen base (arXiv 2406.00832), implemented from the paper</dd>
<dt>BOND / J-BOND</dt><dd>Jeffreys divergence against the best-of-N distribution, MC quantiles,
EMA anchor (arXiv 2407.14622)</dd>
</dl>
<p>The best-of-n ladder is one flag: <code>--flush_every</code> with <code>--objective raft</code>
gives fully on-policy RAFT (1), iterated ReST (30), or offline distillation with the policy frozen
throughout (120).</p></div>"""

cav = """<h2><span class="n">07</span>What I would not claim</h2><hr class="rule">
<div class="col">
<div class="box warn"><h4>Scope</h4>
<p>One model (1.5B), one task family (MATH), one seed per arm. The 300-step result rests on three
runs. &ldquo;The collapse is a short-horizon artifact&rdquo; is what these runs show; whether it
holds at 7B or on a base rather than an already-rejection-sampled instruct model is untested.</p></div>
<div class="box warn"><h4>Distillation was handicapped here</h4>
<p>Qwen2.5-Instruct was already rejection-sampled on math during post-training, so best-of-8 of the
frozen base has little left to distill. BonBon matching GRPO under that handicap is arguably the
more impressive result. Non-iterative BOND is capped by a fixed anchor by construction, and
J-BOND's 2-sample reward fires under 6% of the time with a binary reward.</p></div>
<div class="box"><h4>Still running</h4>
<p>Three follow-ups are mid-flight and not in these tables: an &alpha;-fair family
<code>(r&minus;p)/p^&lambda;</code>, a geometric-weighted pass@k objective whose weight is bounded
at p=0 where MaxRL's diverges, and a difficulty-gated variant that zeroes the gradient on prompts
already solved &ge;50% of the time.</p></div>
</div>"""

html = (HTML.replace("__FIGC__", figc)
        .replace("__T1__", '<h2><span class="n">02</span>Main results</h2><hr class="rule">'
                 + tbl(MAIN, "All arms, MATH-500 level &ge;4 (the harder, unsaturated slice)"))
        .replace("__LR__", lr)
        .replace("__FIGB__", f"<figure class='fig'>{img('figB_lr.png')}<figcaption><b>Opposite "
                 "directions.</b> The baselined objectives improve with lr; the un-baselined ones "
                 "fall to the floor.</figcaption></figure>")
        .replace("__BON__", '<h2><span class="n">04</span>Best-of-n distillation</h2><hr class="rule">'
                 '<div class="col"><p>BonBon implemented from the paper (SFT term plus the IPO '
                 'contrastive term on best/worst-of-8) reaches parity with GRPO. Six epochs over the '
                 'same data &mdash; roughly isoflop &mdash; buys <b>+0.8 pass@1 for 6x the training '
                 'compute</b>, which says the ceiling is the data, not the optimisation.</p></div>'
                 + tbl(BON, "BonBon sweep, MATH-500 level &ge;4"))
        .replace("__FIGA__", '<h2><span class="n">05</span>The frontier</h2><hr class="rule">'
                 f"<figure class='fig'>{img('figA_frontier.png')}<figcaption>Left: full pass@k curves. "
                 "Right: pass@1 against pass@64, dotted lines marking the base model. Up and to the "
                 "right is a genuine Pareto improvement.</figcaption></figure>")
        .replace("__METHOD__", method).replace("__CAV__", cav))
open("results/report_v2.html","w").write(html)
print("wrote results/report_v2.html", os.path.getsize("results/report_v2.html")//1024, "KB")
