"""Turn the measurements saved by the lessons into the plots used in the README.

    python lessons/02_why_rollouts_drift.py && python make_figures.py

Nothing here computes a result - it only draws what the lessons measured, so a
figure can never disagree with the run that produced it.
"""
import pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from wm.analysis import summarise

FIG = pathlib.Path(__file__).parent / "figures"

# The lessons save what they measured; this script only draws it. Say so plainly
# when the measurements are not there yet, rather than letting a FileNotFoundError
# traceback be a freshly cloned repository's first output.
if not (FIG / "lesson02.npz").exists():
    print(__doc__)
    print("No measurements found in figures/. Run the lessons first:\n"
          "    python lessons/01_a_world_model_in_50_lines.py\n"
          "    python lessons/02_why_rollouts_drift.py\n"
          "then this script again. Lessons 3-6 add their own panels when present.")
    raise SystemExit(1)

d = np.load(FIG / "lesson02.npz")
e_p, err_p, bound = d["e_p"], d["err_p"], d["bound"]
EA, EB = d["EA"], d["EB"]
lam_l, dt_l = float(d["lam_l"]), float(d["dt_l"])
scale, d0, tol = float(d["scale"]), float(d["d0"]), float(d["tol"])
rates = dict(zip([str(x) for x in d["avg_keys"]], d["avg_rates"]))

plt.rcParams.update({"font.size": 8.6, "axes.linewidth": .8, "figure.dpi": 160,
                     "font.family": ["Liberation Sans", "DejaVu Sans", "sans-serif"]})
INK, BLUE, RED, GREEN, SAND = "#1a1a1a", "#1a4f7a", "#c0392b", "#2e7d5b", "#e08e6d"
fig, AX = plt.subplots(2, 2, figsize=(11.4, 7.4))

# ── 1. the bound, against reality ─────────────────────────────────────────
a, k = AX[0, 0], np.arange(1, len(e_p) + 1)
a.plot(k, bound, lw=1.7, ls="--", color=RED, label=r"textbook bound  $\delta\,(L^k-1)/(L-1)$")
q1, q3 = np.percentile(err_p, [25, 75], axis=0)
a.fill_between(k, q1, q3, color=BLUE, alpha=.22, lw=0, label="measured, IQR")
# Both summaries, because which one you plot is itself a result: the median is
# a power law, the mean is not, and the gap between them is the 14% of rollouts
# that have gone over the top of the pendulum by step 90.
a.plot(k, err_p.mean(0), lw=1.6, color=SAND, label="measured, MEAN (dragged by failures)")
a.plot(k, e_p, lw=2.4, color=BLUE, label="measured, MEDIAN (typical rollout)")
kk = 40
a.annotate("%.0f$\\times$ apart already\nat step %d" % (bound[kk-1]/e_p[kk-1], kk),
           xy=(kk, bound[kk-1]), xytext=(kk*1.05, bound[kk-1]*.004), fontsize=8,
           color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1))
a.plot([kk], [bound[kk-1]], "o", ms=4, color=RED)
a.set_yscale("log"); a.set_xlabel("rollout step $k$"); a.set_ylabel("state prediction error")
a.set_title("(a)  Pendulum: the bound has the wrong shape", color=INK, loc="left")
a.set_ylim(1e-3, 5e8)
a.legend(fontsize=7.2, frameon=True, framealpha=.85, edgecolor="none", loc="lower right"); a.grid(alpha=.2, lw=.6)

# ── 2. amplification vs injection, on the median curve ────────────────────
b = AX[0, 1]
t = np.arange(1, EA.shape[1] + 1) * dt_l
mA, mB = summarise(EA, "median"), summarise(EB, "median")
b.plot(t, mA, lw=2.0, color=GREEN, label="A  perturb once, true dynamics")
b.plot(t, mB, lw=2.0, color=BLUE, label="B  world-model rollout")
b.plot(t, d0 * np.exp(lam_l * t), lw=1.3, ls="--", color=RED,
       label=r"$\delta e^{\lambda t}$,  $\lambda=%.3f$ measured" % lam_l)
b.axhline(scale, color="#999", lw=.9, ls="-.")
b.text(t[3], scale * 1.15, "attractor scale", fontsize=7.2, color="#777")
b.set_yscale("log"); b.set_xlabel("time  $t=k\\,\\Delta t$"); b.set_ylabel("state prediction error (median)")
b.set_title("(b)  Lorenz: injection lifts the curve, not the rate", color=INK, loc="left")
b.legend(fontsize=7.4, frameon=False, loc="lower right"); b.grid(alpha=.2, lw=.6)

# ── 3. the average you pick changes the answer ────────────────────────────
c = AX[1, 0]
order = ["median", "geometric", "mean"]
x = np.arange(len(order)); w = .34
for i, (curve, col, lab) in enumerate([("A", GREEN, "A  perturbation"), ("B", BLUE, "B  model rollout")]):
    vals = [rates["%s|%s" % (curve, o)] for o in order]
    bars = c.bar(x + (i - .5) * w, vals, w, color=col, alpha=.85, label=lab)
    for xi, v in zip(x + (i - .5) * w, vals):
        c.text(xi, v + .012, "%+.0f%%" % (100 * (v / lam_l - 1)), ha="center", fontsize=7.4,
               color=col, weight="bold")
c.axhline(lam_l, color=RED, lw=1.6, ls="--")
c.text(2.44, lam_l - .020, r"measured $\lambda$ = %.3f" % lam_l, fontsize=7.4, color=RED, ha="right")
c.set_xticks(x); c.set_xticklabels(["median", "geometric\nmean", "arithmetic\nmean"])
c.set_ylim(.75, 1.06); c.set_ylabel("fitted growth rate")
c.set_title("(c)  Only the median recovers $\\lambda$ for both", color=INK, loc="left")
c.legend(fontsize=7.4, frameon=False, loc="upper left"); c.grid(alpha=.2, lw=.6, axis="y")

# ── 4. horizon is not one number ──────────────────────────────────────────
e = AX[1, 1]
K = err_p.shape[1]
h = np.array([(np.argmax(r > tol) + 1) if (r > tol).any() else K for r in err_p])
p5, p50, p95 = np.percentile(h, [5, 50, 95])
e.hist(h, bins=np.arange(0, K + 2, 2) - .5, color=BLUE, edgecolor="white", lw=.5)
top = e.get_ylim()[1]
for v, col, lab in [(p5, RED, "5th %d" % p5), (p50, INK, "median %d" % p50), (p95, GREEN, "95th %d" % p95)]:
    e.axvline(v, color=col, lw=1.4, ls="--")
    e.text(v + 1.8, top * .60, lab, fontsize=7.6, color=col, rotation=90, va="center",
           bbox=dict(fc="white", ec="none", alpha=.75, pad=1))
e.set_xlabel("steps until this trajectory exceeds the tolerance")
e.set_ylabel("trajectories")
e.set_title("(d)  Same model, same task: horizon varies %.0f$\\times$" % (p95 / max(p5, 1)),
            color=INK, loc="left")
e.grid(alpha=.2, lw=.6, axis="y")

fig.tight_layout()
out = FIG / "rollout-error.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)


# ══════════════════════════════════════════════════════════════════════════
# Lesson 3: what a planner actually needs from a world model
# ══════════════════════════════════════════════════════════════════════════
f3 = FIG / "lesson03.npz"
if f3.exists():
    d3 = np.load(f3)
    HS3, base3, rewards = d3["HS"], d3["base"], d3["rewards"]
    names = [str(x) for x in d3["names"]]
    horizons = d3["horizons"]
    rank = dict(zip([str(x) for x in d3["rank_keys"]], d3["rank_vals"]))
    fig3, A3 = plt.subplots(1, 2, figsize=(11.4, 3.9))

    a3 = A3[0]
    a3.plot(HS3, base3, lw=2.4, color=INK, marker="o", ms=4.5, label="TRUE dynamics (control)")
    for nm, col, row, hz in zip(names, (GREEN, BLUE, RED), rewards, horizons):
        a3.plot(HS3, row, lw=1.9, color=col, marker="s", ms=4, alpha=.9,
                label="model: %s  (usable horizon %.0f)" % (nm, hz))
    a3.axvline(25, color="#999", lw=1, ls="--")
    a3.text(26.5, base3.min() + .95, "H the task needs", fontsize=7.4, color="#777")
    a3.set_xlabel("planning horizon $H$")
    a3.set_ylabel("reward per step (real world)")
    a3.set_title("(a)  A rollout useless after 8 steps still plans 25", color=INK, loc="left")
    a3.legend(fontsize=7.2, frameon=False, loc="lower right")
    a3.grid(alpha=.2, lw=.6)

    b3 = A3[1]
    Hs3 = sorted({int(k.split("|")[1]) for k in rank})
    for nm, col in zip(names, (GREEN, BLUE, RED)):
        rho = [rank["%s|%d" % (nm, h)][1] for h in Hs3]
        b3.plot(Hs3, rho, lw=2.0, color=col, marker="o", ms=5, label="model: %s" % nm)
        for h, r in zip(Hs3, rho):
            reg = rank["%s|%d" % (nm, h)][2]
            if reg > 0.5:
                b3.annotate("regret %.0f" % reg, (h, r), textcoords="offset points",
                            xytext=(-8, -13 if r > 0 else 12), fontsize=7, color=col,
                            ha="right" if h == max(Hs3) else "left")
    b3.axhline(0, color="#999", lw=.8)
    b3.set_ylim(-.45, 1.14)
    b3.set_xlabel("planning horizon $H$")
    b3.set_ylabel("rank correlation of candidate returns")
    b3.set_title("(b)  Planning fails when the ranking fails, not the states",
                 color=INK, loc="left")
    b3.legend(fontsize=7.3, frameon=False, loc="lower left")
    b3.grid(alpha=.2, lw=.6)

    fig3.tight_layout()
    out3 = FIG / "planning.png"
    fig3.savefig(out3, bbox_inches="tight", facecolor="white")
    print("wrote", out3)


# ══════════════════════════════════════════════════════════════════════════
# Lesson 4: three collapse detectors, each fooled by a different failure
# ══════════════════════════════════════════════════════════════════════════
f4 = FIG / "lesson04.npz"
if f4.exists():
    d4 = np.load(f4)
    nm4 = [str(x) for x in d4["names"]]
    loss4, std4, rank4, r24 = d4["loss"], d4["std"], d4["rank"], d4["r2"]
    fl, ce, LAT = d4["floor"], float(d4["ceiling"]), int(d4["latent"])
    fig4, A4 = plt.subplots(1, 3, figsize=(13.0, 3.7))
    cols = (RED, GREEN, BLUE)
    x4 = np.arange(len(nm4))

    a4 = A4[0]
    a4.bar(x4, loss4, .55, color=cols)
    a4.set_yscale("log"); a4.set_xticks(x4); a4.set_xticklabels(nm4)
    a4.set_ylabel("final prediction loss")
    a4.set_title("(a)  The best loss belongs to the worst model", color=INK, loc="left")
    for xi, v in zip(x4, loss4):
        a4.text(xi, v * 1.6, "%.0e" % v, ha="center", fontsize=7.4)
    a4.grid(alpha=.2, lw=.6, axis="y")

    b4 = A4[1]
    w = .38
    b4.bar(x4 - w / 2, std4, w, color=SAND, label="latent std")
    b4.bar(x4 + w / 2, rank4 / LAT, w, color="#7a6f9b", label="effective rank / %d" % LAT)
    b4.axhline(1 / LAT, color="#999", lw=.9, ls=":")
    b4.text(-.46, 1 / LAT + .045, "rank 1 = fully collapsed", fontsize=7, color="#777", ha="left")
    b4.set_xticks(x4); b4.set_xticklabels(nm4); b4.set_ylabel("detector value")
    b4.set_title("(b)  Two detectors, opposite verdicts", color=INK, loc="left")
    b4.legend(fontsize=7.3, frameon=False, loc="upper left"); b4.grid(alpha=.2, lw=.6, axis="y")
    b4.annotate("scale fine,\nrank gone", (2 + w / 2, rank4[2] / LAT), textcoords="offset points",
                xytext=(10, 34), fontsize=7, color="#7a6f9b", ha="center",
                arrowprops=dict(arrowstyle="->", color="#7a6f9b", lw=.9))
    # naive's std bar is 0.0003 - invisible at this scale, which is the point,
    # so the annotation has to say that rather than point at a missing bar.
    b4.annotate("std = %.4f, no bar\nscale gone, rank fine" % std4[0], (0 - w / 2, 0),
                textcoords="offset points", xytext=(6, 52), fontsize=7, color=SAND,
                ha="center", arrowprops=dict(arrowstyle="->", color=SAND, lw=.9))

    c4 = A4[2]
    c4.axhspan(float(fl.min()), float(fl.max()), color="#999", alpha=.2, lw=0)
    c4.axhline(float(fl.mean()), color="#777", lw=1.2, ls="--")
    c4.text(-.42, float(fl.mean()) + .022, "untrained encoder (5 seeds)", fontsize=7,
            color="#555", ha="left")
    c4.axhline(ce, color=INK, lw=1.2, ls="-.")
    c4.text(-.42, ce + .022, "raw observation (ceiling)", fontsize=7, color=INK, ha="left")
    c4.bar(x4, r24, .55, color=cols)
    for xi, v in zip(x4, r24):
        c4.text(xi, v + .02, "%.3f" % v, ha="center", fontsize=7.6)
    c4.set_xticks(x4); c4.set_xticklabels(nm4)
    c4.set_ylim(0, 1.06); c4.set_ylabel("linear probe $R^2$ for the true state")
    c4.set_title("(c)  Below the floor is not a good score", color=INK, loc="left")
    c4.grid(alpha=.2, lw=.6, axis="y")

    fig4.tight_layout()
    out4 = FIG / "collapse.png"
    fig4.savefig(out4, bbox_inches="tight", facecolor="white")
    print("wrote", out4)


# ══════════════════════════════════════════════════════════════════════════
# Lessons 5 and 6: real MuJoCo environments
# ══════════════════════════════════════════════════════════════════════════
f5, f6 = FIG / "lesson05.npz", FIG / "lesson06.npz"
if f5.exists() and f6.exists():
    d5, d6 = np.load(f5), np.load(f6)
    envs = [str(x) for x in d5["envs"]]
    short = [e.split("-")[0] for e in envs]
    fig5, A5 = plt.subplots(1, 3, figsize=(13.4, 3.8))

    # (a) the bound writes the rollout off in 2-3 steps; reality is 20-60
    a5 = A5[0]
    y = np.arange(len(envs)); h = .36
    a5.barh(y + h / 2, d5["k_real"], h, color=BLUE, label="actually exceeds the state scale at")
    a5.barh(y - h / 2, d5["k_bound"], h, color=RED, label="textbook bound says worthless at")
    for yi, v in zip(y - h / 2, d5["k_bound"]):
        a5.text(v + .6, yi, "step %d" % v, va="center", fontsize=7.4, color=RED)
    for yi, v in zip(y + h / 2, d5["k_real"]):
        a5.text(v + .6, yi, "step %d" % v, va="center", fontsize=7.4, color=BLUE)
    a5.set_yticks(y); a5.set_yticklabels(short); a5.set_xlabel("rollout step")
    a5.set_xlim(0, max(d5["k_real"]) * 1.28)
    a5.set_title("(a)  On real dynamics the bound is uninformative", color=INK, loc="left")
    a5.legend(fontsize=7.1, frameon=False, loc="lower right"); a5.grid(alpha=.2, lw=.6, axis="x")

    # (b) transient one-step gain against the sustained rate
    b5 = A5[1]
    ge = [str(x).split("-")[0] for x in d6["gain_envs"]]
    x6 = np.arange(len(ge)); w6 = .38
    b5.bar(x6 - w6 / 2, d6["gain_one"], w6, color=SAND, label="one-step gain (transient)")
    b5.bar(x6 + w6 / 2, d6["gain_asym"], w6, color=BLUE, label=r"$e^{\lambda \Delta t}$ (sustained)")
    for xi, a_, b_ in zip(x6, d6["gain_one"], d6["gain_asym"]):
        b5.text(xi, max(a_, b_) * 1.25, "%.0f$\\times$" % (a_ / b_), ha="center",
                fontsize=7.8, weight="bold", color=INK)
    b5.set_yscale("log"); b5.set_xticks(x6); b5.set_xticklabels(ge, fontsize=7.6)
    b5.set_ylabel("amplification per step"); b5.set_ylim(.5, 60)
    b5.set_title("(b)  Arms: the two agree.  Legs: 5-10$\\times$ apart", color=INK, loc="left")
    b5.legend(fontsize=7.2, frameon=False, loc="upper left"); b5.grid(alpha=.2, lw=.6, axis="y")

    # (c) Reacher: the task wants a SHORT horizon
    c5 = A5[2]
    HS6, rew6, dist6 = d6["HS"], d6["rew"], d6["dist"]
    c5.plot(HS6, rew6, lw=2.2, color=BLUE, marker="o", ms=5, label="world-model planner")
    c5.axhline(float(d6["rand"]), color="#999", lw=1.2, ls="--")
    c5.text(HS6[0], float(d6["rand"]) * .95, "random actions", fontsize=7.2,
            color="#777", ha="left", va="bottom")
    bi = int(np.argmax(rew6))
    c5.plot([HS6[bi]], [rew6[bi]], "o", ms=9, mfc="none", mec=RED, mew=1.6)
    c5.annotate("best at H=%d\nthe pendulum needed 25" % HS6[bi], (HS6[bi], rew6[bi]),
                textcoords="offset points", xytext=(30, -34), fontsize=7.4, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=.9))
    c5.set_xlabel("planning horizon $H$"); c5.set_ylabel("reward per step (real robot)")
    c5.set_title("(c)  Reacher: the horizon the task wants is 5", color=INK, loc="left")
    c5.legend(fontsize=7.3, frameon=False, loc="center right"); c5.grid(alpha=.2, lw=.6)

    fig5.tight_layout()
    out5 = FIG / "real-robots.png"
    fig5.savefig(out5, bbox_inches="tight", facecolor="white")
    print("wrote", out5)
