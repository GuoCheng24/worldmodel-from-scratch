"""Draw what diagnose_tdmpc2.py measured on the released TD-MPC2 checkpoints.

    python make_figure.py          # needs numpy and matplotlib, nothing else

Nothing here computes a result; it plots the committed .npz files, and every
summary it takes is a median or a percentile. Lesson 2 of this repository is
that the mean over rollouts describes a runaway minority rather than the
typical one, so no panel here averages.
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarise import load, facts, PLAN_H, SIZES

runs = load()
if not runs:
    raise SystemExit("no results_mt30-*.npz here yet - run the diagnosis first.")
F = facts(runs)
sizes = [s for s in SIZES if s in F]
big = sizes[-1]
tasks = F[big]["tasks"]
K = runs[big]["err__" + tasks[0]].shape[1]

plt.rcParams.update({"font.size": 8.8, "axes.linewidth": .8, "figure.dpi": 160,
                     "font.family": ["Liberation Sans", "DejaVu Sans", "sans-serif"]})
INK, RED, GREY = "#1a1a1a", "#c0392b", "#8a8a8a"
SHADE = ["#c8d8e4", "#5b8fb0", "#1a4f7a"][-len(sizes):]
COL = ["#1a4f7a", "#c0392b", "#2e7d5b", "#b8860b", "#6a4c93", "#0e8a8a", "#d4703a", "#7a7a7a"]

fig, AX = plt.subplots(1, 3, figsize=(15.6, 4.35))

# -- (a) the error curve, per task, as a share of the latent's real motion ----
a = AX[0]
a.axvspan(1, PLAN_H, color=GREY, alpha=.15, lw=0)
r = runs[big]
YMAX = 190
curves = {}
for c, t in zip(COL, tasks):
    e, n = np.median(r["err__" + t], 0), np.median(r["still__" + t], 0)
    curves[t] = 100 * e / np.maximum(n, 1e-12)
    a.plot(np.arange(1, K + 1), curves[t], lw=1.5, color=c)


def stack(desired, lo, hi, gap):
    """Spread labels so none collide and none leave the axes.

    Placing a label at its curve's last value and nudging on collision looks
    fine until a run puts several curves off the top: the nudged labels land
    outside the axes, and a tight bounding box then grows the figure to
    include them. This keeps every label inside [lo, hi] whatever the data
    does, which is the only version that survives a rerun.
    """
    order = sorted(range(len(desired)), key=lambda i: desired[i])
    y = [min(max(desired[i], lo), hi) for i in range(len(desired))]
    run = lo
    for i in order:                       # push up, in order
        y[i] = max(y[i], run); run = y[i] + gap
    over = y[order[-1]] - hi
    if over > 0:                          # then pull the whole stack back down
        run = hi
        for i in reversed(order):
            y[i] = min(y[i], run); run = y[i] - gap
    return y


ends = [float(curves[t][-1]) for t in tasks]
for t, y in zip(tasks, stack(ends, 6, YMAX - 6, 9.5)):
    a.text(K + 0.4, y, t, color=COL[tasks.index(t)], fontsize=7.6, va="center")
a.axhline(100, color=RED, lw=1.3, ls="--")
a.text(1.3, 104, "error equals the motion: the prediction adds nothing",
       color=RED, fontsize=7.6, va="bottom")
a.text(PLAN_H, YMAX - 4, "  the %d steps the planner rolls out" % PLAN_H,
       fontsize=7.4, color=INK, va="top")
a.set_xlabel("open-loop step k"); a.set_ylabel("median error, % of the latent's real motion")
a.set_title("(a)  mt30-%s: how fast the prediction stops paying" % big, loc="left", fontsize=9.6)
a.set_xlim(1, K + 6.2); a.set_ylim(0, YMAX); a.set_xticks([1, 5, 10, 15, 20])
a.spines[["top", "right"]].set_visible(False)

# -- (b) the decision-relevant share, at the planner's own horizon -----------
b, y = AX[1], np.arange(len(tasks))[::-1]
for j, (s, sh) in enumerate(zip(sizes, SHADE)):
    v = [F[s]["loses"][t] for t in tasks]
    b.barh(y + (j - (len(sizes) - 1) / 2) * .26, v, .25, color=sh,
           label="mt30-" + s, edgecolor="white", linewidth=.4)
b.axvline(50, color=RED, lw=1.2, ls="--")
b.text(51, -.72, "half the starts", color=RED, fontsize=7.4, va="bottom")
b.set_yticks(y); b.set_yticklabels(tasks, fontsize=8)
b.set_xlabel("share of starts where the k=%d prediction loses to standing still" % PLAN_H)
b.set_title("(b)  Worse than assuming nothing changed", loc="left", fontsize=9.6)
b.set_xlim(0, 104); b.set_ylim(-1.1, len(tasks) - .4)
# The bars stop well short of the right edge on every row but one, so the
# key goes there rather than over the title or over the data.
b.legend(fontsize=7.8, frameon=False, ncol=1, loc="center right",
         bbox_to_anchor=(1.02, .46), handlelength=1.1, labelspacing=.9)
b.spines[["top", "right"]].set_visible(False)

# -- (c) the horizon is not one number, on somebody else's model -------------
c = AX[2]
for i, t in enumerate(tasks):
    p5, p50, p95, cens = F[big]["trust"][t]
    yy = len(tasks) - 1 - i
    c.plot([p5, p95], [yy, yy], lw=5, color="#c8d8e4", solid_capstyle="round")
    c.plot([p50], [yy], "o", ms=6, color="#1a4f7a", zorder=3)
    if cens:
        c.annotate("", xy=(K + 1.6, yy), xytext=(K - .2, yy),
                   arrowprops=dict(arrowstyle="-|>", color="#5b8fb0", lw=1.2))
c.axvline(PLAN_H, color=RED, lw=1.3)
c.text(PLAN_H + .35, -.55, "the %d steps the planner uses" % PLAN_H,
       color=RED, fontsize=7.4, va="bottom")
c.set_yticks(np.arange(len(tasks))[::-1]); c.set_yticklabels(tasks, fontsize=8)
c.set_xlabel("steps the prediction still beats standing still, per start")
c.set_title("(c)  mt30-%s: one model, one task, no single horizon" % big,
            loc="left", fontsize=9.6)
c.set_xlim(0, K + 2.4); c.set_ylim(-1.1, len(tasks) - .4)
c.set_xticks([1, 5, 10, 15, 20])
c.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = os.path.join(HERE, "tdmpc2-diagnosis.png")
fig.savefig(out, bbox_inches="tight")
print("  wrote %s" % out)
for s in sizes:
    print("  mt30-%-5s at k=%d loses to standing still on median %.0f%% of starts (%.0f-%.0f%%)"
          % (s, PLAN_H, F[s]["median"], F[s]["lo"], F[s]["hi"]))
