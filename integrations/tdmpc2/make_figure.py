"""Draw what diagnose_tdmpc2.py measured on the released TD-MPC2 checkpoints.

    python diagnose_tdmpc2.py task=mt30 model_size=1  checkpoint=.../mt30-1M.pt
    python diagnose_tdmpc2.py task=mt30 model_size=48 checkpoint=.../mt30-48M.pt
    python make_figure.py

Nothing here computes a result; it only plots the .npz files those runs wrote,
so the figure cannot disagree with the run that produced it.
"""
import pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
PLAN_H = 3          # TD-MPC2 rolls the model forward this many steps to score actions

runs = {}
for size in ("1M", "48M", "317M"):
    f = HERE / ("results_mt30-%s.npz" % size)
    if f.exists():
        runs[size] = np.load(f)
if "48M" not in runs:
    raise SystemExit(__doc__ + "\nNo results_mt30-48M.npz here yet - run the diagnosis first.")

tasks = [f for f in runs["48M"].files if not f.startswith("horizon__")]
plt.rcParams.update({"font.size": 8.6, "axes.linewidth": .8, "figure.dpi": 160,
                     "font.family": ["Liberation Sans", "DejaVu Sans", "sans-serif"]})
INK, RED, GREY = "#1a1a1a", "#c0392b", "#8a8a8a"
COL = ["#1a4f7a", "#c0392b", "#2e7d5b", "#b8860b", "#6a4c93",
       "#0e8a8a", "#d4703a", "#7a7a7a"][:len(tasks)]

fig, AX = plt.subplots(1, 2, figsize=(12.6, 4.4))

# -- A. how much of the latent's real motion the error eats, step by step -----
a = AX[0]
sizes = sorted(runs, key=lambda s: float(s[:-1]))
label_size, big = sizes[-1], runs[sorted(runs, key=lambda s: float(s[:-1]))[-1]]
curves = {t: 100 * big[t][0] / np.maximum(big[t][1], 1e-12) for t in tasks}
K = len(next(iter(curves.values())))
a.axvspan(1, PLAN_H, color=GREY, alpha=.15, lw=0)
# Names sit at the end of their own line, so no legend box lands on the data.
ends = sorted(tasks, key=lambda t: curves[t][-1])
taken = [100.0]      # keep names off the "adds nothing" line they would sit on
for c, t in zip(COL, tasks):
    a.plot(np.arange(1, K + 1), curves[t], lw=1.5, color=c)
for t in ends:
    c = COL[tasks.index(t)]
    y = curves[t][-1]
    while any(abs(y - u) < 9 for u in taken):
        y += 9
    taken.append(y)
    a.text(K + 0.4, y, t, color=c, fontsize=7.6, va="center")
a.axhline(100, color=RED, lw=1.3, ls="--")
a.text(1.3, 107, "error equals the motion: the prediction adds nothing",
       ha="left", va="bottom", color=RED, fontsize=7.6)
a.text(PLAN_H, 186, "  the %d steps the planner rolls out" % PLAN_H,
       fontsize=7.4, color=INK, va="top", ha="left")
a.set_xlabel("open-loop step k")
a.set_ylabel("prediction error, % of the latent's real motion")
a.set_title("mt30-%s: open-loop latent error, per task" % label_size, loc="left", fontsize=9.6)
a.set_xlim(1, K + 6.2); a.set_ylim(0, 190)
a.set_xticks([1, 5, 10, 15, 20])
a.spines[["top", "right"]].set_visible(False)

# -- B. the same quantity at the planner's horizon, across model sizes -------
b = AX[1]
w, x = 0.8 / len(sizes), np.arange(len(tasks))
shades = ["#c8d8e4", "#5b8fb0", "#1a4f7a"][-len(sizes):]
for j, (size, sh) in enumerate(zip(sizes, shades)):
    v = [100 * runs[size][t][0][PLAN_H - 1] / max(runs[size][t][1][PLAN_H - 1], 1e-12) for t in tasks]
    b.bar(x + j * w - 0.4 + w / 2, v, w, color=sh, label="mt30-%s" % size,
          edgecolor="white", linewidth=.5)
b.axhline(100, color=RED, lw=1.3, ls="--")
b.text(len(tasks) - .5, 104, "prediction adds nothing", ha="right", va="bottom",
       color=RED, fontsize=7.6)
b.set_xticks(x); b.set_xticklabels(tasks, rotation=32, ha="right", fontsize=7.6)
b.set_ylabel("error at k=%d, %% of the latent's real motion" % PLAN_H)
b.set_title("At the horizon TD-MPC2 actually plans over", loc="left", fontsize=9.6)
b.legend(fontsize=7.8, frameon=False, loc="upper right")
b.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = HERE / "tdmpc2-diagnosis.png"
fig.savefig(out, bbox_inches="tight")
print("  wrote %s" % out)
for size in sorted(runs, key=lambda s: float(s[:-1])):
    v = [100 * runs[size][t][0][PLAN_H - 1] / max(runs[size][t][1][PLAN_H - 1], 1e-12) for t in tasks]
    print("  mt30-%-5s at k=%d: median %3.0f%% of the real motion, range %.0f%%-%.0f%%"
          % (size, PLAN_H, np.median(v), min(v), max(v)))
