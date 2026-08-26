"""Draw what baselines.py measured. Reads only the committed results.

    python make_figure.py

Medians with interquartile ranges, never means. The per-slice distributions are
wide enough that a bar without them would be a decoration.
"""
import os, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarise import load, stat

plt.rcParams.update({"font.size": 8.6, "axes.linewidth": .8, "figure.dpi": 160,
                     "font.family": ["Liberation Sans", "DejaVu Sans", "sans-serif"]})
INK, RED, PALE, MID, DARK = "#1a1a1a", "#c0392b", "#c8d8e4", "#5b8fb0", "#1a4f7a"

base = load("full_plain")
if base is None:
    raise SystemExit("no results committed here yet - run baselines.py first")

fig, AX = plt.subplots(1, 2, figsize=(11.4, 4.2))

# -- (a) the same predictors, scored globally and where the contrast matters --
a = AX[0]
methods = [("B0_identity", "B0\nhand back\nthe input"),
           ("B2_cond_mean", "B2\n256-entry\nlookup"),
           ("UNet", "U-Net\n2M params")]
x = np.arange(len(methods))
for k, (region, col, lab) in enumerate((("whole_slice", MID, "whole slice"),
                                        ("lesion_box", DARK, "lesion box"))):
    med = [stat(base, region, m, "ssim")[0] for m, _ in methods]
    lo = [stat(base, region, m, "ssim")[1] for m, _ in methods]
    hi = [stat(base, region, m, "ssim")[2] for m, _ in methods]
    a.bar(x + (k - .5) * .34, med, .32, color=col, label=lab, edgecolor="white", lw=.5)
    a.errorbar(x + (k - .5) * .34, med, yerr=[np.array(med) - lo, np.array(hi) - np.array(med)],
               fmt="none", ecolor=INK, elinewidth=.9, capsize=2.5, alpha=.65)
for i, (m, _) in enumerate(methods):
    w = stat(base, "whole_slice", m, "ssim")[0]; l = stat(base, "lesion_box", m, "ssim")[0]
    a.annotate("", xy=(i + .17, l + .012), xytext=(i - .17, w - .012),
               arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1, alpha=.75))
    a.text(i - .30, (w + l) / 2, "%+.2f" % (l - w), color=RED, fontsize=7.6,
           va="center", ha="right")
a.set_xticks(x); a.set_xticklabels([n for _, n in methods], fontsize=8)
a.set_ylabel("SSIM against the real post-contrast slice")
a.set_title("(a)  The same predictors, scored in two places", loc="left", fontsize=9.6)
a.set_ylim(0, 1.0); a.legend(fontsize=8, frameon=False, loc="upper left")
a.spines[["top", "right"]].set_visible(False)

# -- (b) nothing done to the model moves the lesion number -------------------
b = AX[1]
runs = [("full_plain", "plain"), ("full_w50", "lesion loss\nweight x50"),
        ("full_oracle", "one true scalar\nleaked in")]
vals, los, his, names = [], [], [], []
for tag, lab in runs:
    r = load(tag)
    if r is None:
        continue
    s = stat(r, "lesion_box", "UNet", "ssim")
    vals.append(s[0]); los.append(s[1]); his.append(s[2]); names.append(lab)
xx = np.arange(len(vals))
b.bar(xx, vals, .5, color=DARK, edgecolor="white", lw=.5)
b.errorbar(xx, vals, yerr=[np.array(vals) - los, np.array(his) - np.array(vals)],
           fmt="none", ecolor=INK, elinewidth=.9, capsize=2.5, alpha=.65)
b2 = stat(base, "lesion_box", "B2_cond_mean", "ssim")[0]
b.axhline(b2, color=RED, lw=1.3, ls="--")
b.text(-.42, b2 + .015, "B2, a lookup table that trains nothing",
       color=RED, fontsize=7.8, ha="left", va="bottom")
b0 = stat(base, "lesion_box", "B0_identity", "ssim")[0]
b.axhline(b0, color=INK, lw=1, ls=":", alpha=.6)
b.text(-.42, b0 - .018, "B0, hand back the input",
       color=INK, fontsize=7.8, ha="left", va="top", alpha=.8)
for i, v in enumerate(vals):
    b.text(i, v - .035, "%.3f" % v, ha="center", fontsize=8.4, color="white", weight="bold")
b.set_xticks(xx); b.set_xticklabels(names, fontsize=8)
b.set_ylabel("SSIM inside the lesion box")
# Not "nothing moves it": the oracle run moves it by +0.026, and the text is
# careful to report that without interpreting it. A panel title that rounds
# it to zero claims more than the run supports, in a figure whose whole
# point is that overclaiming is the failure mode.
b.set_title("(b)  Two things tried, and how little either moved it",
            loc="left", fontsize=9.6)
b.set_xlim(-.6, len(vals) - .4); b.set_ylim(0, 0.95)
b.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out = os.path.join(HERE, "dce-baselines.png")
fig.savefig(out, bbox_inches="tight")
print("  wrote %s" % out)
for region in ("whole_slice", "lesion_box"):
    row = "  %-12s " % region
    for m, _ in methods:
        row += "%s %.3f   " % (m.split("_")[0], stat(base, region, m, "ssim")[0])
    print(row)
