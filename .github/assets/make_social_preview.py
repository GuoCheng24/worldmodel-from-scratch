"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

The card draws the lesson: a learned model that starts far better than assuming nothing changes,
and is worse than that by the end of a twenty-step rollout. Two curves crossing is legible at the
360 px a Slack unfurl gives a card; a block of terminal output is not.

Data is integrations/tdmpc2/results_mt30-317M.npz, committed here - median over 640 episodes of
TD-MPC2's own released 317M checkpoint on pendulum-swingup.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cardkit import SANS, card  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK = "pendulum-swingup"
d = np.load(ROOT / "integrations/tdmpc2/results_mt30-317M.npz", allow_pickle=True)
err = np.median(d[f"err__{TASK}"], axis=0)
still = np.median(d[f"still__{TASK}"], axis=0)
steps = np.arange(1, len(err) + 1)
cross = int(steps[np.argmax(err > still)]) if (err > still).any() else None
if cross is None:
    raise SystemExit("the model never becomes worse than the still baseline; card copy is wrong")


def chart(ax, accent):
    x0, x1, y0, y1 = 3.95, 10.55, 1.15, 3.55
    top = float(max(err.max(), still.max())) * 1.08

    def X(s):
        return x0 + (x1 - x0) * (s - 1) / (len(err) - 1)

    def Y(v):
        return y0 + (y1 - y0) * v / top

    ax.plot([X(s) for s in steps], [Y(v) for v in still], color="#8c8f94", lw=5, zorder=3)
    ax.plot([X(s) for s in steps], [Y(v) for v in err], color="#cf222e", lw=6, zorder=4)
    ax.plot([X(cross)], [Y(still[cross - 1])], "o", ms=18, color="#cf222e", zorder=5)
    # the dotted drop stops clear of its own label; a vertical rule through
    # text passed every check until the rule test learned about vertical rules
    ax.plot([X(cross), X(cross)], [y0 + 0.08, Y(still[cross - 1])], color="#cf222e", lw=2,
            ls=":", zorder=2)

    ax.text(X(1) - 0.20, Y(err[0]), "the model", fontsize=34, color="#cf222e",
            fontweight="bold", family=SANS, ha="right", va="center")
    ax.text(X(1) - 0.20, Y(still[0]) + 0.30, "assume nothing", fontsize=34, color="#55585c",
            family=SANS, ha="right", va="center")
    ax.text(X(cross), y0 - 0.40, f"step {cross}", fontsize=34, color="#cf222e",
            fontweight="bold", family=SANS, ha="center")
    ax.text(X(len(err)) + 0.16, Y(err[-1]), f"{err[-1] / still[-1]:.0f}x worse",
            fontsize=34, fontweight="bold", color="#cf222e", family=SANS,
            ha="right", va="bottom")


out = card(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#1a7f37", badge="W",
    kicker="SIX RUNNABLE LESSONS  ·  no simulator, no MuJoCo",
    headline="Build a world model in an afternoon",
    evidence=f"TD-MPC2 317M, {TASK}, 640 rollouts",
    chart=chart,
    footer="github.com/GuoCheng24/worldmodel-from-scratch",
    headline_size=44,
)
print(f"written {pathlib.Path(out).name}  crosses at step {cross}, "
      f"ends {err[-1]:.2f} vs {still[-1]:.2f}")
