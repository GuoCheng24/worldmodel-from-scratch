"""Generate the GitHub social-preview card (1280x640). Reproducible: python3 make_social_preview.py"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 12.8, 6.4
fig = plt.figure(figsize=(W, H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
ax.add_patch(plt.Rectangle((0, 0), W, H, color="#0d1117"))
SANS, MONO = "Liberation Sans", "Liberation Mono"
ax.text(0.75, 5.55, "worldmodel-from-scratch", fontsize=34, fontweight="bold", color="#e6edf3", family=SANS)
ax.text(0.75, 4.92, "Build a world model in an afternoon - then find out how far you can trust it.", fontsize=17, color="#8b949e", family=SANS)
ax.add_patch(FancyBboxPatch((0.72, 1.28), 11.36, 3.05, boxstyle="round,pad=0.12", fc="#161b22", ec="#30363d", lw=1.5))
ax.text(0.95, 4.02, ">>> wm.diagnose(model, env)", fontsize=13.5, color="#7d8590", family=MONO)
rows = [
    ("one-step loss        1e-03     looks fine", "#e6edf3"),
    ("control (90 steps)   upright   every run", "#3fb950"),
    ("imagination          falls     between step 13 and 23", "#f85149"),
    ("usable horizon       ~20       not one number, even within a task", "#58a6ff"),
]
y = 3.55
for txt, c in rows:
    ax.text(0.95, y, txt, fontsize=14, color=c, family=MONO); y -= 0.5
ax.text(0.95, y - 0.02, "six lessons | torch + numpy + matplotlib | ~1 min on one GPU | 81/81 README claims checked in CI",
        fontsize=12, color="#7d8590", family=MONO)
ax.text(0.75, 0.62, "The same measurement pointed at TD-MPC2's released checkpoints, through TD-MPC2's own code.",
        fontsize=12.5, color="#8b949e", family=SANS)
fig.savefig(pathlib.Path(__file__).parent / "social-preview.png"); print("written social-preview.png 1280x640")
