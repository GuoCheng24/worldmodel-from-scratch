"""Watch a published world model's trust horizon move, along one real episode.

    TDMPC2_DIR=... python make_animation.py task=mt30 model_size=48 \
        checkpoint=/path/mt30-48M.pt +anim_task=cheetah-run

One number per model is the thing this repository argues against, and the
argument is easier to see than to read. This runs a single episode under
TD-MPC2's own planner and, at every step, asks the same question from that
moment: rolling the learned dynamics forward from here, under the actions the
agent actually took, how many steps before the prediction error reaches the
distance the latent really travels - the point where the rollout is worth no
more than assuming nothing changes at all.

The answer is not a property of the model. It moves, along one episode, on one
task, by more than an order of magnitude.
"""
import sys, os
TD = os.environ.get("TDMPC2_DIR")
if TD is None or not os.path.isdir(TD):
    raise SystemExit("set TDMPC2_DIR to the tdmpc2/tdmpc2 directory of a clone of\n"
                     "https://github.com/nicklashansen/tdmpc2 (see README.md here)")
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, TD)
os.chdir(TD)
import numpy as np, torch, hydra, imageio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2

K = 20          # how far ahead the question is asked
EVERY = 6       # keep every Nth frame, so the gif stays small
FPS = 7
TASKS = ["cheetah-run", "cartpole-swingup"]   # one that recovers, one that does not
INK, BLUE, RED, GREY = "#1a1a1a", "#1a4f7a", "#c0392b", "#e2e2e2"
PLAN_H_LINE = 3


@hydra.main(config_name="config", config_path=os.environ["TDMPC2_DIR"], version_base=None)
def main(cfg):
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    env = make_env(cfg)
    agent = TDMPC2(cfg); agent.load(cfg.checkpoint)
    model = agent.model
    tasks = list(cfg.get("anim_tasks", TASKS))
    # Replanning two episodes to redraw a label is a waste of a GPU; the
    # frames and the trust curve are all the drawing needs.
    cache = os.environ.get("ANIM_CACHE")
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        data = list(z["data"])
        print("  reusing episodes from %s" % cache, flush=True)
        return draw(data, cfg)
    data = []
    for task in tasks:
        idx = cfg.tasks.index(task) if cfg.multitask else None
        obs, O, A, frames = env.reset(task_idx=idx), [], [], []
        for t in range(cfg.episode_length):
            a = agent.act(obs, t0=(t == 0), eval_mode=True, task=idx)
            O.append(obs); A.append(a)
            frames.append(env.render())   # gymnasium's Wrapper.render forwards no args
            obs, r, done, _ = env.step(a)
            if done:
                break
        O.append(obs)

        Ot = torch.stack(O).cuda().float()
        At = torch.stack(A).cuda().float()
        n = len(A) - K
        with torch.no_grad():
            Z = model.encode(Ot, idx)
            z = Z[:n].clone()
            err, still = [], []
            for j in range(K):
                z = model.next(z, At[j:j + n], idx)
                true = Z[1 + j:1 + j + n]
                err.append((z - true).norm(dim=-1))
                still.append((Z[:n] - true).norm(dim=-1))
        E = torch.stack(err, 1).cpu().numpy()
        S = torch.stack(still, 1).cpu().numpy()
        # First step at which the prediction stops being worth more than
        # standing still. Never crossing inside the window is reported as K,
        # not as a larger number the data cannot support.
        crossed = E >= S
        trust = np.where(crossed.any(1), crossed.argmax(1) + 1, K)
        data.append((task, frames, trust, n))
        print("  %-18s trust horizon over the episode: min %d, median %d, max %d"
              % (task, trust.min(), int(np.median(trust)), trust.max()), flush=True)

    if cache:
        np.savez_compressed(cache, data=np.array(data, dtype=object))
    return draw(data, cfg)


def draw(data, cfg):
    plt.rcParams.update({"font.size": 8.4, "figure.dpi": 84,
                         "font.family": ["Liberation Sans", "DejaVu Sans", "sans-serif"]})
    n = min(d[3] for d in data)
    # GitHub shows the first frame until a reader presses play, and a reader who
    # has asked for reduced motion never sees any other. Starting where the two
    # tasks actually disagree makes that one frame carry the finding instead of
    # contradicting it - at step 0 both read 20, which says the opposite.
    gap = np.abs(data[0][2][:n].astype(int) - data[1][2][:n].astype(int))
    start = int(np.argmax(gap >= 10)) if (gap >= 10).any() else 0
    out = []
    for t in range(start, n, EVERY):
        fig = plt.figure(figsize=(6.9, 4.7))
        gs = GridSpec(2, len(data), height_ratios=[1.55, 1], figure=fig,
                      wspace=.04, hspace=.34)
        for j, (task, frames, trust, _) in enumerate(data):
            ax = fig.add_subplot(gs[0, j])
            ax.imshow(frames[t]); ax.axis("off")
            ax.set_title("%s" % task, loc="left", fontsize=9.4, color=INK)
            # The number goes on the frame: it is the thing being watched.
            ax.text(.98, .03, "%d" % trust[t], transform=ax.transAxes, ha="right",
                    va="bottom", fontsize=24, color="#7fb3d5", weight="bold")
            ax.text(.03, .05, "steps still worth\nrolling out", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=7, color="#dfe8ef")

        c = fig.add_subplot(gs[1, :])
        for j, (task, frames, trust, _) in enumerate(data):
            c.plot(np.arange(n), trust[:n], lw=.8, color=(GREY if j else "#dcdcdc"),
                   zorder=1)
            c.plot(np.arange(t + 1), trust[:t + 1], lw=1.5,
                   color=(BLUE if j == 0 else "#d4703a"), label=task, zorder=2)
        c.axhline(PLAN_H_LINE, color=RED, lw=1.1, ls="--")
        c.text(n * .995, PLAN_H_LINE + .8, "the %d steps TD-MPC2 plans over" % PLAN_H_LINE,
               color=RED, fontsize=7.2, ha="right", va="bottom",
               bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=.88))
        c.set_xlim(0, n); c.set_ylim(0, K + 1.6)
        c.set_xlabel("step in the episode", fontsize=8)
        c.set_ylabel("steps the prediction\nstill beats standing still", fontsize=8)
        c.legend(fontsize=7.8, frameon=False, ncol=2, loc="lower left",
                 bbox_to_anchor=(0, 1.0, 1, .16), mode="expand", borderaxespad=0)
        c.spines[["top", "right"]].set_visible(False)

        fig.canvas.draw()
        out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)

    path = os.path.join(HERE, "tdmpc2-trust.gif")
    imageio.mimsave(path, out, fps=FPS, loop=0, subrectangles=True)
    print("  wrote %s (%d frames, %.1f MB)"
          % (path, len(out), os.path.getsize(path) / 1e6), flush=True)


main()
