"""Watch a world model imagine a robot, next to the robot actually doing it.

    MUJOCO_GL=egl python make_imagination.py                 # ~40 s on one GPU
    MUJOCO_GL=egl python make_imagination.py Reacher-v5

This is the whole repository in one picture. A model is trained on random play,
given a starting state and a sequence of actions, and asked where the robot
ends up. The left panel is where it actually ends up. The right panel is the
model's answer, rendered by pushing its predicted states back into the
simulator - which is possible here because the state IS the simulator state,
and is exactly what a latent-space model cannot show you.

They start identical. The question this repository is about is when they stop
being, and the answer is measured with the same `wm.usable_horizon` the lessons
use, not eyeballed off the frames.

The default is the inverted pendulum because its state stays bounded, which
matters more than it sounds. On Reacher the joint angles are unbounded and a
90-step imagination wanders to 3e5, which renders as garbage and makes the
divergence a story about an unbounded coordinate rather than about the model.
Pusher is worse - 4e7. Pick a system where a wrong state is still a state.
"""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import wm

wm.ensure_headless_gl()
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import gymnasium as gym

ENV = sys.argv[1] if len(sys.argv) > 1 else "InvertedPendulum-v5"
K = 90                # steps to imagine
DEV = "cuda" if torch.cuda.is_available() else "cpu"
INK, BLUE, RED = "#1a1a1a", "#1a4f7a", "#c0392b"
FIG = pathlib.Path(__file__).parent / "figures"

rng = np.random.default_rng(0); torch.manual_seed(0)
sys_ = wm.GymSystem(ENV, seed=0)
print("%s: state %d, action %d" % (ENV, sys_.state_dim, sys_.action_dim))

S, A, Y = wm.make_dataset(sys_, 600, 30, rng)
model = wm.WorldModel(sys_.state_dim, sys_.action_dim, hidden=256, depth=3)
loss = wm.fit(model, S, A, Y, steps=8000, device=DEV)
print("  one-step MSE %.3e" % loss)

# The actions come from the repository's own planner, keeping the pole up. Left
# to random torques the pole falls within a few steps and stays down, and two
# fallen poles do not show anything. What is on screen is a model good enough to
# balance - and the same model's imagination drifting anyway.
U_MAX = float(sys_._hi[0])
# Started upright on purpose. Random play leaves the pole flat - three quarters
# of the training states sit at the 1.57 rad stop - and this environment cannot
# swing it back up, so a sampled start would be a fallen pole and the planner
# would have nothing to do. The model learns to balance from that data anyway,
# which is worth noticing on its own.
s0 = np.array([[0.0, 0.02, 0.0, 0.0]])
s0_t = torch.tensor(s0, dtype=torch.float32, device=DEV)


def reward(st, u):
    return -(10.0 * st[..., 1] ** 2 + 0.1 * st[..., 0] ** 2 + 0.01 * (u ** 2).sum(-1))


taken = []


def env_step(st, u):
    """The true dynamics, recording what was actually executed."""
    taken.append(u.detach().cpu().numpy().copy())
    return torch.tensor(sys_.step(st.cpu().numpy(), u.cpu().numpy()),
                        dtype=torch.float32, device=DEV)


mean_r, _, traj = wm.cem_mpc(lambda st, u: model(st.to(DEV), u.to(DEV)), env_step,
                             reward, s0_t, horizon=8, steps=K, u_max=U_MAX,
                             n_candidates=192, iters=4, n_elite=24,
                             action_dim=sys_.action_dim)
acts = np.stack(taken, 1)                       # (1, K, action_dim)
true = traj[:, 1:].cpu().numpy()                # (1, K, state_dim)
upright = float(np.abs(true[0, :, 1]).mean())
print("  planner: mean reward %.3f, mean |pole angle| %.3f rad over %d steps"
      % (float(mean_r), upright, K))

pred = wm.imagine(model, s0, K, acts, device=DEV)

# What to measure it against needs care here. The planner holds the system at a
# fixed point, so the state norm is 0.28 and a tolerance set at "10% of typical
# state size" is 0.028 - which the error crosses at step 2 while the imagined
# pole is still visibly upright. A percentage of a quantity that is near zero is
# not a tolerance. This environment supplies a real one: 0.2 rad is the angle at
# which it declares the episode failed. Measuring the imagined pole against the
# real one, in radians, is a threshold somebody chose for physical reasons.
FAIL = 0.2
ang_true, ang_pred = true[0, :, 1], pred[0, :, 1]
err = np.abs(ang_pred - ang_true)
h = int(wm.usable_horizon(err[None, :], FAIL)[0])
crossed = np.argmax(np.abs(ang_pred) > FAIL) + 1 if (np.abs(ang_pred) > FAIL).any() else K
print("  imagined pole leaves the +-%.1f rad band at step %d; the real one %s"
      % (FAIL, crossed, "never does" if (np.abs(ang_true) <= FAIL).all() else "does too"))
print("  imagination is within %.1f rad of reality for %d of %d steps" % (FAIL, h, K))

render_env = gym.make(ENV, render_mode="rgb_array")
render_env.reset(seed=0)


def film(states):
    out = []
    for s in states:
        render_env.unwrapped.set_state(np.asarray(s[:sys_.nq], dtype=float),
                                       np.asarray(s[sys_.nq:], dtype=float))
        out.append(render_env.render())
    return out


fr_true = film(np.concatenate([s0, true[0]]))
fr_pred = film(np.concatenate([s0, pred[0]]))
# Crop to what the scene actually occupies, over BOTH sequences and ALL frames.
# Taking the bounding box from one frame is the obvious thing and it is wrong:
# the empty space above the pole is as black as MuJoCo's border, so a first
# frame with the pole low crops the pole out of every later one.
stack = np.stack(fr_true + fr_pred)
rows = np.where(stack.max(axis=(0, 2, 3)) > 20)[0]
cols = np.where(stack.max(axis=(0, 1, 3)) > 20)[0]
if len(rows) and len(cols):
    m = 12
    r0, r1 = max(int(rows[0]) - m, 0), min(int(rows[-1]) + m, stack.shape[1])
    c0, c1 = max(int(cols[0]) - m, 0), min(int(cols[-1]) + m, stack.shape[2])
    fr_true = [f[r0:r1, c0:c1] for f in fr_true]
    fr_pred = [f[r0:r1, c0:c1] for f in fr_pred]
    print("  cropped to the scene: rows %d-%d, cols %d-%d" % (r0, r1, c0, c1))
render_env.close(); sys_.close()

plt.rcParams.update({"font.size": 8.6, "figure.dpi": 92,
                     "font.family": ["Liberation Sans", "DejaVu Sans", "sans-serif"]})
a_t = np.concatenate([[s0[0, 1]], ang_true])
a_p = np.concatenate([[s0[0, 1]], ang_pred])
# GitHub shows the first frame until a reader presses play, and at step 0 the
# two panels are identical - which is the one thing this animation is not
# about. Start at the moment the imagination has visibly left, and let the loop
# carry the reader back round to where they were the same.
POSTER = min(crossed + 14, K)
order = list(range(POSTER, K + 1)) + list(range(0, POSTER))
frames = []
for t in order:
    fig = plt.figure(figsize=(6.8, 4.4))
    for j, (frs, title, col) in enumerate(
            ((fr_true, "reality: the planner holds it up", INK),
             (fr_pred, "the same model imagining the same actions",
              RED if t >= crossed else BLUE))):
        ax = fig.add_axes([0.02 + j * 0.49, 0.38, 0.47, 0.56])
        ax.imshow(frs[t]); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=9.2, color=col)
    c = fig.add_axes([0.10, 0.11, 0.86, 0.21])
    c.axhspan(-FAIL, FAIL, color="#eef3f7", lw=0)
    c.axhline(FAIL, color=RED, lw=.9, ls="--"); c.axhline(-FAIL, color=RED, lw=.9, ls="--")
    c.plot(np.arange(K + 1), a_t, lw=.8, color="#dcdcdc")
    c.plot(np.arange(K + 1), a_p, lw=.8, color="#f0d8d0")
    c.plot(np.arange(t + 1), a_t[:t + 1], lw=1.6, color=INK, label="reality")
    c.plot(np.arange(t + 1), a_p[:t + 1], lw=1.6, color=RED, label="imagined")
    c.text(K * .99, FAIL * 1.15, "the angle at which this environment calls it a failure",
           color=RED, fontsize=7.2, ha="right", va="bottom")
    c.set_xlim(0, K)
    lim = max(float(np.abs(a_p).max()) * 1.1, FAIL * 2.2)
    c.set_ylim(-lim, lim)
    c.set_xlabel("step  (same start, same actions)", fontsize=8)
    c.set_ylabel("pole angle (rad)", fontsize=8)
    c.legend(fontsize=7.6, frameon=False, ncol=2, loc="lower left",
             bbox_to_anchor=(0, 1.0, 1, .14), mode="expand", borderaxespad=0)
    c.spines[["top", "right"]].set_visible(False)
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)

frames += [frames[-1]] * 8          # hold the end so a loop is readable
out = FIG / ("imagination-%s.gif" % ENV.split("-")[0].lower())
imageio.mimsave(out, frames, fps=12, loop=0, subrectangles=True)
print("  wrote %s (%d frames, %.1f MB)" % (out, len(frames), out.stat().st_size / 1e6))
