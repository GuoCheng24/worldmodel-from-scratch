"""Animations of what the lessons measure, for the README.

A curve of rollout error tells you that a model drifts. It does not let you see
it drift, and the seeing is what makes the rest of the repository worth reading.
This script produces two short GIFs:

    drift.gif    the true pendulum and the model's imagination of it, started
                 from the same state under the same torques, drawn side by side
                 until they come apart
    swingup.gif  the same model used to plan, swinging the pendulum up and
                 holding it - the task from Lesson 3

Both are rendered from the same code the lessons use, so what you watch is the
thing that was measured rather than an illustration of it.

    python make_visuals.py            (~40 s on one GPU)
"""
import pathlib, sys
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import wm

FIG = pathlib.Path(__file__).parent / "figures"; FIG.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
INK, BLUE, RED, GREEN, SAND = "#1a1a1a", "#1a4f7a", "#c0392b", "#2e7d5b", "#e08e6d"
plt.rcParams.update({"font.size": 9, "font.family": ["Liberation Sans", "DejaVu Sans", "sans-serif"]})

rng = np.random.default_rng(0); torch.manual_seed(0)
pend = wm.Pendulum()
print("Training the same model Lesson 2 uses ...")
S, A, Y = wm.make_dataset(pend, 4000, 30, rng)
model = wm.WorldModel(2, 1)
wm.fit(model, S, A, Y, steps=4000, device=DEV)
model.to(DEV).eval()


def rod(ax, theta, color, lw, alpha=1.0, label=None):
    """A pendulum drawn hanging from the origin; theta=0 is straight down."""
    x, y = np.sin(theta), -np.cos(theta)
    (line,) = ax.plot([0, x], [0, y], lw=lw, color=color, alpha=alpha,
                      solid_capstyle="round", label=label)
    (bob,) = ax.plot([x], [y], "o", ms=lw * 2.6, color=color, alpha=alpha)
    return line, bob


# ══════════════════════════════════════════════════════════════════════════
# 1. drift.gif - what usually happens, and what the minority does
# ══════════════════════════════════════════════════════════════════════════
# A first attempt animated a single median trajectory. It was honest and
# useless: the median error at step 90 is about 0.05 rad, so the two pendulums
# sit on top of each other and there is nothing to watch. The interesting thing
# is not the average - it is that most rollouts stay together while a measured
# 14% go over the top and never come back, and that minority is what drags the
# mean into a different functional form in Lesson 2. So both are shown.
K = 120
s0 = pend.sample_states(600, rng)
acts = pend.sample_actions(600, K, rng)
true = wm.rollout(pend, s0, K, acts)
pred = wm.imagine(model, s0, K, acts, device=DEV)
err = wm.rollout_error(pred, true)
gap = np.abs(pred[..., 0] - true[..., 0])
switched = gap.max(1) > np.pi
frac = 100 * switched.mean()

typical = int(np.argsort(np.abs(err[:, -1] - np.median(err[:, -1])))[0])
if switched.any():
    cand = np.where(switched)[0]
    # the earliest switcher, so the moment of departure is inside the window
    broke = int(cand[np.argmin([np.argmax(gap[c] > np.pi) for c in cand])])
else:
    broke = int(np.argmax(err[:, -1]))
print("  drift.gif: %.0f%% of rollouts switch branch; showing #%d (typical) and #%d (switcher)"
      % (frac, typical, broke))

fig = plt.figure(figsize=(10.2, 4.6))
axA = fig.add_axes([0.02, 0.52, 0.30, 0.42])
axB = fig.add_axes([0.02, 0.05, 0.30, 0.42])
axe = fig.add_axes([0.42, 0.13, 0.55, 0.76])
# The fraction rises with the horizon: Lesson 2 quotes 14% by step 90 and this
# runs further, so the label carries the step rather than a bare percentage that
# would read as contradicting it.
switch_label = "one of the %.0f%% that goes over the top by step %d" % (frac, K)
for ax, ttl, col in ((axA, "a typical rollout", BLUE), (axB, switch_label, RED)):
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.35, 1.35); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([0], [0], "o", ms=4, color=INK, zorder=5)
    ax.set_title(ttl, fontsize=9.5, color=col, pad=2)
axe.set_xlim(0, K); axe.set_yscale("log")
axe.set_ylim(1e-4, max(err[broke].max(), err[typical].max()) * 2.5)
axe.set_xlabel("rollout step"); axe.set_ylabel("state error")
axe.grid(alpha=.2, lw=.6)
axe.set_title("the model is fine until it is not", fontsize=10.5, color=INK)
(cA,) = axe.plot([], [], lw=2.2, color=BLUE, label="typical")
(cB,) = axe.plot([], [], lw=2.2, color=RED, label="switcher")
axe.legend(fontsize=8.6, frameon=False, loc="lower right")
axe.plot([], [], lw=4, color=INK, label="_")
axA.legend([plt.Line2D([], [], lw=4, color=INK), plt.Line2D([], [], lw=4, color=SAND)],
                ["true dynamics", "world model"], loc="upper center", fontsize=8.2,
                frameon=False, ncol=2, bbox_to_anchor=(.5, 1.02))
step_txt = axe.text(.02, .95, "", transform=axe.transAxes, fontsize=9.5, color=INK)
drawn = []


def frame(k):
    for a in drawn: a.remove()
    drawn.clear()
    for ax, idx in ((axA, typical), (axB, broke)):
        drawn.extend(rod(ax, true[idx, k, 0], INK, 6.0))
        drawn.extend(rod(ax, pred[idx, k, 0], SAND, 6.0, alpha=.9))
    cA.set_data(np.arange(1, k + 2), err[typical, :k + 1])
    cB.set_data(np.arange(1, k + 2), err[broke, :k + 1])
    step_txt.set_text("step %d" % (k + 1))
    return drawn + [cA, cB, step_txt]


FuncAnimation(fig, frame, frames=K, blit=False).save(
    FIG / "drift.gif", writer=PillowWriter(fps=16), dpi=88)
plt.close(fig)
print("  wrote figures/drift.gif")


# ══════════════════════════════════════════════════════════════════════════
# 2. arm.gif - a real robot arm, driven by a model learned from random play
# ══════════════════════════════════════════════════════════════════════════
# Needs Gymnasium and MuJoCo, which Lessons 1-4 do not. Skipped rather than
# fatal when they are absent, so this script still produces drift.gif on a
# machine with nothing but torch.
try:
    wm.ensure_headless_gl()
    import gymnasium, mujoco  # noqa: F401
except ImportError:
    print("  arm.gif skipped: pip install gymnasium mujoco (see Lesson 5)")
    raise SystemExit(0)

print("\nTraining the Reacher model Lesson 6 uses ...")
rng = np.random.default_rng(0); torch.manual_seed(0)
S6 = wm.GymSystem("Reacher-v5", seed=0)
s0 = S6.sample_states(3000, rng); A6 = S6.sample_actions(3000, 30, rng)
T6 = wm.rollout(S6, s0, 30, A6)
X6 = np.concatenate([s0[:, None], T6[:, :-1]], 1).reshape(-1, S6.state_dim)
m6 = wm.WorldModel(S6.state_dim, S6.action_dim, hidden=384, depth=3)
wm.fit(m6, X6, A6.reshape(-1, S6.action_dim), T6.reshape(-1, S6.state_dim),
       steps=12000, batch=1024, device=DEV)
m6.to(DEV).eval()

L1, L2 = 0.1, 0.11


def fingertip(s):
    t0, t1 = s[..., 0], s[..., 0] + s[..., 1]
    return torch.stack([L1 * torch.cos(t0) + L2 * torch.cos(t1),
                        L1 * torch.sin(t0) + L2 * torch.sin(t1)], -1)


def reward6(s, u):
    return -((fingertip(s) - s[..., 2:4]).norm(dim=-1) + 0.01 * (u ** 2).sum(-1))


def env_step(s, a):
    return torch.tensor(S6.step(s.cpu().numpy(), a.cpu().numpy()),
                        dtype=torch.float32, device=DEV)


import gymnasium as gym
render_env = gym.make("Reacher-v5", render_mode="rgb_array")
render_env.reset(seed=0)


def film(states):
    """Replay a state sequence through a rendering environment, frame by frame."""
    out = []
    for s in states:
        render_env.unwrapped.set_state(np.asarray(s[:S6.nq]), np.asarray(s[S6.nq:]))
        out.append(render_env.render())
    return out


EP = 60
start = torch.tensor(S6.sample_states(1, rng), dtype=torch.float32, device=DEV)
start = start.repeat(1, 1)
# Same start for both, so the difference on screen is the planner and nothing else.
_, _, traj_plan = wm.cem_mpc(lambda s, a: m6(s, a), env_step, reward6, start, 5, EP,
                             u_max=1.0, n_candidates=128, iters=3, n_elite=16,
                             action_dim=S6.action_dim)
_, _, traj_rand = wm.cem_mpc(lambda s, a: s, env_step, reward6, start, 1, EP,
                             u_max=1.0, n_candidates=8, iters=1, n_elite=2,
                             action_dim=S6.action_dim)
d_plan = (fingertip(traj_plan[0]) - traj_plan[0][..., 2:4]).norm(dim=-1).cpu().numpy()
d_rand = (fingertip(traj_rand[0]) - traj_rand[0][..., 2:4]).norm(dim=-1).cpu().numpy()
print("  final fingertip-to-target: planner %.4f, random %.4f" % (d_plan[-1], d_rand[-1]))

fr_plan = film(traj_plan[0].cpu().numpy())
fr_rand = film(traj_rand[0].cpu().numpy())
# MuJoCo's offscreen buffer leaves a black band across the top of the frame.
# Measure it rather than hard-coding a crop, so a different renderer or
# resolution does not silently cut into the scene.
probe = np.asarray(fr_plan[0])
dark = np.where(probe.reshape(len(probe), -1).max(1) > 20)[0]
top = int(dark[0]) if len(dark) else 0
if top:
    fr_plan = [f[top:] for f in fr_plan]
    fr_rand = [f[top:] for f in fr_rand]
    print("  cropped %d rows of black border from each frame" % top)
render_env.close(); S6.close()

fig = plt.figure(figsize=(8.6, 5.0))
axL = fig.add_axes([0.02, 0.29, 0.44, 0.58])
axR = fig.add_axes([0.50, 0.29, 0.44, 0.58])
axd = fig.add_axes([0.10, 0.08, 0.84, 0.16])
for ax, ttl, col in ((axL, "random actions", INK), (axR, "planning with the learned model", BLUE)):
    ax.axis("off"); ax.set_title(ttl, fontsize=10.5, color=col, pad=3)
imL = axL.imshow(fr_rand[0]); imR = axR.imshow(fr_plan[0])
axd.set_xlim(0, len(d_plan) - 1); axd.set_ylim(0, max(d_rand.max(), d_plan.max()) * 1.1)
axd.set_ylabel("tip to target", fontsize=8.5); axd.set_xlabel("step", fontsize=8.5)
axd.grid(alpha=.2, lw=.6); axd.tick_params(labelsize=7.5)
(lr,) = axd.plot([], [], lw=1.8, color=INK)
(lp,) = axd.plot([], [], lw=2.2, color=BLUE)
cap = fig.text(.5, .965, "", ha="center", fontsize=10, color=INK)


def frame6(k):
    imL.set_data(fr_rand[k]); imR.set_data(fr_plan[k])
    lr.set_data(np.arange(k + 1), d_rand[:k + 1])
    lp.set_data(np.arange(k + 1), d_plan[:k + 1])
    cap.set_text("step %d      random %.3f      planned %.3f" % (k, d_rand[k], d_plan[k]))
    return imL, imR, lr, lp, cap


FuncAnimation(fig, frame6, frames=len(fr_plan), blit=False).save(
    FIG / "arm.gif", writer=PillowWriter(fps=12), dpi=80)
plt.close(fig)
print("  wrote figures/arm.gif")
