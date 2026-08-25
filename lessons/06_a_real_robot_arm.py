"""Lesson 6 - A real robot arm, and where the earlier lessons land on one.

    MUJOCO_GL=egl python lessons/06_a_real_robot_arm.py      (~3 min on one GPU)

Lesson 3 planned a swing-up on a pendulum written by hand and concluded that
the planning horizon is set by the task rather than by model quality. This
lesson runs the same code on a MuJoCo arm, where the answer comes out
different - which is what makes it a test rather than a demonstration.

It also measures the one thing that separates these robots from each other,
and it is not what the earlier lessons would have predicted.
"""
import numpy as np, torch, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import wm

wm.ensure_headless_gl()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"

# ═══ 1. Transient gain against sustained rate ══════════════════════════════
print("[1] Why the compounding bound is catastrophic on some robots and merely")
print("    wrong on others.")
print("    Perturb a state by 1e-6 in a random direction, take ONE step, and")
print("    measure the gain. Compare it to exp(lambda*dt), the sustained rate.")
print("      %-18s %-16s %-18s %s" % ("environment", "one-step gain", "exp(lambda*dt)", "ratio"))
rng = np.random.default_rng(0)
gains = []
for env_id in ("Reacher-v5", "Pusher-v5", "Walker2d-v5", "HalfCheetah-v5"):
    S = wm.GymSystem(env_id, seed=0)
    lam, _ = wm.lyapunov(S, n=60, steps=300, rng=rng, eps=1e-7)
    s = S.sample_states(300, rng); a = S.sample_actions(300, 1, rng)[:, 0]
    eps = 1e-6
    u = rng.standard_normal(s.shape); u /= np.linalg.norm(u, axis=-1, keepdims=True)
    g = float(np.median(np.linalg.norm(S.step(s + eps * u, a) - S.step(s, a), axis=-1) / eps))
    asym = float(np.exp(lam * S.dt))
    gains.append((env_id, g, asym, g / asym))
    print("      %-18s %-16.2f %-18.3f %.1f" % (env_id, g, asym, g / asym))
    S.close()
arms = [x for x in gains if x[0].split("-")[0] in ("Reacher", "Pusher")]
legs = [x for x in gains if x[0].split("-")[0] in ("Walker2d", "HalfCheetah")]
print("    The arms sit at a ratio of about %.1f: one step amplifies a random error"
      % np.mean([x[3] for x in arms]))
print("    by exactly the rate the dynamics sustain. The legged robots sit at about")
print("    %.0f - a single step amplifies %.0fx while the sustained rate is under"
      % (np.mean([x[3] for x in legs]), np.mean([x[1] for x in legs])))
print("    %.2f per step." % np.mean([x[2] for x in legs]))
print("    A random perturbation first picks up the largest singular value of the")
print("    Jacobian; only after it rotates into the growing direction does it")
print("    settle to lambda. The textbook bound compounds the FIRST number as if")
print("    it were the second, which is why Lesson 5 saw it write rollouts off")
print("    within three steps. On an arm the two agree and the bound is merely")
print("    loose; on a leg they differ by %.0fx per step and it is worthless."
      % np.mean([x[3] for x in legs]))

# ═══ 2. Can a learned world model actually drive the arm? ══════════════════
print("\n[2] Reacher: a two-link arm, and a world model learned from random play.")
torch.manual_seed(0); rng = np.random.default_rng(0)
S = wm.GymSystem("Reacher-v5", seed=0)
s0 = S.sample_states(3000, rng); A = S.sample_actions(3000, 30, rng)
T = wm.rollout(S, s0, 30, A)
X = np.concatenate([s0[:, None], T[:, :-1]], 1).reshape(-1, S.state_dim)
m = wm.WorldModel(S.state_dim, S.action_dim, hidden=384, depth=3)
mse = wm.fit(m, X, A.reshape(-1, S.action_dim), T.reshape(-1, S.state_dim),
             steps=12000, batch=1024, device=DEV)
m.to(DEV).eval()
print("    state = qpos(%d) + qvel(%d) = %d, action = %d, one-step MSE %.2e"
      % (S.nq, S.nv, S.state_dim, S.action_dim, mse))

# Reacher's qpos is [joint0, joint1, target_x, target_y]; link lengths from the
# model XML. Forward kinematics in torch so the planner can score its own plans.
L1, L2 = 0.1, 0.11


def fingertip(s):
    t0, t1 = s[..., 0], s[..., 0] + s[..., 1]
    return torch.stack([L1 * torch.cos(t0) + L2 * torch.cos(t1),
                        L1 * torch.sin(t0) + L2 * torch.sin(t1)], -1)


def reward(s, u):
    return -((fingertip(s) - s[..., 2:4]).norm(dim=-1) + 0.01 * (u ** 2).sum(-1))


def env_step(s, a):
    return torch.tensor(S.step(s.cpu().numpy(), a.cpu().numpy()), dtype=torch.float32, device=DEV)


N_EP, EP = 16, 50
s_init = torch.tensor(S.sample_states(N_EP, rng), dtype=torch.float32, device=DEV)
rand, _, _ = wm.cem_mpc(lambda s, a: s, env_step, reward, s_init, 1, EP, u_max=1.0,
                        n_candidates=8, iters=1, n_elite=2, action_dim=S.action_dim)
print("      %-22s %-14s %s" % ("planner", "reward/step", "final fingertip-to-target"))
print("      %-22s %-14.4f %s" % ("random actions", rand.mean().item(), "-"))
HS = (3, 5, 10, 20, 40)
res = []
for H in HS:
    r, sf, _ = wm.cem_mpc(lambda s, a: m(s, a), env_step, reward, s_init, H, EP,
                          u_max=1.0, n_candidates=128, iters=3, n_elite=16,
                          action_dim=S.action_dim)
    d = float((fingertip(sf) - sf[..., 2:4]).norm(dim=-1).median())
    res.append((H, r.mean().item(), d))
    print("      %-22s %-14.4f %.4f" % ("world model, H=%d" % H, r.mean().item(), d))
best = max(res, key=lambda x: x[1])
print("    The model drives the arm to within %.4f of the target, from a random-play"
      % min(x[2] for x in res))
print("    dataset and no reward signal during training.")

# ═══ 3. The horizon claim, tested where the answer differs ═════════════════
print("\n[3] Lesson 3 said the planning horizon is set by the task. Here it is %d,"
      % best[0])
print("    where the pendulum swing-up needed 25. Both are the task's answer:")
print("    a swing-up has to pump energy over many steps before anything good")
print("    happens, and reaching does not - greedy descent toward the target is")
print("    already the right move, so a longer horizon only buys more model error.")
worst = min(res, key=lambda x: x[1])
if worst[0] > best[0]:
    print("    That cost is visible: reward falls from %.4f at H=%d to %.4f at H=%d."
          % (best[1], best[0], worst[1], worst[0]))
    print("    Lesson 3's pendulum never showed this because its model stayed")
    print("    accurate over the whole sweep. Same claim, opposite-looking curve.")
else:
    print("    On this run longer horizons did not cost anything measurable.")

# ═══ 4. What the planner needed from the model ═════════════════════════════
print("\n[4] And the quantity Lesson 3 identified, measured on the real arm.")
probe_np = S.sample_states(64, rng)
probe = torch.tensor(probe_np, dtype=torch.float32, device=DEV)
g = torch.Generator(device=DEV)
print("      %-6s %-16s %-14s %s" % ("H", "state error", "rank rho", "regret"))
rank_rows = []
for H in (5, 20, 40):
    # The SAME action sequence must drive both rollouts. Sampling twice - which
    # is what this line did at first - compares a model rollout against a true
    # rollout of a different experiment, and the "state error" it reports is a
    # measure of the action noise rather than of the model. It came out at 7 to
    # 15 on a system whose whole state has norm ~14, next to a rank correlation
    # of 0.998, which is the kind of pair that should stop you.
    Ah = S.sample_actions(64, H, rng)
    err = float(np.median(wm.rollout_error(
        wm.imagine(m, probe_np, H, Ah, device=DEV),
        wm.rollout(S, probe_np, H, Ah))[:, -1]))
    g.manual_seed(0)
    rho, reg = wm.rank_fidelity(lambda s, a: m(s, a), env_step, reward, probe, H,
                                u_max=1.0, std=0.8, generator=g, action_dim=S.action_dim)
    rank_rows.append((H, err, rho, reg))
    print("      %-6d %-16.4f %-14.3f %.4f" % (H, err, rho, reg))
print("    Same pattern as the pendulum: the ranking holds up while the states")
print("    drift, and the planner only needs the ranking.")

S.close()
FIG.mkdir(exist_ok=True)
np.savez(FIG / "lesson06.npz",
         gain_envs=np.array([x[0] for x in gains]),
         gain_one=np.array([x[1] for x in gains]),
         gain_asym=np.array([x[2] for x in gains]),
         HS=np.array([x[0] for x in res]), rew=np.array([x[1] for x in res]),
         dist=np.array([x[2] for x in res]), rand=float(rand.mean()),
         rank=np.array(rank_rows))
print("\nSaved figures/lesson06.npz")
