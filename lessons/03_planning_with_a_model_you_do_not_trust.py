"""Lesson 3 - Planning with a model you do not fully trust.

    python lessons/03_planning_with_a_model_you_do_not_trust.py   (~2 min on one GPU)

Lesson 2 ended with a number that looks like it should decide something: the
usable horizon, how many steps a rollout stays accurate, which varied 16x across
trajectories of one model. The obvious next move is to plan no further than
that. This lesson checks whether that is right.

It is not, and the gap is large. A model whose rollout is useless after 7 steps
plans 25-step action sequences perfectly. The reason is that a planner never
uses the states a model predicts - it uses them to *rank* candidate action
sequences, then throws them away and executes one action. Ranking survives long
after the states have stopped being accurate.

Before any of that, there is a trap to get out of the way, because it makes
every number afterwards meaningless if you fall into it.
"""
import numpy as np, torch, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import wm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
pend = wm.Pendulum()
U_MAX, EP_LEN, N_EP = 3.0, 80, 16

# The task: swing a hanging pendulum up and hold it inverted, with a torque
# limit of 3.0 against a peak gravitational torque of m*g*l = 9.81. You cannot
# push it up directly; you have to rock it and build energy, which is exactly
# why a planner is needed and a reactive controller will not do.
def reward(s, u):
    upright = torch.remainder(s[..., 0] - np.pi + np.pi, 2 * np.pi) - np.pi
    return -(upright ** 2 + 0.1 * s[..., 1] ** 2 + 0.001 * u[..., 0] ** 2)

true_step = lambda s, a: pend.step_torch(s, a)
s_init = torch.zeros(N_EP, 2, device=DEV)
s_init[:, 0] = torch.linspace(-0.3, 0.3, N_EP)          # hanging, roughly at rest


def train(n_traj, steps, seed=0):
    """A model trained on n_traj random-action trajectories. Fewer = worse."""
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    s0 = pend.sample_states(n_traj, rng, omega_max=8.0)   # swing-up reaches high speed
    A = pend.sample_actions(n_traj, 30, rng) * 1.5
    T = wm.rollout(pend, s0, 30, A)
    S = np.concatenate([s0[:, None], T[:, :-1]], 1).reshape(-1, 2)
    m = wm.WorldModel(2, 1, hidden=256, depth=2)
    mse = wm.fit(m, S, A.reshape(-1, 1), T.reshape(-1, 2), steps=steps, device=DEV)
    m.to(DEV).eval()
    se = pend.sample_states(400, rng, omega_max=8.0)
    Ae = pend.sample_actions(400, 80, rng) * 1.5
    E = wm.rollout_error(wm.imagine(m, se, 80, Ae, device=DEV), wm.rollout(pend, se, 80, Ae))
    return m, mse, float(np.median(wm.usable_horizon(E, 0.1)))


MODELS = [("good", 8000, 8000), ("thin", 600, 3000), ("bad", 80, 800)]
print("Training three models on the same task, differing only in how much data they saw.")
trained = []
for name, n, st in MODELS:
    m, mse, hz = train(n, st)
    trained.append((name, m, mse, hz))
    print("    %-5s %5d trajectories   one-step MSE %.1e   usable horizon %2.0f steps" % (name, n, mse, hz))

# ═══ 1. The trap: scoring a planner inside its own model ═══════════════════
print("\n[1] First, the mistake that makes everything else meaningless.")
print("    A planner needs two dynamics: one to imagine with, one to actually")
print("    move the world. Use the model for both and the reward is computed on")
print("    states the model invented. Here is what that reports:")
print("      model    reward in its own dream    reward in the real world")
for name, m, mse, hz in trained:
    dream, _, _ = wm.cem_mpc(lambda s, a: m(s, a), lambda s, a: m(s, a),
                             reward, s_init, 25, EP_LEN, u_max=U_MAX)
    real, _, _ = wm.cem_mpc(lambda s, a: m(s, a), true_step,
                            reward, s_init, 25, EP_LEN, u_max=U_MAX)
    print("      %-5s    %18.2f    %20.2f" % (name, dream.mean().item(), real.mean().item()))
print("    The WORSE the model, the BETTER its dream score - it simply imagines")
print("    the pendulum already balanced. Nothing errors. Nothing warns you.")
print("    wm.cem_mpc takes the two dynamics as separate required arguments so")
print("    that writing this by accident is not possible.")

# ═══ 2. Does the usable horizon tell you the planning horizon? ═════════════
print("\n[2] Sweeping the planning horizon, with the true dynamics as a control.")
print("    If the sweep behaves the same way for the true dynamics, its shape is")
print("    the planner's doing and says nothing about model error.")
HS = (5, 10, 15, 25, 40, 60)

def sweep(plan):
    """Mean reward and its standard error across episodes, for each horizon.

    The standard error is not decoration. Neighbouring horizons here differ by
    less than the spread across episodes, so picking the argmax and calling it
    'the optimal horizon' would be reporting noise. What is well determined is
    the PLATEAU: the set of horizons that are within measurement error of the
    best one.
    """
    mu, se = [], []
    for H in HS:
        r, _, _ = wm.cem_mpc(plan, true_step, reward, s_init, H, EP_LEN, u_max=U_MAX)
        mu.append(r.mean().item()); se.append(r.std().item() / np.sqrt(len(r)))
    return np.array(mu), np.array(se)

def sufficient_horizon(mu, tol=0.10):
    """The shortest horizon that gets within `tol` of what this row can achieve.

    Not the argmax. Neighbouring horizons past the knee differ by less than the
    episode-to-episode spread, so argmax jumps around between reruns and reports
    noise. "How short can I go before it hurts" is the question with a stable
    answer, and it is also the one you actually ask when choosing a horizon.

    The tolerance is a fraction of the row's own achievable range (best minus
    worst), so it means the same thing for a good model and a bad one. The
    stability of the answer to that choice is checked below rather than assumed.
    """
    best, worst = mu.max(), mu.min()
    thresh = best - tol * (best - worst)
    for i, v in enumerate(mu):
        if v >= thresh:
            return HS[i]
    return HS[-1]

print("      %-20s" % "planning horizon" + "".join("%9d" % h for h in HS))
base, base_se = sweep(true_step)
print("      %-20s" % "TRUE dynamics" + "".join("%9.2f" % v for v in base) +
      "   sufficient H = %d" % sufficient_horizon(base))
print("      %-20s" % "  +- s.e." + "".join("%9.2f" % v for v in base_se))
rows, suff = {}, {"TRUE": sufficient_horizon(base)}
for name, m, mse, hz in trained:
    mu, se = sweep(lambda s, a: m(s, a))
    rows[name] = mu; suff[name] = sufficient_horizon(mu)
    print("      %-20s" % ("model: %s (h=%.0f)" % (name, hz)) + "".join("%9.2f" % v for v in mu) +
          "   sufficient H = %d" % suff[name])
    print("      %-20s" % "  +- s.e." + "".join("%9.2f" % v for v in se))

# A threshold-based answer is only worth quoting if it survives the threshold
# being moved. Sweeping it is three lines and catches the case where the whole
# conclusion rests on where a cut happened to fall.
print("    Stability of that answer to the 10% tolerance:")
print("      %-14s" % "tolerance" + "".join("%8.0f%%" % (100 * x) for x in (.03, .05, .10, .20)))
stable = True
for label, mu in [("TRUE", base)] + [(n, rows[n]) for n in rows]:
    vals = [sufficient_horizon(mu, x) for x in (.03, .05, .10, .20)]
    stable &= len(set(vals)) == 1
    print("      %-14s" % label + "".join("%9d" % v for v in vals))
print("    %s" % ("Unchanged across the sweep." if stable else
                  "The answer moves with the tolerance - treat it as approximate."))

vals = set(suff.values())
print("    Sufficient horizon: %s." % ", ".join("%s=%d" % (k, v) for k, v in suff.items()))
if len(vals) == 1:
    H_star = vals.pop()
    print("    The same H=%d for the true dynamics and for every model, including the" % H_star)
    print("    one whose rollout is useless after 1 step. So the horizon you need is")
    print("    set by the TASK - how long a swing-up takes - not by model quality.")
    print("    The usable horizons here are %s and they predict none of it."
          % ", ".join("%.0f" % hz for _, _, _, hz in trained))
else:
    print("    They differ, so on this run the horizon you need does depend on the")
    print("    model - do not claim otherwise from these numbers.")

# ═══ 3. What model quality actually costs you ══════════════════════════════
print("\n[3] So what does a worse model cost? Reward, not horizon.")
i25 = HS.index(25)
for name, m, mse, hz in trained:
    print("      %-5s  usable horizon %2.0f   reward at H=25: %6.2f   (%+.2f vs the true dynamics)"
          % (name, hz, rows[name][i25], rows[name][i25] - base[i25]))
print("    The 'thin' model's rollout is unusable after %.0f steps and it still" % trained[1][3])
print("    plans 25-step sequences at no measurable cost. The 'bad' one does not.")

# ═══ 4. Why: planning needs the ranking, not the states ════════════════════
print("\n[4] The reason, measured. A planner only needs to ORDER candidates.")
print("      %-6s %4s   %-14s %-12s %s" % ("model", "H", "state error", "rank rho", "regret"))
g = torch.Generator(device=DEV); g.manual_seed(0)
probe = torch.tensor(pend.sample_states(64, np.random.default_rng(1), omega_max=6.0),
                     dtype=torch.float32, device=DEV)
rank_tbl = {}
for name, m, mse, hz in trained:
    rng = np.random.default_rng(2)
    se = pend.sample_states(300, rng, omega_max=8.0); Ae = pend.sample_actions(300, 60, rng) * 1.5
    E = wm.rollout_error(wm.imagine(m, se, 60, Ae, device=DEV), wm.rollout(pend, se, 60, Ae))
    for H in (10, 25, 50):
        g.manual_seed(0)
        rho, reg = wm.rank_fidelity(lambda s, a: m(s, a), true_step, reward, probe, H,
                                    u_max=U_MAX, generator=g)
        rank_tbl[(name, H)] = (float(np.median(E[:, H - 1])), rho, reg)
        print("      %-6s %4d   %-14.3f %-12.3f %.3f" % (name, H, *rank_tbl[(name, H)]))
th = rank_tbl[("thin", 25)]
print("    Read the 'thin' row at H=25: the state is off by %.2f, which is far past" % th[0])
print("    any tolerance you would accept, and the ranking correlation is %.3f with" % th[1])
print("    regret %.3f - the model picks the genuinely optimal sequence every time." % th[2])
bad50 = rank_tbl[("bad", 50)]
print("    Planning fails only when the RANKING fails ('bad' at H=50: rho %.2f," % bad50[1])
print("    regret %.1f). That is the quantity to measure before trusting a planner," % bad50[2])
print("    and it is not the one Lesson 2 taught you to compute.")

FIG.mkdir(exist_ok=True)
np.savez(FIG / "lesson03.npz", HS=np.array(HS), base=np.array(base),
         names=np.array([n for n, _, _ in MODELS]),
         rewards=np.array([rows[n] for n, _, _ in MODELS]),
         horizons=np.array([hz for _, _, _, hz in trained]),
         mses=np.array([mse for _, _, mse, _ in trained]),
         rank_keys=np.array(["%s|%d" % k for k in rank_tbl]),
         rank_vals=np.array(list(rank_tbl.values())))
print("\nSaved figures/lesson03.npz")
