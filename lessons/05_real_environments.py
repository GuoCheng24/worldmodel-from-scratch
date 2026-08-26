"""Lesson 5 - Real environments, and which of Lessons 1-4 survive them.

    MUJOCO_GL=egl python lessons/05_real_environments.py     (~70 s on one GPU)

Everything so far ran on two systems written by hand in this repository,
chosen because their true dynamics are known exactly. That is what made "how
wrong is the model" answerable, and it is also the obvious objection: none of
it has touched a real simulator.

This lesson repeats the measurements on three MuJoCo environments. Some
findings reproduce, one reproduces only sometimes, and one turns out to have
been a statement about the regime rather than about world models. Saying which
is which is the point.

INSTALLING. Lessons 1-4 need nothing but torch, numpy and matplotlib. This one
needs two more:

    pip install gymnasium mujoco

and, on any machine without a display, this before either is imported:

    export MUJOCO_GL=egl

That single variable is the difference between working and not. Measured on the
headless machine this was written on:

    unset            mujoco.FatalError: an OpenGL platform library has not been loaded
    MUJOCO_GL=glfw   the same, after a GLFW X11 warning
    MUJOCO_GL=osmesa AttributeError inside PyOpenGL
    MUJOCO_GL=egl    works

and one wrong combination aborts the process outright rather than raising, so
wrapping the call in try/except does not save you. The good news is that none of
the measurements below render anything: they need states, not pictures. If you
are measuring rather than making videos, you can skip the graphics stack.
"""
import numpy as np, torch, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import wm

backend = wm.ensure_headless_gl()   # must happen before mujoco is imported
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"

# Say out loud what graphics configuration this run is using, so the claim in
# the docstring is something the run confirms rather than something the reader
# has to take on trust. The backend table itself is a recorded measurement -
# re-running it live is not possible, because one of the four values aborts the
# process instead of raising, and a lesson that kills its own interpreter to
# prove a point is not a lesson.
import os
print("Running with MUJOCO_GL=%s (DISPLAY=%r)." % (backend, os.environ.get("DISPLAY")))

ENVS = [("InvertedPendulum-v5", 1200, 60, 8000, 256, 2),
        ("Reacher-v5",          1200, 60, 8000, 256, 2),
        ("HalfCheetah-v5",      4000, 40, 12000, 512, 3)]

print("None of the measurements below render anything, so the graphics backend")
print("only has to be valid, not fast. Confirming that now: importing MuJoCo,")
print("building an environment and stepping it, with no renderer created.")
_probe = wm.GymSystem("Reacher-v5", seed=0)
_s = _probe.sample_states(2, np.random.default_rng(0))
_ = _probe.step(_s, _probe.sample_actions(2, 1, np.random.default_rng(0))[:, 0])
_probe.close()
print("    state-only path works, no renderer touched.\n")

print("Repeating the Lesson 1-4 measurements on three MuJoCo environments.")
print("State is (qpos, qvel) - the full simulator state, not the observation,")
print("which for several of these omits coordinates and would quietly change")
print("the question being asked.\n")

rows = []
for env_id, n_traj, K, steps, hidden, depth in ENVS:
    rng = np.random.default_rng(0); torch.manual_seed(0)
    S = wm.GymSystem(env_id, seed=0)
    lam, lam_sd = wm.lyapunov(S, n=80, steps=400, rng=rng, eps=1e-7)
    s0 = S.sample_states(n_traj, rng); A = S.sample_actions(n_traj, 30, rng)
    T = wm.rollout(S, s0, 30, A)
    X = np.concatenate([s0[:, None], T[:, :-1]], 1).reshape(-1, S.state_dim)
    m = wm.WorldModel(S.state_dim, S.action_dim, hidden=hidden, depth=depth)
    mse = wm.fit(m, X, A.reshape(-1, S.action_dim), T.reshape(-1, S.state_dim),
                 steps=steps, batch=1024, device=DEV)
    se = S.sample_states(300, rng); Ae = S.sample_actions(300, K, rng)
    TT = wm.rollout(S, se, K, Ae)
    E = wm.rollout_error(wm.imagine(m, se, K, Ae, device=DEV), TT)
    scale = float(np.linalg.norm(TT, axis=-1).mean())
    med, mean = wm.summarise(E, "median"), wm.summarise(E, "mean")
    gm = wm.fit_growth(med, dt=S.dt, hi=scale * .3)
    ga = wm.fit_growth(mean, dt=S.dt, hi=scale * .3)
    Lmax, _ = wm.lipschitz(S, se[:60], Ae[:60, 0])
    bound = wm.textbook_bound(Lmax, med[0], np.arange(1, K + 1))
    k_bound = int(np.argmax(bound > scale)) + 1 if (bound > scale).any() else K
    k_real = int(np.argmax(med > scale)) + 1 if (med > scale).any() else K
    tol = float(np.percentile(E[:, -1], 10))
    h = wm.usable_horizon(E, tol); p5, p95 = np.percentile(h, [5, 95])
    rows.append(dict(env=env_id, dim=S.state_dim, dt=S.dt, lam=lam, lam_sd=lam_sd,
                     mse=mse, scale=scale, med_fit=gm, mean_fit=ga, Lmax=Lmax,
                     k_bound=k_bound, k_real=k_real, p5=p5, p95=p95, K=K,
                     med=med, mean=mean, bound=bound))
    print("    %-22s state %2d   lambda %+.3f +- %.3f   one-step MSE %.1e"
          % (env_id, S.state_dim, lam, lam_sd, mse))
    S.close()

# ═══ 1. The textbook bound, on real dynamics ═══════════════════════════════
print("\n[1] The compounding-error bound, restated so the number means something.")
print("    Quoting 'overestimates by 1e56 at step 60' is true and useless - L^k")
print("    on real contact dynamics is astronomical and reads as a straw man.")
print("    The useful form: at what step does each claim the rollout is worthless?")
print("      %-22s %-10s %-24s %s" % ("environment", "L_max", "bound says worthless at", "actually is at"))
for r in rows:
    print("      %-22s %-10.1f %-24s %s"
          % (r["env"], r["Lmax"], "step %d" % r["k_bound"],
             "step %d" % r["k_real"] if r["k_real"] < r["K"] else "still fine at %d" % r["K"]))
kb_max = max(r["k_bound"] for r in rows)
print("    Every one of them writes the rollout off within %d steps of the start," % kb_max)
print("    on systems that stay usable for %d to %d. The bound is not conservative"
      % (min(r["k_real"] for r in rows), max(r["k_real"] for r in rows)))
print("    here, it is uninformative.")
# "The exact step moves by one between runs" is what this used to say, and it
# was measured wrong. Over four runs Reacher's failure step came out 18, 21, 22
# and 26 - eight steps apart, not one - and L_max, being a maximum over sampled
# states, swung from 1403 to 10262 on HalfCheetah. Neither of those changes the
# conclusion, which is why the conclusion is what gets stated as stable; but a
# repository about not overstating what a run supports does not get to round
# its own spread down to "one".
print("    Read the L_max column as one draw, not a constant: it is a maximum over")
print("    sampled states, and across four runs it moved by 7x on HalfCheetah while")
print("    Reacher's failure step ranged 18 to 26. What does not move between runs")
print("    is that the bound is single digits against tens.")

# ═══ 2. Which shape does the error curve take? ═════════════════════════════
print("\n[2] Lesson 2 could not tell the shape of its pendulum curve apart at all,")
print("    and found exponential growth at rate lambda on a chaotic one. Here:")
print("      %-22s %-8s %-13s %-13s %s" % ("environment", "lambda", "median curve", "mean curve", "same?"))
diff = 0
for r in rows:
    mv, av = r["med_fit"]["verdict"], r["mean_fit"]["verdict"]
    same = mv == av
    diff += not same
    print("      %-22s %-8.2f %-13s %-13s %s"
          % (r["env"], r["lam"], "%s(%.1f)" % (mv[:4], r["med_fit"]["resid_ratio"]),
             "%s(%.1f)" % (av[:4], r["mean_fit"]["resid_ratio"]), "yes" if same else "NO"))
print("    (the number in brackets is how many times larger the losing residual is;")
print("     under 1.5 the curve does not distinguish the two shapes and says so)")
amb = sum(r["med_fit"]["verdict"] == "ambiguous" for r in rows)
print("    %d of %d median curves are ambiguous, and %d of %d give a different form"
      % (amb, len(rows), diff, len(rows)))
print("    depending on which average you take. Whether that contrast appears is not")
print("    stable: Lesson 2 saw it on two seeds out of six, and it varies here too.")
print("    Treat it as something to check on your own system, not as a property of")
print("    world models.")

# ═══ 3. The qualification Lesson 2 needs ═══════════════════════════════════
print("\n[3] And a correction to Lesson 2 that only showed up here.")
expo = [r for r in rows if r["med_fit"]["verdict"] == "exponential"]
print("    Lesson 2's one solid result - the amplification-only curve grows at the")
print("    Lyapunov rate - was measured on a chaotic system with an accurate model")
print("    and four decades of range before saturation. On these environments the")
print("    median curve is exponential in %d of %d." % (len(expo), len(rows)))
for r in rows:
    v = r["med_fit"]
    if v["verdict"] == "exponential":
        print("      %-22s rate %.3f vs lambda %.3f -> ratio %.2f"
              % (r["env"], v["exp_rate"], r["lam"], v["exp_rate"] / r["lam"]))
    else:
        print("      %-22s %s, so its exponential rate is a fit that did not win"
              % (r["env"], v["verdict"]))
        print("      %-22s and comparing it to lambda would read a number out of the" % "")
        print("      %-22s wrong model." % "")
print("    Amplification needs something to amplify. With lambda*dt small or the")
print("    model error large, injection dominates for the whole horizon you care")
print("    about, and lambda predicts nothing. That is the common case on robots,")
print("    and Lesson 2 should be read as a statement about the chaotic regime.")

# ═══ 4. What does reproduce ════════════════════════════════════════════════
print("\n[4] What carried over without qualification: the horizon is not one number.")
print("      %-22s %-10s %-10s %s" % ("environment", "5th pct", "95th pct", "spread"))
for r in rows:
    print("      %-22s %-10.0f %-10.0f %.1fx" % (r["env"], r["p5"], r["p95"], r["p95"] / max(r["p5"], 1)))
print("    Same model, same task, trajectories differing by %.0fx to %.0fx in how"
      % (min(r["p95"] / max(r["p5"], 1) for r in rows), max(r["p95"] / max(r["p5"], 1) for r in rows)))
print("    many steps they survive. Every environment, every time we have looked.")

FIG.mkdir(exist_ok=True)
np.savez(FIG / "lesson05.npz",
         envs=np.array([r["env"] for r in rows]),
         lam=np.array([r["lam"] for r in rows]),
         k_bound=np.array([r["k_bound"] for r in rows]),
         k_real=np.array([r["k_real"] for r in rows]),
         p5=np.array([r["p5"] for r in rows]), p95=np.array([r["p95"] for r in rows]),
         med_shape=np.array([r["med_fit"]["verdict"] for r in rows]),
         mean_shape=np.array([r["mean_fit"]["verdict"] for r in rows]),
         **{("med_%d" % i): r["med"] for i, r in enumerate(rows)},
         **{("mean_%d" % i): r["mean"] for i, r in enumerate(rows)},
         **{("bound_%d" % i): r["bound"] for i, r in enumerate(rows)},
         dts=np.array([r["dt"] for r in rows]), scales=np.array([r["scale"] for r in rows]))
print("\nSaved figures/lesson05.npz")
