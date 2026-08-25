"""The `wm.diagnose()` report shown in the README, and the run that produces it.

    python examples/diagnose_lorenz.py        (~1 min on one GPU)

This is Lesson 2's Lorenz model, handed to the one-call diagnosis instead of to
the step-by-step analysis. The point of the example is the last block of the
report: the fitted growth rate lands within a couple of percent of the measured
Lyapunov exponent and the interval still does not cover it, and the report says
so rather than rounding the disagreement away. Lesson 2 explains why - a model
rollout injects a fresh error at every step on top of amplifying the old one,
so it lands NEAR lambda, not on it.
"""
import numpy as np, torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import wm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0); torch.manual_seed(0)
lorenz = wm.Lorenz()

lam, sd = wm.lyapunov(lorenz, n=300, steps=4000, rng=rng)   # measured, not assumed
print("Lorenz lambda = %+.4f +- %.4f (literature: ~0.906)" % (lam, sd))

S, A, Y = wm.make_dataset(lorenz, 6000, 25, rng)
model = wm.WorldModel(lorenz.state_dim, lorenz.action_dim, hidden=512, depth=3)
wm.fit(model, S, A, Y, steps=20000, batch=2048, lr=2e-3, device=DEV)

s0 = lorenz.sample_states(600, rng)
true = wm.rollout(lorenz, s0, 900)
pred = wm.imagine(model, s0, 900, device=DEV)

report = wm.diagnose(pred, true, dt=lorenz.dt, lam=(lam, sd))
print()
print(report)
# The report is also a dict, so a script can act on it rather than read it.
print("\n  as data: median usable horizon = %.0f steps, shape verdict = %s"
      % (report["horizon"]["p50"], report["shape"]["verdict"]))
