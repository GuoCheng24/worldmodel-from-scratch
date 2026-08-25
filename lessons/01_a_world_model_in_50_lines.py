"""Lesson 1 - A world model, and the one design choice that matters.

    python lessons/01_a_world_model_in_50_lines.py      (~10 s on one GPU, ~30 s on CPU)

A world model predicts where you end up given where you are and what you do.
Train it on one-step transitions and it gets very accurate very quickly. This
lesson builds one, then makes two measurements that decide how the rest of the
series goes:

  1. Predicting the state CHANGE beats predicting the next state outright.
     Both are one line apart in code, and one is measurably better.

  2. A model that is excellent at one step can still be useless over twenty.
     One-step loss is the number everyone reports and it does not tell you
     what you want to know.
"""
import numpy as np, torch
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import wm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
rng = np.random.default_rng(0); torch.manual_seed(0)
sys_ = wm.Pendulum()

print("Training data: 4000 pendulum trajectories, 30 steps each.")
S, A, Y = wm.make_dataset(sys_, 4000, 30, rng)

# ── Measurement 1: predict the change, or predict the next state? ──────────
print("\n[1] Does predicting the state CHANGE actually help?")
results = {}
for residual in (True, False):
    torch.manual_seed(0)
    m = wm.WorldModel(sys_.state_dim, sys_.action_dim, residual=residual)
    loss = wm.fit(m, S, A, Y, steps=4000, device=DEV)
    results[residual] = (m, loss)
    print("    %-22s one-step MSE = %.3e" % ("predict change" if residual else "predict next state", loss))
r, d = results[True][1], results[False][1]
print("    -> predicting the change is %.1fx more accurate (MSE)" % (d / r))
print("       Modest, but free, and it is the right default: with dt small,")
print("       s_{t+1} is nearly s_t, so a direct model is partly rewarded for")
print("       copying its input rather than learning the dynamics.")

model = results[True][0]

# ── Measurement 2: does one-step accuracy survive a rollout? ───────────────
print("\n[2] The one-step number is excellent. What happens over 20 steps?")
s0 = sys_.sample_states(1000, rng)
Acts = sys_.sample_actions(1000, 20, rng)
true = wm.rollout(sys_, s0, 20, Acts)
pred = wm.imagine(model, s0, 20, Acts, device=DEV)
e = wm.rollout_error(pred, true).mean(0)

span = np.linalg.norm(true, axis=-1).mean()
print("    step  1  error %.4f   (%.2f%% of typical state size)" % (e[0], 100 * e[0] / span))
print("    step  5  error %.4f   (%.2f%%)   %.0fx the one-step error" % (e[4], 100 * e[4] / span, e[4] / e[0]))
print("    step 20  error %.4f   (%.2f%%)   %.0fx the one-step error" % (e[19], 100 * e[19] / span, e[19] / e[0]))
print("\n    The one-step MSE said %.1e. Nothing about that number told you the" % r)
print("    rollout would be %.0fx worse by step 20. Lesson 2 is about why," % (e[19] / e[0]))
print("    and about what you can measure instead.")

np.savez(pathlib.Path(__file__).parent.parent / "figures" / "lesson01.npz",
         e=e, one_step_residual=r, one_step_direct=d, span=span)
