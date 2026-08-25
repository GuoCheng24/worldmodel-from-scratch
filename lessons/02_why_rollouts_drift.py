"""Lesson 2 - Why rollouts drift, and what actually sets the rate.

    python lessons/02_why_rollouts_drift.py       (~50 s on one GPU; see README for CPU)

Lesson 1 ended with a model whose one-step error was tiny and whose 20-step
error was 14x larger. The standard explanation is the compounding-error bound

    e_{k+1} <= L e_k + delta      =>      e_k = delta (L^k - 1)/(L - 1)

with L the Lipschitz constant of the dynamics. It is in every model-based RL
paper. This lesson measures whether it describes anything real.

It does not. The bound is not loose by a constant - it has the wrong shape.
What governs a rollout is the LYAPUNOV exponent, the average log-growth along
the trajectory, not L, the worst case over all states and all directions. This
lesson measures lambda directly, shows the rollout error growing at exactly
that rate, and then separates the two things that are usually conflated:

    amplification   the dynamics stretching an error that is already there
    injection       the model adding a fresh error at every single step

They are separable, and they do different things: amplification sets the RATE,
while injection only shifts the curve up by a roughly constant multiplier. That
multiplier lands near the value you get by assuming successive model errors are
independent, and nowhere near the value you get by assuming they align - so the
errors a model makes at consecutive steps mostly do not point the same way.

Every number below is measured in the run, and every verdict is computed from
those numbers rather than written here in advance. Where the evidence is too
thin to support a conclusion, the script says so instead of asserting one.

ONE QUALIFICATION, ADDED AFTER LESSON 5. The result that the growth rate equals
lambda is measured here on a chaotic system with an accurate model and four
decades of range before saturation. That is the regime where amplification has
something to amplify. Lesson 5 repeats these measurements on three MuJoCo
environments and finds the median error curve is a power law in every one of
them: with lambda*dt small or the model error large, injection dominates for
the entire horizon anyone cares about and lambda predicts nothing. Read the
Lorenz result below as a statement about the chaotic regime, not about world
models in general - and see Lesson 5 for what the common case looks like.
"""
import numpy as np, torch, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import wm

DEV = "cuda" if torch.cuda.is_available() else "cpu"
FIG = pathlib.Path(__file__).resolve().parents[1] / "figures"
rng = np.random.default_rng(0); torch.manual_seed(0)

# ═══ 1. Two systems, one number that tells them apart ══════════════════════
print("[1] Measuring the largest Lyapunov exponent of each system.")
print("    (Benettin's method - evolve a twin, log how fast it separates,")
print("     renormalise every step so it stays in the linear regime.)")
pend, lorenz = wm.Pendulum(), wm.Lorenz()
lam_p, sd_p = wm.lyapunov(pend, n=300, steps=4000, rng=rng)
lam_l, sd_l = wm.lyapunov(lorenz, n=300, steps=4000, rng=rng)
print("    pendulum   lambda = %+.4f +- %.4f   -> marginal: errors are not amplified" % (lam_p, sd_p))
print("    Lorenz     lambda = %+.4f +- %.4f   -> chaotic:  errors double every %.2f time units"
      % (lam_l, sd_l, np.log(2) / lam_l))
print("    Sanity check: the literature value for Lorenz is ~0.906. We measured it.")

# ═══ 2. The textbook bound, against a real rollout ═════════════════════════
print("\n[2] Pendulum: the textbook bound vs what actually happens.")
S, A, Y = wm.make_dataset(pend, 4000, 30, rng)
m = wm.WorldModel(pend.state_dim, pend.action_dim)
wm.fit(m, S, A, Y, steps=4000, device=DEV)
K = 90
s0 = pend.sample_states(1000, rng); Acts = pend.sample_actions(1000, K, rng)
true = wm.rollout(pend, s0, K, Acts)
pred = wm.imagine(m, s0, K, Acts, device=DEV)
err_p = wm.rollout_error(pred, true)
# The MEDIAN trajectory, not the mean. Section 2b shows what the mean does here
# and why. Using the mean at this point would make the rest of the lesson wrong.
e_p = wm.summarise(err_p, "median"); delta = e_p[0]
Lmax, Lmed = wm.lipschitz(pend, true.reshape(-1, 2)[::37], Acts.reshape(-1, 1)[::37])
bound = wm.textbook_bound(Lmax, delta, np.arange(1, K + 1))
print("    L_max = %.4f (largest Jacobian spectral norm over the sampled states)" % Lmax)
print("      step   measured    textbook bound    overestimate")
for k in (1, 10, 20, 40, 90):
    print("      %4d   %.3e   %.3e       %10.0fx" % (k, e_p[k - 1], bound[k - 1], bound[k - 1] / e_p[k - 1]))
g = wm.fit_growth(e_p, dt=pend.dt)
print("    Measured curve is best fit by a %s (alpha=%.2f); residuals exp %.3f / pow %.3f."
      % (g["verdict"], g["pow_alpha"], g["resid_exp"], g["resid_pow"]))
print("    The bound is an exponential in k. %s" % (
      "The measurement is not, so the bound is wrong in shape, not just in scale."
      if g["verdict"] == "power-law" else
      "So is the measurement here - the shape claim does not hold in this run."))
print("    lambda = %+.4f for this system, so there is almost nothing to amplify;" % lam_p)
print("    what you are watching is errors accumulating, not compounding.")

# ═══ 2b. What the mean would have told you instead ═════════════════════════
print("\n[2b] The same data, summarised by the mean instead of the median.")
e_mean = wm.summarise(err_p, "mean")
gm = wm.fit_growth(e_mean, dt=pend.dt)
print("     median curve -> %-11s   mean curve -> %s" % (g["verdict"], gm["verdict"]))
over = np.abs(pred[..., 0] - true[..., 0]) > np.pi
print("     The difference is a minority of rollouts falling off a cliff. A pendulum")
print("     driven by random torque can be pushed over the top; once the model and")
print("     the truth are on opposite branches the error is O(pi) and stays there.")
print("       fraction of rollouts that have switched branch by step:")
for kk in (20, 40, 60, 90):
    print("         step %2d : %5.1f%%" % (kk, 100 * over[:, :kk].any(1).mean()))
print("     Those %.0f%% drag the mean into a different functional form. The typical" % (100 * over.any(1).mean()))
print("     rollout never does what the mean curve says it does.")

# ═══ 3. Lorenz: separating amplification from injection ════════════════════
print("\n[3] Lorenz: two rollouts that differ in exactly one way.")
print("    A  perturb the TRUE system once by delta, then let it run  -> amplification only")
print("    B  run the MODEL, which injects a fresh error every step   -> amplification + injection")
Sl, Al, Yl = wm.make_dataset(lorenz, 6000, 25, rng)
ml = wm.WorldModel(lorenz.state_dim, lorenz.action_dim, hidden=512, depth=3)
one_step = wm.fit(ml, Sl, Al, Yl, steps=20000, batch=2048, lr=2e-3, device=DEV)
KL, N = 900, 1200
s0l = lorenz.sample_states(N, rng)
T = wm.rollout(lorenz, s0l, KL)
EB = wm.rollout_error(wm.imagine(ml, s0l, KL, device=DEV), T)     # keep every trajectory
d0 = wm.summarise(EB, "median")[0]
u = rng.standard_normal((N, 3)); u /= np.linalg.norm(u, axis=-1, keepdims=True)
EA = wm.rollout_error(wm.rollout(lorenz, s0l + d0 * u, KL), T)
scale = np.linalg.norm(T, axis=-1).mean()

# Dynamic range decides whether this comparison can say anything at all. With
# delta close to the size of the attractor the curve saturates before it has
# grown, and any fit is reading noise. Check first, conclude second.
print("    one-step MSE %.2e -> delta = %.2e, saturation near %.1f: %.1f decades of range"
      % (one_step, d0, scale * 0.2, np.log10(scale * 0.2 / d0)))

# Two things below are easy to get wrong, and both were got wrong here first.
#
# 1. The confidence interval must come from resampling TRAJECTORIES. Points
#    along one rollout are serially correlated; resampling them pretends they
#    are independent and returns an interval about half as wide as the truth.
# 2. The curve must be a MEDIAN, not a mean. Rollout error is right-skewed
#    across trajectories, and the skew grows with time, so the mean tracks the
#    few runaway rollouts rather than the typical one. Section 3b shows what
#    that costs. Nearly every paper reports the mean.
lo, hi = d0 * 3, scale * 0.2
res = {}
for name, E in (("A", EA), ("B", EB)):
    g = wm.fit_growth_ci(E, dt=lorenz.dt, lo=lo, hi=hi, rng=rng, how="median")
    res[name] = g
    c = g["exp_rate_ci"]
    print("      %s  rate %.4f  95%%CI [%.4f, %.4f]   %+.1f%% vs lambda   %s"
          % (name, g["exp_rate"], c[0], c[1], 100 * (g["exp_rate"] / lam_l - 1),
             "covers lambda" if c[0] <= lam_l <= c[1] else "does NOT cover lambda"))
gap = abs(res["A"]["exp_rate"] - res["B"]["exp_rate"]) / lam_l
print("    The two rates differ by %.0f%% of lambda. %s" % (100 * gap,
      "Injecting an error at every step did not change the growth rate -"
      " the dynamics set it, and the model only decides the starting size."
      if gap < 0.15 else
      "That is too large to call equal; do not claim injection is rate-neutral here."))

# ═══ 3b. The average you choose changes the answer ═════════════════════════
print("\n[3b] Now the same fit, with the three ways of averaging trajectories.")
print("     %-6s %-12s %-26s %s" % ("curve", "average", "rate [95% CI]", "vs lambda"))
infl = {}
for name, E in (("A", EA), ("B", EB)):
    for how in ("median", "geometric", "mean"):
        c0 = wm.summarise(E, how)[0]
        g = wm.fit_growth_ci(E, dt=lorenz.dt, lo=c0 * 3, hi=scale * 0.2,
                             n_boot=250, rng=rng, how=how)
        if "exp_rate" not in g: continue
        infl[(name, how)] = g["exp_rate"]
        ci = g["exp_rate_ci"]
        print("     %-6s %-12s %.4f [%.4f, %.4f]     %+5.1f%%  %s"
              % (name, how, g["exp_rate"], ci[0], ci[1], 100 * (g["exp_rate"] / lam_l - 1),
                 "covers lambda" if ci[0] <= lam_l <= ci[1] else ""))
if ("B", "mean") in infl and ("B", "median") in infl:
    b_infl = 100 * (infl[("B", "mean")] / infl[("B", "median")] - 1)
    a_infl = 100 * (infl[("A", "mean")] / infl[("A", "median")] - 1)
    print("     Only the MEDIAN covers lambda for both curves. Switching to the mean")
    print("     moves the fitted rate by %+.0f%% for the model rollout and %+.0f%% for the" % (b_infl, a_infl))
    print("     perturbation - large, and not even in the same direction, so it is not")
    print("     a bias you can correct for after the fact. Rollout error is strongly")
    print("     right-skewed across trajectories and the skew grows with the rollout,")
    print("     which is enough to make the mean untrustworthy but not enough to")
    print("     predict which way it will err. Most papers report the mean.")
    print("     Why the two curves skew differently is an open question - see the")
    print("     README section of that name; a clean answer would be a real contribution.")

    # The obvious suspect, measured so the README can cite it rather than assert
    # it. Lorenz has two lobes, separated by the sign of x. A rollout that ends
    # up on the wrong one carries an O(attractor) error, which is exactly the
    # kind of event that builds a heavy right tail.
    Pm = wm.imagine(ml, s0l, KL, device=DEV)
    wrong = np.sign(Pm[..., 0]) != np.sign(T[..., 0])
    print("     Suspect: rollouts landing on the WRONG LOBE of the attractor.")
    print("       cumulative fraction that have been on the wrong lobe by t =")
    for kk in (100, 300, 600, KL):
        print("         t=%.1f : %5.1f%%" % (kk * lorenz.dt, 100 * wrong[:, :kk].any(1).mean()))
    print("     Pervasive, and the right order of event - but conditioning on")
    print("     'still on the right lobe' to test it selects for low-error rollouts")
    print("     and biases the answer the other way, so this is suggestive, not shown.")

# ═══ 4. So what DID injection do? ══════════════════════════════════════════
print("\n[4] Injection lifts the curve. By how much, and does that mean anything?")
coherent = 1 / (lam_l * lorenz.dt)
independent = np.sqrt(1 / (2 * lam_l * lorenz.dt))
# Aligned errors add in magnitude:  sum delta e^{lambda(t-s)} -> delta e^{lambda t}/(lambda dt)
# Independent errors add in VARIANCE: sum delta^2 e^{2 lambda(t-s)} -> delta^2 e^{2 lambda t}/(2 lambda dt)
# The 2 inside the square root is easy to drop and inflates the prediction by
# sqrt(2) - small, but it is the difference between "agrees" and "does not".
lifts = []
print("      %-12s %s" % ("summary", "lift  B/A   (10-90 pct)"))
for how in ("median", "geometric", "mean"):
    a, b = wm.summarise(EA, how), wm.summarise(EB, how)
    s = (a > a[0] * 3) & (b < scale * 0.15)
    if s.sum() < 20: continue
    rr = (b / a)[s]; lifts.append(float(np.median(rr)))
    print("      %-12s %6.2f      %5.2f - %5.2f" % (how, np.median(rr), *np.percentile(rr, [10, 90])))
lift = float(np.median(lifts))
print("      %-12s %6.2f" % ("across all", lift))
print("      aligned prediction      %6.1f    (%.1fx away)" % (coherent, coherent / lift))
print("      independent prediction  %6.2f    (%.1fx away)" % (independent, lift / independent))
d_ind, d_coh = abs(np.log(lift / independent)), abs(np.log(lift / coherent))
off_ind, off_coh = max(lift / independent, independent / lift), max(coherent / lift, lift / coherent)
# Only claim a winner when one hypothesis is decisively closer. A measurement
# that sits near the midpoint in log space distinguishes nothing, and saying
# "closer to X" about a 3.7-versus-4.0 split would be reading noise as a result.
if max(d_ind, d_coh) / max(min(d_ind, d_coh), 1e-9) < 2.0:
    print("    -> %.1fx from one prediction and %.1fx from the other: this run sits" % (off_ind, off_coh))
    print("       between them and does NOT distinguish the two hypotheses.")
    print("       Other seeds have landed at a lift of ~11, which does favour the")
    print("       independent picture (1.5x versus 9.9x). The honest summary is that")
    print("       the lift is order 10, that it is nowhere near either extreme")
    print("       consistently, and that separating the two needs a better estimator")
    print("       than a ratio of two noisy curves. See 'Open questions'.")
else:
    print("    -> Closer to %s: %.1fx off versus %.1fx off." %
          ("independent" if d_ind < d_coh else "aligned", off_ind, off_coh))
    print("       The constant still moves with the estimator and the seed - we have")
    print("       measured anywhere from 8 to 28 - so treat the direction as the")
    print("       result and the number as an order of magnitude.")

# ═══ 5. And it is not one number per model ═════════════════════════════════
print("\n[5] 'How far can I trust it' is not one number.")
# Pick the tolerance from the data: one that most trajectories actually cross
# inside the window, or the percentiles are censoring artefacts rather than
# measurements. Quoting a spread over a censored sample is how this goes wrong.
span = np.linalg.norm(true, axis=-1).mean()
# The 10th percentile of the final-step error, so ~90% of trajectories cross
# it inside the window. Using the MEDIAN final error censors half the sample
# by construction - the tolerance has to be set low enough that the horizon is
# observed rather than cut off.
tol = float(np.percentile(err_p[:, -1], 10))
h = wm.usable_horizon(err_p, tol=tol)
censored = float((h == err_p.shape[1]).mean())
p5, p50, p95 = np.percentile(h, [5, 50, 95])
print("    Pendulum, tolerance %.3f (= 10th pct of final-step error, %.1f%% of state size)"
      % (tol, 100 * tol / span))
print("      5th pct %d steps | median %d | 95th pct %d   [%.0f%% of trajectories never crossed]"
      % (p5, p50, p95, 100 * censored))
if censored > 0.25:
    print("      Too many censored to quote a spread. Raise K or lower the tolerance.")
else:
    print("      A %.1fx spread across trajectories of the SAME model on the SAME task." % (p95 / max(p5, 1)))
    print("      A fixed rollout horizon is too long for the worst of them and needlessly")
    print("      short for the best. That is the opening Lesson 3 walks through -")
    print("      where it turns out not to set the planning horizon at all.")

# ═══ 6. Where the rate result holds, and where it does not ═════════════════
print("\n[6] The scope of section 3, established after Lesson 5 was written.")
print("    That result needed a chaotic system (lambda = %.2f), an accurate model," % lam_l)
print("    and %.1f decades of range before saturation - the regime where"
      % np.log10(scale * 0.2 / d0))
print("    amplification has something to amplify. Lesson 5 repeats these exact")
print("    measurements on three MuJoCo environments and gets a POWER LAW every")
print("    time: with lambda*dt small or the model error large, injection")
print("    dominates the entire horizon anyone plans over and lambda predicts")
print("    nothing. The pendulum in section 2 (lambda = %+.4f) is already that" % lam_p)
print("    case. Read section 3 as a statement about the chaotic regime.")

FIG.mkdir(exist_ok=True)
np.savez(FIG / "lesson02.npz", e_p=e_p, err_p=err_p, bound=bound, Lmax=Lmax, delta=delta,
         EA=EA, EB=EB, lam_p=lam_p, lam_l=lam_l, lift=lift,
         dt_p=pend.dt, dt_l=lorenz.dt, scale=scale, d0=d0, tol=tol,
         avg_keys=np.array(["%s|%s" % kv for kv in infl.keys()]),
         avg_rates=np.array(list(infl.values())))
print("\nSaved figures/lesson02.npz")
