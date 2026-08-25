"""One call that measures what the reference world-model implementations do not.

DreamerV3 logs open-loop prediction as a picture: six sequences, five steps of
context, the rest imagined, rendered next to the truth with a green-to-red
border marking where imagination begins, for a human to look at. The official
JAX implementation and the most-used PyTorch port do this identically - no
scalar leaves the function. TD-MPC2 does report one, `consistency_loss`: a
discounted mean squared error over its three-step training horizon, taken on
replay batches, sent to wandb but not to the console and not to the saved CSV.
That is a training loss, not a prediction diagnostic. It is never divided by
how far the state actually moved, so it has no scale you can read; it stops at
the training horizon; and it is not reported per task at evaluation. None of
the three answers the question a planner needs answered - how many steps ahead
is this model still worth trusting, on this task. (Audited 2026-08 against the
code, TD-MPC2 at e9f5932; the papers may report more than the code logs.)

So if you want to know how far your rollouts stay usable, you write it yourself.
This is that, written once:

    report = wm.diagnose(pred, true)
    print(report)

`pred` and `true` are (n_trajectories, n_steps, state_dim) arrays - your model's
rollout and the ground truth it was supposed to match, from the same starting
states under the same actions. Nothing here touches your model, so it does not
matter what framework it is in or whether it predicts states, latents or pixels
flattened into a vector.

What it will not do is give you a number it cannot stand behind. If too many
trajectories never cross the tolerance, it says the horizon is censored instead
of quoting a percentile. If the growth curve does not distinguish an exponential
from a power law, it says ambiguous instead of picking one. Those two refusals
are most of the value: this repository shipped both of those mistakes before the
guards existed.
"""
import numpy as np

from .analysis import (fit_growth, fit_growth_ci, rollout_error, summarise,
                       usable_horizon)

__all__ = ["diagnose", "Diagnosis"]


class Diagnosis(dict):
    """A dict of measurements that prints as a report."""

    def __str__(self):
        return self["text"]

    __repr__ = __str__


def _fmt_pct(x):
    return "%+.1f%%" % (100 * x)


def diagnose(pred, true, dt=1.0, tol=None, lam=None, n_boot=300, seed=0):
    """Measure a world model's rollouts against the truth they were meant to match.

    pred, true  (n, k, d) arrays: the model's rollout and the ground truth, from
                the same starting states under the same actions. If these two
                differ in either, what comes back measures that difference and
                not the model - it is the easiest way to get a meaningless
                answer here, and one this repository got wrong once.
    dt          simulator timestep, so growth rates come back per unit time
    tol         error at which a trajectory stops being usable. The default is
                the 10th percentile of the final-step error, chosen so that most
                trajectories cross it inside the window; a tolerance most of them
                never reach gives censored percentiles, not measurements.
    lam         the system's Lyapunov exponent, if you have measured it, so the
                fitted growth rate can be compared against it. Pass it as
                `(value, sd)` if you know its uncertainty - `wm.lyapunov`
                returns exactly that pair - and the comparison is made between
                two intervals instead of against a point. It matters: a fit of
                0.890 [0.878, 0.900] misses a lambda of 0.9046 as a point and
                agrees with it completely once lambda's own +-0.043 is counted.
    """
    lam_sd = None
    if lam is not None and np.ndim(lam) == 1:
        if len(lam) != 2:
            raise ValueError("lam must be a number or a (value, sd) pair, got %d values"
                             % len(lam))
        lam, lam_sd = float(lam[0]), abs(float(lam[1]))

    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    if pred.shape != true.shape:
        raise ValueError("pred %s and true %s must have the same shape"
                         % (pred.shape, true.shape))
    if pred.ndim != 3:
        raise ValueError("expected (n_trajectories, n_steps, state_dim), got %s"
                         % (pred.shape,))
    n, k, _ = pred.shape
    rng = np.random.default_rng(seed)
    err = rollout_error(pred, true)
    scale = float(np.linalg.norm(true, axis=-1).mean())

    # Degenerate inputs produce numbers that look like measurements. A perfect
    # model gives a first-step error of zero and a dynamic range of log10(x/0);
    # ground truth centred on the origin gives a scale of zero and a tolerance
    # of infinity percent of it. Both printed confidently before this guard,
    # which is worse than failing.
    if scale <= 1e-12:
        raise ValueError(
            "the ground-truth states have norm ~0, so there is no scale to "
            "measure error against. Pass states in their own units rather than "
            "centred residuals, or set a tolerance explicitly.")
    if float(err.max()) <= 1e-12:
        return Diagnosis(n=n, steps=k, tol=0.0, scale=scale, censored=1.0,
                         horizon=dict(p5=k, p50=k, p95=k, spread=1.0),
                         shape={}, decades=float("inf"), rates={},
                         warnings=["pred and true are identical to floating "
                                   "point, so there is nothing to diagnose"],
                         error=err,
                         text=("world model rollout diagnosis   %d trajectories x %d steps\n"
                               "\n  pred and true are identical to floating point.\n"
                               "  Either the model is exact, or - far more often - the same\n"
                               "  array was passed twice." % (n, k)))

    if tol is None:
        tol = float(np.percentile(err[:, -1], 10))
    h = usable_horizon(err, tol)
    censored = float((h == k).mean())
    p5, p50, p95 = (float(x) for x in np.percentile(h, [5, 50, 95]))

    med = summarise(err, "median")
    lo, hi = med[0] * 3, scale * 0.2
    decades = float(np.log10(hi / med[0])) if med[0] > 0 and hi > med[0] else 0.0
    shape = fit_growth(med, dt=dt, lo=lo, hi=hi)

    rates = {}
    for how in ("median", "geometric", "mean"):
        c = summarise(err, how)
        g = fit_growth_ci(err, dt=dt, lo=c[0] * 3, hi=scale * 0.2,
                          n_boot=n_boot, rng=rng, how=how)
        if "exp_rate" in g:
            rates[how] = g

    warn = []
    if censored > 0.25:
        warn.append("%.0f%% of trajectories never reach the tolerance, so the "
                    "horizon percentiles are censored - lower tol or extend the "
                    "rollout before quoting them" % (100 * censored))
    if decades < 2:
        warn.append("only %.1f decades between the first-step error and "
                    "saturation, which is not enough to tell a growth shape "
                    "apart" % decades)
    if shape.get("verdict") == "ambiguous":
        warn.append("the exponential and power-law fits are within %.0f%% of "
                    "each other, so the curve does not distinguish them"
                    % (100 * (shape["resid_ratio"] - 1)))
    # A refusal has to reach the caveats, or the report closes with "no caveats:
    # the shape fit is clean" about a fit that was never attempted. It printed
    # exactly that on a six-step rollout before this guard.
    if "exp_rate" not in shape:
        warn.append("the growth shape was not fitted (%s), so nothing below "
                    "describes a curve" % shape.get("verdict", "too few points"))
    if not rates:
        warn.append("no growth rate could be fitted in the window between the "
                    "first-step error and saturation - the rollout is too short "
                    "or too accurate for a rate to mean anything")
    if lam is not None and rates.get("median"):
        rmed = rates["median"]["exp_rate"]
        if abs(lam) <= 0.1 * abs(rmed):
            warn.append("lambda is ~0 (%.4f) while the error grows at %.3f per unit "
                        "time, so amplification is not what you are watching - the "
                        "model is injecting error faster than the dynamics stretch it"
                        % (lam, rmed))
    if len(rates) == 3:
        spread = max(r["exp_rate"] for r in rates.values()) / \
            max(min(r["exp_rate"] for r in rates.values()), 1e-12)
        if spread > 1.05:
            warn.append("the fitted rate moves by %.0f%% depending on whether you "
                        "summarise trajectories by mean, median or geometric mean"
                        % (100 * (spread - 1)))

    L = []
    L.append("world model rollout diagnosis   %d trajectories x %d steps, state dim %d"
             % (n, k, pred.shape[2]))
    L.append("")
    L.append("  usable horizon   tolerance %.4g (%.1f%% of typical state size)"
             % (tol, 100 * tol / max(scale, 1e-12)))
    L.append("      5th %.0f    median %.0f    95th %.0f    spread %.1fx%s"
             % (p5, p50, p95, p95 / max(p5, 1),
                "    [%.0f%% censored]" % (100 * censored) if censored else ""))
    L.append("")
    if all(k_ in shape for k_ in ("resid_exp", "resid_pow", "resid_ratio")):
        L.append("  growth shape     residuals exp %.3f / pow %.3f, ratio %.2f -> %s"
                 % (shape["resid_exp"], shape["resid_pow"], shape["resid_ratio"],
                    shape.get("verdict", "not enough points")))
    else:
        # No fit was attempted, so there are no residuals to quote. Printing
        # nan here reads like a failed computation rather than a refusal.
        L.append("  growth shape     not fitted -> %s"
                 % shape.get("verdict", "not enough points"))
    L.append("      %.1f decades of range before saturation" % decades)
    if rates:
        L.append("")
        L.append("  growth rate      by how you summarise trajectories")
        for how in ("median", "geometric", "mean"):
            if how not in rates:
                continue
            g = rates[how]
            ci = g["exp_rate_ci"]
            extra = ""
            if lam is not None:
                if lam_sd:
                    # Two intervals, not an interval against a point: lambda is
                    # itself an estimate, and calling a 2% gap a disagreement
                    # when lambda carries +-5% is a claim the data cannot make.
                    lo_l, hi_l = lam - 1.96 * lam_sd, lam + 1.96 * lam_sd
                    covers = (", agrees within lambda's own +-%.3f" % (1.96 * lam_sd)
                              if ci[0] <= hi_l and lo_l <= ci[1]
                              else ", disagrees beyond lambda's own +-%.3f" % (1.96 * lam_sd))
                else:
                    covers = ", covers it" if ci[0] <= lam <= ci[1] else ", does not cover it"
                # A ratio to lambda is only readable when lambda is far enough
                # from zero. On a marginal system it is not: a rate of 0.52
                # against a lambda of 0.017 comes out as "+2904%", which is
                # arithmetic rather than information.
                if abs(lam) > 0.1 * abs(g["exp_rate"]):
                    extra = "   %s vs lambda%s" % (_fmt_pct(g["exp_rate"] / lam - 1), covers)
                else:
                    extra = "   lambda=%.4f is ~0 here, so read the gap as %+.3f/unit time%s" % (
                        lam, g["exp_rate"] - lam, covers)
            L.append("      %-10s %.4f  95%%CI [%.4f, %.4f]%s"
                     % (how, g["exp_rate"], ci[0], ci[1], extra))
    if warn:
        L.append("")
        L.append("  read with care")
        for w in warn:
            for i, line in enumerate(_wrap(w, 68)):
                L.append("      %s%s" % ("- " if i == 0 else "  ", line))
    else:
        L.append("")
        L.append("  no caveats: the range, the censoring and the shape fit are all clean")

    return Diagnosis(n=n, steps=k, tol=tol, scale=scale, censored=censored,
                     horizon=dict(p5=p5, p50=p50, p95=p95, spread=p95 / max(p5, 1)),
                     shape=shape, decades=decades, rates=rates,
                     warnings=warn, error=err, text="\n".join(L))


def _wrap(s, width):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out
