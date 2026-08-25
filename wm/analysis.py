"""Measurements. This is the half that other tutorials leave out.

Building a world model is the easy part; knowing how far you may trust it is
the part that decides whether it is useful. Everything here is a measurement
with a ground truth to check it against.
"""
import numpy as np

__all__ = ["rollout_error", "lyapunov", "fit_growth", "usable_horizon",
           "lipschitz", "textbook_bound", "fit_growth_ci", "summarise",
           "rank_fidelity", "effective_rank", "probe_r2"]


def rollout_error(pred, true):
    """Per-trajectory, per-step L2 error. Returns (n, k)."""
    return np.linalg.norm(np.asarray(pred) - np.asarray(true), axis=-1)


def lyapunov(system, n=400, steps=4000, eps=1e-8, rng=None, actions=True):
    """Largest Lyapunov exponent, by Benettin's method.

    Evolve a point and a nearby twin, accumulate the log of how fast they
    separate, and renormalise the gap back to eps at every step so it never
    leaves the linear regime. The mean log-growth per unit time is lambda.

    The renormalisation is the step people drop, and dropping it silently
    caps the estimate: once the pair saturates at the attractor diameter the
    measured growth is zero, so a long run reports lambda ~ 0 for a system
    that is plainly chaotic.

    Both members of the pair are driven by the SAME action sequence, otherwise
    the measurement is of the action noise rather than of the dynamics.
    """
    rng = rng or np.random.default_rng(0)
    from .systems import rollout
    s = system.sample_states(n, rng)
    d = rng.standard_normal(s.shape)
    p = s + eps * d / np.linalg.norm(d, axis=-1, keepdims=True)
    acc = np.zeros(n)
    for _ in range(steps):
        a = system.sample_actions(n, 1, rng)[:, 0] if (actions and system.action_dim) else None
        s, p = system.step(s, a), system.step(p, a)
        gap = np.linalg.norm(p - s, axis=-1)
        acc += np.log(gap / eps)
        p = s + (p - s) / gap[:, None] * eps
    per = acc / (steps * system.dt)
    return float(per.mean()), float(per.std())


def fit_growth(e, dt=1.0, lo=None, hi=None, decisive=1.5):
    """Is this curve exponential or power-law, and at what rate?

    Returns a dict with both fits and their residuals. Whichever has the
    smaller residual in log-space is the better description - reporting only
    one of them is how a power law gets mistaken for an exponential.

    A curve does not always distinguish them, and saying so matters more than
    it sounds. `decisive` is how many times larger the losing residual must be
    before a winner is declared; below that the verdict is "ambiguous". The
    default of 1.5 was set from measurement, not taste: on synthetic curves of
    known shape the ratio is 3 to 23, while the real curves that flip between
    random seeds sit at 1.2. Without this, a binary verdict gets reported on a
    coin flip - two of this repository's headline findings did exactly that,
    holding in two runs out of six, until the seeds were varied.

    Restrict to [lo, hi] to exclude the saturated tail: once the error reaches
    the size of the state space it stops growing for reasons that have nothing
    to do with the model, and including that tail flattens any fit.
    """
    e = np.asarray(e, dtype=float)
    k = np.arange(1, len(e) + 1)
    m = np.ones(len(e), bool)
    if lo is not None: m &= e > lo
    if hi is not None: m &= e < hi
    if m.sum() < 8:
        return {"n": int(m.sum()), "verdict": "too few points"}
    kk, ee, t = k[m], np.log(e[m]), (k[m] * dt)
    ce = np.polyfit(t, ee, 1); cp = np.polyfit(np.log(kk), ee, 1)
    r_exp = float(np.std(ee - np.polyval(ce, t)))
    r_pow = float(np.std(ee - np.polyval(cp, np.log(kk))))
    ratio = max(r_exp, r_pow) / max(min(r_exp, r_pow), 1e-12)
    if ratio < decisive:
        verdict = "ambiguous"
    else:
        verdict = "exponential" if r_exp < r_pow else "power-law"
    return {"n": int(m.sum()), "k_range": (int(kk[0]), int(kk[-1])),
            "exp_rate": float(ce[0]), "pow_alpha": float(cp[0]),
            "resid_exp": r_exp, "resid_pow": r_pow,
            "resid_ratio": float(ratio), "leaning": "exponential" if r_exp < r_pow else "power-law",
            "verdict": verdict}


def usable_horizon(err, tol):
    """First step at which each trajectory's error exceeds tol.

    Trajectories that never exceed it are recorded as the full length, so the
    result is right-censored - quote percentiles, not a mean.
    """
    err = np.asarray(err); k = err.shape[1]
    return np.array([(np.argmax(r > tol) + 1) if (r > tol).any() else k for r in err])


def lipschitz(system, states, actions=None, h=1e-5):
    """Worst-case and median spectral norm of the true one-step Jacobian.

    This is the L that appears in the textbook rollout bound
    e_{k+1} <= L e_k + delta, whose solution is delta (L^k - 1)/(L - 1).

    Lesson 2's point is that L is a worst case over directions and over the
    state space, while what actually governs a rollout is the Lyapunov
    exponent - the average log growth along the trajectory. When the two
    disagree, and they usually do, the bound is not merely loose: it predicts
    the wrong functional form.
    """
    S = np.atleast_2d(np.asarray(states, dtype=float))
    n, d = S.shape
    a = None if actions is None else np.asarray(actions, dtype=float)
    base = system.step(S, a)
    J = np.empty((n, d, d))
    for j in range(d):
        E = np.zeros_like(S); E[:, j] = h
        J[:, :, j] = (system.step(S + E, a) - base) / h
    sv = np.linalg.norm(J, ord=2, axis=(1, 2))
    return float(sv.max()), float(np.median(sv))


def textbook_bound(L, delta, k):
    """delta (L^k - 1)/(L - 1), the classical compounding-error bound."""
    k = np.asarray(k, dtype=float)
    return delta * k if abs(L - 1) < 1e-12 else delta * (L ** k - 1) / (L - 1)


def fit_growth_ci(E, dt=1.0, lo=None, hi=None, n_boot=400, rng=None, how="median"):
    """fit_growth, with a confidence interval from resampling TRAJECTORIES.

    E is the (n_trajectories, n_steps) error matrix, not the averaged curve.
    That distinction is the whole point. Bootstrapping the points of a single
    averaged curve is easy and wrong: successive points of one rollout are
    strongly serially correlated, so resampling them treats correlated data as
    independent and returns an interval roughly half as wide as the truth.
    Resampling whole trajectories respects the one axis that really is i.i.d.

    Returns the fit_growth dict plus 'exp_rate_ci' and 'pow_alpha_ci'.
    """
    rng = rng or np.random.default_rng(0)
    E = np.asarray(E, dtype=float)
    base = fit_growth(summarise(E, how), dt=dt, lo=lo, hi=hi)
    if "exp_rate" not in base:
        return base
    rates, alphas = [], []
    n = len(E)
    for _ in range(n_boot):
        g = fit_growth(summarise(E[rng.integers(0, n, n)], how), dt=dt, lo=lo, hi=hi)
        if "exp_rate" in g:
            rates.append(g["exp_rate"]); alphas.append(g["pow_alpha"])
    if len(rates) < n_boot // 2:
        base["exp_rate_ci"] = base["pow_alpha_ci"] = (float("nan"),) * 2
        return base
    base["exp_rate_ci"] = tuple(np.percentile(rates, [2.5, 97.5]))
    base["pow_alpha_ci"] = tuple(np.percentile(alphas, [2.5, 97.5]))
    base["n_boot"] = len(rates)
    return base


def summarise(E, how="median"):
    """Collapse an (n_trajectories, n_steps) error matrix into one curve.

    Which average you pick changes the answer, and not by a rounding error.
    Rollout error is right-skewed across trajectories: a few unlucky rollouts
    run away while most stay tame. The arithmetic mean follows the runaways, so
    it grows faster than the typical trajectory does, and the growth rate you
    fit to it is not the growth rate of the process.

    Measured on the Lorenz system in Lesson 2: the mean curve gives a rate ~8%
    above the Lyapunov exponent while the median and the geometric mean both
    land within ~1-3% of it. Most papers report the mean.

        median      the typical trajectory. The default, and the safe choice.
        geometric   exp(mean(log e)) - the natural one for exponential growth
        mean        what everyone reports; kept so the difference is visible
    """
    E = np.asarray(E, dtype=float)
    if how == "mean":
        return E.mean(0)
    if how == "median":
        return np.median(E, 0)
    if how == "geometric":
        return np.exp(np.log(np.maximum(E, 1e-300)).mean(0))
    raise ValueError("how must be one of: median, geometric, mean")


def rank_fidelity(plan_step, env_step, reward_fn, s0, horizon, n_candidates=128,
                  u_max=3.0, std=1.8, generator=None, action_dim=1):
    """How well does the model RANK candidate action sequences? Plus the regret.

    A planner never uses the states a model predicts. It uses them only to score
    action sequences, then throws them away and executes the first action of the
    winner. So the quantity that decides whether planning works is not how far
    the rollout stays accurate - it is whether the model orders candidates the
    same way the real world does.

    The two come apart, and by a lot. Lesson 3 measures a model whose rollout
    error passes any reasonable tolerance within 7 steps, yet which ranks
    25-step candidate sequences at a Spearman correlation of 0.995 and picks the
    truly optimal sequence every time. Judging that model by its usable horizon
    would have thrown away a planner that works.

    `action_dim` defaults to 1 because Lessons 1-3 have scalar actions. It was
    hard-coded until Lesson 6 tried a 2-joint arm, and the same assumption was
    baked into `cem_mpc` as well - one wrong constant in two places, which is
    what a defect that comes from an unstated assumption usually looks like.

    Returns (median Spearman rho, median regret), where regret is the true
    return of the best sequence minus the true return of the one the model
    picked - zero means the model's choice was actually optimal.
    """
    import torch
    g = generator
    N, D = s0.shape
    noise = torch.randn(N, n_candidates, horizon, int(action_dim),
                        device=s0.device, generator=g)
    A = (noise * std).clamp(-u_max, u_max)

    def score(step):
        s = s0[:, None].expand(N, n_candidates, D).reshape(-1, D)
        R = torch.zeros(N * n_candidates, device=s0.device)
        for h in range(horizon):
            u = A[:, :, h].reshape(-1, int(action_dim))
            s = step(s, u)
            R = R + reward_fn(s, u)
        return R.view(N, n_candidates)

    with torch.no_grad():
        Rm, Rt = score(plan_step), score(env_step)
        ra = Rm.argsort(1).argsort(1).float(); rb = Rt.argsort(1).argsort(1).float()
        ra = ra - ra.mean(1, keepdim=True); rb = rb - rb.mean(1, keepdim=True)
        rho = (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-9)
        pick = Rm.argmax(1)
        regret = Rt.max(1).values - Rt.gather(1, pick[:, None]).squeeze(1)
    return float(rho.median()), float(regret.median())


def effective_rank(Z):
    """Participation ratio of the covariance spectrum: how many directions the
    representation actually uses, as a continuous number between 1 and dim(Z).

    Useful, and on its own misleading. It is computed from the *shape* of the
    eigenvalue spectrum, so it is blind to overall scale: a representation
    squashed to a std of 0.0003 can still have a flat spectrum and report a
    healthy rank. Lesson 4 has exactly that case, and the case where the
    opposite metric is fooled instead.
    """
    import torch
    Z = torch.as_tensor(Z)
    Z = Z - Z.mean(0)
    C = (Z.T @ Z) / max(len(Z), 1)
    ev = torch.linalg.eigvalsh(C).clamp_min(1e-12)
    p = ev / ev.sum()
    return float(torch.exp(-(p * p.log()).sum()))


def probe_r2(Z, target):
    """R^2 of a least-squares linear probe from Z to target.

    Never read this number on its own. A random, untrained encoder is a strong
    baseline - random features preserve a low-dimensional state remarkably well
    - so "R^2 = 0.81" can be below the floor rather than above it, and in
    Lesson 4 it is. Always report it next to the same probe on an untrained
    encoder of the same shape, and on the raw observation, so that the learned
    number has a floor and a ceiling to sit between.
    """
    import torch
    Z = torch.as_tensor(Z); target = torch.as_tensor(target)
    Z = torch.cat([Z, torch.ones(len(Z), 1, device=Z.device, dtype=Z.dtype)], 1)
    w = torch.linalg.lstsq(Z, target).solution
    ss_res = ((Z @ w - target) ** 2).sum()
    ss_tot = ((target - target.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)
