"""Controls for every measurement in `wm`, run as tests.

These are not unit tests in the usual sense. Each one is a case where the right
answer is known independently - a textbook constant, an analytic derivative, a
synthetic curve of known shape - so that a measurement quietly going wrong
fails here rather than silently changing a conclusion in a lesson.

Several of them are negative controls, which matter more than the positive ones:
a metric that reports something sensible on data with no structure is a metric
that will report something sensible on a broken model too.
"""
import numpy as np
import pytest
import torch

import wm

RNG = lambda s=0: np.random.default_rng(s)


class TestDynamics:
    def test_pendulum_torch_matches_numpy_exactly(self):
        """The planner uses step_torch and the measurements use step. If they
        drift apart, Lesson 3's 'true dynamics' control stops being one."""
        P = wm.Pendulum(); rng = RNG()
        s, a = P.sample_states(256, rng), P.sample_actions(256, 1, rng)[:, 0]
        d = np.abs(P.step(s, a) - P.step_torch(torch.tensor(s), torch.tensor(a)).numpy()).max()
        assert d < 1e-12, "step_torch and step disagree by %.2e" % d

    def test_lorenz_is_on_its_attractor_after_burn_in(self):
        L = wm.Lorenz()
        s = L.sample_states(200, RNG())
        assert np.isfinite(s).all()
        assert 5 < np.linalg.norm(s, axis=-1).mean() < 60

    def test_rollout_is_deterministic(self):
        P = wm.Pendulum(); rng = RNG()
        s0, A = P.sample_states(32, rng), P.sample_actions(32, 20, rng)
        assert np.array_equal(wm.rollout(P, s0, 20, A), wm.rollout(P, s0, 20, A))


class TestLyapunov:
    def test_lorenz_exponent_matches_the_literature(self):
        """0.906 is the standard value. We measure it rather than quote it, so
        this test is what entitles Lesson 2 to say 'measured'."""
        lam, _ = wm.lyapunov(wm.Lorenz(), n=200, steps=3000, rng=RNG(1))
        assert abs(lam - 0.906) < 0.06, "measured lambda = %.4f" % lam

    def test_damped_pendulum_is_marginal_not_chaotic(self):
        lam, sd = wm.lyapunov(wm.Pendulum(), n=200, steps=3000, rng=RNG(2))
        assert abs(lam) < 0.15, "pendulum lambda = %.4f, expected near zero" % lam


class TestGrowthFits:
    def test_a_pure_exponential_is_called_exponential(self):
        e = np.exp(0.5 * np.arange(1, 200) * 0.05)
        assert wm.fit_growth(e, dt=0.05)["verdict"] == "exponential"

    def test_a_pure_power_law_is_called_a_power_law(self):
        assert wm.fit_growth(np.arange(1, 200) ** 1.3)["verdict"] == "power-law"

    def test_a_straight_line_is_a_power_law_not_an_exponential(self):
        assert wm.fit_growth(np.arange(1.0, 200.0))["verdict"] == "power-law"

    def test_the_recovered_rate_is_right(self):
        e = np.exp(0.9 * np.arange(1, 400) * 0.01)
        assert abs(wm.fit_growth(e, dt=0.01)["exp_rate"] - 0.9) < 0.02

    def test_too_few_points_refuses_rather_than_guesses(self):
        assert "exp_rate" not in wm.fit_growth(np.arange(1.0, 5.0))

    def test_the_interval_covers_the_truth_about_95pc_of_the_time(self):
        """A confidence interval that does not cover is worse than none, and
        this is the test that caught it: bootstrapping the POINTS of one
        averaged curve gave intervals roughly half as wide as they should be,
        because points along a rollout are serially correlated. Resampling
        trajectories fixed it. Coverage below ~85/100 means that regressed."""
        rng, K, dt, TRUE = RNG(3), 400, 0.01, 0.9
        cover = 0
        for _ in range(40):
            E = np.array([np.exp(TRUE * np.arange(1, K + 1) * dt) * np.exp(rng.normal(0, .3, K))
                          for _ in range(200)])
            lo, hi = wm.fit_growth_ci(E, dt=dt, n_boot=150, rng=rng)["exp_rate_ci"]
            cover += lo <= TRUE <= hi
        assert cover >= 32, "coverage %d/40, intervals are too narrow" % cover


class TestLipschitz:
    def test_numerical_jacobian_matches_the_analytic_one(self):
        """For the pendulum the Jacobian is two lines of calculus, so the
        finite-difference estimate has something exact to be checked against."""
        P = wm.Pendulum(); rng = RNG(4)
        s = P.sample_states(500, rng); a = P.sample_actions(500, 1, rng)[:, 0]
        J = np.zeros((len(s), 2, 2))
        J[:, 0, 0] = 1.0; J[:, 0, 1] = P.dt
        J[:, 1, 0] = -P.dt * (P.g / P.l) * np.cos(s[:, 0])
        J[:, 1, 1] = 1.0 - P.dt * P.b
        analytic = np.linalg.norm(J, ord=2, axis=(1, 2)).max()
        assert abs(wm.lipschitz(P, s, a)[0] - analytic) < 1e-3

    def test_the_bound_reduces_to_delta_times_k_when_l_is_one(self):
        k = np.arange(1, 20)
        assert np.allclose(wm.textbook_bound(1.0, 0.5, k), 0.5 * k)


class TestSummaries:
    def test_the_three_averages_agree_when_the_noise_is_stationary(self):
        """They only come apart when the spread grows with time, which is the
        whole point of Lesson 2's section 3b. If they disagree here, the
        difference measured there is an artefact of the estimator instead."""
        rng = RNG(5)
        E = np.array([np.exp(0.9 * np.arange(1, 400) * 0.01) * np.exp(rng.normal(0, .4, 399))
                      for _ in range(300)])
        rates = [wm.fit_growth_ci(E, dt=.01, n_boot=80, rng=RNG(6), how=h)["exp_rate"]
                 for h in ("mean", "median", "geometric")]
        assert max(rates) - min(rates) < 0.02, "estimators disagree on iid noise: %s" % rates

    def test_usable_horizon_censors_rather_than_lies(self):
        E = np.array([[0.01] * 10, [0.01] * 5 + [1.0] * 5])
        h = wm.usable_horizon(E, tol=0.5)
        assert h[0] == 10 and h[1] == 6

    def test_an_unknown_average_is_an_error_not_a_default(self):
        with pytest.raises(ValueError):
            wm.summarise(np.ones((3, 4)), how="mode")


class TestRepresentation:
    def test_effective_rank_counts_directions(self):
        torch.manual_seed(0)
        assert abs(wm.effective_rank(torch.randn(4000, 8)) - 8) < 0.3
        assert abs(wm.effective_rank(torch.randn(4000, 1) @ torch.randn(1, 8)) - 1) < 0.05

    def test_effective_rank_is_blind_to_scale(self):
        """Not a bug - a documented blind spot, and Lesson 4 turns on it. A
        representation squashed to a std of 1e-4 still reports a healthy rank,
        which is why it cannot be used as a collapse detector on its own."""
        torch.manual_seed(0)
        Z = torch.randn(4000, 8)
        assert abs(wm.effective_rank(Z) - wm.effective_rank(Z * 1e-4)) < 0.01

    def test_probe_r2_is_one_when_the_target_is_present(self):
        torch.manual_seed(0)
        y = torch.randn(2000, 2)
        assert wm.probe_r2(torch.cat([y, torch.randn(2000, 6)], 1), y) > 0.99

    def test_probe_r2_is_zero_on_noise(self):
        torch.manual_seed(0)
        assert abs(wm.probe_r2(torch.randn(2000, 8), torch.randn(2000, 2))) < 0.05


class TestPlanningMetrics:
    @staticmethod
    def _reward(s, u):
        up = torch.remainder(s[..., 0] - np.pi + np.pi, 2 * np.pi) - np.pi
        return -(up ** 2 + 0.1 * s[..., 1] ** 2)

    def test_perfect_model_ranks_perfectly(self):
        P = wm.Pendulum()
        step = lambda s, a: P.step_torch(s, a)
        s0 = torch.tensor(P.sample_states(32, RNG(7)), dtype=torch.float32)
        rho, regret = wm.rank_fidelity(step, step, self._reward, s0, 20)
        assert rho > 0.999 and regret < 1e-5

    def test_a_useless_model_ranks_at_chance(self):
        P = wm.Pendulum()
        s0 = torch.tensor(P.sample_states(32, RNG(8)), dtype=torch.float32)
        rho, regret = wm.rank_fidelity(lambda s, a: s + 0.05 * torch.randn_like(s),
                                       lambda s, a: P.step_torch(s, a),
                                       self._reward, s0, 20)
        assert abs(rho) < 0.5 and regret > 0.1

    def test_the_planner_beats_random_with_the_true_dynamics(self):
        """If this fails the planner is broken, and every model comparison in
        Lesson 3 is measuring the planner rather than the model."""
        P = wm.Pendulum()
        step = lambda s, a: P.step_torch(s, a)
        s0 = torch.zeros(8, 2); s0[:, 0] = torch.linspace(-.3, .3, 8)
        good, _, _ = wm.cem_mpc(step, step, self._reward, s0, 25, 40, u_max=3.0,
                                n_candidates=96, iters=3, n_elite=12)
        lazy, _, _ = wm.cem_mpc(lambda s, a: s, step, self._reward, s0, 1, 40, u_max=3.0,
                                n_candidates=8, iters=1, n_elite=2)
        assert good.mean() > lazy.mean() + 0.5

    def test_action_dim_is_not_hard_coded(self):
        """It was, in two places, until Lesson 6 tried a two-joint arm."""
        for A in (1, 2, 5):
            s0 = torch.zeros(4, 3)
            step = lambda s, a: s + 0.01 * a.sum(-1, keepdim=True)
            r, sf, traj = wm.cem_mpc(step, step, lambda s, u: -s.norm(dim=-1),
                                     s0, 4, 3, n_candidates=16, iters=2, n_elite=4,
                                     action_dim=A)
            assert sf.shape == s0.shape
            rho, _ = wm.rank_fidelity(step, step, lambda s, u: -s.norm(dim=-1),
                                      s0, 4, n_candidates=16, action_dim=A)
            assert rho > 0.99


class TestDiagnose:
    """The one-call report is the part someone will point at their own model, so
    it is the part that must refuse to invent a number."""

    @staticmethod
    def _synthetic(rate=0.9, dt=0.01, K=400, N=200, noise=0.25, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(1, K + 1) * dt
        th = rng.uniform(0, 2 * np.pi, (N, K))
        true = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1)
        mag = (3e-4 * np.exp(rate * t))[None, :] * np.exp(rng.normal(0, noise, (N, K)))
        u = rng.standard_normal((N, K, 3)); u /= np.linalg.norm(u, axis=-1, keepdims=True)
        return true + mag[..., None] * u, true

    def test_it_recovers_a_growth_rate_it_was_given(self):
        pred, true = self._synthetic(rate=0.9)
        r = wm.diagnose(pred, true, dt=0.01, lam=0.9, n_boot=120)
        lo, hi = r["rates"]["median"]["exp_rate_ci"]
        assert lo <= 0.9 <= hi, "interval [%.4f, %.4f] misses 0.9" % (lo, hi)
        assert r["shape"]["verdict"] == "exponential"

    def test_a_clean_measurement_carries_no_caveats(self):
        pred, true = self._synthetic()
        assert wm.diagnose(pred, true, dt=0.01, n_boot=120)["warnings"] == []

    def test_identical_arrays_are_named_as_such(self):
        _, true = self._synthetic()
        r = wm.diagnose(true, true)
        assert "identical" in str(r)
        assert r["rates"] == {}

    def test_truth_with_no_scale_is_refused(self):
        """Centred residuals passed as states used to yield a tolerance of
        800000000000% of the state size, printed without complaint."""
        with pytest.raises(ValueError, match="norm ~0"):
            wm.diagnose(np.ones((5, 10, 3)), np.zeros((5, 10, 3)))

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            wm.diagnose(np.ones((5, 10, 3)), np.ones((5, 11, 3)))

    def test_a_censored_horizon_is_declared_not_quoted(self):
        pred, true = self._synthetic()
        r = wm.diagnose(pred, true, dt=0.01, tol=1e9, n_boot=60)
        assert r["censored"] == 1.0
        assert any("censored" in w for w in r["warnings"])

    def test_lambda_with_an_error_bar_is_compared_as_an_interval(self):
        """A measured lambda is an estimate, so comparing a fit interval against
        it as a point manufactures disagreements. The Lorenz example lands 1.6%
        below a lambda of 0.9046 whose own sd is 0.043: a real agreement that
        the point comparison called a miss."""
        pred, true = self._synthetic(rate=0.9)
        point = wm.diagnose(pred, true, dt=0.01, lam=0.80, n_boot=120)
        wide = wm.diagnose(pred, true, dt=0.01, lam=(0.80, 0.10), n_boot=120)
        assert "does not cover it" in str(point)
        assert "agrees within" in str(wide)
        # A wide error bar must not turn every gap into agreement.
        near = wm.diagnose(pred, true, dt=0.01, lam=(0.80, 0.0005), n_boot=120)
        assert "disagrees beyond" in str(near)

    def test_a_malformed_lambda_pair_is_refused(self):
        pred, true = self._synthetic()
        with pytest.raises(ValueError, match="value, sd"):
            wm.diagnose(pred, true, dt=0.01, lam=(0.9, 0.1, 0.2), n_boot=60)

    def test_a_refused_fit_prints_no_nan(self):
        """When there is too little dynamic range to fit anything, the report
        has to say so. Printing `residuals exp nan / pow nan` reads like a
        failed computation rather than a deliberate refusal."""
        pred, true = self._synthetic(K=6, N=40, noise=0.02)
        r = wm.diagnose(pred, true, dt=0.01, n_boot=60)
        text = str(r)
        assert "nan" not in text, text
        assert "not fitted" in text
        # And the refusal has to reach the caveats: this printed "no caveats:
        # ... the shape fit are all clean" about a fit that never ran.
        assert "no caveats" not in text, text
        assert any("not fitted" in w for w in r["warnings"]), r["warnings"]

    def test_a_curve_that_settles_neither_shape_is_called_ambiguous(self):
        """The other refusal the module docstring calls most of the value.

        Nothing tested it. `exponential` and `power-law` each had a test; the
        verdict for "this curve does not tell you which" did not, so the branch
        that exists to stop a coin flip being reported as a finding was itself
        unchecked.

        The curve is the geometric mean of exp(0.28k) and k^3.75, which no
        straight line in either log-space describes better than the other -
        their residuals here come out within 12%, under the 1.5x the code
        requires before it will name a winner.
        """
        K, N = 40, 24
        k = np.arange(1, K + 1)
        e = np.sqrt(np.exp(0.28 * k) * k ** 3.75)
        e = e / e[-1] * 0.19        # rescaling shifts log by a constant, so it
        ang = np.linspace(0, 2 * np.pi, N, endpoint=False)   # cannot change
        unit = np.stack([np.cos(ang), np.sin(ang)], 1)       # either residual
        true = np.stack([unit] * K, 1)
        pred = true.copy()
        pred[:, :, 0] += e[None, :]
        r = wm.diagnose(pred, true, n_boot=60)
        assert r["shape"]["verdict"] == "ambiguous", r["shape"]
        assert r["shape"]["resid_ratio"] < 1.5, r["shape"]
        # And it has to reach the reader, not just the dict.
        assert any("does not distinguish" in w for w in r["warnings"]), r["warnings"]
        assert "no caveats" not in str(r)

    def test_the_summary_disagreement_is_surfaced(self):
        """Heteroscedastic noise makes mean, median and geometric mean disagree.
        That disagreement is the finding, so it has to reach the reader."""
        rng = np.random.default_rng(1)
        K, N, dt = 400, 300, 0.01
        t = np.arange(1, K + 1) * dt
        th = rng.uniform(0, 2 * np.pi, (N, K))
        true = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], -1)
        widening = 0.05 + 1.2 * t
        mag = (3e-4 * np.exp(0.9 * t))[None, :] * np.exp(rng.normal(0, 1, (N, K)) * widening)
        u = rng.standard_normal((N, K, 3)); u /= np.linalg.norm(u, axis=-1, keepdims=True)
        r = wm.diagnose(true + mag[..., None] * u, true, dt=dt, n_boot=120)
        assert any("summarise" in w for w in r["warnings"]), r["warnings"]
