"""Ground-truth dynamical systems, in numpy. No simulator, no MuJoCo, no OpenGL.

Every lesson needs a world whose true dynamics we already know, so that "how
wrong is the model" is a question with an exact answer. These two systems are
chosen because they sit on opposite sides of the one property that turns out to
govern everything in Lesson 2 - the Lyapunov exponent.

    Pendulum   lambda ~ 0      errors accumulate but are not amplified
    Lorenz     lambda ~ +0.90  errors are amplified exponentially

Both are ~15 lines. You can read them, and you should: half the value of a
from-scratch tutorial is that nothing is hiding behind an import.
"""
import numpy as np

__all__ = ["Pendulum", "Lorenz", "rollout"]


class Pendulum:
    """Damped pendulum driven by a torque. State (theta, theta_dot), action (u,).

    theta_ddot = -(g/l) sin(theta) - b theta_dot + u / (m l^2)

    Integrated with explicit Euler, which is what makes it a fair stand-in for
    a learned model's setting: a fixed-step map, not an adaptive solver.
    """
    state_dim, action_dim = 2, 1

    def __init__(self, g=9.81, l=1.0, m=1.0, b=0.10, dt=0.05):
        self.g, self.l, self.m, self.b, self.dt = g, l, m, b, dt

    def step(self, s, a):
        th, om = s[..., 0], s[..., 1]
        dom = -(self.g / self.l) * np.sin(th) - self.b * om + a[..., 0] / (self.m * self.l ** 2)
        return np.stack([th + self.dt * om, om + self.dt * dom], -1)

    def step_torch(self, s, a):
        """The same map in torch, so a planner can use the TRUE dynamics.

        Lesson 3 needs this as a positive control: run the identical planner on
        the true system and on the learned one, and any difference between them
        is model error rather than a quirk of the planner.
        """
        import torch
        th, om = s[..., 0], s[..., 1]
        dom = -(self.g / self.l) * torch.sin(th) - self.b * om + a[..., 0] / (self.m * self.l ** 2)
        return torch.stack([th + self.dt * om, om + self.dt * dom], -1)

    def sample_states(self, n, rng, omega_max=1.5):
        """Uniform over angle, and over angular velocity up to omega_max.

        The default is enough for Lessons 1-2, which only ever roll out from
        rest-like states. Lesson 3 swings the pendulum up and reaches far higher
        speeds, so it widens this - a model trained on the narrow range is not
        wrong, it is simply being asked about states it has never seen, and that
        confounds the question Lesson 3 is asking.
        """
        return np.stack([rng.uniform(-np.pi, np.pi, n),
                         rng.uniform(-omega_max, omega_max, n)], -1)

    def sample_actions(self, n, k, rng):
        return rng.uniform(-2.0, 2.0, (n, k, 1))


class Lorenz:
    """The Lorenz system, integrated with RK4. Autonomous - no action.

    At the classical parameters this is chaotic, with a largest Lyapunov
    exponent near 0.906. Lesson 2 measures that number rather than quoting it,
    and then checks whether rollout error grows at exactly that rate.
    """
    state_dim, action_dim = 3, 0

    def __init__(self, sigma=10.0, rho=28.0, beta=8 / 3, dt=0.01):
        self.sigma, self.rho, self.beta, self.dt = sigma, rho, beta, dt

    def _f(self, s):
        x, y, z = s[..., 0], s[..., 1], s[..., 2]
        return np.stack([self.sigma * (y - x), x * (self.rho - z) - y, x * y - self.beta * z], -1)

    def step(self, s, a=None):
        dt, f = self.dt, self._f
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def sample_states(self, n, rng, burn_in=2000):
        """Random points, then burned in so they lie on the attractor.

        Skipping the burn-in is a real and easy mistake: off-attractor
        transients have completely different error growth, and mixing them in
        is enough to hide the exponential rate Lesson 2 is looking for.
        """
        s = rng.standard_normal((n, 3)) * 2 + np.array([0.0, 0.0, 25.0])
        return rollout(self, s, burn_in)[:, -1]

    def sample_actions(self, n, k, rng):
        return np.zeros((n, k, 0))


def rollout(system, s0, k, actions=None):
    """Roll the TRUE dynamics forward k steps. Returns (n, k, state_dim)."""
    s, out = np.asarray(s0, dtype=float).copy(), []
    for i in range(k):
        a = None if actions is None else actions[:, i]
        s = system.step(s, a)
        out.append(s.copy())
    return np.stack(out, 1)
