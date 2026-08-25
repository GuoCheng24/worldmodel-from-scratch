"""An adapter that makes a Gymnasium MuJoCo environment look like the hand-written
systems in `wm.systems`, so every measurement from Lessons 1-4 runs unchanged.

The interface those lessons need is small: set an exact state, apply an action,
read the next state, and do it deterministically. MuJoCo supports all of that
through `set_state`, which is what makes these environments usable as ground
truth rather than only as a training loop.

Two practical notes, both of which cost time to discover:

1. `MUJOCO_GL` must be set BEFORE mujoco is imported, and on a headless machine
   the only value that works here is `egl`. With it unset, rendering raises
   `mujoco.FatalError: an OpenGL platform library has not been loaded`; with
   `glfw` it does the same; with `osmesa` it fails inside PyOpenGL. Worse, one
   wrong combination aborts the process outright rather than raising, so a
   try/except around the render call does not save you. `ensure_headless_gl()`
   below sets it, and must be called before the import.

2. None of the measurements in Lessons 1-4 need rendering at all. They need
   states. If you are only measuring, skip the graphics stack entirely.
"""
import os
import numpy as np

__all__ = ["ensure_headless_gl", "GymSystem", "AVAILABLE"]

AVAILABLE = ("InvertedPendulum-v5", "Reacher-v5", "HalfCheetah-v5",
             "Hopper-v5", "Walker2d-v5", "Pusher-v5")


def ensure_headless_gl(backend="egl"):
    """Set MUJOCO_GL if it is unset. Call before importing mujoco or gymnasium."""
    os.environ.setdefault("MUJOCO_GL", backend)
    return os.environ["MUJOCO_GL"]


def _require():
    try:
        import gymnasium  # noqa: F401
        import mujoco     # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Lessons 5 and 6 need Gymnasium and MuJoCo, which Lessons 1-4 do not:\n"
            "    pip install gymnasium mujoco\n"
            "and on a headless machine, before importing either:\n"
            "    export MUJOCO_GL=egl\n"
            "(wm.ensure_headless_gl() does that for you.)\n"
            "Original error: %s" % e)


class GymSystem:
    """A MuJoCo environment with the same surface as wm.Pendulum / wm.Lorenz.

    State is (qpos, qvel) concatenated, which is the full simulator state - not
    the observation, which for several of these environments omits coordinates.
    Measuring model error against the observation instead would quietly change
    the question being asked.

    Stepping is one environment at a time because MuJoCo is not batched here, so
    keep n modest: a few hundred trajectories of a hundred steps is seconds.
    """

    def __init__(self, env_id, seed=0):
        _require()
        import gymnasium as gym
        if env_id not in AVAILABLE:
            raise ValueError("unknown env %r; known: %s" % (env_id, ", ".join(AVAILABLE)))
        self.env_id = env_id
        self.env = gym.make(env_id)
        self.env.reset(seed=seed)
        u = self.env.unwrapped
        self.nq, self.nv = int(u.model.nq), int(u.model.nv)
        self.state_dim = self.nq + self.nv
        self.action_dim = int(np.prod(self.env.action_space.shape))
        self.dt = float(u.dt)
        self._rng = np.random.default_rng(seed)
        self._lo = self.env.action_space.low
        self._hi = self.env.action_space.high

    def _set(self, s):
        self.env.unwrapped.set_state(np.asarray(s[:self.nq], dtype=float),
                                     np.asarray(s[self.nq:], dtype=float))

    def _read(self):
        u = self.env.unwrapped
        return np.concatenate([u.data.qpos.copy(), u.data.qvel.copy()])

    def step(self, s, a):
        """Batched over the leading axis, by looping. Deterministic."""
        s = np.atleast_2d(np.asarray(s, dtype=float))
        a = np.atleast_2d(np.asarray(a, dtype=float))
        out = np.empty_like(s)
        for i in range(len(s)):
            self._set(s[i])
            self.env.unwrapped.do_simulation(np.clip(a[i], self._lo, self._hi),
                                             self.env.unwrapped.frame_skip)
            out[i] = self._read()
        return out

    def sample_states(self, n, rng=None, spread=1.0):
        """States reached by a random policy, so they lie on the reachable set.

        Sampling qpos/qvel uniformly instead would put most of them in
        configurations the simulator never visits - self-intersecting arms,
        cheetahs on their backs - and a model's error there says nothing about
        the states you will actually roll out from.
        """
        rng = rng or self._rng
        states = []
        while len(states) < n:
            self.env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
            for _ in range(int(rng.integers(1, 60))):
                self.env.step(self.env.action_space.sample() * spread)
            states.append(self._read())
        return np.stack(states[:n])

    def sample_actions(self, n, k, rng=None, spread=1.0):
        rng = rng or self._rng
        return rng.uniform(self._lo, self._hi, (n, k, self.action_dim)) * spread

    def close(self):
        self.env.close()
