"""worldmodel-from-scratch - build a world model, then find out where it breaks."""
from .systems import Pendulum, Lorenz, rollout
from .models import WorldModel, fit, imagine, make_dataset
from .planning import cem_mpc
from .gym_systems import ensure_headless_gl, GymSystem, AVAILABLE as GYM_ENVS
from .analysis import (rollout_error, lyapunov, fit_growth, usable_horizon,
                       lipschitz, textbook_bound, fit_growth_ci,
                       summarise, rank_fidelity,
                       effective_rank, probe_r2)

__version__ = "0.1.0"
__all__ = ["Pendulum", "Lorenz", "rollout", "WorldModel", "fit", "imagine",
           "make_dataset", "rollout_error", "lyapunov", "fit_growth", "usable_horizon",
           "lipschitz", "textbook_bound", "fit_growth_ci", "summarise",
           "rank_fidelity", "effective_rank", "probe_r2", "cem_mpc",
           "ensure_headless_gl", "GymSystem", "GYM_ENVS"]
