"""CEM model-predictive control: plan with a model, execute one step, replan.

This is the simplest planner that actually needs a world model to be good, and
it is the setting where "how far can I trust the rollout" stops being a
diagnostic and starts costing you reward. At every timestep it samples action
sequences, scores them by rolling them through the dynamics, keeps the best
few, and refits a Gaussian to them - a few rounds of that and the mean sequence
is a plan. Only its first action is executed, then the whole thing repeats.

The planning horizon is a free parameter, and Lesson 3 is about what happens
when you sweep it.
"""
import torch

__all__ = ["cem_mpc"]


@torch.no_grad()
def cem_mpc(plan_step, env_step, reward_fn, s0, horizon, steps, u_max=3.0,
            n_candidates=192, iters=4, n_elite=24, init_std=None, action_dim=1):
    """Run CEM-MPC for `steps` timesteps from each row of s0.

    action_dim             width of the action vector; 1 unless you say otherwise
    plan_step(s, a) -> s'  the dynamics the planner IMAGINES with (your model)
    env_step(s, a)  -> s'  the dynamics that actually ADVANCE the world
    reward_fn(s, a) -> r   per-transition reward, scored on the real state
    horizon                how many steps the planner imagines before deciding

    The two dynamics arguments are separate and both required, on purpose. Pass
    the model for both and you are scoring the planner inside its own dream:
    the reward is computed on states the model invented, so a WORSE model gets a
    BETTER score - it simply imagines the pendulum already balanced. We measured
    a model with a one-step usable horizon "achieving" three times the reward of
    a planner given the exact true dynamics. Nothing in the code complains.

    This is not a hypothetical footnote. It is the single easiest way to publish
    a model-based result that cannot be reproduced on hardware, and it is why
    this function will not let you supply one callable for both roles.

    Returns (mean reward per step, final states, full state trajectory).

    The elite set is refit each iteration from the elites of the previous one,
    and the mean is warm-started from the previous timestep's plan shifted by
    one. Both matter: without the shift the planner rediscovers the same plan
    from scratch every step and a short horizon looks far worse than it is.
    """
    device = s0.device
    N, D = s0.shape
    A = int(action_dim)
    init_std = u_max * 0.6 if init_std is None else init_std
    s = s0.clone()
    mu = torch.zeros(N, horizon, A, device=device)
    total = torch.zeros(N, device=device)
    traj = [s.clone()]
    for _ in range(steps):
        sd = torch.full_like(mu, init_std)
        for _ in range(iters):
            noise = torch.randn(N, n_candidates, horizon, A, device=device)
            a = (mu[:, None] + sd[:, None] * noise).clamp(-u_max, u_max)
            ss = s[:, None].expand(N, n_candidates, D).reshape(-1, D)
            R = torch.zeros(N * n_candidates, device=device)
            for h in range(horizon):
                uh = a[:, :, h].reshape(-1, A)
                ss = plan_step(ss, uh)              # imagined
                R = R + reward_fn(ss, uh)
            idx = R.view(N, n_candidates).topk(n_elite, dim=1).indices
            elites = torch.gather(a, 1, idx[..., None, None].expand(N, n_elite, horizon, A))
            mu, sd = elites.mean(1), elites.std(1) + 1e-3
        u = mu[:, 0].clamp(-u_max, u_max)
        s = env_step(s, u)                          # real
        total = total + reward_fn(s, u)
        traj.append(s.clone())
        mu = torch.cat([mu[:, 1:], torch.zeros(N, 1, A, device=device)], 1)
    return total / steps, s, torch.stack(traj, 1)
