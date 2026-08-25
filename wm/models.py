"""A world model, small enough to read in one sitting.

A world model answers one question: given where you are and what you do, where
do you end up? That is it. Everything else in the literature - latents,
recurrence, tokenizers, diffusion heads - is a way of making that question
answerable when the state is an image. Here the state is a vector, so the
answer fits in a few lines, and nothing is hidden.

The one design choice worth arguing about is that the network predicts the
*change* in state rather than the next state. Predicting the next state makes
the identity map a strong local optimum: with dt small, s_{t+1} is almost s_t,
so a model that learns "copy the input" already scores well and has little
gradient pressure to learn the dynamics. Lesson 1 measures that difference.
"""
import numpy as np
import torch
import torch.nn as nn

__all__ = ["WorldModel", "fit", "imagine"]


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, depth=2, residual=True):
        super().__init__()
        self.residual = residual
        layers, d = [], state_dim + action_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, state_dim)]
        self.net = nn.Sequential(*layers)
        # Input normalisation, filled in by fit(). Kept as buffers so that
        # saving the model saves the statistics with it - forgetting this is a
        # classic source of "it worked in the notebook and not on reload".
        self.register_buffer("mu", torch.zeros(state_dim))
        self.register_buffer("sd", torch.ones(state_dim))

    def forward(self, s, a=None):
        x = (s - self.mu) / self.sd
        if a is not None and a.shape[-1] > 0:
            x = torch.cat([x, a], -1)
        out = self.net(x)
        return s + out * self.sd if self.residual else out


def fit(model, S, A, S_next, steps=6000, batch=512, lr=1e-3, device="cpu", log_every=0):
    """One-step supervised training. Returns the final loss."""
    model.to(device)
    S = torch.as_tensor(S, dtype=torch.float32, device=device)
    A = torch.as_tensor(A, dtype=torch.float32, device=device)
    Y = torch.as_tensor(S_next, dtype=torch.float32, device=device)
    with torch.no_grad():
        model.mu.copy_(S.mean(0)); model.sd.copy_(S.std(0).clamp_min(1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss = torch.tensor(float("nan"))
    for i in range(steps):
        j = torch.randint(0, len(S), (batch,), device=device)
        loss = ((model(S[j], A[j]) - Y[j]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log_every and i % log_every == 0:
            print("    step %6d   loss %.3e" % (i, loss.item()))
    return float(loss.detach())


@torch.no_grad()
def imagine(model, s0, k, actions=None, device="cpu"):
    """Roll the MODEL forward k steps, feeding its own output back in.

    This feedback is the whole subject of Lesson 2. Training only ever showed
    the model true states; here it must consume its own predictions, and the
    gap between those two distributions is where rollouts go wrong.
    """
    model.to(device).eval()
    s = torch.as_tensor(s0, dtype=torch.float32, device=device)
    A = None if actions is None else torch.as_tensor(actions, dtype=torch.float32, device=device)
    out = []
    for i in range(k):
        s = model(s, None if A is None else A[:, i])
        out.append(s.clone())
    return torch.stack(out, 1).cpu().numpy()


def make_dataset(system, n, k, rng):
    """n trajectories of length k -> flat (s, a, s_next) triples."""
    s0 = system.sample_states(n, rng)
    A = system.sample_actions(n, k, rng)
    from .systems import rollout
    T = rollout(system, s0, k, A if system.action_dim else None)
    S = np.concatenate([s0[:, None], T[:, :-1]], 1).reshape(-1, system.state_dim)
    # An action-free system gives A shape (n, k, 0); numpy cannot infer -1
    # against a zero-width axis, so build the empty block explicitly.
    A_flat = A.reshape(-1, system.action_dim) if system.action_dim else np.zeros((len(S), 0))
    return S, A_flat, T.reshape(-1, system.state_dim)
