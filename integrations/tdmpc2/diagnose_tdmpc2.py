"""Run wm.diagnose() on a published TD-MPC2 world model.

TD-MPC2 has no decoder, so its open-loop prediction error lives in latent
space: roll the learned dynamics forward from enc(o_t) under the actions the
agent actually took, and compare against enc(o_{t+k}) - the same target its
consistency loss uses. That loss carries weight 20, the largest term in the
objective, and is trained over a 3-step horizon. Nothing in the codebase turns
it into a reported number, at 3 steps or at any other.

This script does. It uses TD-MPC2's own code, its own released checkpoint and
its own environments; see README.md in this directory for the pinned versions
and the exact commands.

    TDMPC2_DIR=/path/to/tdmpc2/tdmpc2 python diagnose_tdmpc2.py \
        task=mt30 model_size=1 checkpoint=/path/to/mt30-1M.pt
"""
import sys, os
TD = os.environ.get("TDMPC2_DIR")
if TD is None or not os.path.isdir(TD):
    raise SystemExit("set TDMPC2_DIR to the tdmpc2/tdmpc2 directory of a clone of\n"
                     "https://github.com/nicklashansen/tdmpc2 (see README.md here)")
HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, TD)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
os.chdir(TD)
import numpy as np, torch, hydra
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
import wm

K = 20        # rollout length to diagnose, well past their 3-step training horizon
STARTS = 64   # start points sampled from each episode
EPISODES = 10 # per-task percentages move by tens of points at 3; at 10 they settle
DEFAULT_TASKS = ["cartpole-swingup", "walker-walk", "cheetah-run", "finger-spin",
                 "reacher-easy", "cup-catch", "pendulum-swingup", "hopper-stand"]


def episode(env, agent, task_idx, steps):
    """One episode driven by TD-MPC2's own planner."""
    obs, O, A = env.reset(task_idx=task_idx), [], []
    for t in range(steps):
        a = agent.act(obs, t0=(t == 0), eval_mode=True, task=task_idx)
        O.append(obs); A.append(a)
        obs, r, done, _ = env.step(a)
        if done:
            break
    O.append(obs)
    return torch.stack(O).cuda().float(), torch.stack(A).cuda().float()


def rollout_pair(model, O, A, task_idx, k, starts):
    """(pred, true, nochange) latent trajectories, all shaped (n, k, latent)."""
    n_avail = len(A) - k
    idx = np.linspace(0, n_avail - 1, min(starts, n_avail)).astype(int)
    with torch.no_grad():
        Z = model.encode(O, task_idx)
        true = torch.stack([Z[i + 1:i + 1 + k] for i in idx])
        z, preds = Z[idx], []
        for j in range(k):
            z = model.next(z, torch.stack([A[i + j] for i in idx]), task_idx)
            preds.append(z)
        pred = torch.stack(preds, dim=1)
    Z0 = Z[idx].cpu().numpy()
    return pred.cpu().numpy(), true.cpu().numpy(), np.repeat(Z0[:, None, :], k, axis=1)


@hydra.main(config_name="config", config_path=os.environ["TDMPC2_DIR"], version_base=None)
def main(cfg):
    cfg = parse_cfg(cfg)
    # Their own seeding. It does not make a rerun bit-identical: MPPI amplifies
    # GPU floating-point nondeterminism into different action sequences, so two
    # runs of this file visit different states. Measured, the per-task figures
    # move by at most 1.8 points and 0.6 on average, and the median across
    # tasks did not move at all - see README.md.
    set_seed(cfg.seed)
    env = make_env(cfg)
    agent = TDMPC2(cfg); agent.load(cfg.checkpoint)
    model = agent.model
    tasks = list(cfg.get("diagnose_tasks", DEFAULT_TASKS))
    ckpt = os.path.basename(cfg.checkpoint)

    print("\n  checkpoint      %s" % ckpt)
    print("  latent_dim      %d   (TD-MPC2 predicts latents; there is no decoder)" % cfg.latent_dim)
    print("  trained horizon %d   (consistency_coef %s - the largest term in their loss)"
          % (cfg.horizon, cfg.consistency_coef))
    print("  diagnosing %d tasks, %d-step open-loop rollouts from %d starts each\n"
          % (len(tasks), K, STARTS), flush=True)

    rows, store = [], {}
    for task in tasks:
        task_idx = cfg.tasks.index(task) if cfg.multitask else None
        Ps, Ts, Ns = [], [], []
        for _ in range(EPISODES):
            O, A = episode(env, agent, task_idx, cfg.episode_length)
            if len(A) - K <= 0:
                continue
            P, T, N = rollout_pair(model, O, A, task_idx, K, STARTS)
            Ps.append(P); Ts.append(T); Ns.append(N)
        if not Ps:
            print("  %-22s every episode too short, skipped" % task); continue
        # Pool start points across episodes; each is an independent rollout.
        P, T, N = np.concatenate(Ps), np.concatenate(Ts), np.concatenate(Ns)
        d = wm.diagnose(P, T)
        # Per-rollout error, kept in full. Lesson 2 of this repository is that
        # the mean over trajectories tracks a runaway minority rather than the
        # typical rollout, so the summary must not be baked in here - storing
        # the matrix lets any summary be recomputed later, and lets the figures
        # show the spread instead of one line through it.
        err = np.linalg.norm(P - T, axis=-1).astype(np.float32)      # (n, k)
        still = np.linalg.norm(N - T, axis=-1).astype(np.float32)    # (n, k)
        e_m, e_n = np.median(err, axis=0), np.median(still, axis=0)
        h = d["horizon"]
        # Two readings at the planner's own horizon, neither of which needs a
        # tolerance to be chosen. `still` is the distance the latent actually
        # travelled, so the ratio says what share of the real motion the error
        # already eats - and `loses` counts the starts where it eats all of it,
        # which is the question a planner is really asking.
        H = min(cfg.horizon, K) - 1
        loses = 100.0 * float((err[:, H] >= still[:, H]).mean())
        rows.append((task, h["p50"], h["p5"], h["p95"], loses,
                     int((e_m < e_n).sum()), e_m[H] / max(e_n[H], 1e-12)))
        store["err__" + task] = err
        store["still__" + task] = still
        store["horizon__" + task] = np.array([h["p5"], h["p50"], h["p95"]])
        # A 317M model takes tens of minutes per task, so this line has to
        # arrive when the task finishes rather than when the buffer fills.
        print("  %-22s %d episodes, %d rollouts" % (task, len(Ps), len(P)), flush=True)

    print("\n  %-20s %-22s %-22s %s"
          % ("task", "at k=%d, share of starts" % cfg.horizon,
             "error at k=%d, %% of" % cfg.horizon, "usable horizon"))
    print("  %-20s %-22s %-22s %s"
          % ("", "worse than standing still", "the latent's real motion", "median [5th, 95th]"))
    print("  " + "-" * 88)
    for t, p50, p5, p95, loses, w, ratio in rows:
        print("  %-20s %-22s %-22s %s"
              % (t, "%.0f%%" % loses, "%.0f%%" % (100 * ratio),
                 "%.0f  [%.0f, %.0f]" % (p50, p5, p95)))
    L = sorted(r[4] for r in rows)
    print("\n  median across tasks: %.0f%% of starts are better served by assuming"
          % (0.5 * (L[len(L) // 2] + L[(len(L) - 1) // 2])))
    print("  nothing changed, at the horizon TD-MPC2 plans over. Range %.0f-%.0f%%."
          % (L[0], L[-1]))
    print("  It rolls this model forward %d steps to score every candidate action"
          % cfg.horizon)
    print("  sequence, on every task, at every model size.")
    out = os.path.join(HERE, "results_%s.npz" % ckpt.replace(".pt", ""))
    np.savez_compressed(out, **store)
    print("  wrote %s" % out)


main()
