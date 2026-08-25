# Diagnosing a published TD-MPC2 world model

`wm.diagnose()` run on TD-MPC2's own released checkpoints, using TD-MPC2's own
code and its own environments. Nothing in this directory modifies their
repository; the whole thing is a caller.

![](tdmpc2-diagnosis.png)

## The question

TD-MPC2 plans by rolling its learned dynamics forward and scoring the result:
`_estimate_value` calls `model.next` for `cfg.horizon` steps, then bootstraps
with `Q`. `cfg.horizon` is `3`, on every task. So the planner's decisions rest
on 3-step open-loop latent prediction — and how good that is, per task, decides
how much the plan is worth.

TD-MPC2 does log a related scalar. `consistency_loss` is a `rho`-discounted
mean squared error over those same 3 steps, taken on replay batches, sent to
wandb but not to the console (`CONSOLE_FORMAT` carries only iteration, episode,
step, reward, success and elapsed time) and not to the saved CSV (`step` and
`episode_reward` only). It is a training loss: never divided by how far the
latent actually moved, so it has no readable scale, stopping at the training
horizon, and not broken out per task at evaluation.

## What it measures here

TD-MPC2 has no decoder, so open-loop error lives in latent space. From
`enc(o_t)`, roll `model.next` forward under the actions the agent actually
took, and compare against `enc(o_{t+k})` — the same target the consistency loss
uses. Two readings come out:

- **usable horizon** — how many steps before the error crosses a tolerance
  (`wm.diagnose`'s default is the 10th percentile of the final-step error, so
  that most trajectories cross it inside the window), reported as a
  distribution over start points rather than as a single number.
- **error as a percentage of the latent's real motion** — the model's error at
  step k divided by `||enc(o_{t+k}) - enc(o_t)||`, the distance the latent
  actually travelled. This needs no tolerance to be chosen. At 100% the
  prediction is worth exactly as much as predicting no change at all.

Measured over 8 dm_control tasks from the mt30 set, 10 episodes each, 64 start
points per episode (640 rollouts per task), 20 open-loop steps.

## What it found

**At the horizon TD-MPC2 actually plans over, prediction quality varies about
tenfold across tasks — for one model.**

| task | error at k=3, mt30-1M | error at k=3, mt30-48M | usable horizon, mt30-48M |
|---|---|---|---|
| cartpole-swingup | 90% | 68% | 14 |
| walker-walk | 115% | 12% | 5 |
| cheetah-run | 63% | 52% | 4 |
| finger-spin | 58% | 7% | 13 |
| reacher-easy | 41% | 30% | 11 |
| cup-catch | 161% | 24% | 2 |
| pendulum-swingup | 14% | 10% | 4 |
| hopper-stand | 116% | 19% | 9 |
| **median** | **77%** | **22%** | **7** |

On the 1M model, three tasks are **above 100%**: at the horizon the planner
scores its candidate actions over, the prediction error exceeds the distance
the latent moves, so the rollout carries less information than assuming nothing
changes. Four of the eight are also beaten by predicting no change at the very
first step. Scaling to 48M fixes that — 0 of 8 beaten at one step, and the
worst case falls from 161% to 68% — but the spread across tasks stays wide,
7% to 68%.

The usable horizon behaves the same way. For mt30-48M the median runs from 2
steps (`cup-catch`) to 14 (`cartpole-swingup`). On `cup-catch` the predictions
are past tolerance before the planner's own 3-step lookahead finishes.

None of this is visible in the reward curve, and none of it is a number the
codebase produces.

## Reproducing it

The pinned environment is TD-MPC2's own `docker/environment.yaml`:

```
python 3.11   torch 2.7.1   gymnasium 0.29.1   mujoco 3.1.2
dm-control 1.0.16   tensordict 0.8.3   torchrl 0.8.1   numpy 1.26.4
```

plus `hydra-core 1.3.2`, `hydra-submitit-launcher`, `omegaconf`, `termcolor`,
`pandas`, `imageio`, `kornia`, `h5py`. A CUDA device is required —
`evaluate.py` asserts it.

```bash
git clone https://github.com/nicklashansen/tdmpc2
export TDMPC2_DIR=$PWD/tdmpc2/tdmpc2
export MUJOCO_GL=egl

# checkpoints: https://huggingface.co/nicklashansen/tdmpc2 -> multitask/
python diagnose_tdmpc2.py task=mt30 model_size=1  checkpoint=/path/mt30-1M.pt
python diagnose_tdmpc2.py task=mt30 model_size=48 checkpoint=/path/mt30-48M.pt
python make_figure.py
```

Measured against tdmpc2 at `e9f5932` (2026-07-13). The multitask checkpoints
load through TD-MPC2's own `TDMPC2.load()` with `strict=True`, so every
parameter is matched by name and shape — the model being diagnosed is provably
the released one.

## A second finding: 14 released checkpoints are in an older layout

Most of the released single-task checkpoints load and run. `walker-walk-1.pt`
evaluates to **987.2** against the **982.9** recorded for that seed in the
repository's own `results/tdmpc2/walker-walk.csv`.

Fourteen do not. They store each MLP as a flat `[Linear, LayerNorm, Mish, ...]`
sequence, where the current code builds the same computation from
`NormedLinear` modules that fold the LayerNorm into the Linear.
`api_model_conversion()` exists for exactly this kind of migration but only
renames the Q-ensemble, so:

```
$ python evaluate.py task=cartpole-swingup checkpoint=.../cartpole-swingup-1.pt
RuntimeError: Error(s) in loading state_dict for WorldModel:
    Missing key(s): "_encoder.state.0.ln.weight", ... (28 keys)
```

Surveying all 312 single-task checkpoints — the parameter names sit in plain
text in the first 256 KB of each file, so a ranged GET is enough to classify one
without downloading it — gives 298 in the current layout and these 14 in the
old one:

```
dmcontrol/cartpole-balance-1.pt        dmcontrol/pendulum-swingup-2.pt
dmcontrol/cartpole-balance-2.pt        dmcontrol/walker-walk-backwards-1.pt
dmcontrol/cartpole-swingup-1.pt        maniskill2/pick-cube-2.pt
dmcontrol/cartpole-swingup-sparse-1.pt maniskill2/stack-cube-1.pt
dmcontrol/cartpole-swingup-sparse-2.pt maniskill2/stack-cube-2.pt
dmcontrol/humanoid-run-3.pt            myosuite/myo-hand-pose-2.pt
dmcontrol/humanoid-stand-3.pt
dmcontrol/humanoid-walk-3.pt
```

```bash
curl -sL -r 0-262143 "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/$f" \
  | grep -ac '_encoder\.state\.0\.ln\.weight'      # 0 means the old layout
```

The full classification of all 312 files is in `checkpoint_layouts.tsv` here.

They are scattered across three of the four families and across seeds of the
same task, which reads like a partial re-upload rather than a design change.
Across all 134 commits in the public history there are only two distinct `mlp`
definitions and both build `NormedLinear`, so no published version of this code
produces the layout those 14 use.

`convert_ckpt.py` here remaps the layout exactly — every name and shape matches
the model the current code builds, verified both ways, with nothing dropped.
**It is not a fix.** The converted `cartpole-swingup-1.pt` scores 233.7 where
`results/tdmpc2/cartpole-swingup.csv` records 866.2 for that seed, and its
reward head correlates 0.68 with the true reward. Since the parameter layout is
recovered exactly, what remains must be in the parameter-free operations, which
the released code does not contain. Use it to reproduce this diagnosis, not to
run these checkpoints.

## Caveats

- 8 of the 30 mt30 tasks, 10 episodes each.
- **Reruns are not bit-identical.** MPPI amplifies GPU floating-point
  nondeterminism into different action sequences, so two runs of this script
  with the same seed visit different states. Measured on mt30-48M, the per-task
  percentages moved by at most 1.8 points and 0.6 on average, and the median
  across tasks did not move at all (22% both times; the range read 8-70% and
  7-68%). Read the individual percentages to the nearest few points and the
  tenfold spread as the finding.
- Trajectories come from the agent's own planner, so this is prediction quality
  on the states the agent actually visits — the relevant distribution for
  planning, and not the same as replay-buffer error.
- Latent error is measured against `enc(o_{t+k})`, which is what TD-MPC2 trains
  against. A drifting encoder would show up here as dynamics error; the two are
  not separable without a decoder, which TD-MPC2 does not have.
