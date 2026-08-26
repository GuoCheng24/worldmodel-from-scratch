# Diagnosing a published TD-MPC2 world model

`wm.diagnose()` run on TD-MPC2's own released checkpoints, using TD-MPC2's own
code and its own environments. Nothing in this directory modifies their
repository; the whole thing is a caller.

![Bars of how often TD-MPC2's three-step prediction is worse than assuming nothing changed, per task at three model sizes, beside the spread of usable horizons](tdmpc2-diagnosis.png)

![Eight curves of open-loop error as a percentage of the latent's real motion, crossing 100 percent at widely different steps](tdmpc2-curves.png)

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
uses. The comparison that decides whether a plan is worth anything is against
the most trivial baseline there is: **assume the state did not change.** If the
prediction is further from the truth than `enc(o_t)` itself, the rollout has
carried the planner no information at all.

Two readings come out of that, and neither needs a tolerance to be chosen:

- **the share of starts where the model loses to standing still**, at the
  planner's own horizon. Bounded, no division, and it is the question a planner
  is really asking.
- **how many steps it keeps winning**, per start — which turns out not to be
  one number per model, or even per task.

Measured over 8 dm_control tasks from the mt30 set, 10 episodes each, 64 start
points per episode (640 rollouts per task), 20 open-loop steps. Every summary
below is a median or a percentile: Lesson 2 of this repository is that the mean
over rollouts describes a runaway minority rather than the typical one, and
that applies here as much as it does there.

## What it found

**At the horizon TD-MPC2 rolls its model forward to score actions, the share of
starts where the prediction is beaten by assuming nothing changed:**

| task | mt30-1M | mt30-48M | mt30-317M |
|---|---|---|---|
| cartpole-swingup | 42% | 44% | **57%** |
| walker-walk | 55% | 0% | 0% |
| cheetah-run | 22% | 22% | 0% |
| finger-spin | 68% | 0% | 0% |
| reacher-easy | 22% | 5% | 4% |
| cup-catch | 60% | 1% | 7% |
| pendulum-swingup | 8% | 4% | 18% |
| hopper-stand | 79% | 9% | 4% |
| **median** | **48%** | **4%** | **4%** |
| range | 8-79% | 0-44% | 0-57% |

**1M to 48M is a real fix.** On the smallest model, half the starts on a median
task are better served by assuming nothing changed — 79% of them on
`hopper-stand`. Forty-eight times the parameters takes the median to 4%, and
four of the eight tasks to 1% or below.

**48M to 317M is nothing at all.** Another 6.6x the parameters leaves the median
exactly where it was, at 4%.

**And `cartpole-swingup` gets worse at every size: 42%, 44%, 57%.** At 317M, on
the majority of starts, rolling the learned dynamics forward three steps is
worse than not rolling them forward — on the task the planner is being asked to
solve. `pendulum-swingup` regresses too, 8% to 4% to 18%. Whatever is wrong
there is not a capacity problem, and no number the codebase reports would show
it.

**Does any of it matter?** Not the way the table suggests. TD-MPC2's own
`evaluate.py` on the same checkpoints, two episodes per task, against the same
share:

| task | mt30-1M | mt30-48M | mt30-317M |
|---|---|---|---|
| cartpole-swingup | 242 / 42% | 1 / 44% | **754 / 57%** |
| walker-walk | 229 / 55% | 982 / 0% | 972 / 0% |
| cheetah-run | 12 / 22% | 30 / 22% | 840 / 0% |
| finger-spin | 360 / 68% | 942 / 0% | 927 / 0% |
| reacher-easy | 0 / 22% | 978 / 5% | 985 / 4% |
| cup-catch | 464 / 60% | 868 / 1% | 974 / 7% |
| pendulum-swingup | 0 / 8% | 846 / 4% | 890 / 18% |
| hopper-stand | 0 / 79% | 492 / 9% | 729 / 4% |

<sub>return out of 1000 / share of starts where the k=3 prediction loses to
standing still. Returns are in `returns_mt30.tsv`; `summarise.py` recomputes
both columns.</sub>

At 317M, `cartpole-swingup` scores **754** while its three-step prediction loses
on **57%** of starts - the worst prediction in the set, on a task the agent
does. The planner replans every step and bootstraps the tail with a value
function, and that is evidently enough to survive a rollout that is worse than
standing still.

The rank correlation between the two columns is **+0.43, -0.86, -0.12** at 1M,
48M and 317M. It looks convincing at 48M and it is an accident of that model
having exactly two failing tasks; at eight tasks, nothing here supports either
number standing in for the other. **A reward curve does not tell you what the
model knows, and this share does not tell you whether the agent will succeed.**
Both are worth measuring, which is the point.

**The horizon is not one number.** Asking the same question at every step of one
episode, rather than once per model:

<img src="tdmpc2-trust.gif" width="69%" alt="Two dm_control episodes playing, each captioned with how many steps its prediction is still worth rolling out, over a time series of that number">

<sub>mt30-48M, one episode each. The number on each frame is how many steps
ahead the model is still worth rolling out from that exact moment. It collapses
and recovers dozens of times within a single episode, and the two tasks live in
different regimes while sharing every weight.</sub>

Across starts, for mt30-317M, the median runs from 1 step (`cartpole-swingup`)
to 20 (`cheetah-run`), and within a single task the 5th and 95th percentiles are
19 steps apart. TD-MPC2 rolls the model forward the same 3 steps regardless.

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

- 8 of the 30 mt30 tasks, 10 episodes each. Each model size was run once,
  except mt30-48M, which was run twice to measure the spread below.
- **Reruns are not bit-identical.** MPPI amplifies GPU floating-point
  nondeterminism into different action sequences, so two runs of this script
  with the same seed visit different states. Measured on mt30-48M, the per-task
  shares moved by at most 2.8 points and 0.4 on average, and the median across
  tasks did not move. `cartpole-swingup` read 44% in both runs, so its
  regression with scale is not a run-to-run artefact. Read the individual
  percentages to the nearest few points.
- Trajectories come from the agent's own planner, so this is prediction quality
  on the states the agent actually visits — the relevant distribution for
  planning, and not the same as replay-buffer error.
- Latent error is measured against `enc(o_{t+k})`, which is what TD-MPC2 trains
  against. A drifting encoder would show up here as dynamics error; the two are
  not separable without a decoder, which TD-MPC2 does not have.

## Whose work this measures

Nothing here is a new model. The checkpoints, the environments, the encoder and
the dynamics are TD-MPC2's; this directory rolls them forward and writes down
what comes out.

> Hansen, N., Su, H., & Wang, X. (2024). *TD-MPC2: Scalable, Robust World Models
> for Continuous Control.* ICLR 2024. <https://arxiv.org/abs/2310.16828>

The tasks come from dm_control (Apache-2.0). TD-MPC2's code and released weights
are MIT, which is what makes the layout survey in the section above possible to
run and to publish: `results_mt30-*.npz` here are latent prediction errors
measured from those weights, no weights are redistributed, and `returns_mt30.tsv`
is the output of their own `evaluate.py`. Cite them, not this, for the models.

