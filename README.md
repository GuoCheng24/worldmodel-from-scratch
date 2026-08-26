# worldmodel-from-scratch

[![test](https://github.com/GuoCheng24/worldmodel-from-scratch/actions/workflows/test.yml/badge.svg)](https://github.com/GuoCheng24/worldmodel-from-scratch/actions/workflows/test.yml) [![claims](https://img.shields.io/badge/README%20claims-81%2F81%20verified-2e7d5b)](check_claims.py) [![python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Build a world model in an afternoon — then find out how far you can trust it.**

[中文版](README.zh-CN.md) · MIT · no simulator, no MuJoCo, no OpenGL — until Lesson 5

A world model predicts where you end up given where you are and what you do. It
is the engine underneath model-based RL, robot policy evaluation, and the
current generation of world-action models. There are good tutorials on how to
build one. This is about the half nobody teaches: **how far the thing stays
right once it is built, what actually governs that, and how to measure it on a
model you did not train.**

Two things come out of that. **Six lessons** that measure the failure modes on
systems whose true dynamics are known exactly, so "how wrong is the model" has
an answer — and **`wm.diagnose()`**, one call that makes the same measurement on
anything, including somebody else's published model.

<p align="center">
  <img src="figures/imagination-invertedpendulum.gif" width="69%" alt="Two rendered panels side by side: the real cart-pole stays upright while the same model's imagined one falls over, with both pole angles plotted below">
</p>

<sub><b>A model good enough to control a system, wrong about it within about
twenty steps.</b> <b>Left</b> — a planner using a model learned from random play
holds a pole upright, for all 90 steps, in every run. <b>Right</b> — the same
model, same start, same actions, imagining where they lead: its pole passes the
angle this environment calls a failure between step 13 and step 23 across five
runs, and the real one never does - the heading above it turns red on the step
that happens. A one-step loss around 1e-03 says nothing
about that. Both panels are the simulator — the imagined states are pushed back
into MuJoCo and rendered, which a latent-space model cannot do.
<code>make_imagination.py</code>.</sub>

<p align="center">
  <img src="integrations/tdmpc2/tdmpc2-diagnosis.png" width="100%" alt="Bars of how often TD-MPC2's three-step prediction is worse than assuming nothing changed, per task at three model sizes, beside the spread of usable horizons">
</p>

<sub>And the same measurement pointed at a model this repository did not train:
TD-MPC2's released <code>mt30</code> checkpoints, through TD-MPC2's own code.
<b>Left</b> — at the horizon its planner rolls out to, how often the learned
dynamics are beaten by assuming the state did not change. Forty-eight times the
parameters fixes that; another 6.6x does not, and on <code>cartpole-swingup</code>
it gets worse at every size. <b>Right</b> — how many steps the prediction keeps
winning, which is not one number even within a single task.</sub>

---

## Why another one

Working through the existing world-model repositories, the same three things
go wrong, and they show up in their issue trackers verbatim:

| | what happens | here |
|---|---|---|
| **Install** | `mujoco.FatalError: gladLoadGL`, `No module named gym.envs.atari`, `Installation broken again` | **`torch`, `numpy`, `matplotlib`.** That is the whole dependency list for Lessons 1–4. Lessons 5–6 add `gymnasium` and `mujoco`, with the one environment variable that makes them work spelled out and tested. |
| **Train** | days of GPU time, and issues titled *"Can you share checkpoint of trained models?"* | **about 10 s and 50 s** on one GPU, together under a minute. Nothing to download. |
| **Read** | issues titled *"Add project explanation"*, *"Why is dyn_discrete always True?"* | The library you have to read is **946 lines**; each lesson is a standalone 72-321 more. Commented for why rather than what. |
| **Break** | — | **This is the point of the repo.** |

## `wm.diagnose()` — the number the reference implementations do not give you

DreamerV3 logs open-loop prediction as a *picture*: six sequences, five steps of
context, the rest imagined, rendered next to the truth with a green-to-red
border where imagination begins, for a human to look at. The official JAX
implementation and the most-used PyTorch port do this identically, and no scalar
leaves the function. TD-MPC2 does report one — `consistency_loss`, an undivided,
`rho`-discounted MSE over its 3-step training horizon, on replay batches, to
wandb but not to the console and not to the saved CSV. None of the three answers
*how many steps ahead is this model still worth trusting, here.*

So this repo writes it once. Two arrays in, one report out; it never touches
your model, so it does not care what framework it is in or whether it predicts
states, latents or flattened pixels.

```python
lam, sd = wm.lyapunov(lorenz, n=300, steps=4000, rng=rng)   # measured, not assumed
report  = wm.diagnose(pred, true, dt=lorenz.dt, lam=(lam, sd))
print(report)          # also a dict: report["horizon"]["p50"]
```

`pred` and `true` are `(n_trajectories, n_steps, state_dim)` — your rollout and
the truth it was supposed to match, from the same starts under the same actions.
On the Lorenz model from Lesson 2 (the whole run is
[`examples/diagnose_lorenz.py`](examples/diagnose_lorenz.py), and this is its
output verbatim):

```
world model rollout diagnosis   600 trajectories x 900 steps, state dim 3

  usable horizon   tolerance 2.787 (10.6% of typical state size)
      5th 333    median 590    95th 853    spread 2.6x    [3% censored]

  growth shape     residuals exp 0.187 / pow 0.797, ratio 4.26 -> exponential
      4.3 decades of range before saturation

  growth rate      by how you summarise trajectories
      median     0.8899  95%CI [0.8784, 0.9004]   -1.6% vs lambda, agrees within lambda's own +-0.084
      geometric  0.8886  95%CI [0.8745, 0.9034]   -1.8% vs lambda, agrees within lambda's own +-0.084
      mean       0.9645  95%CI [0.9282, 1.0211]   +6.6% vs lambda, agrees within lambda's own +-0.084

  read with care
      - the fitted rate moves by 9% depending on whether you summarise
        trajectories by mean, median or geometric mean
```

The last block is the part that matters. It will not hand you a number it cannot
stand behind: if too many trajectories never cross the tolerance it says the
horizon is censored rather than quoting a percentile; if the exponential and
power-law fits are within 50% of each other it says **ambiguous** instead of
picking a winner; and it reports all three summaries because which one you pick
moves the answer by more than most of the effects people publish. This
repository shipped both of those mistakes before the guards existed — the
guards are the fix, kept.

## It runs on a published model, not only on these

`wm.diagnose()` takes two arrays and knows nothing about your model, so it runs
on somebody else's. Here it is on **TD-MPC2's released `mt30` checkpoints**,
using TD-MPC2's own code and environments, loaded through their own
`load_state_dict` with `strict=True`.

<p align="center">
  <img src="integrations/tdmpc2/tdmpc2-curves.png" width="78%" alt="Eight curves of open-loop error as a percentage of the latent's real motion, crossing 100 percent at widely different steps">
</p>

<sub>How fast the prediction stops paying, per task, for the largest released
model. The shaded band is the three steps the planner rolls out; the dashed line
is where the error equals the distance the latent actually travels.</sub>

TD-MPC2 plans by rolling its learned dynamics forward **3 steps** and scoring
the result — on every task. Whether that plan is worth anything turns on a
comparison against the most trivial baseline there is: at that horizon, how
often is the prediction beaten by **assuming the state did not change**?
Measured across 8 dm_control tasks, 640 rollouts each, at three model sizes:

- **mt30-1M**: on a median task, **48%** of starts. Half of them — and **79%**
  on `hopper-stand`. The rollout is carrying the planner no information.
- **mt30-48M**: **4%**. Forty-eight times the parameters fixes it, and takes
  three of the eight tasks to 1% or below.
- **mt30-317M**: **4%**. Another 6.6x buys nothing at all.

And on `cartpole-swingup` the share **rises at every size — 42%, 44%, 57%**.
Scale fixed the catastrophic cases and then stopped.

Which matters less than it sounds, and that is the interesting part. Run
TD-MPC2's own evaluation on the same checkpoints and `cartpole-swingup` — the
task with the worst prediction at every size — **scores 754 of 1000 at 317M**.
The planner does the task while the rollout it scores actions with is, on most
starts, worse than assuming nothing changed. It replans every step and
bootstraps the tail with a value function, and that is evidently enough.

So neither number stands in for the other. Across these eight tasks the rank
correlation between return and this share is **+0.43, −0.86 and −0.12** at 1M,
48M and 317M: no stable relationship, in either direction. A reward curve does
not tell you what the model knows, and this does not tell you whether the agent
will succeed — the same thing the animation at the top of this file shows, on a
system small enough to draw.

<p align="center">
  <img src="integrations/tdmpc2/tdmpc2-trust.gif" width="69%" alt="Two dm_control episodes playing, each captioned with how many steps its prediction is still worth rolling out, over a time series of that number">
</p>

<sub>Nor is it one number per model. Asked at every step of a single episode —
how many steps ahead is the model still worth rolling out, from this exact
moment — the answer collapses and recovers dozens of times, and two tasks
sharing every weight live in different regimes.</sub>

TD-MPC2 does log `consistency_loss` — an undivided, `rho`-discounted MSE over
those same 3 steps, on replay batches, to wandb but not to the console or the
saved CSV. It is a training loss with no readable scale, and nothing in the
codebase answers *how many steps ahead is this model worth trusting, here*.

## The same disease, in a domain with nothing to do with robots

A model that turns a pre-contrast MRI slice into the post-contrast one is
predicting how a system evolves after an intervention, and that literature has
started to say so - the framing has moved from image translation to contrast
*kinetics*. So the first question here transfers: **what does the most trivial
predictor already score?**

<p align="center">
  <img src="integrations/dce-mri/dce-baselines.png" width="100%" alt="Bars of SSIM for three predictors scored over the whole slice and inside the lesion box, and three lesion-box variants that barely differ">
</p>

On public breast DCE-MRI, a 2M-parameter U-Net beats handing back the input by
**+0.07 SSIM and +4.1 dB** on the whole slice. **Inside the lesion - the few
hundred pixels the contrast agent is injected for - it draws level with a
256-entry lookup table** that sees one voxel and no context. Weighting the loss
in the lesion by fifty does not move it; five times the data only brings it up
to that lookup table, not past it.

A breast MRI slice is mostly fat, muscle, air and chest wall, none of which
enhance. A global metric is dominated by the part of the image where doing
nothing is right, so it cannot separate a model that predicts enhancement from
one that preserves anatomy — the same shape as a TD-MPC2 planner that does the
task while its rollout is worse than standing still.
**[integrations/dce-mri/](integrations/dce-mri/)** has the data fetcher, the
protocol, a null control reported as a null, and the one indexing gotcha that
silently invalidates the whole thing if you skip it.

Rerunning it takes two commands: **[integrations/tdmpc2/](integrations/tdmpc2/)**
has the pinned environment, the exact invocations, and a second finding — 14 of
the 312 released *single-task* checkpoints are stored in an older parameter
layout and raise a `RuntimeError` with the current code, one no commit in the
public history produces. The other 298 load and run.

## Quick start

```bash
git clone https://github.com/GuoCheng24/worldmodel-from-scratch
cd worldmodel-from-scratch && pip install -r requirements.txt

python lessons/01_a_world_model_in_50_lines.py     # 11 s
python lessons/02_why_rollouts_drift.py            # 56 s
python lessons/03_planning_with_a_model_you_do_not_trust.py   # 74 s
python lessons/04_latent_models_and_collapse.py               # 31 s

# Lessons 5-6 additionally need `pip install gymnasium mujoco`
MUJOCO_GL=egl python lessons/05_real_environments.py          # 65 s
MUJOCO_GL=egl python lessons/06_a_real_robot_arm.py           # 67 s
python make_figures.py                             # redraws the plots above
python make_visuals.py                             # redraws the animations
MUJOCO_GL=egl python make_imagination.py           # 40 s, the animation at the top
```

<sub>Those times are on one GPU. Lessons 1-4 need nothing beyond the three packages
above and run on a CPU too, at a cost that is real and not uniform: measured on
eight CPU threads, Lessons 1 and 4 take 3x longer, Lesson 3 6x and Lesson 2 10x,
so the four together are about nineteen minutes rather than three.</sub>

Every number quoted below is produced by those two scripts. To check that
claim rather than take it:

```bash
# with the three packages above, Lessons 1-4:
for f in lessons/0[1-4]*.py; do python "$f" > "/tmp/$(basename $f .py).txt"; done
python check_claims.py --lessons=1,2,3,4 /tmp/0*.txt     # 58/58

# all six, once gymnasium and mujoco are installed:
for f in lessons/*.py; do MUJOCO_GL=egl python "$f" > "/tmp/$(basename $f .py).txt"; done
python check_claims.py /tmp/0*.txt                       # 81/81
```

No `--config` and no download anywhere. Lessons 1-4 need no environment variable
either — if `torch` imports, they run. Lessons 5-6 want `MUJOCO_GL=egl` on a
machine with no display, which is the one variable in this repository and is
written into every command that needs it.

## The lessons

**1 · A world model in 50 lines.** Build it, train it, and make two
measurements. Predicting the state *change* rather than the next state is 2.1x
more accurate for one line of code — with `dt` small, `s(t+1)` is nearly
`s(t)`, so a direct model is partly rewarded for copying its input. Then the
one that matters: a model with a one-step MSE of `1.5e-05` is **14x worse by
step 20**. The one-step number everyone reports does not tell you what you
want to know.

**2 · Why rollouts drift.** The standard explanation is the compounding-error
bound `e(k+1) <= L·e(k) + δ`, with `L` the Lipschitz constant. This lesson
measures whether it describes anything real.

It does not. On the pendulum it sits **6780x above the measurement at step 40**
and declares the rollout worthless at **step 25**, on a system whose typical
error is still 2% of the state size at step 90. The gap is on every seed.

<p align="center">
  <img src="figures/rollout-error.png" width="100%" alt="Four panels: the textbook bound diverging from measured error, Lorenz error growth, the fitted rate under three averages, and a histogram of usable horizons">
</p>

<sub>The four measurements this lesson makes. <b>(a)</b> the textbook bound
against the pendulum it is supposed to describe. <b>(b)</b> Lorenz, with error
injection separated from amplification. <b>(c)</b> the same rollouts summarised
three ways, where only the median recovers λ. <b>(d)</b> how far one model gets
on one task, as a distribution rather than a number. All four come from
<code>lessons/02_why_rollouts_drift.py</code> in under a minute on one GPU;
nothing is illustrative, and every interval comes from resampling
trajectories.</sub>

<p align="center">
  <img src="figures/drift.gif" width="88%" alt="A typical rollout and a runaway one animating side by side, their error curves growing apart on a log scale">
</p>

<sub>The same start and the same torques, run through the true dynamics and
through the model. Most rollouts stay together; a measured minority goes over
the top and never comes back, and the error between the two groups differs by
two orders of magnitude. That minority is what Lesson 2 is mostly about.</sub>

**What the shape of that curve is, this repository cannot tell you** — and
saying so is a result. Fitting an exponential and a power law to the pendulum's
median error curve gives residuals within 22% of each other. Across six seeds
the winner came out power-law twice and exponential four times. An earlier
version of this file reported "power law" as a finding; it was a coin flip.
`fit_growth` now refuses a verdict when the losing fit is under 1.5x worse, a
threshold set from measurement — synthetic curves of known shape sit at 3 to
23, the curves that flip sit at 1.2.

What survives is the magnitude, which does not move: the bound is wrong by
three to four orders of magnitude at any horizon you would plan over, whatever
the curve's shape is. The reason is that the pendulum's Lyapunov exponent is
`+0.016`, essentially zero, so there is almost nothing to amplify. What you are
watching is errors **accumulating**, not compounding.

### The average you report changes the answer, and it is not a small effect

Everything above used the **median** trajectory. On Lorenz, where the growth
rate itself is at stake, summarising the same rollouts three ways gives:

| | median | geometric mean | arithmetic mean |
|---|---|---|---|
| A perturb once, true dynamics | **+0.5%** | −1.9% | −6.9% |
| B world-model rollout | **+0.2%** | −1.5% | **+10.8%** |

<sub>Deviation of the fitted growth rate from the measured Lyapunov exponent.
The mean's error reproduced on a different machine at −7.1% and +12.3%.</sub>

Only the median recovers λ for both, and the mean errs in *opposite directions*
for the two curves, so it is not a bias you can correct for after the fact. If
you benchmark a world model on mean rollout error, this is a cost you are
paying and not reporting.

**The mechanism is concrete.** A pendulum driven by random torque can be pushed
over the top; once model and truth are on opposite branches the error is O(π)
and stays there. By step 90, **14% of rollouts have switched branch**. Those
14% are what the mean curve is mostly describing, and the typical rollout never
does what the mean says it does — which is what the animation above shows.

On the pendulum this sometimes goes further and the two summaries give
different functional *forms*, the median a power law and the mean an
exponential. That one is not reliable: across six seeds it happened twice.
Reported here because it is worth checking on your own system, not because it
is a property of world models.

### What actually sets the rate

The **Lyapunov exponent** — the average log-growth along the trajectory — not
`L`, the worst case over every state and direction. The lesson measures λ
directly with Benettin's method (Lorenz: `0.899`, against a literature value of
0.906, measured rather than quoted), then separates the two effects that
usually get conflated:

- **amplification** — the dynamics stretching an error that is already there
- **injection** — the model adding a fresh error at every single step

Perturb the true system *once* and you isolate amplification; run the model and
you get both. On the median trajectory, with intervals from resampling
trajectories:

| | growth rate | 95% CI | vs measured λ |
|---|---|---|---|
| **A** perturb once, true dynamics | 0.9030 | [0.8956, 0.9098] | +0.5%, covers λ |
| **B** world-model rollout | 0.9005 | [0.8872, 0.9112] | +0.2%, covers λ |

**Amplification alone grows at exactly the Lyapunov rate.** Curve A's interval
covered λ on six seeds out of six — the one result in this lesson that never
moved. Curve B, the full model rollout, lands near λ but not reliably on it: it
covered λ on two seeds of six and missed high on the other four, always by under
15%. So the honest form is *injection does not change the growth rate to within
about 15%*, not *it does not change it*. The lesson prints whichever the run
found and says how the seeds came out.

What injection does is lift the whole curve by a constant — measured here at 13
to 30 depending on the estimator, and 8 to 28 across seeds.

Two predictions bracket that: if successive model errors *aligned* they would
sum to ~111x, if they were *independent* the pile-up would be ~7.5x. The
measurement sits between them and, on this run, **does not distinguish the
two** (3.7x from one, 4.0x from the other). Earlier seeds landed near 11, which
does favour independence. The lesson prints whichever verdict the run supports,
including "cannot tell", and this is one of the open questions below.

### Horizon is not one number per model

On the same model and the same task, the 5th and 95th percentile trajectories
differ by **16x** in how many steps they survive. A fixed rollout horizon is too
long for the worst of them and needlessly short for the best.

**3 · Planning with a model you do not trust.** Lesson 2's usable horizon looks
like it should decide the planning horizon. It does not, and the gap is large.

The task is a pendulum swing-up with a torque limit of 3.0 against a peak
gravitational torque of 9.81 — you cannot push it up, you have to rock it, so a
planner is genuinely required. Three models are trained, differing only in how
much data they saw:

| model | one-step MSE | usable horizon | reward at H=25 | vs the TRUE dynamics |
|---|---|---|---|---|
| good | 3.5e-04 | 28 steps | −5.19 | +0.02 |
| thin | 1.6e-03 | **8 steps** | −5.21 | **−0.00** |
| bad | 5.3e-02 | 1 step | −7.13 | −1.93 |

**A model whose rollout is unusable after 8 steps plans 25-step action sequences
at no measurable cost**, against a planner handed the exact true dynamics. The
shortest horizon that suffices is 25 for every row including the true dynamics,
so it is set by the task — how long a swing-up takes — not by model quality.

The reason is that a planner never uses the states a model predicts. It uses
them to *rank* candidate action sequences, then throws them away and executes
one action. Ranking outlives the states by a wide margin:

| model | H | state error | rank correlation | regret |
|---|---|---|---|---|
| thin | 25 | 0.232 | **0.995** | **0.000** |
| bad | 25 | 3.196 | 0.080 | 6.938 |
| bad | 50 | 8.255 | −0.270 | 49.610 |

<sub>Regret is the true return of the best sequence minus that of the one the
model picked. Zero means the model's choice was actually optimal.</sub>

Planning breaks when the *ranking* breaks. That is the number to measure before
trusting a planner, and it is not the one Lesson 2 taught you to compute.

<p align="center">
  <img src="figures/planning.png" width="100%" alt="Reward against planning horizon for three models of different quality, beside the rank correlation of candidate returns that decides which of them plans well">
</p>

**The trap this lesson opens with.** A planner needs two dynamics: one to
imagine with, one to actually move the world. Use the model for both and the
reward is scored on states the model invented:

| model | reward in its own dream | reward in the real world |
|---|---|---|
| good | −5.20 | −5.19 |
| thin | −5.24 | −5.21 |
| **bad** | **−1.75** | **−7.13** |

The worst model wins by a wide margin, because it imagines the pendulum already
balanced. Nothing errors. `wm.cem_mpc` takes the two dynamics as separate
required arguments so that writing this by accident is not possible.

**4 · Latent world models, and the collapse you cannot see in the loss.** Real
world models see pixels, so they encode first and predict in latent space.
Training that on `||f(e(o), a) − e(o')||²` has an exact trivial optimum: make
the encoder constant. Here are the naive objective and two standard fixes, on
observations that are 16 dimensions of pendulum and 48 of irrelevant moving
background:

| objective | prediction loss | latent std | effective rank | probe R² |
|---|---|---|---|---|
| naive | **5.5e-08** | 0.0003 | 5.62 | **0.287** |
| +decoder | 3.0e-01 | 0.3059 | 6.08 | 0.845 |
| +variance | 3.3e-04 | 1.1888 | **1.01** | 0.868 |
| *untrained encoder (floor)* | — | — | — | *0.500* |
| *raw observation (ceiling)* | — | — | — | *0.923* |

**The best loss belongs to the only representation that scores below an
untrained encoder.** Seven orders of magnitude of loss improvement bought a
model worse than random features, and nothing in the training curve says so.

Then all three collapse detectors disagree. Latent std catches `naive` and
passes `+variance`; effective rank passes `naive` and fails `+variance`. Both
are right about something: std is blind to direction, effective rank is
computed from the shape of the spectrum and is blind to scale. Neither is a
collapse detector on its own, and the probe only means anything against a
floor — a random encoder gets 0.500 for free.

<p align="center">
  <img src="figures/collapse.png" width="100%" alt="Three panels: the lowest training loss belongs to the collapsed model, two collapse detectors disagreeing, and probe R-squared against an untrained-encoder floor">
</p>

**5 · Real environments.** Everything so far ran on two systems written by hand
in this repository. This lesson repeats the measurements on three MuJoCo
environments and reports which findings survive.

Installing is two packages and one environment variable, and the variable is the
whole game on a headless machine:

```bash
pip install gymnasium mujoco
export MUJOCO_GL=egl        # before mujoco or gymnasium is imported
```

| `MUJOCO_GL` | what happens here |
|---|---|
| unset | `mujoco.FatalError: an OpenGL platform library has not been loaded` |
| `glfw` | the same, after a GLFW X11 warning |
| `osmesa` | `AttributeError` inside PyOpenGL |
| **`egl`** | **works** |

One wrong combination aborts the process rather than raising, so a `try/except`
around the render call does not save you. The consolation: none of these
measurements render anything. They need states, not pictures.

**The bound gets worse, and the honest way to say so.** Quoting "overestimates by
1e56 at step 60" is true and useless — `L^k` on real dynamics is astronomical
and reads as a straw man. The useful form is when each claim the rollout is
finished:

| environment | `L_max` | bound says worthless at | actually is at |
|---|---|---|---|
| InvertedPendulum | 14.4 | **step 3** | still fine at 60 |
| Reacher | 18.3 | **step 3** | step 25 |
| HalfCheetah | 8887.8 | **step 2** | still fine at 40 |

<sub>One run, and read <code>L_max</code> as one draw rather than a constant: it is a
maximum over sampled states. Across four runs it moved from 1403 to 10262 on
HalfCheetah and from 9.2 to 23.2 on InvertedPendulum, while Reacher's failure step
came out 18, 21, 22 and 26. This table used to say the step "moves by one between
runs" — that was measured wrong, and nothing checked it, which is why the numbers
here had drifted by five steps and a factor of 1.7 from the run that produced
everything else on this page. Both are checked now. What does not move is the middle column against the right one: single
digits against tens.</sub>

**What survives, and what needed a qualification.** The horizon spread carried
over everywhere (1.6x to 19x). The median-versus-mean divergence appeared in 0
or 1 of 3 environments depending on the run — which is itself the finding: on
the pendulum in Lesson 2 it reproduced every time, here it does not, so it is a
risk to check on your own system rather than a property of world models.

And one result turned out to be about the regime rather than about world
models. **Lesson 2's rate-equals-lambda finding needed a chaotic system, an
accurate model, and four decades of range before saturation.** On all three
MuJoCo environments the median curve is a power law, so the exponential rate is
a fit that lost and comparing it to λ would be reading a number out of the wrong
model. Amplification needs something to amplify; on a robot with an imperfect
model, injection dominates the entire horizon you plan over. Lesson 2 now says
so, and points here.

**6 · A real robot arm.** Two questions, on MuJoCo arms and legs.

**Why is the bound catastrophic on some robots and merely loose on others?**
Perturb a state by 1e-6, take one step, measure the gain, and compare it to the
sustained rate:

| | one-step gain | `exp(λ·Δt)` | ratio |
|---|---|---|---|
| Reacher (arm) | 1.00 | 1.019 | **1.0** |
| Pusher (arm) | 0.94 | 1.054 | **0.9** |
| Walker2d (leg) | 5.60 | 1.034 | **5.4** |
| HalfCheetah (leg) | 13.10 | 1.315 | **10.0** |

A random perturbation first picks up the largest singular value of the
Jacobian; only once it rotates into the growing direction does it settle to λ.
**The textbook bound compounds the first number as if it were the second.** On
an arm the two coincide and the bound is merely loose. On a leg they differ by
5–10x *per step*, which is why Lesson 5 saw it write rollouts off at step 2.

**Can a learned world model actually drive the arm?** Reacher, from a random-play
dataset with no reward signal during training:

<p align="center">
  <img src="figures/arm.gif" width="76%" alt="A two-link arm under random actions beside the same arm driven by a planner using the learned model, with tip-to-target distance below">
</p>

<p align="center">
  <img src="figures/arm-still.png" width="94%" alt="Six rendered frames in two rows, random actions above and model-based planning below, at steps 0, 20 and 60 with tip-to-target distances">
</p>

<sub>The same run as three frames, because GitHub wraps animated images in a play
control and a reader who has asked for reduced motion sees only the first.</sub>

| planner | reward/step | final fingertip-to-target |
|---|---|---|
| random actions | −0.1861 | — |
| world model, H=3 | −0.0397 | 0.0004 |
| **world model, H=5** | **−0.0239** | 0.0007 |
| world model, H=10 | −0.0272 | 0.0028 |
| world model, H=20 | −0.0392 | 0.0100 |
| world model, H=40 | −0.0533 | 0.0239 |

**The horizon the task wants is 5 here, where the pendulum swing-up needed 25.**
Both are the task's answer: a swing-up must pump energy over many steps before
anything good happens, and reaching does not — greedy descent is already right,
so a longer horizon only buys more model error. Lesson 3's pendulum never showed
that cost because its model stayed accurate across the whole sweep. Same claim,
opposite-looking curve.

And the quantity Lesson 3 identified, on the real arm:

| H | state error | rank correlation | regret |
|---|---|---|---|
| 5 | 0.2483 | 0.998 | 0.0000 |
| 20 | 2.0501 | 0.990 | 0.0000 |
| 40 | 4.4333 | 0.905 | 0.1981 |

The state drifts by nearly 18x between H=5 and H=40 while the ranking holds,
and the planner pays nothing until the ranking goes.

<p align="center">
  <img src="figures/real-robots.png" width="100%" alt="Three panels: when the textbook bound calls a rollout worthless versus when it actually fails, one-step versus sustained amplification per robot, and reward against planning horizon">
</p>

## Open questions

Genuine ones, not decoration. If you have an answer, it is worth an issue.

1. **Is the pile-up factor closer to the independent or the aligned
   prediction?** The measurement moves between 8 and 30 with the estimator and
   the seed, which straddles the midpoint between 7.5 and 111. Separating them
   needs a better estimator than the ratio of two noisy curves.
2. **Why does the arithmetic mean bias the two Lorenz curves in *opposite*
   directions?** Both have log-error dispersion that widens with time, which
   predicts inflation for both. The model rollout inflates by +11%; the single
   perturbation deflates by −7%. A lognormal σ²/2 correction fits neither.
   Wrong-lobe events are the obvious suspect — by t=9, 97% of rollouts have
   ended up on the wrong lobe of the attractor — but conditioning on survival
   to test it introduces its own selection bias, and we could not remove it
   cleanly.
3. **Does any of this survive on pixels?** Partly answered, and the honest
   split matters. [integrations/dce-mri/](integrations/dce-mri/) measures a
   model that predicts pixels, and the aggregate-metric result carries over
   intact. What does not carry over is the part about compounding error: that
   model is handed a real image every time and never consumes its own output.
   A model rolled forward in its own pixel predictions is still untested here.

## What is coming

A latent world model driving the arm from pixel observations, rolled forward in
its own predictions — which is where Lessons 4 and 6 meet, and the one setting
in this repository where compounding error in pixel space would actually arise.

## A note on how this is written

Every verdict the lessons print is **computed from that run**, not written in
advance. When the evidence is too thin, the script says so instead of
asserting a conclusion — Lesson 2 will refuse to quote a horizon spread if too
many trajectories were censored, and refuse to call a curve exponential if
there is not enough dynamic range to tell. Tutorials that hard-code their
conclusions into `print` statements drift away from their own numbers the
moment a parameter changes. `check_claims.py` makes that failure impossible to
ship quietly: it greps the README's numbers out of a captured run and exits
non-zero if any of them is missing.

**Every table above is one recorded run.** Third digits move between runs, and
some second digits do: the pile-up factor in Lesson 2 has landed anywhere from
8 to 28 across seeds, Lesson 5's bound-versus-reality step shifts by one, and
Lesson 5's median-versus-mean divergence appears in 0 or 1 of 3 environments
depending on the run. `check_claims.py` is written to that reality — it verifies
the qualitative verdict and the order of magnitude, not the last digit, and
where a quantity is genuinely unstable the lesson says so in its own output
rather than letting the README imply otherwise. Anything stated here as a
conclusion has held across every run we have done; anything that has not is
labelled.

## What the badges actually cover

A green badge that quietly stands behind less than it appears to is the same
failure this repository is about, so here is the scope in full.

Every claim in `check_claims.py` is marked **stable** or not:

- **stable** — a qualitative verdict, an order of magnitude, or a statement the
  lesson makes about its own reliability. These held on every seed and every
  machine we have run them on.
- the rest quote numbers from the run that produced this README. They will not
  match a different seed or a different CPU, and that is a measurement rather
  than a defect: a first CI build backed 14 of 25, and all eleven misses were
  real drift.

| | runs in CI | checked before release |
|---|---|---|
| 35 library controls, 28 figure checks, 13 markdown checks and 7 on the checker (`tests/`) | ✓ on Python 3.9, 3.11, 3.12 | ✓ |
| Lessons 1–2, **stable claims** | ✓ | ✓ |
| 12 integration numbers, recomputed from committed results | ✓ needs neither GPU, checkpoint nor scan | ✓ |
| Lessons 1–2, recorded numbers | ✗ different CPU | ✓ |
| Lessons 3–6 | ✗ wants a GPU or MuJoCo | ✓ |

```bash
python check_claims.py /tmp/0*.txt                 # 81/81 against the recorded run
python check_claims.py --stable-only /tmp/0*.txt   # what should hold anywhere
```

CI runs the second form and prints the subset it stands behind. Running the
full set there would fail honestly on every build and teach everyone to ignore
the badge; running only the loose half locally would let the README drift. Both
are checked, each where it means something. An empty selection exits non-zero —
a check that matched nothing is not a check that passed.

The `tests/` suite is two kinds of check, and neither is a unit test. The
library cases are ones where the answer is known independently — the Lorenz
exponent against its literature value, a finite-difference Jacobian against the
analytic one, a synthetic curve of known shape, a confidence interval against
its nominal coverage — plus negative controls, which catch more: a metric that
reports something sensible on structureless data will report something sensible
on a broken model too. The rest hold this page to the repository behind it: that
every figure is legible at the width it is shown and carries alt text, that the
tables render as tables, that every count and size quoted here is the one the
files actually have, and that the checker does what it claims. Each of those
exists because the thing it checks had already gone wrong once.

Four of those tests exist because a bug got past everything else:
`test_action_dim_is_not_hard_coded` (one wrong constant, in two functions, found
when Lesson 6 tried a two-joint arm), `test_the_interval_covers_the_truth...`
(bootstrapping the points of one averaged curve instead of the trajectories,
which halved every interval), and the two shape tests, which now also check that
an ambiguous curve is called ambiguous instead of being given a coin-flip
verdict.


## Honest scope

The lessons use 2-D and 3-D systems with fully observed states, chosen because
the true dynamics are known exactly — which is what makes "how wrong is the
model" a question with an answer. Lessons 5 and 6 carry that into MuJoCo,
[integrations/tdmpc2/](integrations/tdmpc2/) carries the measurement onto a
published latent-space world model, and [integrations/dce-mri/](integrations/dce-mri/)
carries it onto one that predicts pixels.

**What has still not been checked is a model rolled forward in its own pixel
predictions — DreamerV3 and its kind.** That is a different thing from the DCE
model, which predicts pixels but is handed a real image every time and never
consumes its own output; compounding error, which most of this repository is
about, does not arise there. The findings about growth shape and rate should be
read as precise statements about settings where everything is measurable, not
as established facts about video world models.

Each integration also has a limit of its own, stated where it belongs. TD-MPC2
has no decoder, so encoder drift and dynamics error are not separable and what
is reported is the two together — which is also what the planner gets. The DCE
run scores inside a bounding box rather than a segmentation, on one collection,
with one U-Net at one training budget.

## Related work, and what it is good for

If you want a large-scale video world model, go to
[minWM](https://github.com/shengshu-ai/minWM). For a working PyTorch DreamerV3,
[NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch). For TD-MPC2,
[nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2). For a reading
list, [Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM). This repo is not
a replacement for any of them — it is the thing to read first, and the thing to
come back to when a rollout has gone wrong and you want to know why.

Companion: [world-model-map](https://github.com/GuoCheng24/world-model-map) —
what the authors of the major open world models say breaks in their own papers.

## License

MIT © Guo Cheng
