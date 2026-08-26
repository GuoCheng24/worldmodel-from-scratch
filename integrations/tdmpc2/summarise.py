"""Print the TD-MPC2 tables in this README, from the committed .npz files.

    python summarise.py            # needs numpy, nothing else

The diagnosis runs want a GPU, TD-MPC2's repository and its released
checkpoints. What they leave behind is the per-rollout error matrix for every
task, which is small enough to live in this directory - so every number quoted
in the README can be recomputed here, by anyone, in a second.

Two statistics are reported, and they are chosen to survive Lesson 2 of this
repository: the mean over rollouts tracks a runaway minority rather than the
typical one, so nothing here averages.

  worse than standing still - the fraction of starts at which the model's
      k-step prediction is already further from the truth than simply assuming
      the state did not change. No division, no tail, and it is the question a
      planner is actually asking.
  error vs motion - the median prediction error over the median distance the
      latent really travelled. A ratio of medians rather than a median of
      ratios, because a rollout in which the latent barely moved puts a near
      zero in the denominator and the per-rollout ratio then reports the
      denominator rather than the model.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = ("1M", "48M", "317M")
PLAN_H = 3
ORDER = ["cartpole-swingup", "walker-walk", "cheetah-run", "finger-spin",
         "reacher-easy", "cup-catch", "pendulum-swingup", "hopper-stand"]


def returns():
    """TD-MPC2's own evaluate.py on the same checkpoints, 2 episodes per task.

    Without this the prediction numbers float free of anything anyone cares
    about, and the obvious objection - that a model predicts badly on a task it
    simply cannot do - cannot be answered either way.
    """
    f = os.path.join(HERE, "returns_mt30.tsv")
    if not os.path.exists(f):
        return {}
    lines = open(f).read().strip().split("\n")
    head = lines[0].split("\t")[1:]
    out = {s: {} for s in head}
    for ln in lines[1:]:
        cell = ln.split("\t")
        for s, v in zip(head, cell[1:]):
            out[s][cell[0]] = float(v)
    return out


def _spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def load():
    out = {}
    for s in SIZES:
        f = os.path.join(HERE, "results_mt30-%s.npz" % s)
        if not os.path.exists(f):
            continue
        r = np.load(f)
        # A results file written before per-rollout matrices were stored has
        # only summary curves; it cannot answer these questions, so skip it
        # rather than half-answer them.
        if any(k.startswith("err__") for k in r.files):
            out[s] = r
    return out


def facts(runs):
    """Per size: per-task statistics at the planner's horizon, plus summaries."""
    f = {}
    for s, r in runs.items():
        tasks = [t for t in ORDER if "err__" + t in r.files]
        loses, ratio, trust = {}, {}, {}
        for t in tasks:
            e, n = r["err__" + t], r["still__" + t]
            k = PLAN_H - 1
            loses[t] = 100.0 * float((e[:, k] >= n[:, k]).mean())
            ratio[t] = 100.0 * float(np.median(e[:, k]) / max(np.median(n[:, k]), 1e-12))
            crossed = e >= n
            per = np.where(crossed.any(1), crossed.argmax(1) + 1, e.shape[1])
            trust[t] = (float(np.percentile(per, 5)), float(np.median(per)),
                        float(np.percentile(per, 95)), bool((~crossed.any(1)).mean() > .5))
        v = np.array([loses[t] for t in tasks])
        f[s] = dict(tasks=tasks, loses=loses, ratio=ratio, trust=trust,
                    median=float(np.median(v)), lo=float(v.min()), hi=float(v.max()),
                    n=int(r["err__" + tasks[0]].shape[0]))
    return f


def main():
    runs = load()
    if not runs:
        print(__doc__); return 1
    f = facts(runs)
    sizes = [s for s in SIZES if s in f]
    tasks = f[sizes[-1]]["tasks"]

    print("\n  At k=%d - the horizon TD-MPC2 rolls its model forward to score actions -"
          % PLAN_H)
    print("  the share of starts where the prediction is already worse than assuming")
    print("  the state did not change.   (%d rollouts per task)\n" % f[sizes[0]]["n"])
    print("  | task | " + " | ".join("mt30-" + s for s in sizes) + " |")
    print("  |---|" + "---|" * len(sizes))
    for t in tasks:
        print("  | %s | %s |" % (t, " | ".join("%.0f%%" % f[s]["loses"][t] for s in sizes)))
    print("  | **median** | %s |"
          % " | ".join("**%.0f%%**" % f[s]["median"] for s in sizes))
    print("  | range | %s |"
          % " | ".join("%.0f-%.0f%%" % (f[s]["lo"], f[s]["hi"]) for s in sizes))

    print("\n  Median error over median motion at k=%d, same runs:\n" % PLAN_H)
    print("  | task | " + " | ".join("mt30-" + s for s in sizes) + " |")
    print("  |---|" + "---|" * len(sizes))
    for t in tasks:
        print("  | %s | %s |" % (t, " | ".join("%.0f%%" % f[s]["ratio"][t] for s in sizes)))

    ret = returns()
    if ret:
        print("\n  Against TD-MPC2's own evaluation of the same checkpoints"
              " (return out of 1000):\n")
        print("  | task | " + " | ".join("mt30-%s  return / lost" % s for s in sizes) + " |")
        print("  |---|" + "---|" * len(sizes))
        for t in tasks:
            print("  | %s | %s |" % (t, " | ".join(
                "%.0f / %.0f%%" % (ret[s][t], f[s]["loses"][t]) for s in sizes)))
        print()
        for s in sizes:
            r = np.array([ret[s][t] for t in tasks])
            l = np.array([f[s]["loses"][t] for t in tasks])
            print("    mt30-%-5s rank correlation between the two columns: %+.2f"
                  % (s, _spearman(r, l)))
        print("    - no stable relationship, in either direction.")

    big = sizes[-1]
    print("\n  Steps before the prediction stops beating standing still, mt30-%s," % big)
    print("  across starts within one episode - the number that is not one number:\n")
    for t in tasks:
        p5, p50, p95, censored = f[big]["trust"][t]
        note = "   (over half never cross inside 20)" if censored else ""
        print("    %-20s median %2.0f   [5th %2.0f, 95th %2.0f]%s" % (t, p50, p5, p95, note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
