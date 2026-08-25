"""Check that every claim in the README is backed by the lessons' real output.

    for f in lessons/*.py; do python "$f" > "/tmp/$(basename $f .py).txt"; done
    python check_claims.py /tmp/0*.txt

The README of a repository like this makes numerical claims, and a reader's
only defence is to run the code and compare by hand. This does that comparison
mechanically: every number quoted in the README appears below as a pattern that
must be found in the captured output. A claim that drifts away from the code -
because a parameter changed, or because a sentence was written before the run -
fails here rather than in someone else's terminal.

Exit code 0 if every claim is backed, 1 otherwise. Run it before every commit
that touches either the lessons or the README. Pass `--lessons=1,2` to check
only the claims a partial run could support - CI uses that, because Lessons 3-6
want a GPU or extra packages.

Numbers move a little with the seed; the patterns below are written to the
precision the README actually quotes, and are deliberately tight on the
qualitative verdicts (which must not move at all) and looser on the last digit
of a rate.
"""
import pathlib, re, sys

CLAIMS = [
    # ---- Lesson 1 ----
    (1, "lesson 1: one-step MSE 1.5e-05",          r"one-step MSE = 1\.5\d\de-05"),
    (1, "lesson 1: change beats direct, 2.1x",     r"change is 2\.1x more accurate"),
    (1, "lesson 1: step 20 is 14x worse",          r"step 20 .*?14x the one-step"),
    # ---- Lesson 2 ----
    (2, "L_max = 1.29",                            r"L_max = 1\.29"),
    (2, "bound overshoots 6780x at step 40",       r"\b40\b\s+\S+\s+\S+\s+6780x"),
    (2, "median curve is a power law",             r"best fit by a power-law"),
    (2, "pendulum lambda = +0.016",                r"lambda = \+0\.01\d"),
    (2, "median -> power law, mean -> exp",        r"median curve -> power-law\s+mean curve -> exponential"),
    (2, "14% have switched branch by step 90",     r"step 90 :\s+1[34]\.\d%"),
    (2, "Lorenz lambda = 0.899",                   r"Lorenz\s+lambda = \+0\.89\d"),
    (2, "A rate 0.9030, CI covers lambda",         r"A  rate 0\.9030  95%CI \[0\.8956, 0\.9098\].*?covers lambda"),
    (2, "B rate 0.9005, CI covers lambda",         r"B  rate 0\.9005  95%CI \[0\.8872, 0\.9112\].*?covers lambda"),
    (2, "injection does not change the rate",      r"did not change the growth rate"),
    (2, "A: +0.5% / -1.9% / -6.9%",                r"A\s+median\s+0\.9030 .*?\+ ?0\.5%.*?A\s+geometric.*?-1\.9%.*?A\s+mean.*?-6\.9%"),
    (2, "B: +0.2% / -1.5% / +10.8%",               r"B\s+median\s+0\.9005 .*?\+ ?0\.2%.*?B\s+geometric.*?-1\.5%.*?B\s+mean.*?\+10\.8%"),
    (2, "only the median covers lambda",           r"Only the MEDIAN covers lambda"),
    (2, "mean moves the rate +11% / -7%",          r"\+11% for the model rollout and -7%"),
    (2, "97% reach the wrong lobe by t=9",         r"t=9\.0 :\s+9[5-9]\.\d%"),
    (2, "lift measured 13-30 across estimators",   r"median\s+2[0-9]\.\d\d.*?mean\s+1[0-9]\.\d\d"),
    (2, "aligned prediction 111",                  r"aligned prediction\s+111\.\d"),
    (2, "independent prediction 7.5",              r"independent prediction\s+7\.4\d"),
    (2, "this run cannot separate the two",        r"does NOT distinguish the two hypotheses"),
    (2, "horizon varies 16x",                      r"A 1[56]\.\dx spread"),
    (2, "nothing censored",                        r"\[0% of trajectories never crossed\]"),
    # ---- Lesson 3 ----
    (3, "bad model scores -1.75 in its dream",     r"bad\s+-1\.7\d\s+-7\.1\d"),
    (3, "good/thin dream and reality agree",       r"good\s+-5\.\d\d\s+-5\.\d\d"),
    (3, "three models: horizons 28 / 8 / 1",       r"good.*?usable horizon 28.*?thin.*?usable horizon\s+8.*?bad.*?usable horizon\s+1"),
    (3, "sufficient H = 25 for every row",         r"Sufficient horizon: TRUE=25, good=25, thin=25, bad=25"),
    (3, "thin costs nothing at H=25",              r"thin\s+usable horizon\s+8\s+reward at H=25:\s+-5\.2\d\s+\(-0\.00"),
    (3, "bad costs 1.93 at H=25",                  r"bad\s+usable horizon\s+1\s+reward at H=25:\s+-7\.1\d\s+\(-1\.9\d"),
    (3, "usable horizons predict none of it",      r"usable horizons here are 28, 8, 1 and they predict none of it"),
    (3, "thin at H=25: rho 0.995, regret 0",       r"thin\s+25\s+0\.2\d\d\s+0\.99\d\s+0\.000"),
    (3, "bad at H=50: rho negative, regret ~50",   r"bad\s+50\s+8\.\d\d\d\s+-0\.2\d\d\s+49\.\d"),
    # ---- Lesson 4 ----
    (4, "ceiling 0.923 on raw observation",        r"ceiling.*?R2 = 0\.92\d"),
    (4, "floor 0.47-0.54 on untrained encoder",    r"floor.*?R2 = 0\.4[6-7]\d to 0\.5[3-4]\d"),
    (4, "naive: loss 5.5e-08, probe 0.287",        r"naive\s+5\.5\de-08\s+0\.000\d\s+5\.\d\d\s+0\.28\d"),
    (4, "+decoder: loss 3.0e-01, probe 0.845",     r"\+decoder\s+3\.0\de-01\s+0\.30\d\d\s+6\.\d\d\s+0\.84\d"),
    (4, "+variance: rank 1.01, probe 0.868",       r"\+variance\s+3\.\d\de-04\s+1\.1\d\d\d\s+1\.0\d\s+0\.86\d"),
    (4, "naive is below the random floor",         r"only one BELOW the random-encoder floor"),
    (4, "std and rank disagree",                   r"latent std\s+catches 'naive'.*?effective rank passes 'naive'"),
    # ---- Lesson 5 ----
    (5, "run confirms MUJOCO_GL=egl",              r"Running with MUJOCO_GL=egl"),
    (5, "state-only path needs no renderer",       r"state-only path works, no renderer touched"),
    (5, "bound speaks in single digits",           r"InvertedPendulum-v5\s+[\d.]+\s+step [1-9]\b"),
    (5, "...about systems good for tens",          r"writes the rollout off within [1-9] steps"),
    (5, "median is power-law on all three",        r"median curve is power-law in every case"),
    (5, "a lost fit must not be quoted",           r"LOST, so comparing it to lambda"),
    (5, "divergence count is reported",            r"[0-3] of 3 environments give a different functional form"),
    (5, "its instability is reported too",         r"we have seen 0 and 1 of 3"),
    (5, "horizon spread carries over",             r"InvertedPendulum-v5\s+1\s+\d+\s+1[0-9]\.\dx"),
    # ---- Lesson 2 ----
    (2, "Lesson 2 states its own scope",           r"Read section 3 as a statement about the chaotic regime"),
    # ---- Lesson 6 ----
    (6, "Reacher arm: ratio 1.0",                  r"Reacher-v5\s+1\.0\d\s+1\.0[12]\d\s+1\.0"),
    (6, "HalfCheetah leg: ratio is ~10x",          r"HalfCheetah-v5\s+1[0-9]\.\d\d\s+1\.\d\d\d\s+(?:[89]|1[0-9])\.\d"),
    (6, "arms agree, legs do not",                 r"On an arm the two agree and the bound is merely"),
    (6, "random actions score -0.19",              r"random actions\s+-0\.19\d\d"),
    (6, "H=5 is best on Reacher",                  r"world model, H=5\s+-0\.02\d\d"),
    (6, "fingertip within 0.001 of target",        r"world model, H=[35]\s+-0\.0\d\d\d\s+0\.00\d\d"),
    (6, "longer H costs reward",                   r"world model, H=40\s+-0\.0[3-9]\d\d"),
    (6, "task sets the horizon, again",            r"Lesson 3 said the planning horizon is set by the task. Here it is 5"),
    (6, "H=5: rho 0.998, regret 0",                r"^      5\s+0\.3\d\d\d\s+0\.99\d\s+0\.0000"),
    (6, "H=40: rho drops, regret appears",         r"^      40\s+\d+\.\d+\s+0\.[4-7]\d\d\s+[0-9]\.\d\d\d"),
]


def main(argv):
    lessons, paths = set(), []
    for a in argv:
        if a.startswith("--lessons="):
            lessons = {int(x) for x in a.split("=", 1)[1].split(",")}
        else:
            paths.append(a)
    if not paths:
        print(__doc__); return 2
    missing = [p for p in paths if not pathlib.Path(p).exists()]
    if missing:
        print("cannot read: %s" % ", ".join(missing)); return 2
    run = "".join(pathlib.Path(p).read_text() for p in paths)
    bad, checked = [], 0
    for lesson, claim, pattern in CLAIMS:
        if lessons and lesson not in lessons:
            continue
        ok = re.search(pattern, run, re.S | re.M) is not None
        checked += 1
        if not ok:
            bad.append(claim)
        print("  L%d  %-40s %s" % (lesson, claim, "ok" if ok else "NOT FOUND IN OUTPUT"))
    print("\n  %d/%d README claims backed by the captured run%s."
          % (checked - len(bad), checked,
             " (lessons %s only)" % ",".join(map(str, sorted(lessons))) if lessons else ""))
    if bad:
        print("  Fix the README or fix the code - do not ship a claim the run does not make.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
