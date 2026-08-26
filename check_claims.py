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

Each claim is marked stable or not. A STABLE claim is one that held on every
seed and on every machine we have run it on - a qualitative verdict, an order of
magnitude, or a statement the lesson makes about its own reliability. The rest
quote numbers from the recorded run and will not match a different seed or a
different CPU, which is a fact about floating point and about how much these
quantities actually move, not a defect.

    check_claims.py <files>                 everything, against the run that produced the README
    check_claims.py --stable-only <files>   only what should hold anywhere - what CI checks

Marking that split explicitly is the point. Running the full set in CI would
fail on a different machine and teach everyone to ignore a red badge; running
only the loose half everywhere would let the README drift. Both are checked, in
the place where each is meaningful.
"""
import pathlib, re, sys

CLAIMS = [
    # ---- Lesson 1 ----
    (1, False, "lesson 1: one-step MSE 1.5e-05",                r'one-step MSE = 1\.5\d\de-05'),
    (1, False, "lesson 1: change beats direct, 2.1x",           r'change is 2\.1x more accurate'),
    (1, False, "lesson 1: step 20 is 14x worse",                r'step 20 .*?14x the one-step'),
    (1, True,  "lesson 1: the change wins, direction only",     r'STABLE: predicting the change wins on every machine'),
    (1, True,  "lesson 1: 20-step gap is at least 5x",          r'STABLE: the 20-step error is at least 5x'),
    # ---- Lesson 2 ----
    (2, True, "L_max = 1.29",                                  r'L_max = 1\.29'),
    (2, False, "bound overshoots 6780x at step 40",             r'\b40\b\s+\S+\s+\S+\s+6780x'),
    (2, True, "shape verdict reports its own decisiveness",    r'residuals exp [\d.]+ / pow [\d.]+, a ratio of [\d.]+ -> (?:ambiguous|power-law|exponential)'),
    (2, False, "an ambiguous curve is not given a verdict",     r'ratio of 1\.\d+ -> ambiguous|does not\s+distinguish the shapes'),
    (2, False, "pendulum lambda = +0.016",                      r'lambda = \+0\.01\d'),
    (2, True, "both summaries are shown with ratios",          r'median curve -> \S+\s+\(ratio [\d.]+\)\s+mean curve -> \S+\s+\(ratio [\d.]+\)'),
    (2, True, "shape contrast is called unreliable",           r'shape contrast is NOT the|Two different functional forms'),
    (2, False, "14% have switched branch by step 90",           r'step 90 :\s+1[34]\.\d%'),
    (2, False, "Lorenz lambda = 0.899",                         r'Lorenz\s+lambda = \+0\.89\d'),
    (2, True, "A rate ~0.90 with an interval covering lambda", r'A  rate 0\.9\d\d\d  95%CI \[0\.\d+, 0\.\d+\].*?covers lambda'),
    (2, True, "B rate reported with its interval",             r'B  rate 0\.9\d\d\d  95%CI \[0\.\d+, 0\.\d+\]'),
    (2, True, "A holds on six seeds out of six",               r'solid: it held on six seeds out of six'),
    (2, True, "B is stated as near lambda, not equal",         r"lands near lambda', not as an equality|the honest form of the claim"),
    (2, False, "A: +0.5% / -1.9% / -6.9%",                      r'A\s+median\s+0\.9030 .*?\+ ?0\.5%.*?A\s+geometric.*?-1\.9%.*?A\s+mean.*?-6\.9%'),
    (2, False, "B: +0.2% / -1.5% / +10.8%",                     r'B\s+median\s+0\.9005 .*?\+ ?0\.2%.*?B\s+geometric.*?-1\.5%.*?B\s+mean.*?\+10\.8%'),
    (2, True, "only the median covers lambda",                 r'Only the MEDIAN covers lambda'),
    (2, False, "mean moves the rate +11% / -7%",                r'\+11% for the model rollout and -7%'),
    (2, False, "97% reach the wrong lobe by t=9",               r't=9\.0 :\s+9[5-9]\.\d%'),
    (2, False, "lift measured 13-30 across estimators",         r'median\s+2[0-9]\.\d\d.*?mean\s+1[0-9]\.\d\d'),
    (2, True, "aligned prediction 111",                        r'aligned prediction\s+111\.\d'),
    (2, True, "independent prediction 7.5",                    r'independent prediction\s+7\.4\d'),
    (2, True, "this run cannot separate the two",              r'does NOT distinguish the two hypotheses'),
    (2, False, "horizon varies 16x",                            r'A 1[56]\.\dx spread'),
    (2, False, "nothing censored",                              r'\[0% of trajectories never crossed\]'),
    # ---- Lesson 3 ----
    (3, True, "bad model scores -1.75 in its dream",           r'bad\s+-1\.7\d\s+-7\.1\d'),
    (3, False, "good/thin dream and reality agree",             r'good\s+-5\.\d\d\s+-5\.\d\d'),
    (3, False, "three models: horizons 28 / 8 / 1",             r'good.*?usable horizon 28.*?thin.*?usable horizon\s+8.*?bad.*?usable horizon\s+1'),
    (3, True, "sufficient H = 25 for every row",               r'Sufficient horizon: TRUE=25, good=25, thin=25, bad=25'),
    (3, False, "thin costs nothing at H=25",                    r'thin\s+usable horizon\s+8\s+reward at H=25:\s+-5\.2\d\s+\(-0\.00'),
    (3, False, "bad costs 1.93 at H=25",                        r'bad\s+usable horizon\s+1\s+reward at H=25:\s+-7\.1\d\s+\(-1\.9\d'),
    (3, True, "usable horizons predict none of it",            r'usable horizons here are 28, 8, 1 and they predict none of it'),
    (3, False, "thin at H=25: rho 0.995, regret 0",             r'thin\s+25\s+0\.2\d\d\s+0\.99\d\s+0\.000'),
    (3, False, "bad at H=50: rho negative, regret ~50",         r'bad\s+50\s+8\.\d\d\d\s+-0\.2\d\d\s+49\.\d'),
    # ---- Lesson 4 ----
    (4, True, "ceiling 0.923 on raw observation",              r'ceiling.*?R2 = 0\.92\d'),
    (4, True, "floor 0.47-0.54 on untrained encoder",          r'floor.*?R2 = 0\.4[6-7]\d to 0\.5[3-4]\d'),
    (4, False, "naive: loss 5.5e-08, probe 0.287",              r'naive\s+5\.5\de-08\s+0\.000\d\s+5\.\d\d\s+0\.28\d'),
    (4, False, "+decoder: loss 3.0e-01, probe 0.845",           r'\+decoder\s+3\.0\de-01\s+0\.30\d\d\s+6\.\d\d\s+0\.84\d'),
    (4, False, "+variance: rank 1.01, probe 0.868",             r'\+variance\s+3\.\d\de-04\s+1\.1\d\d\d\s+1\.0\d\s+0\.86\d'),
    (4, True, "naive is below the random floor",               r'only one BELOW the random-encoder floor'),
    (4, True, "std and rank disagree",                         r"latent std\s+catches 'naive'.*?effective rank passes 'naive'"),
    # ---- Lesson 5 ----
    (5, True, "run confirms MUJOCO_GL=egl",                    r'Running with MUJOCO_GL=egl'),
    (5, True, "state-only path needs no renderer",             r'state-only path works, no renderer touched'),
    (5, True, "bound speaks in single digits",                 r'InvertedPendulum-v5\s+[\d.]+\s+step [1-9]\b'),
    (5, True, "...about systems good for tens",                r'writes the rollout off within [1-9] steps'),
    (5, True, "how many MuJoCo curves are ambiguous",          r'[0-3] of 3 median curves are ambiguous'),
    (5, True, "a lost fit must not be quoted",                 r'a fit that did not win'),
    (5, True, "divergence count is reported",                  r'[0-3] of 3 give a different form'),
    (5, True, "its instability is reported too",               r'Treat it as something to check on your own system'),
    (5, False, "horizon spread carries over",                   r'InvertedPendulum-v5\s+1\s+\d+\s+1[0-9]\.\dx'),
    # ---- Lesson 2 ----
    (2, True, "Lesson 2 states its own scope",                 r'Read section 3 as a statement about the chaotic regime'),
    # ---- Lesson 6 ----
    (6, False, "Reacher arm: ratio 1.0",                        r'Reacher-v5\s+1\.0\d\s+1\.0[12]\d\s+1\.0'),
    (6, False, "HalfCheetah leg: ratio is ~10x",                r'HalfCheetah-v5\s+1[0-9]\.\d\d\s+1\.\d\d\d\s+(?:[89]|1[0-9])\.\d'),
    (6, True, "arms agree, legs do not",                       r'On an arm the two agree and the bound is merely'),
    (6, False, "random actions score -0.20",                    r'random actions\s+-0\.20\d\d'),
    (6, False, "H=5 is best on Reacher",                        r'world model, H=5\s+-0\.02\d\d'),
    (6, False, "fingertip within 0.001 of target",              r'world model, H=[35]\s+-0\.0\d\d\d\s+0\.00\d\d'),
    (6, False, "longer H costs reward",                         r'world model, H=40\s+-0\.0[3-9]\d\d'),
    (6, True, "task sets the horizon, again",                  r'Lesson 3 said the planning horizon is set by the task. Here it is 5'),
    (6, False, "H=5: rho ~1, regret 0",                         r'^      5\s+0\.\d+\s+0\.99\d\s+0\.0000'),
    (6, False, "H=40: rho drops, regret appears",               r'^      40\s+\d+\.\d+\s+0\.[5-9]\d\d\s+0*[0-9]\.\d{3,4}'),
]


PLAN_H = 3        # TD-MPC2 rolls its model forward this many steps to score actions
N_INTEGRATION = 12 # claims recomputed from integrations/*/ artefacts


def _load_module(path, name):
    """Import a file under a name of our choosing.

    Both integration directories ship a summarise.py. Adding each to sys.path
    and importing by name gives whichever was imported first, silently, and the
    second one's functions then have the wrong signature.
    """
    import importlib.util
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _tdmpc2_facts():
    """Recompute the TD-MPC2 numbers the README quotes, from the committed .npz.

    The integration runs need a GPU, TD-MPC2's repository and its released
    checkpoints, so they cannot run here. What they leave behind - the
    per-rollout error matrix for every task - is small enough to live in the
    repository, which is why it does. The arithmetic is imported from
    integrations/tdmpc2/summarise.py rather than repeated, so this file and
    that one cannot drift into disagreeing about what the runs said.
    """
    d = pathlib.Path(__file__).parent / "integrations" / "tdmpc2"
    if not d.is_dir():
        return None
    summarise = _load_module(d / "summarise.py", "tdmpc2_summarise")
    if summarise is None:
        return None
    runs = summarise.load()
    if len(runs) < 3:
        return None
    f = summarise.facts(runs)
    ret = summarise.returns()
    if not ret:
        return None
    import numpy as np
    f["returns"] = ret
    dce = _dce_facts()
    if dce is None:
        return None
    f["dce"] = dce
    f["rho"] = {}
    for s in ("1M", "48M", "317M"):
        tasks = f[s]["tasks"]
        f["rho"][s] = summarise._spearman(
            np.array([ret[s][t] for t in tasks]),
            np.array([f[s]["loses"][t] for t in tasks]))
    return f


def _sig(x):
    """Format the way the READMEs do: a real minus sign, not a hyphen."""
    return ("%+.2f" % x).replace("-", "\u2212")


def _dce_facts():
    """Recompute the DCE-MRI numbers from the committed per-slice results."""
    d = pathlib.Path(__file__).parent / "integrations" / "dce-mri"
    if not d.is_dir():
        return None
    dce = _load_module(d / "summarise.py", "dce_summarise")
    if dce is None:
        return None
    base, w50 = dce.load("full_plain"), dce.load("full_w50")
    if not base:
        return None
    g = lambda rows, reg, m, c: dce.stat(rows, reg, m, c)[0]
    out = dict(
        d_ssim=g(base, "whole_slice", "UNet", "ssim") - g(base, "whole_slice", "B0_identity", "ssim"),
        d_psnr=g(base, "whole_slice", "UNet", "psnr") - g(base, "whole_slice", "B0_identity", "psnr"),
        lesion_gap=abs(g(base, "lesion_box", "UNet", "ssim") - g(base, "lesion_box", "B2_cond_mean", "ssim")))
    if w50:
        out["w50_shift"] = abs(g(w50, "lesion_box", "UNet", "ssim") - g(base, "lesion_box", "UNet", "ssim"))
    return out


def integration_claims(f):
    """(name, {readme: regex}) for every TD-MPC2 number the READMEs quote."""
    b = lambda x: r"\*\*%s\*\*" % re.escape(x)
    pct = lambda x: b("%.0f%%" % x)
    tasks = f["317M"]["tasks"]
    worst1M = max(tasks, key=lambda t: f["1M"]["loses"][t])
    cart = "cartpole-swingup"
    trio_en = ", ".join("%.0f%%" % f[s]["loses"][cart] for s in ("1M", "48M", "317M"))
    trio_zh = "、".join("%.0f%%" % f[s]["loses"][cart] for s in ("1M", "48M", "317M"))
    tiny = sum(1 for t in tasks if f["48M"]["loses"][t] <= 1)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
             6: "six", 7: "seven", 8: "eight"}
    return [
        ("mt30-1M median share of starts lost",
         {"README.md": b("mt30-1M") + r": on a median task, " + pct(f["1M"]["median"]),
          "README.zh-CN.md": b("mt30-1M") + r":中位任务上有 " + pct(f["1M"]["median"])}),
        ("mt30-48M median share of starts lost",
         {"README.md": b("mt30-48M") + r": " + pct(f["48M"]["median"]),
          "README.zh-CN.md": b("mt30-48M") + r":" + pct(f["48M"]["median"])}),
        ("mt30-317M median share of starts lost",
         {"README.md": b("mt30-317M") + r": " + pct(f["317M"]["median"]),
          "README.zh-CN.md": b("mt30-317M") + r":" + pct(f["317M"]["median"])}),
        ("worst task on mt30-1M",
         {"README.md": pct(f["1M"]["loses"][worst1M]) + r"[^\n]{0,12}\n?[^\n]{0,20}"
                       + re.escape(worst1M),
          "README.zh-CN.md": re.escape(worst1M) + r"[^\n]{0,12}" + pct(f["1M"]["loses"][worst1M])}),
        ("cartpole-swingup rises at every size",
         {"README.md": re.escape(trio_en), "README.zh-CN.md": re.escape(trio_zh)}),
        ("tasks at or below 1% on mt30-48M",
         {"README.md": r"%s of the eight tasks to 1%% or below" % words[tiny],
          "README.zh-CN.md": r"8 个任务里有 %d 个降到 1%% 及以下" % tiny}),
        ("8 tasks were measured",
         {"README.md": r"8 dm_control tasks", "README.zh-CN.md": r"8 个 dm_control"}),
        ("mt30-317M return on the worst-predicted task",
         {"README.md": r"\*\*[^*]{0,12}%.0f of 1000 at 317M\*\*" % f["returns"]["317M"][cart],
          "README.zh-CN.md": b("\u5728 317M \u4e0a\u62ff\u5230 %.0f/1000" % f["returns"]["317M"][cart])}),
        ("DCE: U-Net beats the input globally",
         {"README.md": r"\*\*\+%.2f SSIM and \+%.1f dB\*\*" % (f["dce"]["d_ssim"], f["dce"]["d_psnr"]),
          "README.zh-CN.md": r"\*\*\+%.2f SSIM、\+%.1f dB\*\*" % (f["dce"]["d_ssim"], f["dce"]["d_psnr"])}),
        ("DCE: in the lesion it only draws level with the lookup table",
         {"README.md": r"draws\s+level\s+with\s+a\s+256-entry\s+lookup\s+table",
          "README.zh-CN.md": r"只和一张 256 格"}
         if f["dce"]["lesion_gap"] < 0.01 else
         {"README.md": r"__gap_is_%.3f_not_level__" % f["dce"]["lesion_gap"],
          "README.zh-CN.md": r"__gap_is_%.3f_not_level__" % f["dce"]["lesion_gap"]}),
        ("DCE: fifty times the lesion loss weight changes nothing",
         {"README.md": r"[Ww]eighting\s+the\s+loss\s+in\s+the\s+lesion\s+by\s+fifty\s+does\s+not\s+move\s+it",
          "README.zh-CN.md": r"加权\s*50\s*倍[,，]\s*纹丝不动"}
         if f["dce"].get("w50_shift", 1) < 0.03 else
         {"README.md": r"__w50_moved_%.3f__" % f["dce"].get("w50_shift", 1),
          "README.zh-CN.md": r"__w50_moved_%.3f__" % f["dce"].get("w50_shift", 1)}),
        ("rank correlation is unstable across sizes",
         {"README.md": b(", ".join(_sig(f["rho"][s]) for s in ("1M", "48M"))
                         .replace(", " + _sig(f["rho"]["48M"]),
                                  ", " + _sig(f["rho"]["48M"]) + " and " + _sig(f["rho"]["317M"]))),
          "README.zh-CN.md": b("\u3001".join(_sig(f["rho"][s]) for s in ("1M", "48M", "317M")))}),
    ]


def badge_matches_claim_count(total):
    """The README's claim-count badge must equal the number of claims here.

    A badge with a number in it drifts the moment a claim is added, and drifts
    silently, in the most prominent line of the file. That is the exact failure
    this script exists to prevent, so the badge is held to the same standard as
    everything else it advertises. It is checked against the source rather than
    against a captured run, because it is a statement about the repository and
    not about a measurement.

    It was already wrong once: the badge read 60/60 while there were 65 claims,
    and nothing caught it until someone looked at the rendered page.
    """
    import re as _re
    bad = []
    pattern = _re.compile(r"README" + "%20" + r"claims-(\d+)" + "%2F" + r"(\d+)" + "%20" + "verified")
    for name in ("README.md", "README.zh-CN.md"):
        f = pathlib.Path(__file__).parent / name
        if not f.exists():
            continue
        m = pattern.search(f.read_text())
        if not m:
            bad.append(name + ": no claim-count badge found")
        elif int(m.group(1)) != total or int(m.group(2)) != total:
            bad.append("%s: badge says %s/%s, there are %d claims"
                       % (name, m.group(1), m.group(2), total))
    return bad


def main(argv):
    lessons, paths, stable_only = set(), [], False
    for a in argv:
        if a.startswith("--lessons="):
            lessons = {int(x) for x in a.split("=", 1)[1].split(",")}
        elif a == "--stable-only":
            stable_only = True
        else:
            paths.append(a)
    if not paths:
        print(__doc__); return 2
    missing = [p for p in paths if not pathlib.Path(p).exists()]
    if missing:
        print("cannot read: %s" % ", ".join(missing)); return 2
    run = "".join(pathlib.Path(p).read_text() for p in paths)
    bad, checked, badge_bad = [], 0, badge_matches_claim_count(len(CLAIMS) + N_INTEGRATION)
    for problem in badge_bad:
        print("  BADGE  " + problem)
    for lesson, stable, claim, pattern in CLAIMS:
        if lessons and lesson not in lessons:
            continue
        if stable_only and not stable:
            continue
        ok = re.search(pattern, run, re.S | re.M) is not None
        checked += 1
        if not ok:
            bad.append(claim)
        print("  L%d %s %-40s %s" % (lesson, "S" if stable else " ", claim,
                                    "ok" if ok else "NOT FOUND IN OUTPUT"))
    # Integration claims are recomputed from committed arrays, so they need no
    # captured run and no GPU - they are checked on every invocation.
    facts = _tdmpc2_facts()
    if facts is None:
        bad.append("integrations/tdmpc2/*.npz missing - the README's TD-MPC2 "
                   "numbers cannot be recomputed")
        print("  I  integration results not found in integrations/tdmpc2/")
    else:
        claims = integration_claims(facts)
        if len(claims) != N_INTEGRATION:
            bad.append("N_INTEGRATION says %d, integration_claims() returns %d"
                       % (N_INTEGRATION, len(claims)))
        for name, per_file in claims:
            checked += 1
            missing = []
            for fname, pat in per_file.items():
                f = pathlib.Path(__file__).parent / fname
                if not f.exists() or not re.search(pat, f.read_text()):
                    missing.append(fname)
            if missing:
                bad.append(name)
            print("  I  %-40s %s" % (name, "ok" if not missing
                                     else "NOT IN " + ", ".join(missing)))

    print("\n  %d/%d README claims backed by the captured run%s."
          % (checked - len(bad), checked,
             "".join([" (lessons %s)" % ",".join(map(str, sorted(lessons))) if lessons else "",
                      ", seed- and machine-stable claims only" if stable_only else ""])))
    if not checked:
        print("  No claims matched those filters. An empty check is not a passing one.")
        return 1
    if badge_bad:
        print("  The claim-count badge is out of date - it is checked because it was"
              " once wrong by five.")
    if bad or badge_bad:
        print("  Fix the README or fix the code - do not ship a claim the run does not make.")
    return 1 if (bad or badge_bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
