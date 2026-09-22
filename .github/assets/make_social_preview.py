"""Generate the GitHub social-preview card (1200x630). Reproducible: python3 make_social_preview.py

The previous card put its message in a block of 14 pt monospace. A social card is unfurled at about
360 px wide in Slack, where that is grey noise, so the message is in the headline now and the
terminal panel is texture beside it. Layout shared across this account via bin/lightcard.py.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.home() / "bin"))
from lightcard import draw  # noqa: E402

out = draw(
    out=str(pathlib.Path(__file__).parent / "social-preview.png"),
    accent="#3fb950", badge="W", headline_size=42,
    kicker="SIX RUNNABLE LESSONS  ·  no simulator, no MuJoCo",
    headline="Build a world model in an afternoon",
    subline="then find out how far you can trust it",
    body=["A convincing one-step loss, and an",
          "imagination that fails between step",
          "13 and 23. Every number in the README",
          "comes from a lesson you can run."],
    panel=[("$ python lessons/02_why_rollouts_drift.py", "dim"),
           ("one-step loss    1e-03   looks fine", "ok"),
           ("control          upright every run", "ok"),
           ("imagination      fails   step 13-23", "red"),
           ("usable horizon   ~20     not one number", "warn")],
    footer="github.com/GuoCheng24/worldmodel-from-scratch",
)
print(f"written {pathlib.Path(out).name} 1200x630")
