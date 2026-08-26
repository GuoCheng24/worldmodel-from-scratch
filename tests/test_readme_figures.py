"""Every figure in the READMEs has to be readable at the size it is shown at.

Two mistakes prompted this file, both of which shipped and neither of which any
existing check could see.

The first: a figure laid out three panels across at 15.6 inches. Rendered at
100% in a README, which is about 900 px wide, its labels landed at 7 px - the
smallest text in the repository and too small to read. The size a label ends up
at is `pointsize x 900 / (72 x figure_width_in_inches)`; the dpi cancels, so a
higher-resolution export does not help. What matters is how far the image is
scaled down, and past roughly 2.4x nothing in this repository stays legible.

The second: an animation rendered 579 px wide and displayed at 734. Upscaling a
gif only blurs it.

Neither is recoverable from the image alone, but both show up in the ratio of
displayed width to native width, which is.
"""
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_readme_markdown import _docs
COLUMN_PX = 896          # GitHub's README content column, near enough
MIN_SCALE = 0.40         # below this, 8-9 pt labels stop being readable
MAX_SCALE = 1.0          # above this, the image is being upscaled


def _figures():
    """(readme, path-from-repo-root, displayed width in percent) for every
    figure any README in the repository sizes explicitly. Paths in a README are
    relative to that file, not to the root."""
    # Same list the markdown checks use, for the same reason: globbing the tree
    # picks up .pytest_cache/README.md, so the set of documents would depend on
    # where the reader last ran pytest rather than on what is in the repository.
    for f in _docs():
        rel = f.parent.relative_to(ROOT)
        for m in re.finditer(r'<img src="([^"]+)" width="(\d+)%"', f.read_text()):
            yield str(f.relative_to(ROOT)), str(rel / m.group(1)), int(m.group(2))


def test_readme_references_some_figures():
    """A silent zero here would make every other check in this file vacuous."""
    assert len(list(_figures())) >= 6


@pytest.mark.parametrize("readme,src,pct", list(_figures()))
def test_figure_is_shown_at_a_readable_scale(readme, src, pct):
    from PIL import Image

    path = ROOT / src
    assert path.exists(), "%s references %s, which is not in the repository" % (readme, src)
    with Image.open(path) as im:
        native = im.width
    scale = COLUMN_PX * pct / 100.0 / native
    assert scale <= MAX_SCALE, (
        "%s shows %s at %d%% of the column, %.0f px, but it is only %d px wide - "
        "upscaling it only blurs it" % (readme, src, pct, COLUMN_PX * pct / 100.0, native))
    assert scale >= MIN_SCALE, (
        "%s shows %s scaled to %.2fx of its %d px, which put this figure's labels "
        "below the size anything else in the repository uses. Draw it narrower "
        "or show it wider." % (readme, src, scale, native))


def _collected():
    """How many test instances pytest actually collects, per file.

    These counts used to be reconstructed by hand - "that file has two tests
    parametrised over every document, so subtract two and add twice the number
    of documents". That arithmetic states a fact about one test file inside a
    different one, and it was wrong the moment the document list changed: it
    read 13 in the tree it was written in and 9 in a fresh clone, which is the
    form the CI failure took. Ask pytest instead of predicting it.

    --collect-only imports the test modules but runs nothing, so this does not
    recurse into itself. no:cacheprovider keeps the call from writing the very
    .pytest_cache directory whose stray README started all this.
    """
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"),
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=str(ROOT))
    counts = {}
    for line in out.stdout.splitlines():
        if "::" in line:
            counts[pathlib.Path(line.split("::")[0].strip()).name] = \
                counts.get(pathlib.Path(line.split("::")[0].strip()).name, 0) + 1
    if not counts:
        raise AssertionError("pytest --collect-only collected nothing:\n"
                             + out.stdout + out.stderr)
    return counts


def test_readme_states_the_right_test_counts():
    """The counts in the badge section are prose, and prose drifts.

    They were four commits behind by the time anyone looked - 22 figure checks
    where there were 26. The claim count next to them is checked by
    check_claims.py; this checks these.
    """
    counts = _collected()
    lib = counts["test_library.py"]
    figs = counts["test_readme_figures.py"]
    md = counts["test_readme_markdown.py"]
    # Both READMEs, because only the English one was checked and the translated
    # line is a second copy of the same three numbers - which is to say a second
    # thing that can drift with nothing watching it.
    for name, pattern in (
            ("README.md",
             r"(\d+) library controls, (\d+) figure checks and (\d+) markdown checks"),
            ("README.zh-CN.md",
             r"(\d+) 项库正对照 \+ (\d+) 项配图检查 \+ (\d+) 项 markdown 检查")):
        text = (ROOT / name).read_text()
        m = re.search(pattern, text)
        assert m, "%s no longer states the test counts where this can read them" % name
        for got, want, what in ((m.group(1), lib, "library controls"),
                                (m.group(2), figs, "figure checks"),
                                (m.group(3), md, "markdown checks")):
            assert int(got) == want, ("%s says %s %s, the tests collect %d"
                                      % (name, got, what, want))
