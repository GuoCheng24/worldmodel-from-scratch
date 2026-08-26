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

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLUMN_PX = 896          # GitHub's README content column, near enough
MIN_SCALE = 0.40         # below this, 8-9 pt labels stop being readable
MAX_SCALE = 1.0          # above this, the image is being upscaled


def _figures():
    """(readme, path-from-repo-root, displayed width in percent) for every
    figure any README in the repository sizes explicitly. Paths in a README are
    relative to that file, not to the root."""
    for f in sorted(ROOT.glob("**/README*.md")):
        if ".git" in f.parts:
            continue
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
