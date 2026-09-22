"""A social card that survives being unfurled at 360 px.

The failure this exists to prevent: a card whose message is a block of 14 pt
monospace. Rendered at the width Slack actually gives it, that is grey noise.
What survives the reduction is one large headline, so the headline carries the
message and the terminal panel is texture beside it, not the point.

Canvas, palette and proportions are taken from docxaudit/docs/social-card.png,
the one card in this account that already worked at that size.

    from lightcard import draw
    draw(out="social-preview.png", accent="#b8860b", badge="D",
         kicker="PYTHON PACKAGE  ·  pip install docxaudit",
         headline="What your converter dropped",
         subline="it reported success and lost a table",
         body=["Pandoc says it worked.", "..."],
         panel=[("$ docxaudit paper.docx", "dim"), ("ERROR [TBL_NO_GRID]", "red")],
         footer="github.com/GuoCheng24/docxaudit")
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BG, INK, MUTE = "#fbfaf8", "#1b1b1a", "#5d5d5b"
PANEL_X0 = 6.00                       # left edge of the dark panel, in canvas units
PANEL_BG, PANEL_DIM, PANEL_INK = "#14120f", "#8a867e", "#e8e4dc"
PANEL_RED, PANEL_WARN, PANEL_OK = "#e06c60", "#d8a13a", "#6bbf73"
SANS, MONO = "Liberation Sans", "Liberation Mono"
COLOR = {"dim": PANEL_DIM, "ink": PANEL_INK, "red": PANEL_RED,
         "warn": PANEL_WARN, "ok": PANEL_OK}


def draw(out, accent, badge, kicker, headline, subline, body, panel, footer,
         headline_size=46, panel_title=None):
    W, H = 12.0, 6.30
    fig = plt.figure(figsize=(W, H), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), W, H, color=BG))
    ax.add_patch(plt.Rectangle((0, 0), 0.09, H, color=accent))       # left rule

    ax.add_patch(FancyBboxPatch((10.87, 5.30), 0.74, 0.55, boxstyle="round,pad=0.04",
                                fc=accent, ec="none"))
    ax.text(11.24, 5.575, badge, fontsize=27, fontweight="bold", color=BG,
            family=SANS, ha="center", va="center")

    ax.text(0.79, 5.40, kicker, fontsize=14.5, color=accent, family=SANS)
    ax.text(0.79, 4.32, headline, fontsize=headline_size, fontweight="bold",
            color=INK, family=SANS)
    ax.text(0.79, 3.65, subline, fontsize=21, color=INK, family=SANS)
    ax.add_patch(plt.Rectangle((0.79, 3.38), 1.32, 0.035, color=accent))

    y = 2.68
    for line in body:
        ax.text(0.79, y, line, fontsize=16.5, color=MUTE, family=SANS)
        y -= 0.40

    ax.add_patch(FancyBboxPatch((PANEL_X0, 1.40), 5.45, 2.42, boxstyle="round,pad=0.06",
                                fc=PANEL_BG, ec="none"))
    if panel_title:
        ax.text(PANEL_X0 + 0.22, 3.44, panel_title, fontsize=12.5, color=PANEL_DIM, family=MONO)
    y = 3.44 if not panel_title else 3.05
    for text, kind in panel:
        ax.text(PANEL_X0 + 0.22, y, text, fontsize=12.5, color=COLOR[kind], family=MONO)
        y -= 0.36

    ax.text(0.79, 0.60, footer, fontsize=15, color=MUTE, family=SANS)

    # The left column must not reach the panel. Dark text drawn on top of the dark
    # panel is still dark text on a dark ground: matplotlib puts it above, and a
    # reader sees nothing. The first card built from this template lost its subline
    # and two body lines that way, and sciglyph's checker is documented as not
    # covering text-behind-artwork - so the template refuses instead.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_left_px = PANEL_X0 * 100
    spill = []
    for t in ax.texts:
        box = t.get_window_extent(renderer)
        if box.x0 < panel_left_px and box.x1 > panel_left_px and box.y0 < 3.90 * 100:
            spill.append((t.get_text()[:44], round(box.x1 / 100, 2)))
    overflow = [(t.get_text()[:44], round(t.get_window_extent(renderer).x1 / 100, 2))
                for t in ax.texts
                if t.get_window_extent(renderer).x1 > (W - 0.28) * 100]
    if overflow:
        raise ValueError(
            "text runs past the right edge of the canvas and will be cut off:\n  "
            + "\n  ".join(f"{txt!r} reaches x={x} (canvas ends at %.2f)" % W
                           for txt, x in overflow))

    if spill:
        raise ValueError(
            "left-column text runs under the panel at x=%.2f; shorten it:\n  "
            % PANEL_X0
            + "\n  ".join(f"{txt!r} reaches x={x}" for txt, x in spill))

    try:                       # the author's own layout checker, when available
        import pathlib
        import sys
        sys.path.insert(0, str(pathlib.Path.home() / "sciglyph"))
        from sciglyph import report
        report(fig, ax)
    except Exception:
        pass

    fig.savefig(out)
    return out
