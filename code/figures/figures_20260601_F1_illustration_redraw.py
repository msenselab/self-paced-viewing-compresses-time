"""Redraw of Figure 1 design schematic (fig1_illustration.png).

The previous fig1_illustration.png was an AI-generated illustration, which
Elsevier journals do not accept. This script rebuilds the figure
programmatically (matplotlib), using neutral placeholders for stimulus images
to avoid implying fixed picture counts or unequal picture sizes.

Per the user's decision, the only retained AI element is the bottom-right
"Experimental setup" cartoon, which is cropped directly from the original
fig1_illustration.png and composited in unchanged.

Run:
    .venv/bin/python code/figures_20260601_F1_illustration_redraw.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures_20260519_common import COLORS, save_figure, set_style  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]  # open_data/
# Crop source for the retained "Experimental setup" cartoon: the ORIGINAL
# AI illustration (1536x1024), preserved separately so this script never reads
# back its own composited output (fig1_illustration.png), which would recursively
# nest the setup panel.
ORIGINAL = ROOT / "data" / "figure_assets" / "fig1_illustration_original_ai.png"

FIG_W, FIG_H = 9.2, 6.6  # inches; bg axes works in inches with equal aspect

# The top-row schematic uses equal placeholder squares rather than real photos
# or duration-scaled thumbnails. Labels communicate that Scrolling/Watching
# contain a variable number of non-fixed pictures within a block.
NONFIXED_LABELS = ["Picture\n1", "Picture\n2", "Picture\n3", "...", "Picture\nn"]
PICTURE_BOX = 0.40

INK = "#303030"
SCREEN_BG = "#6f6f6f"   # PsychoPy grey background of the real experiment
BEZEL = "#1f1f1f"
BTN_HILITE = "#E8A33D"  # colored left mouse button (the click)


def placeholder_box(ax, x, y, w, h, label, *, z: float = 5, rotate: bool = False,
                    fontsize: float = 8) -> None:
    """Draw a labelled grey placeholder in inch coordinates (origin bottom-left).

    Stimulus photographs are drawn from open-access repositories (Unsplash,
    Pexels, Pixabay) whose licences cover research use but leave publication
    reuse of the specific images uncertain. To avoid any copyright conflict at
    the journal-screening stage, the schematic shows text placeholders
    ("Picture N") rather than the actual photographs.
    """
    ax.add_patch(patches.Rectangle(
        (x, y), w, h, facecolor="#eaeaea", edgecolor=INK, linewidth=0.6, zorder=z))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            rotation=90 if rotate else 0, color="#3a3a3a", fontsize=fontsize,
            linespacing=1.05, zorder=z + 1)


def add_image(fig, img: np.ndarray, x: float, y: float, w: float, h: float, *, z: float = 5) -> None:
    """Place an image in inch coordinates (origin bottom-left)."""
    ax = fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H], zorder=z)
    ax.imshow(img, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(INK)
        s.set_linewidth(0.6)


def rounded_panel(ax, x, y, w, h, *, lw=1.0, ec="#AEAEAE") -> None:
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.10",
            facecolor="white", edgecolor=ec, linewidth=lw, zorder=1,
        )
    )


def mouse_icon(ax, cx, cy, s=1.0) -> None:
    # body
    ax.add_patch(patches.FancyBboxPatch(
        (cx - 0.12 * s, cy - 0.18 * s), 0.24 * s, 0.36 * s,
        boxstyle="round,pad=0.0,rounding_size=0.11", facecolor="white",
        edgecolor=INK, linewidth=1.1, zorder=4))
    # left button (colored = the active click), right button (white)
    ax.add_patch(patches.Rectangle((cx - 0.105 * s, cy + 0.015 * s), 0.085 * s, 0.145 * s,
                 facecolor=BTN_HILITE, edgecolor=INK, linewidth=0.7, zorder=5))
    ax.add_patch(patches.Rectangle((cx + 0.02 * s, cy + 0.015 * s), 0.085 * s, 0.145 * s,
                 facecolor="white", edgecolor=INK, linewidth=0.7, zorder=5))
    # scroll wheel between buttons
    ax.add_patch(patches.FancyBboxPatch(
        (cx - 0.016 * s, cy + 0.05 * s), 0.032 * s, 0.08 * s,
        boxstyle="round,pad=0.0,rounding_size=0.015", facecolor="#BBBBBB",
        edgecolor=INK, linewidth=0.5, zorder=6))
    # divider below the buttons
    ax.plot([cx - 0.105 * s, cx + 0.105 * s], [cy + 0.0 * s, cy + 0.0 * s],
            color=INK, lw=0.7, zorder=5)


def clock_icon(ax, cx, cy, s=1.0) -> None:
    ax.add_patch(patches.Circle((cx, cy), 0.16 * s, fill=False, edgecolor=INK, linewidth=1.1, zorder=4))
    ax.plot([cx, cx], [cy, cy + 0.10 * s], color=INK, lw=1.1, zorder=4)
    ax.plot([cx, cx + 0.08 * s], [cy, cy + 0.02 * s], color=INK, lw=1.1, zorder=4)
    # small "replay" curved arrow above the clock
    arc = patches.Arc((cx, cy + 0.26 * s), 0.22 * s, 0.16 * s, angle=0,
                      theta1=30, theta2=320, edgecolor=INK, linewidth=1.0, zorder=4)
    ax.add_patch(arc)
    ax.annotate("", xy=(cx + 0.10 * s, cy + 0.30 * s), xytext=(cx + 0.07 * s, cy + 0.22 * s),
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": INK}, zorder=4)


def eye_icon(ax, cx, cy, s=1.0) -> None:
    ax.add_patch(patches.Ellipse((cx, cy), 0.40 * s, 0.22 * s, fill=False, edgecolor=INK, linewidth=1.1, zorder=4))
    ax.add_patch(patches.Circle((cx, cy), 0.07 * s, color=INK, zorder=4))


def draw_condition(ax, fig, left, bottom, width, height, *, title, color,
                   single, caption, icon) -> None:
    rounded_panel(ax, left, bottom, width, height)
    cx = left + width / 2

    # title
    ax.text(cx, bottom + height - 0.30, title, ha="center", va="center",
            color=color, fontsize=14, fontweight="bold", zorder=6)

    # image strip
    sx0 = left + 0.24
    sw = width - 0.48
    th_top = bottom + height - 0.82
    th_h = 0.66

    box = PICTURE_BOX
    y_box = th_top - box

    if single:
        # Baseline uses the same square placeholder grammar, but the label makes
        # clear that the block repeats one fixed picture rather than a sequence.
        placeholder_box(ax, sx0 + (sw - box) / 2, y_box, box, box,
                        "fixed\npicture", z=5, fontsize=5.7)
    else:
        labels = NONFIXED_LABELS
        n = len(labels)
        # Slightly smaller boxes leave room for a legible fixation cross between them.
        y_box += box * 0.05
        box *= 0.9
        gap = (sw - n * box) / (n - 1)
        xi = sx0
        # A fixation cross precedes every picture, including the first.
        ax.text(xi - gap / 2 - 0.02, y_box + box / 2, "+", ha="center", va="center",
                color=INK, fontsize=9, zorder=6)
        for i, label in enumerate(labels):
            placeholder_box(ax, xi, y_box, box, box, label, z=5, fontsize=5.5)
            if i < n - 1:  # fixation cross at picture-to-picture transition
                gx = xi + box + gap / 2
                ax.text(gx, y_box + box / 2, "+", ha="center", va="center",
                        color=INK, fontsize=9, zorder=6)
            xi += box + gap
        # The fixation intervals carry the main behavioural result, so label them.
        ax.text(cx, y_box - 0.18, "+  fixation interval (0.7–1.0 s) before each picture",
                ha="center", va="center", fontsize=6.6, color=INK, zorder=6)

    # block-time arrow
    ay = th_top - th_h - 0.20
    ax.annotate("", xy=(sx0 + sw, ay), xytext=(sx0, ay),
                arrowprops={"arrowstyle": "->", "lw": 1.0, "color": INK}, zorder=6)
    ax.text(cx, ay - 0.18, "block time", ha="center", va="center", fontsize=8, color=INK, zorder=6)

    # condition icon
    icon(ax, cx, bottom + height - 2.40, 1.0)

    # caption
    ax.text(cx, bottom + 0.62, caption, ha="center", va="center",
            fontsize=8.2, color=INK, linespacing=1.35, zorder=6)


def screen(ax, x, y, w, h) -> tuple[float, float, float, float]:
    """Draw a monitor (dark bezel + grey experiment screen). Returns screen rect."""
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.0,rounding_size=0.05", facecolor=BEZEL,
                 edgecolor="#000000", linewidth=0.6, zorder=4))
    m = 0.06
    sx, sy, sw, sh = x + m, y + m, w - 2 * m, h - 2 * m
    ax.add_patch(patches.Rectangle((sx, sy), sw, sh, facecolor=SCREEN_BG,
                 edgecolor="none", zorder=5))
    return sx, sy, sw, sh


def draw_endofblock(ax, fig, left, bottom, width, height) -> None:
    rounded_panel(ax, left, bottom, width, height)
    ax.text(left + width / 2, bottom + height - 0.30, "End-of-block tasks",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK, zorder=6)

    box_w, box_h = 1.50, 1.10
    by = bottom + 0.85
    bx1 = left + 0.30
    bx2 = left + width - 0.30 - box_w

    # --- duration estimate: grey experiment screen with a numeric counter ---
    sx, sy, sw, sh = screen(ax, bx1, by, box_w, box_h)
    ax.text(sx + sw / 2, sy + sh * 0.60, "14.5", ha="center", va="center",
            fontsize=19, fontweight="bold", color="white", zorder=7)
    ax.text(sx + sw / 2, sy + sh * 0.30, "seconds", ha="center", va="center",
            fontsize=8, color="#e0e0e0", zorder=7)
    ax.text(sx + sw / 2, sy + sh * 0.10, "scroll wheel ↑↓", ha="center", va="center",
            fontsize=6.5, color="#bdbdbd", zorder=7)
    ax.text(bx1 + box_w / 2, by - 0.22, "Duration estimate", ha="center", va="center",
            fontsize=8.5, color=INK, zorder=6)

    # arrow between screens
    ax.annotate("", xy=(bx2 - 0.10, by + box_h / 2), xytext=(bx1 + box_w + 0.10, by + box_h / 2),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#555555"}, zorder=6)

    # --- recognition probe: actual task asks "Did you see this picture?";
    # no literal '?' placeholder is used in the experiment display.
    sx, sy, sw, sh = screen(ax, bx2, by, box_w, box_h)
    ax.text(sx + sw / 2, sy + sh - 0.11, "Did you see this picture?",
            ha="center", va="top", fontsize=4.8, color="white", zorder=7)
    iw = ih = 0.36
    placeholder_box(ax, sx + (sw - iw) / 2, sy + 0.40, iw, ih,
                    "probe\npicture", z=7, fontsize=5.0)
    for k, lab in enumerate(["Old", "New"]):
        bw2 = 0.52
        bxk = sx + (sw - (2 * bw2 + 0.12)) / 2 + k * (bw2 + 0.12)
        ax.add_patch(patches.FancyBboxPatch((bxk, sy + 0.10), bw2, 0.26,
                     boxstyle="round,pad=0.0,rounding_size=0.05", facecolor="white",
                     edgecolor="#333333", linewidth=0.8, zorder=8))
        ax.text(bxk + bw2 / 2, sy + 0.23, lab, ha="center", va="center",
                fontsize=7.5, color=INK, zorder=9)
    ax.text(bx2 + box_w / 2, by - 0.22, "Recognition probe", ha="center", va="center",
            fontsize=8.5, color=INK, zorder=6)


def draw_setup(ax, fig, left, bottom, width, height) -> None:
    """Composite the retained (AI) experimental-setup panel cropped from the original."""
    rounded_panel(ax, left, bottom, width, height)
    im = Image.open(ORIGINAL)
    W, H = im.size
    crop = im.crop((int(0.487 * W), int(0.49 * H), W - 2, H - 2)).convert("RGB")
    pad = 0.10
    add_image(fig, np.asarray(crop), left + pad, bottom + 0.30,
              width - 2 * pad, height - 0.30 - pad)
    ax.text(left + width / 2, bottom + 0.16,
            "64-channel EEG  ·  monocular right-eye eye-tracking",
            ha="center", va="center", fontsize=8, color=INK, zorder=6)


def main() -> None:
    set_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.set_aspect("equal")
    ax.axis("off")

    margin_x, gap = 0.18, 0.22
    panel_w = (FIG_W - 2 * margin_x - 2 * gap) / 3
    top_b, top_h = 3.10, 3.35
    lefts = [margin_x + i * (panel_w + gap) for i in range(3)]

    draw_condition(
        ax, fig, lefts[0], top_b, panel_w, top_h,
        title="Scrolling", color=COLORS["active"], single=False,
        caption="self-paced mouse click\nminimum 0.8 s per image", icon=mouse_icon,
    )
    draw_condition(
        ax, fig, lefts[1], top_b, panel_w, top_h,
        title="Watching", color=COLORS["passive"], single=False,
        caption="durations yoked to Scrolling\n(re-ordered); different images", icon=clock_icon,
    )
    draw_condition(
        ax, fig, lefts[2], top_b, panel_w, top_h,
        title="Baseline", color=COLORS["constant"], single=True,
        caption="one photograph for the whole block\nduration matched to Scrolling", icon=eye_icon,
    )

    draw_endofblock(ax, fig, margin_x, 0.22, 3.95, 2.55)
    draw_setup(ax, fig, 4.45, 0.18, FIG_W - 4.45 - margin_x, 2.62)

    save_figure(fig, "fig1_illustration_redraw")
    plt.close(fig)


if __name__ == "__main__":
    main()
