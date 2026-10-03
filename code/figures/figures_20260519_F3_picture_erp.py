"""Figure 3 - Picture-locked posterior ERP, Scrolling versus Watching.

Panel A: Baseline-corrected posterior-occipital grand-average waveforms.
Panel B: Scrolling-minus-Watching difference waveform, shown both baseline-
         corrected (solid) and uncorrected (dashed). The uncorrected curve is
         reconstructed by shifting the corrected curve by the group-mean pre-onset
         amplitude difference from the uncorrected-epoch audit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import pandas as pd

from figures_20260519_common import (
    COLORS,
    ROOT,
    add_panel_label,
    channel_waveforms,
    condition_evokeds,
    save_figure,
    set_style,
    within_subject_ci,
)


POSTERIOR_OCC = ["PO3", "PO4", "POz", "PO7", "PO8", "O1", "Oz", "O2"]
AN6_CSV = ROOT / "data" / "revision" / "an6_preonset_subject_level.csv"

WINDOWS = [
    ("P1-like early positive", 80, 130, "#B3CDE3"),
    ("P2-like posterior positive", 140, 300, "#CCEBC5"),
    ("LPP", 300, 500, "#FED9A6"),
]


def main() -> None:
    set_style()
    mne.set_log_level("ERROR")

    evokeds = condition_evokeds(
        "picture_onset-epo.fif", ("active", "passive"), tmin=-0.2, tmax=1.0
    )
    times, active = channel_waveforms(evokeds["active"], POSTERIOR_OCC)
    _, passive = channel_waveforms(evokeds["passive"], POSTERIOR_OCC)
    ci_active, ci_passive = within_subject_ci([active, passive])
    active_mean = active.mean(axis=0)
    passive_mean = passive.mean(axis=0)

    an6 = pd.read_csv(AN6_CSV).query("cluster == 'posterior_occipital'")
    pre_scrolling = float(an6.query("condition == 'active'")["preonset_mean_uV"].mean())
    pre_watching = float(an6.query("condition == 'passive'")["preonset_mean_uV"].mean())
    preonset_delta = pre_scrolling - pre_watching

    diff_corrected = active_mean - passive_mean
    diff_uncorrected = diff_corrected + preonset_delta

    fig = plt.figure(figsize=(6.8, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.25], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax)

    for label, data, ci, color in [
        ("Scrolling", active, ci_active, COLORS["active"]),
        ("Watching", passive, ci_passive, COLORS["passive"]),
    ]:
        mean = data.mean(axis=0)
        ax.plot(times, mean, color=color, linewidth=1.55, label=label)
        ax.fill_between(times, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)

    ax.axvspan(-200, 0, color="#EEEEEE", alpha=0.55, zorder=0)
    for name, start, stop, shade in WINDOWS:
        ax.axvspan(start, stop, color=shade, alpha=0.30, zorder=0)
        label = {"P1-like early positive": "P1-like", "P2-like posterior positive": "P2"}.get(name, name)
        ax.text(
            (start + stop) / 2,
            0.955,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="#333333",
        )
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlim(-200, 600)
    ax.set_ylim(19.5, -2.5)  # ERP convention: positive voltage plotted downward
    ax.set_ylabel(r"Posterior amplitude ($\mu$V)")
    ax.set_title("Posterior-occipital cluster", loc="left", fontsize=8.5, pad=2)
    ax.legend(frameon=False, loc="upper right")
    ax.tick_params(labelbottom=True)
    add_panel_label(ax, "A")

    ax_b.axhline(0, color="#777777", linewidth=0.7)
    ax_b.axvspan(-200, 0, color="#EEEEEE", alpha=0.65, zorder=0)
    ax_b.axvline(0, color="#333333", linewidth=0.8, linestyle=(0, (3, 2)))
    ax_b.plot(times, diff_corrected, color="black", linewidth=1.35, label="baseline-corrected")
    ax_b.plot(times, diff_uncorrected, color="#888888", linewidth=1.1, linestyle="--", label="uncorrected")
    ax_b.set_xlim(-200, 600)
    ax_b.set_ylim(2.0, -2.0)  # ERP convention: positive voltage plotted downward
    ax_b.set_xlabel("Time from picture onset (ms)")
    ax_b.set_ylabel(r"Scrolling - Watching ($\mu$V)")
    ax_b.legend(frameon=False, fontsize=7, loc="lower right")
    add_panel_label(ax_b, "B")

    save_figure(fig, "fig3_picture_erp")
    plt.close(fig)


if __name__ == "__main__":
    main()
