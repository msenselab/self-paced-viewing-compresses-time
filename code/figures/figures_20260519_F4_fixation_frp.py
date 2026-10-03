"""Figure 4 — Fixation-locked occipital FRP, Scrolling/Watching/Baseline.

Wave-1 revision (AN16): Panel B is drawn as a subject-mean dot/line plot
instead of a bar chart with a truncated (1.2 µV) baseline. The lambda-
amplitude axis now spans the full subject range from a 0 µV reference, so
condition differences are not visually exaggerated by an arbitrary bar
origin and no subject points are clipped.

Renamed from figures_20260519_F5_fixation_frp.py during the Wave-1 figure
renumbering (former Figure 5 -> Figure 4 after the recognition figure was
dropped).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mne
import pandas as pd

from figures_20260519_common import (
    COLORS,
    COND_LABELS,
    CONDS,
    ROOT,
    add_panel_label,
    add_subject_lines,
    bar_summary,
    channel_waveforms,
    condition_evokeds,
    draw_bars,
    save_figure,
    set_style,
    within_subject_ci,
)


OCCIPITAL = ["O1", "Oz", "O2"]
FRP_SUMMARY = ROOT / "data" / "group" / "frp_summary_all.csv"


def main() -> None:
    set_style()
    mne.set_log_level("ERROR")
    evokeds = condition_evokeds("fixation_onset-epo.fif", CONDS, tmin=-0.2, tmax=0.8)
    arrays = []
    times = None
    for cond in CONDS:
        times, data = channel_waveforms(evokeds[cond], OCCIPITAL)
        arrays.append(data)
    cis = within_subject_ci(arrays)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1.42, 1.0], "wspace": 0.42})

    # --- Panel A: occipital fixation-locked grand averages ---
    ax = axes[0]
    for cond, data, ci in zip(CONDS, arrays, cis):
        mean = data.mean(axis=0)
        ax.plot(times, mean, color=COLORS[cond], linewidth=1.45, label=COND_LABELS[cond])
        ax.fill_between(times, mean - ci, mean + ci, color=COLORS[cond], alpha=0.15, linewidth=0)
    ax.axvspan(60, 120, color="#C7E9C0", alpha=0.45, zorder=0)
    ax.text(90, 0.96, "lambda", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7)
    ax.axhline(0, color="#777777", linewidth=0.7)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlim(-200, 800)
    ax.invert_yaxis()  # ERP convention: positive voltage plotted downward
    ax.set_xlabel("Time from fixation onset (ms)")
    ax.set_ylabel("Occipital amplitude (µV)")
    ax.legend(frameon=False, loc="lower right")
    add_panel_label(ax, "A")

    # --- Panel B: subject-mean occipital lambda amplitude (bar + subject lines) ---
    frp_csv = pd.read_csv(FRP_SUMMARY)
    lambda_df = (
        frp_csv.query("cluster == 'occipital' and window == 'lambda'")
        .assign(subject_id=lambda x: x["subject"].map(lambda s: str(s).zfill(3)))
        .rename(columns={"mean_amp_uv": "value"})
        [["subject_id", "condition", "value"]]
    )

    ax = axes[1]
    ax.axhline(0, color="#777777", linewidth=0.7, zorder=0)
    sm = bar_summary(lambda_df)
    x = draw_bars(ax, sm)
    add_subject_lines(ax, lambda_df, x)

    ax.set_ylabel("Lambda amplitude (µV)")
    ax.set_ylim(-1.4, 10.5)

    def bracket(x0: float, x1: float, y: float, text: str) -> None:
        top = y + 0.22
        ax.plot([x0, x0, x1, x1], [y, top, top, y], color="#333333", linewidth=0.7)
        ax.text((x0 + x1) / 2, top + 0.12, text, ha="center", va="bottom", fontsize=7.5)

    bracket(0, 1, 7.3, "**")   # S vs W p_holm=.004
    bracket(1, 2, 8.0, "***")  # W vs B p_holm<.001
    bracket(0, 2, 8.9, "***")  # S vs B p_holm<.001
    add_panel_label(ax, "B")

    save_figure(fig, "fig4_fixation_frp")
    plt.close(fig)


if __name__ == "__main__":
    main()
