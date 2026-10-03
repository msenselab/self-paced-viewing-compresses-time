from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures_20260519_common import (
    COLORS,
    COND_LABELS,
    CONDS,
    ROOT,
    add_panel_label,
    add_subject_lines,
    assert_canonical_subjects,
    bar_summary,
    bootstrap_ci,
    draw_bars,
    save_figure,
    set_style,
    subject_condition_means,
)


TRIALS = ROOT / "data" / "group" / "behavioral_trial_level.csv"
# Written by code/revision/analysis_20260519_fixation_interval_accounting.py (run that first).
GAP = ROOT / "output" / "revision" / "fixation_interval_accounting_20260519" / "cell_level_gap_adjusted_20260519.csv"


def load_cell_level() -> pd.DataFrame:
    """Participant x condition x block-length cell means of the time-estimation trials."""
    trials = pd.read_csv(TRIALS)
    trials = trials[trials["phase"] == "time_estimation"]
    trials = trials.assign(sub=trials["sub"].astype(str).str.replace("sub-", "").str.zfill(3))
    return trials.groupby(["sub", "session_type", "n_pictures"], as_index=False).agg(
        estimated_duration_s=("estimated_duration_s", "mean"),
        actual_duration_s=("actual_duration_s", "mean"),
        estimation_ratio=("estimation_ratio", "mean"),
    )


def annotate_pair(ax, x0: float, x1: float, y: float, text: str) -> None:
    ax.plot([x0, x0, x1, x1], [y, y * 1.015, y * 1.015, y], color="#333333", linewidth=0.65)
    ax.text((x0 + x1) / 2, y * 1.02, text, ha="center", va="bottom", fontsize=7)


def add_actual_duration_ref(ax, value: float, show_label: bool = True) -> None:
    """Mid-grey dashed line marking the mean actual block duration (AN17)."""
    ax.axhline(value, color="#777777", linewidth=1.0, linestyle=(0, (5, 3)), zorder=1)
    if show_label:
        ax.text(
            ax.get_xlim()[1], value + 0.8, f"mean actual duration ({value:.1f} s)",
            ha="right", va="bottom", fontsize=6.8, color="#555555",
        )


def panel_bars(ax, data: pd.DataFrame, ylabel: str, panel: str, ylim: tuple[float, float], ref: float | None = None) -> None:
    summary = bar_summary(data)
    x = draw_bars(ax, summary)
    add_subject_lines(ax, data, x)
    if ref is not None:
        ax.axhline(ref, color="#777777", linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    add_panel_label(ax, panel)


def main() -> None:
    set_style()
    behavior = load_cell_level()
    gap = pd.read_csv(GAP)
    assert_canonical_subjects(behavior["sub"], "Figure 2 behavioral input")
    assert_canonical_subjects(gap["sub"], "Figure 2 gap-adjusted input")

    # Mean actual block duration, computed the same way the estimate bars are
    # aggregated (subject x condition cell means, then averaged). The three
    # conditions are matched on exposure, so a single reference line is exact.
    actual_by_cell = behavior.groupby(["sub", "session_type"])["actual_duration_s"].mean()
    actual_duration = float(actual_by_cell.mean())
    print(f"Figure 2 mean actual block duration: {actual_duration:.4f} s")

    duration = subject_condition_means(behavior, "estimated_duration_s")
    ratio = subject_condition_means(behavior, "estimation_ratio")
    gap_subject = (
        gap.assign(subject_id=lambda x: x["sub"].map(lambda s: str(s).zfill(3)))
        .groupby(["subject_id", "session_type"], as_index=False)[["estimated_duration_s", "gap_s", "estimate_plus_gap_s"]]
        .mean()
        .rename(columns={"session_type": "condition"})
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.42})

    # Axis limits cover every participant mean (max Baseline 102.7 s, max ratio 1.14);
    # all three Holm-corrected contrasts are marked (Watching-Baseline p = .005 / .0009).
    panel_bars(axes[0], duration, "Estimated duration (s)", "A", (0, 125))
    annotate_pair(axes[0], 0, 1, 105.0, "**")
    annotate_pair(axes[0], 1, 2, 105.0, "**")
    annotate_pair(axes[0], 0, 2, 114.0, "**")
    add_actual_duration_ref(axes[0], actual_duration)

    panel_bars(axes[1], ratio, "Estimate / actual", "B", (0, 1.38), ref=1.0)
    annotate_pair(axes[1], 0, 1, 1.17, "*")
    annotate_pair(axes[1], 1, 2, 1.17, "***")
    annotate_pair(axes[1], 0, 2, 1.27, "***")

    ax = axes[2]
    x = np.arange(len(CONDS), dtype=float)
    width = 0.28
    for i, cond in enumerate(CONDS):
        sub = gap_subject.loc[gap_subject["condition"] == cond]
        raw = sub["estimated_duration_s"].mean()
        gap_mean = sub["gap_s"].mean()
        total = sub["estimate_plus_gap_s"].mean()
        raw_low, raw_high = bootstrap_ci(sub["estimated_duration_s"].to_numpy(float), seed=20260519 + i)
        ax.bar(x[i] - width / 2, raw, width=width, color=COLORS[cond], edgecolor="black", linewidth=0.7, zorder=2)
        ax.errorbar(x[i] - width / 2, raw, yerr=[[raw - raw_low], [raw_high - raw]], color="black", capsize=2.5, linewidth=0.8, zorder=3)
        ax.bar(x[i] + width / 2, raw, width=width, color=COLORS[cond], edgecolor="black", linewidth=0.7, alpha=0.45, zorder=2)
        # White face + dense hatch keeps the gap segment unambiguous in greyscale (AN17).
        ax.bar(
            x[i] + width / 2, gap_mean, bottom=raw, width=width, facecolor="white",
            edgecolor="black", linewidth=0.7, hatch="////", zorder=2,
        )
        ax.text(x[i] + width / 2, total + 1.2, f"{total:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDS], rotation=20, ha="right")
    ax.set_ylabel("Duration accounted for (s)")
    ax.set_ylim(0, 65)
    add_actual_duration_ref(ax, actual_duration, show_label=False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor="#666666", edgecolor="black", label="Raw estimate"),
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="////", label="Fixation intervals"),
        ],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.88),
        fontsize=7,
        handlelength=1.6,
        handletextpad=0.5,
        borderaxespad=0.3,
    )
    add_panel_label(ax, "C")

    save_figure(fig, "fig2_time_compression_gap")
    plt.close(fig)


if __name__ == "__main__":
    main()
