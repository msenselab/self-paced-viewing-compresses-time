"""Figure 5 — Eye-tracking oculomotor summary: fixation duration, fixation rate, saccade amplitude."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from figures_20260519_common import (
    CONDS,
    ROOT,
    SUBJECTS,
    add_panel_label,
    add_subject_lines,
    assert_canonical_subjects,
    bar_summary,
    draw_bars,
    save_figure,
    set_style,
    subject_string,
)

ET_DIR = ROOT / "data" / "et_summaries"
PX_PER_DEG = 35.3


def load_et_data() -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in sorted(ET_DIR.glob("sub-*_et_summary.csv"))]
    if not dfs:
        raise FileNotFoundError("No sub-*_et_summary.csv files found under data/et_summaries/")
    all_et = pd.concat(dfs, ignore_index=True)
    all_et["subject_id"] = all_et["subject"].map(subject_string)
    all_et = all_et[all_et["subject_id"].isin(SUBJECTS)].copy()
    all_et["sacc_amp_deg"] = all_et["mean_sacc_amp"] / PX_PER_DEG
    return all_et


def metric_df(all_et: pd.DataFrame, col: str) -> pd.DataFrame:
    return (
        all_et[["subject_id", "block_type", col]]
        .rename(columns={"block_type": "condition", col: "value"})
        .copy()
    )


def sig_label(p: float) -> str | None:
    if p < .001: return "***"
    if p < .01:  return "**"
    if p < .05:  return "*"
    return None


def annotate_pair(ax, x0: float, x1: float, y: float, text: str) -> None:
    ax.plot([x0, x0, x1, x1], [y, y * 1.015, y * 1.015, y], color="#333333", linewidth=0.65)
    ax.text((x0 + x1) / 2, y * 1.022, text, ha="center", va="bottom", fontsize=7.5)


def draw_panel(ax, df: pd.DataFrame, ylabel: str, label: str, ylim: tuple,
               contrasts: list[tuple]) -> None:
    summary = bar_summary(df)
    x = draw_bars(ax, summary)
    add_subject_lines(ax, df, x)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    for x0, x1, y, p in contrasts:
        stars = sig_label(p)
        if stars:
            annotate_pair(ax, x0, x1, y, stars)
    add_panel_label(ax, label)


def main() -> None:
    set_style()
    all_et = load_et_data()
    assert_canonical_subjects(all_et["subject_id"], "Figure 5 ET input")

    fix_dur  = metric_df(all_et, "mean_fix_dur")
    fix_rate = metric_df(all_et, "fix_rate")
    sacc_amp = metric_df(all_et, "sacc_amp_deg")

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={"wspace": 0.46})

    # A: Fixation duration — S ≈ W (ns, bracket omitted), S < B (*), W < B (**)
    draw_panel(axes[0], fix_dur, "Fixation duration (ms)", "A", (0, 640), [
        (1, 2, 490, 0.003652),  # W vs B (narrow, lower)
        (0, 2, 548, 0.014396),  # S vs B (wide, higher)
    ])

    # B: Fixation rate — full gradient (**/***/***)
    draw_panel(axes[1], fix_rate, "Fixation rate (fix s⁻¹)", "B", (0, 4.7), [
        (0, 1, 3.60, 0.002986),  # S vs W
        (1, 2, 4.00, 3e-05),     # W vs B
        (0, 2, 4.35, 8e-06),     # S vs B
    ])

    # C: Saccade amplitude — W vs B (*), S vs W (***), S vs B (***)
    draw_panel(axes[2], sacc_amp, "Saccade amplitude (°)", "C", (0, 13.5), [
        (1, 2,  9.8, 0.032687),  # W vs B
        (0, 1, 11.2, 2.9e-05),  # S vs W
        (0, 2, 12.3, 2e-06),    # S vs B
    ])

    save_figure(fig, "fig5_eye_movements")
    plt.close(fig)


if __name__ == "__main__":
    main()
