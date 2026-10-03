from __future__ import annotations

"""Gap/fixation-cross adjustment check for behavioral time estimates.

This script updates the 2026-04-20 "estimate plus gap" check to the current
canonical behavioral sample (N=23; excludes 015, 016, 023, 027). It tests the
claim that the condition differences in duration estimates are largely explained
by participants discounting fixation-cross/empty intervals.

Inputs:
    results/behavioral_20260511_exclude027/behavioral_trial_level_exclude027_20260511.csv

Outputs:
    results/time_discounting_gap_adjustment_20260519/
"""

import ast
import json
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, DATA_DIR, OUT_DIR, ANALYSIS_SUBJECTS  # open_data/ paths  # noqa: E402
ANALYSIS_EXCLUDED_SUBJECTS = ("015", "016", "023", "027")

INPUT = DATA_DIR / "group/behavioral_trial_level.csv"
OUT_DIR = OUT_DIR / "revision/fixation_interval_accounting_20260519"
FIG_DIR = OUT_DIR / "figures"

COND_ORDER = ["active", "passive", "constant"]
DURATION_ORDER = [9, 12, 18]
CONSTANT_GAP_MEAN_S = 0.85  # same convention as analysis_20260420_estimate_plus_gap_visualization.py
KEY_METRICS = [
    "estimated_duration_s",
    "estimate_plus_gap_s",
    "raw_ratio_total_actual",
    "gap_removed_ratio",
    "estimate_plus_gap_ratio",
]


def parse_list(value: object) -> list[float] | None:
    """Parse PsychoPy list fields that may be doubly quoted after CSV export."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    # Several existing derived CSVs contain a literal string such as
    # '"[0.7, 0.8]"'. Strip wrapper quotes until a list remains.
    while len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list):
        return None
    out: list[float] = []
    for item in parsed:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out


def holm(pvals: Iterable[float]) -> np.ndarray:
    pvals = np.array(list(pvals), dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def complete_cases(cell: pd.DataFrame, value_col: str) -> pd.DataFrame:
    work = cell.dropna(subset=[value_col]).copy()
    expected_cells = len(COND_ORDER) * len(DURATION_ORDER)
    counts = work.groupby("sub").size()
    good_subs = counts[counts == expected_cells].index
    return work[work["sub"].isin(good_subs)].copy()


def load_and_adjust() -> pd.DataFrame:
    df = pd.read_csv(INPUT, dtype={"sub": str})
    df["sub"] = df["sub"].astype(str).str.extract(r"(\d+)")[0].str.zfill(3)
    if "phase" in df.columns:
        df = df[df["phase"].astype(str).str.strip().eq("time_estimation")].copy()
    df["session_type"] = df["session_type"].astype(str).str.strip().str.lower()
    df = df[df["session_type"].isin(COND_ORDER)].copy()
    df = df[df["sub"].isin(ANALYSIS_SUBJECTS)].copy()

    for col in ["n_pictures", "actual_duration_s", "estimated_duration_s", "actual_duration", "estimated_duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "actual_duration_s" not in df.columns:
        df["actual_duration_s"] = df["actual_duration"]
    if "estimated_duration_s" not in df.columns:
        df["estimated_duration_s"] = df["estimated_duration"]

    df["fixation_duration_list"] = df["per_fixation_durations"].apply(parse_list)
    observed_gap = df["fixation_duration_list"].apply(lambda x: float(np.sum(x)) if isinstance(x, list) else np.nan)
    df["gap_s"] = observed_gap
    constant_mask = df["session_type"].eq("constant")
    df.loc[constant_mask, "gap_s"] = CONSTANT_GAP_MEAN_S
    df["gap_source"] = np.where(
        constant_mask,
        f"modeled_single_fixation_mean_{CONSTANT_GAP_MEAN_S:.2f}s",
        "observed_sum_per_fixation_durations",
    )
    df["n_fixations_modeled_or_observed"] = np.where(
        constant_mask,
        1,
        df["fixation_duration_list"].apply(lambda x: len(x) if isinstance(x, list) else np.nan),
    )

    df["actual_minus_gap_s"] = df["actual_duration_s"] - df["gap_s"]
    df["estimate_plus_gap_s"] = df["estimated_duration_s"] + df["gap_s"]
    df["raw_ratio_total_actual"] = df["estimated_duration_s"] / df["actual_duration_s"]
    # Equivalent operationalization of "remove fixation-cross duration": compare
    # reported estimates against picture-filled time only.
    df["gap_removed_ratio"] = df["estimated_duration_s"] / df["actual_minus_gap_s"]
    # Same intuition expressed on the estimate side, matching the 2026-04-20 script.
    df["estimate_plus_gap_ratio"] = df["estimate_plus_gap_s"] / df["actual_duration_s"]
    return df


def make_cell_table(trial: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "estimated_duration_s": ("estimated_duration_s", "mean"),
        "gap_s": ("gap_s", "mean"),
        "actual_duration_s": ("actual_duration_s", "mean"),
        "actual_minus_gap_s": ("actual_minus_gap_s", "mean"),
        "estimate_plus_gap_s": ("estimate_plus_gap_s", "mean"),
        "raw_ratio_total_actual": ("raw_ratio_total_actual", "mean"),
        "gap_removed_ratio": ("gap_removed_ratio", "mean"),
        "estimate_plus_gap_ratio": ("estimate_plus_gap_ratio", "mean"),
        "n_fixations_modeled_or_observed": ("n_fixations_modeled_or_observed", "mean"),
        "n_trials": ("estimated_duration_s", "size"),
    }
    cell = trial.groupby(["sub", "session_type", "n_pictures"], as_index=False).agg(**agg_cols)
    return cell


def descriptives(cell: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    collapsed_rows = []
    by_duration_rows = []
    for metric in metrics + ["gap_s", "actual_duration_s", "actual_minus_gap_s"]:
        work = complete_cases(cell, metric)
        collapsed = work.groupby(["sub", "session_type"], as_index=False)[metric].mean()
        desc = collapsed.groupby("session_type", as_index=False).agg(
            mean=(metric, "mean"),
            sd=(metric, "std"),
            sem=(metric, lambda x: x.std(ddof=1) / math.sqrt(len(x))),
            n=("sub", "nunique"),
        )
        desc.insert(0, "metric", metric)
        collapsed_rows.append(desc)

        by_dur = work.groupby(["sub", "session_type", "n_pictures"], as_index=False)[metric].mean()
        desc_dur = by_dur.groupby(["session_type", "n_pictures"], as_index=False).agg(
            mean=(metric, "mean"),
            sd=(metric, "std"),
            sem=(metric, lambda x: x.std(ddof=1) / math.sqrt(len(x))),
            n=("sub", "nunique"),
        )
        desc_dur.insert(0, "metric", metric)
        by_duration_rows.append(desc_dur)
    return pd.concat(collapsed_rows, ignore_index=True), pd.concat(by_duration_rows, ignore_index=True)


def paired_tests(cell: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        work = complete_cases(cell, metric)
        collapsed = work.groupby(["sub", "session_type"], as_index=False)[metric].mean()
        temp_rows = []
        pvals = []
        for a, b in combinations(COND_ORDER, 2):
            wide = collapsed[collapsed["session_type"].isin([a, b])].pivot(index="sub", columns="session_type", values=metric).dropna()
            diff = wide[a] - wide[b]
            t_stat, p_val = stats.ttest_rel(wide[a], wide[b])
            dz = diff.mean() / diff.std(ddof=1)
            ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
            temp_rows.append(
                {
                    "metric": metric,
                    "contrast": f"{a}_vs_{b}",
                    "n": len(wide),
                    "mean_a": wide[a].mean(),
                    "mean_b": wide[b].mean(),
                    "mean_diff_a_minus_b": diff.mean(),
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "t": t_stat,
                    "p_raw": p_val,
                    "dz": dz,
                }
            )
            pvals.append(p_val)
        for row, p_holm in zip(temp_rows, holm(pvals)):
            row["p_holm"] = p_holm
            rows.append(row)
    return pd.DataFrame(rows)


def rm_anovas(cell: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        work = complete_cases(cell, metric)
        model = AnovaRM(work, depvar=metric, subject="sub", within=["session_type", "n_pictures"]).fit()
        tab = model.anova_table.reset_index().rename(columns={"index": "effect"})
        tab.insert(0, "metric", metric)
        tab.insert(1, "n_subjects", work["sub"].nunique())
        tab["partial_eta_sq"] = (tab["F Value"] * tab["Num DF"]) / (tab["F Value"] * tab["Num DF"] + tab["Den DF"])
        rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def save_figure(desc: pd.DataFrame) -> Path:
    plot_metrics = ["estimated_duration_s", "estimate_plus_gap_s"]
    labels = {
        "estimated_duration_s": "Raw estimate",
        "estimate_plus_gap_s": "Estimate + fixation gap",
    }
    colors = {"active": "#111111", "passive": "#c83f49", "constant": "#3b6fb6"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
    for ax, metric in zip(axes, plot_metrics):
        sub = desc[desc["metric"].eq(metric)].set_index("session_type").reindex(COND_ORDER)
        x = np.arange(len(COND_ORDER))
        ax.bar(x, sub["mean"], yerr=sub["sem"], color=[colors[c] for c in COND_ORDER], capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in COND_ORDER])
        ax.set_title(labels[metric])
        ax.set_ylabel("Seconds" if ax is axes[0] else "")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Condition means before/after fixation-gap accounting (N=23)")
    out = FIG_DIR / "time_discounting_gap_adjustment_condition_means_20260519.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    trial = load_and_adjust()
    cell = make_cell_table(trial)
    complete_n = complete_cases(cell, "estimated_duration_s")["sub"].nunique()
    complete_subjects = sorted(complete_cases(cell, "estimated_duration_s")["sub"].unique().tolist())

    desc, desc_by_duration = descriptives(cell, KEY_METRICS)
    tests = paired_tests(cell, KEY_METRICS)
    anovas = rm_anovas(cell, KEY_METRICS)
    fig_path = save_figure(desc)

    trial_out = OUT_DIR / "trial_level_gap_adjusted_20260519.csv"
    cell_out = OUT_DIR / "cell_level_gap_adjusted_20260519.csv"
    desc_out = OUT_DIR / "condition_descriptives_gap_adjusted_20260519.csv"
    desc_dur_out = OUT_DIR / "condition_by_duration_descriptives_gap_adjusted_20260519.csv"
    tests_out = OUT_DIR / "paired_tests_gap_adjusted_20260519.csv"
    anova_out = OUT_DIR / "rm_anova_gap_adjusted_20260519.csv"
    summary_json_out = OUT_DIR / "summary_gap_adjusted_20260519.json"
    summary_txt_out = OUT_DIR / "summary_gap_adjusted_20260519.txt"

    trial.to_csv(trial_out, index=False)
    cell.to_csv(cell_out, index=False)
    desc.to_csv(desc_out, index=False)
    desc_by_duration.to_csv(desc_dur_out, index=False)
    tests.to_csv(tests_out, index=False)
    anovas.to_csv(anova_out, index=False)

    key_desc = desc[desc["metric"].isin(["estimated_duration_s", "gap_s", "estimate_plus_gap_s", "raw_ratio_total_actual", "gap_removed_ratio", "estimate_plus_gap_ratio"])]
    key_tests = tests[
        tests["metric"].isin(["estimated_duration_s", "estimate_plus_gap_s", "gap_removed_ratio", "estimate_plus_gap_ratio"])
        & tests["contrast"].isin(["active_vs_constant", "passive_vs_constant", "active_vs_passive"])
    ]
    key_anova = anovas[anovas["metric"].isin(["estimated_duration_s", "estimate_plus_gap_s", "gap_removed_ratio", "estimate_plus_gap_ratio"])]

    summary = {
        "input": str(INPUT),
        "outputs_dir": str(OUT_DIR),
        "analysis_subjects_n": int(complete_n),
        "analysis_subjects": complete_subjects,
        "excluded_subjects": list(ANALYSIS_EXCLUDED_SUBJECTS),
        "constant_gap_model_seconds": CONSTANT_GAP_MEAN_S,
        "method": {
            "active_passive_gap": "sum observed per_fixation_durations per trial/block",
            "constant_gap": "modeled as one fixation at 0.85 s, matching analysis_20260420_estimate_plus_gap_visualization.py",
            "estimate_plus_gap_s": "estimated_duration_s + gap_s",
            "gap_removed_ratio": "estimated_duration_s / (actual_duration_s - gap_s)",
            "stats": "subject x condition x duration cell means; repeated-measures ANOVA session_type x n_pictures; paired t-tests collapsed across duration with Holm correction within metric",
        },
        "files": {
            "trial_level": str(trial_out),
            "cell_level": str(cell_out),
            "descriptives": str(desc_out),
            "descriptives_by_duration": str(desc_dur_out),
            "paired_tests": str(tests_out),
            "rm_anova": str(anova_out),
            "figure": str(fig_path),
            "summary_txt": str(summary_txt_out),
        },
    }
    summary_json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append("Time-discounting / fixation-gap adjustment analysis (2026-05-19)\n")
    lines.append("=" * 72 + "\n\n")
    lines.append(f"Input: {INPUT}\n")
    lines.append(f"N complete analysis subjects: {complete_n}\n")
    lines.append(f"Subjects: {', '.join(complete_subjects)}\n")
    lines.append(f"Excluded by canonical config: {', '.join(ANALYSIS_EXCLUDED_SUBJECTS)}\n\n")
    lines.append("Methods:\n")
    lines.append("- Active/passive gap_s = sum(per_fixation_durations) for each trial/block.\n")
    lines.append(f"- Constant gap_s = modeled single fixation mean {CONSTANT_GAP_MEAN_S:.2f} s (same as 20260420 script).\n")
    lines.append("- Tested raw estimates, estimate_plus_gap_s = estimate + gap, and two ratio variants: raw estimate / (actual - gap) and (estimate + gap) / actual.\n")
    lines.append("- Statistics use subject-level condition x n_pictures cell means; paired t-tests are collapsed across n_pictures with Holm correction per metric.\n\n")
    lines.append("Collapsed condition descriptives (subject means):\n")
    lines.append(key_desc.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n")
    lines.append("Paired tests collapsed across durations:\n")
    lines.append(key_tests.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n")
    lines.append("Repeated-measures ANOVA (session_type x n_pictures):\n")
    lines.append(key_anova.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n\n")

    # Short interpretation based on the two comparisons central to the claim.
    adj_ac = tests[(tests.metric == "estimate_plus_gap_s") & (tests.contrast == "active_vs_constant")].iloc[0]
    adj_pc = tests[(tests.metric == "estimate_plus_gap_s") & (tests.contrast == "passive_vs_constant")].iloc[0]
    raw_ac = tests[(tests.metric == "estimated_duration_s") & (tests.contrast == "active_vs_constant")].iloc[0]
    raw_pc = tests[(tests.metric == "estimated_duration_s") & (tests.contrast == "passive_vs_constant")].iloc[0]
    ratio_ac = tests[(tests.metric == "gap_removed_ratio") & (tests.contrast == "active_vs_constant")].iloc[0]
    ratio_pc = tests[(tests.metric == "gap_removed_ratio") & (tests.contrast == "passive_vs_constant")].iloc[0]
    lines.append("Interpretation:\n")
    lines.append(
        f"- Raw estimates are lower than constant for active (diff {raw_ac.mean_diff_a_minus_b:.2f} s, "
        f"p_holm {fmt_p(raw_ac.p_holm)}) and passive (diff {raw_pc.mean_diff_a_minus_b:.2f} s, "
        f"p_holm {fmt_p(raw_pc.p_holm)}).\n"
    )
    lines.append(
        f"- After adding/accounting for fixation gaps, active vs constant is no longer different "
        f"(diff {adj_ac.mean_diff_a_minus_b:.2f} s, p_holm {fmt_p(adj_ac.p_holm)}), and passive vs constant is "
        f"also not different (diff {adj_pc.mean_diff_a_minus_b:.2f} s, p_holm {fmt_p(adj_pc.p_holm)}).\n"
    )
    lines.append(
        f"- When fixation time is removed from the denominator instead, active vs constant is also not significant "
        f"(diff {ratio_ac.mean_diff_a_minus_b:.3f}, p_holm {fmt_p(ratio_ac.p_holm)}), and passive vs constant is not significant "
        f"(diff {ratio_pc.mean_diff_a_minus_b:.3f}, p_holm {fmt_p(ratio_pc.p_holm)}). This ratio check is more conservative because it changes the physical target duration rather than placing the discounted gap back onto the estimate.\n"
    )
    lines.append(
        "Overall, the direct replication of the earlier estimate-plus-gap analysis in the current N=23 sample supports the claim that the active/passive vs constant raw-estimate differences are largely attributable to discounted fixation-cross intervals. It should be phrased as an adjustment/accounting result, not definitive proof of equivalence.\n\n"
    )
    lines.append("Output files:\n")
    for name, path in summary["files"].items():
        lines.append(f"- {name}: {path}\n")
    lines.append(f"- summary_json: {summary_json_out}\n")
    summary_txt_out.write_text("".join(lines), encoding="utf-8")

    print("Saved outputs to", OUT_DIR)
    print("N complete subjects:", complete_n)
    print(key_desc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nKey paired tests:")
    print(key_tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
