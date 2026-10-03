from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # open_data/
FIGURES_DIR = ROOT / "output" / "figures"
# Epoch files (picture_onset-epo.fif, fixation_onset-epo.fif per participant) are derived from the raw
# EEG and are not part of this repository; they are available from the corresponding author on request.
DERIVATIVES_DIR = ROOT / "data" / "derivatives"

SUBJECTS = tuple(f"{s:03d}" for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26))
CONDS = ("active", "passive", "constant")
COND_LABELS = {
    "active": "Scrolling",
    "passive": "Watching",
    "constant": "Baseline",
}
COLORS = {
    "active": "#000000",
    "passive": "#D55E00",
    "constant": "#0072B2",
}
GAP_COLOR = "#BDBDBD"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def subject_string(subject_id: str | int) -> str:
    return str(subject_id).replace("sub-", "").zfill(3)


def assert_canonical_subjects(subjects: list[str] | pd.Series, context: str) -> None:
    found = tuple(sorted({subject_string(s) for s in subjects}))
    if found != SUBJECTS:
        missing = sorted(set(SUBJECTS) - set(found))
        extra = sorted(set(found) - set(SUBJECTS))
        raise ValueError(f"{context}: expected canonical N=23; missing={missing}; extra={extra}")
    print(f"{context}: canonical subjects N=23 ({', '.join(SUBJECTS)})")


def subject_condition_means(
    df: pd.DataFrame,
    value: str,
    *,
    subject: str = "sub",
    condition: str = "session_type",
) -> pd.DataFrame:
    return (
        df.assign(subject_id=lambda x: x[subject].map(subject_string))
        .groupby(["subject_id", condition], as_index=False)[value]
        .mean()
        .rename(columns={condition: "condition", value: "value"})
    )


def bootstrap_ci(values: np.ndarray, *, seed: int = 20260519, n_boot: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return tuple(np.percentile(boot, [2.5, 97.5]))


def bar_summary(df: pd.DataFrame, *, seed: int = 20260519) -> pd.DataFrame:
    rows = []
    for idx, cond in enumerate(CONDS):
        values = df.loc[df["condition"] == cond, "value"].to_numpy(float)
        lo, hi = bootstrap_ci(values, seed=seed + idx)
        rows.append(
            {
                "condition": cond,
                "label": COND_LABELS[cond],
                "mean": values.mean(),
                "ci_low": lo,
                "ci_high": hi,
                "n": values.size,
            }
        )
    return pd.DataFrame(rows)


def draw_bars(ax, summary: pd.DataFrame, *, width: float = 0.62, edgecolor: str = "black") -> np.ndarray:
    x = np.arange(len(CONDS), dtype=float)
    for i, cond in enumerate(CONDS):
        row = summary.loc[summary["condition"] == cond].iloc[0]
        err = [[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]]
        ax.bar(
            x[i],
            row["mean"],
            width=width,
            color=COLORS[cond],
            edgecolor=edgecolor,
            linewidth=0.7,
            yerr=err,
            error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8, "ecolor": "black"},
            zorder=2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABELS[c] for c in CONDS], rotation=20, ha="right")
    return x


def add_subject_lines(ax, df: pd.DataFrame, x: np.ndarray, *, jitter: float = 0.04, seed: int = 20260519) -> None:
    rng = np.random.default_rng(seed)
    wide = df.pivot(index="subject_id", columns="condition", values="value").loc[:, list(CONDS)]
    for _, row in wide.iterrows():
        offset = rng.normal(0, jitter)
        ax.plot(x + offset, row.to_numpy(float), color="#555555", alpha=0.18, linewidth=0.6, zorder=1)
        ax.scatter(x + offset, row.to_numpy(float), color="white", edgecolor="#333333", s=9, linewidth=0.35, alpha=0.65, zorder=3)


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")


def save_figure(fig, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png = FIGURES_DIR / f"{stem}.png"
    pdf = FIGURES_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"saved {png}")
    print(f"saved {pdf}")


def condition_evokeds(epoch_name: str, conditions: tuple[str, ...], *, tmin: float, tmax: float):
    import mne

    evokeds = {cond: [] for cond in conditions}
    included = []
    for sub in SUBJECTS:
        epo_path = DERIVATIVES_DIR / f"sub-{sub}" / epoch_name
        if not epo_path.exists():
            raise FileNotFoundError(epo_path)
        epochs = mne.read_epochs(epo_path, preload=True, verbose="ERROR").crop(tmin=tmin, tmax=tmax)
        missing = [cond for cond in conditions if cond not in epochs.event_id or len(epochs[cond]) == 0]
        if missing:
            raise ValueError(f"sub-{sub} missing {missing} in {epoch_name}")
        for cond in conditions:
            evokeds[cond].append(epochs[cond].average())
        included.append(sub)
        del epochs
    assert_canonical_subjects(included, epoch_name)
    return evokeds


def channel_waveforms(evokeds: list, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for evoked in evokeds:
        picks = [evoked.ch_names.index(ch) for ch in channels if ch in evoked.ch_names]
        if not picks:
            raise ValueError(f"No requested channels found: {channels}")
        rows.append(evoked.data[picks].mean(axis=0) * 1e6)
    return evokeds[0].times * 1000, np.asarray(rows)


def within_subject_ci(condition_arrays: list[np.ndarray]) -> list[np.ndarray]:
    stacked = np.stack(condition_arrays, axis=0)
    subject_mean = stacked.mean(axis=0, keepdims=True)
    grand_mean = stacked.mean(axis=(0, 1), keepdims=True)
    normalized = stacked - subject_mean + grand_mean
    sem = normalized.std(axis=1, ddof=1) / np.sqrt(normalized.shape[1])
    return [1.96 * sem[i] for i in range(normalized.shape[0])]
