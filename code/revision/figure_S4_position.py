"""Supplementary Figure S4: session position and the Scrolling-Watching contrast.

(A) Watching and Baseline estimates by within-repetition position (2nd vs 3rd), participant means.
(B) Position and context effects (estimation ratio), observed vs predicted by the sequential-assimilation
    model, participant means with 95% CI.
Run: uv run python manuscript/revision-tmb-20260908/analyses/figure_S4_position.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, DATA_DIR, OUT_DIR  # open_data/ paths
TRIAL_CSV = DATA_DIR / "group/behavioral_trial_level.csv"
OUT = OUT_DIR / "revision/figures"
OUT.mkdir(parents=True, exist_ok=True)
COND = {"active": "Scrolling", "passive": "Watching", "constant": "Baseline"}
COL = {"Scrolling": "#222222", "Watching": "#d95f02", "Baseline": "#1f77b4"}

d = pd.read_csv(TRIAL_CSV)
d = d[d.phase == "time_estimation"].copy()
d["sub"] = d["sub"].astype(str).str.zfill(3)
d["condition"] = d.session_type.map(COND)
d["position"] = ((d.session_order - 1) % 3) + 1

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"width_ratios": [1, 1.05]})

# (A) Watching / Baseline by position
ax = axes[0]
pm = d[d.condition.isin(["Watching", "Baseline"])].groupby(["sub", "condition", "position"])["estimated_duration_s"].mean().unstack("position")
x0 = {"Watching": 0.0, "Baseline": 1.6}
for c in ["Watching", "Baseline"]:
    sub = pm.xs(c, level="condition").dropna()
    xs = np.array([x0[c], x0[c] + 0.6])
    for _, row in sub.iterrows():
        ax.plot(xs, [row[2], row[3]], color=COL[c], alpha=0.25, lw=0.8)
    m = sub.mean().values
    ci = stats.t.interval(0.95, len(sub) - 1, m, stats.sem(sub.values, axis=0))
    ax.errorbar(xs, m, yerr=[m - ci[0], ci[1] - m], fmt="o-", color=COL[c], lw=2, capsize=3, zorder=5)
ax.set_xticks([0, 0.6, 1.6, 2.2])
ax.set_xticklabels(["2nd\nafter\nScrolling", "3rd\nafter\nBaseline", "2nd\nafter\nScrolling", "3rd\nafter\nWatching"], fontsize=8)
ax.text(0.3, ax.get_ylim()[1] * 0.98, "Watching", ha="center", va="top", color=COL["Watching"], fontweight="bold")
ax.text(1.9, ax.get_ylim()[1] * 0.98, "Baseline", ha="center", va="top", color=COL["Baseline"], fontweight="bold")
ax.set_ylabel("Estimated duration (s)")
ax.set_xlabel("Position within repetition")
ax.set_title("A  Estimates by session position", loc="left")

# (B) sequential assimilation: observed vs predicted position and context effects
# (from analysis_20261001_sequential_assimilation.py; run that script first)
ax = axes[1]
SA = OUT_DIR / "revision/sequential_assimilation_20261001"
pv = pd.read_csv(SA / "predicted_vs_observed_ci.csv")
g = pd.read_csv(SA / "models.csv").query("model == 'M_lag1' and term == 'lag1_c'").iloc[0].b
rows = [("Watching: 3rd − 2nd", "Watching 3rd − 2nd"), ("Baseline: 3rd − 2nd", "Baseline 3rd − 2nd"),
        ("Scrolling − Watching, Watching 2nd", "S − W, Watching\nafter Scrolling"),
        ("Scrolling − Watching, Watching 3rd", "S − W, Watching\nafter Baseline")]
for i, (q, _) in enumerate(rows):
    for src, off, face in [("Observed", -0.15, "#222222"), ("Assimilation model", 0.15, "white")]:
        r = pv[(pv.quantity == q) & (pv.source == src)].iloc[0]
        ax.plot([r.ci_low, r.ci_high], [i + off] * 2, color="#222222", lw=1.2, zorder=2)
        ax.plot(r["mean"], i + off, "o", ms=6, mfc=face, mec="#222222", mew=0.9, zorder=3,
                label=(f"Assimilation model (g = {g:.2f})" if src != "Observed" else "Observed") if i == 0 else None)
ax.axvline(0, color="#999999", lw=0.8, ls="--", zorder=0)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([lab for _, lab in rows])
ax.invert_yaxis()
ax.set_xlabel("Difference (estimation ratio)")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2)
ax.set_title("B  Sequential assimilation", loc="left")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"figS4_position_context.{ext}", dpi=300, bbox_inches="tight")
print("saved", OUT / "figS4_position_context.png")
