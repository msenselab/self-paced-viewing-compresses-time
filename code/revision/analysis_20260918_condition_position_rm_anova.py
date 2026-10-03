"""Primary 2 x 2 repeated-measures ANOVA for reviewer order concern.

Factors:
- condition: Watching vs Baseline
- position: 2nd vs 3rd within repetition

Participant-level cell means are used. Participants missing any of the four cells
are excluded from both DVs. Follow-up position contrasts are paired within each
condition and Holm-corrected across Watching and Baseline.

Run:
    uv run python manuscript/revision-tmb-20260908/analyses/analysis_20260918_condition_position_rm_anova.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, DATA_DIR, OUT_DIR  # open_data/ paths
TRIAL_CSV = DATA_DIR / "group/behavioral_trial_level.csv"
OUT = OUT_DIR / "revision/condition_position_rm_anova_20260918"
OUT.mkdir(parents=True, exist_ok=True)

COND = {"active": "Scrolling", "passive": "Watching", "constant": "Baseline"}
CANON_EXCL = {"015", "016", "023", "027"}

raw = pd.read_csv(TRIAL_CSV)
raw["sub"] = raw["sub"].astype(str).str.zfill(3)
raw = raw[(raw["phase"] == "time_estimation") & ~raw["sub"].isin(CANON_EXCL)].copy()
raw["condition"] = raw["session_type"].map(COND)
raw["position"] = ((raw["session_order"] - 1) % 3) + 1
raw["ratio"] = raw["estimated_duration_s"] / raw["actual_duration_s"]
wb = raw[raw["condition"].isin(["Watching", "Baseline"]) & raw["position"].isin([2, 3])].copy()

# Verify the design coding before aggregation.
assert set(wb["condition"].dropna().unique()) == {"Watching", "Baseline"}
assert set(wb["position"].unique()) == {2, 3}
assert not ((wb["condition"] == "Scrolling")).any()

cell_counts = wb.groupby(["sub", "condition", "position"]).size().rename("n_blocks").reset_index()
cell_counts.to_csv(OUT / "cell_counts.csv", index=False)

results = {"trial_csv": str(TRIAL_CSV), "canonical_exclusions": sorted(CANON_EXCL), "dvs": {}}
all_anova = []
all_desc = []
all_follow = []

for dv, dv_label in [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]:
    cell = wb.groupby(["sub", "condition", "position"], as_index=False)[dv].mean()
    wide = cell.pivot(index="sub", columns=["condition", "position"], values=dv)
    required = pd.MultiIndex.from_product([["Watching", "Baseline"], [2, 3]], names=["condition", "position"])
    wide = wide.reindex(columns=required)
    complete_subs = wide.dropna().index.tolist()
    cell_complete = cell[cell["sub"].isin(complete_subs)].copy()
    assert cell_complete["sub"].nunique() == len(complete_subs)
    assert len(cell_complete) == 4 * len(complete_subs)

    aov = pg.rm_anova(
        data=cell_complete,
        dv=dv,
        within=["condition", "position"],
        subject="sub",
        correction=False,
        detailed=True,
        effsize="np2",
    )
    aov.insert(0, "dv", dv_label)
    aov.insert(1, "n", len(complete_subs))
    all_anova.append(aov)

    desc = (
        cell_complete.groupby(["condition", "position"])[dv]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    desc.insert(0, "dv", dv_label)
    all_desc.append(desc)

    follow_rows = []
    for condition in ["Watching", "Baseline"]:
        w = wide.loc[complete_subs, condition]
        diff = w[3] - w[2]
        t, p = stats.ttest_rel(w[3], w[2])
        ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
        follow_rows.append({
            "dv": dv_label,
            "condition": condition,
            "contrast": "3rd - 2nd",
            "n": len(diff),
            "mean_2nd": w[2].mean(),
            "sd_2nd": w[2].std(ddof=1),
            "mean_3rd": w[3].mean(),
            "sd_3rd": w[3].std(ddof=1),
            "mean_diff": diff.mean(),
            "sd_diff": diff.std(ddof=1),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "t": t,
            "df": len(diff) - 1,
            "p_unc": p,
            "dz": diff.mean() / diff.std(ddof=1),
        })
    p_holm = multipletests([r["p_unc"] for r in follow_rows], method="holm")[1]
    for row, adjusted in zip(follow_rows, p_holm):
        row["p_holm"] = adjusted
    all_follow.append(pd.DataFrame(follow_rows))

    results["dvs"][dv_label] = {
        "n_complete": len(complete_subs),
        "excluded_for_incomplete_cells": sorted(set(cell["sub"]) - set(complete_subs)),
        "anova": aov.replace({np.nan: None}).to_dict(orient="records"),
        "descriptives": desc.to_dict(orient="records"),
        "followups": follow_rows,
    }

anova = pd.concat(all_anova, ignore_index=True)
descriptives = pd.concat(all_desc, ignore_index=True)
followups = pd.concat(all_follow, ignore_index=True)
anova.to_csv(OUT / "anova_2x2.csv", index=False)
descriptives.to_csv(OUT / "cell_descriptives.csv", index=False)
followups.to_csv(OUT / "position_followups_holm.csv", index=False)
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

print("Complete-case N by DV:")
for dv, value in results["dvs"].items():
    print(f"  {dv}: N={value['n_complete']}; excluded={value['excluded_for_incomplete_cells']}")
print("\n2 x 2 repeated-measures ANOVA")
print(anova[["dv", "n", "Source", "ddof1", "ddof2", "F", "p_unc", "np2"]].round(6).to_string(index=False))
print("\nCell descriptives")
print(descriptives.round(6).to_string(index=False))
print("\nPosition follow-ups (Holm within DV)")
print(followups.round(6).to_string(index=False))
