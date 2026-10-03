"""A2h: Baseline contrasts by session position (Supplement S4, Figure S4.2B).

Rows "2" and "3": participant mean of Baseline blocks at that position versus participant mean of Watching
blocks at the same position (participants with both). Rows "Baseline at k vs its Scrolling": Baseline blocks
at position k paired with the Scrolling block of the same repetition and block length.
Run from the open_data/ root: python code/revision/analysis_20260915_baseline_contrasts_by_position.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, DATA_DIR, OUT_DIR  # open_data/ paths
TRIAL_CSV = DATA_DIR / "group/behavioral_trial_level.csv"
OUT = OUT_DIR / "revision"
OUT.mkdir(parents=True, exist_ok=True)
COND = {"active": "Scrolling", "passive": "Watching", "constant": "Baseline"}

d = pd.read_csv(TRIAL_CSV)
d = d[d.phase == "time_estimation"].copy()
d["sub"] = d["sub"].astype(str).str.zfill(3)
d["condition"] = d.session_type.map(COND)
d["position"] = ((d.session_order - 1) % 3) + 1
d["ratio"] = d.estimated_duration_s / d.actual_duration_s


def paired(x, y):
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    diff = x - y
    t, p = stats.ttest_rel(x, y)
    lo, hi = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=stats.sem(diff))
    return dict(n=len(diff), baseline=x.mean(), watching=y.mean(), diff=diff.mean(), ci_low=lo, ci_high=hi,
                t=t, p=p, dz=diff.mean() / diff.std(ddof=1))


rows = []
for dv in ["estimated_duration_s", "ratio"]:
    m = d.groupby(["sub", "condition", "position"])[dv].mean()
    for pos in [2, 3]:
        rows.append(dict(dv=dv, position=str(pos), **paired(m.xs(("Baseline", pos), level=(1, 2)),
                                                             m.xs(("Watching", pos), level=(1, 2)))))
    base = d[d.condition == "Baseline"][["sub", "rep", "n_pictures", "position", dv]]
    scr = d[d.condition == "Scrolling"][["sub", "rep", "n_pictures", dv]]
    pr = base.merge(scr, on=["sub", "rep", "n_pictures"], suffixes=("_b", "_s"))
    for pos in [2, 3]:
        s = pr[pr.position == pos].groupby("sub")[[f"{dv}_b", f"{dv}_s"]].mean()
        rows.append(dict(dv=dv, position=f"Baseline at {pos} vs its Scrolling", **paired(s[f"{dv}_b"], s[f"{dv}_s"])))
pd.DataFrame(rows).to_csv(OUT / "A2h_baseline_contrasts_by_position.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string())
