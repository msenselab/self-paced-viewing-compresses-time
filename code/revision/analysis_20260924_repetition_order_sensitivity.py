"""Repetition-level and run-order sensitivity analyses for the Scrolling-first confound.

Design facts used here (run_experiment.py):
- 4 repetitions x 3 condition runs x 3 blocks = 36 blocks. Scrolling always opened a
  repetition; Watching and Baseline alternated between runs 2 and 3.
- All three runs of a repetition used the same three block durations (yoked to Scrolling).
- Every run began with an instruction screen and ended with an optional short break; after
  runs 3, 6 and 9 (end of repetitions 1-3) the eye tracker was recalibrated. Scrolling runs
  in repetitions 2-4 therefore always followed the recalibration break; the repetition-1
  Scrolling run followed the practice block.

Analyses (raw estimate primary, estimation ratio secondary):
1. Paired condition contrasts within each repetition (participant means; durations are
   matched within a repetition, so no duration adjustment is needed).
2. Condition (Scrolling, Watching) x Repetition likelihood-ratio test (ML fits).
3. Sensitivity of the Scrolling - Watching contrast to the first and last run.
   Primary: participant-level matched pairs (Scrolling and Watching blocks of the same
   repetition and block length, the estimator of the main-text contrast), with and without
   pairs that involve run 1 or run 12. Pairing within a repetition conditions on repetition.
   Secondary: block-level mixed models, all runs vs runs 2-11, with no time term, a linear
   run-order term, or repetition as a factor.
4. Descriptive: condition contrasts at each block position within a run (1st/2nd/3rd).

Mixed models follow the Supplement S4.6 convention: participant random intercept and random
slopes for block length (9/12/18 pictures, also a fixed factor), Watching as the reference
condition. Block-level models are less precise than the paired estimator because participants
differ widely in how their estimates scale with block length.

Run:
    uv run python manuscript/revision-tmb-20260908/analyses/analysis_20260924_repetition_order_sensitivity.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, DATA_DIR, OUT_DIR  # open_data/ paths
TRIAL_CSV = DATA_DIR / "group/behavioral_trial_level.csv"
OUT = OUT_DIR / "revision/repetition_order_sensitivity_20260924"
OUT.mkdir(parents=True, exist_ok=True)

COND = {"active": "Scrolling", "passive": "Watching", "constant": "Baseline"}
CANON_EXCL = {"015", "016", "023", "027"}
DVS = [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]
REF = "C(condition, Treatment('Watching'))"

raw = pd.read_csv(TRIAL_CSV)
raw["sub"] = raw["sub"].astype(str).str.zfill(3)
d = raw[(raw["phase"] == "time_estimation") & ~raw["sub"].isin(CANON_EXCL)].copy()
d["condition"] = d["session_type"].map(COND)
d["run"] = d["session_order"].astype(int)
d["position"] = (d["run"] - 1) % 3 + 1
d["ratio"] = d["estimated_duration_s"] / d["actual_duration_s"]
d["npic"] = pd.Categorical(d["n_pictures"].astype(int), [9, 12, 18])
d["run_c"] = d["run"] - 6.5
d["block_in_run"] = d["block_order_in_session"].astype(int)

# Design checks.
assert d["sub"].nunique() == 23
assert (d["rep"] == (d["run"] - 1) // 3 + 1).all()
assert (d.loc[d["condition"] == "Scrolling", "position"] == 1).all()
assert set(d.loc[d["condition"] != "Scrolling", "position"]) == {2, 3}
assert (d.groupby(["sub", "run"]).size() == 3).all()


def paired_row(diff: pd.Series, **labels) -> dict:
    diff = diff.dropna()
    n = len(diff)
    t, p = stats.ttest_1samp(diff, 0.0)
    lo, hi = stats.t.interval(0.95, n - 1, loc=diff.mean(), scale=stats.sem(diff))
    return {**labels, "n": n, "mean_diff": diff.mean(), "sd_diff": diff.std(ddof=1),
            "ci_low": lo, "ci_high": hi, "t": t, "df": n - 1, "p": p,
            "dz": diff.mean() / diff.std(ddof=1)}


def cond_effects(fit, dv_label: str, **labels) -> list[dict]:
    ci = fit.conf_int()
    rows = []
    for name, lab in [("Scrolling", "Scrolling - Watching"), ("Baseline", "Baseline - Watching")]:
        term = f"{REF}[T.{name}]"
        rows.append({**labels, "dv": dv_label, "contrast": lab, "b": fit.params[term],
                     "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1], "p": fit.pvalues[term]})
    return rows


results: dict = {"trial_csv": str(TRIAL_CSV), "canonical_exclusions": sorted(CANON_EXCL)}

# 1. Paired contrasts within each repetition.
rep_rows = []
for dv, dv_label in DVS:
    m = d.groupby(["sub", "rep", "condition"])[dv].mean().unstack("condition")
    for rep, g in m.groupby(level="rep"):
        for a, b in [("Scrolling", "Watching"), ("Baseline", "Watching"), ("Baseline", "Scrolling")]:
            rep_rows.append(paired_row(g[a] - g[b], dv=dv_label, repetition=rep, contrast=f"{a} - {b}"))
rep_tab = pd.DataFrame(rep_rows)
rep_tab.to_csv(OUT / "paired_contrasts_by_repetition.csv", index=False)

# Repetition level (all conditions): participant means per repetition, paired vs repetition 1.
lvl = d.groupby(["sub", "rep"])["estimated_duration_s"].mean().unstack("rep")
level_rows = [paired_row(lvl[r] - lvl[1], contrast=f"repetition {r} - repetition 1") for r in [2, 3, 4]]
level_rows += [paired_row(lvl[4] - lvl[2], contrast="repetition 4 - repetition 2")]
pd.DataFrame(level_rows).to_csv(OUT / "repetition_level_shift.csv", index=False)

# 2. Condition x Repetition for Scrolling vs Watching (ML likelihood-ratio test).
lr_rows = []
sw = d[d["condition"] != "Baseline"].copy()
for dv, dv_label in DVS:
    m0 = smf.mixedlm(f"{dv} ~ C(condition) + C(rep) + C(npic)", sw, groups=sw["sub"]).fit(reml=False)
    m1 = smf.mixedlm(f"{dv} ~ C(condition) * C(rep) + C(npic)", sw, groups=sw["sub"]).fit(reml=False)
    lr = 2 * (m1.llf - m0.llf)
    lr_rows.append({"dv": dv_label, "test": "Scrolling/Watching x Repetition", "chi2": lr, "df": 3,
                    "p": stats.chi2.sf(lr, 3), "n_sub": sw["sub"].nunique(), "n_blocks": len(sw)})
pd.DataFrame(lr_rows).to_csv(OUT / "condition_by_repetition_lrt.csv", index=False)

# 3a. Matched-pair sensitivity (primary).
sc = d[d["condition"] == "Scrolling"][["sub", "rep", "n_pictures", "run", "estimated_duration_s", "ratio"]]
wa = d[d["condition"] == "Watching"][["sub", "rep", "n_pictures", "run", "estimated_duration_s", "ratio"]]
pairs = sc.merge(wa, on=["sub", "rep", "n_pictures"], suffixes=("_s", "_w"))
assert len(pairs) == len(sc)
pair_rows = []
for dv, dv_label in DVS:
    pairs["diff"] = pairs[f"{dv}_s"] - pairs[f"{dv}_w"]
    for lab, q in [("all pairs", pairs), ("runs 1 and 12 removed", pairs[(pairs.run_s > 1) & (pairs.run_w < 12)])]:
        row = paired_row(q.groupby("sub")["diff"].mean(), dv=dv_label, sample=lab, contrast="Scrolling - Watching")
        row["n_pairs"] = len(q)
        pair_rows.append(row)
pd.DataFrame(pair_rows).to_csv(OUT / "matched_pair_sensitivity.csv", index=False)

# 3b. Block-level sensitivity models (secondary).
inner = d[(d["run"] > 1) & (d["run"] < 12)].copy()
specs = [
    ("M1", "all runs", d, ""),
    ("M2", "all runs", d, " + C(rep)"),
    ("M3", "runs 2-11", inner, ""),
    ("M4", "runs 2-11", inner, " + run_c"),
    ("M5", "runs 2-11", inner, " + C(rep)"),
]
time_label = {"": "none", " + run_c": "run order (linear)", " + C(rep)": "repetition (factor)"}
sens_rows = []
for dv, dv_label in DVS:
    for key, data_label, data, time_term in specs:
        model = smf.mixedlm(f"{dv} ~ {REF} + C(npic){time_term}", data, groups=data["sub"],
                            re_formula="~C(npic)").fit(reml=True, method=["lbfgs"])
        conv = bool(model.converged)
        sens_rows += cond_effects(model, dv_label, model=key, data=data_label, time_term=time_label[time_term],
                                  n_sub=data["sub"].nunique(), n_blocks=len(data), converged=conv)
sens = pd.DataFrame(sens_rows)
sens.to_csv(OUT / "sensitivity_models.csv", index=False)

# 4. Condition contrasts at each block position within a run (descriptive).
blk_rows = []
for dv, dv_label in DVS:
    for b in [1, 2, 3]:
        sub_d = d[d["block_in_run"] == b]
        model = smf.mixedlm(f"{dv} ~ {REF} + C(npic) + C(rep)", sub_d, groups=sub_d["sub"],
                            re_formula="~C(npic)").fit(reml=True, method=["lbfgs"])
        blk_rows += cond_effects(model, dv_label, block_in_run=b, n_blocks=len(sub_d))
pd.DataFrame(blk_rows).to_csv(OUT / "contrasts_by_block_in_run.csv", index=False)

results.update({
    "paired_by_repetition": rep_rows,
    "repetition_level_shift": level_rows,
    "condition_by_repetition_lrt": lr_rows,
    "matched_pair_sensitivity": pair_rows,
    "sensitivity_models": sens_rows,
    "contrasts_by_block_in_run": blk_rows,
})
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")

pd.set_option("display.width", 200)
print("Paired contrasts by repetition (estimate, s)")
print(rep_tab[rep_tab.dv == "estimate_s"].round(3).to_string(index=False))
print("\nRepetition level shift (estimate, s)")
print(pd.DataFrame(level_rows).round(3).to_string(index=False))
print("\nCondition x Repetition LRT")
print(pd.DataFrame(lr_rows).round(4).to_string(index=False))
print("\nMatched-pair sensitivity")
print(pd.DataFrame(pair_rows).round(3).to_string(index=False))
print("\nSensitivity models")
print(sens.round(3).to_string(index=False))
print("\nContrasts by block position within run")
print(pd.DataFrame(blk_rows).round(3).to_string(index=False))
