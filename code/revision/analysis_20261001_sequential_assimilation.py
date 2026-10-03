"""Post hoc: does sequential assimilation explain the asymmetric position effect (Figure S4.1A)?

Watching shifted with position and Baseline did not. Watching follows either Scrolling (short estimates) or
Baseline (long estimates); Baseline follows Scrolling or Watching (both short). Assimilation of each estimate
toward the preceding one predicts exactly this asymmetry. Analysis added after the data were seen
(post hoc, exploratory; author suggestion). Model:
    current estimate = condition-specific estimate + g * (preceding estimate - condition-specific estimate)
fitted on the estimation ratio (estimate / actual duration), because successive blocks differ in length.

Block-level linear mixed models, participant random intercept, block length and repetition as factors:
  M_pos      ratio ~ condition * position                       (the position effect to be explained)
  M_lag1     ratio ~ condition + lag1                           (assimilation to the preceding block)
  M_lag1pos  ratio ~ condition * position + lag1                (does the interaction survive assimilation?)
  M_seq      ratio ~ condition + prevseq                        (assimilation to the preceding sequence mean)
  M_seqpos   ratio ~ condition * position + prevseq
Predicted vs observed: fitted values of the assimilation models (without position terms) are averaged the
same way as Figure S4.1 (Watching/Baseline by position; Scrolling - Watching by Watching position).
Run from the open_data/ root: python code/revision/analysis_20261001_sequential_assimilation.py
"""
from pathlib import Path
import json
import warnings

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
OUT = OUT_DIR / "revision/sequential_assimilation_20261001"
OUT.mkdir(parents=True, exist_ok=True)
COND = {"active": "Scrolling", "passive": "Watching", "constant": "Baseline"}

d = pd.read_csv(TRIAL_CSV)
d = d[d.phase == "time_estimation"].copy()
d["sub"] = d["sub"].astype(str).str.zfill(3)
d["cond"] = pd.Categorical(d.session_type.map(COND), ["Watching", "Scrolling", "Baseline"])
d["position"] = ((d.session_order - 1) % 3) + 1
d["ratio"] = d.estimated_duration_s / d.actual_duration_s
d = d.sort_values(["sub", "session_order", "block_order_in_session"]).reset_index(drop=True)
d["lag1"] = d.groupby("sub")["ratio"].shift(1)
d["prev_cond"] = d.groupby("sub")["cond"].shift(1)
seq = d.groupby(["sub", "session_order"], as_index=False).agg(seq_mean=("ratio", "mean"))
seq["prevseq"] = seq.groupby("sub")["seq_mean"].shift(1)
d = d.merge(seq[["sub", "session_order", "prevseq"]], on=["sub", "session_order"], how="left")
# common rows: every block that has a preceding sequence (drops the first sequence of each session)
m = d.dropna(subset=["lag1", "prevseq"]).copy()
for v in ["lag1", "prevseq"]:
    m[v + "_c"] = m[v] - m[v].mean()
m["pos"] = m.position.astype(str)
m["len"] = m.n_pictures.astype(str)
m["rep_f"] = m.rep.astype(str)
# Scrolling is always 1st, so position is coded within Watching and Baseline (interaction = W3 - B3)
m["W3"] = ((m.cond == "Watching") & (m.position == 3)).astype(int)
m["B3"] = ((m.cond == "Baseline") & (m.position == 3)).astype(int)

COV = " + C(len) + C(rep_f)"
specs = {
    "M_pos": "ratio ~ cond + W3 + B3" + COV,
    "M_lag1": "ratio ~ cond + lag1_c" + COV,
    "M_lag1pos": "ratio ~ cond + W3 + B3 + lag1_c" + COV,
    "M_seq": "ratio ~ cond + prevseq_c" + COV,
    "M_seqpos": "ratio ~ cond + W3 + B3 + prevseq_c" + COV,
}
fits, rows = {}, []
for name, f in specs.items():
    fit = smf.mixedlm(f, m, groups=m["sub"]).fit(reml=False)
    fits[name] = fit
    ci = fit.conf_int()
    for term in fit.params.index:
        if term == "Group Var" or term.startswith("C(len)") or term.startswith("C(rep_f)") or term == "Intercept":
            continue
        rows.append(dict(model=name, term=term, b=fit.params[term], ci_low=ci.loc[term, 0], ci_high=ci.loc[term, 1],
                         p=fit.pvalues[term]))
    rows.append(dict(model=name, term="AIC", b=fit.aic))
tab = pd.DataFrame(rows)
for name in ["M_pos", "M_lag1pos", "M_seqpos"]:  # interaction contrast W3 - B3
    fit = fits[name]; c = np.zeros(len(fit.fe_params)); idx = list(fit.fe_params.index)
    c[idx.index("W3")], c[idx.index("B3")] = 1, -1
    est = float(c @ fit.fe_params); se = float(np.sqrt(c @ fit.cov_params().loc[idx, idx].values @ c))
    tab.loc[len(tab)] = dict(model=name, term="W3 - B3 (interaction)", b=est, ci_low=est - 1.96 * se,
                             ci_high=est + 1.96 * se, p=2 * stats.norm.sf(abs(est / se)))
tab.to_csv(OUT / "models.csv", index=False)

# --- predicted vs observed, aggregated like Figure S4.1
def summarise(df, col):
    w = df[df.cond == "Watching"].groupby(["sub", "position"])[col].mean().unstack()
    b = df[df.cond == "Baseline"].groupby(["sub", "position"])[col].mean().unstack()
    sc = df[df.cond == "Scrolling"][["sub", "rep", "n_pictures", col]]
    wa = df[df.cond == "Watching"][["sub", "rep", "n_pictures", "position", col]]
    pr = sc.merge(wa, on=["sub", "rep", "n_pictures"], suffixes=("_s", "_w"))
    pr["diff"] = pr[f"{col}_s"] - pr[f"{col}_w"]
    split = pr.groupby(["sub", "position"])["diff"].mean().unstack()
    return {"Watching 3rd - 2nd": (w[3] - w[2]).mean(), "Baseline 3rd - 2nd": (b[3] - b[2]).mean(),
            "S - W, Watching 2nd": split[2].mean(), "S - W, Watching 3rd": split[3].mean()}

m["fit_lag1"] = fits["M_lag1"].fittedvalues
m["fit_seq"] = fits["M_seq"].fittedvalues
comp = pd.DataFrame({"observed": summarise(m, "ratio"), "lag-1 model": summarise(m, "fit_lag1"),
                     "sequence model": summarise(m, "fit_seq")})
comp.to_csv(OUT / "predicted_vs_observed.csv")


def per_participant(df, col):
    w = df[df.cond == "Watching"].groupby(["sub", "position"])[col].mean().unstack()
    b = df[df.cond == "Baseline"].groupby(["sub", "position"])[col].mean().unstack()
    sc = df[df.cond == "Scrolling"][["sub", "rep", "n_pictures", col]]
    wa = df[df.cond == "Watching"][["sub", "rep", "n_pictures", "position", col]]
    pr = sc.merge(wa, on=["sub", "rep", "n_pictures"], suffixes=("_s", "_w"))
    pr["diff"] = pr[f"{col}_s"] - pr[f"{col}_w"]
    split = pr.groupby(["sub", "position"])["diff"].mean().unstack()
    return {"Watching: 3rd − 2nd": w[3] - w[2], "Baseline: 3rd − 2nd": b[3] - b[2],
            "Scrolling − Watching, Watching 2nd": split[2], "Scrolling − Watching, Watching 3rd": split[3]}


ci_rows = []
for src, col in [("Observed", "ratio"), ("Assimilation model", "fit_lag1")]:
    for q, s in per_participant(m, col).items():
        s = s.dropna()
        lo, hi = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
        ci_rows.append(dict(quantity=q, source=src, n=len(s), mean=s.mean(), ci_low=lo, ci_high=hi))
pd.DataFrame(ci_rows).to_csv(OUT / "predicted_vs_observed_ci.csv", index=False)

# --- carry-over at condition switches (first block of a sequence) vs within a sequence
m["first"] = (m.block_order_in_session == 1).astype(int)
m["lag_switch"] = m.lag1_c * m["first"]
m["lag_within"] = m.lag1_c * (1 - m["first"])
m["sub_cond"] = m["sub"] + "_" + m.cond.astype(str)
sw_rows = []
for label, extra in [("participant intercept", {}), ("+ participant x condition variance", {"vc_formula": {"sc": "0 + C(sub_cond)"}})]:
    fit = smf.mixedlm("ratio ~ cond + first + lag_switch + lag_within" + COV, m, groups=m["sub"], **extra).fit(reml=False)
    ci = fit.conf_int()
    for k in ["lag_switch", "lag_within"]:
        sw_rows.append(dict(model=label, term=k, g=fit.params[k], ci_low=ci.loc[k, 0], ci_high=ci.loc[k, 1], p=fit.pvalues[k]))
pd.DataFrame(sw_rows).to_csv(OUT / "carryover_switch_vs_within.csv", index=False)

# --- implied condition effect without assimilation: b / (1 - g), CI by parametric draws
res = {}
for name, g_term in [("M_lag1", "lag1_c"), ("M_seq", "prevseq_c")]:
    fit = fits[name]
    keys = ["cond[T.Scrolling]", "cond[T.Baseline]", g_term]
    mu = fit.params[keys].values
    cov = fit.cov_params().loc[keys, keys].values
    draws = np.random.default_rng(1).multivariate_normal(mu, cov, 20000)
    for i, lab in enumerate(["Scrolling - Watching", "Baseline - Watching"]):
        imp = draws[:, i] / (1 - draws[:, 2])
        res[f"{name}: {lab}, implied b/(1-g)"] = [mu[i] / (1 - mu[2]), *np.percentile(imp, [2.5, 97.5])]
pd.DataFrame(res, index=["estimate", "ci_low", "ci_high"]).T.to_csv(OUT / "implied_condition_effects.csv")

pd.set_option("display.width", 200)
print(f"blocks analysed: {len(m)} from {m['sub'].nunique()} participants\n")
print(tab[tab.term.str.contains("lag1|prevseq|Scrolling|Baseline|W3|B3|AIC")].round(4).to_string(index=False))
print("\npredicted vs observed (ratio):\n", comp.round(4).to_string())
print("\ncarry-over at switches vs within:\n", pd.DataFrame(sw_rows).round(3).to_string(index=False))
print("\nimplied condition effects:\n", pd.DataFrame(res, index=["est", "lo", "hi"]).T.round(4).to_string())
