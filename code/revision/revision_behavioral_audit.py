"""Revision round 1 (TMB-2026-0261): reviewer-requested behavioral checks.

Implements analysis-spec.md items A1 (paired-contrast recomputation and
precision), A2 (session-position test for Watching/Baseline; repetition
trend), A3 (RM-ANOVA sensitivity bound), and A7 (yoking verification).

Run:  uv run python manuscript/revision-tmb-20260908/analyses/revision_behavioral_audit.py
Outputs: manuscript/revision-tmb-20260908/analyses/output/
"""
from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.stats.multitest import multipletests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, DATA_DIR, OUT_DIR  # open_data/ paths
TRIAL_CSV = DATA_DIR / "group/behavioral_trial_level.csv"
OUT = OUT_DIR / "revision"
OUT.mkdir(parents=True, exist_ok=True)

COND = {"active": "Scrolling", "passive": "Watching", "constant": "Baseline"}
CANON_EXCL = {"015", "016", "023", "027"}
results: dict = {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def dz_ci(t: float, n: int, conf: float = 0.95):
    df = n - 1
    alpha = 1 - conf

    def cdf(d):
        v = stats.nct.cdf(t, df, d)
        return (1.0 if d < t else 0.0) if np.isnan(v) else v

    span = 60 + abs(t)
    lo = optimize.brentq(lambda d: cdf(d) - (1 - alpha / 2), t - span, t + span)
    hi = optimize.brentq(lambda d: cdf(d) - alpha / 2, t - span, t + span)
    return t / np.sqrt(n), lo / np.sqrt(n), hi / np.sqrt(n)


def paired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    d = x - y
    n = len(d)
    t, p = stats.ttest_rel(x, y)
    dz, lo, hi = dz_ci(t, n)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d))
    r = stats.pearsonr(x, y)[0]
    return dict(n=n, mean_x=x.mean(), sd_x=x.std(ddof=1), mean_y=y.mean(),
                sd_y=y.std(ddof=1), mean_diff=d.mean(), sd_diff=d.std(ddof=1),
                se_diff=stats.sem(d), ci_low=ci[0], ci_high=ci[1],
                r_within=r, t=t, df=n - 1, p=p, dz=dz, dz_lo=lo, dz_hi=hi,
                # what the between-person SDs would imply if (wrongly) treated as independent
                t_if_independent=(x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / n + y.var(ddof=1) / n))


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------
raw = pd.read_csv(TRIAL_CSV)
raw["sub"] = raw["sub"].astype(str).str.zfill(3)
# sub-027 is absent from the canonical file; take its rows from the all-subjects QC file
ALL_CSV = DATA_DIR / "group/behavioral_trial_level_excluded.csv"  # sub-015, 016, 027 (sensitivity samples only)
allsub = pd.read_csv(ALL_CSV)
allsub["sub"] = allsub["sub"].astype(str).str.zfill(3)
extra = allsub[allsub["sub"].isin(["015", "016", "027"])]
raw = pd.concat([raw, extra[raw.columns.intersection(extra.columns)]], ignore_index=True)
raw = raw[raw["phase"] == "time_estimation"].copy()
raw["condition"] = raw["session_type"].map(COND)
raw["position"] = ((raw["session_order"] - 1) % 3) + 1  # 1..3 within repetition
raw["ratio"] = raw["estimated_duration_s"] / raw["actual_duration_s"]
subs_in_file = sorted(raw["sub"].unique())
canon = raw[~raw["sub"].isin(CANON_EXCL)].copy()
keep027 = raw[~raw["sub"].isin(CANON_EXCL - {"027"})].copy()
keep_all = raw[~raw["sub"].isin({"023"})].copy()  # N = 26: only the incomplete session excluded
results["n_keep_all"] = keep_all["sub"].nunique()
results["subjects_in_file"] = subs_in_file
results["n_canonical"] = canon["sub"].nunique()
results["n_keep027"] = keep027["sub"].nunique()
print("subjects in file:", subs_in_file)
print("canonical N =", results["n_canonical"], "| keep-027 N =", results["n_keep027"])
assert results["n_canonical"] == 23

# sanity: position coding — Scrolling must always be position 1
xt = pd.crosstab(canon["condition"], canon["position"])
print(xt)
assert xt.loc["Scrolling", 1] == len(canon[canon.condition == "Scrolling"])
results["position_crosstab"] = xt.to_dict()


# ==========================================================================
# A1. paired-contrast recomputation and precision
# ==========================================================================
def contrast_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for dv, name in [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]:
        wide = df.groupby(["sub", "condition"])[dv].mean().unstack()
        pairs = [("Scrolling", "Watching"), ("Scrolling", "Baseline"), ("Watching", "Baseline")]
        recs = [dict(sample=label, dv=name, contrast=f"{a} - {b}", **paired(wide[a], wide[b]))
                for a, b in pairs]
        ps = [r["p"] for r in recs]
        holm = multipletests(ps, method="holm")[1]
        for r, ph in zip(recs, holm):
            r["p_holm"] = ph
        rows += recs
        # condition descriptives (between-person SD of condition means)
        results.setdefault("descriptives", {})[f"{label}:{name}"] = {
            c: dict(mean=float(wide[c].mean()), sd_between=float(wide[c].std(ddof=1)))
            for c in wide.columns}
    return pd.DataFrame(rows)


a1 = pd.concat([contrast_table(canon, "canonical_N23"), contrast_table(keep027, "keep027_N24"),
                contrast_table(keep_all, "keep015_016_027_N26")])
a1.to_csv(OUT / "A1_paired_contrasts.csv", index=False)
print("\n=== A1 paired contrasts ===")
print(a1[["sample", "dv", "contrast", "mean_diff", "sd_diff", "ci_low", "ci_high", "r_within",
          "t", "p", "p_holm", "dz", "dz_lo", "dz_hi", "t_if_independent"]].round(3).to_string())

# participant-level paired data for a supplementary figure/table
wide_est = canon.groupby(["sub", "condition"])["estimated_duration_s"].mean().unstack()
wide_est.to_csv(OUT / "A1_participant_condition_means_estimate.csv")

# 3x3 RM-ANOVA reproduction (raw estimate and ratio) to confirm the submitted omnibus
for dv, name in [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]:
    cellm = canon.groupby(["sub", "condition", "n_pictures"])[dv].mean().reset_index()
    aov = pg.rm_anova(data=cellm, dv=dv, within=["condition", "n_pictures"], subject="sub",
                      correction=True, detailed=True)
    aov.columns = [c.replace("-", "_") for c in aov.columns]
    aov["np2"] = aov["F"] * aov["ddof1"] / (aov["F"] * aov["ddof1"] + aov["ddof2"])
    aov.to_csv(OUT / f"A1_rm_anova_{name}.csv", index=False)
    results.setdefault("rm_anova", {})[name] = aov.to_dict(orient="records")
    print(f"\n=== A1 RM-ANOVA {name} ===")
    print(aov[["Source", "F", "ddof1", "ddof2", "p_unc", "p_GG_corr", "eps", "np2"]].round(4).to_string())


# ==========================================================================
# A2. session-position test (Watching / Baseline at position 2 vs 3)
# ==========================================================================
wb = canon[canon["condition"].isin(["Watching", "Baseline"])].copy()
wb["pos3"] = (wb["position"] == 3).astype(int)
wb["rep_c"] = wb["rep"] - 2.5
wb["cond"] = pd.Categorical(wb["condition"], ["Watching", "Baseline"])
wb["npic"] = pd.Categorical(wb["n_pictures"].astype(int), [9, 12, 18])

a2 = {}
for dv, name in [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]:
    m = smf.mixedlm(f"{dv} ~ cond * pos3 + C(npic) + rep_c", wb, groups=wb["sub"]).fit(reml=True)
    ci = m.conf_int()
    tab = pd.DataFrame({"estimate": m.params, "se": m.bse, "ci_low": ci[0], "ci_high": ci[1],
                        "z": m.tvalues, "p": m.pvalues})
    tab.to_csv(OUT / f"A2_position_mixed_{name}.csv")
    a2[name] = tab.to_dict(orient="index")
    print(f"\n=== A2 mixed model ({name}) ===")
    print(tab.round(4).to_string())
    # descriptive: participant-level means by condition x position, paired diff 3rd-2nd
    pm = wb.groupby(["sub", "condition", "position"])[dv].mean().unstack("position")
    desc = []
    for c in ["Watching", "Baseline"]:
        sub = pm.xs(c, level="condition")
        r = paired(sub[3], sub[2])
        desc.append(dict(dv=name, condition=c, mean_pos2=r["mean_y"], mean_pos3=r["mean_x"],
                         diff_3_minus_2=r["mean_diff"], ci_low=r["ci_low"], ci_high=r["ci_high"],
                         t=r["t"], p=r["p"], dz=r["dz"], n=r["n"]))
    # pooled across the two conditions
    pooled = wb.groupby(["sub", "position"])[dv].mean().unstack("position")
    r = paired(pooled[3], pooled[2])
    desc.append(dict(dv=name, condition="Watching+Baseline pooled", mean_pos2=r["mean_y"],
                     mean_pos3=r["mean_x"], diff_3_minus_2=r["mean_diff"], ci_low=r["ci_low"],
                     ci_high=r["ci_high"], t=r["t"], p=r["p"], dz=r["dz"], n=r["n"]))
    desc = pd.DataFrame(desc)
    desc.to_csv(OUT / f"A2_position_descriptives_{name}.csv", index=False)
    a2[f"{name}_descriptives"] = desc.to_dict(orient="records")
    print(desc.round(3).to_string())

# repetition trend (learning / fatigue proxy), all three conditions
canon["rep_c"] = canon["rep"] - 2.5
canon["cond"] = pd.Categorical(canon["condition"], ["Scrolling", "Watching", "Baseline"])
canon["npic"] = pd.Categorical(canon["n_pictures"].astype(int), [9, 12, 18])
for dv, name in [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]:
    m = smf.mixedlm(f"{dv} ~ cond * rep_c + C(npic)", canon, groups=canon["sub"]).fit(reml=True)
    ci = m.conf_int()
    tab = pd.DataFrame({"estimate": m.params, "se": m.bse, "ci_low": ci[0], "ci_high": ci[1],
                        "z": m.tvalues, "p": m.pvalues})
    tab.to_csv(OUT / f"A2_repetition_mixed_{name}.csv")
    a2[f"repetition_{name}"] = tab.to_dict(orient="index")
    print(f"\n=== A2 repetition trend ({name}) ===")
    print(tab.round(4).to_string())
    # condition x repetition RM-ANOVA (as in the submission's 'repetition analysis')
    cm = canon.groupby(["sub", "condition", "rep"])[dv].mean().reset_index()
    aov = pg.rm_anova(data=cm, dv=dv, within=["condition", "rep"], subject="sub", correction=True)
    aov.columns = [c.replace("-", "_") for c in aov.columns]
    aov.to_csv(OUT / f"A2_condition_x_repetition_anova_{name}.csv", index=False)
    a2[f"cond_x_rep_anova_{name}"] = aov.to_dict(orient="records")
    print(aov[["Source", "F", "ddof1", "ddof2", "p_unc", "p_GG_corr"]].round(4).to_string())
results["A2"] = a2


# --------------------------------------------------------------------------
# A2c. Scrolling - Watching split by Watching position (pairs within repetition x block length)
# --------------------------------------------------------------------------
a2c = []
for dv, name in [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]:
    sc = canon[canon.condition == "Scrolling"][["sub", "rep", "n_pictures", dv]].rename(columns={dv: "scroll"})
    wa = canon[canon.condition == "Watching"][["sub", "rep", "n_pictures", "position", dv]].rename(columns={dv: "watch"})
    pr = sc.merge(wa, on=["sub", "rep", "n_pictures"])
    pr["diff"] = pr["scroll"] - pr["watch"]
    for pos in [2, 3]:
        sm = pr[pr.position == pos].groupby("sub")[["scroll", "watch"]].mean()
        r = paired(sm["scroll"], sm["watch"])
        a2c.append(dict(dv=name, watching_position=pos, n=r["n"], mean_scrolling=r["mean_x"],
                        mean_watching=r["mean_y"], diff=r["mean_diff"], ci_low=r["ci_low"],
                        ci_high=r["ci_high"], t=r["t"], p=r["p"], dz=r["dz"]))
    # mixed model: diff ~ pos3 with participant intercept (block-level pairs)
    pr["pos3"] = (pr.position == 3).astype(int)
    m = smf.mixedlm("diff ~ pos3", pr, groups=pr["sub"]).fit(reml=True)
    ci = m.conf_int()
    a2c.append(dict(dv=name, watching_position="mixed: diff ~ pos3", n=pr["sub"].nunique(),
                    diff=m.params["Intercept"], ci_low=ci.loc["Intercept", 0], ci_high=ci.loc["Intercept", 1],
                    p=m.pvalues["Intercept"], t=m.tvalues["Intercept"],
                    mean_scrolling=m.params["pos3"], mean_watching=m.pvalues["pos3"]))
a2c = pd.DataFrame(a2c)
a2c.to_csv(OUT / "A2c_scrolling_minus_watching_by_position.csv", index=False)
results["A2c"] = a2c.to_dict(orient="records")
print("\n=== A2c Scrolling - Watching by Watching position (mixed rows: mean_scrolling=pos3 coef, mean_watching=pos3 p) ===")
print(a2c.round(3).to_string())

# --------------------------------------------------------------------------
# A2d. lag-1 preceding-estimate model
# --------------------------------------------------------------------------
a2d = {}
seq = canon.sort_values(["sub", "session_order", "block_order_in_session"]).copy()
seq["prev_est"] = seq.groupby("sub")["estimated_duration_s"].shift(1)
seq["prev_ratio"] = seq.groupby("sub")["ratio"].shift(1)
seq = seq.dropna(subset=["prev_est"])
seq["prev_est_c"] = seq["prev_est"] - seq["prev_est"].mean()
seq["prev_ratio_c"] = seq["prev_ratio"] - seq["prev_ratio"].mean()
for dv, name, lag in [("estimated_duration_s", "estimate_s", "prev_est_c"), ("ratio", "ratio", "prev_ratio_c")]:
    for label, formula in [("no_lag", f"{dv} ~ C(cond, Treatment('Watching')) + C(npic) + rep_c"),
                           ("with_lag", f"{dv} ~ C(cond, Treatment('Watching')) + {lag} + C(npic) + rep_c")]:
        m = smf.mixedlm(formula, seq, groups=seq["sub"]).fit(reml=True)
        ci = m.conf_int()
        tab = pd.DataFrame({"estimate": m.params, "se": m.bse, "ci_low": ci[0], "ci_high": ci[1],
                            "z": m.tvalues, "p": m.pvalues})
        tab.to_csv(OUT / f"A2d_lag1_{name}_{label}.csv")
        a2d[f"{name}_{label}"] = tab.to_dict(orient="index")
        print(f"\n=== A2d lag-1 ({name}, {label}; n blocks = {len(seq)}) ===")
        print(tab.round(4).to_string())
results["A2d"] = a2d


# --------------------------------------------------------------------------
# A2e. assimilation prediction for Scrolling: preceding block Baseline vs Watching
# --------------------------------------------------------------------------
seq_all = canon.sort_values(["sub", "session_order", "block_order_in_session"]).copy()
seq_all["prev_cond"] = seq_all.groupby("sub")["condition"].shift(1)
seq_all["prev_est"] = seq_all.groupby("sub")["estimated_duration_s"].shift(1)
sc = seq_all[(seq_all.condition == "Scrolling") & (seq_all.rep > 1)].copy()
# the block immediately before a Scrolling *session* is the last block of the previous repetition;
# within a Scrolling session, blocks 2 and 3 are preceded by Scrolling blocks. Use session-level
# preceding condition: last block of the previous session.
sess_last = seq_all.groupby(["sub", "session_order"]).tail(1)[["sub", "session_order", "condition", "estimated_duration_s"]]
sess_last = sess_last.rename(columns={"condition": "prev_sess_cond", "estimated_duration_s": "prev_sess_last_est"})
sess_last["session_order"] = sess_last["session_order"] + 1
sc = sc.merge(sess_last, on=["sub", "session_order"], how="left")
a2e = []
for dv, name in [("estimated_duration_s", "estimate_s"), ("ratio", "ratio")]:
    pm = sc.groupby(["sub", "prev_sess_cond"])[dv].mean().unstack()
    r = paired(pm["Baseline"], pm["Watching"])
    a2e.append(dict(dv=name, mean_scrolling_after_baseline=r["mean_x"], mean_scrolling_after_watching=r["mean_y"],
                    diff=r["mean_diff"], ci_low=r["ci_low"], ci_high=r["ci_high"], t=r["t"], p=r["p"], dz=r["dz"], n=r["n"]))
a2e = pd.DataFrame(a2e)
a2e.to_csv(OUT / "A2e_scrolling_by_preceding_condition.csv", index=False)
results["A2e"] = a2e.to_dict(orient="records")
print("\n=== A2e Scrolling (reps 2-4) by preceding session condition ===")
print(a2e.round(3).to_string())

# --------------------------------------------------------------------------
# A2f. robustness of the A2c split: counterbalancing group, repetition, sign counts
# --------------------------------------------------------------------------
sc_ = canon[canon.condition == "Scrolling"][["sub", "rep", "n_pictures", "estimated_duration_s", "ratio"]].rename(
    columns={"estimated_duration_s": "scroll_est", "ratio": "scroll_ratio"})
wa_ = canon[canon.condition == "Watching"][["sub", "rep", "n_pictures", "position", "estimated_duration_s", "ratio"]].rename(
    columns={"estimated_duration_s": "watch_est", "ratio": "watch_ratio"})
pr = sc_.merge(wa_, on=["sub", "rep", "n_pictures"])
pr["d_est"] = pr.scroll_est - pr.watch_est
pr["d_ratio"] = pr.scroll_ratio - pr.watch_ratio
# counterbalancing group: Watching second in odd reps?
grp = pr[pr.rep == 1].groupby("sub")["position"].first().rename("watch_pos_rep1")
pr = pr.merge(grp, on="sub")
pr["cb_group"] = np.where(pr.watch_pos_rep1 == 2, "Watching 2nd in odd reps", "Watching 2nd in even reps")
rows = []
for g, gg in pr.groupby("cb_group"):
    for pos in [2, 3]:
        sm = gg[gg.position == pos].groupby("sub")[["d_est", "d_ratio"]].mean()
        rows.append(dict(split="cb_group", level=g, watching_position=pos, n=len(sm),
                         mean_d_est=sm.d_est.mean(), ci_est=stats.t.interval(0.95, len(sm) - 1, sm.d_est.mean(), stats.sem(sm.d_est)),
                         mean_d_ratio=sm.d_ratio.mean(), prop_scroll_shorter=float((sm.d_est < 0).mean())))
for rep, gg in pr.groupby("rep"):
    for pos in [2, 3]:
        sm = gg[gg.position == pos].groupby("sub")[["d_est", "d_ratio"]].mean()
        if len(sm) < 3:
            continue
        rows.append(dict(split="rep", level=int(rep), watching_position=pos, n=len(sm),
                         mean_d_est=sm.d_est.mean(), ci_est=stats.t.interval(0.95, len(sm) - 1, sm.d_est.mean(), stats.sem(sm.d_est)),
                         mean_d_ratio=sm.d_ratio.mean(), prop_scroll_shorter=float((sm.d_est < 0).mean())))
for pos in [2, 3]:
    sm = pr[pr.position == pos].groupby("sub")[["d_est", "d_ratio"]].mean()
    rows.append(dict(split="all", level="all", watching_position=pos, n=len(sm),
                     mean_d_est=sm.d_est.mean(), ci_est=stats.t.interval(0.95, len(sm) - 1, sm.d_est.mean(), stats.sem(sm.d_est)),
                     mean_d_ratio=sm.d_ratio.mean(), prop_scroll_shorter=float((sm.d_est < 0).mean())))
a2f = pd.DataFrame(rows)
a2f.to_csv(OUT / "A2f_split_robustness.csv", index=False)
results["A2f"] = a2f.to_dict(orient="records")
print("\n=== A2f robustness of the split ===")
print(a2f.round(3).to_string())

# --------------------------------------------------------------------------
# A2g. lag-1 models with participant random slopes for block length
# --------------------------------------------------------------------------
a2g = {}
for dv, name, lag in [("estimated_duration_s", "estimate_s", "prev_est_c"), ("ratio", "ratio", "prev_ratio_c")]:
    for label, formula in [("no_lag", f"{dv} ~ C(cond, Treatment('Watching')) + C(npic) + rep_c"),
                           ("with_lag", f"{dv} ~ C(cond, Treatment('Watching')) + {lag} + C(npic) + rep_c")]:
        m = smf.mixedlm(formula, seq, groups=seq["sub"], re_formula="~C(npic)").fit(reml=True, method=["lbfgs"])
        ci = m.conf_int()
        tab = pd.DataFrame({"estimate": m.params, "se": m.bse, "ci_low": ci[0], "ci_high": ci[1],
                            "z": m.tvalues, "p": m.pvalues})
        tab.to_csv(OUT / f"A2g_lag1_randslope_{name}_{label}.csv")
        a2g[f"{name}_{label}"] = tab.loc[[i for i in tab.index if "cond" in i or "prev" in i]].to_dict(orient="index")
        print(f"\n=== A2g lag-1 with random slopes ({name}, {label}; converged={m.converged}) ===")
        print(tab.loc[[i for i in tab.index if "cond" in i or "prev" in i or "Intercept" in i]].round(4).to_string())
results["A2g"] = a2g


# ==========================================================================
# A3. RM-ANOVA sensitivity bound (G*Power 3.0 'within factors' convention)
# ==========================================================================
def min_f_within(n, m, df1_full, eps, rho, alpha=0.05, power=0.80):
    """Smallest Cohen's f detectable with power for a within-subject effect.
    lambda = f^2 * n * m * eps / (1 - rho); df1 = df1_full*eps; df2 = (n-1)*df1_full*eps.
    m = number of repeated measurements entering the effect."""
    df1 = df1_full * eps
    df2 = (n - 1) * df1_full * eps
    fcrit = stats.f.ppf(1 - alpha, df1, df2)

    def pw(f):
        lam = f ** 2 * n * m * eps / (1 - rho)
        v = stats.ncf.sf(fcrit, df1, df2, lam)
        return 1.0 if np.isnan(v) else v

    return optimize.brentq(lambda f: pw(f) - power, 1e-4, 3)


cell9 = canon.groupby(["sub", "condition", "n_pictures"])["estimated_duration_s"].mean().unstack(["condition", "n_pictures"])
corr9 = cell9.corr().values
rho9 = float(corr9[np.triu_indices(9, 1)].mean())
cond3 = canon.groupby(["sub", "condition"])["estimated_duration_s"].mean().unstack()
rho3 = float(cond3.corr().values[np.triu_indices(3, 1)].mean())
aov_est = pd.read_csv(OUT / "A1_rm_anova_estimate_s.csv")
eps_cond = float(aov_est.loc[aov_est.Source == "condition", "eps"].iloc[0])
eps_int = float(aov_est.loc[aov_est.Source == "condition * n_pictures", "eps"].iloc[0])
a3 = dict(
    n=23, rho_condition_means=rho3, rho_nine_cells=rho9, eps_condition=eps_cond, eps_interaction=eps_int,
    min_f_condition_main_eps_obs=min_f_within(23, 3, 2, eps_cond, rho3),
    min_f_condition_main_eps1=min_f_within(23, 3, 2, 1.0, rho3),
    min_f_interaction_eps_obs=min_f_within(23, 9, 4, eps_int, rho9),
    min_f_interaction_eps1=min_f_within(23, 9, 4, 1.0, rho9),
    # original paired-t plan
    n_for_d064_paired=int(np.ceil(pg.power_ttest(d=0.64, power=0.8, alpha=0.05, contrast="paired"))),
    min_dz_paired_n23=float(pg.power_ttest(n=23, power=0.8, alpha=0.05, contrast="paired")),
)
# observed interaction f for reference (from np2: f = sqrt(np2/(1-np2)))
np2_int = float(aov_est.loc[aov_est.Source == "condition * n_pictures", "np2"].iloc[0]) if "np2" in aov_est else np.nan
a3["observed_interaction_np2"] = np2_int
a3["observed_interaction_f"] = float(np.sqrt(np2_int / (1 - np2_int))) if np.isfinite(np2_int) else np.nan
# A3b. sensitivity in partial-eta-squared units (comparable with the observed RM-ANOVA np2):
# find the noncentrality giving 80% power at the GG-corrected dfs, then np2_min = lambda / (lambda + df_error_uncorrected)
def min_np2(df1_full, df2_full, eps, alpha=0.05, power=0.80):
    df1, df2 = df1_full * eps, df2_full * eps
    fcrit = stats.f.ppf(1 - alpha, df1, df2)
    lam = optimize.brentq(lambda l: stats.ncf.sf(fcrit, df1, df2, l) - power, 1e-3, 500)
    return lam, lam / (lam + df2_full)
lam_c, np2_c = min_np2(2, 44, eps_cond)
lam_i, np2_i = min_np2(4, 88, eps_int)
a3.update(dict(min_np2_condition_main=np2_c, lambda_condition_main=lam_c,
               min_np2_interaction=np2_i, lambda_interaction=lam_i,
               observed_condition_np2=float(aov_est.loc[aov_est.Source == "condition", "np2"].iloc[0])))
results["A3"] = a3
print("\n=== A3 sensitivity ===")
print(json.dumps(a3, indent=2))


# ==========================================================================
# A7. yoking verification (durations equal as multisets; images disjoint)
# ==========================================================================
def parse_list(s):
    s = str(s).strip().strip('"')
    return literal_eval(s)


chk = []
for sub, g in canon.groupby("sub"):
    for (rep, n), gg in g.groupby(["rep", "n_pictures"]):
        act = gg[gg.session_type == "active"]
        pas = gg[gg.session_type == "passive"]
        con = gg[gg.session_type == "constant"]
        if len(act) != 1 or len(pas) != 1:
            chk.append(dict(sub=sub, rep=rep, n=n, status="missing"))
            continue
        a_pic = sorted(parse_list(act.per_picture_durations.iloc[0]))
        p_pic = sorted(parse_list(pas.per_picture_durations.iloc[0]))
        a_fix = sorted(parse_list(act.per_fixation_durations.iloc[0]))
        p_fix = sorted(parse_list(pas.per_fixation_durations.iloc[0]))
        a_img = set(parse_list(act.image_files.iloc[0]))
        p_img = set(parse_list(pas.image_files.iloc[0]))
        c_img = set([con.constant_image.iloc[0]]) if len(con) else set()
        chk.append(dict(
            sub=sub, rep=rep, n=n, status="ok",
            pic_dur_max_abs_diff=float(np.max(np.abs(np.array(a_pic) - np.array(p_pic)))),
            fix_dur_max_abs_diff=float(np.max(np.abs(np.array(a_fix) - np.array(p_fix)))),
            gap_sum_active=float(np.sum(a_fix)), gap_sum_passive=float(np.sum(p_fix)),
            block_dur_active=float(act.actual_duration_s.iloc[0]),
            block_dur_passive=float(pas.actual_duration_s.iloc[0]),
            block_dur_constant=float(con.actual_duration_s.iloc[0]) if len(con) else np.nan,
            n_shared_images_active_passive=len(a_img & p_img),
            constant_image_in_active_or_passive=len(c_img & (a_img | p_img)),
            same_order=bool(parse_list(act.per_picture_durations.iloc[0]) == parse_list(pas.per_picture_durations.iloc[0])),
        ))
chk = pd.DataFrame(chk)
chk.to_csv(OUT / "A7_yoking_check.csv", index=False)
ok = chk[chk.status == "ok"]
a7 = dict(
    n_cells=len(chk), n_ok=int((chk.status == "ok").sum()),
    max_pic_dur_diff=float(ok.pic_dur_max_abs_diff.max()),
    max_fix_dur_diff=float(ok.fix_dur_max_abs_diff.max()),
    max_gap_sum_diff=float((ok.gap_sum_active - ok.gap_sum_passive).abs().max()),
    max_block_dur_diff_active_passive=float((ok.block_dur_active - ok.block_dur_passive).abs().max()),
    max_block_dur_diff_active_constant=float((ok.block_dur_active - ok.block_dur_constant).abs().max()),
    mean_block_dur_diff_active_passive=float((ok.block_dur_passive - ok.block_dur_active).mean()),
    mean_block_dur_diff_active_constant=float((ok.block_dur_constant - ok.block_dur_active).mean()),
    cells_with_shared_images=int((ok.n_shared_images_active_passive > 0).sum()),
    cells_constant_image_reused=int((ok.constant_image_in_active_or_passive > 0).sum()),
    cells_same_order=int(ok.same_order.sum()),
    mean_gap_s_sequential=float(ok.gap_sum_active.mean()),
)
results["A7"] = a7
print("\n=== A7 yoking ===")
print(json.dumps(a7, indent=2))

with open(OUT / "revision_behavioral_audit_results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print("\nwrote", OUT)
