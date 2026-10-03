"""Supplementary figures that replace the long supplement tables (revision round 1).

Every figure is built from the analysis outputs, not from the manuscript tables:
  S2.1  retained EEG epochs per participant            (A4_retained_*_by_subject.csv)
  S3.1  paired contrasts before/after fixation-interval accounting
                                                       (paired_tests_gap_adjusted_20260519.csv)
  S4.2  Scrolling - Watching and Baseline contrasts across repetitions, contexts, and models
                                                       (repetition_order_sensitivity_20260924/, A2c, A2g, A2h)
  S4.3  paired contrasts in the analysed and exclusion-sensitivity samples (A1_paired_contrasts.csv)
  S5.1  picture-locked window differences, baseline-corrected vs uncorrected (erp_three_components_20260608/)
  S6.1  exploratory FRP grid: three clusters x four windows x three contrasts (group_frp/frp_condition_comparisons.csv)
  S6.2  single-fixation mixed models and the saccade-amplitude covariate
                                                       (frp_image_random_effect_20260521/, frp_lambda_saccade_covariate_20260602/)
Figure S4.1 is figS4_position_context (figure_S4_position.py).
Run: uv run python manuscript/revision-tmb-20260908/analyses/figures_supplement.py
"""
from pathlib import Path
import re

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
OUTDIR = OUT_DIR / "revision"
RV = DATA_DIR / "revision"  # EEG/ET-derived intermediate tables
OUT = OUT_DIR / "revision/figures"
OUT.mkdir(parents=True, exist_ok=True)
COL = {"Scrolling": "#222222", "Watching": "#d95f02", "Baseline": "#1f77b4"}
INK, MUTED, GRID = "#222222", "#6b6b6b", "#bdbdbd"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": INK, "ytick.color": INK})


def save(fig, name):
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / f"{name}.png")


def zero(ax, axis="x"):
    (ax.axvline if axis == "x" else ax.axhline)(0, color=GRID, lw=0.8, ls="--", zorder=0)


def forest(ax, rows, series, xlabel, group_gap=0.6, legend_below=False):
    """rows: list of (group, label). series: list of dicts(name, marker, face, data{(group,label): (est, lo, hi)})."""
    y, ylab, ypos, prev = 0.0, [], {}, None
    for g, lab in rows:
        if prev is not None and g != prev:
            y += group_gap
        ypos[(g, lab)] = y
        ylab.append((y, lab))
        prev = g
        y += 1
    k = len(series)
    offs = np.linspace(-0.18, 0.18, k) if k > 1 else [0.0]
    for s, off in zip(series, offs):
        first = True
        for key, (est, lo, hi) in s["data"].items():
            if key not in ypos:
                continue
            yy = ypos[key] + off
            ax.plot([lo, hi], [yy, yy], color=s.get("line", INK), lw=1.2, zorder=2)
            ax.plot(est, yy, s["marker"], ms=6, mfc=s["face"], mec=INK, mew=0.9, zorder=3,
                    label=s["name"] if first else None)
            first = False
    ax.set_yticks([p for p, _ in ylab])
    ax.set_yticklabels([l for _, l in ylab])
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel(xlabel)
    # group headers
    seen = []
    for (g, _), p in ypos.items():
        if g and g not in seen:
            seen.append(g)
            ax.text(-0.02, p - 0.62, g, transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=8.5, fontweight="bold", color=INK)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    if len(series) > 1:
        if legend_below:
            ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, ncol=len(series),
                      loc="upper center", bbox_to_anchor=(0.5, -0.32), handletextpad=0.3, columnspacing=1.2)
        else:
            ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8, loc="best")
    return ypos


def t_ci(diff, t, df=22):
    se = diff / t
    q = stats.t.ppf(0.975, df)
    return diff - q * se, diff + q * se


# ---------------------------------------------------------------- S2.1 retained epochs
pic = pd.read_csv(RV / "retained_picture_epochs_by_subject.csv", dtype={"subject": str})
fix = pd.read_csv(RV / "retained_fixation_epochs_by_subject.csv", dtype={"subject": str})
cond = [("active", "Scrolling"), ("passive", "Watching"), ("constant", "Baseline")]
pic["rej"] = 100 * (1 - pic[[f"final_{c}" for c, _ in cond]].sum(axis=1) / pic[[f"raw_{c}" for c, _ in cond]].sum(axis=1))
flag = pic.loc[pic.rej > 30, "subject"].tolist()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
rng = np.random.default_rng(3)
for ax, df_, stem, ylabel, title in [
        (axes[0], pic, "final", "Picture epochs retained (% of presented)", "A  Picture-locked"),
        (axes[1], fix, "final", "Fixation epochs retained (count)", "B  Fixation-locked")]:
    for i, (c, name) in enumerate(cond):
        v = 100 * df_[f"final_{c}"] / df_[f"raw_{c}"] if df_ is pic else df_[f"final_{c}"]
        jit = rng.uniform(-0.13, 0.13, len(v))
        isf = df_["subject"].isin(flag).values
        ax.scatter(i + jit[~isf], v[~isf], s=14, color=COL[name], alpha=0.55, lw=0, zorder=3)
        ax.scatter(i + jit[isf], v[isf], s=30, facecolor="white", edgecolor=COL[name], lw=1.2, zorder=4)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=INK, lw=2, zorder=5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for _, n in cond])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
axes[0].set_ylim(0, 105)
axes[0].text(0.98, 0.04, "open markers: participants\nwith > 30% picture epochs rejected", transform=axes[0].transAxes,
             ha="right", va="bottom", fontsize=7.5, color=MUTED)
save(fig, "figS2_1_retained_epochs")

# ---------------------------------------------------------------- S3.1 fixation-interval accounting
gp = pd.read_csv(OUTDIR / "fixation_interval_accounting_20260519/paired_tests_gap_adjusted_20260519.csv")
cmap = {"active_vs_passive": "Scrolling − Watching", "active_vs_constant": "Scrolling − Baseline",
        "passive_vs_constant": "Watching − Baseline"}
get = lambda m: {("", cmap[r.contrast]): (r.mean_diff_a_minus_b, r.ci95_low, r.ci95_high)
                 for r in gp[gp.metric == m].itertuples()}
rows = [("", v) for v in cmap.values()]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
forest(axes[0], rows, [
    {"name": "Raw estimate", "marker": "o", "face": "white", "data": get("estimated_duration_s")},
    {"name": "Estimate + fixation intervals", "marker": "o", "face": INK, "data": get("estimate_plus_gap_s")}],
    "Paired difference (s)", legend_below=True)
axes[0].set_title("A  Duration (s)", loc="left")
forest(axes[1], rows, [
    {"name": "Raw ratio", "marker": "s", "face": "white", "data": get("raw_ratio_total_actual")},
    {"name": "Fixation-removed ratio", "marker": "s", "face": INK, "data": get("gap_removed_ratio")}],
    "Paired difference (ratio)", legend_below=True)
axes[1].set_yticklabels([])
axes[1].set_title("B  Estimation ratio", loc="left")
save(fig, "figS3_1_fixation_accounting")

# ---------------------------------------------------------------- S4.2 contexts, repetitions, models
ro = OUTDIR / "repetition_order_sensitivity_20260924"
rep = pd.read_csv(ro / "paired_contrasts_by_repetition.csv")
mp = pd.read_csv(ro / "matched_pair_sensitivity.csv")
sm = pd.read_csv(ro / "sensitivity_models.csv")
pos = pd.read_csv(OUTDIR / "A2c_scrolling_minus_watching_by_position.csv")
bpos = pd.read_csv(OUTDIR / "A2h_baseline_contrasts_by_position.csv")
lag = {k: pd.read_csv(OUTDIR / f"A2g_lag1_randslope_estimate_s_{k}.csv", index_col=0) for k in ["no_lag", "with_lag"]}

sw = {}
G1, G2, G3, G4 = "All blocks", "By repetition", "By context", "Sensitivity"
r = mp[(mp.dv == "estimate_s") & (mp["sample"] == "all pairs")].iloc[0]
sw[(G1, "Matched pairs")] = (r.mean_diff, r.ci_low, r.ci_high)
for x in rep[(rep.dv == "estimate_s") & (rep.contrast == "Scrolling - Watching")].itertuples():
    sw[(G2, f"Repetition {x.repetition}")] = (x.mean_diff, x.ci_low, x.ci_high)
for x in pos[(pos.dv == "estimate_s") & pos.watching_position.isin(["2", "3", 2, 3])].itertuples():
    lab = "Watching right after Scrolling" if str(x.watching_position) == "2" else "Watching after Baseline"
    sw[(G3, lab)] = (x.diff, x.ci_low, x.ci_high)
r = mp[(mp.dv == "estimate_s") & (mp["sample"] == "runs 1 and 12 removed")].iloc[0]
sw[(G4, "First & last sequence removed")] = (r.mean_diff, r.ci_low, r.ci_high)
mlab = {("runs 2-11", "none"): "Model, seq. 2–11, no time term",
        ("runs 2-11", "run order (linear)"): "Model, seq. 2–11, linear order",
        ("runs 2-11", "repetition (factor)"): "Model, seq. 2–11, repetition"}
for x in sm[(sm.dv == "estimate_s") & (sm.contrast == "Scrolling - Watching")].itertuples():
    if (x.data, x.time_term) in mlab:
        sw[(G4, mlab[(x.data, x.time_term)])] = (x.b, x.ci_low, x.ci_high)
term = [i for i in lag["with_lag"].index if "T.Scrolling" in i][0]
for k, lab in [("no_lag", "Lag-1 model, without lag term"), ("with_lag", "Lag-1 model, with lag term")]:
    e = lag[k].loc[term]
    sw[(G4, lab)] = (e.estimate, e.ci_low, e.ci_high)

bw, bs = {}, {}
for x in rep[(rep.dv == "estimate_s")].itertuples():
    tgt = bw if x.contrast == "Baseline - Watching" else bs if x.contrast == "Baseline - Scrolling" else None
    if tgt is not None:
        tgt[(G2, f"Repetition {x.repetition}")] = (x.mean_diff, x.ci_low, x.ci_high)
for x in bpos[bpos.dv == "estimated_duration_s"].itertuples():
    p = str(x.position)
    if p in ("2", "3"):
        bw[(G3, "Watching 2nd (after Scrolling)" if p == "2" else "Watching 3rd (after Baseline)")] = (x.diff, x.ci_low, x.ci_high)
    elif "at 2" in p:
        bs[(G3, "Baseline 2nd")] = (x.diff, x.ci_low, x.ci_high)
    elif "at 3" in p:
        bs[(G3, "Baseline 3rd")] = (x.diff, x.ci_low, x.ci_high)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
forest(axes[0], list(sw.keys()), [{"name": "Scrolling − Watching", "marker": "D", "face": INK, "data": sw}],
       "Scrolling − Watching (s)")
axes[0].set_title("A  Self-paced contrast", loc="left", pad=14)
rowsB = [(G2, f"Repetition {i}") for i in range(1, 5)] + [
    (G3, "Watching 2nd (after Scrolling)"), (G3, "Watching 3rd (after Baseline)"),
    (G3, "Baseline 2nd"), (G3, "Baseline 3rd")]
forest(axes[1], rowsB, [
    {"name": "Baseline − Watching", "marker": "o", "face": COL["Watching"], "data": bw},
    {"name": "Baseline − Scrolling", "marker": "s", "face": "white", "data": bs}],
    "Baseline − sequential condition (s)")
axes[1].set_title("B  Baseline contrasts", loc="left", pad=14)
axes[1].get_legend().remove()
axes[1].legend(*axes[1].get_legend_handles_labels(), frameon=False, fontsize=8, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, -0.12))
save(fig, "figS4_2_contexts_models")

# ---------------------------------------------------------------- S4.3 exclusion sensitivity
a1 = pd.read_csv(OUTDIR / "A1_paired_contrasts.csv")
samples = [("canonical_N23", "N = 23 (analysed)", "o", INK),
           ("keep027_N24", "N = 24", "s", "#8c8c8c"),
           ("keep015_016_027_N26", "N = 26", "^", "white")]
crow = ["Scrolling − Watching", "Scrolling − Baseline", "Watching − Baseline"]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
for ax, dv, xl, ttl in [(axes[0], "estimate_s", "Paired difference (s)", "A  Estimate"),
                        (axes[1], "ratio", "Paired difference (ratio)", "B  Estimation ratio")]:
    ser = []
    for key, name, mk, face in samples:
        sub = a1[(a1["sample"] == key) & (a1.dv == dv)]
        ser.append({"name": name, "marker": mk, "face": face,
                    "data": {("", c.replace(" - ", " − ")): (x.mean_diff, x.ci_low, x.ci_high)
                             for c, x in zip(sub.contrast, sub.itertuples())}})
    forest(ax, [("", c) for c in crow], ser, xl)
    ax.set_title(ttl, loc="left")
axes[1].set_yticklabels([])
axes[1].get_legend().remove()
save(fig, "figS4_3_exclusion_sensitivity")

# ---------------------------------------------------------------- S5.1 ERP windows
bc = pd.read_csv(RV / "erp_windows_baseline_corrected.csv")
uc = pd.read_csv(RV / "erp_windows_uncorrected.csv")
lab = {"P1_like_early_positive": "P1-like (80–130 ms)", "P2_like_posterior_positive": "P2-like (140–300 ms)"}
wl = lambda w, l: lab.get(w, "LPP (300–500 ms)" if "LPP" in w.upper() or "late" in w.lower() else l)
dat = lambda d: {("", wl(x.window, x.window_label)): (x.mean_diff_uV, x.ci95_low_uV, x.ci95_high_uV) for x in d.itertuples()}
fig, ax = plt.subplots(figsize=(4.6, 2.2))
forest(ax, [("", wl(x.window, x.window_label)) for x in bc.itertuples()], [
    {"name": "Baseline-corrected", "marker": "o", "face": INK, "data": dat(bc)},
    {"name": "Uncorrected", "marker": "o", "face": "white", "data": dat(uc)}],
    "Scrolling − Watching (µV)", legend_below=True)
save(fig, "figS5_1_erp_windows")

# ---------------------------------------------------------------- S6.1 FRP grid
fr = pd.read_csv(RV / "frp_condition_comparisons.csv")
fr = fr[fr.cluster.isin(["occipital", "posterior", "frontal"])].copy()
fr["lo"], fr["hi"] = zip(*[t_ci(d, t) for d, t in zip(fr.mean_diff_uv, fr.t_stat)])
wins = [("lambda", "Lambda\n60–120"), ("N1", "N1\n120–200"), ("P2", "P2\n200–350"), ("late", "Late\n350–600")]
comps = [("active_vs_passive", "Scrolling − Watching", "D", INK),
         ("active_vs_constant", "Scrolling − Baseline", "s", "white"),
         ("passive_vs_constant", "Watching − Baseline", "o", COL["Watching"])]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
for ax, cl in zip(axes, ["occipital", "posterior", "frontal"]):
    for j, (ckey, cname, mk, face) in enumerate(comps):
        for i, (wkey, _) in enumerate(wins):
            x = fr[(fr.cluster == cl) & (fr.comparison == ckey) & fr.window.str.lower().str.startswith(wkey.lower())]
            if x.empty:
                continue
            x = x.iloc[0]
            xx = i + (j - 1) * 0.22
            ax.plot([xx, xx], [x.lo, x.hi], color=INK, lw=1.1, zorder=2)
            ax.plot(xx, x.mean_diff_uv, mk, ms=5.5, mfc=face, mec=INK, mew=0.9, zorder=3,
                    label=cname if (i == 0 and cl == "occipital") else None)
    zero(ax, "y")
    ax.set_xticks(range(4))
    ax.set_xticklabels([w for _, w in wins], fontsize=7.5)
    ax.set_title(cl.capitalize(), loc="left")
axes[0].set_ylabel("Paired difference (µV)")
fig.legend(loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"figS6_1_frp_grid.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved", OUT / "figS6_1_frp_grid.png")

# ---------------------------------------------------------------- S6.2 single-fixation models
mm = pd.read_csv(RV / "frp_mixedlm_condition_image_re.csv")
sign = {"active_minus_passive": ("Scrolling − Watching", 1), "active_minus_constant": ("Scrolling − Baseline", 1),
        "constant_minus_passive": ("Watching − Baseline", -1)}
wl6 = {"lambda": "Lambda", "N1_frp": "N1", "P2_frp": "P2", "Late_frp": "Late"}
txt = (RV / "lambda_saccade_covariate_lmer_output.txt").read_text()
cov = {}
for tag, lab in [("m0", "Without saccade amplitude"), ("m1", "With saccade amplitude")]:
    block = txt.split(f"==== {tag}:")[1]
    est, se = map(float, re.search(r"conditionScrolling\s+([-\d.e]+)\s+([-\d.e]+)", block).groups())
    cov[lab] = (est, est - 1.96 * se, est + 1.96 * se)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
for j, (ck, (cname, sg)) in enumerate(sign.items()):
    mk, face = {0: ("D", INK), 1: ("s", "white"), 2: ("o", COL["Watching"])}[j]
    for i, (wk, wlab) in enumerate(wl6.items()):
        x = mm[(mm.window == wk) & (mm.contrast == ck)]
        if x.empty:
            continue
        x = x.iloc[0]
        est, lo, hi = sg * x.estimate_uv, sorted([sg * x.ci_low, sg * x.ci_high])[0], sorted([sg * x.ci_low, sg * x.ci_high])[1]
        xx = i + (j - 1) * 0.22
        ax.plot([xx, xx], [lo, hi], color=INK, lw=1.1, zorder=2)
        ax.plot(xx, est, mk, ms=5.5, mfc=face, mec=INK, mew=0.9, zorder=3, label=cname if i == 0 else None)
zero(ax, "y")
ax.set_xticks(range(4))
ax.set_xticklabels(list(wl6.values()))
ax.set_ylabel("Model coefficient (µV)")
ax.set_title("A  Occipital, crossed participant and image effects", loc="left", fontsize=8.5)
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14), handletextpad=0.3)
ax = axes[1]
for i, (lab, (est, lo, hi)) in enumerate(cov.items()):
    ax.plot([i, i], [lo, hi], color=INK, lw=1.2)
    ax.plot(i, est, "D", ms=6, mfc=INK if i == 0 else "white", mec=INK)
zero(ax, "y")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Without", "With"])
ax.set_xlabel("Saccade amplitude in model")
ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("Scrolling − Watching lambda (µV)")
ax.set_title("B  Saccade-amplitude covariate", loc="left", fontsize=8.5)
save(fig, "figS6_2_frp_models")
