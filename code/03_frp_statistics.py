"""
FRP Statistics
==============
Reads pre-computed per-subject FRP amplitude summaries from
data/frp_summaries/ and data/group/frp_summary_all.csv.

Reproduces:
  - Paired t-tests (scrolling vs watching vs baseline) per window × cluster
  - Bar summary figure
  - Lambda (fixation-related P1) comparison across conditions

No MNE or raw EEG data required.

Output: output/frp_statistics/

Usage:
    python code/03_frp_statistics.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    ANALYSIS_SUBJECTS, FRP_DIR, GROUP_DIR,
    COLORS, COND_LABELS, FRP_WINDOWS, make_out,
)

OUT = make_out('frp_statistics')
CONDS = ['active', 'passive', 'constant']


def load_all_frp() -> pd.DataFrame:
    frames = []
    for sub in ANALYSIS_SUBJECTS:
        p = FRP_DIR / f'sub-{sub}_frp_summary.csv'
        if not p.exists():
            print(f'  sub-{sub}: missing FRP summary, skipping')
            continue
        df = pd.read_csv(p)
        # normalise column names — individual files use slightly different schema
        df = df.rename(columns={'mean_amp_uv': 'amplitude_uV', 'condition': 'condition'})
        df['sub'] = sub
        frames.append(df)
    if not frames:
        sys.exit('ERROR: no FRP summary files found in data/frp_summaries/')
    return pd.concat(frames, ignore_index=True)


def pivot_frp(df: pd.DataFrame, cluster: str, window: str) -> pd.DataFrame:
    sub_df = df[(df['cluster'] == cluster) & (df['window'] == window)]
    amp_col = 'amplitude_uV' if 'amplitude_uV' in sub_df.columns else 'mean_amp_uv'
    cond_col = 'condition' if 'condition' in sub_df.columns else 'condition'
    return sub_df.pivot_table(index='sub', columns=cond_col, values=amp_col)


def run_statistics(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    clusters = df['cluster'].unique()
    windows  = df['window'].unique()
    available_conds = df['condition'].unique() if 'condition' in df.columns else df['condition'].unique()
    cond_pairs = [(c1, c2) for c1, c2 in combinations(CONDS, 2)
                  if c1 in available_conds and c2 in available_conds]

    for cl in clusters:
        for win in windows:
            piv = pivot_frp(df, cl, win)
            for c1, c2 in cond_pairs:
                if c1 not in piv.columns or c2 not in piv.columns:
                    continue
                valid = piv[[c1, c2]].dropna()
                if len(valid) < 5:
                    continue
                t, p = stats.ttest_rel(valid[c1], valid[c2])
                diff = valid[c1] - valid[c2]
                d = diff.mean() / diff.std()
                results.append(dict(
                    cluster=cl, window=win,
                    condition1=c1, condition2=c2,
                    mean_c1=valid[c1].mean(), sem_c1=valid[c1].sem(),
                    mean_c2=valid[c2].mean(), sem_c2=valid[c2].sem(),
                    mean_diff=diff.mean(), sem_diff=diff.sem(),
                    t=t, df=len(valid)-1, p=p, d=d, n=len(valid),
                ))

    res_df = pd.DataFrame(results)
    if len(res_df) > 1:
        _, p_fdr, _, _ = multipletests(res_df['p'], method='fdr_bh')
        res_df['p_fdr'] = p_fdr
    return res_df


def print_results(res_df: pd.DataFrame) -> None:
    print('\n' + '=' * 70)
    print('FRP STATISTICS (Paired t-tests, all condition pairs)')
    print('=' * 70)
    for cl in res_df['cluster'].unique():
        print(f'\n  Cluster: {cl}')
        for win in [w for w in list(FRP_WINDOWS.keys()) if w in res_df['window'].unique()]:
            sub = res_df[(res_df['cluster'] == cl) & (res_df['window'] == win)]
            for _, row in sub.iterrows():
                sig = ('***' if row['p'] < .001 else '**' if row['p'] < .01
                       else '*' if row['p'] < .05 else '  ')
                print(f'    {win:>10}  {row["condition1"]} vs {row["condition2"]}:'
                      f'  t({row["df"]}) = {row["t"]:+.2f}'
                      f'  p = {row["p"]:.4f}{sig}  d = {row["d"]:.2f}'
                      f'  Δ = {row["mean_diff"]:+.3f} µV')


def bar_summary_figure(df: pd.DataFrame) -> None:
    amp_col  = 'amplitude_uV' if 'amplitude_uV' in df.columns else 'mean_amp_uv'
    cond_col = 'condition'
    cluster  = 'occipital' if 'occipital' in df['cluster'].unique() else df['cluster'].unique()[0]
    windows  = [w for w in list(FRP_WINDOWS.keys()) if w in df['window'].unique()]
    conds    = [c for c in CONDS if c in df[cond_col].unique()]

    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4.5), sharey=False)
    if len(windows) == 1:
        axes = [axes]

    for ax, win in zip(axes, windows):
        piv = pivot_frp(df, cluster, win)
        x = range(len(conds))
        means = [piv[c].mean() if c in piv.columns else np.nan for c in conds]
        sems  = [piv[c].sem()  if c in piv.columns else np.nan for c in conds]
        ax.bar(x, means, yerr=sems,
               color=[COLORS.get(c, 'gray') for c in conds],
               alpha=0.75, capsize=5, width=0.55)
        for i, c in enumerate(conds):
            if c not in piv.columns:
                continue
            vals = piv[c].dropna()
            jitter = np.random.default_rng(42 + i).normal(0, 0.08, len(vals))
            ax.scatter(i + jitter, vals, color='black', alpha=0.35, s=14, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABELS.get(c, c) for c in conds], rotation=20, ha='right')
        ax.set_title(win)
        if ax is axes[0]:
            ax.set_ylabel('Amplitude (µV)')

    fig.suptitle(f'FRP Amplitude — {cluster.capitalize()} Cluster (N={df["sub"].nunique()})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'frp_bar_summary.png', dpi=150)
    plt.close(fig)
    print('  Saved: frp_bar_summary.png')


def main() -> None:
    print('=' * 60)
    print('FRP STATISTICS')
    print('=' * 60)

    group_csv = GROUP_DIR / 'frp_summary_all.csv'
    if group_csv.exists():
        df = pd.read_csv(group_csv)
        # group CSV uses 'subject' not 'sub'
        if 'subject' in df.columns and 'sub' not in df.columns:
            df = df.rename(columns={'subject': 'sub'})
        print(f'  Loaded group CSV: {df["sub"].nunique()} subjects')
    else:
        df = load_all_frp()

    df['sub'] = df['sub'].astype(str).str.zfill(3)
    df = df[df['sub'].isin(ANALYSIS_SUBJECTS)].copy()

    # Normalise amplitude column name
    if 'mean_amp_uv' in df.columns and 'amplitude_uV' not in df.columns:
        df = df.rename(columns={'mean_amp_uv': 'amplitude_uV'})

    print(f'  Windows:  {sorted(df["window"].unique())}')
    print(f'  Clusters: {sorted(df["cluster"].unique())}')
    print(f'  N subjects: {df["sub"].nunique()}')

    res_df = run_statistics(df)
    print_results(res_df)
    bar_summary_figure(df)

    res_df.to_csv(OUT / 'frp_paired_ttests.csv', index=False)
    df.to_csv(OUT / 'frp_subject_amplitudes.csv', index=False)
    print(f'\n  Saved: frp_paired_ttests.csv, frp_subject_amplitudes.csv')
    print(f'All results saved to: output/frp_statistics/')


if __name__ == '__main__':
    main()
