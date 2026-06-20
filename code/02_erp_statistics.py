"""
ERP Statistics
==============
Reads pre-computed per-subject ERP amplitude summaries from
data/erp_summaries/ and data/group/erp_summary_all.csv.

Reproduces:
  - Paired t-tests (scrolling vs watching) per time window × cluster
  - Bar summary figure
  - Outputs a merged subject-level table

No MNE or raw EEG data required.

Output: output/erp_statistics/

Usage:
    python code/02_erp_statistics.py
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
    ANALYSIS_SUBJECTS, ERP_DIR, GROUP_DIR,
    COLORS, COND_LABELS, make_out,
)

OUT = make_out('erp_statistics')
CONDS = ['active', 'passive']


def load_all_erp() -> pd.DataFrame:
    """Stack per-subject ERP summary CSVs."""
    frames = []
    for sub in ANALYSIS_SUBJECTS:
        p = ERP_DIR / f'sub-{sub}_erp_summary.csv'
        if not p.exists():
            print(f'  sub-{sub}: missing ERP summary, skipping')
            continue
        df = pd.read_csv(p)
        df['sub'] = sub
        frames.append(df)
    if not frames:
        sys.exit('ERROR: no ERP summary files found in data/erp_summaries/')
    combined = pd.concat(frames, ignore_index=True)
    print(f'  Loaded {len(combined["sub"].unique())} subjects')
    return combined


def pivot_erp(df: pd.DataFrame, cluster: str, window: str) -> pd.DataFrame:
    """Pivot to subject × condition for a given cluster × window."""
    sub_df = df[(df['cluster'] == cluster) & (df['window'] == window)]
    return sub_df.pivot(index='sub', columns='condition', values='amplitude_uV')


def run_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Paired t-tests for each window × cluster × comparison."""
    results = []
    clusters = df['cluster'].unique()
    windows  = df['window'].unique()
    cond_pairs = list(combinations(CONDS, 2)) + [('active', 'constant'), ('passive', 'constant')]
    cond_pairs = [p for p in cond_pairs if all(c in df['condition'].unique() for c in p)]

    for cl in clusters:
        for win in windows:
            piv = pivot_erp(df, cl, win)
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
    print('ERP STATISTICS (Paired t-tests, scrolling vs watching)')
    print('=' * 70)
    aw = res_df[(res_df['condition1'] == 'active') & (res_df['condition2'] == 'passive')]
    for cl in aw['cluster'].unique():
        print(f'\n  Cluster: {cl}')
        for _, row in aw[aw['cluster'] == cl].iterrows():
            sig = ('***' if row['p'] < .001 else '**' if row['p'] < .01
                   else '*' if row['p'] < .05 else '')
            sig_fdr = '*' if row.get('p_fdr', 1) < .05 else ''
            print(f'    {row["window"]:>8}:  t({row["df"]}) = {row["t"]:+.2f}'
                  f'  p = {row["p"]:.4f}{sig}  d = {row["d"]:.2f}'
                  f'  Δ = {row["mean_diff"]:+.3f} µV'
                  + (f'  [p_fdr={row["p_fdr"]:.4f}{sig_fdr}]' if 'p_fdr' in row else ''))


def bar_summary_figure(df: pd.DataFrame) -> None:
    """Bar chart: mean ERP amplitude per condition × window (main cluster: occipital)."""
    cluster = 'occipital' if 'occipital' in df['cluster'].unique() else df['cluster'].unique()[0]
    windows = [w for w in ['P1', 'N1', 'P2', 'P3', 'Late'] if w in df['window'].unique()]
    conds   = [c for c in ['active', 'passive', 'constant'] if c in df['condition'].unique()]

    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4.5), sharey=False)
    if len(windows) == 1:
        axes = [axes]

    for ax, win in zip(axes, windows):
        piv = pivot_erp(df, cluster, win)
        x = range(len(conds))
        means = [piv[c].mean() if c in piv.columns else np.nan for c in conds]
        sems  = [piv[c].sem()  if c in piv.columns else np.nan for c in conds]
        bars  = ax.bar(x, means, yerr=sems,
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
        ax.set_title(win); ax.set_ylabel('Amplitude (µV)') if ax is axes[0] else None

    fig.suptitle(f'ERP Amplitude — {cluster.capitalize()} Cluster (N={len(df["sub"].unique())})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'erp_bar_summary.png', dpi=150)
    plt.close(fig)
    print('  Saved: erp_bar_summary.png')


def main() -> None:
    print('=' * 60)
    print('ERP STATISTICS')
    print('=' * 60)

    # Prefer the stacked group CSV; fall back to loading individual files
    group_csv = GROUP_DIR / 'erp_summary_all.csv'
    if group_csv.exists():
        df = pd.read_csv(group_csv)
        print(f'  Loaded group CSV: {len(df["sub"].unique())} subjects')
    else:
        df = load_all_erp()

    # Filter to canonical subjects
    df = df[df['sub'].astype(str).str.zfill(3).isin(ANALYSIS_SUBJECTS)].copy()
    df['sub'] = df['sub'].astype(str).str.zfill(3)

    print(f'  Windows:  {sorted(df["window"].unique())}')
    print(f'  Clusters: {sorted(df["cluster"].unique())}')
    print(f'  N subjects: {df["sub"].nunique()}')

    res_df = run_statistics(df)
    print_results(res_df)
    bar_summary_figure(df)

    res_df.to_csv(OUT / 'erp_paired_ttests.csv', index=False)
    df.to_csv(OUT / 'erp_subject_amplitudes.csv', index=False)
    print(f'\n  Saved: erp_paired_ttests.csv, erp_subject_amplitudes.csv')
    print(f'All results saved to: output/erp_statistics/')


if __name__ == '__main__':
    main()
