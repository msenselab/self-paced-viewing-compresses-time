"""
Eye-Tracking Analysis
=====================
Reads per-subject ET summary CSVs from data/et_summaries/ and reproduces:
  - Fixation count and duration by condition (paired t-tests)
  - Saccade amplitude and peak velocity by condition
  - Pupil diameter by condition
  - Group-level bar and raincloud figures

Output: output/et_analysis/

Usage:
    python code/04_et_analysis.py
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
from config import ANALYSIS_SUBJECTS, ET_DIR, COLORS, COND_LABELS, make_out

OUT = make_out('et_analysis')
CONDS = ['active', 'passive', 'constant']


def load_et_summaries() -> pd.DataFrame:
    frames = []
    for sub in ANALYSIS_SUBJECTS:
        p = ET_DIR / f'sub-{sub}_et_summary.csv'
        if not p.exists():
            print(f'  sub-{sub}: missing ET summary, skipping')
            continue
        df = pd.read_csv(p)
        df['sub'] = sub
        frames.append(df)
    if not frames:
        sys.exit('ERROR: no ET summary files found in data/et_summaries/')
    combined = pd.concat(frames, ignore_index=True)
    cond_col = 'block_type' if 'block_type' in combined.columns else 'condition'
    combined = combined.rename(columns={cond_col: 'condition'})
    print(f'  Loaded ET summaries: {combined["sub"].nunique()} subjects')
    return combined


def pivot_measure(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot_table(index='sub', columns='condition', values=col)


def paired_tests(df: pd.DataFrame, col: str, label: str) -> list[dict]:
    piv = pivot_measure(df, col)
    results = []
    pairs = [(c1, c2) for c1, c2 in combinations(CONDS, 2)
             if c1 in piv.columns and c2 in piv.columns]
    print(f'\n  {label}:')
    for c1, c2 in pairs:
        valid = piv[[c1, c2]].dropna()
        if len(valid) < 5:
            continue
        t, p = stats.ttest_rel(valid[c1], valid[c2])
        diff = valid[c1] - valid[c2]
        d = diff.mean() / diff.std()
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        print(f'    {c1} vs {c2}: t({len(valid)-1}) = {t:+.2f}  p = {p:.4f}{sig}  d = {d:.2f}')
        results.append(dict(measure=label, condition1=c1, condition2=c2,
                            t=t, df=len(valid)-1, p=p, d=d, n=len(valid)))
    return results


def bar_panel(df: pd.DataFrame, measures: list[tuple[str, str]], title: str, filename: str) -> None:
    ncols = len(measures)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]
    conds = [c for c in CONDS if c in df['condition'].unique()]

    for ax, (col, ylabel) in zip(axes, measures):
        piv = pivot_measure(df, col)
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
        ax.set_ylabel(ylabel)

    fig.suptitle(f'{title} (N={df["sub"].nunique()})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / filename, dpi=150)
    plt.close(fig)
    print(f'  Saved: {filename}')


def main() -> None:
    print('=' * 60)
    print('EYE-TRACKING ANALYSIS')
    print('=' * 60)

    df = load_et_summaries()
    df['sub'] = df['sub'].astype(str).str.zfill(3)
    df = df[df['sub'].isin(ANALYSIS_SUBJECTS)].copy()

    # Convert numeric columns
    numeric_cols = ['n_fix', 'mean_fix_dur', 'fix_rate',
                    'n_sacc', 'mean_sacc_amp', 'mean_sacc_pv', 'sacc_rate',
                    'pupil_mean', 'pupil_sd']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f'\n  Conditions: {sorted(df["condition"].unique())}')
    print(f'  N subjects: {df["sub"].nunique()}')

    # Group means
    print('\n' + '=' * 60)
    print('GROUP MEANS')
    print('=' * 60)
    group = df.groupby('condition')[numeric_cols].agg(['mean', 'sem'])
    for cond in CONDS:
        if cond not in df['condition'].unique():
            continue
        print(f'\n  {COND_LABELS.get(cond, cond)}:')
        for col in numeric_cols:
            if col not in df.columns:
                continue
            sub_vals = df[df['condition'] == cond][col].dropna()
            print(f'    {col:>18}: M = {sub_vals.mean():.3f}  SEM = {sub_vals.sem():.3f}  N = {len(sub_vals)}')

    # Inferential statistics
    print('\n' + '=' * 60)
    print('INFERENTIAL STATISTICS (Paired t-tests)')
    print('=' * 60)
    all_results = []
    measure_map = {
        'n_fix':        'Fixation count',
        'mean_fix_dur': 'Mean fixation duration (ms)',
        'fix_rate':     'Fixation rate (fix/s)',
        'mean_sacc_amp':'Mean saccade amplitude (px)',
        'mean_sacc_pv': 'Saccade peak velocity (px/s)',
        'pupil_mean':   'Mean pupil diameter (AU)',
    }
    for col, label in measure_map.items():
        if col in df.columns:
            all_results.extend(paired_tests(df, col, label))

    res_df = pd.DataFrame(all_results)
    if len(res_df) > 1:
        _, p_fdr, _, _ = multipletests(res_df['p'], method='fdr_bh')
        res_df['p_fdr'] = p_fdr

    # Figures
    bar_panel(df,
              [('n_fix', 'Fixation count'),
               ('mean_fix_dur', 'Mean fix. duration (ms)'),
               ('fix_rate', 'Fixation rate (fix/s)')],
              'Fixation Metrics', 'fixation_summary.png')

    bar_panel(df,
              [('mean_sacc_amp', 'Saccade amplitude (px)'),
               ('mean_sacc_pv', 'Peak velocity (px/s)'),
               ('sacc_rate', 'Saccade rate (sacc/s)')],
              'Saccade Metrics', 'saccade_summary.png')

    if 'pupil_mean' in df.columns:
        bar_panel(df,
                  [('pupil_mean', 'Pupil diameter (AU)'),
                   ('pupil_sd', 'Pupil SD (AU)')],
                  'Pupil Diameter', 'pupil_summary.png')

    # Save outputs
    res_df.to_csv(OUT / 'et_statistical_tests.csv', index=False)
    df.to_csv(OUT / 'et_subject_summaries.csv', index=False)
    print(f'\n  Saved: et_statistical_tests.csv, et_subject_summaries.csv')
    print(f'All results saved to: output/et_analysis/')


if __name__ == '__main__':
    main()
