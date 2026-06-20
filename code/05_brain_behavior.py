"""
Brain-Behavior Correlations
============================
Reads pre-computed ERP amplitude summaries (data/group/erp_summary_all.csv)
and behavioral data (data/behavioral/) to reproduce brain-behavior correlations.

No MNE or epoch files required.

Reproduces:
  - Per-condition ERP amplitude vs time estimation ratio (Pearson r)
  - ERP difference (scrolling - watching) vs estimation ratio difference
  - ERP amplitude vs recognition accuracy
  - FDR-corrected results table + scatter plots

Output: output/brain_behavior/

Usage:
    python code/05_brain_behavior.py
"""
import sys
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
    ANALYSIS_SUBJECTS, GROUP_DIR, BEH_DIR,
    COLORS, COND_LABELS, ERP_WINDOWS, make_out,
)

OUT = make_out('brain_behavior')
CONDS = ['active', 'passive']


# ── Behavioral loader (minimal — same CSV parser as 01_behavioral_analysis.py) ──
def parse_behavioral_csv(csv_path: Path) -> pd.DataFrame:
    rows = []
    with open(csv_path) as fh:
        lines = fh.readlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0].strip().rstrip(',').split(',')
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        fields, current, in_quotes, bracket_depth = [], [], False, 0
        for ch in line:
            if ch == '"' and bracket_depth == 0:
                in_quotes = not in_quotes; current.append(ch)
            elif ch == '[' and in_quotes:
                bracket_depth += 1; current.append(ch)
            elif ch == ']' and bracket_depth > 0:
                bracket_depth -= 1; current.append(ch)
            elif ch == ',' and not in_quotes and bracket_depth == 0:
                fields.append(''.join(current).strip()); current = []
            else:
                current.append(ch)
        fields.append(''.join(current).strip())
        row = {h.strip(): (fields[i] if i < len(fields) else '') for i, h in enumerate(header)}
        rows.append(row)
    return pd.DataFrame(rows)


def load_behavioral_summary() -> pd.DataFrame:
    """Return per-subject × condition estimation ratio and recognition accuracy."""
    rows = []
    for sub in ANALYSIS_SUBJECTS:
        csv_path = BEH_DIR / f'sub-{sub}_behavior.csv'
        if not csv_path.exists():
            continue
        df = parse_behavioral_csv(csv_path)
        if df.empty:
            continue
        est = df[df['phase'] == 'time_estimation'].copy()
        est['session_type'] = est['session_type'].str.strip().str.lower()
        est['estimation_ratio'] = pd.to_numeric(est['estimation_ratio'], errors='coerce')
        row = {'sub': sub}
        for cond in ['active', 'passive', 'constant']:
            v = est[est['session_type'] == cond]['estimation_ratio'].dropna()
            row[f'est_ratio_{cond}'] = v.mean() if len(v) else np.nan
        if not np.isnan(row.get('est_ratio_active', np.nan)) and not np.isnan(row.get('est_ratio_passive', np.nan)):
            row['est_ratio_diff'] = row['est_ratio_active'] - row['est_ratio_passive']
        else:
            row['est_ratio_diff'] = np.nan

        # Recognition (parse recognition_correct from the row)
        recog = est[est['recognition_image'].astype(str).str.strip().ne('')]
        for col in ['recognition_is_old', 'recognition_correct']:
            if col in recog.columns:
                recog = recog.copy()
                recog[col] = recog[col].astype(str).str.strip().str.lower().map(
                    {'true': True, 'false': False, '1': True, '0': False})
        for cond in ['active', 'passive']:
            v = recog[recog['session_type'] == cond]['recognition_correct'].dropna()
            row[f'recog_acc_{cond}'] = v.mean() if len(v) else np.nan
        if not np.isnan(row.get('recog_acc_active', np.nan)) and not np.isnan(row.get('recog_acc_passive', np.nan)):
            row['recog_diff'] = row['recog_acc_active'] - row['recog_acc_passive']
        else:
            row['recog_diff'] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_erp_wide(df_erp: pd.DataFrame) -> pd.DataFrame:
    """Pivot ERP long-format to wide (one row per subject)."""
    # Use occipital cluster as primary ROI (consistent with paper)
    cluster = 'occipital' if 'occipital' in df_erp['cluster'].unique() else df_erp['cluster'].unique()[0]
    sub_df = df_erp[df_erp['cluster'] == cluster].copy()
    sub_df['sub'] = sub_df['sub'].astype(str).str.zfill(3)
    amp_col = 'amplitude_uV' if 'amplitude_uV' in sub_df.columns else 'mean_amp_uv'

    wide = sub_df.pivot_table(index='sub', columns=['condition', 'window'], values=amp_col)
    wide.columns = [f'erp_{c}_{w}' for c, w in wide.columns]
    wide = wide.reset_index()

    # Compute scrolling - watching differences
    for win in ERP_WINDOWS:
        a, p = f'erp_active_{win}', f'erp_passive_{win}'
        if a in wide.columns and p in wide.columns:
            wide[f'erp_diff_{win}'] = wide[a] - wide[p]
    return wide


def run_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    results = []
    windows = list(ERP_WINDOWS.keys())

    print('\n1. ERP vs Time Estimation Ratio (within condition)')
    print('-' * 55)
    for cond in CONDS:
        for win in windows:
            x_col = f'erp_{cond}_{win}'
            y_col = f'est_ratio_{cond}'
            if x_col not in merged or y_col not in merged:
                continue
            valid = merged[[x_col, y_col]].dropna()
            if len(valid) < 5:
                continue
            r, p = stats.pearsonr(valid[x_col], valid[y_col])
            sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            print(f'  {cond:>8} {win:<8}: r = {r:+.3f}  p = {p:.4f}{sig}')
            results.append(dict(comparison='ERP_vs_EstRatio', condition=cond,
                                window=win, r=r, p=p, n=len(valid)))

    print('\n2. ERP difference (A-W) vs Estimation Ratio difference (A-W)')
    print('-' * 55)
    for win in windows:
        x_col = f'erp_diff_{win}'
        y_col = 'est_ratio_diff'
        if x_col not in merged or y_col not in merged:
            continue
        valid = merged[[x_col, y_col]].dropna()
        if len(valid) < 5:
            continue
        r, p = stats.pearsonr(valid[x_col], valid[y_col])
        sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
        print(f'  {win:<8}: r = {r:+.3f}  p = {p:.4f}{sig}')
        results.append(dict(comparison='ERPdiff_vs_EstRatioDiff', condition='A-W',
                            window=win, r=r, p=p, n=len(valid)))

    print('\n3. ERP vs Recognition Accuracy (within condition)')
    print('-' * 55)
    for cond in CONDS:
        for win in windows:
            x_col = f'erp_{cond}_{win}'
            y_col = f'recog_acc_{cond}'
            if x_col not in merged or y_col not in merged:
                continue
            valid = merged[[x_col, y_col]].dropna()
            if len(valid) < 5:
                continue
            r, p = stats.pearsonr(valid[x_col], valid[y_col])
            sig = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            print(f'  {cond:>8} {win:<8}: r = {r:+.3f}  p = {p:.4f}{sig}')
            results.append(dict(comparison='ERP_vs_RecogAcc', condition=cond,
                                window=win, r=r, p=p, n=len(valid)))

    res_df = pd.DataFrame(results)
    if len(res_df) > 1:
        _, p_fdr, _, _ = multipletests(res_df['p'], method='fdr_bh')
        res_df['p_fdr'] = p_fdr
        sig_n = (p_fdr < 0.05).sum()
        print(f'\nFDR correction (BH) applied: {sig_n}/{len(res_df)} significant')
    return res_df


def scatter_plots(merged: pd.DataFrame) -> None:
    windows = [w for w in ERP_WINDOWS if f'erp_active_{w}' in merged.columns]

    # ERP vs estimation ratio (scrolling)
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4))
    if len(windows) == 1:
        axes = [axes]
    fig.suptitle('ERP (Scrolling) vs Time Estimation Ratio', fontsize=12)
    for ax, win in zip(axes, windows):
        x = merged[f'erp_active_{win}'].values
        y = merged['est_ratio_active'].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 3:
            ax.scatter(x[mask], y[mask], c='k', s=30, alpha=0.7)
            r, p = stats.pearsonr(x[mask], y[mask])
            m_, b_ = np.polyfit(x[mask], y[mask], 1)
            xr = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xr, m_ * xr + b_, 'r--', alpha=0.5)
            ax.set_title(f'{win}\nr={r:.2f}, p={p:.3f}', fontsize=10)
        ax.set_xlabel('ERP (µV)')
        if ax is axes[0]:
            ax.set_ylabel('Estimation ratio')
    plt.tight_layout()
    fig.savefig(OUT / 'scatter_erp_vs_estimation.png', dpi=150)
    plt.close(fig)

    # ERP difference vs estimation difference
    fig, axes = plt.subplots(1, len(windows), figsize=(4 * len(windows), 4))
    if len(windows) == 1:
        axes = [axes]
    fig.suptitle('ERP Difference (A−W) vs Estimation Ratio Difference', fontsize=12)
    for ax, win in zip(axes, windows):
        x_col, y_col = f'erp_diff_{win}', 'est_ratio_diff'
        if x_col not in merged.columns:
            continue
        x = merged[x_col].values
        y = merged[y_col].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 3:
            ax.scatter(x[mask], y[mask], c='k', s=30, alpha=0.7)
            r, p = stats.pearsonr(x[mask], y[mask])
            m_, b_ = np.polyfit(x[mask], y[mask], 1)
            xr = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xr, m_ * xr + b_, 'r--', alpha=0.5)
            ax.set_title(f'{win}\nr={r:.2f}, p={p:.3f}', fontsize=10)
        ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
        ax.set_xlabel('ΔERP (µV)')
        if ax is axes[0]:
            ax.set_ylabel('ΔEstimation ratio')
    plt.tight_layout()
    fig.savefig(OUT / 'scatter_erp_diff_vs_estimation_diff.png', dpi=150)
    plt.close(fig)
    print('  Saved: scatter_erp_vs_estimation.png, scatter_erp_diff_vs_estimation_diff.png')


def main() -> None:
    print('=' * 60)
    print('BRAIN-BEHAVIOR CORRELATIONS')
    print('=' * 60)

    # Load ERP amplitudes from group CSV
    erp_csv = GROUP_DIR / 'erp_summary_all.csv'
    if not erp_csv.exists():
        sys.exit(f'ERROR: {erp_csv} not found. Run 02_erp_statistics.py first or check data/group/.')
    df_erp = pd.read_csv(erp_csv)

    # Load behavioral data
    print('\nLoading behavioral data...')
    beh_df = load_behavioral_summary()
    print(f'  {len(beh_df)} subjects with behavioral data')

    # Pivot ERP to wide format
    erp_wide = load_erp_wide(df_erp)
    erp_wide['sub'] = erp_wide['sub'].astype(str).str.zfill(3)

    # Merge
    merged = beh_df.merge(erp_wide, on='sub')
    merged = merged[merged['sub'].isin(ANALYSIS_SUBJECTS)].copy()
    print(f'  {len(merged)} subjects in merged dataset')

    # Run correlations
    print('\n' + '=' * 60)
    print('CORRELATIONS')
    print('=' * 60)
    res_df = run_correlations(merged)

    # Scatter plots
    print('\nGenerating scatter plots...')
    scatter_plots(merged)

    # Save
    res_df.to_csv(OUT / 'brain_behavior_correlations.csv', index=False)
    merged.to_csv(OUT / 'merged_erp_behavioral.csv', index=False)
    print(f'\n  Saved: brain_behavior_correlations.csv, merged_erp_behavioral.csv')
    print(f'All results saved to: output/brain_behavior/')


if __name__ == '__main__':
    main()
