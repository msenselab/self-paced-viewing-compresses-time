"""
Behavioral Analysis
===================
Reads the raw behavioral CSVs in data/behavioral/ and reproduces:
  - Time estimation ratio by condition (paired t-tests, Friedman)
  - Recognition accuracy, RT, and d' (scrolling vs watching)
  - Per-picture viewing duration patterns

Output: output/behavioral/

Usage:
    python code/01_behavioral_analysis.py
"""
import sys
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANALYSIS_SUBJECTS, BEH_DIR, COLORS, COND_LABELS, make_out

OUT = make_out('behavioral')


# ── CSV parser (handles embedded arrays with commas) ─────────────────────────
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


def load_all_behavioral() -> pd.DataFrame:
    all_rows = []
    for sub in ANALYSIS_SUBJECTS:
        csv_path = BEH_DIR / f'sub-{sub}_behavior.csv'
        if not csv_path.exists():
            print(f'  sub-{sub}: missing behavioral CSV, skipping')
            continue
        df = parse_behavioral_csv(csv_path)
        if df.empty:
            continue
        df['sub'] = sub
        all_rows.append(df)
        print(f'  sub-{sub}: {len(df)} rows')
    if not all_rows:
        sys.exit('ERROR: no behavioral data found in data/behavioral/')
    return pd.concat(all_rows, ignore_index=True)


def clean_data(df: pd.DataFrame):
    est = df[df['phase'] == 'time_estimation'].copy()
    for col in ['actual_duration', 'estimated_duration', 'estimation_ratio',
                'estimation_error', 'n_pictures', 'rep']:
        if col in est.columns:
            est[col] = pd.to_numeric(est[col], errors='coerce')
    est['session_type'] = est['session_type'].str.strip().str.lower()
    est = est[est['session_type'].isin(['active', 'passive', 'constant'])].copy()

    recog = est[est['recognition_image'].astype(str).str.strip().ne('')].copy()
    if not recog.empty:
        for col in ['recognition_rt']:
            if col in recog.columns:
                recog[col] = pd.to_numeric(recog[col], errors='coerce')
        for col in ['recognition_is_old', 'recognition_correct']:
            if col in recog.columns:
                recog[col] = recog[col].astype(str).str.strip().str.lower().map(
                    {'true': True, 'false': False, '1': True, '0': False})
        recog['recognition_response'] = recog['recognition_response'].astype(str).str.strip().str.lower()
    return est, recog


# ── Estimation ───────────────────────────────────────────────────────────────
def estimation_analysis(est: pd.DataFrame) -> tuple:
    print('\n' + '=' * 60)
    print('TIME ESTIMATION ANALYSIS')
    print('=' * 60)

    sub_means = est.groupby(['sub', 'session_type'])['estimation_ratio'].mean().reset_index()
    pivot = sub_means.pivot(index='sub', columns='session_type', values='estimation_ratio')
    n = len(pivot)
    conds = [c for c in ['active', 'passive', 'constant'] if c in pivot.columns]

    print(f'\nN = {n} subjects')
    for cond in conds:
        m, s = pivot[cond].mean(), pivot[cond].sem()
        print(f'  {cond:>10}: M = {m:.3f}  SEM = {s:.3f}')

    print('\nOne-sample t-tests vs 1.0:')
    for cond in conds:
        t, p = stats.ttest_1samp(pivot[cond].dropna(), 1.0)
        d = (pivot[cond].mean() - 1.0) / pivot[cond].std()
        print(f'  {cond:>10}: t({len(pivot[cond].dropna())-1}) = {t:.2f}  p = {p:.4f}  d = {d:.2f}')

    print('\nPaired comparisons:')
    comp_results = []
    for c1, c2 in combinations(conds, 2):
        valid = pivot[[c1, c2]].dropna()
        if len(valid) >= 3:
            t, p = stats.ttest_rel(valid[c1], valid[c2])
            d = (valid[c1] - valid[c2]).mean() / (valid[c1] - valid[c2]).std()
            print(f'  {c1} vs {c2}: t({len(valid)-1}) = {t:.2f}  p = {p:.4f}  d = {d:.2f}')
            comp_results.append(dict(comparison=f'{c1}_vs_{c2}', t=t, p=p, d=d, n=len(valid)))

    if len(conds) >= 3:
        valid3 = pivot[conds].dropna()
        if len(valid3) >= 3:
            F, p = stats.friedmanchisquare(*[valid3[c] for c in conds])
            print(f'\n  Friedman: χ²({len(conds)-1}) = {F:.2f}  p = {p:.4f}')

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax = axes[0]
    x = range(len(conds))
    ax.bar(x, [pivot[c].mean() for c in conds],
           yerr=[pivot[c].sem() for c in conds],
           color=[COLORS.get(c, 'gray') for c in conds],
           alpha=0.7, capsize=5, width=0.6)
    for i, c in enumerate(conds):
        jitter = np.random.default_rng(42).normal(0, 0.06, len(pivot[c].dropna()))
        ax.scatter(i + jitter, pivot[c].dropna(), color='black', alpha=0.4, s=20, zorder=5)
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='Veridical')
    ax.set_xticks(x); ax.set_xticklabels([COND_LABELS.get(c, c) for c in conds])
    ax.set_ylabel('Estimation Ratio (estimated / actual)'); ax.set_title('Time Estimation'); ax.legend()

    ax = axes[1]
    for cond in conds:
        pm = est[est['session_type'] == cond].groupby('n_pictures')['estimation_ratio']
        ax.errorbar(pm.mean().index, pm.mean().values, yerr=pm.sem().values,
                    color=COLORS.get(cond, 'gray'), marker='o', lw=2,
                    label=COND_LABELS.get(cond, cond), capsize=3)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('Number of Pictures'); ax.set_ylabel('Estimation Ratio')
    ax.set_title('Estimation by Pic Count'); ax.legend()

    ax = axes[2]
    for cond in conds:
        pm = est[est['session_type'] == cond].groupby('rep')['estimation_ratio']
        ax.errorbar(pm.mean().index, pm.mean().values, yerr=pm.sem().values,
                    color=COLORS.get(cond, 'gray'), marker='o', lw=2,
                    label=COND_LABELS.get(cond, cond), capsize=3)
    ax.axhline(1.0, color='gray', ls='--', lw=1)
    ax.set_xlabel('Repetition'); ax.set_ylabel('Estimation Ratio')
    ax.set_title('Estimation by Repetition'); ax.legend()

    fig.suptitle(f'Time Estimation Analysis (N={n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'estimation_by_condition.png', dpi=150)
    plt.close(fig)
    print(f'\n  Saved: estimation_by_condition.png')
    return pivot, comp_results


# ── Recognition ──────────────────────────────────────────────────────────────
def recognition_analysis(recog: pd.DataFrame) -> tuple:
    print('\n' + '=' * 60)
    print('RECOGNITION ANALYSIS')
    print('=' * 60)
    if recog.empty:
        print('  No recognition data found.')
        return None, []

    recog = recog[recog['session_type'].isin(['active', 'passive'])].copy()
    acc = recog.groupby(['sub', 'session_type'])['recognition_correct'].mean().reset_index()
    pivot_acc = acc.pivot(index='sub', columns='session_type', values='recognition_correct')

    correct_only = recog[recog['recognition_correct'] == True]
    rt = correct_only.groupby(['sub', 'session_type'])['recognition_rt'].mean().reset_index()
    pivot_rt = rt.pivot(index='sub', columns='session_type', values='recognition_rt')

    n = len(pivot_acc)
    print(f'\nN = {n} subjects')

    for measure, piv, label in [('Accuracy', pivot_acc, ''), ('RT (s)', pivot_rt, '')]:
        print(f'\n{measure}:')
        for cond in ['active', 'passive']:
            if cond in piv.columns:
                m, s = piv[cond].mean(), piv[cond].sem()
                print(f'  {cond:>10}: M = {m:.3f}  SEM = {s:.3f}')

    comp_results = []
    for piv, label in [(pivot_acc, 'Accuracy'), (pivot_rt, 'RT')]:
        valid = piv[['active', 'passive']].dropna()
        if len(valid) >= 3:
            t, p = stats.ttest_rel(valid['active'], valid['passive'])
            d = (valid['active'] - valid['passive']).mean() / (valid['active'] - valid['passive']).std()
            print(f'  {label} scrolling vs watching: t({len(valid)-1}) = {t:.2f}  p = {p:.4f}  d = {d:.2f}')
            comp_results.append(dict(measure=label, t=t, p=p, d=d, n=len(valid)))

    # Signal detection
    dprime_rows = []
    for sub in recog['sub'].unique():
        for cond in ['active', 'passive']:
            sc = recog[(recog['sub'] == sub) & (recog['session_type'] == cond)]
            if sc.empty:
                continue
            old_t = sc[sc['recognition_is_old'] == True]
            new_t = sc[sc['recognition_is_old'] == False]
            hr = (old_t['recognition_response'] == 'old').mean() if len(old_t) else np.nan
            fa = (new_t['recognition_response'] == 'old').mean() if len(new_t) else np.nan
            no, nn = len(old_t), len(new_t)
            if hr == 1.0: hr = 1 - 1/(2*no)
            if hr == 0.0: hr = 1/(2*no)
            if fa == 1.0: fa = 1 - 1/(2*nn)
            if fa == 0.0: fa = 1/(2*nn)
            dp = stats.norm.ppf(hr) - stats.norm.ppf(fa) if not (np.isnan(hr) or np.isnan(fa)) else np.nan
            c_ = -0.5 * (stats.norm.ppf(hr) + stats.norm.ppf(fa)) if not (np.isnan(hr) or np.isnan(fa)) else np.nan
            dprime_rows.append(dict(sub=sub, session_type=cond, hit_rate=hr, fa_rate=fa, d_prime=dp, criterion=c_))

    dprime_df = pd.DataFrame(dprime_rows)
    pivot_dp = dprime_df.pivot(index='sub', columns='session_type', values='d_prime')
    print("\nSignal detection (d'):")
    for cond in ['active', 'passive']:
        if cond in pivot_dp.columns:
            m, s = pivot_dp[cond].mean(), pivot_dp[cond].sem()
            print(f"  d' {cond:>10}: M = {m:.2f}  SEM = {s:.2f}")
    valid_dp = pivot_dp[['active', 'passive']].dropna()
    if len(valid_dp) >= 3:
        t, p = stats.ttest_rel(valid_dp['active'], valid_dp['passive'])
        d = (valid_dp['active'] - valid_dp['passive']).mean() / (valid_dp['active'] - valid_dp['passive']).std()
        print(f"  d' scrolling vs watching: t({len(valid_dp)-1}) = {t:.2f}  p = {p:.4f}  d = {d:.2f}")

    # Figure
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    pivot_c = dprime_df.pivot(index='sub', columns='session_type', values='criterion')
    for ax, (piv, ylabel, title) in zip(axes, [
        (pivot_acc, 'Accuracy',        'Recognition Accuracy'),
        (pivot_rt,  'RT (s)',          'Recognition RT'),
        (pivot_dp,  "d'",              "Sensitivity (d')"),
        (pivot_c,   'Criterion (c)',   'Response Bias'),
    ]):
        for i, cond in enumerate(['active', 'passive']):
            if cond not in piv.columns:
                continue
            vals = piv[cond].dropna()
            jitter = np.random.default_rng(42).normal(0, 0.05, len(vals))
            ax.scatter(i + jitter, vals, color=COLORS.get(cond, 'gray'), alpha=0.5, s=25)
            ax.bar(i, vals.mean(), yerr=vals.sem(), color=COLORS.get(cond, 'gray'), alpha=0.4, capsize=5, width=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Scrolling', 'Watching'])
        ax.set_ylabel(ylabel); ax.set_title(title)
    axes[3].axhline(0, color='gray', ls='--', lw=1)

    fig.suptitle(f'Recognition Performance (N={n})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'recognition_performance.png', dpi=150)
    plt.close(fig)
    dprime_df.to_csv(OUT / 'dprime_table.csv', index=False)
    print(f'\n  Saved: recognition_performance.png, dprime_table.csv')
    return pivot_acc, comp_results


# ── Viewing durations ────────────────────────────────────────────────────────
def viewing_duration_analysis(est: pd.DataFrame) -> None:
    print('\n' + '=' * 60)
    print('VIEWING DURATION ANALYSIS')
    print('=' * 60)
    dur_rows = []
    for _, row in est.iterrows():
        ppd = row.get('per_picture_durations', '')
        if not isinstance(ppd, str) or not ppd.strip():
            continue
        ppd_clean = ppd.strip().strip('"').strip('[]')
        try:
            durs = [float(x.strip()) for x in ppd_clean.split(',') if x.strip()]
        except ValueError:
            continue
        for i, d in enumerate(durs):
            dur_rows.append(dict(sub=row['sub'], session_type=row['session_type'],
                                 n_pictures=row['n_pictures'], pic_idx=i+1, pic_duration=d))
    if not dur_rows:
        print('  No per-picture duration data found.')
        return
    dur_df = pd.DataFrame(dur_rows)
    dur_df['pic_duration'] = pd.to_numeric(dur_df['pic_duration'], errors='coerce')
    dur_df['n_pictures'] = pd.to_numeric(dur_df['n_pictures'], errors='coerce')
    dur_df = dur_df.dropna(subset=['pic_duration'])

    print(f'  Total viewing episodes: {len(dur_df)}')
    for cond in ['active', 'passive', 'constant']:
        v = dur_df[dur_df['session_type'] == cond]['pic_duration']
        if len(v):
            print(f'  {cond:>10}: M = {v.mean():.2f}s  SD = {v.std():.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    conds = ['active', 'passive', 'constant']
    ax = axes[0]
    for cond in conds:
        pm = dur_df[dur_df['session_type'] == cond].groupby('pic_idx')['pic_duration']
        idx = pm.mean().index[pm.mean().index <= 18]
        ax.errorbar(idx, pm.mean()[idx], yerr=pm.sem()[idx],
                    color=COLORS.get(cond, 'gray'), marker='.', lw=1.5,
                    label=COND_LABELS.get(cond, cond), capsize=2, alpha=0.8)
    ax.set_xlabel('Picture Position'); ax.set_ylabel('Viewing Duration (s)')
    ax.set_title('Duration by Picture Position'); ax.legend()

    ax = axes[1]
    dur_df['pos_bin'] = pd.cut(
        (dur_df['pic_idx'] - 1) / (dur_df['n_pictures'] - 1).clip(lower=1),
        bins=5, labels=['1st', '2nd', '3rd', '4th', '5th'])
    for cond in conds:
        pm = dur_df[dur_df['session_type'] == cond].groupby('pos_bin', observed=True)['pic_duration']
        ax.errorbar(range(len(pm.mean())), pm.mean().values, yerr=pm.sem().values,
                    color=COLORS.get(cond, 'gray'), marker='o', lw=2,
                    label=COND_LABELS.get(cond, cond), capsize=3)
    ax.set_xticks(range(5)); ax.set_xticklabels(['Early', '', 'Mid', '', 'Late'])
    ax.set_xlabel('Normalized Position'); ax.set_ylabel('Viewing Duration (s)')
    ax.set_title('Duration by Block Position'); ax.legend()

    fig.suptitle('Per-Picture Viewing Duration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'viewing_durations.png', dpi=150)
    plt.close(fig)
    print('  Saved: viewing_durations.png')


# ── Summary CSV ──────────────────────────────────────────────────────────────
def save_summary(est: pd.DataFrame, recog: pd.DataFrame) -> None:
    rows = []
    for sub in sorted(est['sub'].unique()):
        row = {'sub': sub}
        for cond in ['active', 'passive', 'constant']:
            sc = est[(est['sub'] == sub) & (est['session_type'] == cond)]
            row[f'est_ratio_{cond}'] = sc['estimation_ratio'].mean() if len(sc) else np.nan
            row[f'n_blocks_{cond}'] = len(sc)
        if not recog.empty:
            for cond in ['active', 'passive']:
                sc = recog[(recog['sub'] == sub) & (recog['session_type'] == cond)]
                if len(sc):
                    row[f'recog_acc_{cond}'] = sc['recognition_correct'].mean()
                    row[f'recog_rt_{cond}'] = pd.to_numeric(
                        sc[sc['recognition_correct'] == True]['recognition_rt'], errors='coerce').mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'behavioral_summary.csv', index=False)
    print(f'\n  Saved: behavioral_summary.csv ({len(rows)} subjects)')


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print('=' * 60)
    print('BEHAVIORAL ANALYSIS')
    print('=' * 60)
    df = load_all_behavioral()
    print(f'\n  Total rows: {len(df)}')
    est, recog = clean_data(df)
    pivot_est, est_results   = estimation_analysis(est)
    pivot_acc, recog_results = recognition_analysis(recog)
    viewing_duration_analysis(est)
    save_summary(est, recog)

    all_stats = ([{**r, 'measure': 'estimation_ratio'} for r in est_results]
                 + list(recog_results))
    pd.DataFrame(all_stats).to_csv(OUT / 'statistical_tests.csv', index=False)
    print(f'\nAll results saved to: output/behavioral/')


if __name__ == '__main__':
    main()
