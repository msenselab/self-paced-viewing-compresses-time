"""
ERP Waveform Figures
====================
Reads grand-average ERP files (data/group/grand_avg_*-ave.fif) and generates
publication-ready waveform plots and topographic maps.

Requires MNE-Python (pip install mne).

Output: output/erp_waveforms/

Usage:
    python code/06_erp_waveforms.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    sys.exit('ERROR: MNE-Python is required for this script.\n'
             'Install it with:  pip install mne')

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import GROUP_DIR, COLORS, COND_LABELS, ERP_WINDOWS, make_out

OUT = make_out('erp_waveforms')

POSTERIOR_CHS = ['O1', 'Oz', 'O2', 'PO3', 'PO4', 'POz', 'PO7', 'PO8']
PARIETAL_CHS  = ['Pz', 'CPz', 'P3', 'P4']


def load_grand_averages() -> dict:
    evokeds = {}
    for cond, label in [('active', 'Scrolling'), ('passive', 'Watching')]:
        fif = GROUP_DIR / f'grand_avg_{cond}-ave.fif'
        if not fif.exists():
            print(f'  WARNING: {fif.name} not found, skipping {cond}')
            continue
        evk = mne.read_evokeds(str(fif), verbose=False)
        evokeds[label] = evk[0] if isinstance(evk, list) else evk
        print(f'  Loaded: {fif.name}  ({evokeds[label].nave} epochs)')

    diff_fif = GROUP_DIR / 'grand_avg_difference-ave.fif'
    if diff_fif.exists():
        diff = mne.read_evokeds(str(diff_fif), verbose=False)
        evokeds['Difference'] = diff[0] if isinstance(diff, list) else diff
        print(f'  Loaded: {diff_fif.name}')
    return evokeds


def pick_cluster(evk: mne.Evoked, channels: list[str]) -> list[str]:
    return [ch for ch in channels if ch in evk.ch_names]


def waveform_figure(evokeds: dict) -> None:
    """Grand-average ERP waveforms for posterior and parietal clusters."""
    if len(evokeds) < 2:
        print('  Skipping waveform figure (need at least 2 conditions).')
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    cond_evks = {k: v for k, v in evokeds.items() if k != 'Difference'}
    diff_evk  = evokeds.get('Difference')

    for ax, (cluster_name, chs) in zip(axes, [
        ('Posterior (O1/Oz/O2/PO3–PO8)', POSTERIOR_CHS),
        ('Parietal (Pz/CPz/P3/P4)',       PARIETAL_CHS),
    ]):
        for label, evk in cond_evks.items():
            picks = pick_cluster(evk, chs)
            if not picks:
                continue
            data_uv = evk.data[np.ix_(
                [evk.ch_names.index(ch) for ch in picks],
                np.ones(len(evk.times), dtype=bool)
            )].mean(axis=0) * 1e6
            color = COLORS.get('active' if 'croll' in label else 'passive', 'gray')
            ax.plot(evk.times * 1000, data_uv, color=color, lw=1.8, label=label)

        if diff_evk is not None:
            picks = pick_cluster(diff_evk, chs)
            if picks:
                data_uv = diff_evk.data[np.ix_(
                    [diff_evk.ch_names.index(ch) for ch in picks],
                    np.ones(len(diff_evk.times), dtype=bool)
                )].mean(axis=0) * 1e6
                ax.plot(diff_evk.times * 1000, data_uv,
                        color='purple', lw=1.4, ls='--', alpha=0.8, label='Difference')

        for win_name, (t0, t1) in ERP_WINDOWS.items():
            ax.axvspan(t0 * 1000, t1 * 1000, alpha=0.05, color='gray')
        ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5, ls='--')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(cluster_name)
        ax.legend(loc='upper right', fontsize=9)
        ax.invert_yaxis()

    axes[-1].set_xlabel('Time (ms)')
    fig.suptitle('Grand-Average ERP — Picture Onset Locked', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'grand_average_waveforms.png', dpi=150)
    plt.close(fig)
    print('  Saved: grand_average_waveforms.png')


def topomap_figure(evokeds: dict) -> None:
    """Topographic maps at the centre of each ERP window."""
    cond_evks = {k: v for k, v in evokeds.items() if k != 'Difference'}
    if not cond_evks:
        return

    windows = list(ERP_WINDOWS.items())
    nrows = len(cond_evks) + (1 if 'Difference' in evokeds else 0)
    ncols = len(windows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))

    row_order = list(cond_evks.items())
    if 'Difference' in evokeds:
        row_order.append(('Difference', evokeds['Difference']))

    for row_idx, (label, evk) in enumerate(row_order):
        for col_idx, (win_name, (t0, t1)) in enumerate(windows):
            ax = axes[row_idx, col_idx] if nrows > 1 else axes[col_idx]
            t_mid = (t0 + t1) / 2
            try:
                evk.plot_topomap(times=[t_mid], axes=[ax], show=False, colorbar=False,
                                 time_unit='s', vlim=(-5, 5) if 'Diff' in label else None)
            except Exception:
                ax.set_visible(False)
                continue
            if row_idx == 0:
                ax.set_title(f'{win_name}\n{int(t_mid*1000)} ms', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=10)

    fig.suptitle('Topographic Maps — ERP Windows', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'grand_average_topomaps.png', dpi=150)
    plt.close(fig)
    print('  Saved: grand_average_topomaps.png')


def main() -> None:
    print('=' * 60)
    print('ERP WAVEFORM FIGURES')
    print('=' * 60)

    evokeds = load_grand_averages()
    if not evokeds:
        sys.exit('ERROR: no grand-average .fif files found in data/group/.')

    waveform_figure(evokeds)
    topomap_figure(evokeds)
    print(f'\nAll figures saved to: output/erp_waveforms/')


if __name__ == '__main__':
    main()
