# Self-paced Viewing Compresses Time — Open Data & Analysis Code

This repository contains intermediate data and analysis code to reproduce the
statistics and figures reported in:

> **The Attentional Thief: How Self-paced Visual Exploration Compresses Subjective Time**
>  Chunyu Qu, Artyom Zinchenko, Siyi Chen, and Zhuanghua Shi

Raw EEG (BrainVision) and raw eye-tracking (EyeLink EDF) files are not
included due to data-volume.
All reported inferential statistics and figures are fully reproducible from
the intermediate data provided here.

## Repository Layout

```
open_data/
├── data/
│   ├── behavioral/          # raw trial-level behavioral CSV per participant
│   ├── erp_summaries/       # per-participant ERP amplitude by condition × window × cluster
│   ├── frp_summaries/       # per-participant FRP amplitude by condition × window × cluster
│   ├── et_summaries/        # per-participant eye-tracking metrics (fixations, saccades, pupil)
│   └── group/               # group-level aggregates and grand-average ERP files
│       ├── erp_summary_all.csv
│       ├── frp_summary_all.csv
│       ├── behavioral_summary.csv
│       ├── frp_epoch_counts.csv
│       ├── brain_behavior_correlations.csv
│       ├── merged_erp_behavioral.csv
│       ├── grand_avg_active-ave.fif
│       ├── grand_avg_passive-ave.fif
│       └── grand_avg_difference-ave.fif
└── code/
    ├── config.py                  # shared paths and constants
    ├── 01_behavioral_analysis.py  # time estimation and recognition memory
    ├── 02_erp_statistics.py       # ERP amplitude comparisons (no MNE needed)
    ├── 03_frp_statistics.py       # FRP amplitude comparisons (no MNE needed)
    ├── 04_et_analysis.py          # fixation, saccade, and pupil statistics
    ├── 05_brain_behavior.py       # ERP × behavioral correlations (no MNE needed)
    └── 06_erp_waveforms.py        # ERP waveform figures from grand-average .fif (needs MNE)
```


## Participants

N = 23 valid participants (sub-001 through sub-026, excluding sub-015, sub-016,
sub-023, sub-027 due to behavioural outlier or data-quality criteria).

Subject IDs in all files are zero-padded three-digit strings (e.g., `001`).

## Conditions

| Code       | Label    | Description                                    |
|------------|----------|------------------------------------------------|
| `active`   | Scrolling | Participant controls image advance (scrolling) |
| `passive`  | Watching  | Yoked passive viewing                          |
| `constant` | Baseline  | Fixed-duration constant image display          |


## Data Files

### `data/behavioral/sub-XXX_behavior.csv`

Raw trial-level output from the PsychoPy experiment. Key columns:

| Column               | Description                                      |
|----------------------|--------------------------------------------------|
| `phase`              | `time_estimation` or `recognition`               |
| `session_type`       | `active`, `passive`, or `constant`               |
| `actual_duration`    | True block duration (s)                          |
| `estimated_duration` | Participant's estimate (s)                       |
| `estimation_ratio`   | `estimated / actual` (1.0 = veridical)           |
| `n_pictures`         | Number of images shown in the block              |
| `recognition_correct`| Boolean — correct recognition response           |
| `recognition_rt`     | Recognition response time (s)                    |

### `data/erp_summaries/sub-XXX_erp_summary.csv`

Mean ERP amplitude (µV) per participant × condition × time window × electrode cluster.
Picture-onset locked epochs (−200 to 1000 ms, baseline corrected).

| Column         | Description                                           |
|----------------|-------------------------------------------------------|
| `condition`    | `active` / `passive` / `constant`                     |
| `window`       | `P1` / `N1` / `P2` / `P3` / `Late`                   |
| `cluster`      | `occipital` / `posterior` / `parietal` / `frontal`    |
| `amplitude_uV` | Mean amplitude in that window × cluster               |

### `data/frp_summaries/sub-XXX_frp_summary.csv`

Fixation-onset locked FRP amplitudes (µV). Same structure as ERP summaries.
Windows: `lambda` (60–120 ms), `N1_frp` (120–200 ms),
`P2_frp` (200–350 ms), `Late_frp` (350–600 ms).

The `n` column gives the number of fixation-onset epochs contributing to this
participant's average.

### `data/et_summaries/sub-XXX_et_summary.csv`

Per-participant, per-block-type eye-tracking summary metrics:
fixation count and duration, saccade amplitude and peak velocity, pupil diameter.

### `data/et_summaries/sub-XXX_et_fixations.csv`

Individual fixation events with coordinates, duration, and block type.

### `data/group/erp_summary_all.csv` / `frp_summary_all.csv`

Long-format stacks of all participants' ERP/FRP summaries — equivalent to
concatenating the per-participant files. Used directly by scripts 02 and 03.

### `data/group/grand_avg_*-ave.fif`

MNE-Python grand-average Evoked objects for the `active`, `passive`, and
`active − passive` difference waveforms. Required only for `06_erp_waveforms.py`.
Read with `mne.read_evokeds(path)`.


## Reproduction Steps

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

MNE-Python is only required for `06_erp_waveforms.py`; all other scripts depend
only on NumPy, pandas, SciPy, Matplotlib, and statsmodels.

### 2. Run scripts from the `open_data/` root

All paths are relative to the repository root. Run each script with:

```bash
python code/01_behavioral_analysis.py   # time estimation + recognition
python code/02_erp_statistics.py        # ERP paired t-tests + bar figures
python code/03_frp_statistics.py        # FRP paired t-tests + bar figures
python code/04_et_analysis.py           # fixation / saccade / pupil stats
python code/05_brain_behavior.py        # ERP × behavioral correlations
python code/06_erp_waveforms.py         # ERP waveform + topomap figures (needs MNE)
```

Output is written to `output/<script_name>/` (created automatically).


## EEG Preprocessing Summary

The ERP/FRP summaries were derived from continuous EEG preprocessed as follows:

1. **Filtering** — 0.1–30 Hz bandpass, 50 Hz notch
2. **Online reference recovery** — FCz re-added to scalp
3. **Re-referencing** — average of TP9 + TP10 (mastoids)
4. **ICA** — applied with ICLabel; eye (≥0.8) and artefact (≥0.9) components rejected
5. **Epoching** — picture onset: −200 to 1000 ms; fixation onset: −200 to 800 ms
6. **Epoch rejection** — AutoReject + ±150 µV threshold
7. **Baseline correction** — −200 to 0 ms

Preprocessing code is available in the full project repository.

---

## License

Data and code are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
