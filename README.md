# Self-paced Viewing Compresses Time — Open Data & Analysis Code

This repository contains intermediate data and analysis code to reproduce the
statistics and figures reported in:

> **The Attentional Thief: Scrolling-Like Picture Streams Compress Subjective Time**
>  Chunyu Qu, Artyom Zinchenko, Siyi Chen, and Zhuanghua Shi

Raw EEG (BrainVision) and raw eye-tracking (EyeLink EDF) files are not
included due to data-volume.
All reported inferential statistics are reproducible from the intermediate data provided here, and so
are the main and supplementary figures except the waveform panels of Figures 4 and 5, which need
per-participant EEG epoch files (available from the corresponding author on request).

## Repository Layout

```
open_data/
├── data/
│   ├── behavioral/          # raw trial-level behavioral CSV per participant
│   ├── behavioral_excluded/ # raw behavioral CSV of excluded sub-015, 016, 027 (sensitivity analyses only)
│   ├── revision/            # EEG/ET-derived tables behind the revised Supplementary Materials
│   ├── figure_assets/       # experimental-setup illustration composited into Figure 1
│   ├── erp_summaries/       # per-participant ERP amplitude by condition × window × cluster
│   ├── frp_summaries/       # per-participant FRP amplitude by condition × window × cluster
│   ├── et_summaries/        # per-participant eye-tracking metrics (fixations, saccades, pupil)
│   └── group/               # group-level aggregates and grand-average ERP files
│       ├── behavioral_trial_level.csv           # cleaned trial-level data, analysed N = 23
│       ├── behavioral_trial_level_excluded.csv  # same format, sub-015, 016, 027
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
    ├── revision/                  # analyses and figures added in revision (see below)
    ├── figures/                   # scripts for the manuscript's Figures 1–5 (see below)
    ├── config.py                  # shared paths and constants
    ├── 01_behavioral_analysis.py  # time estimation and recognition memory
    ├── 02_erp_statistics.py       # ERP amplitude comparisons (no MNE needed)
    ├── 03_frp_statistics.py       # FRP amplitude comparisons (no MNE needed)
    ├── 04_et_analysis.py          # fixation, saccade, and pupil statistics
    ├── 05_brain_behavior.py       # ERP × behavioral correlations (no MNE needed)
    └── 06_erp_waveforms.py        # ERP waveform figures from grand-average .fif (needs MNE)
```


## Participants

N = 23 analysed participants (sub-001 through sub-026). Four were excluded during data collection by
investigator decision: sub-023 for technical failure (very high EEG impedance and eye-tracker failure);
sub-015 and sub-016 because their Scrolling–Watching differences in raw estimates lay more than 2 SD from
the mean, one in each direction; and sub-027 because they misunderstood the task and estimated durations
objectively rather than as perceived. The study was not preregistered.

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


## Revision Analyses (Supplementary Materials)

The analyses and figures added in the first revision are in `code/revision/`. Run them from the
`open_data/` root in this order; outputs go to `output/revision/` and figures to `output/revision/figures/`.

| Script | Reproduces |
|---|---|
| `revision_behavioral_audit.py` | Paired contrasts with precision and exclusion sensitivity (N = 23, 24, 26), position models, lag-1 models, split by Watching position (A1–A2g) |
| `analysis_20260915_baseline_contrasts_by_position.py` | Baseline contrasts by session position (A2h) |
| `analysis_20260918_condition_position_rm_anova.py` | 2 × 2 Condition × Position ANOVA (Table S4.1) |
| `diagnostic_20260918_ratio_denominator.py` | Same ANOVA on actual block duration (ratio denominator check) |
| `analysis_20260924_repetition_order_sensitivity.py` | Contrasts by repetition, first/last-sequence and time-term sensitivity models |
| `analysis_20260519_fixation_interval_accounting.py` | Fixation-interval accounting (Supplement S3) |
| `frp_single_fixation_models.R` | Single-fixation FRP models with participant and image random intercepts (Figure S6.2A; R with lme4, lmerTest, emmeans) |
| `fit_lambda_saccade_covariate.R` | Saccade-amplitude covariate on the lambda response (Figure S6.2B; R with lme4 and lmerTest) |
| `analysis_20261001_sequential_assimilation.py` | Sequential-assimilation models (post hoc; Figure S4.1B) |
| `figure_S4_position.py` | Figure S4.1 (after the sequential-assimilation script) |
| `figures_supplement.py` | Figures S2.1, S3.1, S4.2, S4.3, S5.1, S6.1, S6.2 (run after the scripts above) |

The behavioral analyses start from `data/group/behavioral_trial_level.csv`. The raw PsychoPy files in
`data/behavioral/` contain list-valued columns that break naive CSV parsing; the trial-level file is the
cleaned version of the same records. The three excluded participants (`data/behavioral_excluded/`,
`data/group/behavioral_trial_level_excluded.csv`) enter only the exclusion-sensitivity samples
(N = 24 adds sub-027; N = 26 adds sub-015, 016, and 027); sub-023 (technical failure) is not used.

`data/revision/` holds the EEG- and eye-tracking-derived inputs that cannot be regenerated without the raw
recordings: retained epoch counts per participant, picture-locked window amplitudes with and without baseline
correction, the fixation-locked condition comparisons, single-fixation epoch amplitudes, the
Scrolling/Watching single-fixation lambda data with incoming saccade amplitude, the reported lme4 output, and
the yoking check of all 267 Scrolling–Watching block pairs.

## Main Figures

`code/figures/` holds the scripts that draw the manuscript's Figures 1–5. Run them from the `open_data/` root;
figures are written to `output/figures/`. Script names follow the order in which the figures were drafted,
so the mapping to the published numbering is:

| Manuscript figure | Script | Inputs |
|---|---|---|
| Figure 1 (design schematic) | `figures_20260601_F1_illustration_redraw.py` | `data/figure_assets/fig1_illustration_original_ai.png` (needs Pillow) |
| Figure 2 (time compression, fixation intervals) | `figures_20260519_F2_time_compression.py` | `data/group/behavioral_trial_level.csv`; run `code/revision/analysis_20260519_fixation_interval_accounting.py` first |
| Figure 3 (eye movements) | `figures_20260519_F5_eye_movements.py` | `data/et_summaries/sub-*_et_summary.csv` |
| Figure 4 (picture-locked ERP) | `figures_20260519_F3_picture_erp.py` | `data/revision/an6_preonset_subject_level.csv`; waveform panels need `data/derivatives/sub-*/picture_onset-epo.fif` (MNE; on request) |
| Figure 5 (fixation-locked FRP) | `figures_20260519_F4_fixation_frp.py` | `data/group/frp_summary_all.csv`; waveform panel needs `data/derivatives/sub-*/fixation_onset-epo.fif` (MNE; on request) |

Figures 1, 2, and 3 regenerated from this repository are pixel-identical to the published versions.

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
