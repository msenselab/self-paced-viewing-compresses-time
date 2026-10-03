"""Diagnostic for the estimation-ratio denominator in the position analysis."""
from pathlib import Path

import pandas as pd
import pingouin as pg

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ROOT, DATA_DIR, OUT_DIR  # open_data/ paths
SRC = DATA_DIR / "group/behavioral_trial_level.csv"
OUT = OUT_DIR / "revision/condition_position_rm_anova_20260918"

x = pd.read_csv(SRC)
x['sub'] = x['sub'].astype(str).str.zfill(3)
x = x.loc[x['phase'].eq('time_estimation')]
x = x.loc[x['sub'].isin({'015', '016', '023', '027'}).eq(False)].copy()
x['condition'] = x['session_type'].map({'active': 'Scrolling', 'passive': 'Watching', 'constant': 'Baseline'})
x['position'] = ((x['session_order'] - 1) % 3) + 1
x = x.loc[x['condition'].isin(['Watching', 'Baseline'])]

cell = x.groupby(['sub', 'condition', 'position'], as_index=False)['actual_duration_s'].mean()
wide = cell.pivot(index='sub', columns=['condition', 'position'], values='actual_duration_s')
complete = wide.dropna().index
cell = cell.loc[cell['sub'].isin(complete)]
assert len(complete) == 22

aov = pg.rm_anova(data=cell, dv='actual_duration_s', within=['condition', 'position'], subject='sub', detailed=True, effsize='np2')
desc = cell.groupby(['condition', 'position'])['actual_duration_s'].agg(mean='mean', sd='std', n='count').reset_index()
aov.to_csv(OUT / 'actual_duration_denominator_anova.csv', index=False)
desc.to_csv(OUT / 'actual_duration_denominator_descriptives.csv', index=False)
print('Actual block duration: 2 x 2 RM-ANOVA')
print(aov[['Source', 'ddof1', 'ddof2', 'F', 'p_unc', 'np2']].round(6).to_string(index=False))
print('\nCell means')
print(desc.round(6).to_string(index=False))
