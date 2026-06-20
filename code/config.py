"""
Attentional Thief — open data configuration.
All analysis scripts import ROOT, DATA_DIR, and OUT_DIR from here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # open_data/
DATA_DIR = ROOT / 'data'
OUT_DIR = ROOT / 'output'

# Directories within data/
BEH_DIR = DATA_DIR / 'behavioral'
ERP_DIR = DATA_DIR / 'erp_summaries'
FRP_DIR = DATA_DIR / 'frp_summaries'
ET_DIR = DATA_DIR / 'et_summaries'
GROUP_DIR = DATA_DIR / 'group'

ANALYSIS_SUBJECTS = (
    '001', '002', '003', '004', '005', '006', '007', '008', '009', '010',
    '011', '012', '013', '014', '017', '018', '019', '020', '021', '022',
    '024', '025', '026',
)

COLORS = {
    'active':   '#111111',
    'passive':  '#d62728',
    'constant': '#1f77b4',
}

COND_LABELS = {'active': 'Scrolling', 'passive': 'Watching', 'constant': 'Baseline'}

CLUSTERS = {
    'occipital':  ['O1', 'Oz', 'O2'],
    'posterior':  ['PO3', 'PO4', 'POz', 'PO7', 'PO8'],
    'parietal':   ['Pz', 'CPz', 'P3', 'P4'],
    'frontal':    ['F3', 'Fz', 'F4'],
}

ERP_WINDOWS = {
    'P1':   (0.080, 0.130),
    'N1':   (0.140, 0.200),
    'P2':   (0.200, 0.300),
    'P3':   (0.300, 0.500),
    'Late': (0.500, 1.000),
}

FRP_WINDOWS = {
    'lambda':   (0.060, 0.120),
    'N1_frp':   (0.120, 0.200),
    'P2_frp':   (0.200, 0.350),
    'Late_frp': (0.350, 0.600),
}


def make_out(*parts: str) -> Path:
    """Create and return an output subdirectory."""
    d = OUT_DIR.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
