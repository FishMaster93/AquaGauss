"""All paths used by the code, relative to the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'data' / 'manifest_video.csv'
PRETRAINED = ROOT / 'pretrained'
WORKSPACE = ROOT / 'workspace'      # encoder checkpoints and logs
FEAT_ROOT = ROOT / 'features'       # frozen encoder features
RESULTS = ROOT / 'results'
