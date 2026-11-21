"""Frozen stage-end research panel, evaluated in four original-size native batches."""
import gzip
import hashlib
import json
from pathlib import Path


FILE = Path(__file__).resolve().parents[2]/'reports/supermimic_research_scale_20261121/independent512.json.gz'
SHA256 = '077de9a4b35e23f7d07ee58d1cc055e1edf05d5d6eadf8f5bce2446221a042ee'


def load_panel():
    data = FILE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError('Independent panel changed since declaration')
    return json.loads(gzip.decompress(data))


def panel_batch(index, protocol_sha256, seed):
    panel = load_panel()
    if (index not in range(4) or panel['protocol_sha256'] != protocol_sha256
            or panel['native_reset_seed'] != seed):
        raise ValueError('Independent panel batch, reset seed or scoring protocol mismatch')
    cases = panel['cases'][128*index:128*(index+1)]
    identity = dict(panel_sha256=SHA256, batch=index, case_count=len(cases), total_count=512)
    return cases, identity
