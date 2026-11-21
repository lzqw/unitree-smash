"""Frozen stage-end research panel, evaluated in four original-size native batches."""
import gzip
import hashlib
import json
from pathlib import Path


FILE = Path(__file__).resolve().parents[2]/'reports/supermimic_research_scale_20261121/independent512.json.gz'
SHA256 = '077de9a4b35e23f7d07ee58d1cc055e1edf05d5d6eadf8f5bce2446221a042ee'


FIXED128_SHA256 = '0e2c5cf34a165ec7aa6b68d73bbc2f7d27b38a98aad4b3f4b73cfc066365df0d'


def _read(path, sha):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError('Independent panel changed since declaration')
    return json.loads(gzip.decompress(data))


def load_panel():
    return _read(FILE, SHA256)


def fixed128_cases(protocol_sha256, seed):
    panel = _read(FILE.with_name('fixed128.json.gz'), FIXED128_SHA256)
    if panel['protocol_sha256'] != protocol_sha256 or panel['native_reset_seed'] != seed:
        raise ValueError('Fixed128 reset seed or scoring protocol mismatch')
    return panel['cases'], dict(panel_sha256=FIXED128_SHA256, case_count=128)


def panel_batch(index, protocol_sha256, seed):
    panel = load_panel()
    if (index not in range(4) or panel['protocol_sha256'] != protocol_sha256
            or panel['native_reset_seed'] != seed):
        raise ValueError('Independent panel batch, reset seed or scoring protocol mismatch')
    cases = panel['cases'][128*index:128*(index+1)]
    identity = dict(panel_sha256=SHA256, batch=index, case_count=len(cases), total_count=512)
    return cases, identity
