"""Portable construction of the explicitly unexecuted shared-prior family."""
import hashlib
import importlib.util
from pathlib import Path


def make_controller(family, candidate_id, count, device):
    if family.get('schema') != 'shared-recovery-prior-proposals-v1':
        raise ValueError('Require the explicit proposal family container')
    directory = Path(__file__).resolve().parent
    for name, digest in family['factory_source_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Proposed controller source changed: ' + name)
    candidates = [row for row in family['candidates'] if row['candidate_id'] == candidate_id]
    if len(candidates) != 1:
        raise ValueError('Require one declared candidate ID')
    path = directory / 'prior_controller.py'
    spec = importlib.util.spec_from_file_location('portable_recovery_prior_controller', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = candidates[0]
    return module.RecoveryPriorController(family['source_public_policy'], family['seed_metadata'],
        family['seed_arrays'], count, device,
        **{key: candidate[key] for key in ['group', 'table', 'alignment', 'gain']})
