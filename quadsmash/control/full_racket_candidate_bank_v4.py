"""Validated TRAIN proposals for unchanged controller coordinates and bounds."""
import numpy as np
import torch


def validate_candidate_bank(bank, seed, asset_sha256, names, reference, lower, upper, capacity):
    if bank.get('schema') != 'full-racket-candidate-bank-v4':
        raise ValueError('Unknown candidate bank schema')
    if bank.get('asset_sha256') != asset_sha256 or bank.get('seed') != seed:
        raise ValueError('Candidate bank asset/controller seed mismatch')
    if bank.get('parameter_names') != names:
        raise ValueError('Candidate bank parameter order mismatch')
    reference, lower, upper = [np.asarray(x, dtype=np.float32) for x in (reference, lower, upper)]
    expected = np.asarray(bank.get('reference_parameters'), dtype=np.float32)
    if expected.shape != reference.shape or not np.array_equal(expected, reference):
        raise ValueError('Candidate bank reference differs from warm-start reference')
    rows = np.asarray(bank.get('parameters'), dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != len(names) or not 1 <= len(rows) <= capacity:
        raise ValueError('Candidate bank shape/capacity mismatch')
    if lower.shape != reference.shape or upper.shape != reference.shape or np.any(upper <= lower):
        raise ValueError('Invalid controller search bounds')
    if not all(np.isfinite(x).all() for x in (rows, lower, upper, reference)):
        raise ValueError('Finite candidate parameters and bounds required')
    if np.any(rows < lower) or np.any(rows > upper):
        raise ValueError('Candidate bank exceeds original controller search bounds')
    return rows.copy()


def install_candidate_bank(population, rows, generation):
    """Keep both references; seed generation0only, then let CEM adapt freely."""
    if generation == 0:
        if population.ndim != 2 or rows.ndim != 2 or rows.shape[1] != population.shape[1] or len(rows) > len(population)-2:
            raise ValueError('Candidate bank cannot overwrite reserved references')
        population[2:2+len(rows)] = torch.as_tensor(rows, device=population.device, dtype=population.dtype)
    return population
