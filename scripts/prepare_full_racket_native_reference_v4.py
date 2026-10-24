#!/usr/bin/env python3
"""Embed an actually executed contact/recovery candidate into32 coordinates.

This preserves its26 original parameters and seed exactly. Extra velocity
coordinates reproduce the legacy formula. New rollouts must remeasure fitness.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_arm_search_v4 import extend_arm_velocity_parameters, VELOCITY_PARAMETER_NAMES
from quadsmash.control.full_racket_teacher_v4 import PARAMETER_NAMES
from quadsmash.control.full_racket_contact_curriculum_v4 import usable_air_contact


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('--candidate', type=int, required=True)
    parser.add_argument('--body-run', type=Path, required=True)
    parser.add_argument('--body-generation', type=int, required=True)
    parser.add_argument('--body-candidate', type=int, required=True)
    parser.add_argument('--velocity-bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    source_path = args.source_run / f'population_{args.generation:03d}.pt'
    body_path = args.body_run / f'population_{args.body_generation:03d}.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    body = torch.load(body_path, map_location='cpu', weights_only=True)
    rows_path = args.source_run / f'episodes_{args.generation:03d}.json'
    body_rows_path = args.body_run / f'episodes_{args.body_generation:03d}.json'
    rows = [r for r in json.loads(rows_path.read_text()) if r['candidate_id'] == args.candidate]
    body_rows = [r for r in json.loads(body_rows_path.read_text()) if r['candidate_id'] == args.body_candidate]
    assert len(rows) == len(body_rows) == 2
    assert any(usable_air_contact(r) and r['recovery_success'] for r in rows)
    assert all(r['recovery_success'] and not r['numerical_failures'] for r in body_rows)
    assert source['asset_sha256'] == body['asset_sha256']
    assert source['controller_template']['seed'] == body['controller_template']['seed']
    baseline = copy.deepcopy(source['controller_template'])
    baseline['parameters'] = source['parameters'][args.candidate].clone()
    assert baseline['parameters'].shape == (26,)
    assert baseline['seed'].get('causal_arm_prepose_gain', 0.) == 0.
    baseline.pop('training_result', None)
    provenance = {str(p.resolve().relative_to(ROOT)): sha(p) for p in
                  [source_path, body_path, rows_path, body_rows_path, args.velocity_bank, Path(__file__)]}
    baseline.update(kind='Unexecuted new initialization preserving an actually executed26-parameter contact/recovery candidate.',
                    formal_acceptance=False, native_execution=False, performance_inherited=False,
                    initialization_provenance=provenance)
    augmented = copy.deepcopy(baseline)
    augmented['parameters'] = extend_arm_velocity_parameters(baseline['parameters'], baseline['seed']['joint_velocity_seed_rad_s'])
    augmented['parameter_names'] = PARAMETER_NAMES + ['intercept_plane_x', 'strike_timing_residual'] + VELOCITY_PARAMETER_NAMES
    assert torch.equal(augmented['parameters'][:26], source['parameters'][args.candidate])
    velocity = np.asarray(json.loads(args.velocity_bank.read_text())['joint_velocity_seeds_rad_s'])
    assert velocity.shape == (4, 6) and np.isfinite(velocity).all() and np.abs(velocity).max() <= 5.
    velocity[0] = np.asarray(baseline['seed']['joint_velocity_seed_rad_s'])
    body_bank = dict(seed=baseline['seed'], asset_sha256=baseline['asset_sha256'],
                     parameters=body['parameters'][args.body_candidate].tolist(),
                     source_provenance=provenance,
                     scope='Historical paired body recovery proposal; changed coupled trajectories must be measured again.',
                     native_execution_with_new_controller=False, performance_inherited=False)
    velocity_bank = dict(reference_seed=baseline['seed'], asset_sha256=baseline['asset_sha256'],
                         joint_velocity_seeds_rad_s=velocity.tolist(), source_provenance=provenance,
                         scope='Original direction plus three momentum/pitch proposals; no inherited fitness.',
                         native_execution=False, formal_acceptance=False)
    args.output.mkdir(parents=True)
    torch.save(baseline, args.output / 'legacy26_reference.pt')
    torch.save(augmented, args.output / 'policy.pt')
    (args.output / 'RECOVERY_BANK.json').write_text(json.dumps(body_bank, indent=2) + '\n')
    (args.output / 'VELOCITY_BANK.json').write_text(json.dumps(velocity_bank, indent=2) + '\n')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), source_candidate=args.candidate,
                  source_generation=args.generation, body_candidate=args.body_candidate,
                  source_native_examples=[{k: r[k] for k in ['source_training_index', 'env_id', 'airborne_string_hit',
                      'recovery_success', 'outgoing_velocity_mps', 'first_landing_m', 'complete_smash']} for r in rows],
                  original26_parameters_bitwise_preserved=True, original_seed_preserved=True,
                  causal_prepose_gain=0., extra_velocity_coordinates_embed_legacy_formula=True,
                  native_execution=False, formal_acceptance=False, performance_inherited=False,
                  policy_sha256=sha(args.output / 'policy.pt'), sources=provenance)
    (args.output / 'INITIALIZATION.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
