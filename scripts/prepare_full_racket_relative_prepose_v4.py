#!/usr/bin/env python3
"""Calibrate optional causal preparation from one actually executed TRAIN chain."""
import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.control.full_racket_relative_prepose_v4 import public_target_body, validate_relative_prepose


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--population', type=Path, required=True)
    parser.add_argument('--reference-env-id', type=int, required=True)
    parser.add_argument('--gain', type=float, required=True)
    parser.add_argument('--candidate-bank', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    if not np.isfinite(args.gain) or not 0. <= args.gain <= 1.:
        raise ValueError('Preparation gain must be in[0,1]')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    population = torch.load(args.population, map_location='cpu', weights_only=False)
    if not source.get('native_execution') or population['schema'] != 'full-racket-ball-search-state-v4':
        raise ValueError('Reference must come from an actually executed TRAIN policy and population')
    if source['seed'].get('causal_arm_prepose_gain', 0.) != 0.:
        raise ValueError('Calibrate a reference without an existing causal preparation correction')
    case_count = len(population['training_case_indices'])
    env = args.reference_env_id
    if not 0 <= env < population['candidate_count']*case_count:
        raise ValueError('Reference environment is not in this population')
    candidate = env//case_count
    if not torch.equal(source['parameters'], population['parameters'][candidate]):
        raise ValueError('Reference policy differs from the actual candidate parameters')
    if source['seed'] != population['controller_template']['seed']:
        raise ValueError('Reference policy and population use different controller seeds')
    if source['asset_sha256'] != population['asset_sha256']:
        raise ValueError('Reference policy belongs to another physical model')
    row = population['rows'][env]
    if row['env_id'] != env or not row['airborne_string_hit'] or not row['recovery_success']:
        raise ValueError('Reference must retain an actual same-episode airborne strings/recovery chain')
    suffix = args.population.stem.rsplit('_', 1)[1]
    plans_path = args.population.with_name('causal_plans_'+suffix+'.json')
    episodes_path = args.population.with_name('episodes_'+suffix+'.json')
    if json.dumps(json.loads(episodes_path.read_text()), sort_keys=True) != json.dumps(population['rows'], sort_keys=True):
        raise ValueError('Population and separately recorded native episodes disagree')
    groups = [g for g in json.loads(plans_path.read_text()) if env in g['env_ids']]
    if len(groups) != 1:
        raise ValueError('Need exactly one recorded causal plan for this first episode')
    plan = groups[0]
    index = plan['env_ids'].index(env)
    public = np.array(plan['used_public_observations'][index:index+1])
    forecast = np.array(plan['position_m'][index:index+1])
    if not plan['valid'][index] or not 2.-1e-5 <= public[0, 775] <= 2.01+1e-5:
        raise ValueError('Reference requires a valid causal forecast at the initial planning observation')
    reference, ready = public_target_body(forecast, public)
    if not ready[0]:
        raise ValueError('Invalid public body axes or reference command')
    calibration = validate_relative_prepose(dict(frame='robot_root_body',
        reference_target_body_m=reference[0].tolist(), reference_capture_clock_s=float(public[0, 775]),
        reference_intercept_plane_x_m=float(source['parameters'][24]),
        provenance=dict(native_policy_sha256=digest(args.checkpoint),
                        native_population_sha256=digest(args.population),
                        native_causal_plans_sha256=digest(plans_path), reference_env_id=env,
                        note='Offline TRAIN calibration only. Runtime reads no env/case ID or future state.')))
    warm = copy.deepcopy(source)
    warm['seed'] = dict(warm['seed'], relative_contact_prepose=calibration,
                        causal_arm_prepose_gain=args.gain)
    warm.pop('training_result', None)
    warm.update(native_execution=False, performance_inherited=False, formal_acceptance=False,
                kind='Unexecuted public-forecast preparation relative to a measured TRAIN reference; requires fresh native evaluation.')
    bank = None
    if args.candidate_bank:
        bank = json.loads(args.candidate_bank.read_text())
        if (bank['asset_sha256'] != source['asset_sha256'] or bank['seed'] != source['seed']
                or not np.array_equal(np.array(bank['reference_parameters'], dtype=np.float32),
                                      source['parameters'].numpy())):
            raise ValueError('Candidate bank differs from the source policy')
        bank.update(seed=warm['seed'], native_execution=False, performance_inherited=False)
    args.output.mkdir(parents=True)
    torch.save(warm, args.output/'policy.pt')
    if bank is not None:
        (args.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank, indent=2)+'\n')
    sources = [args.checkpoint, args.population, plans_path, episodes_path, Path(__file__),
               ROOT/'quadsmash/control/full_racket_relative_prepose_v4.py',
               ROOT/'quadsmash/control/full_racket_rally_controller_v4.py']
    if args.candidate_bank:
        sources.append(args.candidate_bank)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), gain=args.gain,
                  scope='CPU-prepared optional causal controller calibration only; no source performance is inherited.',
                  reference=calibration, original_parameters_preserved=True,
                  native_execution=False, performance_inherited=False, formal_acceptance=False,
                  source_sha256={str(p.resolve()): digest(p) for p in sources},
                  policy_sha256=digest(args.output/'policy.pt'))
    if bank is not None:
        report['bank_sha256'] = digest(args.output/'CANDIDATE_BANK.json')
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
