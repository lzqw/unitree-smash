#!/usr/bin/env python3
"""Retain an actual attacking stroke and vary previously recovered body motions."""
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
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank
from quadsmash.control.full_racket_contact_curriculum_v4 import usable_air_contact
from quadsmash.control.full_racket_teacher_v4 import LOW, HIGH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['attack-run', 'attack-snapshot', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--recovery-runs', type=Path, nargs=2, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    rows_path = args.attack_snapshot / 'episodes.partial.json'
    state_path = args.attack_snapshot / 'partial_search_state.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    rows = json.loads(rows_path.read_text())
    attacks = [r for r in rows if usable_air_contact(r) and r['released'] and r['crossed_net']
        and np.isfinite(r['outgoing_velocity_mps']).all()
        and np.linalg.norm(r['outgoing_velocity_mps']) >= 8
        and r['outgoing_velocity_mps'][0] > 0 and r['outgoing_velocity_mps'][2] <= -1]
    attack = max(attacks, key=lambda r: np.linalg.norm(r['outgoing_velocity_mps']))
    generation = int(state['generation'])
    start_path = args.attack_run / f'generation_start_{generation:03d}.pt'
    start = torch.load(start_path, map_location='cpu', weights_only=True)
    assert start['training_case_indices'] == [1, 65]
    candidate = attack['env_id'] // 2
    reference = state['parameters'][candidate].numpy().copy()
    assert reference.shape == (32,) and not attack['complete_smash']
    warm = copy.deepcopy(start['controller_template'])
    recovery, sources = [], [rows_path, state_path, start_path]
    for run in args.recovery_runs:
        recovered = []
        for path in sorted(run.glob('episodes_*.json')):
            if not path.stem.rsplit('_', 1)[1].isdigit():
                continue
            recorded = json.loads(path.read_text())
            for row in recorded:
                if usable_air_contact(row) and row['recovery_success']:
                    recovered.append((row['jump']['recovery_s'], path, row))
        _, path, row = max(recovered, key=lambda item: item[0])
        population_path = path.with_name('population_' + path.stem.rsplit('_', 1)[1] + '.pt')
        population = torch.load(population_path, map_location='cpu', weights_only=True)
        assert population['asset_sha256'] == start['asset_sha256']
        assert population['controller_template']['seed']['joint_position_rad'] == warm['seed']['joint_position_rad']
        recovery.append(dict(run=run.name, env_id=row['env_id'], candidate_id=row['candidate_id'],
            actual_recovery_s=row['jump']['recovery_s'], actual_airborne_strings=True,
            actual_crossed_net=row['crossed_net'], actual_complete_smash=row['complete_smash'],
            parameters=population['parameters'][row['candidate_id']].numpy()))
        sources += [path, population_path]
    # All attacking arm, face, interception and swing parameters stay fixed in
    # the bank. Only measured body coordinates and recovery start time vary.
    body_indices = [0, 1, 2, 4, 5, 6, 17, 20, 21, 22, 23]
    low = np.array(LOW + [.8, -.12] + [-5.] * 6, dtype=np.float32)
    high = np.array(HIGH + [1.45, .12] + [5.] * 6, dtype=np.float32)
    low[22:24], high[22:24] = -.3, .3
    proposals, labels = [], []
    for source in recovery:
        for fraction in [.2, .4, .6, .8, 1.]:
            for recovery_time in [1.55, 1.58, 1.61, 1.64, 1.67, 1.70]:
                proposal = reference.copy()
                proposal[body_indices] = ((1-fraction) * reference[body_indices]
                                         + fraction * source['parameters'][body_indices])
                proposal[3] = recovery_time
                proposals.append(np.clip(proposal, low, high).astype(np.float32))
                labels.append(dict(kind='actual_attack_preserved_recovered_body_blend',
                    body_source=source['run'], recovered_body_fraction=fraction,
                    recovery_local_time_s=recovery_time, performance_inherited=False))
    proposals = np.stack(proposals)
    assert proposals.shape == (60, 32) and len({tuple(p) for p in proposals}) == 60
    fixed = [i for i in range(32) if i not in [3, *body_indices]]
    np.testing.assert_array_equal(proposals[:, fixed], np.tile(reference[fixed], (60, 1)))
    assert np.max(abs(proposals[:, 26:32] * proposals[:, 7:8])) <= 5.
    warm.pop('training_result', None)
    warm.update(parameters=torch.tensor(reference), native_execution=False,
        performance_inherited=False, formal_acceptance=False,
        kind='Next TRAIN search reference is exact actual CEM16attacking candidate; new recovered-body blends are unexecuted and all fitness must be remeasured.')
    bank = dict(schema='full-racket-candidate-bank-v4', seed=warm['seed'],
        asset_sha256=start['asset_sha256'], parameter_names=start['parameter_names'],
        reference_parameters=reference.tolist(), parameters=proposals.tolist(), proposal_labels=labels,
        native_execution=False, performance_inherited=False,
        controller_revision='actual-attack-retained-recovered-body-blends-v1')
    validate_candidate_bank(bank, warm['seed'], start['asset_sha256'], start['parameter_names'], reference, low, high, 62)
    sources += [Path(__file__), ROOT / 'quadsmash/control/full_racket_contact_curriculum_v4.py']
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    warm['initialization_provenance'] = provenance
    bank['initialization_provenance'] = provenance
    args.output.mkdir(parents=True)
    torch.save(warm, args.output / 'policy.pt')
    (args.output / 'CANDIDATE_BANK.json').write_text(json.dumps(bank, indent=2) + '\n')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='Unexecuted next TRAIN proposals. Retain all32actual attacking-reference parameters; bank varies11body coordinates and recovery timing using two prior same-episode airborne-string/recovery motions. Coupled performance does not transfer.',
        actual_attack_source=dict(run=args.attack_run.name, generation=generation, candidate_id=candidate,
            env_id=attack['env_id'], outgoing_velocity_mps=attack['outgoing_velocity_mps'],
            outgoing_speed_mps=float(np.linalg.norm(attack['outgoing_velocity_mps'])),
            recovery_success=attack['recovery_success'], complete_smash=attack['complete_smash']),
        body_sources=[{k: v for k, v in source.items() if k != 'parameters'} for source in recovery],
        body_blend_fractions=[.2, .4, .6, .8, 1.], recovery_local_times_s=[1.55, 1.58, 1.61, 1.64, 1.67, 1.70],
        reference_all32_parameters_preserved_exactly=True, bank_fixed_coordinates=fixed,
        all_proposals_keep_actual_attack_arm_face_interception_and_velocity_parameters=True,
        all_physical_assets_hardware_rules_success_gates_unchanged=True,
        recommended_training_flag='--retain-contact-recovery',
        policy_sha256=hashlib.sha256((args.output / 'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((args.output / 'CANDIDATE_BANK.json').read_bytes()).hexdigest(),
        native_execution=False, formal_acceptance=False, source_sha256=provenance)
    (args.output / 'REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()
