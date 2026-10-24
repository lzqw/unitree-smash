#!/usr/bin/env python3
"""Preserve a measured two-case contact motion and search recovery/face commands."""
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
    for name in ['checkpoint', 'bank', 'contact-run', 'recovery-run', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    warm = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    bank = json.loads(args.bank.read_text())
    assert bank['seed'] == warm['seed'] and warm['seed'].get('measured_face_feedback') is not None
    population_path = args.contact_run/'population_000.pt'
    source = torch.load(population_path, map_location='cpu', weights_only=True)
    assert source['asset_sha256'] == warm['asset_sha256']
    groups = [[row for row in source['rows'] if row['candidate_id'] == index]
              for index in range(len(source['parameters']))]
    rank = [(sum(usable_air_contact(row) for row in rows), min(row['training_score'] for row in rows))
            for rows in groups]
    selected = max(range(len(groups)), key=lambda index: rank[index])
    assert len(groups[selected]) == 2 and rank[selected][0] == 2
    reference = source['parameters'][selected].numpy().copy()
    assert np.array_equal(reference[26:32], warm['parameters'].numpy()[26:32])
    low = np.array(LOW+[.8, -.12]+[-5.]*6, dtype=np.float32)
    high = np.array(HIGH+[1.45, .12]+[5.]*6, dtype=np.float32)
    low[22:24], high[22:24] = -.3, .3
    proposals, labels = [], []
    for recovery in [1.55, 1.58, 1.62, 1.70]:
        for gain in [.2, .6, 1., 1.4]:
            for lead_delta in [-.015, 0., .015]:
                candidate = reference.copy()
                candidate[3], candidate[15] = recovery, gain
                candidate[14] = np.clip(reference[14]+lead_delta, low[14], high[14])
                proposals.append(candidate)
                labels.append(dict(kind='measured_contact_recovery_face_timing',
                    recovery_local_time=recovery, face_and_pitch_gain=gain, arm_lead_delta_s=lead_delta,
                    performance_inherited=False))
    recovery_path = args.recovery_run/'population_000.pt'
    recovered = torch.load(recovery_path, map_location='cpu', weights_only=True)
    recoveries = [row for row in recovered['rows'] if row['recovery_success']]
    assert len(recoveries) == 6 and recovered['asset_sha256'] == warm['asset_sha256']
    for row in recoveries:
        for gain in [.8, 1.4]:
            candidate = recovered['parameters'][row['candidate_id']].numpy().copy()
            assert np.array_equal(candidate[26:32], reference[26:32])
            candidate[15] = gain
            proposals.append(candidate)
            labels.append(dict(kind='measured_missed_ball_recovery_new_face_mode',
                source_env_id=row['env_id'], source_candidate_id=row['candidate_id'],
                face_and_pitch_gain=gain, performance_inherited=False))
    assert len(proposals) == len({tuple(row) for row in proposals}) == 60
    warm = copy.deepcopy(warm)
    warm['parameters'] = torch.tensor(reference)
    warm.pop('training_result', None)
    warm.update(native_execution=False, performance_inherited=False, formal_acceptance=False,
        kind='Unexecuted final-face projection plus recovery/arm timing around an actual two-case contact motion.')
    bank = copy.deepcopy(bank)
    bank.update(reference_parameters=reference.tolist(), parameters=np.stack(proposals).tolist(),
        proposal_labels=labels, controller_revision='contact-recovery-final-face-projection-v4',
        native_execution=False, performance_inherited=False)
    checked = validate_candidate_bank(bank, warm['seed'], warm['asset_sha256'], warm['parameter_names'],
                                      reference, low, high, 62)
    assert checked.shape == (60, 32)
    sources = [args.checkpoint, args.bank, population_path, recovery_path, Path(__file__)]
    provenance = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    warm['initialization_provenance'] = provenance
    bank['measured_contact_recovery_provenance'] = provenance
    args.output.mkdir(parents=True)
    torch.save(warm, args.output/'policy.pt')
    (args.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank, indent=2)+'\n')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU-only new motion proposals. The old candidate contacted both TRAIN cases; its outcomes do not transfer to the changed controller.',
        source_contact_candidate=selected, source_contact_training_cases=[r['source_training_index'] for r in groups[selected]],
        source_contacts_without_robot_rule_faults=2,
        source_contact_recovery_successes=sum(r['recovery_success'] for r in groups[selected]),
        source_contact_complete_smashes=sum(r['complete_smash'] for r in groups[selected]),
        recovery_local_times=[1.55, 1.58, 1.62, 1.70], face_and_pitch_gains=[.2, .6, 1., 1.4],
        arm_lead_deltas_s=[-.015, 0., .015], proposal_groups=dict(contact_motion_recovery_face_timing=48, prior_missed_ball_recovery=12),
        all_six_reference_velocity_directions_preserved=True, original_search_bounds_and_physics_unchanged=True,
        native_execution=False, performance_inherited=False, formal_acceptance=False,
        policy_sha256=hashlib.sha256((args.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((args.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest(), sources=provenance)
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
