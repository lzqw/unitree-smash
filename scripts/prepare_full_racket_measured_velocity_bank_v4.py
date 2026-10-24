#!/usr/bin/env python3
"""Prepare measured-state arm-rate and recovery proposals; no outcome transfer."""
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
    for name in ['checkpoint', 'bank', 'contact-run', 'velocity-report', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    warm = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    bank = json.loads(args.bank.read_text())
    assert bank['seed'] == warm['seed']
    source_path = args.contact_run/'population_000.pt'
    source = torch.load(source_path, map_location='cpu', weights_only=True)
    assert source['asset_sha256'] == warm['asset_sha256']
    groups = [[r for r in source['rows'] if r['candidate_id'] == i] for i in range(len(source['parameters']))]
    ranks = [(sum(usable_air_contact(r) for r in group), min(r['training_score'] for r in group)) for group in groups]
    selected = max(range(len(groups)), key=lambda i: ranks[i])
    assert ranks[selected][0] >= 1
    reference = source['parameters'][selected].numpy().copy()
    rates = json.loads(args.velocity_report.read_text())
    assert rates['maximum_point_position_error_m'] < .0005
    assert rates['maximum_point_velocity_reconstruction_error_mps'] < .005
    assert rates['maximum_normal_rate_reconstruction_error_per_s'] < .005
    seeds = np.array([p['joint_velocity_seed_rad_s'] for p in rates['selected_joint_rate_proposals']])
    assert seeds.shape == (4, 6) and np.isfinite(seeds).all() and np.max(abs(seeds)) <= 5.000001
    low = np.array(LOW+[.8, -.12]+[-5.]*6, dtype=np.float32)
    high = np.array(HIGH+[1.45, .12]+[5.]*6, dtype=np.float32)
    low[22:24], high[22:24] = -.3, .3
    recovered = [r for r in source['rows'] if r['recovery_success']]
    assert recovered, 'Preserve an actually measured recovery configuration'
    recovery_row = max(recovered, key=lambda r: (r['had_contact'], r['jump']['recovery_s']))
    recovery_reference = source['parameters'][recovery_row['candidate_id']].numpy().copy()
    body_variants = []
    for recovery in [1.55, 1.58, 1.62]:
        body = reference.copy()
        body[3] = recovery
        body_variants.append((body, 'selected_air_contact_reference'))
    for offset in [0., .03]:
        body = recovery_reference.copy()
        body[3] = np.clip(body[3]+offset, low[3], high[3])
        body_variants.append((body, 'measured_contact_recovery_configuration'))
    proposals, labels = [], []
    for index, velocity in enumerate(seeds):
        for lead_delta in [-.025, 0., .025]:
            for body, body_kind in body_variants:
                proposal = body.copy()
                proposal[7] = 1. # The proposed seed itself obeys the5rad/s bound.
                proposal[26:32] = velocity
                proposal[14] = body[14]+lead_delta
                proposal = np.clip(proposal, low, high).astype(np.float32)
                proposals.append(proposal)
                labels.append(dict(kind='measured_contact_velocity_and_recovery',
                    rate_proposal=index, source_velocity_env_id=rates['selected_joint_rate_proposals'][index]['source_env_id'],
                    arm_lead_delta_s=lead_delta, recovery_local_time=float(body[3]),
                    body_configuration_source=body_kind, performance_inherited=False))
    assert len(proposals) == len({tuple(row) for row in proposals}) == 60
    warm = copy.deepcopy(warm)
    warm['parameters'] = torch.tensor(reference)
    warm.pop('training_result', None)
    warm.update(native_execution=False, performance_inherited=False, formal_acceptance=False,
        kind='Unexecuted measured-contact arm-velocity and recovery proposals; native fitness must be measured again.')
    bank = copy.deepcopy(bank)
    bank.update(reference_parameters=reference.tolist(), parameters=np.stack(proposals).tolist(),
        proposal_labels=labels, controller_revision='measured-contact-velocity-recovery-v1',
        native_execution=False, performance_inherited=False,
        scope='Four new bounded arm-velocity directions from actual TRAIN states, with arm/recovery timing proposals. No actual outcome inherited.')
    rows = validate_candidate_bank(bank, warm['seed'], warm['asset_sha256'], warm['parameter_names'], reference, low, high, 62)
    assert rows.shape == (60, 32) and np.max(abs(rows[:, 26:32]*rows[:, 7:8])) <= 5.
    sources = [args.checkpoint, args.bank, source_path, args.velocity_report, Path(__file__)]
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    warm['initialization_provenance'] = provenance
    bank['measured_velocity_provenance'] = provenance
    args.output.mkdir(parents=True)
    torch.save(warm, args.output/'policy.pt')
    (args.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank, indent=2)+'\n')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU-only proposal preparation. Instantaneous kinematic rate objectives do not prove torque/position tracking, real collision, attack or recovery feasibility.',
        selected_native_reference_candidate=selected,
        source_usable_air_contacts_in_two_cases=ranks[selected][0],
        source_reference_recovery_successes=sum(r['recovery_success'] for r in groups[selected]),
        source_reference_complete_smashes=sum(r['complete_smash'] for r in groups[selected]),
        source_recovery_configuration_candidate=recovery_row['candidate_id'],
        source_recovery_configuration_env=recovery_row['env_id'],
        source_recovery_configuration_had_contact=recovery_row['had_contact'],
        source_recovery_configuration_frame_hit=recovery_row['frame_hit'],
        source_recovery_duration_s=recovery_row['jump']['recovery_s'],
        reference_all32parameters_retained_from_selected_native_candidate=True,
        proposed_velocity_seeds_rad_s=seeds.tolist(), proposal_count=60, proposal_effective_joint_rate_limit_rad_s=5.,
        original_physical_assets_hardware_limits_rules_success_gates_unchanged=True,
        native_execution=False, performance_inherited=False, formal_acceptance=False,
        policy_sha256=hashlib.sha256((args.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((args.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest(), sources=provenance)
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
