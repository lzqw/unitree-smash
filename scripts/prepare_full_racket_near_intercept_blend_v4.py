#!/usr/bin/env python3
"""Prepare nearer causal interception and measured stroke blends, without dynamics."""
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
from quadsmash.control.full_racket_teacher_v4 import LOW, HIGH
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint', 'bank', 'reference-run', 'power-snapshot', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    warm = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    bank = json.loads(args.bank.read_text())
    assert bank['seed'] == warm['seed']
    population_path = args.reference_run/'population_000.pt'
    source = torch.load(population_path, map_location='cpu', weights_only=True)
    assert source['asset_sha256'] == warm['asset_sha256']
    # Preserve the best native contact motion and the actually recovered frame
    # configuration. Their outcomes never transfer to this changed family.
    selected = int(np.argmax([min(r['training_score'] for r in source['rows'] if r['candidate_id'] == i)
        if any(r['airborne_string_hit'] and not r['robot_net_fault'] for r in source['rows'] if r['candidate_id'] == i)
        else -np.inf for i in range(len(source['parameters']))]))
    air = source['parameters'][selected].numpy().copy()
    recovered = [r for r in source['rows'] if r['recovery_success']]
    frame_row = max(recovered, key=lambda r: (r['had_contact'], r['jump']['recovery_s']))
    recovery = source['parameters'][frame_row['candidate_id']].numpy().copy()
    contact_path = args.power_snapshot/'contacts.partial.json'
    state_path = args.power_snapshot/'partial_search_state.pt'
    contacts = [c for c in json.loads(contact_path.read_text())
        if c['episode_index'] == 0 and c['env_id'] >= 4]
    power = max(contacts, key=lambda c: c['point_velocity_mps'][0])
    assert power['point_velocity_mps'][0] > .5
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    count = state['parameters'].shape[0]
    assert count == 64 and state['parameters'].shape[1] == 32
    powerful = state['parameters'][power['env_id']//2].numpy()
    fast_rate = powerful[26:32]*powerful[7]
    old_rate = air[26:32]*air[7]
    assert np.max(abs(fast_rate)) <= 5. and np.max(abs(old_rate)) <= 5.
    low = np.array(LOW+[.8, -.12]+[-5.]*6, dtype=np.float32)
    high = np.array(HIGH+[1.45, .12]+[5.]*6, dtype=np.float32)
    low[22:24], high[22:24] = -.3, .3
    setups = [(air, 1.55, 1.05), (air, 1.58, 1.15), (air, 1.62, 1.25),
              (recovery, 1.55, 1.05), (recovery, 1.58, 1.15)]
    proposals, labels = [], []
    for blend in [0., 1/3, 2/3, 1.]:
        rate = (1-blend)*old_rate+blend*fast_rate
        for lead in [0., .04, .08]:
            for setup, recovery_time, plane in setups:
                candidate = setup.copy()
                candidate[3], candidate[7], candidate[14] = recovery_time, 1., lead
                candidate[24], candidate[25] = plane, 0.
                candidate[26:32] = rate
                candidate = np.clip(candidate, low, high).astype(np.float32)
                proposals.append(candidate)
                labels.append(dict(kind='nearer_interception_measured_stroke_blend',
                    measured_fast_rate_fraction=blend, arm_lead_s=lead, recovery_local_time=recovery_time,
                    interception_plane_x_m=plane, body_reference=('contact' if setup is air else 'recovery'),
                    performance_inherited=False))
    assert len(proposals) == len({tuple(row) for row in proposals}) == 60
    reference = air.copy()
    reference[7], reference[14], reference[24], reference[25] = 1., .04, 1.15, 0.
    reference[26:32] = old_rate
    warm = copy.deepcopy(warm)
    velocity = list(warm['seed']['desired_point_velocity_mps'])
    velocity[0] = 1.5
    warm['seed'] = dict(warm['seed'], desired_point_velocity_mps=velocity)
    warm['parameters'] = torch.tensor(reference)
    warm['event_calibration_s'] = 0.
    warm.pop('training_result', None)
    warm.update(native_execution=False, performance_inherited=False, formal_acceptance=False,
        kind='Unexecuted nearer causal contact/positive arm-lead/stroke-blend proposals. Original task gates unchanged.',
        initial_timing_scope='Predicted current-ball arrival commands body event; arm-lead parameters separately explore actuator lag. Zero calibration is a new initialization assumption.')
    bank = copy.deepcopy(bank)
    bank.update(seed=warm['seed'], reference_parameters=reference.tolist(), parameters=np.stack(proposals).tolist(),
        proposal_labels=labels, controller_revision='nearer-causal-interception-stroke-blend-v1',
        native_execution=False, performance_inherited=False)
    rows = validate_candidate_bank(bank, warm['seed'], warm['asset_sha256'], warm['parameter_names'], reference, low, high, 62)
    assert rows.shape == (60, 32) and np.max(abs(rows[:, 26:32]*rows[:, 7:8])) <= 5.
    paths = [args.checkpoint, args.bank, population_path, contact_path, state_path, Path(__file__)]
    provenance = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    warm['initialization_provenance'] = provenance
    bank['near_intercept_blend_provenance'] = provenance
    args.output.mkdir(parents=True)
    torch.save(warm, args.output/'policy.pt')
    (args.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank, indent=2)+'\n')
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU-only new command family, preserving measured contact/recovery bodies and blending old rates with an actually forward-moving frame-hit stroke. No outcome inherited.',
        source_contact_candidate=selected, source_recovery_candidate=frame_row['candidate_id'],
        source_fast_frame_env=power['env_id'], source_fast_event_is_strings=power['string_contact'],
        source_fast_actual_point_velocity_mps=power['point_velocity_mps'],
        old_effective_joint_rates_rad_s=old_rate.tolist(), measured_forward_stroke_rate_seed_rad_s=fast_rate.tolist(),
        stroke_blend_fractions=[0., 1/3, 2/3, 1.], arm_leads_s=[0., .04, .08],
        interception_planes_x_m=[1.05, 1.15, 1.25], command_forward_point_velocity_mps=1.5,
        body_event_calibration_s=0., unchanged_ball_attack_speed_gate_mps=8.,
        all_physical_assets_hardware_rules_and_success_gates_unchanged=True,
        native_execution=False, formal_acceptance=False,
        policy_sha256=hashlib.sha256((args.output/'policy.pt').read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256((args.output/'CANDIDATE_BANK.json').read_bytes()).hexdigest(), sources=provenance)
    (args.output/'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
