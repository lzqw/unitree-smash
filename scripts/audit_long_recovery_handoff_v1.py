"""Replay optimized full18 recovery, then feedback through original episode end.

CPU diagnostic only. The candidate file supplies commands, never an injected
terminal state. Controller memory advances on every predicted control tick.
This does not certify native execution or produce expert demonstration labels.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_joint_sequence_proposal_v1 import plain
from scripts.pure_control_moving_foot_recovery_v7 import PureControlMovingFootRecoveryV7
from scripts.recovery_reference_library_v1 import RecoveryReferenceLibraryV1
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.authored_net_clearance_v1 import AuthoredNetClearanceV1, CourtCollisionPathConstraintV1
from scripts.foot_separation_path_constraint_v1 import FootSeparationPathConstraintV1
from quadsmash.physics.court_v4 import load_court


def evaluate(configuration, nominal, candidate, bank, deadline):
    controller = PureControlMovingFootRecoveryV7(PositionPDInterface(**configuration['interface']), nominal)
    library = RecoveryReferenceLibraryV1(configuration['model_reference_library'], controller.model)
    source = next(r for r in library.records if r['source_key'] == candidate['source_key'])
    state = library.state(source, -1)
    geometry = FootSeparationPathConstraintV1(
        CourtCollisionPathConstraintV1(controller.geometry.ground,
            AuthoredNetClearanceV1(controller.geometry.ground, load_court())),
        controller.geometry.spheres, margin_m=0.)
    planned = candidate['fine']['trajectory']
    if not candidate['fine']['model_path_complete'] or len(planned) != 160:
        raise ValueError('Need a complete fine-model 1.6-second candidate')
    required = 800 - int(bank[source['source_case']]['control_tick'])
    rows = []
    failure = None
    settling = stable = longest = 0.
    for k in range(required):
        if time.monotonic() >= deadline:
            failure = dict(step=k, reason='compute_budget')
            break
        rotation = state.root_rotation_world
        velocity = rotation @ state.v[:3]
        angles = np.array([np.arctan2(rotation[2, 1], rotation[2, 2]),
                           np.arcsin(np.clip(-rotation[2, 0], -1, 1))])
        near_settled = (state.contacts.any() and max(abs(angles)) < np.deg2rad(30)
                        and abs(velocity[2]) < .5 and np.linalg.norm(state.v[3:6]) < 2.)
        settling = settling + .01 if near_settled else 0.
        phase = 'AIR_PREPARE' if not state.contacts.any() else (
            'SUSTAIN' if settling >= .15 else 'BUFFER_LOAD')
        feedback, diagnostic = controller.command(state, phase, 100. + k * .01)
        target = np.asarray(planned[k]['target']) if k < len(planned) else feedback
        source_name = 'optimized_full18_command' if k < len(planned) else diagnostic['source']
        try:
            prediction = predict_self_contact_spheres(controller.model, controller.interface, state,
                target, controller.geometry.spheres, geometry=geometry, substeps=160)
        except (ValueError, np.linalg.LinAlgError) as error:
            failure = dict(step=k, reason='model_rejected', error=repr(error), command_source=source_name)
            break
        violations = {n: v for n, v in prediction['diagnostics'].items() if
            (n == 'nonfoot_ground_violation_m' and v > 1e-6) or
            (n in ['joint_position_violation_rad', 'joint_velocity_violation_rad_s',
                   'motor_torque_violation_nm'] and v > 3e-4)}
        if violations:
            failure = dict(step=k, reason='physical_path_violation', violations=violations,
                           collision=prediction['first_ground_violation'], command_source=source_name)
            break
        state = prediction['state']
        state.previous_command_target = target.copy()
        rotation = state.root_rotation_world
        velocity = rotation @ state.v[:3]
        angles = np.array([np.arctan2(rotation[2, 1], rotation[2, 2]),
                           np.arcsin(np.clip(-rotation[2, 0], -1, 1))])
        stable_now = (state.contacts.any() and max(abs(angles)) <= np.deg2rad(15)
            and np.linalg.norm(velocity[:2]) <= .3 and abs(velocity[2]) <= .1
            and np.linalg.norm(state.v[3:6]) <= 1. and max(abs(state.joint_velocity)) <= 1.)
        stable = stable + .01 if stable_now else 0.
        longest = max(longest, stable)
        reference_error = float(np.max(abs(state.q - np.asarray(planned[k]['q'])))) if k < len(planned) else None
        rows.append(dict(step=k, phase=phase, command_source=source_name,
            solver_accepted=diagnostic['solver_accepted'], q=state.q.copy(), v=state.v.copy(),
            target=target.copy(), contacts=state.contacts.copy(), stable_s=stable,
            planned_replay_q_max_abs=reference_error))
    return dict(source_case=source['source_case'], source_key=source['source_key'],
        steps=len(rows), required_steps=required, failure=failure, trajectory=rows,
        final_model_stable_s=stable, longest_model_stable_s=longest,
        model_episode_recovery=failure is None and len(rows) == required and stable >= .8,
        native_success=False, valid_expert_labels=False,
        phase_uses_predicted_impulse_contacts=True, candidate_prefix_s=1.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['configuration', 'fixture-run', 'bank', 'candidate-run', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=900.)
    args = parser.parse_args()
    if args.output.exists() or not 60 <= args.max_seconds <= 1800:
        raise ValueError('Fresh bounded diagnostic required')
    read = lambda path: json.loads(path.read_text())
    configuration = read(args.configuration)
    nominal = read(args.fixture_run / 'native_command_inputs.json')['default_joint_position_rad']
    bank = {r['case']: r for r in read(args.bank)['queries']}
    paths = [p for p in sorted(args.candidate_run.glob('case_*.json'))
             if (read(p).get('fine') or {}).get('model_path_complete')]
    if not paths:
        raise ValueError('No complete fine-model candidate in current snapshot')
    args.output.mkdir(parents=True)

    def write(name, value):
        path = args.output / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    pins = [Path(__file__), args.configuration, args.bank,
            args.fixture_run / 'native_command_inputs.json', Path(configuration['model_reference_library']), *paths]
    write('inputs.json', dict(source_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in pins},
        selected_cases=[read(p)['source_case'] for p in paths], candidate_snapshot=True,
        max_seconds=args.max_seconds, plant_substeps=160, training=False))
    started = time.monotonic()
    results = []
    for path in paths:
        candidate = read(path)
        write('status.json', dict(state='RUNNING', case=candidate['source_case'], completed=len(results), total=len(paths)))
        result = evaluate(configuration, nominal, candidate, bank, started + args.max_seconds)
        write('case_%03d.json' % result['source_case'], result)
        results.append({k: v for k, v in result.items() if k != 'trajectory'})
    summary = dict(state='COMPLETED', cases=results, wall_seconds=time.monotonic() - started,
                   native_success=False, valid_expert_labels=False)
    write('result.json', summary)
    write('status.json', summary)
    print(json.dumps(plain(summary)), flush=True)


if __name__ == '__main__':
    main()
