"""Check the local recovery candidate on other recorded, same-clock states.

Offline diagnostic only: outcome labels select a failure cohort, never a runtime
gate. This deliberately tests the existing case-3 candidate without retiming or
changing its reference. A model pass is not a native rescue or expert label.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.compare_frozen_complete5_v2 import read_run
from scripts.recorded_nonfoot_contact_history_v1 import RecordedNonfootContactHistory
from scripts.loaded_pinocchio_model_v1 import rotation_wxyz
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.feedback_tail_rollout_v1 import evaluate_feedback_tail
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['bundle', 'validation', 'warm-plan', 'sphere-audit', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--nearest', type=int, default=5)
    parser.add_argument('--substeps', type=int, choices=[16, 160], default=16)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.nearest <= 25:
        raise ValueError('New output and bounded diagnostic cohort required')
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = read(args.bundle/'manifest.json')
    baseline = Path(manifest['baseline'])
    protocol_path = ROOT/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json'
    protocol = read(protocol_path); protocol['_root'] = str(ROOT)
    score, _ = read_run(baseline, protocol)
    if score['complete5'] != 27 or score['ball5_except_recovery'] != 53:
        raise ValueError('Require declared frozen baseline')
    trace_manifest_path = baseline/'fixed_policy_traces/manifest.json'
    trace_manifest = read(trace_manifest_path)
    inputs_path = baseline/'native_command_inputs.json'
    inputs = read(inputs_path)
    default = np.asarray(inputs['default_joint_position_rad'])
    limits = np.asarray(inputs['joint_limits_rad'])
    controller = build_controller(manifest['configuration']); model = controller.qp.model
    if not np.array_equal(limits, controller.qp.interface.limits):
        raise ValueError('Different native command interface')
    archive = args.bundle/manifest['bundle_name']
    if sha(archive) != manifest['bundle_sha256']:
        raise ValueError('Changed case-3 source')
    with np.load(archive, allow_pickle=False) as data:
        query = data['case_003_query_public'].copy()
    feedback, _ = load_verified_policy(model, controller, query, args.warm_plan, args.validation)
    poses_path = baseline/'all_first_episodes.npz'
    geometry_path = baseline/'geometry.json'
    root_index = read(geometry_path)['body_names'].index('base')
    with np.load(poses_path, allow_pickle=False) as data:
        poses = data['link_poses'].copy(); valid = data['valid_first_episode'].copy()
    events_path = baseline/'first_nonfoot_contact_details.json'
    events = [row for row in read(events_path) if row['physics_frame'] % 320 == 0]
    history = RecordedNonfootContactHistory(baseline)
    if abs(history.dt - .0000625) > 1e-12:
        raise ValueError('Unexpected physics clock')
    sources = [Path(__file__), args.bundle/'manifest.json', archive, args.validation,
        args.warm_plan, args.sphere_audit, protocol_path, trace_manifest_path, inputs_path,
        poses_path, geometry_path, events_path, baseline/'contact_telemetry_metadata.json', model.path]
    sources += [baseline/name for name in score['source_sha256']]
    cohort = [row['training_index'] for row in score['stages']
              if row['ball5_except_recovery'] and not row['complete5']]
    entries = {row['source_training_index']: row for row in trace_manifest['episodes']}
    states = {}; rows = []; checks = []; max_error = dict(position=0., rotation=0., joint=0., velocity=0.)
    for case in cohort:
        entry = entries[case]
        if entry['env_id'] != case:
            raise ValueError('Case/lane mapping differs')
        path = baseline/'fixed_policy_traces'/entry['data_file']; sources.append(path)
        if sha(path) != entry['data_sha256']:
            raise ValueError('Changed original trace')
        with np.load(path, allow_pickle=False) as data:
            public = unpack_public(data['packed_public'])
            actions = data['dispatched_actions'].copy()
        # Independent native q/v/link-pose samples verify reconstruction; these
        # are not future contact labels supplied to the controller.
        count = 0
        for event in events:
            tick = event['physics_frame']//160
            if tick >= len(public) or not valid[tick//2, case]:
                continue
            state = model.state_from_public(public[tick], default)
            pose = np.asarray(event['body_link_pose_world_all_lanes'])[case, root_index]
            rotation = rotation_wxyz(pose[3:])
            twist = np.asarray(event['root_link_twist_world_all_lanes'])[case]
            expected_v = np.r_[rotation.T @ twist[:3], rotation.T @ twist[3:],
                np.asarray(event['joint_velocity_rad_s_all_lanes'])[case]]
            actual_v = np.r_[state.v[:6], state.joint_velocity]
            errors = dict(position=float(np.max(abs(state.q[:3]-pose[:3]))),
                rotation=float(np.max(abs(state.root_rotation_world-rotation))),
                joint=float(np.max(abs(state.joint_position-np.asarray(event['joint_position_rad_all_lanes'])[case]))),
                velocity=float(np.max(abs(actual_v-expected_v))))
            for key, value in errors.items():
                max_error[key] = max(max_error[key], value)
            if any(value > (2e-4 if key == 'velocity' else 2e-5) for key, value in errors.items()):
                raise ValueError(f'Native state reconstruction differs: {case}, {tick}, {errors}')
            count += 1
        if count == 0:
            raise ValueError('No independent native state checks')
        checks.append(dict(case=case, native_samples=count))
        tick = 460
        if len(public) <= tick:
            rows.append(dict(case=case, control_tick=tick,
                             exclusions=['first_episode_ended_before_query']))
            continue
        state = model.state_from_public(public[tick], default)
        pose = poses[tick//2, case, root_index]
        if (not valid[tick//2, case] or np.max(abs(state.q[:3]-pose[:3])) > 2e-5
                or np.max(abs(state.root_rotation_world-rotation_wxyz(pose[3:]))) > 2e-5):
            raise ValueError('Query pose does not match native recording')
        state.previous_command_target = limits[:, 0]+.5*(actions[tick-1]+1.)*np.diff(limits).ravel()
        reasons = []
        if public[tick, 774] <= .5: reasons.append('not_observed_recovery_phase')
        if public[:tick+1, 773].max() <= .5: reasons.append('no_prior_observed_airborne_hit')
        if history.seen(case, tick, public[tick, 775]): reasons.append('nonfoot_fault_already_observed')
        rows.append(dict(case=case, control_tick=tick, exclusions=reasons))
        states[case] = state
    # Distances only rank which real states to diagnose, not teacher confidence.
    anchor = states[3]
    eligible = []
    for row in rows:
        if row['case'] not in states:
            row['diagnostic_state_distance'] = None
            continue
        state = states[row['case']]
        distance = (np.mean(((state.joint_position-anchor.joint_position)/.25)**2)
                    + np.mean(((state.v-anchor.v)/2.)**2)
                    + np.mean(((state.root_rotation_world-anchor.root_rotation_world)/.2)**2)
                    + ((state.q[2]-anchor.q[2])/.1)**2)
        row['diagnostic_state_distance'] = float(distance)
        if not row['exclusions'] and row['case'] != 3: eligible.append(row)
    selected = sorted(eligible, key=lambda row: (row['diagnostic_state_distance'], row['case']))[:args.nearest]
    authored_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    recovery_path = ROOT/'configs/evaluation_protocol_v2.json'
    geometry = AuthoredGroundClearance(authored_path, model, margin_m=0.)
    recovery = read(recovery_path)['recovery']; spheres = read(args.sphere_audit)['spheres']
    sources += [authored_path, recovery_path]
    sources += [Path(module.__file__) for name, module in list(sys.modules.items())
                if name.startswith('scripts.') and getattr(module, '__file__', None)]
    hashes = {str(path.resolve()): sha(path) for path in sources}
    results = []; args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in selected:
        result = evaluate_feedback_tail(controller, states[row['case']], feedback, spheres, geometry,
                                        recovery, control_tick=460, substeps=args.substeps, time_limit_s=180.)
        results.append(dict(case=row['case'], result=result))
        print(json.dumps(plain(dict(case=row['case'], accepted=result['accepted'], failure=result['failure'],
            completed_steps=result['completed_model_steps'], terminal=result['terminal'],
            trailing_window_s=result['trailing_model_kinematic_window_s']))), flush=True)
        args.output.with_suffix('.partial.json').write_text(json.dumps(plain(results), allow_nan=False)+'\n')
    for path, digest in hashes.items():
        if sha(path) != digest: raise ValueError('Changed input '+path)
    payload = dict(scope=__doc__, native_execution=False, valid_expert_labels=False,
        new_native_successes=None, source_sha256=hashes, native_state_checks=checks,
        maximum_reconstruction_error=max_error, cohort=rows, selected_cases=[row['case'] for row in selected],
        results=results, integration_substeps=args.substeps,
        limitations=['Same-clock local transfer of a single-case controller, not a general recovery policy.',
            'Selection uses known failure outcomes offline; it is not a causal intervention rule.',
            'Contact flags are measured-force hysteresis, not geometric or guaranteed future support.',
            'Model results require fine integration and full native execution before any rescue claim.'])
    args.output.write_text(json.dumps(plain(payload), indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
