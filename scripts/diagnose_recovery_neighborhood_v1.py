"""Separate pre-existing model geometry from local reference/command mismatch."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['result', 'bundle', 'validation', 'warm-plan', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve earlier evidence')
    read = lambda p: json.loads(Path(p).read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result = read(args.result)
    for path, digest in result['source_sha256'].items():
        if sha(path) != digest: raise ValueError('Changed diagnostic source '+path)
    manifest = read(args.bundle/'manifest.json')
    root = Path(manifest['baseline'])
    native = read(root/'native_command_inputs.json')
    default = np.asarray(native['default_joint_position_rad'])
    limits = np.asarray(native['joint_limits_rad'])
    controller = build_controller(manifest['configuration']); model = controller.qp.model
    archive = args.bundle/manifest['bundle_name']
    with np.load(archive, allow_pickle=False) as data:
        query = data['case_003_query_public'].copy()
    policy, _ = load_verified_policy(model, controller, query, args.warm_plan, args.validation)
    geometry_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    geometry = AuthoredGroundClearance(geometry_path, model, margin_m=0.)
    rows = []
    for row in result['results']:
        case = row['case']; path = root/'fixed_policy_traces'/f'case_{case:03d}.npz'
        if str(path.resolve()) not in result['source_sha256']: raise ValueError('Unbound trace')
        with np.load(path, allow_pickle=False) as data:
            public = unpack_public(data['packed_public'])[460]
            previous_action = data['dispatched_actions'][459].copy()
        state = model.state_from_public(public, default)
        previous = limits[:, 0]+.5*(previous_action+1.)*np.diff(limits).ravel()
        state.previous_command_target = previous
        distances, _ = geometry.evaluate(state.q)
        worst = int(np.argmin(distances))
        target, mode = policy.action(0, state, previous)
        command_delta = target-previous
        last_states = row['result']['predicted_states']
        last_joint_velocity = None if not last_states else np.asarray(last_states[-1]['v'])[model.v_indices]
        rows.append(dict(case=case, initial_authored_minimum_gap_m=float(distances[worst]),
            initial_closest_body=geometry.shapes[worst]['body'],
            initial_joint_velocity_rad_s=state.joint_velocity,
            candidate_first_command_delta_rad=command_delta,
            largest_leg_command_delta_rad=float(np.max(abs(command_delta[:12]))),
            largest_arm_command_delta_rad=float(np.max(abs(command_delta[12:]))),
            last_completed_model_joint_velocity_rad_s=last_joint_velocity,
            failure=row['result']['failure'], model_steps_completed=row['result']['completed_model_steps']))
    sources = [args.result, Path(__file__), geometry_path]
    sources += [Path(module.__file__) for name, module in list(sys.modules.items())
                if name.startswith('scripts.') and getattr(module, '__file__', None)]
    hashes = {str(path.resolve()): sha(path) for path in sources}
    payload = dict(scope=__doc__, native_execution=False, valid_expert_labels=False, rows=rows,
        joint_order=model.joint_order, inherited_source_sha256=result['source_sha256'], source_sha256=hashes,
        limitations=['Authored geometric overlap is not measured nonfoot force or a PhysX collision certificate.',
            'Command mismatch is diagnostic, not proof that smoothing alone restores feasibility.',
            'Last completed state precedes the failing integration interval; it is not the peak violation.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plain(payload), indent=2, allow_nan=False)+'\n')
    print(json.dumps(plain(rows)))


if __name__ == '__main__': main()
