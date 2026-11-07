"""Test a command-continuous seed for the existing local recovery reference.

The first requested target equals the actual previous target. Its initial
reference offset vanishes smoothly over the existing 80-step prefix. This is
an offline candidate, not a rescue certificate or a deployed MPC change.
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
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.native_terminal_attitude_v1 import load_verified_policy
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.feedback_tail_rollout_v1 import evaluate_feedback_tail
from scripts.audit_joint_sequence_proposal_v1 import plain


class CommandContinuousPrefixV1:
    def __init__(self, base, initial_state, actual_previous):
        self.base = base
        self.interface = base.interface
        self.actual_previous = np.asarray(actual_previous, float).copy()
        if (self.actual_previous.shape != (18,) or not np.isfinite(self.actual_previous).all()
                or np.any(self.actual_previous < self.interface.limits[:, 0])
                or np.any(self.actual_previous > self.interface.limits[:, 1])):
            raise ValueError('Actual bounded command history required')
        nominal, _ = base.action(0, initial_state, self.actual_previous)
        self.initial_offset = self.actual_previous-nominal

    def action(self, offset, state, previous):
        if not 0 <= offset <= 339:
            raise ValueError('Original local recovery phase required')
        target, mode = self.base.action(offset, state, previous)
        phase = np.clip(offset/80., 0., 1.)
        blend = 1.-3.*phase**2+2.*phase**3
        proposed = np.asarray(target, float)+blend*self.initial_offset
        proposed = np.clip(proposed, self.interface.limits[:, 0], self.interface.limits[:, 1])
        proposed[12:] = np.clip(proposed[12:], previous[12:]-.01*self.interface.arm_rate,
                               previous[12:]+.01*self.interface.arm_rate)
        return proposed, 'command_continuous_local_prefix' if offset < 80 else mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['comparison', 'bundle', 'validation', 'warm-plan', 'sphere-audit', 'output']:
        parser.add_argument('--'+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve previous experiment')
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    comparison = read(args.comparison)
    if comparison['integration_substeps'] != 160:
        raise ValueError('Require same native-step comparison')
    for path, digest in comparison['source_sha256'].items():
        if sha(path) != digest: raise ValueError('Changed comparison '+path)
    manifest = read(args.bundle/'manifest.json'); baseline = Path(manifest['baseline'])
    archive = args.bundle/manifest['bundle_name']
    with np.load(archive, allow_pickle=False) as data:
        query = data['case_003_query_public'].copy()
    native = read(baseline/'native_command_inputs.json')
    default = np.asarray(native['default_joint_position_rad'])
    limits = np.asarray(native['joint_limits_rad'])
    controller = build_controller(manifest['configuration']); model = controller.qp.model
    base, _ = load_verified_policy(model, controller, query, args.warm_plan, args.validation)
    geometry_path = ROOT/'configs/go2_piper_authored_ground_geometry_v1.json'
    protocol_path = ROOT/'configs/evaluation_protocol_v2.json'
    geometry = AuthoredGroundClearance(geometry_path, model, margin_m=0.)
    protocol = read(protocol_path)['recovery']; spheres = read(args.sphere_audit)['spheres']
    sources = [Path(__file__), args.comparison, args.bundle/'manifest.json', archive,
        args.validation, args.warm_plan, args.sphere_audit, geometry_path, protocol_path]
    sources += [Path(module.__file__) for name, module in list(sys.modules.items())
                if name.startswith('scripts.') and getattr(module, '__file__', None)]
    hashes = {str(path.resolve()): sha(path) for path in sources}
    rows = []; args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in comparison['selected_cases']:
        path = baseline/'fixed_policy_traces'/f'case_{case:03d}.npz'
        if str(path.resolve()) not in comparison['source_sha256']:
            raise ValueError('Missing compared native trace')
        with np.load(path, allow_pickle=False) as data:
            public = unpack_public(data['packed_public'])[460]
            previous_action = data['dispatched_actions'][459]
        state = model.state_from_public(public, default)
        state.previous_command_target = limits[:, 0]+.5*(previous_action+1.)*np.diff(limits).ravel()
        feedback = CommandContinuousPrefixV1(base, state, state.previous_command_target)
        first, _ = feedback.action(0, state, state.previous_command_target)
        error = float(np.max(abs(first-state.previous_command_target)))
        if error > 1e-12: raise ValueError('Initial command is not continuous')
        result = evaluate_feedback_tail(controller, state, feedback, spheres, geometry, protocol,
                                        control_tick=460, substeps=160, time_limit_s=180.)
        rows.append(dict(case=case, initial_command_error_rad=error,
                         initial_reference_offset_rad=feedback.initial_offset, result=result))
        print(json.dumps(plain(dict(case=case, accepted=result['accepted'], failure=result['failure'],
            completed_steps=result['completed_model_steps'], terminal=result['terminal'],
            trailing_window_s=result['trailing_model_kinematic_window_s']))), flush=True)
        args.output.with_suffix('.partial.json').write_text(json.dumps(plain(rows), allow_nan=False)+'\n')
    for path, digest in {**comparison['source_sha256'], **hashes}.items():
        if sha(path) != digest: raise ValueError('Changed input '+path)
    args.output.write_text(json.dumps(plain(dict(scope=__doc__, rows=rows, source_sha256=hashes,
        inherited_source_sha256=comparison['source_sha256'], native_execution=False,
        valid_expert_labels=False, new_native_successes=None, task_thresholds_changed=False,
        prefix_steps=80, integration_substeps=160,
        limitations=['Only the reference handoff changes; no stepping/contact sequence optimization.',
            'A failed candidate does not establish physical impossibility.',
            'Known failure cohort selects diagnostics; no causal online gate or generalization claim.'])),
        indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
