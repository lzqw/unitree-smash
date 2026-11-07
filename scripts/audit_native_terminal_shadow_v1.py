"""Audit a native terminal-feedback shadow run without relaxing lane isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.audit_native_terminal_support_v1 import state_at
from scripts.native_terminal_support_v1 import load_verified_policy
from scripts.parallel_landing_solver_v1 import build_controller


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'bundle', 'output']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Preserve earlier audit evidence')
    read = lambda name: json.loads((a.run / name).read_text())
    manifest = json.loads((a.bundle / 'manifest.json').read_text())
    baseline = Path(manifest['baseline'])
    old_inputs = json.loads((baseline / 'INPUTS.json').read_text())
    inputs, job, metrics, result = (read(n) for n in
        ['INPUTS.json', 'job.json', 'metrics.json', 'sequence_native_result.json'])
    for key in ['policy_sha256', 'protocol_sha256', 'training_suite', 'origin_precision_receipt_sha256',
                'origin_spacing_m', 'training_indices', 'reset_noise_amplitudes', 'ground_extent_diagnostic',
                'recovery_assistance_execution', 'support_config', 'support_disabled', 'residual_actor_enabled']:
        if inputs[key] != old_inputs[key]:
            raise ValueError('Changed task/control contract ' + key)
    for key in ['landing_mpc_cases', 'learned_recovery_command_head', 'reactive_vhsip_cases', 'expert_bank']:
        if inputs[key] is not None:
            raise ValueError('Unexpected additional controller ' + key)
    if (job['status'] != 'SUCCEEDED' or job['exit_code'] != 0 or job['timed_out']
            or metrics['numerical_failures'] or metrics['actual_control_env_steps'] != 102400
            or inputs['training_indices'] != list(range(128)) or not inputs['native_shadow_only']
            or not result['native_shadow_only'] or result['teacher_steps'] != 0
            or result['applied_steps'] != 0 or result['shadow_proposal_steps'] != 340
            or result['student_dispatch_steps'] != 800 or not result['query_match']['passed']
            or result['valid_expert_labels'] or result['sustained_terminal_feedback_implemented']
            or result['sustained_mpc_feedback_implemented']):
        raise ValueError('Require complete, unchanged, genuinely nonexecuted shadow proposals')
    if read('contact_telemetry_metadata.json')['physics_dt_s'] != json.loads(
            (baseline / 'contact_telemetry_metadata.json').read_text())['physics_dt_s']:
        raise ValueError('Changed native physics step')
    with np.load(a.run / 'sequence_dispatch.npz', allow_pickle=False) as z:
        student, final, valid = (z[k].copy() for k in ['student', 'final', 'first_episode'])
    if student.shape != (800, 128, 18) or not np.array_equal(student, final):
        raise ValueError('Shadow control changed an actual native command')
    with np.load(a.run / 'sequence_native_trace.npz', allow_pickle=False) as z:
        trace = {k: z[k].copy() for k in z.files}
    case, start = result['case'], result['start_tick']
    limits = np.asarray(manifest['configuration']['interface']['limits'])
    targets = limits[:, 0] + .5 * (final[:, case] + 1.) * np.diff(limits).ravel()
    c = build_controller(manifest['configuration'])
    with np.load(a.bundle / manifest['bundle_name'], allow_pickle=False) as z:
        query = z[f'case_{case:03d}_query_public'].copy()
    policy, _ = load_verified_policy(c.qp.model, c, query,
                                     Path(inputs['frictional_warm_plan']), Path(result['validation_path']))
    maximum = 0.
    if len(result['records']) != 340:
        raise ValueError('Missing shadow proposal/state records')
    for i, row in enumerate(result['records']):
        tick = start + i
        if row['control_tick'] != tick or row['mode'] != 'exact_student_dispatch_shadow_not_executed':
            raise ValueError('Wrong shadow control clock or label')
        if not np.array_equal(trace['final_action'][i], student[tick, case]):
            raise ValueError('Trace describes a proposal as execution')
        if not np.allclose(trace['previous_target_rad'][i], targets[tick-1], atol=1e-6, rtol=0):
            raise ValueError('Shadow prediction used counterfactual command history')
        if not np.allclose(trace['final_target_rad'][i], targets[tick], atol=1e-7, rtol=0):
            raise ValueError('Wrong actual-command record')
        expected, mode = policy.action(i, state_at(c.qp.model, trace, i), trace['previous_target_rad'][i])
        err = float(np.max(abs(expected-trace['shadow_requested_target_rad'][i])))
        maximum = max(maximum, err)
        if err > 1e-8 or mode != row['shadow_mode']:
            raise ValueError('Shadow proposal cannot be reproduced from the actual state')
    traces = read('fixed_policy_traces/manifest.json')
    old = json.loads((baseline / 'fixed_policy_traces/manifest.json').read_text())
    old_rows = {row['source_training_index']: row for row in old['episodes']}
    if sorted(row['source_training_index'] for row in traces['episodes']) != list(range(128)):
        raise ValueError('Missing full-panel trace coverage')
    comparisons, sources = [], []
    for row in traces['episodes']:
        lane = row['source_training_index']
        if row['env_id'] != lane:
            raise ValueError('Unexpected lane mapping')
        path = a.run / 'fixed_policy_traces' / row['data_file']
        prior = baseline / 'fixed_policy_traces' / old_rows[lane]['data_file']
        if sha(path) != row['data_sha256'] or sha(prior) != old_rows[lane]['data_sha256']:
            raise ValueError('Changed trajectory artifact')
        sources.extend([path, prior])
        with np.load(path, allow_pickle=False) as z:
            obs, actions = unpack_public(z['packed_public']), z['dispatched_actions'].copy()
        with np.load(prior, allow_pickle=False) as z:
            old_obs, old_actions = unpack_public(z['packed_public']), z['dispatched_actions'].copy()
        n, count = len(actions), min(len(actions), len(old_actions))
        if not np.array_equal(actions, final[:n, lane]) or not valid[:n, lane].all() or valid[n:, lane].any():
            raise ValueError('Recorded action was not dispatched in the first episode')
        comparisons.append(dict(case=lane, same_episode_length=n == len(old_actions),
            samples=count, public_max_error=float(np.max(abs(obs[:count]-old_obs[:count]))),
            action_max_error=float(np.max(abs(actions[:count]-old_actions[:count])))))
    passed = all(row['same_episode_length'] and row['public_max_error'] <= 5e-4
                 and row['action_max_error'] <= 5e-6 for row in comparisons)
    sources += [a.run / n for n in ['job.json', 'INPUTS.json', 'metrics.json', 'sequence_native_result.json',
        'sequence_dispatch.npz', 'sequence_native_trace.npz', 'fixed_policy_traces/manifest.json']]
    sources += [Path(__file__), a.bundle / 'manifest.json', Path(result['validation_path'])]
    report = dict(scope=__doc__, passed_dispatch_integrity=passed,
        every_dispatched_action_equals_current_student=True, native_shadow_only=True,
        shadow_proposals=340, maximum_recomputed_shadow_request_error_rad=maximum,
        per_case_comparison=comparisons, complete5=metrics['complete5'],
        valid_expert_labels=False, automatic_promotion=False, teacher_off_learning_result=False,
        source_sha256={str(path.resolve()): sha(path) for path in sources})
    with a.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['per_case_comparison', 'source_sha256']}))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
