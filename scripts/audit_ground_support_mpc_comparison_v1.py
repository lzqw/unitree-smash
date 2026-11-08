"""Compare the ground-input MPC with its parent and original student.

This preserves the failed strict historical-isolation audit. It separately
checks actual command dispatch and exact parent replay; neither is a claim of
generalization, complete physical fidelity, or teacher-off learning.
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
from scripts.check_native_foot_space_mpc_v1 import verify_feedback_dispatch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'parent', 'bundle', 'ground-audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve prior evidence')
    read = lambda path: json.loads(Path(path).read_text())
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest = read(args.bundle / 'manifest.json')
    baseline = Path(manifest['baseline'])
    ground = read(args.ground_audit)
    if not ground['passed']:
        raise ValueError('Complete ground input audit required')
    for path, digest in ground['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed ground-input evidence ' + path)
    for run in [args.run, args.parent, baseline]:
        job = read(run / 'job.json')
        if job['status'] != 'SUCCEEDED' or job['exit_code'] != 0:
            raise ValueError('Completed native source required')
        if read(run / 'INPUTS.json')['policy_sha256'] != manifest['policy_sha256']:
            raise ValueError('Different frozen student')
    native = read(args.run / 'sequence_native_result.json')
    case, start = native['case'], native['start_tick']
    if case != ground['case'] or native['applied_steps'] != 520 or start != 280:
        raise ValueError('Expected sustained declared case and clock')
    with np.load(args.run / 'sequence_dispatch.npz', allow_pickle=False) as z:
        student, final = z['student'].copy(), z['final'].copy()
        first = z['first_episode'].copy()
    if student.shape != (800, 128, 18) or final.shape != student.shape:
        raise ValueError('Full dispatch required')
    others = np.arange(128) != case
    if not np.array_equal(student[:, others], final[:, others]) or not np.array_equal(student[:start], final[:start]):
        raise ValueError('Unexpected teacher action outside declared lane or prefix')
    feedback = verify_feedback_dispatch(args.run, args.bundle)
    actual_manifest = read(args.run / 'fixed_policy_traces/manifest.json')
    sources = [Path(__file__), args.ground_audit, args.bundle / 'manifest.json',
               args.run / 'sequence_dispatch.npz', args.run / 'sequence_native_result.json']
    reports = {}
    for name, other_run in [('original_student', baseline), ('previous_mpc', args.parent)]:
        reference = read(other_run / 'fixed_policy_traces/manifest.json')
        entries = {row['source_training_index']: row for row in reference['episodes']}
        rows = []
        for entry in actual_manifest['episodes']:
            lane = entry['source_training_index']
            if lane != entry['env_id']:
                raise ValueError('Case/lane mapping changed')
            path = args.run / 'fixed_policy_traces' / entry['data_file']
            other = other_run / 'fixed_policy_traces' / entries[lane]['data_file']
            sources.extend([path, other])
            if sha(path) != entry['data_sha256'] or sha(other) != entries[lane]['data_sha256']:
                raise ValueError('Changed original trace')
            with np.load(path, allow_pickle=False) as z:
                obs, actions = unpack_public(z['packed_public']), z['dispatched_actions'].copy()
            with np.load(other, allow_pickle=False) as z:
                old_obs, old_actions = unpack_public(z['packed_public']), z['dispatched_actions'].copy()
            n = len(actions)
            if not np.array_equal(actions, final[:n, lane]) or not first[:n, lane].all() or first[n:, lane].any():
                raise ValueError('Recorded actions or episode mask differ from dispatch')
            common = min(n, len(old_actions))
            differing = np.flatnonzero(np.any(obs[:common] != old_obs[:common], axis=1)
                                      | np.any(actions[:common] != old_actions[:common], axis=1))
            equal = n == len(old_actions) and not len(differing)
            first_difference = int(differing[0]) if len(differing) else (None if equal else common)
            rows.append(dict(case=lane, samples=n, reference_samples=len(old_actions),
                bitwise_equal=bool(equal), first_difference_tick=first_difference,
                common_observation_error=float(np.max(abs(obs[:common] - old_obs[:common]))),
                common_action_error=float(np.max(abs(actions[:common] - old_actions[:common])))))
        if len(rows) != 128 or len(entries) != 128:
            raise ValueError('Full original128 comparison required')
        reports[name] = dict(reference=str(other_run),
            differing_cases=[r['case'] for r in rows if not r['bitwise_equal']], rows=rows)
        sources.append(other_run / 'fixed_policy_traces/manifest.json')
    strict_equal = all(r['bitwise_equal'] for r in reports['original_student']['rows'] if r['case'] != case)
    sources.extend([args.run / 'fixed_policy_traces/manifest.json', args.run / 'metrics.json',
        args.run / 'job.json', args.run / 'INPUTS.json'])
    result = dict(scope=__doc__, native_execution=True, case=case,
        inactive_commands_equal_current_student=True, feedback_dispatch=feedback,
        comparisons=reports, strict_original_untouched_isolation_passed=strict_equal,
        previous_strict_failure_not_overridden=True,
        unique_divergence_cause_established=False, valid_expert_labels=False,
        teacher_off_learning_result=False,
        source_sha256={str(path.resolve()): sha(path) for path in sources})
    temporary = args.output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print(json.dumps(dict(case=case, strict_original_untouched_isolation_passed=strict_equal,
        differing_cases={k: v['differing_cases'] for k, v in reports.items()})))


if __name__ == '__main__':
    main()
