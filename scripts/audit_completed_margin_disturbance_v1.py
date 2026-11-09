"""Compare completed fixed/feedback model disturbance branches without promotion.

Checks same-plan histories before the artificial impulse and the actual issued
commands. One source state and three perturbations are not 12 independent cases.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.retry_budgeted_settling_feedback_v1 import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); read = lambda p: json.loads(Path(p).read_text())
    if args.output.exists() or not (args.run/'result.json').exists():
        raise ValueError('Completed result and fresh audit required')
    result = read(args.run/'result.json'); inputs = read(args.run/'inputs.json')
    if (result['state'] != 'COMPLETED_PAIRED_MARGIN_DISTURBANCE' or
            result['worker_exceptions'] != 0 or len(result['results']) != 12):
        raise ValueError('Complete twelve-branch experiment required')
    for filename, digest in inputs['source_sha256'].items():
        if sha(filename) != digest: raise ValueError('Changed experiment source '+filename)
    expected = {(p,m,d) for p in ['baseline','optimized'] for m in ['fixed','feedback'] for d in [0.,-.05,.05]}
    keys = [(r['plan'],r['mode'],r['pitch_delta_rad_s']) for r in result['results']]
    if len(set(keys)) != 12 or set(keys) != expected: raise ValueError('Wrong paired membership')
    rows, files, checks = {}, [Path(__file__), args.run/'inputs.json', args.run/'result.json'], []
    for summary in result['results']:
        key = (summary['plan'], summary['mode'], summary['pitch_delta_rad_s'])
        path = args.run/(key[0]+'_'+key[1]+'_'+str(key[2])+'.json'); data = read(path); files.append(path)
        if {k:v for k,v in data.items() if k not in ['trajectory','initial_actual_state']} != summary:
            raise ValueError('Summary differs from branch evidence')
        trace, initial = data['trajectory'], data['initial_actual_state']
        if len(trace) != data['steps'] or any(r['tick'] != initial['tick']+i for i,r in enumerate(trace)):
            raise ValueError('Incomplete or misaligned checked trajectory')
        full = data['failure'] is None and len(trace) == 800-initial['tick']
        passed = full and trace[-1]['phase']['control_sample_stable_duration_s'] >= .8
        if (full != data['full_model_path'] or passed != data['model_terminal_window_passed'] or
                data['native_success'] or data['valid_expert_labels']):
            raise ValueError('Incorrect result classification')
        disturbance = data['disturbance']
        if disturbance is None or not np.allclose(disturbance['world_angular_velocity_delta_rad_s'],
                [0.,key[2],0.], atol=1e-12, rtol=0.):
            raise ValueError('Declared disturbance not applied')
        snapshot = args.run/(key[0]+'_sequence.json'); files.append(snapshot)
        nominal = read(snapshot)
        if nominal['initial_actual_state'] != initial: raise ValueError('Different initial state')
        differences = [float(np.max(abs(np.asarray(r['target'])-nominal['trajectory'][i]['target']))) for i,r in enumerate(trace)]
        maximum = max(differences, default=0.)
        if maximum > data['maximum_command_difference_rad']+1e-10:
            raise ValueError('Command modification telemetry understated')
        if key[1]=='fixed' and maximum != 0.: raise ValueError('Fixed branch changed nominal commands')
        checks.append(dict(plan=key[0], mode=key[1], pitch_delta_rad_s=key[2],
            checked_steps=len(trace), model_complete_and_stable=passed, failure=data['failure'],
            final_stable_s=data['final_model_stable_s'],
            maximum_checked_command_change_rad=maximum,
            first_checked_change_tick=next((r['tick'] for r,x in zip(trace,differences) if x>1e-8),None)))
        rows[key] = data
    prefix_checks = []
    for plan in ['baseline','optimized']:
        original = rows[(plan,'fixed',0.)]
        for mode in ['fixed','feedback']:
            for delta in [0.,-.05,.05]:
                branch = rows[(plan,mode,delta)]
                if branch['initial_actual_state'] != original['initial_actual_state']:
                    raise ValueError('Paired release state mismatch')
                prefix = [r for r in branch['trajectory'] if r['tick']<inputs['disturbance_tick']]
                expected_prefix = [r for r in original['trajectory'] if r['tick']<inputs['disturbance_tick']]
                if len(prefix) != len(expected_prefix): raise ValueError('Incomplete pre-disturbance execution')
                maximum = max(float(np.max(abs(np.asarray(a[name])-b[name])))
                    for a,b in zip(prefix,expected_prefix) for name in ['q','v','target','ground_force'])
                if maximum > 1e-10: raise ValueError('Paired execution already differs before disturbance')
                prefix_checks.append(dict(plan=plan,mode=mode,pitch_delta_rad_s=delta,
                    checked_prefix_steps=len(prefix),maximum_component_difference=maximum))
    pairs = []
    for plan in ['baseline','optimized']:
        for delta in [0.,-.05,.05]:
            fixed = rows[(plan,'fixed',delta)]; feedback = rows[(plan,'feedback',delta)]
            a,b = fixed['model_terminal_window_passed'],feedback['model_terminal_window_passed']
            pairs.append(dict(plan=plan,pitch_delta_rad_s=delta,fixed_passed=a,feedback_passed=b,
                feedback_rescued=bool(b and not a),feedback_lost=bool(a and not b)))
    write(args.output, dict(scope=__doc__, branches=checks, paired_results=pairs,
        pre_disturbance_checks=prefix_checks, source_states=1, native_execution=False,
        shared_model_not_native_robustness=True, valid_expert_labels=False,
        source_sha256={str(p.resolve()):sha(p) for p in files}))
    print(json.dumps(dict(paired_results=pairs, source_states=1, native_execution=False)))


if __name__ == '__main__': main()
