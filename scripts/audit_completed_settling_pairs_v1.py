"""Audit completed, immutable paired MODEL rollouts; never infer native success."""
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
    for name in ['source', 'retries', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    result = read(args.retries / 'result.json')
    if result['state'] != 'COMPLETED_BUDGET_ONLY_REPLAY' or args.output.exists():
        raise ValueError('Completed replay and fresh output required')
    inputs = read(args.retries / 'inputs.json')
    for path, digest in inputs['source_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed replay input: ' + path)
    files = [Path(__file__), args.retries / 'result.json', args.retries / 'inputs.json']
    grouped = {}
    for summary in result['results']:
        key, mode = summary['state_key'], summary['mode']
        if mode not in ['plan_end', 'state_settling'] or mode in grouped.get(key, {}):
            raise ValueError('Invalid paired membership')
        filename = key.replace(':', '_') + '_' + mode + '.json'
        path = args.retries / filename
        if not path.exists():
            path = args.source / filename
        row = read(path)
        for field in ['state_key', 'mode', 'steps', 'failure', 'full_model_path']:
            if row[field] != summary[field]:
                raise ValueError('Per-case artifact and completed summary disagree')
        if (row.get('failure') or {}).get('reason') == 'compute_budget':
            raise ValueError('Cannot characterize an unfinished pair')
        if row['native_success'] or row['valid_expert_labels']:
            raise ValueError('Expected model-only evidence')
        trace = row['trajectory']
        start = row['initial_actual_state']['tick']
        if len(trace) != row['steps'] or any(r['tick'] != start + i for i, r in enumerate(trace)):
            raise ValueError('Discontinuous recorded trajectory')
        grouped.setdefault(key, {})[mode] = row
        files.append(path)
    pairs = []
    for key, modes in sorted(grouped.items()):
        if set(modes) != {'plan_end', 'state_settling'}:
            raise ValueError('Missing comparison mode')
        old, new = modes['plan_end'], modes['state_settling']
        if old['initial_actual_state'] != new['initial_actual_state']:
            raise ValueError('Different actual starting states')
        reference_path = args.source / 'candidate_snapshots' / ('state_' + key.replace(':', '_') + '.json')
        if inputs['source_sha256'].get(str(reference_path.resolve())) != sha(reference_path):
            raise ValueError('Unpinned reference')
        reference = read(reference_path)
        if reference['initial_actual_state'] != old['initial_actual_state']:
            raise ValueError('Reference starts elsewhere')
        files.append(reference_path)
        common = min(old['steps'], new['steps'])
        differences = {field: [float(np.max(np.abs(np.asarray(old['trajectory'][i][field]) -
            new['trajectory'][i][field]))) for i in range(common)] for field in ['target', 'q', 'v']}
        first = {field: next((i for i, delta in enumerate(ds) if delta != 0.), None)
            for field, ds in differences.items()}
        command_step = first['target']
        prefix = common if command_step is None else command_step
        if any(max(differences[field][:prefix], default=0.) != 0. for field in ['q', 'v']):
            raise ValueError('State diverged before command; cannot attribute to switching')
        start = old['initial_actual_state']['tick']
        pairs.append(dict(state_key=key, reference_steps=len(reference['fine']['trajectory']),
            common_steps=common, exact_common_prefix_steps=prefix,
            first_difference_tick={field: None if step is None else start + step for field, step in first.items()},
            first_command_difference_rad=None if command_step is None else differences['target'][command_step],
            state_rows_are_after_command_integration=True,
            capture_tick=new['state_settling_capture_tick'],
            plan_end=dict(steps=old['steps'], full_model_path=old['full_model_path'],
                stable_s=old['final_model_stable_s'], failure=old['failure']),
            state_settling=dict(steps=new['steps'], full_model_path=new['full_model_path'],
                stable_s=new['final_model_stable_s'], failure=new['failure'])))
    output = dict(pairs=pairs, model_pairs=len(pairs),
        plan_end_full_paths=sum(p['plan_end']['full_model_path'] for p in pairs),
        state_settling_full_paths=sum(p['state_settling']['full_model_path'] for p in pairs),
        native_success=False, valid_expert_labels=False, training=False,
        source_sha256={str(path.resolve()): sha(path) for path in files})
    args.output.mkdir(parents=True)
    write(args.output / 'result.json', output)
    print(json.dumps({k: v for k, v in output.items() if k != 'source_sha256'}))


if __name__ == '__main__':
    main()
