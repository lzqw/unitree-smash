"""Add fine-checked cohort model plans without changing existing native inputs.

New records use the actual causal release-state bank. No successful episode
label, TRAIN number or future event is consulted by the runtime state matcher.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['base-library', 'search', 'bank', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if args.output.exists():
        raise ValueError('Fresh immutable model library required')
    base = read(args.base_library)
    if base['schema'] != 'model-recovery-reference-library-v2' or base['native_success'] or base['valid_expert_labels']:
        raise ValueError('Explicit model-only base required')
    bank = {r['case']: r['query'] for r in read(args.bank)['queries']}
    refs = {r['source_case']: r for r in base['references']}
    added = []
    pins = {str(p.resolve()): digest(p) for p in [args.base_library, args.bank]}
    for path in sorted(args.search.glob('case_*.json')):
        result = read(path)
        fine = result.get('fine')
        if not fine or not fine['model_path_complete']:
            continue
        case = result['source_case']
        if case in refs:
            raise ValueError('Cohort expansion must preserve existing candidate records')
        query = bank[case]
        key = hashlib.sha256(json.dumps(query, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if result['source_key'] != key or result['query'] != query:
            raise ValueError('Candidate does not originate at the declared recorded release state')
        trajectory = fine['trajectory']
        if len(trajectory) != 160 or fine['required_steps'] != 160:
            raise ValueError('Full 1.6-second fine-model sequence required')
        for k, row in enumerate(trajectory):
            if row['step'] != k:
                raise ValueError('Non-contiguous model sequence')
            for name, width in [('q', 25), ('v', 24), ('target', 18), ('contacts', 4)]:
                if np.shape(row[name]) != (width,) or not np.isfinite(row[name]).all():
                    raise ValueError('Malformed model state/command')
        pins[str(path.resolve())] = digest(path)
        refs[case] = dict(source_case=case, source_key=digest(path), parent_source_key=key,
            query=query, trajectory=trajectory, terminal=fine['terminal'], native_success=False,
            prefix_model_terminal_stable_s=fine['terminal_stable_s'])
        added.append(case)
    if not added:
        raise ValueError('No fine-checked new model path')
    output = dict(schema='model-recovery-reference-library-v2', control_dt_s=.01,
        references=[refs[k] for k in sorted(refs)], added_source_cases=added, source_sha256=pins,
        native_success=False, valid_expert_labels=False, training=False,
        full_episode_validation_required=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(references=len(refs), added_source_cases=added, native_success=False)))


if __name__ == '__main__':
    main()
