"""Replace short model plans with fine-checked full18 recovery candidates.

This preserves causal initial-state matching. Model path completion does not
assert sustained recovery or native success; those are separate evaluations.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['base-library', 'search', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh immutable library required')
    read = lambda path: json.loads(path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    base = read(args.base_library)
    if base['schema'] != 'model-recovery-reference-library-v1':
        raise ValueError('Expected original short-reference library')
    refs = {r['source_case']: r for r in base['references']}
    pins = {str(args.base_library.resolve()): digest(args.base_library)}
    replaced = []
    for path in sorted(args.search.glob('case_*.json')):
        result = read(path)
        fine = result.get('fine')
        if not fine or not fine['model_path_complete']:
            continue
        case = result['source_case']
        old = refs[case]
        if result['source_key'] != old['source_key'] or len(fine['trajectory']) != fine['required_steps']:
            raise ValueError('Candidate provenance or trajectory mismatch')
        pins[str(path.resolve())] = digest(path)
        refs[case] = dict(source_case=case, source_key=digest(path),
            parent_source_key=old['source_key'], query=old['query'], trajectory=fine['trajectory'],
            terminal=fine['terminal'], native_success=False,
            prefix_model_terminal_stable_s=fine['terminal_stable_s'])
        replaced.append(case)
    if not replaced:
        raise ValueError('No longer fine-checked candidate')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(schema='model-recovery-reference-library-v2',
        control_dt_s=.01, references=[refs[k] for k in sorted(refs)],
        replaced_source_cases=replaced, source_sha256=pins, native_success=False,
        valid_expert_labels=False, training=False), indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(references=len(refs), replaced_source_cases=replaced, native_success=False)))


if __name__ == '__main__':
    main()
