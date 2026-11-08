"""Build model references for explicit full-episode feedback evaluation.

Keeps the original query states and all unchanged reference records. A stable
model terminal window still needs full-episode feedback and native execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['base-library','search','bank','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--include-unstable-paths-for-diagnosis',action='store_true',
        help='Include collision-free prefixes whose endpoint is not yet stable; never expert labels')
    args=parser.parse_args()
    read=lambda p:json.loads(Path(p).read_text())
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if args.output.exists():raise ValueError('Fresh immutable output required')
    base=read(args.base_library)
    if base['schema']!='model-recovery-reference-library-v2' or base['native_success'] or base['valid_expert_labels']:
        raise ValueError('Model-only source required')
    if read(args.search/'result.json')['state']!='COMPLETED_SUSTAINED_SUPPORT_SEARCH':
        raise ValueError('Completed search required')
    bank={r['case']:r['query'] for r in read(args.bank)['queries']}
    refs={r['source_case']:r for r in base['references']}
    changed=[]
    pins={str(p.resolve()):digest(p) for p in [Path(__file__),args.base_library,args.bank,args.search/'result.json']}
    for path in sorted(args.search.glob('case_*.json')):
        result=read(path);fine=result['fine'];case=result['source_case']
        if not fine['model_path_complete'] or (fine['terminal_stable_s']<.8 and not args.include_unstable_paths_for_diagnosis):
            continue
        if case not in refs or result['query']!=bank[case] or refs[case]['query']!=bank[case]:
            raise ValueError('Reference initial state changed')
        if fine['substeps']!=160 or fine['failure'] is not None:
            raise ValueError('Fine plant verification required')
        trajectory=fine['trajectory']
        if len(trajectory)!=fine['required_steps'] or not 1<=len(trajectory)<=800:
            raise ValueError('Invalid candidate horizon')
        for k,row in enumerate(trajectory):
            if row['step']!=k:raise ValueError('Discontinuous candidate sequence')
            for name,width in [('q',25),('v',24),('target',18),('contacts',4)]:
                if np.shape(row[name])!=(width,) or not np.isfinite(row[name]).all():
                    raise ValueError('Malformed state or command')
        pins[str(path.resolve())]=digest(path)
        refs[case]=dict(source_case=case,source_key=digest(path),parent_source_key=refs[case]['source_key'],
            query=bank[case],trajectory=trajectory,terminal=fine['terminal'],native_success=False,
            model_terminal_stable_s=fine['terminal_stable_s'],full_episode_feedback_verified=False)
        changed.append(case)
    if not changed:raise ValueError('No candidate meets the declared model-path inclusion rule')
    result=dict(schema='model-recovery-reference-library-v2',control_dt_s=.01,
        references=[refs[k] for k in sorted(refs)],replaced_source_cases=changed,source_sha256=pins,
        native_success=False,valid_expert_labels=False,training=False,full_episode_validation_required=True,
        includes_unstable_endpoints_for_diagnosis=args.include_unstable_paths_for_diagnosis,
        original_stability_requirement_unchanged=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(references=len(refs),replaced_source_cases=changed,native_success=False)))


if __name__=='__main__':main()
