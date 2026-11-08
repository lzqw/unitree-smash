"""Extend a frozen initial-state reference bank with model-checked recoveries.

The native controller's existing state matcher chooses references; source case
numbers are provenance, never a runtime intervention mask. Native evaluation
and preservation of verified cases remain separate mandatory checks.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['base','addition','end-check','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(f.read_text())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Fresh immutable reference bank required')
    base,addition=read(a.base),read(a.addition)
    pins={}
    for source in [base,addition]:
        if source['schema']!='model-recovery-reference-library-v2' or source['native_success'] or source['valid_expert_labels']:
            raise ValueError('Model-only references required')
        for f,h in source['source_sha256'].items():
            if sha(f)!=h or (f in pins and pins[f]!=h):raise ValueError('Changed/conflicting source '+f)
            pins[f]=h
    checks=read(a.end_check)
    end={r['state_key']:r for r in checks['results']}
    known={r['source_key'] for r in base['references']}
    for r in addition['references']:
        if r['source_key'] in known:raise ValueError('Duplicate reference')
        known.add(r['source_key'])
        key=str(r['source_case'])+':'+str(r['source_actual_tick'])
        check=end[key]
        if not check['complete_model_feedback'] or not check['model_terminal_window_passed'] or check['end_time_s']!=8.:
            raise ValueError('Complete feedback and actual final observation required')
    references=base['references']+addition['references']
    for f in [Path(__file__),a.base,a.addition,a.end_check]:pins[str(f.resolve())]=sha(f)
    result=dict(schema=base['schema'],control_dt_s=.01,references=references,
        source_sha256=pins,native_success=False,valid_expert_labels=False,training=False,
        original_reference_count=len(base['references']),added_reference_count=len(addition['references']),
        base_reference_order_and_values_preserved=True,
        matching='Unchanged causal state matcher; no source-case runtime gate',
        full_model_feedback_is_not_native_certification=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['references','source_sha256']}))


if __name__=='__main__':main()
