"""Combine model-only plans after feedback and final 8-second observation checks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--libraries',type=Path,nargs='+',required=True)
    p.add_argument('--end-audits',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists() or len(a.libraries)!=len(a.end_audits):raise ValueError('Fresh paired inputs required')
    references=[];pins={str(Path(__file__).resolve()):sha(__file__)};keys=set()
    for lp,ap in zip(a.libraries,a.end_audits):
        lib,audit=read(lp),read(ap)
        if lib['native_success'] or lib['valid_expert_labels'] or audit['acceptance_changed']:
            raise ValueError('Original criteria and model-only labels required')
        if lib['schema']!='model-recovery-reference-library-v2' or lib['control_dt_s']!=.01:
            raise ValueError('Incompatible library')
        for data in [lib,audit]:
            for f,h in data['source_sha256'].items():
                if sha(f)!=h or (f in pins and pins[f]!=h):raise ValueError('Changed or conflicting source '+f)
                pins[f]=h
        for row in lib['references']:
            key=str(row['source_case'])+':'+str(row['source_actual_tick'])
            checks=[x for x in audit['results'] if x['state_key']==key]
            if (len(checks)!=1 or not checks[0].get('model_terminal_window_passed')
                    or checks[0]['end_time_s']!=8.):raise ValueError('Missing full terminal observation '+key)
            for f in [row['source_search'],row['source_feedback']]:
                if audit['source_sha256'].get(f)!=sha(f):raise ValueError('Audit refers to another plan')
            if row['source_key'] in keys:raise ValueError('Duplicate plan')
            keys.add(row['source_key']);references.append(dict(row,
                model_end_observation_stable_s=checks[0]['end_observation_stable_s']))
        pins[str(lp.resolve())]=sha(lp);pins[str(ap.resolve())]=sha(ap)
    data=dict(schema='model-recovery-reference-library-v2',control_dt_s=.01,references=references,
        source_sha256=pins,native_success=False,valid_expert_labels=False,training=False,
        full_model_feedback_is_not_native_certification=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(model_candidates=len(references),source_cases=[r['source_case'] for r in references],native_success=False)))


if __name__=='__main__':main()
