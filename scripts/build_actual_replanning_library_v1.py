"""Keep only computed plans that passed full-episode feedback in the model.

Acceptance here is model-only. Native deployment and recovery remain unproven.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['search','feedback','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists():raise ValueError('Immutable new library required')
    if read(a.search/'result.json')['state']!='COMPLETED_ACTUAL_RECOVERY_SEQUENCE_SEARCH' or read(a.feedback/'result.json')['state']!='COMPLETED_REPLANNED_FEEDBACK_MODEL':
        raise ValueError('Completed planning and feedback results required')
    for folder in [a.search,a.feedback]:
        for f,h in read(folder/'inputs.json')['source_sha256'].items():
            if sha(f)!=h:raise ValueError('Changed source '+f)
    references=[];excluded=[];files=[Path(__file__),a.search/'inputs.json',a.search/'result.json',a.feedback/'inputs.json',a.feedback/'result.json']
    for result in read(a.feedback/'result.json')['results']:
        key=result['state_key'];path=a.search/('state_'+key.replace(':','_')+'.json')
        feedback=a.feedback/('state_'+key.replace(':','_')+'.json');r=read(path);initial=r['initial_actual_state'];fine=r['fine']
        if (result['failure'] is not None or result['steps']!=result['required_steps'] or result['final_model_stable_s']<.8):
            excluded.append(dict(state_key=key,reason='Full model feedback did not satisfy original terminal window'));continue
        if not fine['model_path_complete'] or fine['substeps']!=160 or fine['failure'] is not None:
            raise ValueError('Inconsistent planning result')
        files.extend([path,feedback])
        references.append(dict(source_case=initial['case'],source_key=sha(path),
            query=dict(q=initial['q'],v=initial['v'],target=initial['previous'],contacts=initial['phase']['ground_contacts']),
            trajectory=fine['trajectory'],native_success=False,valid_expert_labels=False,
            source_actual_tick=initial['tick'],model_feedback_stable_s=result['final_model_stable_s'],
            source_search=str(path.resolve()),source_feedback=str(feedback.resolve())))
    if not references:raise ValueError('No full-model recovery candidate; do not fabricate a library')
    result=dict(schema='model-recovery-reference-library-v2',control_dt_s=.01,references=references,
        source_sha256={str(f.resolve()):sha(f) for f in files},excluded=excluded,
        native_success=False,valid_expert_labels=False,training=False,
        full_model_feedback_is_not_native_certification=True)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(reference_count=len(references),source_cases=[r['source_case'] for r in references],native_success=False)))


if __name__=='__main__':main()
