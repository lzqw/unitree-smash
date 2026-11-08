"""Check causal first-release coverage and commands of an expanded model bank."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_stance_foot_reference_v15 import PureControlStanceFootReferenceV15
from scripts.recovery_reference_library_v2 import RecoveryReferenceLibraryV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['configuration','library','run','feedback','output']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    if a.output.exists() or read(a.run/'job.json')['status']!='SUCCEEDED':raise ValueError('Fresh completed-history check required')
    cfg=read(a.configuration);bank=read(a.library);feedback=read(a.feedback/'result.json')
    for data in [bank,read(a.feedback/'inputs.json')]:
        for f,h in data['source_sha256'].items():
            if sha(f)!=h:raise ValueError('Changed input '+f)
    nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    c=PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
    new=RecoveryReferenceLibraryV2(a.library,c.model);old_keys={r['source_key'] for r in c.library.records}
    new_keys={r['source_key'] for r in new.records}-old_keys
    with np.load(a.run/'pure_control_commands.npz') as z:arrays={k:z[k].copy() for k in z.files}
    diagnostics=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    changed=[];preserved=0;unmatched=0;before_count=0;after_count=0
    for case in sorted(set(arrays['lane'].tolist())):
        i=int(np.flatnonzero(arrays['lane']==case)[0]);row=diagnostics[i]
        if (row['tick'],row['lane'])!=(int(arrays['tick'][i]),case) or not row['phase']['owns_final_recovery_command']:
            raise ValueError('Not a causal release record')
        record={k:arrays[k][i].copy() for k in ['q','v','previous','ground_force']}
        record['phase']=row['phase'];state=state_from_record(c.model,record)
        before,_=c.library.match(state);after,score=new.match(state)
        before_count+=int(before is not None);after_count+=int(after is not None)
        if before is not None:
            if after is None or after['source_key']!=before['source_key']:raise ValueError('Previously selected reference changed')
            preserved+=1;continue
        if after is None:unmatched+=1;continue
        if after['source_key'] not in new_keys:raise ValueError('Unexpected new match')
        key=str(case)+':'+str(int(arrays['tick'][i]))
        found=[r for r in feedback['results'] if r['state_key']==key]
        if len(found)!=1 or found[0]['failure'] is not None or found[0]['steps']!=800-int(arrays['tick'][i]):
            raise ValueError('Unverified first-match state '+key)
        evidence=read(a.feedback/('state_'+key.replace(':','_')+'.json'))
        planned=read(after['source_search'])['initial_actual_state']
        if any(not np.array_equal(record[k],planned[k]) for k in ['q','v','previous','ground_force']):
            raise ValueError('Feedback used another initial state')
        test=PureControlStanceFootReferenceV15(PositionPDInterface(**cfg['interface']),nominal,a.library)
        target,diag=test.command(state,row['phase']['phase'],float(arrays['time_s'][i]))
        error=float(np.max(abs(target-evidence['trajectory'][0]['target'])))
        if error>2e-6 or test.chosen['source_key']!=after['source_key']:raise ValueError('Initial command mismatch')
        changed.append(dict(case=case,actual_tick=int(arrays['tick'][i]),source_key=after['source_key'],
            match_score=score,first_command_error_rad=error,feedback_state_key=key))
    if {r['source_key'] for r in changed}!=new_keys:raise ValueError('Some new references lack exact causal entry evidence')
    files=[Path(__file__),a.configuration,a.library,a.feedback/'inputs.json',a.feedback/'result.json']
    files += [a.feedback/('state_'+r['feedback_state_key'].replace(':','_')+'.json') for r in changed]
    files += [a.run/n for n in ['job.json','native_command_inputs.json','pure_control_commands.npz','pure_control_diagnostics.jsonl']]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(passed=True,causal_release_count=len(set(arrays['lane'].tolist())),
        matched_before=before_count,matched_after=after_count,unchanged_old_matches=preserved,
        remaining_unmatched=unmatched,new_matches=changed,
        source_sha256={str(f.resolve()):sha(f) for f in files},native_success=False,
        runtime_gate_uses_case_ids=False,native_execution=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}))


if __name__=='__main__':main()
