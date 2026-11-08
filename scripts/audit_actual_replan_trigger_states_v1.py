"""Find first causal plan matches on complete native histories without control replay.

This measures the current local matcher's trigger state, not rescue success.
No success/failure label or TRAIN ID participates in runtime eligibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from scripts.recovery_reference_library_v2 import RecoveryReferenceLibraryV2
from scripts.audit_actual_handoff_model_v1 import state_from_record
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','prefix-library','replanning-library','output']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if a.output.exists() or read(a.run/'job.json')['status']!='SUCCEEDED':
        raise ValueError('Fresh audit of completed native history required')
    model=LoadedPinocchioModelV1();library=RecoveryReferenceLibraryV2(a.replanning_library,model)
    refs={r['source_key']:r for r in read(a.prefix_library)['references']}
    with np.load(a.run/'pure_control_commands.npz') as z:
        data={k:z[k].copy() for k in ['tick','lane','time_s','q','v','previous','ground_force']}
    rows=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    if len(rows)!=len(data['lane']):raise ValueError('Misaligned records')
    matches=[];unmatched=[]
    for lane in sorted(set(data['lane'].tolist())):
        ids=np.flatnonzero(data['lane']==lane);first=int(ids[0])
        source=refs.get(rows[first]['diagnostic'].get('offline_reference_source_key'))
        found=False;eligible=0
        for i in ids:
            i=int(i);row=rows[i]
            if (row['tick'],row['lane'])!=(data['tick'][i],data['lane'][i]):raise ValueError('Record mismatch')
            if source is None or int(round((float(data['time_s'][i])-float(data['time_s'][first]))/.01))<len(source['trajectory']):
                continue
            eligible+=1
            record={k:data[k][i].copy() for k in ['q','v','previous','ground_force']}
            record.update(case=int(lane),tick=int(data['tick'][i]),time_s=float(data['time_s'][i]),phase=row['phase'])
            state=state_from_record(model,record);chosen,score=library.match(state)
            if chosen is not None:
                matches.append(dict(case=int(lane),actual_tick=record['tick'],query_source_case=chosen['source_case'],
                    original_query_tick=chosen['source_actual_tick'],score=score,source_key=chosen['source_key'],
                    first_matching_actual_state=record,native_success=False))
                found=True;break
        if not found:unmatched.append(dict(case=int(lane),eligible_observations=eligible))
    files=[Path(__file__),a.run/'job.json',a.run/'pure_control_commands.npz',a.run/'pure_control_diagnostics.jsonl',
           a.prefix_library,a.replanning_library]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(matches=matches,unmatched=unmatched,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        old_history_scan_not_new_closed_loop=True,native_success=False,valid_expert_labels=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ['first_matching_actual_state','source_key']} for r in matches]))


if __name__=='__main__':main()
