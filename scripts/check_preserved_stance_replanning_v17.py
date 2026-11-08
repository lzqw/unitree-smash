"""Replay actual prefixes and check causal state matching for computed replans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.pure_control_preserved_stance_replanning_v17 import PureControlPreservedStanceReplanningV17
from scripts.pure_control_settled_stance_v14 import PureControlSettledStanceV14
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.audit_actual_handoff_model_v1 import state_from_record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','configuration','library','output']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();read=lambda f:json.loads(Path(f).read_text())
    if a.output.exists():raise ValueError('Fresh check required')
    cfg=read(a.configuration);nominal=read(a.run/'native_command_inputs.json')['default_joint_position_rad']
    with np.load(a.run/'pure_control_commands.npz') as z:arrays={k:z[k].copy() for k in z.files}
    diag=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    rows=[]
    for source in read(a.library)['references']:
        case,end=source['source_case'],source['source_actual_tick']
        old=PureControlSettledStanceV14(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'])
        new=PureControlPreservedStanceReplanningV17(PositionPDInterface(**cfg['interface']),nominal,cfg['model_reference_library'],a.library)
        ids=np.flatnonzero((arrays['lane']==case)&(arrays['tick']<=end));errors=[];observed=[]
        if len(ids)<2 or np.any(np.diff(arrays['tick'][ids])!=1):raise ValueError('Incomplete actual prefix')
        for j,i in enumerate(ids):
            i=int(i);r=diag[i]
            record={k:arrays[k][i].copy() for k in ['q','v','previous','ground_force']};record['phase']=r['phase']
            state=state_from_record(new.model,record);q,v=state.q.copy(),state.v.copy()
            time_s=float(arrays['time_s'][i]);phase=r['phase']['phase']
            original,_=old.command(state,phase,time_s);target,d=new.command(state,phase,time_s)
            if not np.array_equal(q,state.q) or not np.array_equal(v,state.v):raise ValueError('Controller altered observed state')
            if int(arrays['tick'][i])<end:
                error=float(np.max(abs(target-original)));errors.append(error)
                if error>1e-12 or d['replanned']:raise ValueError('Prefix changed before causal state match')
                observed.append(float(np.max(abs(target-arrays['previous'][int(ids[j+1])]))))
                if observed[-1]>2e-6:raise ValueError('Prefix command did not reproduce native readback')
            else:
                if not d['replanned'] or new.replanned is None:raise ValueError('Covered actual state failed to select plan')
                if np.any(target<new.interface.limits[:,0]) or np.any(target>new.interface.limits[:,1]) or not np.isfinite(target).all():
                    raise ValueError('Invalid replan command')
                chosen,score=new.replanning_library.match(state)
                label=chosen['source_case'];chosen['source_case']=-999
                if new.replanning_library.match(state)[0]['source_key']!=chosen['source_key']:
                    raise ValueError('Runtime matching depends on TRAIN label')
                chosen['source_case']=label
                state.q[0]+=.3;state.root_position_world[0]+=.3
                if new.replanning_library.match(state)[0] is not None:raise ValueError('Uncovered state silently matched')
                rows.append(dict(source_case=case,actual_replanning_tick=end,prefix_steps=len(errors),
                    maximum_prefix_change_rad=max(errors,default=0.),maximum_native_readback_error_rad=max(observed,default=0.),
                    covered_state_match_score=score,first_replanned_command_source=d['source'],
                    first_replanned_vs_old_command_max_rad=float(np.max(abs(target-original)))))
    files=[Path(__file__),a.run/'pure_control_commands.npz',a.run/'pure_control_diagnostics.jsonl',a.run/'native_command_inputs.json',a.configuration,a.library]
    files += [Path(m.__file__) for n,m in list(sys.modules.items()) if n.startswith('scripts.') and getattr(m,'__file__',None)]
    result=dict(passed=True,checks=rows,source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        native_execution=False,valid_expert_labels=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows))


if __name__=='__main__':main()
