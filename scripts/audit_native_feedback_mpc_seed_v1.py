"""Compare shifted and state-feedback warm seeds at the same failed native MPC query.

All seeds face the original nonlinear path and 80-step terminal checks. This
does not execute a new native continuation or relax MPC terminal requirements.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import audit_terminal_feedback_native_queries_v1 as queries
from scripts import stress_terminal_support_commands_v1 as stress
from scripts import search_frictional_recovery_cem_v1 as core
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate
from scripts.audit_native_frictional_recovery_v1 import state_at


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','native-run','output']:
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve prior evidence')
    queries.initialize(*[str(path) for path in [args.bundle,args.warm_plan,args.validation,args.sphere_audit,args.native_run]])
    controller,base,_,_,spheres,geometry,_=stress.CONTEXT
    record=json.loads((args.native_run/'sequence_native_result.json').read_text())
    failure=record['planning_records'][-1];last=record['planning_records'][-2]
    if not record['planning_failed'] or failure['decision']['accepted'] or not last['decision']['accepted']:
        raise ValueError('Require recorded failed query with a previous accepted plan')
    tick=failure['tick'];shift=tick-last['tick']
    if not 1<=shift<=10 or tick+80>800:raise ValueError('Unexpected actual query interval')
    initial=state_at(controller.qp.model,queries.TRACE,tick-record['start_tick'])
    for name,value in [('observed_q',initial.q),('observed_v',initial.v),('actual_previous_target_rad',initial.previous_command_target)]:
        if not np.array_equal(failure[name],value):raise ValueError('Wrong recorded query '+name)
    old=np.asarray(last['decision']['commands'])
    shifted=np.vstack([old[shift:],np.repeat(old[-1:],shift,axis=0)])
    files=[Path(__file__),args.bundle/'manifest.json',args.warm_plan,args.validation,args.sphere_audit,
           args.native_run/'sequence_native_result.json',args.native_run/'sequence_native_trace.npz']
    files += [Path(value.__file__) for key,value in list(sys.modules.items())
              if key.startswith('scripts.') and getattr(value,'__file__',None)]
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    hashes={str(path.resolve()):sha(path) for path in files};rows=[]
    for mode,amplitude in [('shifted_previous_plan',None),('original_feedback',0.),('arm_margin_feedback',-.006)]:
        start=time.monotonic();seed=shifted.copy();states=[]
        if amplitude is not None:
            policy=ArmPrefixCandidate(base,controller.qp.interface,amplitude);state=copy.deepcopy(initial);commands=[]
            for current in range(tick,tick+80):
                target,_=policy.action(current-460,state,state.previous_command_target)
                result=stress.predict_frictional_spheres(controller.qp.model,controller.qp.interface,state,
                                                        target,spheres,geometry=geometry)
                state=result['state'];state.previous_command_target=target.copy()
                commands.append(target.copy());states.append(state.q.copy())
            seed=np.asarray(commands)
        core.CONTEXT=(controller,copy.deepcopy(initial),initial.previous_command_target.copy(),seed,spheres,geometry)
        verified=core.evaluate(np.zeros_like(seed),save_states=True)
        if states and verified['solved']:
            error=max(float(np.max(abs(a-np.asarray(b['q'])))) for a,b in zip(states,verified['states']))
            if error>1e-10:raise ValueError('Independent warm seed replay changed trajectory')
        rows.append(dict(mode=mode,arm_correction_rad=amplitude,verification=verified,
                         generation_and_verification_seconds=time.monotonic()-start))
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Input changed: '+path)
    output=dict(scope=__doc__,case=3,query_tick=tick,rows=rows,source_sha256=hashes,
                same_actual_native_state=True,native_continuation_executed=False,valid_expert_labels=False)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(mode=r['mode'],accepted=r['verification']['accepted'],
                          terminal=r['verification'].get('terminal'),seconds=r['generation_and_verification_seconds']) for r in rows]))


if __name__=='__main__':main()
