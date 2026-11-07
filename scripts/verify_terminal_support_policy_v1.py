"""Replay a complete trajectory/feedback candidate from a measured native query."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import copy
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.audit_native_frictional_recovery_v1 import state_at
from scripts.terminal_support_policy_v1 import TerminalSupportPolicyV1
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.extend_recovery_to_episode_v1 import kinematic_window
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','native-run','warm-plan','tail-search','sphere-audit','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous full-sequence evidence')
    read=lambda path:json.loads(path.read_text())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    manifest=read(a.bundle/'manifest.json');warm=read(a.warm_plan);tails=read(a.tail_search)
    sphere=read(a.sphere_audit);native=read(a.native_run/'sequence_native_result.json')
    for evidence in [warm,tails,sphere]:
        for path,digest in evidence['source_sha256'].items():
            if sha(path)!=digest:raise ValueError('Changed declared source '+path)
    candidates=[r for r in tails['rows'] if Path(r['source']).resolve()==a.warm_plan.resolve()
        and r['failure'] is None and r['valid_steps']==260 and r['model_window']['longest_model_kinematic_window_s']>=.8]
    if not candidates:raise ValueError('No complete verified terminal candidate')
    chosen=max(candidates,key=lambda r:r['model_window']['longest_model_kinematic_window_s'])
    if native['case']!=warm['case'] or native['start_tick']!=460:raise ValueError('Different query clock/case')
    with np.load(a.native_run/'sequence_native_trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in z.files}
    c=build_controller(manifest['configuration']);state=state_at(c.qp.model,trace,0)
    initial=copy.deepcopy(state);policy=TerminalSupportPolicyV1(c.qp.model,c.qp.interface,state,warm,chosen)
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',c.qp.model,margin_m=0.)
    protocol=read(ROOT/'configs/evaluation_protocol_v2.json')['recovery']
    files=[Path(__file__),a.bundle/'manifest.json',a.native_run/'sequence_native_trace.npz',
        a.native_run/'sequence_native_result.json',a.warm_plan,a.tail_search,a.sphere_audit,
        ROOT/'configs/evaluation_protocol_v2.json',c.qp.model.path]
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    hashes={str(f.resolve()):sha(f) for f in files};states=[];commands=[];records=[];failure=None;began=time.monotonic()
    for offset in range(340):
        command,mode=policy.action(offset,state,state.previous_command_target)
        result=predict_frictional_spheres(c.qp.model,c.qp.interface,state,command,sphere['spheres'],geometry=geometry)
        d=result['diagnostics'];bad={k:v for k,v in d.items() if (k=='nonfoot_ground_violation_m' and v>1e-6) or
            (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        records.append(dict(offset=offset,mode=mode,diagnostics=d))
        if bad:failure=dict(offset=offset,violations=bad);break
        state=result['state'];state.previous_command_target=command.copy()
        states.append(copy.deepcopy(state));commands.append(command.copy())
    window=kinematic_window(c.qp.model,states,protocol,.01)
    for path,digest in hashes.items():
        if sha(path)!=digest:raise ValueError('Source changed during replay '+path)
    out=dict(scope=__doc__,case=warm['case'],start_tick=460,needed_steps=340,executed_model_steps=len(states),
        model_candidate_accepted=failure is None and len(states)==340 and window['longest_model_kinematic_window_s']>=.8,
        failure=failure,model_window=window,initial_q=initial.q,initial_v=initial.v,
        initial_contacts=initial.contacts,actual_previous_command=initial.previous_command_target,
        selected_tail={k:chosen[k] for k in ['mode','posture_gain','velocity_gain','hip_bias_rad','initial_static_reference']},
        states=[dict(q=s.q,v=s.v,contacts=s.contacts) for s in states],commands=commands,records=records,
        source_sha256=hashes,wall_seconds=time.monotonic()-began,measured_native_initial_state=True,
        native_continuation_executed=False,valid_expert_labels=False)
    a.output.write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:out[k] for k in ['model_candidate_accepted','executed_model_steps','failure','model_window']}))


if __name__=='__main__':main()
