"""Compare feedback with nominal command replay under declared command pulses.

All trials start at the same recorded native query and execute the complete
remaining model interval. No state teleport, changed motor limits, or native
success claim. This finite perturbation bank is not a certified region.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.native_terminal_support_v1 import load_verified_policy
from scripts.extend_frictional_recovery_v1 import set_state
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.extend_recovery_to_episode_v1 import kinematic_window

CONTEXT=None


def initialize(bundle,warm,validation,sphere):
    global CONTEXT
    bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text())
    c=build_controller(manifest['configuration']);record=json.loads(Path(validation).read_text())
    with np.load(bundle/manifest['bundle_name'],allow_pickle=False) as z:
        public=z[f"case_{record['case']:03d}_query_public"].copy()
    policy,record=load_verified_policy(c.qp.model,c,public,warm,validation)
    initial=set_state(c.qp.model.state_from_public(public,np.zeros(18)),c.qp.model,
        record['initial_q'],record['initial_v'],record['initial_contacts'],record['actual_previous_command'])
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',c.qp.model,margin_m=0.)
    protocol=json.loads((ROOT/'configs/evaluation_protocol_v2.json').read_text())['recovery']
    CONTEXT=(c,policy,initial,record,json.loads(Path(sphere).read_text())['spheres'],geometry,protocol)


def evaluate(task):
    c,policy,initial,record,spheres,geometry,protocol=CONTEXT
    state=copy.deepcopy(initial);states=[];failure=None;maximum_request_delta=0.;start=time.monotonic()
    for offset in range(340):
        if task['feedback']:target,_=policy.action(offset,state,state.previous_command_target)
        else:target=np.asarray(record['commands'][offset]).copy()
        original=target.copy()
        if task['start']<=offset<task['end']:
            target[task['joint_start']:task['joint_end']]+=task['amplitude_rad']
        target=np.clip(target,c.qp.interface.limits[:,0],c.qp.interface.limits[:,1])
        target[12:]=np.clip(target[12:],state.previous_command_target[12:]-.1,state.previous_command_target[12:]+.1)
        maximum_request_delta=max(maximum_request_delta,float(np.max(abs(target-original))))
        try:item=predict_frictional_spheres(c.qp.model,c.qp.interface,state,target,spheres,geometry=geometry)
        except (ValueError,np.linalg.LinAlgError) as error:
            failure=dict(offset=offset,prediction_error=repr(error));break
        diagnostics=item['diagnostics'];bad={k:v for k,v in diagnostics.items() if
            (k=='nonfoot_ground_violation_m' and v>1e-6) or
            (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
        if bad:failure=dict(offset=offset,violations=bad);break
        state=item['state'];state.previous_command_target=target.copy();states.append(copy.deepcopy(state))
    window=kinematic_window(c.qp.model,states,protocol,.01)
    return dict(task=task,valid_steps=len(states),failure=failure,model_window=window,
        model_success=failure is None and len(states)==340 and window['longest_model_kinematic_window_s']>=.8,
        maximum_applied_pulse_rad=maximum_request_delta,wall_seconds=time.monotonic()-start,
        native_execution=False,valid_expert_labels=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['bundle','warm-plan','validation','sphere-audit','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous perturbation evidence')
    initialize(a.bundle,a.warm_plan,a.validation,a.sphere_audit)
    files=[Path(__file__),a.bundle/'manifest.json',a.warm_plan,a.validation,a.sphere_audit,
        ROOT/'configs/evaluation_protocol_v2.json']
    files += [Path(v.__file__) for k,v in list(sys.modules.items()) if k.startswith('scripts.') and getattr(v,'__file__',None)]
    sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();hashes={str(f.resolve()):sha(f) for f in files}
    tasks=[]
    for feedback in [False,True]:
        tasks.append(dict(name='nominal',feedback=feedback,start=0,end=0,joint_start=0,joint_end=4,amplitude_rad=0.))
        for label,start,end,j0,j1 in [('prefix_hips',0,10,0,4),('terminal_calves',80,100,8,12)]:
            for amplitude in [-.02,-.005,.005,.02]:
                tasks.append(dict(name=label,feedback=feedback,start=start,end=end,joint_start=j0,joint_end=j1,amplitude_rad=amplitude))
    rows=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,
        initargs=(str(a.bundle),str(a.warm_plan),str(a.validation),str(a.sphere_audit))) as pool:
        for row in pool.map(evaluate,tasks):
            rows.append(row);print(json.dumps(row),flush=True)
            a.output.with_suffix('.partial.json').write_text(json.dumps(dict(rows=rows,complete=False),indent=2)+'\n')
    for name,digest in hashes.items():
        if sha(name)!=digest:raise ValueError('Input changed '+name)
    for row in rows:
        if row['task']['name']=='nominal' and not row['model_success']:raise ValueError('Nominal validation did not reproduce')
    out=dict(scope=__doc__,rows=rows,complete=True,source_sha256=hashes,wall_seconds=time.monotonic()-started,
        native_execution=False,valid_expert_labels=False,generalization_proven=False)
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
