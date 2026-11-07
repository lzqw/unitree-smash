#!/usr/bin/env python3
"""Check whether successful MPC commands fit the current student action range.

Replay the original production student on the actual teacher-visited states.
Verify its recorded pre-MPC request, then compute the bounded mixer's reachable
command interval. No clipped inverse is exported as an expert residual label.
"""
import argparse,hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
from tensordict import TensorDict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.build_verified_fixed_retention_v2 as production
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.touchdown_contact_state_v1 import TouchdownContactStateV1,TouchdownContactSettingsV1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('Preserve prior evidence')
    read=lambda f:json.loads((args.run/f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    inputs=read('INPUTS.json');metrics=read('metrics.json');logs=read('landing_mpc_records.json')
    if read('job.json')['status']!='SUCCEEDED' or sha(args.policy)!=inputs['policy_sha256']:
        raise ValueError('Require completed matching policy execution')
    successes={x['training_index'] for x in metrics['complete5_case_outcomes'] if x['complete5']}
    corrected={x['case'] for x in logs if x.get('applied_first_step')}
    cases=sorted(successes&corrected)
    if not cases:raise ValueError('No actually corrected complete successes')
    torch.set_num_threads(1)
    policy=torch.load(args.policy,map_location='cpu',weights_only=False)
    file=ROOT/'scripts/evaluate_shared_target_ground64_persistent_recovery_v1.py'
    production.TouchdownContactStateV1=TouchdownContactStateV1
    production.TouchdownContactSettingsV1=TouchdownContactSettingsV1
    production.selected_definitions(file,['load_shared_factory','SharedTargetResidualController'],production.__dict__)
    controller=production.SharedTargetResidualController(policy,len(cases),'cpu');controller.reset()
    native=read('native_command_inputs.json');default=torch.tensor(native['default_joint_position_rad'])
    limits=torch.tensor(native['joint_limits_rad']);lim=limits.numpy()
    frames=[];commands=[];sources={str(args.policy):sha(args.policy),str(file):sha(file)}
    for case in cases:
        path=args.run/'fixed_policy_traces'/f'case_{case:03d}.npz';sources[str(path)]=sha(path)
        with np.load(path) as z:
            frames.append(unpack_public(z['packed_public']));commands.append(z['dispatched_actions'].copy())
    frames=np.stack(frames);commands=np.stack(commands)
    if frames.shape!=(len(cases),800,794):raise ValueError('Require complete ground-start episodes')
    by_tick={(r['case'],r['control_tick']):r for r in logs if r.get('episode_index')==0}
    original=controller.mixer.action;last_base=[]
    def capture(base,*a,**kw):
        last_base[:] = [base.detach().numpy().copy()];return original(base,*a,**kw)
    controller.mixer.action=capture
    scale=np.asarray(policy['residual_joint_scale_rad']);rate=float(policy['residual_arm_target_rate_rad_s'])
    rows=[];request_errors=[]
    with torch.inference_mode():
        for tick in range(800):
            raw=torch.from_numpy(frames[:,tick].copy())
            student=controller.action(TensorDict({'policy':raw},batch_size=[len(cases)]),default,limits).numpy()
            base_q=lim[:,0]+.5*(last_base[0]+1.)*(lim[:,1]-lim[:,0])
            student_q=lim[:,0]+.5*(student+1.)*(lim[:,1]-lim[:,0])
            for lane,case in enumerate(cases):
                r=by_tick.get((case,tick))
                if r is None:continue
                request_errors.append(float(np.max(abs(student_q[lane]-np.asarray(r['requested_student_joint_reference'])))))
                low=np.maximum(base_q[lane]-scale,lim[:,0]);high=np.minimum(base_q[lane]+scale,lim[:,1])
                previous=np.asarray(r['previous_dispatched_target']);teacher=np.asarray(r['final_dispatched_joint_target_rad'])
                low[12:]=np.clip(low[12:],previous[12:]-rate*.01,previous[12:]+rate*.01)
                high[12:]=np.clip(high[12:],previous[12:]-rate*.01,previous[12:]+rate*.01)
                violation=np.maximum(np.maximum(low-teacher,teacher-high),0.)
                rows.append(dict(case=case,tick=tick,applied=r['applied_first_step'],
                    max_command_interval_violation_rad=float(violation.max()),
                    violating_joint_indices=np.flatnonzero(violation>5e-4).tolist()))
    maximum=max(request_errors,default=float('inf'))
    if maximum>5e-4:raise ValueError('Original student request replay differs: '+str(maximum))
    summaries=[]
    for case in cases:
        subset=[r for r in rows if r['case']==case];bad=[r for r in subset if r['max_command_interval_violation_rad']>5e-4]
        summaries.append(dict(case=case,corrected_complete5=True,checked_steps=len(subset),
            outside_current_residual_range_steps=len(bad),
            maximum_interval_violation_rad=max(r['max_command_interval_violation_rad'] for r in subset),
            first_outside_range=None if not bad else bad[0]))
    result=dict(scope=__doc__,cases=cases,residual_scale_rad=scale.tolist(),arm_target_rate_rad_s=rate,
        student_pre_mpc_request_replay_error_max_rad=maximum,command_tolerance_rad=5e-4,
        summaries=summaries,records=rows,source_sha256=sources,expert_residual_labels_exported=False,
        actual_student_learning=False,native_new_execution=False)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(student_request_replay_error=maximum,summaries=summaries)))


if __name__=='__main__':main()
