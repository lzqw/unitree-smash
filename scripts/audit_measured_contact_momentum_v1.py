"""Explain one-step errors using measured normal AND friction impulses.

This is a retrospective momentum diagnostic. It uses the measured END of the
interval to obtain force, so it is explicitly not a predictive model, controller,
or expert label. No new force is injected into the native simulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.check_constrained_recovery_suffix_v1 import source_context
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.native_recovery_query_bank_v1 import state_from_sample
from scripts.frictional_position_limited_step_v1 import position_limited_velocity_step
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['plan','run','audit','baseline','output']:
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh output required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=read(a.audit);baseline=read(a.baseline)
    if not audit['filtered_observation_passed'] or not audit['passive_parent_replay']:
        raise ValueError('Transparent complete filtered native observation required')
    pins=dict(audit['source_sha256']);pins.update(baseline['source_sha256'])
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed input '+f)
    source=read(read(a.plan)['source']);*_,context=source_context(source['configuration']['source'])
    initialize(*context);w=_WORKER;model=w['controller'].qp.model;interface=w['controller'].qp.interface
    meta=read(a.run/'foot_impact_observation.json')
    with np.load(a.run/'foot_impact_observation.npz',allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
    with np.load(a.run/'filtered_foot_pairs.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    if not np.array_equal(data['physics_frame'],obs['physics_frame']):raise ValueError('Different force and state clocks')
    dt=meta['physics_dt_s'];select=np.zeros((24,18));select[model.v_indices,np.arange(18)]=1.
    old={(r['case'],r['physics_frame']):r for r in baseline['rows']}
    spheres={s['body']:s for s in w['spheres']}
    rows=[];began=time.monotonic()
    for lane,case in enumerate(meta['lanes']):
        states=[state_from_sample(model,{k:v[i,lane] for k,v in obs.items() if k!='physics_frame'},
            meta['origins_world'][lane]) for i in range(len(obs['physics_frame']))]
        for i,frame in enumerate(obs['physics_frame'][:-1]):
            before,after=states[i:i+2];snap=model.snapshot(before.q,before.v)
            normals=data['pair_normal_force_world_n'][i+1,lane].sum(1)
            friction=data['pair_friction_force_world_n'][i+1,lane].sum(1)
            friction_moment=data['pair_friction_moment_about_foot_origin_nm'][i+1,lane].sum(1)
            row=dict(case=case,physics_frame=int(frame),model_joint_velocity_error_rad_s=old[case,int(frame)]['joint_velocity_error_rad_s'])
            for geometry,q in [('start',before.q),('end',after.q)]:
                pin.computeJointJacobians(model.model,model.data,q)
                pin.updateFramePlacements(model.model,model.data)
                normal_impulse=np.zeros(24);friction_impulse=np.zeros(24)
                for foot,name in enumerate(model.feet):
                    fid=model.body_frames[name]
                    jac=pin.getFrameJacobian(model.model,model.data,fid,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    offset=model.data.oMf[fid].rotation@np.asarray(spheres[name]['center_body_m'])
                    normal_impulse += dt*jac.T@np.r_[normals[foot],np.cross(offset,normals[foot])]
                    friction_impulse += dt*jac.T@np.r_[friction[foot],friction_moment[foot]]
                modes=['normal_only','normal_plus_friction'] if geometry=='start' else ['normal_plus_friction']
                for mode in modes:
                    impulse=normal_impulse+(friction_impulse if mode=='normal_plus_friction' else 0.)
                    result=position_limited_velocity_step(snap['mass_matrix'],before.v,
                        snap['nonlinear']-impulse/dt,select,interface.kp,interface.kd,
                        after.previous_command_target-before.q[model.q_indices],interface.effort,dt,
                        np.empty((0,24)),np.empty(0),interface.friction,interface.velocity,
                        before.q[model.q_indices],interface.limits)
                    error=result['velocity']-after.v
                    row[geometry+'_'+mode]=dict(joint_velocity_error_rad_s=float(np.max(abs(error[model.v_indices]))),
                        base_linear_error_m_s=float(np.linalg.norm(error[:3])),
                        base_angular_error_rad_s=float(np.linalg.norm(error[3:6])),
                        generalized_contact_impulse=impulse)
            rows.append(row)
    fields=['start_normal_only','start_normal_plus_friction','end_normal_plus_friction']
    summaries={}
    for case in meta['lanes']:
        selected=[r for r in rows if r['case']==case]
        summaries[str(case)]={f:dict(maximum_joint_velocity_error_rad_s=max(r[f]['joint_velocity_error_rad_s'] for r in selected),
            median_joint_velocity_error_rad_s=float(np.median([r[f]['joint_velocity_error_rad_s'] for r in selected]))) for f in fields}
        summaries[str(case)]['large_model_errors']=[r for r in selected if r['model_joint_velocity_error_rad_s']>1.]
    for f in [a.plan,a.audit,a.baseline,a.run/'filtered_foot_pairs.npz',Path(__file__)]:pins[str(f.resolve())]=sha(f)
    for f,h in pins.items():
        if sha(f)!=h:raise ValueError('Changed source during analysis '+f)
    result=dict(scope=__doc__,summaries=summaries,rows=rows,source_sha256=pins,seconds=time.monotonic()-began,
        uses_measured_future_contact=True,causal_predictor=False,controller_changed=False,valid_expert_labels=False,
        limitations=['Implicit drive and limit equations still approximate native finite-iteration response.',
            'Normal moment uses authored sphere center; friction moment is native measured.',
            'Start/end Jacobians bracket geometry timing; neither is a native TGS replication.'])
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print(json.dumps({c:{f:v for f,v in s.items() if f!='large_model_errors'} for c,s in summaries.items()}))


if __name__=='__main__':main()
