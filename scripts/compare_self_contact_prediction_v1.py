"""Paired native one-step prediction with audited hip/mount reactions added.

No parameter fitting, state teleportation, native physics changes or success
relabeling. Baseline and revised predictions use the same recorded commands.
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
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.receding_full_command_search_v1 import initialize_court
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.predict_self_contact_spheres_v2 import predict_self_contact_spheres
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1
from scripts.audit_joint_sequence_proposal_v1 import plain


class RecordingResponse(HipMountCollisionResponseV1):
    def __init__(self,model):super().__init__(model);self.impulses=[]
    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs);self.impulses.append(result['impulse'].copy());return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','model-inputs','baseline','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();read=lambda path:json.loads(Path(path).read_text())
    if a.output.exists():raise ValueError('Fresh result required')
    x=read(a.model_inputs);s=read(read(x['configuration']['model_plan'])['source'])
    *_,context=source_context(s['configuration']['source']);initialize_court(*context)
    w=_WORKER;c=w['controller'];model=c.qp.model
    def arrays(name,keys):
        with np.load(a.run/name) as z:return {k:z[k].copy() for k in keys}
    d=arrays('pure_control_commands.npz',['lane','tick','q','v','previous','final'])
    ground=arrays('pure_control_ground_forces.npz',['interval_peak_normal_n'])
    contact=arrays('landing_contact_detail.npz',['normal_impulse','friction_impulse','net_peak','foot_pose','foot_twist'])
    names=read(a.run/'pure_control_ground_forces.json')['body_names'];non=[i for i,n in enumerate(names) if not n.endswith('_foot')]
    first=read(a.baseline)['records'];records=[];start=time.monotonic()
    for before in first:
        if 'control_tick' not in before:continue
        case=before['case'];tick=before['control_tick']
        ids=np.flatnonzero((d['lane']==case)&(d['tick']==tick));nxt=np.flatnonzero((d['lane']==case)&(d['tick']==tick+1))
        if len(ids)!=1 or len(nxt)!=1:raise ValueError('Consecutive actual states required')
        i,j=int(ids[0]),int(nxt[0])
        state=state_from_row(dict(q=d['q'][i],v=d['v'][i],target=d['previous'][i],contacts=before['native_contacts']))
        snap=model.snapshot(state.q,state.v)
        actual_twist=contact['foot_twist'][tick*10-1,case]
        model_twist=np.asarray([pin.getFrameJacobian(model.model,model.data,model.body_frames[n],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)@state.v for n in model.feet])
        row=dict(case=case,control_tick=tick,baseline=before,predicted=False,
            no_nonfoot_ground_or_net_interval=bool(ground['interval_peak_normal_n'][tick,case,non].max()==0 and contact['net_peak'][tick,case].max()==0),
            initial_foot_twist_maximum_error=float(np.max(abs(model_twist-actual_twist))),
            initial_front_hip_rad=state.joint_position[:2],measured_friction_impulse_ns=contact['friction_impulse'][tick,case].sum(0),
            measured_normal_impulse_ns=contact['normal_impulse'][tick,case].sum(0))
        response=RecordingResponse(model)
        try:
            pred=predict_self_contact_spheres(model,c.qp.interface,state,d['final'][i],w['spheres'],
                geometry=w['geometry'],substeps=160,collision_response=response)
            y=pred['state'];actual_q=d['q'][j];actual_v=d['v'][j]
            rp=pin.Quaternion(y.q[3:7]).matrix();ra=pin.Quaternion(actual_q[3:7]).matrix()
            difference=pin.difference(model.model,actual_q,y.q)
            impulse=np.asarray(response.impulses).sum(0).reshape(4,3).sum(0)
            row.update(predicted=True,position_error_m=float(np.linalg.norm(y.q[:3]-actual_q[:3])),
                attitude_error_rad=float(np.linalg.norm(difference[3:6])),
                linear_velocity_error_m_s=float(np.linalg.norm(rp@y.v[:3]-ra@actual_v[:3])),
                angular_velocity_error_rad_s=float(np.linalg.norm(rp@y.v[3:6]-ra@actual_v[3:6])),
                maximum_joint_velocity_error_rad_s=float(np.max(abs(y.v[model.v_indices]-actual_v[model.v_indices]))),
                predicted_ground_impulse_ns=impulse,
                measured_ground_impulse_error_ns=impulse-row['measured_friction_impulse_ns']-row['measured_normal_impulse_ns'],
                diagnostics=pred['diagnostics'])
        except (ValueError,np.linalg.LinAlgError) as e:row['model_rejection']=repr(e)
        records.append(row)
    paired=[r for r in records if r['predicted'] and r['baseline']['predicted']]
    clean=[r for r in paired if r['no_nonfoot_ground_or_net_interval']]
    def summarize(rows):
        keys=['angular_velocity_error_rad_s','maximum_joint_velocity_error_rad_s','linear_velocity_error_m_s']
        return dict(cases=len(rows),errors={k:dict(
            before_median=float(np.median([r['baseline'][k] for r in rows])),
            after_median=float(np.median([r[k] for r in rows])),
            before_maximum=max(r['baseline'][k] for r in rows),after_maximum=max(r[k] for r in rows)) for k in keys}) if rows else dict(cases=0)
    paths=[Path(__file__),a.baseline,a.model_inputs,a.run/'pure_control_commands.npz',a.run/'landing_contact_detail.npz',
        ROOT/'scripts/hip_mount_collision_response_v1.py',ROOT/'scripts/predict_self_contact_spheres_v2.py',
        ROOT/'configs/go2_piper_hip_mount_boundaries_v1.json']
    result=dict(records=records,paired=summarize(paired),clean_paired=summarize(clean),
        prediction_returns=sum(r['predicted'] for r in records),attempts=len(records),wall_seconds=time.monotonic()-start,
        native_execution=False,actuator_parameters_changed=False,friction_coefficient_changed=False,
        added_response='Audited hip/mount internal unilateral impulse; zero restitution; 10ms penetration decay',
        full_recovery_success_not_inferred=True,
        source_sha256={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']}))


if __name__=='__main__':main()
