"""Check directly recorded foot friction against native linear momentum.

Uses each run's own consecutive native states, not cross-run identity. Clean
interval selection is retrospective evidence filtering, never a control gate.
Contact-force/point telemetry does not certify a stable landing or identify a
unique cause of any model mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1


def read_arrays(path,keys=None):
    with np.load(path) as z:return {k:z[k].copy() for k in (keys or z.files)}


def interval_balance(before,after,mass,normal,friction,dt=.01):
    external=np.asarray(after)-np.asarray(before)-np.array([0.,0.,-mass*9.81*dt])
    normal=np.asarray(normal).sum(0);friction=np.asarray(friction).sum(0)
    return dict(external_impulse_from_momentum_ns=external.tolist(),normal_impulse_ns=normal.tolist(),
        measured_friction_impulse_ns=friction.tolist(),
        measured_force_balance_residual_ns=(external-normal-friction).tolist(),
        reversed_friction_sign_residual_ns=(external-normal+friction).tolist(),
        normal_only_residual_ns=(external-normal).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();read=lambda name:json.loads((a.run/name).read_text())
    if a.output.exists():raise ValueError('Fresh output required')
    meta=read('landing_contact_detail.json');base=read('pure_control_ground_forces.json')
    if (not meta['complete'] or not base['complete'] or meta['maximum_points_per_pair']>8
        or read('job.json')['status']!='SUCCEEDED'):
        raise ValueError('Full native evidence and admitted telemetry required')
    data=read_arrays(a.run/'pure_control_commands.npz')
    detail=read_arrays(a.run/'landing_contact_detail.npz')
    ground=read_arrays(a.run/'pure_control_ground_forces.npz',['interval_peak_normal_n'])
    names=base['body_names'];non=[i for i,n in enumerate(names) if not n.endswith('_foot')]
    diagnostics=[json.loads(line) for line in (a.run/'pure_control_diagnostics.jsonl').read_text().splitlines()]
    model=LoadedPinocchioModelV1();mass=sum(i.mass for i in model.model.inertias)
    records=[];excluded=dict(no_consecutive_state=0,no_sampled_support=0,nonfoot_ground=0,robot_net=0)
    for lane in sorted(set(data['lane'].tolist())):
        ids=np.flatnonzero(data['lane']==lane)
        # One first clean supported interval per case, not a chosen success.
        for i,j in zip(ids[:-1],ids[1:]):
            tick=int(data['tick'][i])
            if data['tick'][j]!=tick+1:excluded['no_consecutive_state']+=1;continue
            if not any(diagnostics[i]['phase']['ground_contacts']):excluded['no_sampled_support']+=1;continue
            if ground['interval_peak_normal_n'][tick,lane,non].max()>0:excluded['nonfoot_ground']+=1;continue
            if detail['net_peak'][tick,lane].max()>0:excluded['robot_net']+=1;continue
            h0=model.snapshot(data['q'][i],data['v'][i])['momentum_world'][:3].copy()
            h1=model.snapshot(data['q'][j],data['v'][j])['momentum_world'][:3].copy()
            normal=detail['normal_impulse'][tick,lane];friction=detail['friction_impulse'][tick,lane]
            sl=slice(tick*10,(tick+1)*10)
            pose=detail['foot_pose'][sl,lane];twist=detail['foot_twist'][sl,lane]
            point=detail['friction_point'][sl,lane];force=detail['friction_force'][sl,lane]
            point_velocity=twist[:,:,:3]+np.cross(twist[:,:,3:],point-pose[:,:,:3])
            loaded=detail['friction_count'][sl,lane]>0
            rows=dict(case=int(lane),control_tick=tick,model_mass_kg=mass,
                sampled_contacts=diagnostics[i]['phase']['ground_contacts'],
                normal_impulse_per_foot_ns=normal.tolist(),friction_impulse_per_foot_ns=friction.tolist(),
                **interval_balance(h0,h1,mass,normal,friction),
                millisecond_force_n=force.tolist(),millisecond_friction_points_m=point.tolist(),
                millisecond_friction_counts=detail['friction_count'][sl,lane].tolist(),
                millisecond_end_frame_point_velocity_m_s=np.where(loaded[:,:,None],point_velocity,0.).tolist(),
                approximate_force_dot_end_velocity_w=np.where(loaded,(force*point_velocity).sum(-1),0.).tolist(),
                force_dot_velocity_is_not_integrated_contact_work=True,
                clean_native_ground_and_net_interval=True,stability_not_inferred=True)
            records.append(rows);break
    residual=[np.linalg.norm(r['measured_force_balance_residual_ns']) for r in records]
    before=[np.linalg.norm(r['normal_only_residual_ns']) for r in records]
    result=dict(records=records,clean_cases=len(records),excluded_intervals=excluded,
        maximum_normal_view_difference_n=meta['maximum_normal_view_difference_n'],
        momentum_residual_ns=dict(median=float(np.median(residual)),maximum=max(residual)) if residual else None,
        normal_only_residual_ns=dict(median=float(np.median(before)),maximum=max(before)) if before else None,
        native_friction_directly_measured=True,control_logic_changed=False,training=False,
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path(__file__),a.run/'pure_control_commands.npz',a.run/'landing_contact_detail.npz',
             a.run/'pure_control_ground_forces.npz',a.run/'pure_control_diagnostics.jsonl']})
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','source_sha256']}))


if __name__=='__main__':main()
