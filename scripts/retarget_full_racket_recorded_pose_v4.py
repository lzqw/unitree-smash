#!/usr/bin/env python3
"""Reproducible CPU teacher proposal from an immutable TRAIN state snapshot."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.optimize import least_squares,lsq_linear
from scipy.spatial.transform import Rotation
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--uv-z',type=float,default=.09);p.add_argument('--normal-z',type=float,default=-.08)
    p.add_argument('--point-vx',type=float,default=3.2);p.add_argument('--point-vz',type=float,default=-.3)
    p.add_argument('--q1-low',type=float,default=-1.);p.add_argument('--q1-high',type=float,default=0.)
    p.add_argument('--fixed-leg-clock',action='store_true');p.add_argument('--starts',type=int,default=24)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output path')
    source=json.loads(a.snapshot.read_text());obs=np.array(source['public_observation'])
    pose=np.array(source['body_link_poses'])[0];rot=Rotation.from_quat(pose[3:][[1,2,3,0]])
    root_rpy=rot.as_euler('xyz');target=np.array(source['ball_position_m'])
    reset=np.array(json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad'])
    actual=reset+obs[21:27];model=FullRacketGeometryV4();bounds=model.bounds.copy()
    if not model.bounds[0,0]<=a.q1_low<a.q1_high<=model.bounds[0,1]:
        raise ValueError('Pose-search interval must stay inside physical bounds')
    if a.starts<1:raise ValueError('At least one pose search start required')
    bounds[0]=[a.q1_low,a.q1_high]
    normal=np.array([1.,0.,a.normal_z]);normal/=np.linalg.norm(normal)
    def residual(q):
        point,n=model.contact_center(q,uv=(0,a.uv_z),face=1,base_position=pose[:3],base_rpy=root_rpy)
        return np.r_[5*(point-target),n-normal,.001*(q-actual)]
    rng=np.random.default_rng(725603);rows=[]
    starts=[actual,*rng.uniform(bounds[:,0]+1e-4,bounds[:,1]-1e-4,(a.starts-1,6))]
    for start in starts:
        fit=least_squares(residual,np.clip(start,bounds[:,0]+1e-5,bounds[:,1]-1e-5),
            bounds=(bounds[:,0]+1e-5,bounds[:,1]-1e-5),max_nfev=700)
        point,n=model.contact_center(fit.x,uv=(0,a.uv_z),face=1,base_position=pose[:3],base_rpy=root_rpy)
        error=float(np.linalg.norm(point-target));normal_error=float(np.linalg.norm(n-normal))
        rows.append(dict(joint_position_rad=fit.x.tolist(),position_error_m=error,normal_error=normal_error,
            distance_to_recorded_joint_pose_rad=float(np.linalg.norm(fit.x-actual)),score=error+.2*normal_error))
    good=[row for row in rows if row['position_error_m']<.002 and row['normal_error']<.02]
    chosen=min(good,key=lambda x:x['distance_to_recorded_joint_pose_rad']) if good else min(rows,key=lambda x:x['score'])
    q=np.array(chosen['joint_position_rad']);kinematics=FullRacketKinematicsV4()
    point,n,jp,jn=[value[0].numpy() for value in kinematics(torch.tensor(q[None]),
        torch.tensor([.0256,0.,a.uv_z],dtype=torch.float64))]
    requested=np.array([a.point_vx,0.,a.point_vz])
    relative=rot.inv().apply(requested)-obs[:3]-np.cross(obs[3:6],point)
    desired_normal_rate=-np.cross(obs[3:6],n)
    jacobian=np.r_[jp,.3*jn];rhs=np.r_[relative,.3*desired_normal_rate]
    room=np.minimum(q-model.bounds[:,0],model.bounds[:,1]-q)
    maximum=np.maximum(np.minimum(4.8,room/.08),1e-5)
    velocity=lsq_linear(np.r_[jacobian,.02*np.eye(6)],np.r_[rhs,np.zeros(6)],bounds=(-maximum,maximum)).x
    projected=rot.apply(obs[:3]+np.cross(obs[3:6],point)+jp@velocity)
    normal_rate=rot.apply(np.cross(obs[3:6],n)+jn@velocity)
    seed=dict(chosen,joint_position_rad=q.tolist(),joint_velocity_seed_rad_s=velocity.tolist(),string_uv_m=[0.,a.uv_z],
        face_sign=1,planned_root_position_m=pose[:3].tolist(),planned_root_rpy_rad=root_rpy.tolist(),
        desired_ball_center_m=target.tolist(),outward_contact_normal_world=normal.tolist(),
        original_velocity_solver_request_mps=requested.tolist(),desired_point_velocity_mps=projected.tolist(),
        predicted_point_velocity_mps=projected.tolist(),predicted_normal_rate_world_rad_s=normal_rate.tolist(),
        nominal_projected_pitch_rad=float(np.arctan2(obs[6],-obs[8])),
        leg_event_time_gain=0. if a.fixed_leg_clock else 1.)
    result=dict(schema='full-racket-ik-seeds-v4',kind='CPU TRAIN proposal only, not native contact/recovery or dynamic feasibility evidence.',
        seeds=[seed],all_search_results=rows,source_snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),
        source=dict(run=source['source_run'],generation=source['generation'],env_id=source['env_id'],time_s=source['recorded_clock_s']),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        pose_search_bounds_rad=bounds.tolist(),physical_changes=False,formal_acceptance=False,
        limitations=['Uses measured TRAIN base motion only to initialize an offline teacher. Runtime still consumes public observations only.',
            'Joint velocity projection is local kinematics, not prediction of the changed whole-body reaction.',
            'Source recovery labels are preserved in the snapshot and do not transfer to the changed teacher. New native rollout must establish every outcome.',
            'Original actual outgoing8m/s/downward/overnet/landing/recovery requirements are unchanged.'])
    a.output.mkdir(parents=True)
    (a.output/'TRAIN_SOURCE_SNAPSHOT.json').write_bytes(a.snapshot.read_bytes())
    (a.output/'seeds.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(position_error_m=chosen['position_error_m'],normal_error=chosen['normal_error'],
        projected_point_velocity_mps=projected.tolist(),dynamic_execution=False)),flush=True)


if __name__=='__main__':main()
