#!/usr/bin/env python3
"""Prepare unexecuted body/face/recovery controls around a native power shot."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_teacher_v4 import LOW,HIGH,PARAMETER_NAMES
from quadsmash.control.full_racket_arm_search_v4 import VELOCITY_PARAMETER_NAMES
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_candidate_bank_v4 import validate_candidate_bank


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--contact',type=Path,required=True)
    p.add_argument('--recovery-bank',type=Path,required=True)
    p.add_argument('--native-recovery-population',type=Path,help='Optional completed native population supplying two additional recovered body proposals')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh output directory')
    source=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    reference=source['parameters'].numpy().copy()
    if reference.shape!=(32,):raise ValueError('Native32-parameter power reference required')
    event=json.loads(a.contact.read_text());body=json.loads(a.recovery_bank.read_text())
    assert event['string_contact'] and event['first_racket_contact']
    assert source['asset_sha256']==body['asset_sha256'] and source['seed']==body['seed']
    assert source['seed'].get('causal_arm_prepose_gain',0.)==0.
    low=np.array(LOW+[.8,-.12]+[-5.]*6,dtype=np.float32)
    high=np.array(HIGH+[1.45,.12]+[5.]*6,dtype=np.float32)
    low[22:24]=-.3;high[22:24]=.3
    names=PARAMETER_NAMES+['intercept_plane_x','strike_timing_residual']+VELOCITY_PARAMETER_NAMES
    assert source['parameter_names']==names
    poses=np.asarray(event['body_link_poses_local'])
    root_rotation=Rotation.from_quat(poses[0,[4,5,6,3]]).as_matrix()
    racket_rotation=Rotation.from_quat(poses[27,[4,5,6,3]]).as_matrix()
    contact_offset=racket_rotation.T@(np.asarray(event['point_m'])-poses[27,:3])
    actual=np.asarray(event['joint_position_rad'][-6:])
    kin=FullRacketKinematicsV4().to(dtype=torch.float64)
    with torch.inference_mode():
        point,n,jp,jn=[x[0].numpy() for x in kin(torch.tensor(actual[None]),torch.tensor(contact_offset))]
    reconstruction=np.linalg.norm(poses[0,:3]+root_rotation@point-event['point_m'])
    assert reconstruction<1e-4
    jp,jn=root_rotation@jp,root_rotation@jn
    normal=root_rotation@n
    desired=np.asarray(source['seed']['outward_contact_normal_world']);desired/=np.linalg.norm(desired)
    model=FullRacketGeometryV4()
    dl=np.maximum.reduce([low[8:14]-reference[8:14],model.bounds[:,0]-actual,np.full(6,-.12)])
    du=np.minimum.reduce([high[8:14]-reference[8:14],model.bounds[:,1]-actual,np.full(6,.12)])
    assert np.all(du>dl)
    fit=lsq_linear(np.r_[5*jp,jn,.04*np.eye(6)],np.r_[np.zeros(3),desired-normal,np.zeros(6)],
                   bounds=(dl,du),tol=1e-12,max_iter=500)
    assert fit.success
    delta=fit.x
    proposals=[];labels=[]
    def add(value,**label):
        proposals.append(np.clip(value,low,high).astype(np.float32));labels.append(label)
    for waypoint,speed in [(.75,float(reference[0])),(.90,.60),(1.05,.65)]:
        for velocity_gain in [float(reference[7]),1.]:
            for extension in [False,True]:
                for phase in [0.,-.06]:
                    row=reference.copy();row[1]=waypoint;row[0]=speed;row[7]=velocity_gain;row[2]+=phase
                    if extension:row[5]-=.06;row[6]+=.08
                    add(row,kind='body_preparation',waypoint_x_m=waypoint,walk_command_mps=speed,
                        arm_velocity_gain=velocity_gain,push_extension=extension,leg_phase_delta_s=phase)
    for waypoint in [.80,.95]:
        for lead in [-.02,0.,.02]:
            for fraction in [.5,1.]:
                row=reference.copy();row[1]=waypoint;row[14]+=lead;row[8:14]+=fraction*delta
                add(row,kind='body_and_face',waypoint_x_m=waypoint,arm_lead_delta_s=lead,face_correction_fraction=fraction)
    body_indices=[0,1,2,3,4,5,6,17,20,21,22,23]
    bodies=[('historical_paired',np.asarray(body['parameters']))]
    native_recovery_sources=[]
    if a.native_recovery_population:
        native=torch.load(a.native_recovery_population,map_location='cpu',weights_only=True)
        assert native['asset_sha256']==source['asset_sha256'] and native['controller_template']['seed']==source['seed']
        recovered=sorted({r['candidate_id'] for r in native['rows'] if r['recovery_success'] and not r['numerical_failures']})
        assert len(recovered)>=2
        for index in recovered[:2]:
            bodies.append((f'current_native_{index}',native['parameters'][index,:26].numpy()))
            native_recovery_sources.append(dict(candidate_id=index,recovered_training_indices=[r['source_training_index'] for r in native['rows']
                if r['candidate_id']==index and r['recovery_success']],performance_inherited=False))
    if len(bodies)==3:
        variants=[(name,measured,fraction,waypoint) for name,measured in bodies for fraction in [.5,1.] for waypoint in [.80,.95]]
    else:
        variants=[(bodies[0][0],bodies[0][1],fraction,waypoint) for fraction in [.25,.5,.75,1.] for waypoint in [.75,.90,1.05]]
    for name,measured,fraction,waypoint in variants:
        row=reference.copy();row[body_indices]=(1-fraction)*reference[body_indices]+fraction*measured[body_indices]
        row[1]=waypoint
        add(row,kind='power_arm_recovery_body',body_source=name,measured_body_fraction=fraction,waypoint_x_m=waypoint)
    for waypoint,speed in [(.80,.60),(.95,.65)]:
        for velocity_gain in [float(reference[7]),1.]:
            for fraction in [0.,.5,1.]:
                row=reference.copy();row[1]=waypoint;row[0]=speed;row[7]=velocity_gain
                row[24]=1.25;row[2]-=.02;row[5]-=.06;row[6]+=.08;row[8:14]+=fraction*delta
                add(row,kind='combined_high_interception',waypoint_x_m=waypoint,interception_plane_x_m=1.25,
                    arm_velocity_gain=velocity_gain,face_correction_fraction=fraction,leg_phase_delta_s=-.02,push_extension=True)
    inputs=[a.checkpoint,a.contact,a.recovery_bank,Path(__file__)]+([a.native_recovery_population] if a.native_recovery_population else [])
    provenance={str(path.resolve()):sha(path) for path in inputs}
    bank=dict(schema='full-racket-candidate-bank-v4',asset_sha256=source['asset_sha256'],seed=source['seed'],
              parameter_names=names,reference_parameters=reference.tolist(),parameters=np.asarray(proposals).tolist(),
              proposal_labels=labels,source_provenance=provenance,native_execution=False,formal_acceptance=False,
              schedule='generation0only; subsequent generations use all available Gaussian/CEM candidates')
    checked=validate_candidate_bank(bank,source['seed'],source['asset_sha256'],names,reference,low,high,62)
    assert checked.shape==(60,32)
    warm=copy.deepcopy(source);warm.pop('training_result',None)
    warm.update(kind='Unexecuted body/face bank initialization from an actual native power candidate; no inherited fitness.',
                native_execution=False,formal_acceptance=False,performance_inherited=False,initialization_provenance=provenance)
    a.output.mkdir(parents=True)
    torch.save(warm,a.output/'policy.pt')
    (a.output/'CANDIDATE_BANK.json').write_text(json.dumps(bank,indent=2)+'\n')
    with torch.inference_mode():
        moved_point,moved_normal,_,_=kin(torch.tensor((actual+delta)[None]),torch.tensor(contact_offset))
    actual_normal_after=root_rotation@moved_normal[0].numpy()
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='CPU-only bounded body,face and recovery proposals; no new dynamics or inherited outcome.',
                candidate_count=64,reference_candidates=2,first_generation_bank=60,first_generation_gaussian=2,
                later_generations_gaussian=62,body_preparation_proposals=24,face_proposals=12,recovery_body_proposals=12,
                combined_high_interception_proposals=12,interception_planes_x_m=[1.25,1.40],
                original32_reference_parameters_preserved=bool(np.array_equal(warm['parameters'].numpy(),reference)),
                source_contact_reconstruction_error_m=float(reconstruction),offline_face_joint_delta_rad=delta.tolist(),
                recorded_normal_world=normal.tolist(),offline_corrected_normal_world=actual_normal_after.tolist(),
                offline_face_contact_shift_m=float(np.linalg.norm(moved_point[0].numpy()-point)),
                waypoint_range_m=[.75,1.05],maximum_push_coordinate_change_rad=.08,
                native_recovery_sources=native_recovery_sources,velocity_directions_coherent=True,
                unchanged_physical_joint_velocity_torque_limits=True,unchanged_physics_and_success_gates=True,
                policy_sha256=sha(a.output/'policy.pt'),bank_sha256=sha(a.output/'CANDIDATE_BANK.json'),
                sources=provenance,native_execution=False,formal_acceptance=False)
    (a.output/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
