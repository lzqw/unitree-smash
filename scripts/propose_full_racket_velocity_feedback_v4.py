#!/usr/bin/env python3
"""Offline bounded velocity proposals from actual native strings contacts.

The measured body motion is held fixed in this local calculation. These are
controller proposals, not executed velocities or coupled feasibility evidence.
"""
import argparse,hashlib,io,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();output=a.output.resolve()
    if output.exists():raise ValueError('Use a fresh proposal directory')
    progress=json.loads((run/'progress.json').read_text());generation=progress['generation']
    raw=(run/'contacts.partial.json').read_bytes();state_raw=(run/'partial_search_state.pt').read_bytes()
    state=torch.load(io.BytesIO(state_raw),map_location='cpu',weights_only=False)
    assert state['generation']==generation==json.loads((run/'progress.json').read_text())['generation']
    events=[x for x in json.loads(raw) if x['episode_index']==0 and x['first_racket_contact'] and x['string_contact']]
    kin=FullRacketKinematicsV4();bounds=FullRacketGeometryV4().bounds;rows=[]
    for event in events:
        q=np.asarray(event['joint_position_rad'][-6:]);qd=np.asarray(event['joint_velocity_rad_s'][-6:])
        poses=np.asarray(event['body_link_poses_local']);root=Rotation.from_quat(poses[0,3:][[1,2,3,0]]).as_matrix()
        racket=Rotation.from_quat(poses[-1,3:][[1,2,3,0]]).as_matrix()
        offset=racket.T@(np.array(event['point_m'])-poses[-1,:3])
        point,normal,jp,jn=[x[0].numpy() for x in kin(torch.tensor(q[None]),torch.tensor(offset))]
        pose_error=float(np.linalg.norm(poses[0,:3]+root@point-event['point_m']))
        assert pose_error<1e-5
        jp=root@jp;jn=root@jn;n=np.asarray(event['face_normal']);velocity=np.asarray(event['point_velocity_mps'])
        arm=jp@qd;transport=velocity-arm;omega=np.asarray(event['root_angular_velocity_rad_s'])
        rotation=np.cross(omega,np.asarray(event['point_m'])-poses[0,:3]);translation=transport-rotation
        normal_jacobian=n@jp;goal_normal_speed=1.2
        # Preserve current world face direction while seeking forward normal
        # velocity. Native source joints and the original5rad/s arm limit apply.
        matrix=np.r_[3*normal_jacobian[None],.3*jn,.03*np.eye(6)]
        rhs=np.r_[3*(goal_normal_speed-n@transport),-.3*np.cross(omega,n),.03*qd]
        lower=np.maximum(-5.,(bounds[:,0]-q)/.08)
        upper=np.minimum(5.,(bounds[:,1]-q)/.08)
        assert np.all(upper>lower)
        fit=lsq_linear(matrix,rhs,bounds=(lower,upper),tol=1e-12,max_iter=1000)
        assert fit.success
        proposal_velocity=transport+jp@fit.x;normal_rate=np.cross(omega,n)+jn@fit.x
        un_contribution=normal_jacobian*qd
        max_arm=float(np.sum(np.where(normal_jacobian>=0,normal_jacobian*upper,normal_jacobian*lower)))
        rows.append(dict(env_id=event['env_id'],candidate_id=event['env_id']//2,generation=generation,
            time_s=event['time_s'],native_contact_point_m=event['point_m'],native_face_normal_world=event['face_normal'],
            native_point_velocity_mps=velocity.tolist(),native_joint_velocity_rad_s=qd.tolist(),
            actual_arm_relative_normal_contributions_mps=un_contribution.tolist(),
            arm_relative_normal_speed_mps=float(n@arm),body_rotation_normal_speed_mps=float(n@rotation),
            root_translation_normal_speed_mps=float(n@translation),actual_total_normal_speed_mps=float(n@velocity),
            reconstructed_contact_position_error_m=pose_error,local_goal_normal_speed_mps=goal_normal_speed,
            proposed_joint_velocity_rad_s=fit.x.tolist(),local_proposed_point_velocity_mps=proposal_velocity.tolist(),
            local_proposed_normal_speed_mps=float(n@proposal_velocity),local_proposed_world_normal_rate_rad_s=normal_rate.tolist(),
            proposal_joint_position_lookahead_s=.08,proposal_joint_target_rad=(q+.08*fit.x).tolist(),
            fixed_state_max_normal_speed_from_joint_box_mps=float(n@transport)+max_arm,
            weighted_solver_residual=float(np.linalg.norm(matrix@fit.x-rhs)),native_execution_of_proposal=False))
    output.mkdir(parents=True)
    (output/'SOURCE_CONTACTS.json').write_bytes(raw);(output/'SOURCE_PARTIAL_SEARCH_STATE.pt').write_bytes(state_raw)
    report=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='Offline fixed-recorded-state kinematic velocity feedback proposals only, not native policy performance.',
        source_run=str(run.relative_to(ROOT)),generation=generation,observed_progress=progress,
        source_contacts_sha256=hashlib.sha256(raw).hexdigest(),source_search_state_sha256=hashlib.sha256(state_raw).hexdigest(),
        rows=rows,formal_acceptance=False,
        limitations=['Measured body angular/linear motion is held fixed; changing arm motion changes the coupled body response.',
            'A locally solvable joint-velocity target does not establish actuator tracking, legal earlier contact, net clearance, first target landing or recovery.',
            'Joint target lookahead uses unchanged source joint position bounds and5rad/s arm speed; all native torque limits remain required.',
            'Only actual first-contact strings examples are included; same-stroke frame-to-strings contacts are outside this diagnostic subset.'],
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'PLAN.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(examples=len(rows),rows=[{k:r[k] for k in ['env_id','arm_relative_normal_speed_mps','body_rotation_normal_speed_mps','actual_total_normal_speed_mps','local_proposed_normal_speed_mps','proposed_joint_velocity_rad_s']} for r in rows])),flush=True)


if __name__=='__main__':main()
