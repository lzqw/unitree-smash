"""Fixed-root standard-PiPER IK seeds at actual contact/earlier ball samples.

No native execution, root motion, actuator trajectory, collision or recovery
proof. Unsuccessful bounded local optimization is not an unreachable proof.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[3]
MAIN = ROOT.parent/'unitree_smash'
HERE = Path(__file__).resolve().parent
RUN = MAIN/'reports/G07_full_racket_v4/residual_ppo_centered_leg06_after075_intermediates_n2048_probe8_fixed_train128_01'
BASE = MAIN/'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01/policy.pt'
sys.path.insert(0,str(MAIN))
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    assert not path.exists()
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def joint_angles(poses,names,geometry):
    rotations = Rotation.from_quat(poses[:,[4,5,6,3]]).as_matrix()
    result = []
    for i,(_,origin) in enumerate(geometry.joints):
        parent = names.index('base_link' if i == 0 else f'link{i}')
        child = names.index(f'link{i+1}')
        turn = origin.T@rotations[parent].T@rotations[child]
        angle = np.arctan2(turn[1,0],turn[0,0])
        options = [angle+2*np.pi*k for k in [-1,0,1]]
        inside = [x for x in options if geometry.bounds[i,0]-1e-4 <= x <= geometry.bounds[i,1]+1e-4]
        assert len(inside) == 1
        result.append(inside[0])
    return np.asarray(result)


@torch.inference_mode()
def fit(model,starts,position,normal,limits,offset):
    q = starts.clamp(limits[:,0],limits[:,1]);eye = torch.eye(6,dtype=torch.float64)[None]
    for _ in range(140):
        p,n,jp,jn = model(q,offset)
        jacobian = torch.cat([jp,.3*jn],1); error = torch.cat([position-p,.3*(normal-n)],1)
        delta = (jacobian.transpose(1,2)@torch.linalg.solve(jacobian@jacobian.transpose(1,2)+.04**2*eye,error[:,:,None])).squeeze(-1)
        delta = delta.clamp(-.12,.12)
        best_q,best_cost = q, error.square().sum(1)
        for scale in [1.,.5,.2]:
            candidate = (q+scale*delta).clamp(limits[:,0],limits[:,1]);cp,cn,*_ = model(candidate,offset)
            cost = (position-cp).square().sum(1)+.09*(normal-cn).square().sum(1)
            better = cost < best_cost;best_q = torch.where(better[:,None],candidate,best_q)
            best_cost = torch.where(better,cost,best_cost)
        q = best_q
    return q


def main():
    sources = [Path(__file__),HERE/'EARLIER_POINT_RESULT.json',RUN/'contacts.json',RUN/'all_first_episodes.npz',
        RUN/'geometry.json',BASE,MAIN/'quadsmash/control/full_racket_geometry_v4.py',
        MAIN/'quadsmash/control/full_racket_kinematics_v4.py',MAIN/'configs/assembly_v2.json',
        MAIN/'configs/full_racket_v4_r3.json',MAIN/'quadsmash/physics/full_racket_v4.py',
        MAIN/'quadsmash/control/piper_geometry.py']
    geometry = FullRacketGeometryV4();sources.append(geometry.urdf)
    fingerprints = {str(p):sha(p) for p in sources}
    save(HERE/'ARM_REACH_PLAN.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256=fingerprints,
        FK_position_max_error_m=.0001,FK_normal_max_error_rad=.001,joint_reconstruction_max_error_rad=.0002,
        IK_seed_position_tolerance_m=.02,IK_seed_normal_tolerance_deg=10.,iterations=140,starts=3,
        original_hardware_joint_bounds=True,original5rad_s_used_only_as_necessary_travel_time_lower_bound=True,
        root='Each actually recorded root pose stays fixed; no root pose optimization or writes.',
        contact_point='The unchanged teacher string_uv,not an enlarged face or extended arm.',
        scope=__doc__,new_native_execution=False,formal_acceptance=False,goal_complete=False))
    base = torch.load(BASE,map_location='cpu',weights_only=False)
    offset = torch.tensor([.0256,*base['seed']['string_uv_m']],dtype=torch.float64)
    names = read(RUN/'geometry.json')['body_names'];racket_id = names.index('Racket');base_id = names.index('base')
    contacts = {x['env_id']:x for x in read(RUN/'contacts.json') if x['first_racket_contact']}
    earlier = read(HERE/'EARLIER_POINT_RESULT.json')['actual_earlier_ball_and_racket_samples']
    samples,angle_errors = [],[]
    with np.load(RUN/'all_first_episodes.npz') as arrays:
        for item in earlier:
            index = item['TRAIN_index']; event = contacts[index]
            records = [('actual_contact',np.asarray(event['body_link_poses_local']),np.asarray(event['ball_position_m']))]
            for number,previous in enumerate(item['earlier_samples']):
                ids = np.flatnonzero(arrays['valid_first_episode'][:,index].astype(bool)
                    & np.isclose(arrays['episode_clock_s'][:,index],previous['actual_time_s'],atol=1e-6,rtol=0))
                assert len(ids) == 1
                records.append((f'earlier{number+1}',arrays['link_poses'][ids[0],index].astype(float),np.asarray(previous['actual_ball_position_m'])))
            for label,poses,ball in records:
                angles = joint_angles(poses,names,geometry)
                if label == 'actual_contact':angle_errors.append(float(np.max(abs(angles-np.asarray(event['joint_position_rad'])[12:]))))
                rotation = Rotation.from_quat(poses[base_id,[4,5,6,3]]).as_matrix()
                nr = Rotation.from_quat(poses[racket_id,[4,5,6,3]]).as_matrix()[:,0]
                samples.append(dict(TRAIN_index=index,label=label,angles=angles,root_position=poses[base_id,:3],
                    root_rotation=rotation,actual_tool_center=poses[racket_id,:3],actual_normal=nr,ball=ball))
    model = FullRacketKinematicsV4().to(dtype=torch.float64)
    measured = torch.from_numpy(np.asarray([x['angles'] for x in samples]))
    with torch.inference_mode(): center,normal,*_ = model(measured)
    predicted_p = np.asarray([x['root_position']+x['root_rotation']@p for x,p in zip(samples,center.numpy())])
    predicted_n = np.asarray([x['root_rotation']@p for x,p in zip(samples,normal.numpy())])
    actual_p,actual_n = np.asarray([x['actual_tool_center'] for x in samples]),np.asarray([x['actual_normal'] for x in samples])
    fk_p_error = float(np.linalg.norm(predicted_p-actual_p,axis=1).max())
    fk_n_error = float(np.arccos(np.clip((predicted_n*actual_n).sum(1),-1,1)).max())
    valid = fk_p_error <= .0001 and fk_n_error <= .001 and max(angle_errors) <= .0002
    save(HERE/'ARM_ACTUAL_FK_CHECK.json',dict(actual_native_pose_samples=len(samples),FK_position_max_error_m=fk_p_error,
        FK_normal_max_error_rad=fk_n_error,joint_reconstruction_max_error_rad=max(angle_errors),passed=valid,
        new_native_execution=False,formal_acceptance=False))
    assert valid,'Preserve actual geometry mismatch before using IK'
    seed_normal = np.asarray(base['seed']['outward_contact_normal_world']);a=float(base['parameters'][37])
    requested = np.array([np.cos(a)*seed_normal[0]+np.sin(a)*seed_normal[2],seed_normal[1],
        -np.sin(a)*seed_normal[0]+np.cos(a)*seed_normal[2]])
    source_q = np.asarray(base['seed']['joint_position_rad'])+base['parameters'][8:14].numpy()
    limits = torch.from_numpy(geometry.bounds);entries = []
    for normal_mode in ['recorded_contact_normal','unchanged_teacher_requested_normal']:
        targets = np.asarray([x['root_rotation'].T@(x['ball']-x['root_position']) for x in samples])
        target_normals = np.asarray([x['root_rotation'].T@(contacts[x['TRAIN_index']]['face_normal'] if normal_mode == 'recorded_contact_normal' else requested) for x in samples])
        target_normals /= np.linalg.norm(target_normals,axis=1,keepdims=True)
        starts = torch.cat([measured,torch.tensor(source_q).repeat(len(samples),1),limits.mean(1).repeat(len(samples),1)])
        target = torch.from_numpy(np.tile(targets,(3,1)));tn = torch.from_numpy(np.tile(target_normals,(3,1)))
        q = fit(model,starts,target,tn,limits,offset)
        with torch.inference_mode(): point,n,*_ = model(q,offset)
        pe = (point-target).norm(dim=1).numpy().reshape(3,len(samples))
        ne = torch.acos((n*tn).sum(1).clamp(-1,1)).numpy().reshape(3,len(samples))
        cost = pe**2+.09*ne**2;chosen = cost.argmin(0);q=q.numpy().reshape(3,len(samples),6)
        for i,sample in enumerate(samples):
            row=int(chosen[i]);position_error=float(pe[row,i]);normal_error=float(np.rad2deg(ne[row,i]))
            entries.append(dict(TRAIN_index=sample['TRAIN_index'],sample=sample['label'],normal_mode=normal_mode,
                position_error_m=position_error,normal_error_deg=normal_error,
                local_IK_seed_found=position_error<=.02 and normal_error<=10.,
                joint_seed_rad=q[row,i].tolist(),max_joint_change_from_recorded_rad=float(np.max(abs(q[row,i]-sample['angles']))),
                joint_travel_time_at5rad_s_lower_bound_s=float(np.max(abs(q[row,i]-sample['angles']))/5.)))
    assert all(sha(Path(name)) == digest for name,digest in fingerprints.items())
    counts = {mode:{label:sum(x['local_IK_seed_found'] for x in entries if x['normal_mode']==mode and x['sample']==label)
        for label in ['actual_contact','earlier1','earlier2']} for mode in ['recorded_contact_normal','unchanged_teacher_requested_normal']}
    save(HERE/'ARM_REACH_RESULT.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),counts=counts,entries=entries,
        source_sha256=fingerprints,root_pose_fixed=True,original_joint_bounds_preserved=True,
        collision_dynamic_torque_tracking_impact_and_recovery_unverified=True,local_failure_not_unreachable_proof=True,
        new_native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False))
    print(json.dumps(dict(FK_position_max_error_m=fk_p_error,counts=counts)),flush=True)


if __name__ == '__main__': main()
