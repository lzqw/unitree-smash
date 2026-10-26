"""Generate static whole-racket arm seeds for a later near-net experiment.

The recorded successful body pose is a reference. Any forward shift is a
hypothetical walking goal, never an edited reset or a simulated trajectory.
Incoming targets use recorded public observations and the pinned flight model.
This does not evaluate dynamics, velocity, collisions, recovery or task success.
"""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.full_racket_body_residual_v4 import make_rally_controller
from quadsmash.control.initial_intercept import predict_initial_intercept
from quadsmash.control.full_racket_train_evaluation_v4 import training_cases,summarize_training


def read(path):
    return json.loads(path.read_text())


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def main():
    assert not (HERE/'PLAN.json').exists()
    reports=ROOT/'reports/G07_full_racket_v4'
    selected=read(reports/'body_residual48_after_grid16_queue01/ADMISSION_DECISION.json')['source']
    source_path,run=Path(selected['policy']),Path(selected['evaluation'])
    protocol=read(ROOT/'configs/g07_full_racket_eval_v4_r3.json')
    protocol['_root']=str(ROOT)
    cases,indices=training_cases(ROOT,protocol)
    rows=read(run/'episodes.json')
    summary=summarize_training(rows,cases,indices,protocol)
    assert summary==selected['summary'] and read(run/'job.json')['status']=='SUCCEEDED'
    assert sha(source_path)==read(run/'INPUTS.json')['policy_sha256']
    public_plan=read(run/'causal_plans.json')
    assert len(public_plan)==1 and public_plan[0]['env_ids']==list(range(128))
    observations=np.asarray(public_plan[0]['used_public_observations'],dtype=float)
    assert observations.shape==(128,794) and np.isfinite(observations).all()
    assert (observations[:,67]>.5).all()
    for name,digest in protocol['physical_configs'].items():
        assert sha(ROOT/name)==digest,name
    assert sha(ROOT/protocol['asset']['path'])==protocol['asset']['sha256']
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    reference=make_rally_controller(source,1,'cpu').reference_controller
    original_hit=(reference.teacher.arm+reference.parameters[0,8:14]).detach().cpu().numpy()
    ground_arm=np.asarray(read(run/'initial_states.json')[29]['joint_position_rad'])[12:]
    with np.load(run/'all_first_episodes.npz') as data:
        clocks=data['episode_clock_s'][:,29]
        valid=data['valid_first_episode'][:,29].astype(bool)
        frame=np.where(valid&(clocks<=rows[29]['contact_time_s']))[0][-1]
        recorded_root=data['link_poses'][frame,29,0].copy().astype(float)
        recorded_poses=data['link_poses'][frame,29].copy().astype(float)
        frame_clock=float(clocks[frame])
    assert rows[29]['complete_smash'] and read(run/'geometry.json')['body_names'][0]=='base'
    rpy=Rotation.from_quat(recorded_root[3:][[1,2,3,0]]).as_euler('xyz')
    model=FullRacketGeometryV4()
    provenance=read(reports/'body_residual_leg15_fresh_n2048_updates48_01/trajectory_traces/manifest.json')['provenance']
    native_bounds=np.asarray(provenance['joint_limits_rad'])[12:]
    bounds=np.column_stack((np.maximum(model.bounds[:,0],native_bounds[:,0]),
        np.minimum(model.bounds[:,1],native_bounds[:,1])))
    initial=np.clip(original_hit,bounds[:,0]+1e-8,bounds[:,1]-1e-8)
    profiles=[(0.,1.08),(.1,1.08),(.1,1.12),(.2,1.16)]
    pitches=[0.,2.5,5.,7.5]
    uv=[0.,.06]
    sources=[Path(__file__),source_path,run/'INPUTS.json',run/'job.json',run/'metrics.json',
        run/'episodes.json',run/'causal_plans.json',run/'initial_states.json',run/'all_first_episodes.npz',
        ROOT/'configs/g07_full_racket_eval_v4_r3.json',ROOT/'configs/shuttle_physics_v2.json',
        ROOT/'configs/assembly_v2.json',ROOT/'quadsmash/control/full_racket_geometry_v4.py',
        ROOT/'quadsmash/control/piper_geometry.py',ROOT/'quadsmash/control/initial_intercept.py',model.urdf]
    sources += [ROOT/name for name in protocol['physical_configs']]
    sources += [ROOT/protocol['asset']['path']]
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_sha256={str(p):sha(p) for p in sources},
        recorded_reference_case=29,recorded_reference_frame=int(frame),recorded_reference_clock_s=frame_clock,
        recorded_reference_contact_time_s=rows[29]['contact_time_s'],recorded_reference_root=recorded_root.tolist(),
        body_advance_m_and_ball_plane_m=profiles,downward_normal_pitch_degrees=pitches,aim_uv_m=uv,
        source_height_intercept=reference.height_intercept,
        revision_reason='Preserve earlier failures/7 normal-only seeds. Add near-vertical shaft roll choices, native and wrist-mirrored IK starts, and the original1.08 or modestly earlier1.12/1.16 planes.',
        original16_failure_records='reports/G07_full_racket_v4/near_net_pose_bank_20261026',
        proposed_prediction='Known initial public observations, RK4 original drag/gravity, prospective fixed x plane; no future ball or case labels as controller input.',
        target_aggregation='Median valid forecast of the original128 TRAIN cases; one shared seed per profile, not a case-indexed lookup policy.',
        body_advance_semantics='Hypothetical autonomous walking goal after original ground reset; no source pose or initialization is edited.',
        search_tolerances_only=dict(position_m=.005,normal_angle_deg=3.,full_rotation_deg=3.),
        shaft_roll_search_degrees=[-15.,0.,15.],
        seed_choice='Among static pose matches choose smallest max joint displacement from ground divided by5rad/s. This is only a target-rate lower bound, not execution time.',
        search_tolerances_are_not_acceptance=True,
        scope='Static joint-position search seeds only. Velocity, torque, self/net collisions, body tracking, legal contact and recovery require later native execution.',
        native_execution=False,formal_cases_read=False,formal_acceptance=False,goal_complete=False)
    (HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    rng=np.random.default_rng(750501)
    rotations=Rotation.from_quat(recorded_poses[:,3:][:,[1,2,3,0]]).as_matrix()
    names=read(run/'geometry.json')['body_names']
    links=[names.index(name) for name in ['base_link','link1','link2','link3','link4','link5','link6']]
    actual_q=[]
    for j in range(6):
        relative=model.joints[j][1].T@rotations[links[j]].T@rotations[links[j+1]]
        actual_q.append(np.arctan2(relative[1,0],relative[0,0]))
    actual_q=np.asarray(actual_q)
    mirrored=actual_q.copy()
    mirrored[3]=(mirrored[3]+2*np.pi)%(2*np.pi)-np.pi
    mirrored[4]=-mirrored[4]
    mirrored[5]=(mirrored[5]+2*np.pi)%(2*np.pi)-np.pi
    guesses=[initial,ground_arm,actual_q,mirrored,
        *[rng.uniform(bounds[:,0],bounds[:,1]) for _ in range(2)]]
    guesses=[np.clip(guess,bounds[:,0]+1e-8,bounds[:,1]-1e-8) for guess in guesses]
    physics=read(ROOT/'configs/shuttle_physics_v2.json')
    results=[]
    for advance,plane in profiles:
        forecast=predict_initial_intercept(observations,[1.10,0,1.80],[1,0,0],plane,physics,dt=.001,max_seconds=1.5)
        keep=forecast['valid']&np.isfinite(forecast['position_m']).all(1)
        assert keep.any()
        target=np.median(forecast['position_m'][keep],axis=0)
        requested_root=recorded_root[:3]+np.array([advance,0,0])
        for pitch in pitches:
            angle=np.deg2rad(pitch)
            normal=np.array([np.cos(angle),0,-np.sin(angle)])

            up=np.array([0.,0.,1.])-normal[2]*normal
            up/=np.linalg.norm(up)
            sideways=np.cross(up,normal)
            attempts=[]
            for roll in [-15.,0.,15.]:
                radians=np.deg2rad(roll)
                shaft=np.cos(radians)*up+np.sin(radians)*sideways
                desired_rotation=np.column_stack((normal,np.cross(shaft,normal),shaft))

                def residual(q):
                    point,_=model.contact_center(q,uv=uv,face=reference.teacher.face_sign,
                        base_position=requested_root,base_rpy=rpy)
                    _,rotation=model.forward(q,base_position=requested_root,base_rpy=rpy)
                    angle_error=Rotation.from_matrix(desired_rotation@rotation.T).as_rotvec()
                    return np.r_[point-target,.25*angle_error,.0002*(q-ground_arm)]

                for guess in guesses:
                    trial=least_squares(residual,guess,bounds=(bounds[:,0],bounds[:,1]),
                        max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
                    point,direction=model.contact_center(trial.x,uv=uv,face=reference.teacher.face_sign,
                        base_position=requested_root,base_rpy=rpy)
                    _,rotation=model.forward(trial.x,base_position=requested_root,base_rpy=rpy)
                    position_error=float(np.linalg.norm(point-target))
                    normal_error=float(np.rad2deg(np.arccos(np.clip(direction@normal,-1,1))))
                    rotation_error=float(np.rad2deg(Rotation.from_matrix(desired_rotation@rotation.T).magnitude()))
                    found=position_error<=.005 and normal_error<=3. and rotation_error<=3.
                    margin=float(np.minimum(trial.x-bounds[:,0],bounds[:,1]-trial.x).min())
                    minimum_time=float(abs(trial.x-ground_arm).max()/5.)
                    attempts.append(dict(fit=trial,point=point,direction=direction,
                        position_error=position_error,normal_error=normal_error,rotation_error=rotation_error,
                        found=found,margin=margin,minimum_time=minimum_time,roll=roll,
                        desired_rotation=desired_rotation))
            feasible=[item for item in attempts if item['found']]
            if feasible:
                chosen=min(feasible,key=lambda item:(item['minimum_time'],-item['margin']))
            else:
                chosen=min(attempts,key=lambda item:item['position_error']+.25*np.deg2rad(item['rotation_error']))
            fit,point,direction=[chosen[key] for key in ['fit','point','direction']]
            position_error,normal_error=[chosen[key] for key in ['position_error','normal_error']]
            assert np.isfinite(fit.x).all() and ((fit.x>=bounds[:,0])&(fit.x<=bounds[:,1])).all()
            found=chosen['found']
            results.append(dict(candidate_id=len(results),requested_walk_advance_m=advance,ball_plane_x_m=plane,
                desired_down_pitch_deg=pitch,valid_public_forecasts=int(keep.sum()),
                median_forecast_ball_center_m=target.tolist(),requested_body_root_position_m=requested_root.tolist(),
                desired_normal_world=normal.tolist(),desired_shaft_roll_deg=chosen['roll'],
                desired_racket_rotation_world=chosen['desired_rotation'].tolist(),
                rotation_error_deg=chosen['rotation_error'],joint_position_rad=fit.x.tolist(),
                reconstructed_ball_center_m=point.tolist(),reconstructed_normal_world=direction.tolist(),
                position_error_m=position_error,normal_error_deg=normal_error,pose_search_found=bool(found),
                maximum_delta_from_source_hit_rad=float(abs(fit.x-initial).max()),
                target_rate_only_lower_bound_from_ground_s=float(abs(fit.x-ground_arm).max()/5.),
                minimum_joint_position_bound_margin_rad=float(np.minimum(fit.x-bounds[:,0],bounds[:,1]-fit.x).min()),
                native_execution=False,physical_strike_feasibility_proven=False))
    assert all(sha(Path(name))==digest for name,digest in plan['source_sha256'].items())
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),candidates=results,
        pose_seeds_found=sum(item['pose_search_found'] for item in results),
        source_native_data_and_policy_unchanged=True,original_ground_reset_and_all_original_scores_retained=True,
        no_pose_found_is_not_an_unreachability_proof=True,
        source_successes_not_inherited=True,full_pose_optimized=True,native_execution=False,formal_cases_read=False,
        formal_acceptance=False,goal_complete=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    columns=['candidate_id','requested_walk_advance_m','ball_plane_x_m','desired_down_pitch_deg',
        'position_error_m','normal_error_deg','rotation_error_deg','desired_shaft_roll_deg','pose_search_found','target_rate_only_lower_bound_from_ground_s',
        'minimum_joint_position_bound_margin_rad']
    with (HERE/'CANDIDATES.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n')
        writer.writeheader()
        writer.writerows({key:item[key] for key in columns} for item in results)
    print(json.dumps(dict(candidates=len(results),pose_seeds_found=result['pose_seeds_found'],native_execution=False)),flush=True)


if __name__=='__main__':
    main()
