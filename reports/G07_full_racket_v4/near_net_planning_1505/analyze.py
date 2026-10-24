"""Causal plane forecasts, free-flight margins and matched-body static IK only."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from quadsmash.control.initial_intercept import predict_initial_intercept
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.physics.smash_feasibility import predict_flights
from quadsmash.physics.court_v4 import top_height

OUT=Path(__file__).parent
RUN=ROOT/'reports/G07_full_racket_v4/ball_attack_recovered_normal38_cem21'
assert not (OUT/'REPORT.json').exists()
torch.set_num_threads(2)
plan_path=RUN/'causal_plans_000.json'
plans=json.loads(plan_path.read_text())[0]
index=plans['env_ids'].index(3)
public=np.array(plans['used_public_observations'][index:index+1])
physics_path=ROOT/'configs/shuttle_physics_v2.json';physics=json.loads(physics_path.read_text())
court_path=ROOT/'configs/court_net_v4.json';court=json.loads(court_path.read_text())
event_path=RUN/'contacts_000.json'
event=next(e for e in json.loads(event_path.read_text()) if e['env_id']==3 and e['episode_index']==0 and e['string_contact'])
row_path=RUN/'episodes_000.json';row=json.loads(row_path.read_text())[3]
actual_velocity=np.array(row['outgoing_velocity_mps'])
warm_path=ROOT/'reports/G07_full_racket_v4/normal38_gen0_best_for_slew_1444/policy_evidence.pt'
warm=torch.load(warm_path,map_location='cpu',weights_only=False)
contract_path=ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json'
limits=json.loads(contract_path.read_text())['limits']
lower,upper=np.array([limits[f'joint{i}'] for i in range(1,7)]).T
observed_q=np.array(event['joint_position_rad'][-6:])
root=np.array(event['body_link_poses_local'][0])
rotation=Rotation.from_quat(root[[4,5,6,3]]).as_matrix()
legacy_path=ROOT/'reports/G07_full_racket_v4/matched_flange_kinematics_1245/LEGACY_MOUNT.json'
legacy=json.loads(legacy_path.read_text())
assert legacy['local_rotation0_wxyz']==legacy['local_rotation1_wxyz']==[1.,0.,0.,0.]
full=FullRacketKinematicsV4().double()
short=FullRacketKinematicsV4().double()
short.tool_offset.copy_(torch.tensor(legacy['local_position0_m'],dtype=torch.float64))
short.tool_rotation.copy_(torch.eye(3,dtype=torch.float64))
uv=warm['seed']['string_uv_m'];assert uv==[0.,.1]
models={'full_racket':(full,[.0256,*uv]),'legacy_shortboard':(short,[.0325,*uv])}
normals={'observed_native_full_racket_face':np.array(event['face_normal']),
         'teacher_world_command':np.array(warm['seed']['outward_contact_normal_world'])}
vx,vz=np.meshgrid(np.linspace(6.5,10.,71),np.linspace(-4.,-1.,61))
velocities=np.column_stack([vx.ravel(),np.full(vx.size,actual_velocity[1]),vz.ravel()])
records=[];grids={}
for plane in [1.05,1.15,1.25,1.35]:
    forecast=predict_initial_intercept(public,[1.1,0.,1.8],[1.,0.,0.],plane,physics,dt=.001,max_seconds=1.5)
    assert forecast['valid'][0]
    ball=forecast['position_m'][0]
    flight=predict_flights(ball,velocities,dt=.0005,duration=2.,net_x=court['net_plane_x_m'],
        net_height=court['net_center_top_m'],net_half_width=court['net_width_m']/2,
        net_half_thickness=court['net_equivalent_thickness_m']/2,
        target_x=court['target_x_limits_m'],target_y=court['target_y_half_width_m'])
    y=flight['net_position'][:,1]
    height=np.max(top_height(np.stack([y-.025,y,y+.025]),court),axis=0)
    margin=flight['conservative_net_exit_margin']-(height-court['net_center_top_m'])
    landing=flight['landing_position']
    interior=np.minimum.reduce([landing[:,0]-court['target_x_limits_m'][0],
        court['target_x_limits_m'][1]-landing[:,0],court['target_y_half_width_m']-abs(landing[:,1])])
    feasible=(np.linalg.norm(velocities,axis=1)>=8)&(margin>=.005)&(interior>=0)
    robust=feasible&(margin>=.03)&(interior>=.15)
    best=None
    if feasible.any():
        idx=np.where(feasible,np.linalg.norm(velocities-actual_velocity,axis=1),np.inf).argmin()
        best=dict(velocity_mps=velocities[idx].tolist(),speed_mps=float(np.linalg.norm(velocities[idx])),
            landing_m=landing[idx].tolist(),net_clearance_m=float(margin[idx]),target_interior_margin_m=float(interior[idx]))
    static={}
    for model_name,(kin,offset) in models.items():
        static[model_name]={}
        offset=torch.tensor(offset,dtype=torch.float64)
        for normal_name,normal_target in normals.items():
            normal_target=normal_target/np.linalg.norm(normal_target)
            def evaluate(q):
                with torch.inference_mode():p,n,jp,jn=kin(torch.tensor(q[None]),offset)
                return p[0].numpy()@rotation.T+root[:3],n[0].numpy()@rotation.T,rotation@jp[0].numpy(),rotation@jn[0].numpy()
            def residual(q):
                p,n,*_=evaluate(q);return np.r_[p-ball,.3*(n-normal_target)]
            def jacobian(q):
                _,_,jp,jn=evaluate(q);return np.r_[jp,.3*jn]
            fit=least_squares(residual,np.clip(observed_q,lower+1e-7,upper-1e-7),jac=jacobian,
                bounds=(lower,upper),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            p,n,*_=evaluate(fit.x)
            static[model_name][normal_name]=dict(joints_rad=fit.x.tolist(),position_error_m=float(np.linalg.norm(p-ball)),
                face_error_deg=float(np.degrees(np.arccos(np.clip(n@normal_target,-1,1)))),
                maximum_joint_change_from_observed_full_racket_pose_rad=float(abs(fit.x-observed_q).max()),
                solver_converged=bool(fit.success))
    records.append(dict(intercept_plane_x_m=plane,predicted_ball_center_m=ball.tolist(),
        causal_arrival_clock_s=float(public[0,775]+forecast['arrival_time_s'][0]),
        sampled_outgoing_velocities=len(velocities),feasible_free_flight_targets=int(feasible.sum()),
        planning_margin_targets_3cm_net_and15cm_target_interior=int(robust.sum()),nearest_velocity_proposal=best,
        static_joint_limit_IK_at_same_observed_body_pose=static))
    grids[f'plane_{plane:.2f}_feasible']=feasible
    grids[f'plane_{plane:.2f}_landing']=landing
    grids[f'plane_{plane:.2f}_net_clearance']=margin
np.savez_compressed(OUT/'planning_grids.npz',outgoing_velocity_mps=velocities,**grids)
sources=[plan_path,event_path,row_path,warm_path,contract_path,legacy_path,physics_path,court_path,Path(__file__),
    ROOT/'quadsmash/control/initial_intercept.py',ROOT/'quadsmash/control/full_racket_kinematics_v4.py',
    ROOT/'quadsmash/physics/smash_feasibility.py']
report=dict(created_at=datetime.now(timezone.utc).isoformat(),
    scope='Planning-only causal ball forecasts from one saved TRAIN65public input, free-flight velocity grids and static joint-limit IK at one fixed observed body pose. Old/new tool transforms differ but body pose and hardware joint bounds match. No native collision, full-tool net clearance, dynamic reach, body recovery or distribution success is proven.',
    source_native_env_id=3,held_body_pose_from_time_s=event['time_s'],records=records,
    original_target_x_m=court['target_x_limits_m'],original_target_y_half_width_m=court['target_y_half_width_m'],
    stricter_buffers_are_planning_preferences_only=True,
    old_payload_dynamics_simulated=False,changed_simulator_state=False,formal_acceptance=False,
    source_sha256={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k!='static_joint_limit_IK_at_same_observed_body_pose'} for row in records],indent=2))
