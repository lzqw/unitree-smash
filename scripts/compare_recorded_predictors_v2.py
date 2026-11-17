"""Compare existing offline contact models on the SAME native state and command.

No parameter fitting, native simulation, controller modification or execution
certificate. Future measured state appears only in post-prediction errors.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1,PublicState,rotation_wxyz
from scripts.whole_body_support_qp_v1 import PositionPDInterface
from scripts.unilateral_implicit_prediction_v1 import predict_unilateral_support
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.predict_speed_limited_spheres_v1 import predict_speed_limited_spheres
from scripts.audit_sphere_touchdown_predictions_v1 import spheres_from_asset
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.diagnostic_nonfoot_spheres_v1 import nonfoot_spheres
from scripts.hip_mount_collision_response_v1 import HipMountCollisionResponseV1


def measured_state(model,data,tick,lane):
    R=rotation_wxyz(data['root_quaternion'][tick,lane]);q=pin.neutral(model.model);v=np.zeros(model.model.nv)
    q[:3]=data['root_position'][tick,lane];q[3:7]=pin.Quaternion(R).coeffs()
    q[model.q_indices]=data['joint_position'][tick,lane];v[model.v_indices]=data['joint_velocity'][tick,lane]
    v[:6]=data['public'][tick,lane,:6]  # native link-origin, body-frame twist
    state=PublicState(q,v,R,q[:3].copy(),q[model.q_indices].copy(),v[model.v_indices].copy(),data['contacts'][tick,lane].copy(),0.)
    state.previous_command_target=data['previous_target'][tick,lane].copy()
    state.measured_force_world=data['foot_force'][tick,lane].copy()
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--spheres',type=Path);p.add_argument('--rollout',action='store_true');p.add_argument('--method',choices=('current_body_anchor','authored_sphere_contact','sphere_plus_sim_speed_constraint','sphere_speed_and_position_stops','sphere_stops_and_nonfoot_response','sphere_stops_nonfoot_and_hip_response'));p.add_argument('--start-tick',type=int,default=249);p.add_argument('--steps',type=int,default=14);p.add_argument('--solver-iterations',type=int,default=400);a=p.parse_args()
    if a.output.exists():raise ValueError('Do not overwrite evidence')
    a.output.mkdir(parents=True)
    if not (0<=a.start_tick and 1<=a.steps<=80 and a.start_tick+a.steps<800):raise ValueError('Bounded recorded prediction window required')
    root=Path(__file__).resolve().parents[1];source=a.native/'control_state.npz';z=np.load(source,allow_pickle=False)
    interface_path=root/'reports/codex_teacher_joint_eval_20261116/controller.json'
    interface=PositionPDInterface(**json.loads(interface_path.read_text())['interface'])
    protocol=json.loads((root/'configs/g07_full_racket_eval_v5_contact_recovery_origin0.json').read_text());asset=root/protocol['asset']['path']
    if hashlib.sha256(asset.read_bytes()).hexdigest()!=protocol['asset']['sha256']:raise ValueError('Asset drift')
    model=LoadedPinocchioModelV1()
    if a.spheres is None:spheres=spheres_from_asset(asset,model.feet)
    else:
        record=json.loads(a.spheres.read_text())
        if record['asset_sha256']!=protocol['asset']['sha256']:raise ValueError('Cached sphere asset differs')
        if not all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in record['dependency_sha256'].items()):raise ValueError('Cached sphere dependencies drifted')
        spheres=record['spheres']
    if [s['body'] for s in spheres]!=model.feet:raise ValueError('Foot order mismatch')
    methods={'current_body_anchor':lambda s,t:predict_unilateral_support(model,interface,s,t,substeps=160,enforce_joint_stops=True),
      'authored_sphere_contact':lambda s,t:predict_frictional_spheres(model,interface,s,t,spheres,substeps=160),
      'sphere_plus_sim_speed_constraint':lambda s,t:predict_speed_limited_spheres(model,interface,s,t,spheres,substeps=160)}
    methods['sphere_speed_and_position_stops']=lambda s,t:predict_speed_limited_spheres(
        model,interface,s,t,spheres,substeps=160,enforce_joint_stops=True)
    if a.method in ('sphere_stops_and_nonfoot_response','sphere_stops_nonfoot_and_hip_response'):
        geometry=AuthoredGroundClearance(root/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
        extra_spheres=nonfoot_spheres(geometry,model.feet)
        hip_response=HipMountCollisionResponseV1(model) if a.method=='sphere_stops_nonfoot_and_hip_response' else None
        methods[a.method]=lambda s,t:predict_speed_limited_spheres(model,interface,s,t,spheres,substeps=160,
            enforce_joint_stops=True,diagnostic_nonfoot_spheres=extra_spheres,diagnostic_hip_mount_response=hip_response,contact_max_iterations=a.solver_iterations)
    if a.method is not None:methods={a.method:methods[a.method]}
    rows=[]
    # Entire predeclared window, both original lanes; no success subset.
    for lane in range(2):
        histories={name:measured_state(model,z,a.start_tick,lane) for name in methods};failed=set()
        for tick in range(a.start_tick,a.start_tick+a.steps):
            initial=measured_state(model,z,tick,lane);truth=measured_state(model,z,tick+1,lane)
            target=z['previous_target'][tick+1,lane].copy()
            for name,predict in methods.items():
                if a.rollout and name in failed:continue
                if a.rollout:initial=histories[name]
                start=time.perf_counter();row=dict(lane=lane,tick=tick,method=name)
                try:
                    result=predict(copy.deepcopy(initial),target.copy());s=result['state']
                    if not np.isfinite(np.r_[s.q,s.v]).all():raise ValueError('Nonfinite output')
                    histories[name]=copy.deepcopy(s)
                    initial_positions,initial_rotations=model.poses(s.q);actual_positions,actual_rotations=model.poses(truth.q)
                    racket=model.names.index('Racket');difference=pin.difference(model.model,truth.q,s.q)
                    row.update(solved=True,joint_position_max_error_rad=float(max(abs(s.joint_position-truth.joint_position))),
                      joint_velocity_max_error_rad_s=float(max(abs(s.joint_velocity-truth.joint_velocity))),
                      root_position_error_m=float(np.linalg.norm(s.q[:3]-truth.q[:3])),root_rotation_error_rad=float(np.linalg.norm(difference[3:6])),
                      racket_position_error_m=float(np.linalg.norm(initial_positions[racket]-actual_positions[racket])),
                      predicted_q=s.q.tolist(),predicted_v=s.v.tolist(),native_q=truth.q.tolist(),native_v=truth.v.tolist(),
                      predicted_joint_velocity=s.joint_velocity.tolist(),native_joint_velocity=truth.joint_velocity.tolist(),diagnostics=result['diagnostics'],
                      predicted_contacts=s.contacts.tolist(),native_contacts=truth.contacts.tolist())
                except (ValueError,np.linalg.LinAlgError,FloatingPointError) as e:
                    row.update(solved=False,error=str(e));failed.add(name)
                row['seconds']=time.perf_counter()-start;rows.append(row)
                with (a.output/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(lane=lane,tick=tick,finished=len(rows))),flush=True)
    summary={}
    for name in methods:
        selected=[r for r in rows if r['method']==name];solved=[r for r in selected if r['solved']]
        summary[name]=dict(solved=len(solved),attempted=len(selected),requested=2*a.steps,not_a_physical_acceptance_count=True,
            error_stats={key:dict(mean=float(np.mean([r[key] for r in solved])),max=float(max(r[key] for r in solved))) for key in
                ('joint_velocity_max_error_rad_s','joint_position_max_error_rad','root_position_error_m','racket_position_error_m')} if solved else {})
    (a.output/'RESULTS.json').write_text(json.dumps(dict(mode='open_loop' if a.rollout else 'one_step',start_tick=a.start_tick,steps=a.steps,solver_iterations=a.solver_iterations,summary=summary,rows=rows,spheres=spheres,joint_order=model.joint_order,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),interface_sha256=hashlib.sha256(interface_path.read_bytes()).hexdigest(),asset_sha256=protocol['asset']['sha256'],
        policy_modified=False,native_execution=False,physx_equivalent=False,valid_teacher_reference=False,
        speed_constraint_caveat='Existing experimental ideal simulator constraint; reaction not calibrated as actual actuator braking. Never native admission.'),indent=2))
if __name__=='__main__':main()
