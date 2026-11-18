"""Attribute terminal contact residual without replacing predicted/native state.

Counterfactual arm poses are geometry diagnostics, never executable commands.
The only future state consumed is a complete causal CPU prediction.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pinocchio as pin
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1
from quadsmash.control.position_anchored_contact_search_v1 import position_anchor
from quadsmash.control.contact_pose_reachability_v1 import diagnose_contact_pose
from quadsmash.control.contact_reach_bound_v1 import string_strip_reach_bound
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4


def contact_kinematics(model,q,offset):
    q=np.asarray(q,float);offset=np.asarray(offset,float)
    if q.shape!=(model.model.nq,) or offset.shape!=(3,) or not np.isfinite(np.r_[q,offset]).all():
        raise ValueError('Finite full state and authored contact offset required')
    pin.computeJointJacobians(model.model,model.data,q)
    pin.updateFramePlacements(model.model,model.data)
    frame=model.body_frames['Racket'];pose=model.data.oMf[frame]
    lever=pose.rotation@offset
    J=pin.getFrameJacobian(model.model,model.data,frame,pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).copy()
    p=(pose.translation+lever).copy();n=pose.rotation[:,0].copy()
    return p,n,(J[:3]-pin.skew(lever)@J[3:]).copy(),(-pin.skew(n)@J[3:]).copy()


def decompose_position(actual,reference,target):
    values=np.asarray([actual,reference,target],float)
    if values.shape!=(3,3) or not np.isfinite(values).all():raise ValueError('Finite positions required')
    actual,reference,target=values
    tracking=actual-reference;reference_error=reference-target;total=actual-target
    return dict(actual_minus_target_world_m=total,actual_minus_reference_world_m=tracking,
        reference_minus_target_world_m=reference_error,
        vector_identity_error_m=float(np.linalg.norm(total-tracking-reference_error)),
        actual_error_m=float(np.linalg.norm(total)),reference_error_m=float(np.linalg.norm(reference_error)),
        tracking_displacement_m=float(np.linalg.norm(tracking)),norms_are_not_additive=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve prior evidence')
    r=json.loads(a.prefix.read_text())
    if hashlib.sha256(a.prefix.read_bytes()).hexdigest()!=a.prefix.with_suffix('.sha256').read_text().strip():raise ValueError('Prefix hash mismatch')
    for path,digest in r['input_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Input drift: '+path)
    s=r['trials'][r['selected_index']]['result']
    original=Path('reports/experimental_fullbody_gate_20261118/REPLAY.json.gz')
    initial=Path('reports/push_off_task_20261118/DIAGNOSED.json')
    start=json.loads(initial.read_text())['rows'][0]
    calls=json.load(gzip.open(original,'rt'))['physical_calls'];matches=[c for c in calls if c['lane']==start['lane']]
    if len(matches)!=1:raise ValueError('Ambiguous contact event')
    m=matches[0]['metadata'];trace=s['trajectory']
    if not s['path_valid'] or len(trace)!=m['contact_index']:raise ValueError('Complete predicted prefix required')
    q=np.array(trace[-1]['q']);v=np.array(trace[-1]['v']);q0=np.array(start['initial_q'])
    model=LoadedPinocchioModelV1();armq=model.q_indices[12:];armv=model.v_indices[12:]
    R0=pin.Quaternion(q0[3:7]).matrix();R=pin.Quaternion(q[3:7]).matrix()
    target=q0[:3]+R0@np.array(m['target_position_world_m'])
    normal=R0@np.array(m['target_normal_world']);velocity=R0@np.array(m['retargeted_velocity_current_frame_m_s'])
    offset=np.array(m['contact_point_local']);offset[0]+=m['task_physics']['collision_radius_m']
    actual,n,J,Jn=contact_kinematics(model,q,offset)
    ref=np.array(r['selected_arm_reference_rad'])[-1]
    if not np.array_equal(ref,np.array(trace[-1]['planned_arm_reference_rad'])):raise ValueError('Reference mismatch')
    qref=q.copy();qref[armq]=ref;reference=contact_kinematics(model,qref,offset)[0]
    command=np.array(trace[-1]['final_target_rad'])[12:];qc=q.copy();qc[armq]=command
    commanded=contact_kinematics(model,qc,offset)[0]
    def kinematics(qa):
        local=q.copy();local[armq]=qa;pos,n,jp,jn=contact_kinematics(model,local,offset)
        return pos,n,jp[:,armv],jn[:,armv]
    cfg=json.loads(Path('reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface']
    limits=np.array(cfg['limits'])[12:];speed=np.array(cfg['velocity'])[12:]
    anchor=position_anchor(kinematics,[q[armq],ref,command],limits[:,0],limits[:,1],target)
    basev=v.copy();basev[armv]=0
    def base_velocity(point):
        # Frozen terminal root twist; arm/leg rates excluded. Body-frame root
        # velocity is transported by the recorded terminal root rotation.
        return R@v[:3]+np.cross(R@v[3:6],point-q[:3])
    diagnosis=diagnose_contact_pose(kinematics,anchor['q_anchor'],limits[:,0],limits[:,1],speed,target,normal,velocity,base_velocity_fn=base_velocity)
    bound=string_strip_reach_bound(FullRacketGeometryV4(),R.T@(target-q[:3]),m['task_string_strip'],ball_radius_m=m['task_physics']['collision_radius_m'],face_sign=float(np.sign(m['contact_point_local'][0])))
    # Geometric targets only: no mass relocation, velocity injection or
    # command installation. The bound ignores limits and is necessary only.
    direction=R.T@(target-q[:3])-np.asarray(bound['shoulder_body_m'])
    direction/=np.linalg.norm(direction)
    translation_lower=R@direction*bound['conditional_translation_lower_bound_m']
    translation_witness=target-np.asarray(anchor['position_anchor'])
    result=dict(unrealized_base_targets=dict(
        necessary_radial_translation_world_m=translation_lower,
        local_ik_contact_translation_world_m=translation_witness,
        local_translation_is_global_minimum=False,leg_commands_generated=False,
        fullbody_revalidation_required=True),prefix_sha256=hashlib.sha256(a.prefix.read_bytes()).hexdigest(),input_sha256=r['input_sha256'],
        semantics='fixed predicted terminal base; counterfactual arm IK is not a dynamic trajectory',
        position_decomposition=decompose_position(actual,reference,target),command_pose_error_m=float(np.linalg.norm(commanded-target)),
        actual_contact_world_m=actual,reference_contact_world_m=reference,target_contact_world_m=target,
        terminal_root_position_world_m=q[:3],terminal_root_rotation_world=R,
        velocity_decomposition=dict(actual_world_m_s=J@v,base_world_m_s=J@basev,arm_world_m_s=J[:,armv]@v[armv],
            base_formula_error_m_s=float(np.linalg.norm(J@basev-base_velocity(actual)))),
        position_anchor=anchor,pose_diagnosis=diagnosis,reach_bound=bound,
        future_native_states_read=False,native_execution=False,valid_teacher_reference=False)
    if abs(result['position_decomposition']['actual_error_m']-s['terminal_errors']['position_m'])>1e-10:raise ValueError('Terminal source/FK mismatch')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x))+'\n')
    print(json.dumps(dict(actual_error=result['position_decomposition']['actual_error_m'],reference_error=result['position_decomposition']['reference_error_m'],position_only_error=anchor['position_error_anchor_m'],reach_deficit=bound['conditional_translation_lower_bound_m'])))
if __name__=='__main__':main()
