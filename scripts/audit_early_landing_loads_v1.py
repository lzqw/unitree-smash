"""Replay a failed candidate and measure its model support impulses, not native forces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pinocchio as pin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.loaded_pinocchio_model_v1 import PublicState
from scripts.authored_ground_clearance_v1 import AuthoredGroundClearance
from scripts.search_early_landing_trajectory_v1 import EarlyReference
from scripts.search_whole_body_early_landing_v2 import WholeBodyEarlyReference
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.audit_sphere_touchdown_predictions_v1 import sphere_gaps
from scripts.audit_joint_sequence_proposal_v1 import plain
import scripts.predict_frictional_spheres_v1 as predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Fresh audit output required')
    read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    record=read(a.search)
    for f,h in record['source_sha256'].items():
        if sha(f)!=h:raise ValueError('Changed search source '+f)
    cfg=record['configuration'];query=read(cfg['query']);manifest=read(Path(cfg['bundle'])/'manifest.json')
    controller=build_controller(manifest['configuration']);model=controller.qp.model;interface=controller.qp.interface
    q=np.asarray(query['initial_q']);v=np.asarray(query['initial_v'])
    state=PublicState(q,v,pin.Quaternion(q[3:7]).matrix(),q[:3].copy(),q[model.q_indices].copy(),
                      v[model.v_indices].copy(),np.asarray(query['initial_contacts'],bool),0.)
    state.previous_command_target=np.asarray(query['actual_previous_command'])
    candidate=record['fine_candidates'][0];saved=candidate['fine']
    if saved['substeps']!=160 or saved['failure'] is None or saved['failure']['reason']!='path_constraints':
        raise ValueError('A completed fine path rejection is required')
    geometry=AuthoredGroundClearance(ROOT/'configs/go2_piper_authored_ground_geometry_v1.json',model,margin_m=0.)
    spheres=read(cfg['sphere_audit'])['spheres']
    shape=np.asarray(candidate['knots_rad']).shape
    if shape not in [(3,12),(3,18)]:raise ValueError('Unsupported reference parameterization')
    factory=WholeBodyEarlyReference if shape==(3,18) else EarlyReference
    policy=TerminalAttitudeFeedbackV1(factory(model,interface,state,candidate['knots_rad']),
        proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
    original=predictor.frictional_velocity_step;captured=[];rows=[];maximum_error=0.
    def observed(*args,**kwargs):
        result=original(*args,**kwargs)
        # All actuated joints are revolute scalar coordinates: infer their
        # post-microstep positions from the unchanged PD error and velocity.
        joint_v=result['velocity'][model.v_indices]
        joint_q=target-np.asarray(args[6])+float(args[8])*joint_v
        captured.append((result['impulse'].copy(),result['torque'].copy(),joint_q.copy(),joint_v.copy()))
        return result
    # Only this offline process is patched. The returned physics solution is
    # untouched and the original function is restored even on exceptions.
    with patch.object(predictor,'frictional_velocity_step',observed):
        for step in range(saved['failure']['step']+1):
            captured.clear();target,_=policy.action(80+step,state,state.previous_command_target)
            prediction=predictor.predict_frictional_spheres(model,interface,state,target,spheres,substeps=160,geometry=geometry)
            if len(captured)!=160:raise ValueError('Incomplete physical microstep observation')
            following=prediction['state'];following.previous_command_target=target.copy()
            if step<len(saved['trajectory']):
                expected=saved['trajectory'][step]
                error=max(float(np.max(abs(following.q-expected['q']))),float(np.max(abs(following.v-expected['v']))),
                          float(np.max(abs(target-expected['target']))))
                maximum_error=max(maximum_error,error)
                if error>1e-12:raise ValueError('Observer changed the candidate replay')
            impulse=np.stack([x[0] for x in captured]);torques=np.stack([x[1] for x in captured])
            joint_q=np.stack([x[2] for x in captured]);joint_v=np.stack([x[3] for x in captured])
            if np.max(abs(joint_q[-1]-following.joint_position))>1e-12:
                raise ValueError('Microstep joint reconstruction mismatch')
            forces=impulse[:,2::3]/(.01/160)
            snap=model.snapshot(following.q,following.v)
            rows.append(dict(step=step,mean_normal_force_n=forces.mean(0),peak_normal_force_n=forces.max(0),
                loaded_microstep_fraction=(forces>1.).mean(0),
                effort_saturation_fraction=(abs(torques)>=interface.effort[None,:]-1e-6).mean(0),
                maximum_position_excess_rad=np.maximum(0.,np.maximum(
                    interface.limits[:,0]-joint_q,joint_q-interface.limits[:,1])).max(0),
                maximum_velocity_excess_rad_s=np.maximum(0.,abs(joint_v)-interface.velocity).max(0),
                interval_end_joint_position=following.joint_position,
                interval_end_joint_velocity=following.joint_velocity,
                root_linear_velocity_world=following.root_rotation_world@following.v[:3],
                root_angular_velocity_world=following.root_rotation_world@following.v[3:6],
                foot_surface_gap_m=sphere_gaps(model,following.q,spheres),com_world=snap['com_world'],
                com_velocity_world=snap['com_velocity_world'],
                first_ground_violation=prediction['first_ground_violation'],diagnostics=prediction['diagnostics']))
            state=following
    if predictor.frictional_velocity_step is not original:raise ValueError('Observer was not removed')
    sources={str(a.search.resolve()):sha(a.search),str(Path(__file__).resolve()):sha(__file__)}
    out=dict(scope=__doc__,case=3,tick=record['tick'],feet=model.feet,joint_order=model.joint_order,
        mass_kg=model.mass,weight_n=model.mass*9.81,maximum_saved_replay_difference=maximum_error,
        rows=rows,source_sha256=sources,search_source_sha256=record['source_sha256'],
        native_forces=False,valid_expert_labels=False,
        limitation='Model impulse-derived mean forces do not establish real foot/head loading or prove infeasibility.')
    a.output.write_text(json.dumps(plain(out),indent=2,allow_nan=False)+'\n')
    print(json.dumps(plain({k:v for k,v in out.items() if k in ['weight_n','maximum_saved_replay_difference','rows']})))


if __name__=='__main__':main()
