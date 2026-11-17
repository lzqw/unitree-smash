"""Offline conditional contact refinement on fully integrated model prefixes.

Consumes only causal CPU predictions plus their original current observation.
No native future state is read. Revised endpoints are NOT execution references:
new Hermite and complete 18D verification remain mandatory.
"""
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np
import torch
from quadsmash.control.action_conditioned_base_v1 import ActionConditionedBaseV1
from quadsmash.control.position_anchored_contact_search_v1 import position_anchor
from quadsmash.control.task_constrained_hit_search_v2 import optimize_task_hit
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.control.contact_reach_bound_v1 import string_strip_reach_bound
from quadsmash.control.impact_outcome_evaluator_v1 import ImpactOutcomeEvaluator
from quadsmash.physics.full_racket_v4 import load_racket
from scripts.loaded_pinocchio_model_v1 import LoadedPinocchioModelV1

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True);p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve previous experiment')
    x=json.load(gzip.open(a.replay,'rt')) if a.replay.suffix=='.gz' else json.loads(a.replay.read_text())
    digest=hashlib.sha256(a.observations.read_bytes()).hexdigest()
    original=[v for k,v in x['input_sha256'].items() if k.endswith('public_observations.npy')]
    if original!=[digest]:raise ValueError('Current-state observation source mismatch')
    observations=np.load(a.observations,mmap_mode='r')
    cfg=json.loads((ROOT/'reports/codex_teacher_joint_eval_20261116/controller.json').read_text())['interface']
    limits=np.asarray(cfg['limits']);speed=np.asarray(cfg['velocity'])[12:]
    defaults=json.loads((ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json').read_text())['defaults_before_neutral_arm_override']
    model=LoadedPinocchioModelV1();default=np.array([defaults[n] for n in model.joint_order])
    default[12:]=json.loads((ROOT/'configs/full_racket_reset_v4.json').read_text())['arm_joint_positions_rad']
    fk=FullRacketKinematicsV4().to(dtype=torch.float32);rows=[]
    for call in x['physical_calls']:
        r=call['result'];m=call['metadata'];lane=call['lane'];trace=r.get('state_trace') or []
        if len(trace)!=m['contact_index'] or r.get('rejection')!='actual_racket_terminal_residual':continue
        # Locate the unique originating planning event by full command digest,
        # not a historical success/failure label or lane-specific control rule.
        matches=[]
        for plan in x['plans']:
            clocks=dict(zip(plan['env_ids'],plan['observation_clock_s']))
            for summary in plan['contact_first_plan_summaries']:
                for anchor in summary.get('search_diagnostics',{}).get('anchors',[]):
                    physical=anchor.get('physical_result') or {}
                    if summary['lane']==lane and physical.get('replayed_command_sha256')==r['replayed_command_sha256']:
                        matches.append((clocks[lane],anchor))
        if len(matches)!=1:raise ValueError('Ambiguous source event')
        clock,old_anchor=matches[0];tick=int(np.argmin(abs(observations[:,lane,775]-clock)))
        if abs(float(observations[tick,lane,775])-clock)>1e-7:raise ValueError('Missing current state')
        state=model.state_from_public(np.asarray(observations[tick,lane]),default)
        if np.linalg.norm(state.q[:3]-r['geometry_witness_initial']['root_position_world_m'])>1e-8:
            raise ValueError('Initial state mismatch')
        commands=np.asarray(r['replayed_target_sequence_rad'])
        base=ActionConditionedBaseV1(state.q,state.v,commands,trace)
        base.validate_context(state.q,state.v,commands)
        transport=base(m['contact_time_s']);R=transport['rotation_current_from_future'];d=transport['translation_current_m']
        vb=transport['linear_velocity_current_m_s'];omega=transport['angular_velocity_current_rad_s']
        body_target=R.T@(np.asarray(m['target_position_world_m'])-d)
        local=np.asarray(m['contact_point_local']);radius=m['task_physics']['collision_radius_m']
        def kinematics(q,contact=local):
            offset=np.asarray(contact).copy();offset[0]+=radius
            with torch.inference_mode():values=fk(torch.tensor(q,dtype=torch.float32)[None],torch.tensor(offset,dtype=torch.float32))
            return tuple(y[0].numpy().astype(float) for y in values)
        q_initial=np.asarray(call['arm_sequence_rad'])[0];q_previous=np.asarray(call['arm_sequence_rad'])[m['contact_index']]
        anchor=position_anchor(kinematics,[q_initial,q_previous],limits[12:,0],limits[12:,1],body_target)
        row=dict(lane=lane,clock_s=clock,duration_s=m['contact_time_s'],current_observation_tick=tick,
            previous_terminal_error_m=r['actual_racket_position_error_m'],transport=transport,
            target_future_body_m=body_target,position_anchor=anchor,
            reach_bound=string_strip_reach_bound(FullRacketGeometryV4(),body_target,m['task_string_strip'],ball_radius_m=radius,face_sign=float(np.sign(local[0]))),
            changed_command_requires_revalidation=True,fullbody_prediction_valid=False,
            native_execution=False,valid_teacher_reference=False)
        if anchor['position_error_anchor_m']>.0005:
            row['rejection']='ACTION_CONDITIONED_POSITION_ANCHOR_OUTSIDE_FINE_BUDGET'
        else:
            basis=state.root_rotation_world;origin=state.root_position_world
            ball=origin+basis@np.asarray(m['target_position_world_m'])
            incoming=basis@np.asarray(m['incoming_velocity_planning'])
            impact=ImpactOutcomeEvaluator(m['task_physics'],m['task_court'],load_racket())
            def forward(metadata):
                n=basis@metadata['achieved_normal_current_frame']
                proxy=origin+basis@metadata['achieved_position_current_frame_m']
                velocity=basis@metadata['achieved_velocity_current_frame_m_s']
                return impact.evaluate(ball,incoming,proxy-radius*n,velocity,n,metadata['contact_point_local'])
            def base_velocity(point):return R.T@(vb+np.cross(omega,R@point))
            trial=optimize_task_hit(kinematics,anchor['q_anchor'],old_anchor['qd_rad_s'],
                limits[12:,0],limits[12:,1],speed,body_target,local,m['task_string_strip'],
                base_velocity,R,d,m,forward,m['task_court'],m['incoming_velocity_planning'])
            row['joint_task_refinement']=trial
            row['rejection']=None if trial['accepted'] else 'ACTION_CONDITIONED_JOINT_TASK_INFEASIBLE'
        rows.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(source_sha256=hashlib.sha256(a.replay.read_bytes()).hexdigest(),
        observations_sha256=digest,scope='conditional offline optimization, not independent fullbody validation',
        future_native_states_read=False,training=False,rows=rows),indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else float(v))+'\n')
    print(json.dumps(dict(examined=len(rows),rejections=[r['rejection'] for r in rows])))
if __name__=='__main__':main()
