"""Execute a model-only receding recovery with a declared mid-course stress.

Starts at the recorded native query, advances dynamics between planning calls,
and applies one explicit 0.5-degree model-state perturbation at tick 400. Native
physics is never reset or modified. This is not a native task success metric.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_native_replanned_suffix_v1 import validate
from scripts.check_constrained_recovery_suffix_v1 import source_context,state_from_row
from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.current_state_recovery_planner_v1 import CurrentStateRecoveryPlannerV1,build_feedback_plan
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.predict_frictional_spheres_v1 import predict_frictional_spheres
from scripts.perturb_early_support_domain_v1 import perturb_state
from scripts.feedback_tail_rollout_v1 import terminal_measures,terminal_violations,kinematically_stable
from scripts.audit_joint_sequence_proposal_v1 import plain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=1800.)
    a=p.parse_args()
    if a.output.exists() or not 900<=a.max_seconds<=2400:raise ValueError('Fresh output and bounded experiment required')
    read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=read(a.source);start,expected=validate(source)
    if start!=350:raise ValueError('Current declared actual-state source required')
    _,knots,_,audit,initargs=source_context(source['configuration']['source'])
    initialize(*initargs);w=_WORKER;controller=w['controller']
    state=state_from_row(source['query'])
    plan=dict(original_knots_rad=knots,corrections=[dict(control_tick=start,
        previous_command=state.previous_command_target.copy(),knots_rad=np.asarray(source['best']['corrections_rad']))])
    hashes=dict(source['source_sha256']);hashes[str(a.source.resolve())]=sha(a.source)
    for name,module in list(sys.modules.items()):
        if name.startswith('scripts.') and getattr(module,'__file__',None):
            path=Path(module.__file__).resolve();hashes[str(path)]=sha(path)
    hashes[str(Path(__file__).resolve())]=sha(__file__)
    for path,h in hashes.items():
        if sha(path)!=h:raise ValueError('Changed source '+path)
    began=time.monotonic();deadline=began+a.max_seconds
    planner=CurrentStateRecoveryPlannerV1(initargs,audit,max_seconds=600.)
    geometry=HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02)
    rows=[];calls=[];failure=None;stable=0;longest=0;policy=None;first_plan_exact=False;stress=None
    a.output.parent.mkdir(parents=True,exist_ok=True)
    def checkpoint():
        tmp=a.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(plain(dict(completed_steps=len(rows),planning_calls=len(calls),
            calls=[dict(tick=c['control_tick'],accepted=c['accepted'],wall_seconds=c['wall_seconds']) for c in calls],
            failure=failure,stress=stress,first_plan_exact=first_plan_exact)),allow_nan=False)+'\n')
        tmp.replace(a.output.with_suffix('.partial.json'))
    try:
        for tick in range(start,800):
            if time.monotonic()>=deadline:
                failure=dict(control_tick=tick,reason='experiment_compute_budget');break
            if tick==400:
                before=copy.deepcopy(state)
                spec=dict(kind='orientation',world_axis=0,amount=float(np.deg2rad(.5)),name='midcourse_world_roll_plus_half_degree')
                state=perturb_state(state,spec);stable=0
                stress=dict(control_tick=tick,spec=spec,model_only=True,before_q=before.q,before_v=before.v,
                    after_q=state.q.copy(),after_v=state.v.copy(),previous_command_preserved=np.array_equal(
                        state.previous_command_target,before.previous_command_target))
            if tick in [350,400,550,700,750]:
                remaining=deadline-time.monotonic()
                if remaining<120:
                    failure=dict(control_tick=tick,reason='insufficient_declared_planning_budget');break
                planner.max_seconds=min(600.,remaining)
                result=planner.solve(state,tick,plan,preceding_stable_ticks=stable);calls.append(result)
                print(json.dumps(dict(control_tick=tick,accepted=result['accepted'],wall_seconds=result['wall_seconds'])),flush=True)
                if not result['accepted']:
                    failure=dict(control_tick=tick,reason='no_verified_current_state_plan');checkpoint();break
                if tick==350:
                    trajectory=result['prediction']['trajectory']
                    first_plan_exact=all(np.array_equal(np.asarray([r[k] for r in trajectory]),expected[k])
                        for k in ['q','v','target','contacts'])
                    if not first_plan_exact:raise ValueError('Planner warm start changed the verified original suffix')
                plan=result['plan'];policy=build_feedback_plan(w,plan);checkpoint()
            previous=state.previous_command_target.copy();target,mode=policy.action(80+tick-280,state,previous)
            if (np.any(target<controller.qp.interface.limits[:,0]-1e-10) or
                    np.any(target>controller.qp.interface.limits[:,1]+1e-10) or
                    np.max(abs(target[12:]-previous[12:]))>.1+1e-10):
                raise ValueError('Receding controller violated actual command mapping')
            try:
                prediction=predict_frictional_spheres(controller.qp.model,controller.qp.interface,state,target,w['spheres'],
                    geometry=geometry,substeps=160)
            except (ValueError,np.linalg.LinAlgError) as error:
                failure=dict(control_tick=tick,reason='model_prediction_rejected',error=repr(error));break
            bad={k:v for k,v in prediction['diagnostics'].items() if
                (k=='nonfoot_ground_violation_m' and v>1e-6) or
                (k in ['joint_position_violation_rad','joint_velocity_violation_rad_s','motor_torque_violation_nm'] and v>3e-4)}
            if bad:
                failure=dict(control_tick=tick,reason='path_constraints',violations=bad,
                    first_ground_violation=prediction['first_ground_violation']);break
            state=prediction['state'];state.previous_command_target=target.copy()
            measures=terminal_measures(controller,state)
            stable=stable+1 if kinematically_stable(measures,w['protocol']) and state.q[2]>=.12 else 0
            longest=max(longest,stable)
            rows.append(dict(control_tick=tick,target=target,q=state.q.copy(),v=state.v.copy(),contacts=state.contacts.copy(),mode=mode))
    finally:planner.close()
    terminal=terminal_measures(controller,state);violations=terminal_violations(terminal,.01)
    if stable<80:violations['trailing_window_s']=stable*.01
    if state.q[2]<.12:violations['root_height_m']=float(state.q[2])
    passed=failure is None and len(rows)==450 and not violations and first_plan_exact
    for path,h in hashes.items():
        if sha(path)!=h:raise ValueError('Changed source during receding experiment '+path)
    result=dict(scope=__doc__,configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        source_sha256=hashes,passed_model_receding_recovery=passed,failure=failure,terminal=terminal,
        terminal_violations=violations,trailing_window_s=stable*.01,longest_window_s=longest*.01,
        first_plan_exact=first_plan_exact,stress=stress,calls=calls,trajectory=rows,
        final_plan=plan,wall_seconds=time.monotonic()-began,native_execution=False,valid_expert_labels=False,
        limitations=['All dynamics and state perturbations here are model-only.',
            'A full-remainder prediction is not recursive feasibility or a safety certificate.',
            'Actual native execution, timing and teacher-off learning remain separate requirements.'])
    a.output.write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n');checkpoint()


if __name__=='__main__':main()
