#!/usr/bin/env python3
"""Diagnose the later predicted failures after bounded arm feasibility repair.

Only recorded-state CPU rollouts. Removing a family is diagnostic and never
executed; predicted failure is not an observed native failure or hardware proof.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from quadsmash.rl.episode_trace_v4 import unpack_public
from scripts.landing_arm_feasibility_mpc_v1 import LandingArmFeasibilityMPCV1
from scripts.landing_support_mpc_v2 import LandingMPCSettingsV2
from scripts.whole_body_support_qp_v1 import PositionPDInterface,SupportSettings
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.audit_interval_constraint_conflicts_v1 import feasibility
import scripts.interval_command_support_qp_v1 as interval_module


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve evidence')
    selection=json.loads(a.selection.read_text())
    for name,value in selection['source_sha256'].items():
        if sha(name)!=value:raise ValueError('Selection source changed: '+name)
    selected={tuple(k) for k in selection['recorded_panel']['candidate_first_step_accepted']['gained']}
    if Path(selection['source']).resolve()!=a.run.resolve():raise ValueError('Wrong native source')
    read=lambda f:json.loads((a.run/f).read_text())
    if read('job.json')['status']!='SUCCEEDED':raise ValueError('Completed recorded source required')
    native=read('native_command_inputs.json');default=np.asarray(native['default_joint_position_rad'])
    interface=PositionPDInterface(np.asarray(native['joint_limits_rad']),
        np.r_[np.full(12,25.),np.full(6,80.)],np.r_[np.full(12,.5),np.full(6,4.)],
        np.r_[np.full(12,23.5),np.full(6,100.)],np.r_[np.full(12,30.),np.full(6,5.)],.65)
    cfg=SupportSettings(solver_eps=1e-6,residual_tolerance=3e-4,
        swing_foot_weight=1.,swing_foot_kp_s2=60.,swing_foot_kd_s=12.,swing_foot_clearance_m=.005,
        nonfoot_clearance_weight=5.,nonfoot_clearance_m=.05,nonfoot_clearance_kp_s2=60.,nonfoot_clearance_kd_s=15.)
    settings=LandingMPCSettingsV2(horizon_steps=40,minimum_support_feet=1,
        one_foot_max_com_distance_m=.20,landing_leg_posture_blend=.15,
        landing_arm_posture_blend=0.,executor_replay=True,predict_touchdown_contacts=True,contact_mode_fallback=True)
    mpc=LandingArmFeasibilityMPCV1(interface,qp_settings=cfg,settings=settings,arm_target_rate_rad_s=10.)
    mpc.set_posture_target(default)
    logs=read('landing_mpc_records.json');public={};paths=[]
    for case in sorted({x['case'] for x in logs}):
        path=a.run/'fixed_policy_traces'/f'case_{case:03d}.npz';paths.append(path)
        with np.load(path) as z:public[case]=unpack_public(z['packed_public'])
    captured={};labels=[];failures=[]
    original_constraints=interval_module.interval_constraints
    def capture_constraints(*args,**kwargs):
        result=original_constraints(*args,**kwargs)
        labels[:]=[kind.rsplit('_',1)[0] if kind.startswith(('normal_force_','friction_'))
                   else kind for _,kind,_ in result['labels']]
        return result
    class CaptureSolver(DenseActiveSetQP):
        def setup(self,**kwargs):
            captured.clear();captured.update({k:kwargs[k].copy() for k in ('A','l','u')})
            captured['interval_labels']=list(labels)
            return super().setup(**kwargs)
    mpc.qp._make_solver=CaptureSolver
    original_solve=mpc.qp.solve
    def record_solve(state,reference,**kwargs):
        result=original_solve(state,reference,**kwargs)
        if not result['accepted']:
            failures.append(dict(state=copy.deepcopy(state),matrix=copy.deepcopy(captured),
                                 result={k:result.get(k) for k in ['solver_status','rejection','residuals',
                                     'predicted_minimum_root_height_m','arm_feasibility_fallback']}))
        return result
    mpc.qp.solve=record_solve;interval_module.interval_constraints=capture_constraints
    output=[]
    try:
        for row in logs:
            case,tick=row['case'],row['control_tick']
            if row['episode_index']!=0:raise ValueError('First episode required')
            state=mpc.qp.model.state_from_public(public[case][tick],default)
            state.contacts=np.asarray(row['support_contacts'],dtype=bool)
            state.measured_force_world=np.asarray(row['measured_foot_force_world'])
            state.previous_command_target=np.asarray(row['previous_dispatched_target'])
            ref=mpc.admit_command_target(state,np.asarray(row['requested_student_joint_reference']))
            mpc.observe(state,ref,lane=case,control_tick=tick)
            if (case,tick) not in selected:continue
            failures.clear();plan=mpc.solve(state,ref,lane=case,control_tick=tick,force=True)
            rejected=next((s for s in plan.get('horizon',[]) if not s['accepted']),None)
            item=dict(case=case,tick=tick,accepted=bool(plan['accepted']),rejection=plan.get('rejection'),
                terminal=plan.get('terminal_risk'),terminal_support_ok=plan.get('support_margin_terminal_ok'))
            if rejected is not None:
                entry=next(x for x in reversed(failures) if x['state'].contacts.tolist()==rejected['contacts'])
                replayed,_=mpc.replay_horizon(state,plan['horizon'])
                np.testing.assert_allclose(entry['state'].q,replayed.q,atol=1e-10,rtol=0.)
                np.testing.assert_allclose(entry['state'].v,replayed.v,atol=1e-10,rtol=0.)
                matrix=entry['matrix'];n_extra=len(matrix['l'])-len(matrix['interval_labels'])-18-16
                if n_extra not in (0,6):raise ValueError('Unexpected constraint layout')
                families=np.asarray(matrix['interval_labels']+['native_target_bounds']*18+
                                    ['preserve_arm']*n_extra+['root_height']*16)
                check=feasibility(matrix,np.ones(len(families),dtype=bool))
                deletion={str(group):feasibility(matrix,families!=group) for group in sorted(set(families))}
                predicted=entry['state'];contacts=predicted.contacts
                snapshot=mpc.qp.model.snapshot(predicted.q,predicted.v)
                feet=[mpc.qp.model.names.index(x) for x in mpc.qp.model.feet]
                item.update(failed_stage=rejected['stage'],predicted_contacts=contacts.tolist(),
                    failed_qp=entry['result'],linear_feasibility=check,diagnostic_family_removal=deletion,
                    predicted_root_height_m=float(predicted.q[2]),predicted_joint_position_rad=predicted.joint_position.tolist(),
                    predicted_joint_velocity_rad_s=predicted.joint_velocity.tolist(),joint_order=list(mpc.qp.model.joint_order),
                    joint_limits_rad=interface.limits.tolist(),
                    predicted_foot_positions_world=snapshot['body_positions_world'][feet].tolist(),
                    prescribed_arm_constraint_present=bool(n_extra))
            output.append(item);print(json.dumps(item),flush=True)
    finally:
        interval_module.interval_constraints=original_constraints
    if len(output)!=len(selected):raise ValueError('Missing selected states')
    paths += [Path(__file__),a.selection]+[a.run/n for n in ['landing_mpc_records.json','native_command_inputs.json','job.json']]
    paths += [ROOT/'scripts'/n for n in ['landing_arm_feasibility_mpc_v1.py','interval_command_support_qp_v1.py',
        'landing_interval_command_mpc_v1.py','condensed_implicit_drive_v1.py','audit_interval_constraint_conflicts_v1.py']]
    summary=dict(samples=len(output),later_qp_failures=sum('failed_stage' in x for x in output),
                 terminal_failures=sum('failed_stage' not in x and not x['accepted'] for x in output))
    result=dict(scope=__doc__,source=str(a.run.resolve()),native_execution=False,summary=summary,records=output,
        source_sha256={str(x.resolve()):sha(x) for x in paths})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
