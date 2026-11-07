#!/usr/bin/env python3
"""Compare local command sensitivity with refreshed model propagation.

CPU model diagnostic, not native execution. The contact schedule is the
predicted nominal schedule, held identical across perturbations to isolate
continuous dynamics. No future recorded state selects a command or contact.
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
from scripts.parallel_landing_solver_v1 import build_controller
from scripts.landing_joint_sequence_proposal_v1 import knot_basis
from scripts.landing_sequence_linearization_v1 import make_nominal_profile,affine_trajectory_sequence
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.landing_differentiated_sequence_v1 import affine_differentiated_sequence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plans',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=int,nargs='+',default=[2,3,58,61])
    p.add_argument('--prefix-steps',type=int,default=4)
    p.add_argument('--differentiate-dynamics',action='store_true')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve old evidence')
    if not 1<=a.prefix_steps<=40:raise ValueError('Explicit bounded prefix required')
    data=json.loads(a.plans.read_text())
    for path,digest in data['source_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise ValueError('Changed source '+path)
    source=Path(data['baseline']);cfg=data['configuration'];default=np.asarray(cfg['posture_target'])
    limits=np.asarray(cfg['interface']['limits']);basis=knot_basis(40,9)
    selected={r['case']:r for r in data['records']};rows=[];files=[Path(__file__),a.plans]
    for case in a.cases:
        record=selected[case];tick=record['query_tick']
        if not record['proposal']['linear_proposal_solved'] or record['nominal_commands'] is None:
            raise ValueError('Require stored nominal and optimized command sequences')
        path=source/'fixed_policy_traces'/f'case_{case:03d}.npz';files.append(path)
        with np.load(path,allow_pickle=False) as z:
            obs=unpack_public(z['packed_public'])[tick];actions=z['dispatched_actions'][tick-1:tick+1]
        c=build_controller(cfg);state=c.qp.model.state_from_public(obs,default)
        state.previous_command_target=limits[:,0]+.5*(actions[0]+1.)*np.diff(limits).ravel()
        reference=limits[:,0]+.5*(actions[1]+1.)*np.diff(limits).ravel()
        plan=c.solve(state,reference,lane=case,control_tick=tick,force=True)
        profile=make_nominal_profile(c,state,plan)
        if profile is None:raise ValueError('Nominal rollout no longer reproducible')
        nominal=np.stack([n['command'] for n in profile]);saved=np.asarray(record['nominal_commands'])
        agreement=float(np.max(abs(nominal-saved)))
        if agreement>1e-9:raise ValueError('Nominal commands changed')
        delta=np.asarray(record['proposal']['commands'])-nominal
        knots=np.linalg.lstsq(basis,delta,rcond=None)[0]
        reconstruction=float(np.max(abs(basis@knots-delta)))
        if reconstruction>1e-9:raise ValueError('Stored commands do not match knot parameterization')
        affine=list(affine_differentiated_sequence(c,state,basis[:a.prefix_steps],profile[:a.prefix_steps])) if a.differentiate_dynamics else list(affine_trajectory_sequence(c,state,basis,profile))
        measured=[]
        for fraction in [0.,.1,.01,.001,.0001,.00001]:
            following=copy.deepcopy(state);velocities=[];positions=[];roots=[];active_sets=[]
            physical=[];admissions=[]
            for step in range(a.prefix_steps):
                following.contacts=profile[step]['state'].contacts.copy()
                command=nominal[step]+fraction*delta[step]
                c.qp.interface.bind_previous_command(following.previous_command_target)
                lo,hi=c.qp.interface.admitted_target_bounds(following.joint_position,.01)
                admissions.append(float(max(0.,np.max(lo-command),np.max(command-hi))))
                prediction=predict_implicit_support(c.qp.model,c.qp.interface,following,command,
                    substeps=16,contact_velocity_decay_s=c.qp.settings.contact_velocity_decay_s)
                following=prediction['state'];following.previous_command_target=command.copy()
                following.contacts=profile[step]['following'].contacts.copy()
                velocities.append(following.v.copy());positions.append(following.joint_position.copy())
                roots.append(following.root_position_world.copy());physical.append(prediction['diagnostics'])
                active_sets.append(prediction['diagnostics']['maximum_active_set_iterations'])
            measured.append(dict(fraction=fraction,velocities=np.asarray(velocities),
                positions=np.asarray(positions),roots=np.asarray(roots),
                motor_active_set_iterations=active_sets,physical_checks=physical,
                command_admission_max_rad=max(admissions)))
        zero=measured[0];probes=[]
        for m in measured[1:]:
            fraction=m['fraction'];ev=[];eq=[];ep=[]
            for step in range(a.prefix_steps):
                local=affine[(step+1)*16-1]
                ev.append(float(np.max(abs((m['velocities'][step]-zero['velocities'][step])-
                    fraction*local['velocity_map']@knots.ravel()))))
                eq.append(float(np.max(abs((m['positions'][step]-zero['positions'][step])-
                    fraction*local['joint_position_map']@knots.ravel()))))
                ep.append(float(np.max(abs((m['roots'][step]-zero['roots'][step])-
                    fraction*local['displacement_map'][:3]@knots.ravel()))))
            probes.append(dict(fraction=fraction,maximum_command_change_rad=float(np.max(abs(fraction*delta[:a.prefix_steps]))),
                velocity_error_by_step=ev,joint_position_error_by_step=eq,root_position_error_by_step=ep,
                velocity_error_over_fraction=[v/fraction for v in ev],
                motor_active_set_iterations=m['motor_active_set_iterations'],
                command_admission_max_rad=m['command_admission_max_rad'],physical_checks=m['physical_checks']))
        zero_velocity_error=max(float(np.max(abs(zero['velocities'][k]-profile[k]['following'].v))) for k in range(a.prefix_steps))
        row=dict(case=case,query_tick=tick,prefix_steps=a.prefix_steps,
            nominal_command_max_error_rad=agreement,knot_reconstruction_max_error_rad=reconstruction,
            zero_increment_velocity_max_error=zero_velocity_error,
            nominal_contact_schedule=[node['state'].contacts.tolist() for node in profile[:a.prefix_steps]],probes=probes)
        rows.append(row)
        print(json.dumps(dict(case=case,zero_velocity_error=zero_velocity_error,
            final_derivative_error=probes[-1]['velocity_error_over_fraction'][-1])),flush=True)
    files += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('scripts.') and getattr(m,'__file__',None)]
    files.append(ROOT/'configs/go2_piper_racket_landing_model_v1.json')
    result=dict(scope=__doc__,native_execution=False,solver_acceptance_changed=False,
        differentiated_interval_dynamics=a.differentiate_dynamics,records=rows,
        limitation='Comparison is within the existing contact/implicit-drive model, not against PhysX or hardware.',
        source_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
