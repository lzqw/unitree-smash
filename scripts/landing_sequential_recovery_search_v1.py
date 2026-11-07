"""Offline sequential local optimization, never a native control entry.

Intermediate trajectories may violate constraints and are only linearization
points. No such trajectory is authorized for execution: the independent strict
verifier, including the original terminal gate, is the final authority.
"""
import copy
import time
import numpy as np
import pinocchio as pin
from scripts.implicit_support_prediction_v1 import predict_implicit_support
from scripts.landing_differentiated_sequence_v1 import interval_outputs
from scripts.landing_sequence_linearization_v1 import world_displacement
from scripts.frame_point_kinematics_v1 import frame_point_kinematics
from scripts.landing_joint_sequence_proposal_v1 import propose_sequence, verify_sequence


def rollout_for_search(controller, initial, commands, reference):
    """Refresh all states; retain violations instead of concealing/clipping them."""
    qp=controller.qp;model=qp.model;interface=qp.interface;cfg=qp.settings
    H=controller.settings.horizon_steps;dt=controller.settings.control_dt_s
    commands=np.asarray(commands,dtype=float)
    if commands.shape!=(H,18) or not np.isfinite(commands).all():
        raise ValueError('Complete finite command sequence required')
    if controller.settings.touchdown_impact_projection:
        raise ValueError('Hybrid impact derivatives are not implemented')
    previous=copy.deepcopy(interface.previous_command)
    state=copy.deepcopy(initial);profile=[];cost=0.;checks=[]
    snap=copy.deepcopy(model.snapshot(initial.q,initial.v))
    footids=[model.names.index(name) for name in model.feet]
    feet=snap['body_positions_world'][np.asarray(footids)[initial.contacts]]
    if len(feet)==0:raise ValueError('Observed support required for initial reference')
    com_goal=snap['com_world'].copy();com_goal[:2]=feet[:,:2].mean(0)
    ground=float(np.median(feet[:,2]))
    yaw=np.arctan2(initial.root_rotation_world[1,0],initial.root_rotation_world[0,0])
    upright=pin.log3(pin.rpy.rpyToMatrix(0.,0.,yaw)@initial.root_rotation_world.T)
    angles=np.arange(cfg.friction_facets)*2*np.pi/cfg.friction_facets
    cone=np.c_[np.cos(angles),np.sin(angles),
               np.full(cfg.friction_facets,-interface.friction*np.cos(np.pi/cfg.friction_facets))]
    try:
        for tick,command in enumerate(commands):
            interface.bind_previous_command(state.previous_command_target)
            lo,hi=interface.admitted_target_bounds(state.joint_position,dt)
            admission=float(max(0.,np.max(lo-command),np.max(command-hi)))
            snapshot=copy.deepcopy(model.snapshot(state.q,state.v))
            prediction=predict_implicit_support(model,interface,state,command,
                duration_s=dt,substeps=qp.substeps,contact_velocity_decay_s=cfg.contact_velocity_decay_s)
            following=prediction['state'];following.previous_command_target=command.copy()
            following,added=controller._after_prediction(following)
            end_snapshot=copy.deepcopy(model.snapshot(following.q,following.v))
            com_jac,_=qp.com_kinematics(following);com_jac=com_jac.copy()
            transform=np.eye(24);transform[:3,:3]=following.root_rotation_world.T
            transform[3:6,3:6]=following.root_rotation_world.T
            frames=frame_point_kinematics(model,following.q,following.v,
                ('FL_thigh','FR_thigh','RL_thigh','RR_thigh'))
            node=dict(state=copy.deepcopy(state),following=copy.deepcopy(following),snapshot=snapshot,
                command=command.copy(),end_snapshot=end_snapshot,com_jac_world=com_jac@transform,
                world_to_local=transform,frames=frames,
                displacement=world_displacement(model,initial,following.q),added_contacts=added,
                internal_search_only=True)
            values,_=interval_outputs(controller,initial,node,np.zeros(66))
            nf=int(state.contacts.sum());velocity=values[:,model.v_indices]
            joint=values[:,24:42];force=values[:,66:66+3*nf].reshape(-1,nf,3)
            torque=values[:,66+3*nf:]
            violation=dict(command_admission_rad=admission,
                motor_torque_nm=float(max(0.,np.max(abs(torque)-interface.effort))),
                joint_velocity_rad_s=float(max(0.,np.max(abs(velocity)-interface.velocity))),
                joint_position_rad=float(max(0.,np.max(interface.limits[:,0]-joint),np.max(joint-interface.limits[:,1]))),
                joint_damper_rad_s=float(max(0.,np.max(qp.damper_gain_s_inv*(interface.limits[:,0]-joint)-velocity),
                    np.max(velocity+qp.damper_gain_s_inv*(joint-interface.limits[:,1])))),
                normal_force_n=float(max(0.,np.max(-force[:,:,2],initial=0.))),
                friction_n=float(max(0.,np.max(force@cone.T,initial=0.))),
                root_height_m=float(max(0.,qp.minimum_root_height_m-np.min(initial.q[2]+values[:,44]))))
            # Scales organize the search merit only; physical bounds and final
            # tolerances are not changed. Report every unscaled violation.
            scales=dict(command_admission_rad=.1,motor_torque_nm=23.5,joint_velocity_rad_s=5.,
                joint_position_rad=.1,joint_damper_rad_s=5.,normal_force_n=100.,friction_n=100.,root_height_m=.1)
            normalized=sum(violation[k]/scales[k] for k in violation)
            checks.append(dict(physical=violation,normalized_l1=normalized))
            w=10. if tick==H-1 else 1./H
            cost += .5*(.02/H*np.sum((command-reference)**2)
                +.2/H*np.sum((command-state.previous_command_target)**2)
                +100.*w*np.sum((end_snapshot['com_world']-com_goal)**2)
                +5.*w*np.sum((com_jac@following.v)**2)
                +2.*w*np.sum(following.v[3:6]**2)
                +10.*w*np.sum((node['displacement'][3:6]-upright)**2))
            for frame in frames.values():
                cost += .5*cfg.nonfoot_clearance_weight*w*max(0.,ground+cfg.nonfoot_clearance_m-frame['position'][2])**2
            profile.append(node);state=following
        risk=controller.risk(state)
        terminal=bool(controller._terminal_ok(state,risk))
        envelope=max(max(row['physical'].values()) for row in checks)
        # A maximum plus average prevents hiding a brief severe violation in
        # a long horizon. The merit never grants acceptance for native use.
        infeasibility=max(row['normalized_l1'] for row in checks)+sum(row['normalized_l1'] for row in checks)/H
        return dict(profile=profile,checks=checks,cost=cost,infeasibility=infeasibility,
            maximum_physical_violation=envelope,terminal_risk=risk,terminal_ok=terminal,
            physically_feasible=envelope<=cfg.residual_tolerance,native_execution=False)
    finally:
        interface.previous_command=previous


def sequential_recovery_search(controller,state,reference,nominal_commands,*,iterations=6,time_limit_s=180.):
    """Relinearize accepted merit steps, strictly verify final candidates."""
    if type(iterations) is not int or not 1<=iterations<=12 or not 0<time_limit_s<=600:
        raise ValueError('Bounded offline search required')
    start=time.perf_counter();commands=np.asarray(nominal_commands,dtype=float).copy()
    current=rollout_for_search(controller,state,commands,reference)
    initial_verification=verify_sequence(controller,state,commands)
    trace=[];radius=.1;penalty=1000.
    best_commands=None;best_verification=None;best_cost=np.inf
    for iteration in range(iterations):
        if time.perf_counter()-start>=time_limit_s:break
        old_merit=current['cost']+penalty*current['infeasibility']
        proposal=propose_sequence(controller,state,reference,nominal_profile=current['profile'],
            differentiate_dynamics=True,solver_backend='active_set',time_limit_s=min(20.,time_limit_s),
            local_increment_bound_rad=radius)
        entry=dict(iteration=iteration,old_cost=current['cost'],old_infeasibility=current['infeasibility'],
            trust_radius_rad=radius,linear_solved=proposal['linear_proposal_solved'],
            solver_status=proposal.get('solver_status'),trials=[],accepted_search_step=False)
        trace.append(entry)
        if not proposal['linear_proposal_solved']:
            entry['stopped']='Hard local subproblem has no accepted solution';break
        delta=proposal['commands']-commands;choice=None
        for fraction in [1.,.5,.25,.125,.0625,.03125]:
            trial_commands=commands+fraction*delta
            try:
                trial=rollout_for_search(controller,state,trial_commands,reference)
            except (ValueError,np.linalg.LinAlgError) as error:
                entry['trials'].append(dict(fraction=fraction,error=str(error)));continue
            merit=trial['cost']+penalty*trial['infeasibility']
            summary=dict(fraction=fraction,cost=trial['cost'],infeasibility=trial['infeasibility'],
                merit=merit,maximum_physical_violation=trial['maximum_physical_violation'],
                physically_feasible=trial['physically_feasible'],terminal_ok=trial['terminal_ok'])
            entry['trials'].append(summary)
            if trial['physically_feasible'] and trial['terminal_ok']:
                checked=verify_sequence(controller,state,trial_commands)
                summary['strict_verification_accepted']=checked['accepted']
                if checked['accepted'] and trial['cost']<best_cost:
                    best_commands=trial_commands.copy();best_verification=checked;best_cost=trial['cost']
            if merit<old_merit-1e-7*max(1.,abs(old_merit)):
                choice=(trial_commands,trial,fraction);break
        if choice is None:
            radius*=.5;entry['stopped']='No merit reduction; shrink trust region'
            if radius<.003125:break
        else:
            commands,current,fraction=choice
            entry.update(accepted_search_step=True,chosen_fraction=fraction,
                new_cost=current['cost'],new_infeasibility=current['infeasibility'])
    return dict(native_execution=False,accepted=best_commands is not None,
        initial_verification_accepted=initial_verification['accepted'],
        newly_model_feasible=bool(best_commands is not None and not initial_verification['accepted']),
        commands=best_commands,verification=best_verification,iterations=trace,
        final_search_commands=commands,final_search_cost=current['cost'],
        final_search_infeasibility=current['infeasibility'],
        final_search_terminal_risk=current['terminal_risk'],
        wall_seconds=time.perf_counter()-start,
        warning='Search iterates can be infeasible; only commands with strict verification are candidates. No native proof.')

def restore_feasibility(controller,state,reference,commands,*,iterations=8):
    """Project a search iterate towards the original nonlinear constraints.

    The local QP minimizes increment norm, without the task cost that can
    compete with feasibility. All proposals remain search-only until the
    separate verifier accepts both physics and terminal recovery.
    """
    if type(iterations) is not int or not 1<=iterations<=12:
        raise ValueError('Bounded restoration required')
    commands=np.asarray(commands,dtype=float).copy()
    current=rollout_for_search(controller,state,commands,reference)
    trace=[];best=None;radius=.1
    for iteration in range(iterations):
        checked=verify_sequence(controller,state,commands)
        if checked['accepted'] and current['physically_feasible']:
            best=checked;break
        if current['physically_feasible']:
            # Feasibility alone cannot repair a failed terminal objective.
            break
        proposal=propose_sequence(controller,state,reference,nominal_profile=current['profile'],
            differentiate_dynamics=True,solver_backend='active_set',time_limit_s=20.,
            local_increment_bound_rad=radius,feasibility_restoration=True)
        row=dict(iteration=iteration,old_infeasibility=current['infeasibility'],
            old_cost=current['cost'],linear_solved=proposal['linear_proposal_solved'],
            solver_status=proposal.get('solver_status'),trials=[],accepted_search_step=False)
        trace.append(row)
        if not proposal['linear_proposal_solved']:break
        direction=proposal['commands']-commands;choice=None
        for fraction in [1.,.5,.25,.125,.0625,.03125]:
            candidate=commands+fraction*direction
            try: trial=rollout_for_search(controller,state,candidate,reference)
            except (ValueError,np.linalg.LinAlgError) as error:
                row['trials'].append(dict(fraction=fraction,error=str(error)));continue
            row['trials'].append(dict(fraction=fraction,infeasibility=trial['infeasibility'],
                maximum_physical_violation=trial['maximum_physical_violation'],
                cost=trial['cost'],terminal_ok=trial['terminal_ok']))
            if trial['infeasibility']<current['infeasibility']-1e-10:
                choice=(candidate,trial,fraction);break
        if choice is None:
            radius*=.5
            if radius<.003125:break
        else:
            commands,current,fraction=choice
            row.update(accepted_search_step=True,fraction=fraction,
                new_infeasibility=current['infeasibility'],new_cost=current['cost'])
    checked=verify_sequence(controller,state,commands)
    accepted=bool(checked['accepted'] and current['physically_feasible'])
    return dict(accepted=accepted,native_execution=False,
        commands=commands if accepted else None,verification=checked,
        iterations=trace,final_search_commands=commands,
        final_search_infeasibility=current['infeasibility'],final_search_cost=current['cost'],
        final_search_terminal_risk=current['terminal_risk'],
        physically_feasible=current['physically_feasible'],
        maximum_physical_violation=current['maximum_physical_violation'])
