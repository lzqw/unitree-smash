"""Exact native reward recomputation from captured inputs, not model labels.

The boolean event version below is for native replay. A differentiable event
surrogate must be explicitly identified separately by its caller/model.
"""
import torch

FAULTS = ('robot_net_fault','body_ball_fault','wrong_side_fault','double_hit_fault',
          'racket_invasion_fault','body_over_net_fault','under_net_obstruction_fault',
          'engineering_body_crossing','bad_return_fault','net_blocked_fault')


def native_reward_from_record(row, racket_body_index, dt=.01):
    after=row['after'];rally=after['task'];a=after['auxiliary'];p=after['physical']
    contact=rally['legal_initial_contact'] & rally['string_hit'] & rally['airborne_string_hit']
    minimum_speed = after['recovery'].get('minimum_outgoing_speed_mps', 5.)
    attack=contact & rally['released'] & (rally['outgoing_velocity'].norm(dim=1)>=minimum_speed) & (rally['outgoing_velocity'][:,0]>0)
    cross=attack & rally['crossed_net'];target=cross & rally['landed_target']
    recover=contact & after['recovery']['rally_recovery_success']
    body=after['recovery']['body_ground_seen'];paid=row['before']['event_state'].bool()
    faulted=torch.stack([rally[name] for name in FAULTS]).any(0)
    jump=after['recovery']['flight_qualified'] & ~paid[:,0]
    hit=contact & ~paid[:,1];net=cross & ~paid[:,2];land=target & ~paid[:,3]
    recovered=recover & ~paid[:,4];body_event=body & ~paid[:,5]
    frame=rally['frame_hit'] & ~hit & ~paid[:,6];fault=faulted & ~paid[:,7]
    tool=after['body_position_w'][:,racket_body_index]
    proximity=torch.exp(-((p['ball_position_w']-tool).norm(dim=1)/.18).square())
    approach=proximity*(rally['in_play'] & ~rally['had_contact'] & ~body).float()
    f=a['reward_inputs'];head_gap=f['safety_body_height'].min(1).values.clamp_min(0.)
    risk=torch.exp(-head_gap/.04)
    stable=recover.float()*torch.exp(-4.*f['projected_gravity_b'][:,:2].square().sum(1)
        -.4*f['root_com_lin_vel_w'][:,:2].square().sum(1)-.2*f['root_com_ang_vel_w'].square().sum(1))
    continuous=(.5*stable-2.*risk-.05*a['environment']['action_delta']
        -.02*f['applied_torque_all'].square().sum(1)/1000.+.8*approach)*dt
    settled = (row['timeout'] & ~row['terminated']) | after['recovery'].get('terminated_success', False)
    complete=target & after['recovery']['rally_recovery_success'] & ~faulted & settled
    reward=continuous+2.*jump.float()+20.*hit.float()+5.*net.float()
    reward=reward+15.*land.float()+10.*recovered.float()+20.*complete.float()
    reward=reward-4.*frame.float()-5.*((~contact)&(p['elapsed_s']>3.)).float()*dt
    reward=reward-15.*fault.float()-40.*body_event.float()
    reward=torch.nan_to_num(reward,nan=-40.,posinf=40.,neginf=-40.)
    increments=torch.stack((jump,hit,net,land,recovered,body_event,frame,fault),-1)
    return reward, (paid | increments).float()
