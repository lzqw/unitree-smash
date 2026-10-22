"""Observable, contiguous full-COM flight and stable-recovery accounting."""
import torch

FLOAT_FIELDS=('flight_start_com','flight_apex_rise','flight_clear_time','max_com_rise','airborne_time',
    'recovery_run','recovery_time','maximum_body_ground_force','takeoff_time','landing_time')
BOOL_FIELDS=('support_seen','flight_candidate','flight_qualified','valid_jump_landed','airborne_seen',
    'landed_once','recovery_seen','body_ground_seen','rewarded_jump','rewarded_success','rewarded_body_contact')

def initialize(state,n,device):
    for name in FLOAT_FIELDS:setattr(state,name,torch.zeros(n,device=device))
    for name in BOOL_FIELDS:setattr(state,name,torch.zeros(n,dtype=torch.bool,device=device))

def reset(state,ids):
    for name in FLOAT_FIELDS:getattr(state,name)[ids]=0.
    for name in BOOL_FIELDS:getattr(state,name)[ids]=False

def advance(state,*,dt,time,com_height,foot_contact,foot_clearance,roll_pitch,
            horizontal_speed,angular_speed,joint_speed,body_ground_force,protocol):
    support=foot_contact.any(1)
    takeoff=(~support)&state.support_seen&(~state.flight_candidate)
    state.flight_start_com[takeoff]=com_height[takeoff]
    state.flight_apex_rise[takeoff]=0.
    state.flight_clear_time[takeoff]=0.
    state.flight_qualified[takeoff]=False
    state.valid_jump_landed[takeoff]=False
    state.recovery_seen[takeoff]=False
    state.recovery_run[takeoff]=0.
    state.flight_candidate |= takeoff
    state.takeoff_time[takeoff]=time[takeoff]
    state.support_seen |= support
    clear=(~support)&(foot_clearance>.035)&state.flight_candidate
    state.flight_clear_time.copy_(torch.where(clear,state.flight_clear_time+dt,0.))
    state.airborne_seen |= state.flight_clear_time>=.02-1e-7
    rise=torch.where(state.flight_candidate & (~support),com_height-state.flight_start_com,0.)
    torch.maximum(state.flight_apex_rise,rise,out=state.flight_apex_rise)
    torch.maximum(state.max_com_rise,rise,out=state.max_com_rise)
    torch.maximum(state.airborne_time,state.flight_clear_time,out=state.airborne_time)
    # Height and flight duration must belong to the SAME uninterrupted flight.
    state.flight_qualified |= (state.flight_apex_rise>=protocol['jump']['minimum_total_com_rise_from_takeoff_m']) & (
        state.flight_clear_time>=protocol['jump']['minimum_continuous_all_feet_airborne_s']-1e-7)
    landing=support&state.flight_candidate
    state.landing_time[landing]=time[landing]
    state.landed_once |= landing & state.airborne_seen
    state.valid_jump_landed |= landing & state.flight_qualified
    state.flight_candidate[landing]=False
    recovery=protocol['recovery']
    torch.maximum(state.maximum_body_ground_force,body_ground_force,out=state.maximum_body_ground_force)
    state.body_ground_seen |= body_ground_force>recovery['maximum_body_ground_force_n']
    stable=(state.landed_once & support & (roll_pitch.abs().max(1).values<=recovery['maximum_abs_roll_pitch_deg']*torch.pi/180)
        & (horizontal_speed<=recovery['maximum_horizontal_root_speed_m_s'])
        & (angular_speed<=recovery['maximum_root_angular_speed_rad_s'])
        & (joint_speed<=recovery['maximum_abs_joint_velocity_rad_s'])
        & (~state.body_ground_seen))
    state.recovery_run.copy_(torch.where(stable,state.recovery_run+dt,0.))
    torch.maximum(state.recovery_time,state.recovery_run,out=state.recovery_time)
    state.recovery_seen |= state.recovery_run>=recovery['continuous_stable_window_s']-1e-6
    return clear,stable

def success(state):
    return state.valid_jump_landed & state.recovery_seen & (~state.body_ground_seen)
