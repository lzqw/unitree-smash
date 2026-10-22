"""Shared native policy interface; no simulator import is needed for history tests."""
import torch
from quadsmash.physics.native import quat_rotate

HISTORY_LENGTH = 11  # current sample and 0.10 s of history at a 10 ms policy period
FRAME_SIZE = 69
OBSERVATION_SIZE = HISTORY_LENGTH*FRAME_SIZE + 11 + 5 + 1 + 18
FRAME_FIELDS = {"root_linear_velocity_b":3,"root_angular_velocity_b":3,"gravity_b":3,
    "joint_position_offset":18,"joint_velocity":18,"tool_position_b":3,"tool_normal_b":3,
    "tool_relative_linear_velocity_b":3,"tool_relative_angular_velocity_b":3,"foot_contact":4,
    "ball_position_b":3,"ball_relative_velocity_b":3,"ball_present":1,"history_valid":1}


def inverse_rotate(q,vector):
    return quat_rotate(torch.cat((q[:,:1],-q[:,1:]),-1),vector)


def relative_velocity(q,point,velocity,origin,origin_velocity,angular_velocity):
    return inverse_rotate(q,velocity-origin_velocity-torch.linalg.cross(angular_velocity,point-origin))


class PolicyHistory:
    """At most one insertion per policy tick, independent for partial resets."""
    def __init__(self,count,device):
        self.frames=torch.zeros((count,HISTORY_LENGTH,FRAME_SIZE),device=device)
        self.last_tick=torch.full((count,),-1,dtype=torch.long,device=device)

    def reset(self,ids):
        self.frames[ids]=0.
        self.last_tick[ids]=-1

    def observe(self,frame,tick):
        ids=torch.where(self.last_tick!=tick)[0]
        if ids.numel():
            self.frames[ids,1:]=self.frames[ids,:-1].clone()
            self.frames[ids,0]=frame[ids]
            self.last_tick[ids]=tick
        return self.frames.flatten(1)


def observe_native(env):
    d=env.robot.data
    q=d.root_link_quat_w
    root=d.root_link_pos_w
    # The locked Isaac Lab COM-velocity writer does not invalidate its
    # derived root-link velocity cache. Derive this reference-point velocity
    # from the explicitly updated COM state, including after partial resets.
    omega=d.root_com_ang_vel_w
    velocity=d.root_com_lin_vel_w-torch.linalg.cross(omega,quat_rotate(q,d.body_com_pos_b[:,0]))
    tool=d.body_link_pos_w[:,env.racket_body_id]
    tool_q=d.body_link_quat_w[:,env.racket_body_id]
    normal=quat_rotate(tool_q,torch.tensor([1.,0.,0.],device=env.device).expand(env.num_envs,-1))
    fields=[inverse_rotate(q,velocity),inverse_rotate(q,omega),d.projected_gravity_b,
        (d.joint_pos-d.default_joint_pos)[:,env.joint_ids],d.joint_vel[:,env.joint_ids],
        inverse_rotate(q,tool-root),inverse_rotate(q,normal),
        relative_velocity(q,tool,d.body_link_lin_vel_w[:,env.racket_body_id],root,velocity,omega),
        inverse_rotate(q,d.body_link_ang_vel_w[:,env.racket_body_id]-omega),
        env.observed_foot_contact.float()]
    if hasattr(env,'shuttle'):
        ball=env.shuttle.data
        fields += [inverse_rotate(q,ball.root_pos_w-root),
            relative_velocity(q,ball.root_pos_w,ball.root_lin_vel_w,root,velocity,omega),
            torch.ones((env.num_envs,1),device=env.device)]
    else:
        fields += [torch.zeros((env.num_envs,7),device=env.device)]
    fields += [torch.ones((env.num_envs,1),device=env.device)]
    frame=torch.cat(fields,dim=-1)
    history=env.policy_history.observe(frame,env.common_step_counter)
    command=torch.zeros((env.num_envs,11),device=env.device)
    elapsed=env.episode_length_buf.float()*env.step_dt
    if env.cfg.event_goal_enabled:
        goal=env.goal_position_w
        command[:,0]=env.goal_time_s-elapsed
        command[:,1:4]=inverse_rotate(q,goal-root)
        command[:,4:7]=inverse_rotate(q,env.goal_normal_w)
        command[:,7:10]=relative_velocity(q,goal,env.goal_velocity_w,root,velocity,omega)
        command[:,10]=1.
    # Phase labels depend on measured dynamics/contact events. Time is a
    # separate observable clock, never an invented future contact signal.
    airborne=env.observed_airborne
    seen=env.observed_airborne_seen
    hit=getattr(env,'contact_seen',getattr(env,'virtual_event_seen',torch.zeros_like(airborne)))
    phase=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
    phase[(~seen)&(~airborne)&(velocity[:,2]>.05)]=1
    phase[airborne&(~hit)]=2
    phase[airborne&hit]=3
    phase[seen&(~airborne)&env.observed_foot_contact.any(1)]=4
    phase=torch.nn.functional.one_hot(phase,5).float()
    obs=torch.cat((history,command,phase,elapsed[:,None],env.last_action),dim=-1)
    if obs.shape!=(env.num_envs,OBSERVATION_SIZE) or not torch.isfinite(obs).all():
        raise RuntimeError('nonfinite or incompatible native policy observation')
    return {"policy":obs}
