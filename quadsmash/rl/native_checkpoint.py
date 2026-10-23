"""Versioned full native training state, with explicit physics-restore limits."""
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch

SCHEMA='quadsmash-native-training-v2'

def interface(env):
    from quadsmash.tasks.stand.observation_contract import OBSERVATION_SIZE,FRAME_FIELDS,HISTORY_LENGTH
    return dict(schema=SCHEMA,observation_size=OBSERVATION_SIZE,frame_fields=FRAME_FIELDS,
        history_length=HISTORY_LENGTH,policy_dt=env.step_dt,joints=[env.robot.joint_names[j] for j in env.joint_ids],
        asset_sha256=hashlib.sha256(Path(env.cfg.robot.spawn.usd_path).read_bytes()).hexdigest(),
        action_family=getattr(env, 'action_family',
            'stand-diagnostic' if env.__class__.__name__=='CoupledEnv' else 'jump-event-v1'),
        metrics_version='full-com-contiguous-v2' if hasattr(env,'jump_clock') else 'stand-v2')

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']])

def environment_state(env):
    tensors={k:v.detach().cpu().clone() for k,v in vars(env).items() if isinstance(v,torch.Tensor)}
    scalar_names=('common_step_counter','_sim_step_counter','_observation_physics_frame','_last_physics_frame','_jump_physics_frame',
        'completed_episodes','successful_episodes','failed_episodes','_metrics_started','checkpoint_aborted_episodes')
    state=dict(num_envs=env.num_envs,physics_dt=env.physics_dt,environment_type=env.__class__.__name__,
        root=env.robot.data.root_state_w.detach().cpu().clone(),
        joint_position=env.robot.data.joint_pos.detach().cpu().clone(),
        joint_velocity=env.robot.data.joint_vel.detach().cpu().clone(),tensors=tensors,
        scalars={k:getattr(env,k) for k in scalar_names if hasattr(env,k)},
        history=env.policy_history.frames.cpu().clone(),history_tick=env.policy_history.last_tick.cpu().clone(),
        curriculum=getattr(env,'curriculum_state',{}),
        physics_solver_cache='not exposed by PhysX; physical continuation is tested with numerical tolerance, never claimed bitwise')
    if hasattr(env,'shuttle'):
        state['shuttle']=env.shuttle.data.root_state_w.cpu().clone()
        state['contact_episodes']={k:getattr(env.contact_events,k).cpu().clone() for k in ('active','count','clear_frames')}
    return state

@torch.inference_mode()
def restore_environment(env,state):
    if state['num_envs']!=env.num_envs or state['physics_dt']!=env.physics_dt or state['environment_type']!=env.__class__.__name__:
        raise ValueError('resume environment type/count/physics step is incompatible')
    for key,value in state['tensors'].items():
        current=getattr(env,key,None)
        # DirectRLEnv creates reward_buf on the first step, after reset.
        if key=='reward_buf' and current is None and value.shape==(env.num_envs,):
            env.reward_buf=torch.zeros_like(value,device=env.device)
            current=env.reward_buf
        if not isinstance(current,torch.Tensor) or current.shape!=value.shape or current.dtype!=value.dtype:
            raise ValueError(f'incompatible checkpoint buffer: {key}')
    env.scene.reset()
    env.robot.write_root_state_to_sim(state['root'].to(env.device))
    env.robot.write_joint_state_to_sim(state['joint_position'].to(env.device),state['joint_velocity'].to(env.device))
    if 'shuttle' in state:
        env.shuttle.write_root_state_to_sim(state['shuttle'].to(env.device))
        for key,value in state['contact_episodes'].items():getattr(env.contact_events,key).copy_(value.to(env.device))
    for key,value in state['tensors'].items():getattr(env,key).copy_(value.to(env.device))
    for key,value in state['scalars'].items():setattr(env,key,value)
    env.policy_history.frames.copy_(state['history'].to(env.device))
    env.policy_history.last_tick.copy_(state['history_tick'].to(env.device))
    env.curriculum_state=state['curriculum']
    env.robot.set_joint_position_target(env.targets)
    env.scene.write_data_to_sim()
    env.sim.forward()

def training_signature(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
