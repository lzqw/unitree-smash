"""Controlled CPU integration of the production adapter's read-only hooks.

These fixtures do not construct Isaac or demonstrate native performance.
"""
import json
from types import SimpleNamespace

import numpy as np
import torch

from quadsmash.rl.episode_trace_v4 import EpisodeTraceRecorderV4
from quadsmash.rl.full_racket_residual_ppo_v4 import ResidualRslAdapterV4


class WrappedFixture:
    num_envs=2
    device='cpu'

    def __init__(self):
        self.history=torch.zeros(2,10,69);self.clock=torch.zeros(2);self.steps=0;self.applied=[]
        origins=torch.tensor([[128.,-256.,0.],[-128.,256.,0.]])
        positions=origins[:,None].clone()
        quaternions=torch.tensor([[[1.,0.,0.,0.]],[[1.,0.,0.,0.]]])
        self.unwrapped=SimpleNamespace(episode_index=torch.zeros(2,dtype=torch.long),
            rally_records=[],episode_records=[],contact_records=[],residual_raw=torch.zeros(2,18),
            sweep_misses=torch.zeros(2),sweep_unresolved=torch.zeros(2),
            reward_target=torch.zeros(2,3),reward_event_time=torch.zeros(2),reward_plan_valid=torch.zeros(2,dtype=torch.bool),
            robot=SimpleNamespace(data=SimpleNamespace(body_link_pos_w=positions,body_link_quat_w=quaternions)),
            shuttle=SimpleNamespace(data=SimpleNamespace(root_pos_w=origins.clone())),
            scene=SimpleNamespace(env_origins=origins))

    def get_observations(self):
        frame=torch.zeros(2,69);frame[:,0]=self.clock
        tail=torch.zeros(2,35);tail[:,16]=self.clock
        return {'policy':torch.cat([frame,self.history.flatten(1),tail],1)}

    def step(self,action):
        self.applied.append(action.clone());self.steps+=1
        frame=torch.zeros(2,69);frame[:,0]=self.clock
        self.history[:,1:]=self.history[:,:-1].clone();self.history[:,0]=frame
        self.clock+=.01
        dones=torch.zeros(2,dtype=torch.bool)
        if self.steps==2:
            env=self.unwrapped
            env.rally_records.append(dict(env_id=0,episode_index=0,complete_smash=False,numerical_failures=0))
            env.episode_records.append(dict(env_id=0,recovery_s=0.))
            env.contact_records.append(dict(env_id=0,episode_index=0,time_s=.015,string_contact=True))
            env.episode_index[0]+=1;self.history[0]=0.;self.clock[0]=0.;dones[0]=True
        env=self.unwrapped
        env.robot.data.body_link_pos_w[:,:,2]=self.steps*.01
        env.shuttle.data.root_pos_w[:,2]=self.steps*.02
        return self.get_observations(),torch.tensor([1.,2.]),dones,{}


def adapter(tmp_path,enabled):
    wrapped=WrappedFixture();result=ResidualRslAdapterV4.__new__(ResidualRslAdapterV4)
    result.wrapped=wrapped
    result.controller=SimpleNamespace(nominal_target=[0.,0.,0.],target_offset_world=torch.zeros(2,3),
        event_time=torch.zeros(2),planned=torch.zeros(2,dtype=torch.bool),
        action=lambda *args:torch.zeros(2,18),reset=lambda *args:None)
    result.mixer=SimpleNamespace(action=lambda base,residual,*args:(base+residual).clamp(-1,1),reset=lambda *args:None)
    result.default=torch.zeros(18);result.limits=torch.tensor([[-3.,3.]]*18)
    result.stop_requested=lambda:False;result.total_control_env_steps=0;result.last_record_count=0
    result.last_observation=wrapped.get_observations();result.trace_contact_cursor=0;result.trace_iteration=7
    result.trace_recorder=None
    if enabled:
        result.trace_recorder=EpisodeTraceRecorderV4(tmp_path/'trace',2,5,['base'],
            dict(max_success_traces=1,max_failure_traces=1,first_episode_lanes=[0],max_buffer_bytes=1000000),
            dict(native_source_confirmed=False,source_kind='Controlled CPU adapter fixture',training_indices=[0,1]))
    return result


def test_trace_hooks_preserve_actions_rewards_observations_and_automatic_reset(tmp_path):
    traced=adapter(tmp_path,True);plain=adapter(tmp_path,False)
    for _ in range(3):
        residual=torch.full((2,18),.2)
        left=traced.step(residual);right=plain.step(residual)
        assert torch.equal(left[0]['policy'],right[0]['policy'])
        assert torch.equal(left[1],right[1]) and torch.equal(left[2],right[2])
    assert traced.trace_recorder.disabled_error is None
    assert traced.total_control_env_steps==plain.total_control_env_steps==6
    assert all(torch.equal(a,b)for a,b in zip(traced.wrapped.applied,plain.wrapped.applied))
    assert traced.trace_recorder.lengths.tolist()==[1,3]
    path=tmp_path/'trace/lane0000_episode000000.npz';data=np.load(path)
    assert data['link_poses'].shape==(2,1,7)
    assert np.array_equal(data['link_poses'][:,0,2],np.array([0.,.01],dtype=np.float32))
    assert np.all(data['link_poses'][:,:,0:2]==0.)
    assert np.array_equal(data['native_actions'],np.full((2,18),.2,dtype=np.float32))
    meta=json.loads(path.with_suffix('.json').read_text())
    assert meta['actor_iterations_encountered']==[7]
    assert meta['actual_contact_events'][0]['time_s']==.015
    assert not meta['native_execution']


def test_recording_contract_error_is_visible_without_changing_learning_step(tmp_path):
    traced=adapter(tmp_path,True);plain=adapter(tmp_path,False)
    traced.last_observation['policy'][0,69]=1.  # Controlled inconsistent history.
    residual=torch.full((2,18),.2)
    left=traced.step(residual);right=plain.step(residual)
    assert torch.equal(left[0]['policy'],right[0]['policy']) and torch.equal(left[1],right[1])
    assert torch.equal(traced.wrapped.applied[0],plain.wrapped.applied[0])
    stats=traced.trace_recorder.statistics()
    assert not stats['enabled'] and 'losslessly' in stats['disabled_error']
    assert traced.total_control_env_steps==2
