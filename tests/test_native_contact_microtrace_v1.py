from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from scripts.native_contact_microtrace_v1 import NativeContactMicrotraceV1


def environment():
    data = NS(joint_pos=torch.zeros(1,18), joint_vel=torch.zeros(1,18),
              root_link_pos_w=torch.zeros(1,3), root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]),
              root_com_lin_vel_w=torch.zeros(1,3), root_com_ang_vel_w=torch.zeros(1,3),
              body_com_pos_b=torch.zeros(1,4,3), body_pos_w=torch.zeros(1,4,3),
              body_quat_w=torch.tensor([[[1.,0.,0.,0.]]*4]))
    names=['FL_foot','FR_foot','RL_foot','RR_foot']
    e=NS(num_envs=1,joint_ids=list(range(18)),physics_dt=.0000625,
         robot=NS(data=data,body_names=names,joint_names=[str(i) for i in range(18)]),
         feet_contact=NS(body_names=names,data=NS(net_forces_w=torch.zeros(1,4,3))),
         scene=NS(env_origins=torch.zeros(1,3)),targets=torch.ones(1,18),
         observed_foot_contact=torch.zeros(1,4,dtype=torch.bool),episode_index=torch.zeros(1,dtype=torch.int64),
         _observation_physics_frame=-1,callback_count=0)
    def original(frame):
        e.callback_count+=1
        if frame>e._observation_physics_frame:
            e._observation_physics_frame=frame
            force=e.feet_contact.data.net_forces_w.norm(dim=-1)
            e.observed_foot_contact.copy_(torch.where(e.observed_foot_contact,force>.5,force>1.))
        return frame
    e._observe_physics_frame=original
    return e


def test_records_substep_pulse_without_changing_observer_or_command(tmp_path):
    e=environment();plain=environment()
    r=NativeContactMicrotraceV1(e,tmp_path,lane=0,first_frame=1,last_frame=3)
    for frame,force in [(0,0.),(1,0.),(2,10.),(2,10.),(3,0.),(4,0.)]:
        for env in [e,plain]:
            env.feet_contact.data.net_forces_w[0,2,2]=force
            assert env._observe_physics_frame(frame)==frame
        assert torch.equal(e.observed_foot_contact,plain.observed_foot_contact)
        assert torch.equal(e.targets,plain.targets)
    assert e.callback_count==plain.callback_count==6
    e.feet_contact.data.net_forces_w.fill_(999.)
    result=r.save()
    assert result['complete_window'] and result['samples']==3
    with np.load(tmp_path/'contact_microtrace.npz') as x:
        assert x['physics_frame'].tolist()==[1,2,3]
        assert x['foot_normal_force_world_n'][:,2,2].tolist()==[0.,10.,0.]
        assert x['observed_foot_contact'][:,2].tolist()==[False,True,False]
    original=r.original;r.detach()
    assert e._observe_physics_frame is original


def test_missing_frames_and_empty_windows_are_not_complete(tmp_path):
    e=environment();r=NativeContactMicrotraceV1(e,tmp_path,lane=0,first_frame=1,last_frame=3)
    assert not r.save()['complete_window']
    e._observe_physics_frame(1);e._observe_physics_frame(3)
    assert not r.save()['complete_window']


def test_episode_reset_is_preserved_not_filtered_into_first_episode(tmp_path):
    e=environment();r=NativeContactMicrotraceV1(e,tmp_path,lane=0,first_frame=1,last_frame=2)
    e._observe_physics_frame(1);e.episode_index.fill_(1);e._observe_physics_frame(2);r.save()
    with np.load(tmp_path/'contact_microtrace.npz') as x:
        assert x['episode_index'].tolist()==[0,1]


def test_double_install_and_replaced_hook_rejected(tmp_path):
    e=environment();r=NativeContactMicrotraceV1(e,tmp_path,lane=0,first_frame=1,last_frame=2)
    with pytest.raises(ValueError,match='already attached'):
        NativeContactMicrotraceV1(e,tmp_path/'other',lane=0,first_frame=1,last_frame=2)
    e._observe_physics_frame=lambda f:None
    with pytest.raises(ValueError,match='Another hook'):
        r.detach()


def test_sensor_order_mismatch_rejected(tmp_path):
    e=environment();e.feet_contact.body_names=list(reversed(e.feet_contact.body_names))
    with pytest.raises(ValueError,match='order'):
        NativeContactMicrotraceV1(e,tmp_path,lane=0,first_frame=1,last_frame=2)
