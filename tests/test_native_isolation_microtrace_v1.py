"""CPU instrumentation tests only; no physics or controller validation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.native_isolation_microtrace_v1 import NativeIsolationMicrotraceV1


def environment():
    z = lambda *shape: torch.zeros(shape)
    data = SimpleNamespace(joint_pos=z(3, 18), joint_vel=z(3, 18),
        joint_pos_target=z(3, 18), root_link_pos_w=z(3, 3),
        root_link_quat_w=z(3, 4), root_com_lin_vel_w=z(3, 3),
        root_com_ang_vel_w=z(3, 3))
    robot = SimpleNamespace(data=data, joint_names=[str(i) for i in range(18)],
        _joint_pos_target_sim=z(3,18), _joint_vel_target_sim=z(3,18),
        _joint_effort_target_sim=z(3,18), _external_force_b=z(3,2,3),
        _external_torque_b=z(3,2,3), _external_wrench_positions_b=z(3,2,3),
        has_external_wrench=False, _use_global_wrench_frame=False,
        uses_external_wrench_positions=False)
    env = SimpleNamespace(robot=robot, num_envs=3, joint_ids=list(range(18)),
        physics_dt=0.0000625, scene=SimpleNamespace(env_origins=z(3,3)),
        feet_contact=SimpleNamespace(body_names=['FL','FR','RL','RR'],
            data=SimpleNamespace(net_forces_w=z(3,4,3))),
        observed_foot_contact=torch.zeros((3,4),dtype=torch.bool),
        targets=z(3,18), episode_index=torch.zeros(3,dtype=torch.int64),
        shuttle=SimpleNamespace(data=SimpleNamespace(root_pos_w=z(3,3),root_lin_vel_w=z(3,3))))
    env.calls = []
    def original(frame):
        env.calls.append(frame)
        env._observation_physics_frame = frame
        return frame+7
    env._observe_physics_frame = original
    return env


def test_hook_is_passive_and_records_distinct_submission_buffers(tmp_path):
    env = environment(); original = env._observe_physics_frame
    env.targets[2,0] = 1
    env.robot.data.joint_pos_target[2,0] = 2
    env.robot._joint_pos_target_sim[2,0] = 3
    recorder = NativeIsolationMicrotraceV1(env,tmp_path,lanes=(2,0),first_frame=10,last_frame=12)
    for frame in [9,10,10,11,12,13]:
        assert env._observe_physics_frame(frame) == frame+7
    assert env.calls == [9,10,10,11,12,13]
    assert env.targets[2,0] == 1
    assert env.robot._joint_pos_target_sim[2,0] == 3
    with np.load(tmp_path/'isolation_microtrace.npz') as z:
        np.testing.assert_array_equal(z['physics_frame'],[10,11,12])
        np.testing.assert_array_equal(z['env_position_request'][:,0,0],[1,1,1])
        np.testing.assert_array_equal(z['articulation_position_request'][:,0,0],[2,2,2])
        np.testing.assert_array_equal(z['submitted_position_target'][:,0,0],[3,3,3])
    assert recorder.save()['complete_window']
    recorder.detach(); assert env._observe_physics_frame is original


def test_missing_frame_does_not_claim_complete_window(tmp_path):
    env=environment()
    recorder=NativeIsolationMicrotraceV1(env,tmp_path,lanes=(0,),first_frame=10,last_frame=12)
    env._observe_physics_frame(10); env._observe_physics_frame(12)
    assert not recorder.save()['complete_window']
    recorder.detach()


def test_cannot_attach_twice_or_overwrite_another_hook(tmp_path):
    env=environment()
    recorder=NativeIsolationMicrotraceV1(env,tmp_path,lanes=(0,),first_frame=10,last_frame=12)
    with pytest.raises(ValueError):
        NativeIsolationMicrotraceV1(env,tmp_path,lanes=(0,),first_frame=10,last_frame=12)
    env._observe_physics_frame=lambda _:None
    with pytest.raises(ValueError):
        recorder.detach()
