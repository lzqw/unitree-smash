"""Instrumentation checks only; these do not simulate head-ground contact."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from scripts.native_head_ground_observation_v1 import NativeHeadGroundObservationV1


def environment():
    identities = [(i, name) for i in range(128) for name in ['Head_lower', 'Head_upper']][::-1]
    class View:
        filter_count = 1
        sensor_paths = [f'/World/envs/env_{i}/Robot/Go2/{name}' for i, name in identities]
        def get_contact_force_matrix(self, dt):
            assert dt == .0000625
            value = torch.zeros(256,1,3)
            for row, (i, name) in enumerate(identities): value[row,0,2] = 10*i+(name=='Head_upper')
            return value
    class Sim:
        def create_rigid_contact_view(self, pattern, filter_patterns, max_contact_data_count):
            assert filter_patterns == ['/World/Ground/geometry/mesh']
            assert max_contact_data_count == 4096
            return View()
    env = SimpleNamespace(num_envs=128, physics_dt=.0000625, device='cpu',
        robot=SimpleNamespace(body_names=['Head_upper', 'base', 'Head_lower'],
            data=SimpleNamespace(body_link_pos_w=torch.zeros(128,3,3))),
        feet_contact=SimpleNamespace(_physics_sim_view=Sim()),
        episode_index=torch.zeros(128,dtype=torch.int64), calls=[])
    def original(frame):
        env.calls.append(frame); env._observation_physics_frame=frame
        return frame+5
    env._observe_physics_frame=original
    return env


def test_path_mapping_clock_duplicates_and_original_hook(tmp_path):
    env = environment(); original = env._observe_physics_frame
    recorder = NativeHeadGroundObservationV1(env,tmp_path)
    for frame in [1,16,16,17,32]: assert env._observe_physics_frame(frame) == frame+5
    assert env.calls == [1,16,16,17,32]
    assert not recorder.save()['complete_window']
    with np.load(tmp_path/'head_ground_observation.npz') as data:
        np.testing.assert_array_equal(data['physics_frame'],[16,32])
        np.testing.assert_array_equal(data['head_ground_normal_force'][:,3,:,2],[[30,31],[30,31]])
    recorder.detach(); assert env._observe_physics_frame is original


def test_replaced_hook_and_invalid_environment_rejected(tmp_path):
    env=environment(); recorder=NativeHeadGroundObservationV1(env,tmp_path)
    with pytest.raises(ValueError): NativeHeadGroundObservationV1(env,tmp_path)
    env._observe_physics_frame=lambda _:None
    with pytest.raises(ValueError): recorder.detach()
    env=environment(); env.physics_dt=.001
    with pytest.raises(ValueError): NativeHeadGroundObservationV1(env,tmp_path)
