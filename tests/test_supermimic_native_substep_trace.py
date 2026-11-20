"""CPU hook contracts; not evidence of a real PhysX run."""
from types import SimpleNamespace
import pytest
import torch
from quadsmash.rl.supermimic_native_substep_trace_v1 import NativeSubstepTraceV1


class Env:
    def __init__(self):
        n,b,j=2,3,18
        self.device='cpu';self.cfg=SimpleNamespace(decimation=3);self.physics_dt=.001
        self._sim_step_counter=10;self._last_physics_frame=10;self.calls=[]
        self.racket_body_id=1;self.joint_ids=list(range(j));self.ball_clock=torch.zeros(n)
        data=SimpleNamespace(body_link_pos_w=torch.zeros(n,b,3),body_link_quat_w=torch.zeros(n,b,4),
            body_link_lin_vel_w=torch.zeros(n,b,3),body_link_ang_vel_w=torch.zeros(n,b,3),
            joint_pos=torch.zeros(n,j),joint_vel=torch.zeros(n,j),applied_torque=torch.zeros(n,j))
        self.robot=SimpleNamespace(data=data)
        self.shuttle=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(n,13)))
        self.shuttle_contact=SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.zeros(n,1,2,3),
            contact_pos_w=torch.full((n,1,2,3),float('nan'))))
    def _consume_physics_frame(self,frame):
        self.calls.append(frame)
        if frame>self._last_physics_frame:
            self._last_physics_frame=frame;self.ball_clock+=self.physics_dt
            self.robot.data.joint_pos+=1
        return 'original-result'


def test_original_called_once_duplicate_frames_not_recorded_and_hook_restored():
    env=Env();trace=NativeSubstepTraceV1(env)
    for frame in (10,11,11,12,13):
        assert env._consume_physics_frame(frame)=='original-result'
    result=trace.finish()
    assert env.calls==[10,11,11,12,13]
    assert '_consume_physics_frame' not in vars(env)
    assert result['tensors']['physics_frame'].tolist()==[10,11,12,13]
    assert result['tensors']['joint_position'][:,0,0].tolist()==[0,1,2,3]
    assert not result['tensors']['racket_contact_position_valid'].any()
    assert torch.isnan(result['tensors']['racket_contact_position_w']).all()
    env.robot.data.joint_pos+=99
    assert result['tensors']['joint_position'][-1,0,0]==3


def test_incomplete_tick_fails_and_restores_instance_method():
    env=Env();original=env._consume_physics_frame;env._consume_physics_frame=original
    trace=NativeSubstepTraceV1(env);env._consume_physics_frame(11)
    with pytest.raises(RuntimeError,match='Incomplete native substep'):trace.finish()
    assert env._consume_physics_frame is original


def test_trace_overflow_does_not_silently_drop_data():
    env=Env();trace=NativeSubstepTraceV1(env)
    try:
        for frame in (11,12,13):env._consume_physics_frame(frame)
        with pytest.raises(RuntimeError,match='capacity exceeded'):env._consume_physics_frame(14)
    finally:trace.close()


def test_gap_cannot_be_reported_as_complete_native_tick():
    env=Env();trace=NativeSubstepTraceV1(env)
    for frame in (11,13,14):env._consume_physics_frame(frame)
    with pytest.raises(RuntimeError,match='nonconsecutive'):trace.finish()


def test_sparse_manifold_uses_each_filter_start_and_count():
    from quadsmash.rl.supermimic_native_substep_trace_v1 import pack_manifold_frame
    values = torch.arange(7).float()
    points = values[:, None].expand(-1, 3).clone()
    result = pack_manifold_frame((
        values[:, None], points, -points, (-values)[:, None],
        torch.tensor([[2, 0], [1, 1]]), torch.tensor([[4, 999], [2, 0]])))
    assert result["lane"].tolist() == [0, 0, 1, 1]
    assert result["filter_index"].tolist() == [0, 0, 0, 1]
    assert result["normal_force_n"].tolist() == [4., 5., 2., 0.]
    points.fill_(99.)
    assert result["point_w"][:, 0].tolist() == [4., 5., 2., 0.]


def test_manual_frames_have_a_physics_clock_without_installing_task_hook():
    env = Env()
    trace = NativeSubstepTraceV1(env, start_frame=60)
    for frame in (61, 62, 63):
        trace.observe_frame(frame)
    result = trace.finish()['tensors']
    assert env.calls == []
    assert '_consume_physics_frame' not in vars(env)
    torch.testing.assert_close(result['ball_clock'][:, 0], torch.arange(4) * .001)
