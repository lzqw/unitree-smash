import pytest
import torch

from scripts import supermimic_native_arrival_readout as readout


def test_arrival_uses_native_physics_frames_and_selected_lane_order(monkeypatch):
    xyz = torch.arange(18.).reshape(3, 2, 3)
    quaternions = torch.zeros(3, 2, 4)
    quaternions[..., 3] = 1.  # 180 degrees around z, face normal points along -x
    trace = dict(original_lanes=[16, 52], physics_dt_s=0.005, tensors=dict(
        physics_frame=torch.arange(91, 94), racket_quaternion_w=quaternions,
        racket_position_w=xyz, racket_linear_velocity_w=xyz+20, racket_angular_velocity_w=xyz+40))
    monkeypatch.setattr(readout, 'complete_window', lambda raw, spec, count: [trace])
    raw = {'substep_trace': {'count': 3}}
    selection = [(1, 52, 0), (0, 16, 0)]
    poses, frames = readout.measured_at_arrival(raw, {}, selection, [.25, 1.], .01)
    assert frames == [(91, 92, .5), (93, 93, 0.)]
    position = torch.stack(((xyz[0, 1]+xyz[1, 1])/2, xyz[2, 0]))
    torch.testing.assert_close(poses, torch.cat((position, position+20,
        torch.tensor([[-1., 0., 0.]]).expand(2, 3), position+40), dim=-1))
    for lane, fraction, dt in [(16, .25, .01), (52, 1.1, .01), (52, .25, .02)]:
        with pytest.raises(ValueError, match='Arrival lane or substep clock'):
            readout.measured_at_arrival(raw, {}, [(1, lane, 0)], [fraction], dt)
