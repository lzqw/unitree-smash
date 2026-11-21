"""Read measured racket kinematics at a causal arrival time from saved physics frames."""
import torch
from scripts.audit_supermimic_native_window_response import complete_window
from quadsmash.control.reactive_landing_v1 import quat_matrix


def measured_at_arrival(raw, spec, selections, fraction, control_dt, *, precontact=None):
    """Interpolate only adjacent native substeps; keep every selected lane in order."""
    count = raw['substep_trace']['count']-1
    traces = complete_window(raw, spec, count)
    poses = []
    frame_pairs = []
    for (row, lane, tick), alpha in zip(selections, fraction, strict=True):
        trace = traces[tick]
        if (trace['original_lanes'][row] != lane or abs(trace['physics_dt_s']*count-control_dt)>1e-9
                or not 0. <= alpha <= 1.):
            raise ValueError('Arrival lane or substep clock differs from recorded execution')
        tensors = trace['tensors']
        if precontact is not None:
            task = raw['task_after']
            prior = bool(raw['start']['before']['task']['had_contact'][lane])
            occurs = bool(task['had_contact'][lane])
            clock = float(tensors['ball_clock'][0,row])
            stamp = float(task['contact_time'][lane])
            if occurs and not torch.isfinite(task['contact_time'][lane]):
                raise ValueError('Contact latch has no native substep timestamp')
            precontact.append(not prior and (not occurs or clock < stamp))

        at = float(alpha)*count
        start = int(at)
        end = min(start+1, count)
        blend = at-start
        ids = [start,end]
        normal = quat_matrix(tensors['racket_quaternion_w'][ids,row])[:,:,0]
        values = torch.cat((tensors['racket_position_w'][ids,row],tensors['racket_linear_velocity_w'][ids,row],
                            normal,tensors['racket_angular_velocity_w'][ids,row]),1)
        poses.append(torch.lerp(values[0],values[1],blend))
        frame_pairs.append((int(tensors['physics_frame'][start]), int(tensors['physics_frame'][end]), blend))
    return torch.stack(poses), frame_pairs
