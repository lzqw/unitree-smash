"""Bounded orientation commands from measured joints and a public normal goal."""
import math
import torch


def validate_face_feedback(config):
    if config is None:
        return None
    values = dict(damping=float(config.get('damping', .02)),
                  normal_weight=float(config.get('normal_weight', .3)),
                  maximum_joint_delta_rad=float(config.get('maximum_joint_delta_rad', .4)),
                  maximum_point_displacement_m=float(config.get('maximum_point_displacement_m', .03)),
                  measured_normal_lead_gain=float(config.get('measured_normal_lead_gain', .5)),
                  iterations=config.get('iterations', 3))
    if not all(math.isfinite(values[k]) for k in values if k != 'iterations'):
        raise ValueError('Finite face-feedback settings required')
    if not .01 <= values['damping'] <= .1 or not .1 <= values['normal_weight'] <= 1.:
        raise ValueError('Invalid orientation solve weights')
    if not .05 <= values['maximum_joint_delta_rad'] <= .8:
        raise ValueError('Invalid orientation command correction bound')
    if not .005 <= values['maximum_point_displacement_m'] <= .05 or not 0 <= values['measured_normal_lead_gain'] <= 1:
        raise ValueError('Invalid point-preservation or observed-normal lead setting')
    if type(values['iterations']) is not int or not 1 <= values['iterations'] <= 8:
        raise ValueError('Require1..8orientation iterations')
    return values


class FullRacketFaceFeedback(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = validate_face_feedback(config)
        self.register_buffer('eye', torch.eye(6, dtype=torch.float64))

    def forward(self, actual, desired_normal_body, limits, kinematics, contact_offset):
        """Return a joint correction, holding the supplied reference hit point.

        This is an inverse-kinematic command calculation. It does not change
        simulated state, hardware bounds, contacts, or success definitions.
        """
        cfg = self.config
        radius = cfg['maximum_joint_delta_rad']
        lower = torch.maximum(limits[:, 0], actual - radius)
        upper = torch.minimum(limits[:, 1], actual + radius)
        q = actual.clamp(limits[:, 0], limits[:, 1])
        point_target, original_normal, *_ = kinematics(actual, contact_offset)
        weight = cfg['normal_weight']
        for _ in range(cfg['iterations']):
            point, normal, jp, jn = kinematics(q, contact_offset)
            task = torch.cat((jp, weight * jn), dim=1)
            error = torch.cat((point_target - point, weight * (desired_normal_body - normal)), dim=1)
            # Re-solve on free columns when a joint reaches a command or
            # hardware bound. Clipping an unconstrained solve otherwise loses
            # the point-holding contribution of its saturated joints.
            step_lower = (lower - q).clamp_min(-.15)
            step_upper = (upper - q).clamp_max(.15)
            free = torch.ones_like(q, dtype=torch.bool)
            fixed = torch.zeros_like(q)
            for _ in range(6):
                jacobian = task * free[:, None, :]
                residual = error - (task @ fixed[:, :, None]).squeeze(-1)
                solved = torch.linalg.solve(jacobian @ jacobian.transpose(1, 2)
                                            + cfg['damping']**2 * self.eye, residual[:, :, None])
                delta = fixed + (jacobian.transpose(1, 2) @ solved).squeeze(-1)
                violated = free & ((delta < step_lower) | (delta > step_upper))
                fixed = torch.where(violated, delta.clamp(step_lower, step_upper), fixed)
                free = free & ~violated
            delta = delta.clamp(step_lower, step_upper)
            q = (q + delta).clamp(lower, upper)
        final_point, final_normal, *_ = kinematics(q, contact_offset)
        improved = ((desired_normal_body-final_normal).square().sum(1)
                    <= (desired_normal_body-original_normal).square().sum(1))
        close = torch.linalg.vector_norm(final_point-point_target, dim=1) <= cfg['maximum_point_displacement_m']
        return torch.where((improved & close)[:, None], q-actual, torch.zeros_like(q))
