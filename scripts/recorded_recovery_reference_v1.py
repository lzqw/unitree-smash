"""A recorded successful command sequence is only a planning proposal.

The receiving state and support contacts are never replaced by donor states.
Commands enter through the unchanged position limits and arm rate mapping.
"""
import numpy as np
import pinocchio as pin


class RecordedRecoveryReferenceV1:
    def __init__(self, model, interface, initial, targets):
        self.model, self.interface, self.initial = model, interface, initial
        self.targets = np.asarray(targets, float).copy()
        if self.targets.shape != (520, 18) or not np.isfinite(self.targets).all():
            raise ValueError('Full 280..799 command proposal required')

    def action(self, offset, state, previous):
        step = offset - 80
        if type(step) is not int or not 0 <= step < 520:
            raise ValueError('Original episode clock required')
        u = float(np.clip(step * .01 / .04, 0., 1.))
        blend = u*u*(3.-2.*u)
        raw = (1.-blend)*self.initial.previous_command_target + blend*self.targets[step]
        target = np.clip(raw, self.interface.limits[:, 0], self.interface.limits[:, 1])
        target[12:] = np.clip(target[12:], previous[12:]-.01*self.interface.arm_rate,
                             previous[12:]+.01*self.interface.arm_rate)
        return target, 'recorded_command_proposal_with_causal_feedback'


def proposal_targets(previous_targets):
    """Command at t is stored as previous target at t+1, not at t.

The bank ends before dispatch at tick 799. Its final command is explicitly
held from tick 798 as a proposal, never described as a measured tick-799 action.
"""
    values = np.asarray(previous_targets, float)
    if values.shape != (520, 18) or not np.isfinite(values).all():
        raise ValueError('Expected previous targets at ticks 280..799')
    return np.concatenate([values[1:], values[-1:]], axis=0)


def initial_distance(query, donor):
    """Heuristic ranking in explicit units; not a feasibility or safety test."""
    qa, qb = query.q, donor.q
    ra, rb = pin.Quaternion(qa[3:7]).matrix(), pin.Quaternion(qb[3:7]).matrix()
    # Heading and horizontal translation are aligned separately; do not rank
    # otherwise identical trajectories by their global origin or heading.
    ya, yb = np.arctan2(ra[1, 0], ra[0, 0]), np.arctan2(rb[1, 0], rb[0, 0])
    angle = ya-yb; c, s = np.cos(angle), np.sin(angle)
    rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    errors = dict(height_m=float(qa[2]-qb[2]), attitude_rad=pin.log3(ra @ (rz @ rb).T),
        body_linear_velocity_m_s=query.v[:3]-donor.v[:3],
        body_angular_velocity_rad_s=query.v[3:6]-donor.v[3:6],
        joint_position_rad=query.joint_position-donor.joint_position,
        joint_velocity_rad_s=query.joint_velocity-donor.joint_velocity)
    scales = dict(height_m=.05, attitude_rad=.2, body_linear_velocity_m_s=.5,
        body_angular_velocity_rad_s=1., joint_position_rad=.3, joint_velocity_rad_s=2.)
    score = sum(float(np.mean(np.square(np.asarray(errors[k])/scales[k]))) for k in errors)
    return dict(score=score, component_errors=errors, ranking_scales=scales,
                feasibility_claim=False)
