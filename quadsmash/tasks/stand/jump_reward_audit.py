"""Pure reward decomposition and an UNEXECUTED qualified-recovery draft.

These functions never change acceptance state or the environment reward.
Components are signed, weighted rewards per policy step (not rates).
"""
import torch

DRAFT_VERSION = 'qualified-recovery-draft-v1'


def com_reference(time, reset_height):
    return reset_height - .06 * torch.exp(-((time - .93) / .12).square()) \
        + .14 * torch.exp(-((time - 1.25) / .19).square())


def v2c_components(features, dt):
    f = features
    return {
        'height_tracking': 8. * f['height_tracking'] * dt,
        'flight': 3. * f['flight'] * dt,
        'stable_recovery': 3. * f['recovery'] * dt,
        'preparation': .3 * f['prepare_quality'] * dt,
        'recovery_quality': 2. * f['recovery_quality'] * dt,
        'arm_error': -.03 * f['arm_error'] * dt,
        'effort': -.00002 * f['torque_square_sum'] * dt,
        'action_delta': -.005 * f['action_delta'] * dt,
        'settling': -f['settling'] * f['settle_cost'] * dt,
        'first_qualified_jump': 2. * f['first_jump'].float(),
        'first_success': 5. * f['first_success'].float(),
        'first_body_contact': -4. * f['first_body'].float(),
        'terminal': -2. * f['terminated'].float(),
    }


def recovery_draft(components, *, valid_jump_landed, body_ground_seen,
                   recovery_quality, weak_paid, dt, weak_rate=.05, weak_cap=.1):
    """Shadow-only draft: task recovery requires a qualified current landing.

    Retain <=0.1 reward per episode of weak, safe post-landing settling. The
    caller owns a separate diagnostic accumulator; inputs are never mutated.
    Reset weak_paid ONLY on a real new episode, not on takeoff or a new hop.
    """
    if dt <= 0 or weak_rate < 0 or weak_cap < 0:
        raise ValueError('invalid weak recovery reward budget')
    gate = valid_jump_landed & ~body_ground_seen
    weak = torch.minimum((weak_cap - weak_paid).clamp(min=0.),
                         weak_rate * dt * recovery_quality.clamp(0., 1.) * (~body_ground_seen))
    result = dict(components)
    result['stable_recovery'] = components['stable_recovery'] * gate
    result['recovery_quality'] = components['recovery_quality'] * gate
    result['weak_safe_settling'] = weak
    return result, weak_paid + weak


class RecoveryWindows:
    """Read-only substep diagnostics, independent of formal success counters."""
    names = ('post_landing', 'qualified_landing', 'support', 'posture',
             'horizontal_speed', 'angular_speed', 'joint_speed', 'no_body_contact', 'all')

    def __init__(self, n, device):
        self.run = torch.zeros((n, len(self.names)), device=device)
        self.maximum = torch.zeros_like(self.run)
        self.seen = torch.zeros((n, 4), dtype=torch.bool, device=device)
        self.takeoff_velocity = torch.zeros(n, device=device)
        self._candidate = torch.zeros(n, dtype=torch.bool, device=device)

    def observe(self, state, *, dt, support, roll_pitch, horizontal_speed,
                angular_speed, joint_speed, com_vertical_velocity, protocol):
        r = protocol['recovery']
        qualified = state.valid_jump_landed & ~state.flight_candidate
        landing = state.landed_once & ~state.flight_candidate
        criteria = [landing, qualified, support,
                    roll_pitch.abs().amax(1) <= r['maximum_abs_roll_pitch_deg'] * torch.pi / 180,
                    horizontal_speed <= r['maximum_horizontal_root_speed_m_s'],
                    angular_speed <= r['maximum_root_angular_speed_rad_s'],
                    joint_speed <= r['maximum_abs_joint_velocity_rad_s'], ~state.body_ground_seen]
        all_conditions = torch.stack(criteria[2:], 1).all(1)
        mask = torch.stack([*criteria, all_conditions], 1) & landing[:, None]
        self.run.copy_(torch.where(mask, self.run + dt, 0.))
        torch.maximum(self.maximum, self.run, out=self.maximum)
        self.seen |= torch.stack((state.airborne_seen, state.flight_qualified,
                                 state.valid_jump_landed, state.valid_jump_landed & state.recovery_seen), 1)
        takeoff = state.flight_candidate & ~self._candidate
        self.takeoff_velocity[takeoff] = com_vertical_velocity[takeoff]
        self._candidate.copy_(state.flight_candidate)

    def reset(self, ids):
        self.run[ids] = 0.
        self.maximum[ids] = 0.
        self.seen[ids] = False
        self.takeoff_velocity[ids] = 0.
        self._candidate[ids] = False
