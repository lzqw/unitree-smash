"""Opt-in TRAIN shaping from actual strings geometry and release events.

This changes training rewards only. Geometry, collision responses, attack
thresholds, legal events and independent acceptance remain unchanged. The
potential uses present measured ball/tool state, without a future forecast.
"""
import math

import torch

REWARD_VERSION = 'native-full-chain-residual-v3-measured-strings-attack'


def alignment_settings(value):
    if value is None:
        return None
    bounds = dict(contact_potential_weight=(0., 20.), normal_distance_scale_m=(1e-4, 1.),
        # Larger values are allowed for explicitly staged RL branches.  This
        # only changes the dense TRAIN shaping; the actual attack predicate
        # remains the unchanged 8 m/s, forward, downward event.
        attack_progress_weight=(0., 80.), frame_penalty=(0., 40.), discount=(0., 1.))
    if not isinstance(value, dict) or set(value) != set(bounds):
        raise ValueError('Declare exactly the measured string-alignment TRAIN settings')
    result = {}
    for name, (lower, upper) in bounds.items():
        number = value[name]
        if (isinstance(number, bool) or not isinstance(number, (int, float))
                or not math.isfinite(number) or not lower < number <= upper):
            raise ValueError('Invalid measured string-alignment TRAIN setting: '+name)
        result[name] = float(number)
    return result


def attack_aligned_target(legal_air, attack, landed_target, faulted):
    """TRAIN event gate only; never writes the actual target/attack flags."""
    return legal_air & attack & landed_target & (~faulted)


class MeasuredStringAlignmentRewardV4:
    def __init__(self, count, device, settings, *, semiaxes_yz_m, contact_plane_offset_m):
        settings = alignment_settings(settings)
        if settings is None:
            raise ValueError('Explicit opt-in settings required')
        semiaxes = torch.as_tensor(semiaxes_yz_m, dtype=torch.float32, device=device)
        if semiaxes.shape != (2,) or not torch.isfinite(semiaxes).all() or (semiaxes <= 0).any():
            raise ValueError('Require the unchanged positive finite strings semiaxes')
        if not math.isfinite(contact_plane_offset_m) or contact_plane_offset_m <= 0:
            raise ValueError('Use the actual sphere radius plus strings half-thickness')
        self.settings = settings
        self.semiaxes = semiaxes
        self.plane_offset = float(contact_plane_offset_m)
        self.previous_potential = torch.zeros(count, device=device)
        self.seen_release = torch.zeros(count, device=device, dtype=torch.bool)
        self.seen_frame = torch.zeros(count, device=device, dtype=torch.bool)
        self.last_components = torch.zeros((count, 3), device=device)
        # Diagnostics are paid contributions in this job, independent of
        # partial episode resets. They are not policy or native physics state.
        self.job_sums = torch.zeros(3, device=device, dtype=torch.float64)
        self.job_event_counts = torch.zeros(2, device=device, dtype=torch.long)

    def reset(self, ids=None):
        if ids is None:
            self.previous_potential.zero_(); self.seen_release.zero_(); self.seen_frame.zero_()
            self.last_components.zero_()
        else:
            self.previous_potential[ids] = 0.; self.seen_release[ids] = False; self.seen_frame[ids] = False
            self.last_components[ids] = 0.

    def step(self, *, clock_s, ball_relative_tool_m, plane_distance_rate_mps,
            ball_in_play, had_contact, ground_seen, body_contact, faulted,
            legal_air, released, frame_hit, outgoing_velocity_mps, terminated):
        count = len(self.previous_potential)
        flags = [ball_in_play, had_contact, ground_seen, body_contact, faulted,
            legal_air, released, frame_hit, terminated]
        if any(flag.shape != (count,) or flag.dtype != torch.bool for flag in flags):
            raise ValueError('Require one actual measured boolean per lane')
        if (ball_relative_tool_m.shape != (count, 3) or outgoing_velocity_mps.shape != (count, 3)
                or clock_s.shape != (count,) or plane_distance_rate_mps.shape != (count,)):
            raise ValueError('Require measured xyz positions/velocities and one scalar clock/rate per lane')
        if not torch.isfinite(clock_s).all():
            raise ValueError('Finite actual episode clocks required')
        finite_position = torch.isfinite(ball_relative_tool_m).all(1)
        safe_position = torch.where(finite_position[:, None], ball_relative_tool_m, torch.zeros_like(ball_relative_tool_m))
        gap = safe_position[:, 0]-self.plane_offset
        rho_squared = (safe_position[:, 1:]/self.semiaxes).square().sum(1)
        scale = self.settings['normal_distance_scale_m']
        active = (ball_in_play & (~had_contact) & (~ground_seen) & (~body_contact) & (~faulted)
            & finite_position & torch.isfinite(plane_distance_rate_mps)
            & (clock_s >= 2.2) & (clock_s <= 3.3) & (gap >= 0.) & (gap <= 4*scale)
            & (plane_distance_rate_mps < -.25) & (~terminated))
        potential = self.settings['contact_potential_weight']*torch.exp(-(gap/scale).square()-rho_squared)
        potential = torch.where(active, potential, torch.zeros_like(potential))
        shaping = self.settings['discount']*potential-self.previous_potential
        self.previous_potential.copy_(potential.detach())
        first_release = released & (~self.seen_release)
        self.seen_release |= released
        finite_velocity = torch.isfinite(outgoing_velocity_mps).all(1)
        velocity = torch.where(finite_velocity[:, None], outgoing_velocity_mps, torch.zeros_like(outgoing_velocity_mps))
        speed_progress = (velocity.norm(dim=1)/8.).clamp(0., 1.)
        downward_progress = (-velocity[:, 2]).clamp(0., 1.)
        release_eligible = first_release & legal_air & (~faulted) & finite_velocity & (velocity[:, 0] > 0.)
        progress = torch.where(release_eligible,
            self.settings['attack_progress_weight']*speed_progress*downward_progress, torch.zeros_like(speed_progress))
        first_frame = frame_hit & (~self.seen_frame)
        self.seen_frame |= frame_hit
        frame = -self.settings['frame_penalty']*first_frame.float()
        self.last_components.copy_(torch.stack([shaping, progress, frame], 1))
        self.job_sums.add_(self.last_components.double().sum(0))
        self.job_event_counts.add_(torch.stack([release_eligible.sum(), first_frame.sum()]))
        return self.last_components.sum(1)

    def statistics(self):
        names = ['measured_strings_potential_shaping', 'measured_release_attack_progress', 'actual_frame_penalty']
        return dict(scope='Actual paid TRAIN additions in this native job only; no reconstructed terminal-flag rewards or acceptance labels.',
            component_sums={name:float(value) for name,value in zip(names, self.job_sums.cpu())},
            eligible_first_releases=int(self.job_event_counts[0]), actual_first_frame_events=int(self.job_event_counts[1]))
