"""Ordered, force-confirmed G07 event accounting, independent of Isaac imports.

Opt-in v3 diagnostics preserve legacy evidence and reject ground bounces,
multiple racket contacts, and net crossings outside the declared court.
"""
from __future__ import annotations

import math
import torch
from .native import segment_plane_crossing


class OrderedSmashEvents:
    def __init__(self, count, device, *, radius=.025, net_x=1.5,
                 net_height=1.524, net_half_width=1., target_x=(2., 4.),
                 target_y=1., dtype=torch.float32):
        self.radius, self.net_x = radius, net_x
        self.net_height, self.net_half_width = net_height, net_half_width
        self.target_x, self.target_y = target_x, target_y
        self.count = count
        for name in ("contact", "airborne_contact", "effective_face", "released",
                     "crossed", "ground_seen", "landed_target", "invalid_multiple_contact"):
            setattr(self, name, torch.zeros(count, dtype=torch.bool, device=device))
        self.quiet_frames = torch.zeros(count, dtype=torch.long, device=device)
        self.contact_count = torch.zeros_like(self.quiet_frames)
        for name in ("contact_time", "release_time", "crossing_time", "landing_time", "net_margin"):
            setattr(self, name, torch.full((count,), float("nan"), device=device, dtype=dtype))
        for name in ("contact_position", "outgoing_velocity", "net_position", "landing_position"):
            setattr(self, name, torch.full((count, 3), float("nan"), device=device, dtype=dtype))
        self._last_touch_velocity = torch.zeros((count, 3), device=device, dtype=dtype)

    def reset(self, ids):
        for name in ("contact", "airborne_contact", "effective_face", "released", "crossed",
                     "ground_seen", "landed_target", "invalid_multiple_contact"):
            getattr(self, name)[ids] = False
        self.quiet_frames[ids] = self.contact_count[ids] = 0
        for name in ("contact_time", "release_time", "crossing_time", "landing_time", "net_margin",
                     "contact_position", "outgoing_velocity", "net_position", "landing_position"):
            getattr(self, name)[ids] = float("nan")
        self._last_touch_velocity[ids] = 0

    def advance(self, *, previous_position, position, velocity, contact_event,
                touching, airborne, effective_face, ground_touch, time, dt):
        before_ground = ~self.ground_seen
        first = contact_event & (~self.contact)
        self.contact_count += contact_event.long()
        self.invalid_multiple_contact |= self.contact_count > 1
        self.contact_time[first] = time[first]
        self.contact_position[first] = position[first]
        self.airborne_contact[first] = airborne[first] & before_ground[first]
        self.effective_face[first] = effective_face[first]
        self.contact |= contact_event
        self._last_touch_velocity[touching] = velocity[touching]
        quiet = self.contact & (~touching)
        self.quiet_frames = torch.where(quiet, self.quiet_frames + 1,
                                       torch.zeros_like(self.quiet_frames))
        release = quiet & (self.quiet_frames >= 2) & (~self.released)
        self.outgoing_velocity[release] = self._last_touch_velocity[release]
        self.release_time[release] = time[release]
        self.released |= release
        attack = self.attack()
        crossing, fraction, point = segment_plane_crossing(
            previous_position, position, 0, self.net_x)
        first_crossing = crossing & self.released & before_ground & torch.isnan(self.crossing_time)
        self.crossing_time[first_crossing] = time[first_crossing] - dt + fraction[first_crossing] * dt
        self.net_position[first_crossing] = point[first_crossing]
        margin = point[:, 2] - self.radius - self.net_height
        self.net_margin[first_crossing] = margin[first_crossing]
        legal = first_crossing & attack & (margin > 0) & (point[:, 1].abs() + self.radius <= self.net_half_width)
        self.crossed |= legal
        landing, fraction, point = segment_plane_crossing(
            previous_position, position, 2, self.radius, -1)
        first_landing = (landing | ground_touch) & before_ground
        # With a force-only ground contact, the center XY is a conservative
        # proxy and is explicitly tagged by the caller's substep geometry.
        point = torch.where((landing | (~ground_touch))[:, None], point, position)
        self.landing_position[first_landing] = point[first_landing]
        self.landing_time[first_landing] = torch.where(
            landing, time - dt + fraction * dt, time)[first_landing]
        target = ((point[:, 0] >= self.target_x[0]) & (point[:, 0] <= self.target_x[1])
                  & (point[:, 1].abs() <= self.target_y))
        ordered = (self.release_time < self.crossing_time) & (self.crossing_time < self.landing_time)
        self.landed_target |= first_landing & self.crossed & target & attack & ordered
        self.ground_seen |= landing | ground_touch

    def attack(self):
        return (self.released & self.airborne_contact & self.effective_face
                & (~self.invalid_multiple_contact)
                & (self.outgoing_velocity.norm(dim=-1) >= 8.)
                & (self.outgoing_velocity[:, 0] > 0)
                & (self.outgoing_velocity[:, 2] <= -1.))

    def complete(self, recovery):
        return self.attack() & self.landed_target & recovery


def rank_candidate(record):
    """Actual ordered net/first-landing results dominate all local components."""
    numerical = (record.get('numerical_failures', 0) or record.get('sweep_misses', 0)
                 or record.get('sweep_unresolved', 0))
    legal_landing = bool(record.get("ordered_landed_target", False)) and not numerical
    legal_crossing = bool(record.get("ordered_crossed_net", False)) and not numerical
    complete = legal_landing and bool(record.get("recovery_success", False))
    margin = record.get("net_margin_m")
    margin = float(margin) if margin is not None else -2.
    margin = min(.5, max(-2., margin)) if math.isfinite(margin) else -2.
    velocity = record.get("outgoing_velocity_mps", (0., 0., 0.))
    velocity = tuple(float(x) if math.isfinite(float(x)) else 0. for x in velocity)
    forward = velocity[0] > 0
    return (int(complete), int(legal_landing), int(legal_crossing),
            int(forward), margin, -abs(velocity[1]))
