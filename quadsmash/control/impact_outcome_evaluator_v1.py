"""Causal analytic return prediction, never a PhysX response or task scorer.

All inputs are in one environment-local WORLD frame (court coordinates).
The input racket point is on the physical string surface, not ball center.
"""
from copy import deepcopy
import math
import numpy as np
from quadsmash.physics.collision import RacketGeometry, resolve_racket_collision_with_reaction
from quadsmash.physics.full_racket_v4 import string_strips
from quadsmash.physics.court_v4 import top_height
from .initial_intercept import propagate_ball_state


class ImpactOutcomeEvaluator:
    def __init__(self, physics, court, racket, *, dt_s=.01, max_flight_s=4.,
                 contact_tolerance_m=.003):
        self.physics, self.court, self.racket = map(deepcopy, (physics, court, racket))
        if not (0 < dt_s <= .02 and 0 < max_flight_s <= 8 and
                0 <= contact_tolerance_m <= .01):
            raise ValueError('Invalid bounded prediction integration settings')
        self.dt_s, self.max_flight_s = float(dt_s), float(max_flight_s)
        self.tolerance = float(contact_tolerance_m)
        self.strips = string_strips(self.racket)

    def evaluate(self, shuttle_position, shuttle_velocity_before,
                 racket_contact_position, racket_contact_velocity, racket_normal,
                 contact_point_local, robot_state=None):
        vectors = [np.asarray(v, dtype=float) for v in (shuttle_position,
            shuttle_velocity_before, racket_contact_position, racket_contact_velocity,
            racket_normal, contact_point_local)]
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
            raise ValueError('Finite 3D impact vectors required')
        p, incoming, surface, vr, normal, local = vectors
        norm = np.linalg.norm(normal)
        if norm < 1e-12: raise ValueError('Zero contact normal')
        normal = normal/norm
        radius = self.physics['collision_radius_m']
        result = dict(model='analytic_finite_mass_coulomb_impulse_plus_drag',
            native_execution=False, valid_string_contact=False, valid_return=False,
            predicted_target_landing=False, rejection_reason=None,
            incoming_normal_speed=float((incoming-vr)@normal),
            outgoing_velocity=None, predicted_impulse=None, reaction_impulse=None,
            net_clearance=None, landing_position=None)
        def reject(reason):
            result['rejection_reason'] = reason
            return result
        # Actual authored strip boxes, not an invented circular racket radius.
        on_string = any(np.all(np.abs(local-np.asarray(s['center'])) <=
                              .5*np.asarray(s['size'])+1e-9) for s in self.strips)
        if not on_string: return reject('outside_authored_string_surface')
        if np.linalg.norm(p-(surface+radius*normal)) > self.tolerance:
            return reject('ball_string_contact_position_mismatch')
        result['valid_string_contact'] = True
        if result['incoming_normal_speed'] >= 0: return reject('nonapproaching_contact')
        if surface[0] >= self.court['net_plane_x_m']: return reject('contact_on_opponent_side')
        effective_mass = math.inf
        if robot_state is not None:
            # Normal effective mass from full floating-base inertia and point J.
            mass = np.asarray(robot_state['mass_matrix'], dtype=float)
            jac = np.asarray(robot_state['contact_jacobian_world'], dtype=float)
            if (mass.ndim != 2 or mass.shape[0] != mass.shape[1] or
                    jac.shape != (3,len(mass)) or not np.isfinite(mass).all() or
                    not np.isfinite(jac).all() or not np.allclose(mass,mass.T)):
                raise ValueError('Invalid full-body effective-mass input')
            np.linalg.cholesky(mass)
            jn = normal@jac
            inverse_mass = float(jn@np.linalg.solve(mass,jn))
            effective_mass = 1./inverse_mass if inverse_mass > 1e-14 else math.inf
        outgoing, _, event = resolve_racket_collision_with_reaction(incoming,
            RacketGeometry(surface,normal), racket_velocity=vr, racket_mass=effective_mass,
            shuttle_mass=self.physics['mass_kg'], restitution=self.physics['restitution'],
            friction=self.physics['friction'], contact_position=surface)
        result.update(outgoing_velocity=outgoing.tolist(),
            outgoing_speed_m_s=float(np.linalg.norm(outgoing)),
            predicted_impulse=(-event.reaction_impulse).tolist(),
            reaction_impulse=event.reaction_impulse.tolist(),
            effective_racket_mass_kg=None if math.isinf(effective_mass) else effective_mass,
            finite_mass_approximation='normal effective scalar mass; isotropic tangential approximation')
        if outgoing[0] <= 0: return reject('outgoing_not_toward_opponent')
        position, velocity = p.copy(), outgoing.copy()
        crossing = None
        for _ in range(math.ceil(self.max_flight_s/self.dt_s)):
            following, next_velocity = propagate_ball_state(position,velocity,self.dt_s,
                                                            self.physics,dt=self.dt_s)
            plane = self.court['net_plane_x_m']+radius  # trailing-surface crossing
            if crossing is None and position[0] <= plane <= following[0]:
                fraction = (plane-position[0])/(following[0]-position[0])
                crossing = position+fraction*(following-position)
                height = float(top_height(crossing[1],self.court))
                result['net_clearance'] = float(crossing[2]-radius-height)
                result['crossing_position'] = crossing.tolist()
            if following[2] <= radius < position[2]:
                fraction = (position[2]-radius)/(position[2]-following[2])
                landing = position+fraction*(following-position)
                result['landing_position'] = landing.tolist()
                break
            position, velocity = following, next_velocity
        if result['landing_position'] is None: return reject('landing_not_found_in_horizon')
        if crossing is None: return reject('no_net_crossing')
        if abs(crossing[1]) > self.court['net_width_m']/2: return reject('outside_net_width')
        # Conservative geometric coarse filter, NOT a replacement for tape-contact scoring.
        if result['net_clearance'] < 0: return reject('predicted_net_intersection')
        x,y,_ = result['landing_position']
        lo,hi = self.court['opponent_x_limits_m']
        if not (lo <= x <= hi and abs(y) <= self.court['singles_width_m']/2):
            return reject('predicted_out_of_court')
        lo,hi = self.court['target_x_limits_m']
        result['predicted_target_landing'] = bool(lo <= x <= hi and abs(y) <= self.court['target_y_half_width_m'])
        result['valid_return'] = True
        # Dimensionless task cost with physical scales: target half-widths and .1m clearance.
        center = .5*(lo+hi)
        result['ranking_cost'] = float(((x-center)/(.5*(hi-lo)))**2 +
            (y/self.court['target_y_half_width_m'])**2 + max(0.,.1-result['net_clearance'])/.1)
        return result
