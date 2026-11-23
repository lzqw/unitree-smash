"""Sample native control-step wall time at three ball-trajectory phases."""

from time import perf_counter

import torch


class NativePhaseTiming:
    def __init__(self, env, *, samples_per_phase=4, stride=4):
        self.env = env
        self.samples_per_phase = samples_per_phase
        self.stride = stride
        self.samples = {'far': [], 'near_contact': [], 'released_flight': []}

    def step(self, advance, action, observation, tick):
        if tick % self.stride or all(len(group) >= self.samples_per_phase for group in self.samples.values()):
            return advance(action)
        rally = self.env.rally
        ball_present = observation[:, 67] > .5
        distance = (observation[:, 61:64] - observation[:, 45:48]).norm(dim=-1)
        near = ball_present & ~rally.had_contact & (distance < .45)
        flight = ball_present & rally.released & ~rally.ground_seen
        near_count, flight_count = torch.stack((near.sum(), flight.sum())).tolist()
        phase = 'near_contact' if near_count else 'released_flight' if flight_count else 'far'
        if len(self.samples[phase]) >= self.samples_per_phase:
            return advance(action)

        sim = self.env.sim
        original = sim.step
        physics_seconds = 0.
        substeps = 0

        def timed_step(*args, **kwargs):
            nonlocal physics_seconds, substeps
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                physics_seconds += perf_counter() - started
                substeps += 1

        started = perf_counter()
        sim.step = timed_step
        try:
            result = advance(action)
        finally:
            sim.step = original
        elapsed = perf_counter() - started
        self.samples[phase].append(dict(tick=tick, near_contact_lanes=near_count,
            released_flight_lanes=flight_count, physics_substeps=substeps, control_wall_seconds=elapsed,
            simulation_step_wall_seconds=physics_seconds, other_control_wall_seconds=elapsed - physics_seconds))
        return result

    def report(self):
        return dict(scope='sampled whole-batch wall time; simulation calls may include runtime synchronization',
                    samples=self.samples)
