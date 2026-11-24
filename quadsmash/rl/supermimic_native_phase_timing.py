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

        sections = {}
        originals = []

        def watch(owner, attr, name):
            original = getattr(owner, attr, None)
            if original is None or not callable(original):
                return
            row = sections[name] = dict(seconds=0., calls=0)

            def timed(*args, **kwargs):
                started = perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    row['seconds'] += perf_counter() - started
                    row['calls'] += 1

            originals.append((owner, attr, original))
            setattr(owner, attr, timed)

        started = perf_counter()
        try:
            watch(self.env.sim, 'step', 'simulation')
            scene = getattr(self.env, 'scene', None)
            watch(scene, 'write_data_to_sim', 'scene_write')
            watch(scene, 'update', 'scene_update')
            for method, name in (('_apply_action', 'substep_action'), ('_consume_physics_frame', 'ball_events_nested'),
                                 ('_pre_physics_step', 'control_input'), ('_get_dones', 'terminal_checks'),
                                 ('_get_rewards', 'reward'), ('_get_observations', 'observation')):
                watch(self.env, method, name)
            result = advance(action)
        finally:
            for owner, attr, original in reversed(originals):
                setattr(owner, attr, original)
        elapsed = perf_counter() - started
        sim_timing = sections['simulation']
        self.samples[phase].append(dict(tick=tick, near_contact_lanes=near_count,
            released_flight_lanes=flight_count, physics_substeps=sim_timing['calls'], control_wall_seconds=elapsed,
            simulation_step_wall_seconds=sim_timing['seconds'], other_control_wall_seconds=elapsed - sim_timing['seconds'],
            instrumented_sections=sections))
        return result

    def report(self):
        return dict(scope='sampled whole-batch wall time; runtime synchronization may be charged to caller. '
                          'Ball event timing is nested inside substep action and terminal checks.',
                    samples=self.samples)
