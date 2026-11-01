"""Read first nonfoot ground-contact events without changing native commands."""
import torch

from scripts.standing_contact_telemetry_v1 import StandingContactTelemetry


class StrikeGroundContactObserver:
    """Sample the original filtered-force condition at every metric callback.

    Detailed native buffers are copied only when a first-episode lane newly
    crosses 1 N. This is event telemetry, not a continuous force archive.
    """
    def __init__(self, env, output):
        self.telemetry = StandingContactTelemetry(env, output)
        self.seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.callbacks = 0
        self.last_frame = None
        self.telemetry.metadata['sampling_scope'] = 'First nonfoot contact events only; no continuous force archive'
        self.telemetry._atomic('contact_telemetry_metadata.json', self.telemetry.metadata)
        self.telemetry._atomic('first_nonfoot_contact_details.json', [])

    def on_physics(self, env, _roll_pitch):
        self.callbacks += 1
        self.last_frame = env._jump_physics_frame
        force = sum(s.data.force_matrix_w.norm(dim=-1).sum((1, 2)) for s in env.ground_sensors)
        new = (env.episode_index == 0) & (~self.seen) & (force > 1.)
        if not bool(new.any()):
            return
        self.telemetry.on_physics(env)
        self.seen.copy_(torch.as_tensor(self.telemetry.seen, device=env.device))

    def on_reward(self, _env, _components, _reward):
        # The environment invokes this diagnostic hook for every control step.
        # Contact telemetry must not alter reward components or policy state.
        pass

    def on_episodes(self, _env, _episodes):
        # First-event selection uses actual episode_index in on_physics.
        # Episode scoring and reset handling remain owned by the environment.
        pass

    def save(self):
        self.telemetry._atomic('first_nonfoot_contact_details.json', self.telemetry.events)
        self.telemetry._atomic('contact_telemetry_summary.json', dict(
            sampling_scope='First nonfoot contact events only; no continuous force archive',
            metric_callbacks=self.callbacks, last_physics_frame=self.last_frame,
            recorded_first_event_lanes=self.seen.nonzero().flatten().cpu().tolist(),
            event_records=len(self.telemetry.events), original_metric_changed=False))
