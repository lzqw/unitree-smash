"""Explicit ground64 provenance and read-only first-contact monitoring."""
import json
from pathlib import Path

import torch

from scripts.run_landing_validation_sequence_v1 import sha
from scripts.teacher_ground_extent_diagnostic_v1 import compare_teacher


GROUND = dict(original_configured_m=[4096.,4096.,.1], requested_m=[64.,64.,.1])


def verify_ground_source(source, original, checkpoint_path, protocol):
    result = compare_teacher(Path(source),Path(original),protocol)
    inputs = json.loads((Path(source)/'INPUTS.json').read_text())
    if inputs['policy_sha256'] != sha(checkpoint_path) or inputs['ground_extent_diagnostic'] != GROUND:
        raise ValueError('Ground64 source differs from the actual fixed training teacher')
    if result['candidate_contact_events']['suspicious_elevated_downward_patches']:
        raise ValueError('Observed elevated downward ground contact requires diagnosis before training')
    return dict(source=str(source), original_comparison=str(original),
        source_policy_sha256=sha(checkpoint_path), ground_extent_diagnostic=GROUND,
        measured_teacher=result['candidate64'],contact_events=result['candidate_contact_events'],
        original_ground_score_not_inherited=True,formal_acceptance=False)


class FirstGroundForceMonitor:
    """Monitor the first nonfoot ground contact of each actual episode on device.

    A negative vertical force on that event triggers diagnosis at the next
    completed PPO update. Later forces after an already invalid body contact
    are counted separately. Neither rewards, actions nor scoring are modified.
    This screening does not replace raw contact/geometry diagnosis.
    """
    def __init__(self, env):
        if not env.ground_sensors or any(
            sensor.data.force_matrix_w.ndim != 4
            or sensor.data.force_matrix_w.shape[2:] != (1, 3)
            or sensor.data.force_matrix_w.shape[0] != len(env.episode_index)
            for sensor in env.ground_sensors
        ):
            raise ValueError('Require the original single global-ground filter per nonfoot sensor')
        self.episode_index = env.episode_index.clone()
        self.seen = torch.zeros_like(self.episode_index,dtype=torch.bool)
        self.first_count = torch.zeros((),dtype=torch.int64,device=env.device)
        self.anomaly_count = self.first_count.clone()
        self.later_negative_count = self.first_count.clone()
        self.minimum_vertical = torch.zeros((),dtype=torch.float32,device=env.device)
        self.first_anomaly_frame = torch.full_like(self.episode_index,-1)
        self.callbacks = 0

    def on_physics(self, env, _roll_pitch):
        current = env.episode_index
        self.seen &= current == self.episode_index
        self.episode_index.copy_(current)
        forces = [sensor.data.force_matrix_w for sensor in env.ground_sensors]
        total = sum(force.norm(dim=-1).sum((1,2)) for force in forces)
        vertical = torch.cat([force[:,:,0,2] for force in forces],dim=1).amin(dim=1)
        first = (total > 1.) & (~self.seen)
        downward = vertical < -1.
        anomalous = first & downward
        self.first_count += first.sum()
        self.anomaly_count += anomalous.sum()
        self.later_negative_count += (self.seen & downward).sum()
        self.minimum_vertical.copy_(torch.minimum(self.minimum_vertical,vertical.amin()))
        self.first_anomaly_frame.copy_(torch.where(anomalous & (self.first_anomaly_frame < 0),
            env._jump_physics_frame,self.first_anomaly_frame))
        self.seen |= total > 1.
        self.callbacks += 1

    def on_reward(self, _env, _components, _reward):
        pass

    def on_episodes(self, _env, _complete_ids):
        # The next physics callback observes actual episode_index changes.
        pass

    def diagnostics(self):
        where = (self.first_anomaly_frame >= 0).nonzero().flatten()
        return dict(scope='First nonfoot ground-contact sign screening, not full contact correctness certification',
            metric_callbacks=self.callbacks,first_contact_events=int(self.first_count.item()),
            first_contact_downward_events=int(self.anomaly_count.item()),
            later_negative_env_time_samples=int(self.later_negative_count.item()),
            minimum_vertical_force_n=float(self.minimum_vertical.item()),
            first_anomaly_env_ids=where.cpu().tolist(),
            first_anomaly_physics_frames=self.first_anomaly_frame[where].cpu().tolist(),
            reward_action_scoring_unchanged=True)
