"""Development-only batching of separately frozen controllers on public inputs."""
import torch
from .learned_hybrid import LearnedHybridPolicy


class CandidateBatch:
    def __init__(self, checkpoints, candidate_ids, manifest, default, limits, physics, calibration):
        device, count = candidate_ids.device, len(candidate_ids)
        self.indices = [torch.where(candidate_ids==i)[0] for i in range(len(checkpoints))]
        self.policies = [LearnedHybridPolicy(cp, manifest, len(ids), device, default, limits, physics, calibration)
                         for cp, ids in zip(checkpoints, self.indices)]
        self.waypoints = torch.zeros((count, 3), device=device)
        self.arrival = torch.zeros(count, device=device)

    @staticmethod
    def subset(observation, ids):
        return {'policy': observation['policy'][ids]}

    def initialize(self, observation):
        for policy, ids in zip(self.policies, self.indices):
            policy.initialize(self.subset(observation, ids))
            self.waypoints.index_copy_(0, ids, policy.waypoints)
            self.arrival.index_copy_(0, ids, policy.arrival)

    def command_positions(self, observation):
        result = torch.empty_like(self.waypoints)
        for policy, ids in zip(self.policies, self.indices):
            result.index_copy_(0, ids, policy.command_positions(self.subset(observation, ids)))
        return result

    def action(self, observation):
        result = torch.empty((len(self.arrival), 18), device=self.arrival.device)
        for policy, ids in zip(self.policies, self.indices):
            result.index_copy_(0, ids, policy.action(self.subset(observation, ids)))
            self.arrival.index_copy_(0, ids, policy.arrival)
        return result

    @property
    def plan_log(self):
        return [dict(candidate=i, env_ids=ids.cpu().tolist(), plan=policy.plan_log)
                for i, (policy, ids) in enumerate(zip(self.policies, self.indices))]

    @property
    def replan_at_s(self):
        return [policy.replan_at_s for policy in self.policies]

    @property
    def replan_log(self):
        result = []
        for i, (policy, ids) in enumerate(zip(self.policies, self.indices)):
            mapping = ids.cpu().tolist()
            result.extend(dict(row, candidate=i, env_ids=[mapping[j] for j in row['env_ids']])
                          for row in policy.replan_log)
        return result

    @property
    def selection_at_s(self):
        return [policy.selection_at_s for policy in self.policies]

    @property
    def selection_log(self):
        result = []
        for i, (policy, ids) in enumerate(zip(self.policies, self.indices)):
            mapping = ids.cpu().tolist()
            result.extend(dict(row, candidate=i, env_ids=[mapping[j] for j in row['env_ids']])
                          for row in policy.selection_log)
        return result
