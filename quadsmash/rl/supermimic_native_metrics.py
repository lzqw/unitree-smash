"""Measured native episode returns and dynamic task denominators.

No model reward, predicted events, reference-case membership, or scoring-rule
changes. Partial episodes never enter completed-episode statistics.
"""
import math
import torch

RETURN_DEFINITION = 'undiscounted_sum_of_native_control_step_rewards_v1'


class NativeReturnAccumulator:
    def __init__(self, count, device, gamma=0.99**0.5):
        if not 0 < gamma <= 1:
            raise ValueError('Invalid native return discount')
        self.gamma = float(gamma)
        self.returns = torch.zeros(count, device=device, dtype=torch.float64)
        self.discounted = torch.zeros_like(self.returns)
        self.discount = torch.ones_like(self.returns)
        self.steps = torch.zeros(count, device=device, dtype=torch.long)

    @torch.no_grad()
    def add(self, reward):
        value = reward.detach().reshape(-1)
        if value.shape != self.returns.shape:
            raise ValueError('Native reward lane shape mismatch')
        # Rewards already include their own dt/event scaling. Do not multiply
        # by dt or gamma again. Float64 accumulation reduces long-sum drift.
        self.returns.add_(value.to(dtype=self.returns.dtype))
        self.discounted.add_(self.discount * value.to(dtype=self.returns.dtype))
        self.discount.mul_(self.gamma)
        self.steps.add_(1)

    @torch.no_grad()
    def records(self, ids):
        ids = torch.as_tensor(ids, device=self.returns.device, dtype=torch.long).reshape(-1)
        totals = self.returns[ids].cpu().tolist()
        steps = self.steps[ids].cpu().tolist()
        discounted = self.discounted[ids].cpu().tolist()
        return {int(i):dict(native_episode_return=float(value),
                           native_episode_discounted_return=float(dvalue),
                           native_return_gamma=self.gamma,
                           native_episode_length_steps=int(n),
                           native_return_definition=RETURN_DEFINITION)
                for i,value,n,dvalue in zip(ids.cpu().tolist(),totals,steps,discounted)}

    @torch.no_grad()
    def reset(self, ids):
        self.returns[ids] = 0.
        self.discounted[ids] = 0.
        self.discount[ids] = 1.
        self.steps[ids] = 0


def count_first_episode_body_contacts(seen_lanes, episode, body_edge):
    """Accumulate first-episode body contacts per lane across native collection blocks."""
    seen_lanes.logical_or_(((episode == 0) & body_edge).any(dim=0))
    return int(seen_lanes.sum())


class NativeEpisodeStatistics:
    """O(1) streaming summary; each input is one independently scored terminal."""
    def __init__(self):
        self.episodes = self.contacts = self.recoveries = self.ball_ok = self.complete = 0
        self.body_observed = self.body_contacts = 0
        self.return_count = 0
        self.return_mean = self.return_m2 = 0.
        self.return_min = self.return_max = self.last_return = None

    def add(self, row):
        required=('qualified_airborne_string_contact','recovery_success',
                  'ball_side_complete_except_recovery','complete5')
        if any(k not in row or row[k] is None for k in required):
            raise ValueError('Missing measured terminal task evidence')
        value=row.get('native_episode_return')
        if value is not None:
            if (not math.isfinite(float(value)) or row.get('native_episode_length_steps',0)<=0
                    or row.get('native_return_definition') != RETURN_DEFINITION):
                raise ValueError('Invalid native terminal return evidence')
        contact=bool(row['qualified_airborne_string_contact'])
        self.episodes += 1
        if row.get('body_ground_seen') is not None:
            self.body_observed += 1
            self.body_contacts += int(bool(row['body_ground_seen']))
        self.contacts += int(contact)
        self.recoveries += int(contact and bool(row['recovery_success']))
        self.ball_ok += int(bool(row['ball_side_complete_except_recovery']))
        self.complete += int(bool(row['complete5']))
        if value is not None:
            value=float(value);self.return_count += 1
            delta=value-self.return_mean;self.return_mean += delta/self.return_count
            self.return_m2 += delta*(value-self.return_mean)
            self.return_min=value if self.return_min is None else min(self.return_min,value)
            self.return_max=value if self.return_max is None else max(self.return_max,value)
            self.last_return=value

    def snapshot(self):
        n=self.episodes;c=self.contacts
        return dict(completed_episodes=n,
            qualified_airborne_string_contact=self.contacts,
            qualified_airborne_string_contact_rate=self.contacts/n if n else None,
            ball_side_complete_except_recovery=self.ball_ok,
            ball_side_complete_except_recovery_rate=self.ball_ok/n if n else None,
            conditional_recovery_count=self.recoveries,
            conditional_recovery_denominator=c,
            recovery_given_airborne_string_contact=self.recoveries/c if c else None,
            conditional_recovery_scope='actual_qualified_contacts_in_these_completed_episodes',
            complete5=self.complete,complete5_rate=self.complete/n if n else None,
            native_episode_return_observed_count=self.return_count,
            native_episode_return_missing_count=n-self.return_count,
            native_episode_return_mean=self.return_mean if self.return_count else None,
            native_episode_return_std=math.sqrt(max(0.,self.return_m2/self.return_count)) if self.return_count else None,
            native_episode_return_min=self.return_min,native_episode_return_max=self.return_max,
            native_episode_return_last=self.last_return,native_return_definition=RETURN_DEFINITION)


def summarize_native_episodes(rows):
    stats=NativeEpisodeStatistics()
    for row in rows:stats.add(row)
    return stats.snapshot()
