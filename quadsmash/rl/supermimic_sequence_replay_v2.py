"""GPU time-major ring replay with exact windows and reset/session boundaries."""
from __future__ import annotations
import torch


class NotEnoughSequenceData(RuntimeError):
    pass


class SequenceReplay:
    def __init__(self, capacity, num_envs, obs_dim, action_dim, device, event_dim=0):
        if capacity < 2 * num_envs:
            raise ValueError("Replay needs at least two time slots per environment")
        self.slots, self.num_envs = capacity // num_envs, num_envs
        self.device, self.capacity = torch.device(device), (capacity // num_envs) * num_envs
        self.obs_dim, self.action_dim, self.event_dim = obs_dim, action_dim, int(event_dim)
        if self.event_dim < 0:
            raise ValueError("event_dim must be non-negative")
        self.ptr = self.ticks_written = self.session = 0
        spec = {"obs": (obs_dim,), "next_obs": (obs_dim,), "action": (action_dim,),
                "reward": (1,), "done": (1,), "elapsed": (1,)}
        if self.event_dim:
            spec.update(event_state=(self.event_dim,), next_event_state=(self.event_dim,))
        self.data = {k: torch.empty((self.slots, num_envs, *shape), device=device) for k, shape in spec.items()}
        self.episode = torch.full((self.slots, num_envs), -1, dtype=torch.long, device=device)
        self.sessions = torch.full((self.slots,), -1, dtype=torch.long, device=device)
        self.stamps = torch.full((self.slots,), -1, dtype=torch.long, device=device)

    def __len__(self):
        return min(self.ticks_written, self.slots) * self.num_envs

    @torch.no_grad()
    def add(self, obs, action, next_obs, reward, done, elapsed, episode,
            event_state=None, next_event_state=None):
        arrays = dict(obs=obs, action=action, next_obs=next_obs, reward=reward.reshape(-1, 1),
                      done=done.reshape(-1, 1), elapsed=elapsed.reshape(-1, 1))
        if self.event_dim:
            if event_state is None or next_event_state is None:
                raise ValueError("event_state and next_event_state are required")
            arrays.update(event_state=event_state, next_event_state=next_event_state)
        for key, value in arrays.items():
            if value.shape != self.data[key][self.ptr].shape:
                raise ValueError(f"Replay shape mismatch: {key}")
            self.data[key][self.ptr].copy_(value.detach())
        self.episode[self.ptr].copy_(episode.reshape(-1).detach())
        self.sessions[self.ptr] = self.session
        self.stamps[self.ptr] = self.ticks_written
        self.ptr = (self.ptr + 1) % self.slots
        self.ticks_written += 1

    def new_session(self):
        # Restoring a checkpoint does not restore PhysX contact caches. Never
        # splice the first resumed simulator step onto a saved physical episode.
        self.session += 1

    def _rows(self, stamps):
        return stamps % self.slots

    def sample_starts(self, batch):
        count = min(self.ticks_written, self.slots)
        if count == 0:
            raise NotEnoughSequenceData("Replay is empty")
        low = self.ticks_written - count
        stamps = torch.randint(low, self.ticks_written, (batch,), device=self.device)
        lanes = torch.randint(self.num_envs, (batch,), device=self.device)
        return {key: value[self._rows(stamps), lanes] for key, value in self.data.items()}

    def sample_windows(self, batch, horizon):
        if horizon < 1 or horizon > self.slots:
            raise ValueError("Window must fit in replay time slots")
        low = max(0, self.ticks_written - self.slots)
        high = self.ticks_written - horizon + 1
        if high <= low:
            raise NotEnoughSequenceData(f"Need {horizon} consecutive transitions per lane")
        # Rejection sampling is batched on-device. Never silently substitute a
        # one-step batch or a shorter window for the requested physical horizon.
        accepted = []
        remaining = batch
        offsets = torch.arange(horizon, device=self.device)
        for _ in range(8):
            count = max(64, remaining * 2)
            starts = torch.randint(low, high, (count,), device=self.device)
            lanes = torch.randint(self.num_envs, (count,), device=self.device)
            stamps = starts[:, None] + offsets[None]
            rows = self._rows(stamps)
            lane_grid = lanes[:, None].expand_as(rows)
            ep = self.episode[rows, lane_grid]
            valid = (self.stamps[rows] == stamps).all(-1)
            valid &= (self.sessions[rows] == self.sessions[rows[:, :1]]).all(-1)
            valid &= (ep == ep[:, :1]).all(-1)
            if horizon > 1:
                valid &= (self.data["done"][rows[:, :-1], lane_grid[:, :-1], 0] < .5).all(-1)
            selected = torch.where(valid)[0][:remaining]
            accepted.append(torch.stack((starts[selected], lanes[selected]), -1))
            remaining -= selected.numel()
            if remaining == 0:
                break
        if remaining:
            raise NotEnoughSequenceData("Not enough reset-free windows; collect more data")
        chosen = torch.cat(accepted)
        rows = self._rows(chosen[:, :1] + offsets[None])
        lanes = chosen[:, 1:2].expand_as(rows)
        # A list of actual transitions: K targets, hence K+1 state positions.
        return [{key: value[rows[:, k], lanes[:, k]] for key, value in self.data.items()}
                for k in range(horizon)]

    def state_dict(self):
        valid_slots = min(self.ticks_written, self.slots)
        return {"capacity": self.capacity, "num_envs": self.num_envs, "event_dim": self.event_dim,
                "ptr": self.ptr, "ticks_written": self.ticks_written, "session": self.session,
                "data": {k: v[:valid_slots].cpu() for k, v in self.data.items()},
                "episode": self.episode[:valid_slots].cpu(),
                "sessions": self.sessions[:valid_slots].cpu(), "stamps": self.stamps[:valid_slots].cpu()}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity or state["num_envs"] != self.num_envs:
            raise ValueError("Resume replay layout mismatch")
        if int(state.get("event_dim", 0)) != self.event_dim:
            raise ValueError("Resume replay event-state layout mismatch")
        count = min(state["ticks_written"], self.slots)
        for key in self.data:
            self.data[key][:count].copy_(state["data"][key].to(self.device))
        for key in ("episode", "sessions", "stamps"):
            getattr(self, key)[:count].copy_(state[key].to(self.device))
        self.ptr, self.ticks_written, self.session = state["ptr"], state["ticks_written"], state["session"]
        self.new_session()
