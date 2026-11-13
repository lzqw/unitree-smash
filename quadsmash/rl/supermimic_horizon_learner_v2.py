"""Time-aware BPTT/GAE learner for the existing task adapter.

This fixes objectives and window semantics, but intentionally does NOT claim
that the legacy observation/reward predictor equals the official G1 dynamics
representation. Physical-state/event model validation remains a separate gate.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import torch
from torch import nn
from .supermimic_timebase_v2 import Timebase, seconds_to_steps, bptt_return, gae_returns
from .supermimic_sequence_replay_v2 import SequenceReplay, NotEnoughSequenceData


@contextmanager
def frozen(*modules):
    flags = [[p.requires_grad for p in m.parameters()] for m in modules]
    modes = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
            m.requires_grad_(False)
        yield
    finally:
        for m, fs, mode in zip(modules, flags, modes):
            for p, flag in zip(m.parameters(), fs):
                p.requires_grad_(flag)
            m.train(mode)


def optimizer_step(optimizer, params, *, element_clip=None):
    params = list(params)
    norm = nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    if element_clip is not None:
        nn.utils.clip_grad_value_(params, element_clip)
    optimizer.step()
    return norm.detach()


class HorizonTrainer:
    version = "task-timebase-v2"

    def __init__(self, actor, world, critic, replay, timebase: Timebase, *,
                 lr=2e-4, actor_lr=None, world_lr=None, critic_lr=None,
                 reference_actor=None, policy_anchor_coeff=0.0,
                 actor_microbatch=128, critic_epochs=5,
                 critic_minibatches=4, max_step_reward=100., actor_kind="unknown",
                 observation_clock_index=None):
        self.actor, self.world, self.critic, self.replay = actor, world, critic, replay
        self.timebase, self.actor_kind = timebase, actor_kind
        self.reference_actor = reference_actor
        if self.reference_actor is not None:
            self.reference_actor.eval()
            self.reference_actor.requires_grad_(False)
        self.policy_anchor_coeff = float(policy_anchor_coeff)
        if self.policy_anchor_coeff < 0:
            raise ValueError("policy_anchor_coeff must be non-negative")
        self.device = next(actor.parameters()).device
        if actor_microbatch < 1 or critic_epochs < 1 or critic_minibatches < 1 or max_step_reward <= 0:
            raise ValueError("Invalid optimizer configuration")
        self.actor_microbatch = actor_microbatch
        self.critic_epochs, self.critic_minibatches = critic_epochs, critic_minibatches
        self.v_max = max_step_reward / (1.0 - timebase.gamma)
        self.observation_clock_index = observation_clock_index
        self.actor_opt = torch.optim.Adam(actor.parameters(), lr=lr if actor_lr is None else actor_lr, betas=(.9, .95))
        self.world_opt = torch.optim.Adam(world.parameters(), lr=lr if world_lr is None else world_lr)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=lr if critic_lr is None else critic_lr)
        self.world_updates = self.actor_updates = self.critic_updates = 0

    @torch.no_grad()
    def action(self, obs, sample=True):
        return self.actor(obs, sample=sample)[0].clamp(-1., 1.)

    def fit_world(self, updates, batch_size):
        self.world.train()
        losses = []
        for _ in range(updates):
            try:
                window = self.replay.sample_windows(batch_size, self.timebase.world_steps)
            except NotEnoughSequenceData as error:
                return {"world_updates": len(losses), "world_waiting": str(error),
                        "world_window_steps": self.timebase.world_steps}
            prediction = window[0]["obs"]
            loss = prediction.new_zeros(())
            for transition in window:
                delta, reward, done_logit = self.world(prediction, transition["action"])
                prediction = prediction + delta
                # The target is data. Do not detach the predicted input inside
                # an artificial delta target: that changes multi-step gradients.
                loss = loss + (prediction - transition["next_obs"]).square().mean()
                loss = loss + .5 * (reward - transition["reward"]).square().mean()
                loss = loss + .1 * nn.functional.binary_cross_entropy_with_logits(done_logit, transition["done"])
            loss = loss / len(window)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite world-model loss")
            self.world_opt.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_step(self.world_opt, self.world.parameters())
            self.world_updates += 1
            losses.append(loss.detach())
        return {"world_updates": len(losses), "world_loss": float(torch.stack(losses).mean()) if losses else None,
                "world_window_steps": self.timebase.world_steps}

    def _rollout(self, initial, elapsed):
        cfg = self.timebase
        obs = initial
        states, rewards, dones, valid, anchors = [], [], [], [], []
        for k in range(cfg.imagination_steps):
            active = elapsed + k * cfg.control_dt_s < cfg.episode_s - 1e-7
            action = self.actor(obs, sample=False)[0].clamp(-1., 1.)
            if self.reference_actor is not None and self.policy_anchor_coeff > 0:
                with torch.no_grad():
                    reference_action = self.reference_actor(obs, sample=False)[0].clamp(-1., 1.)
                anchor_penalty = (action - reference_action).square().mean(-1, keepdim=True)
            else:
                anchor_penalty = torch.zeros_like(action[..., :1])
            anchors.append(anchor_penalty)
            delta, reward, terminal = self.world(obs, action)
            next_obs = obs + delta
            # The real observation clock is deterministic; the model cannot
            # gain reward by slowing down episode time.
            if self.observation_clock_index is not None:
                next_obs = next_obs.clone()
                next_obs[:, self.observation_clock_index:self.observation_clock_index+1] = elapsed + (k+1)*cfg.control_dt_s
            at_end = elapsed + (k + 1) * cfg.control_dt_s >= cfg.episode_s - 1e-7
            done = torch.where(at_end, torch.ones_like(terminal), terminal.sigmoid())
            states.append(obs.detach())
            rewards.append(torch.where(active, reward, torch.zeros_like(reward)))
            dones.append(done)
            valid.append(active)
            obs = torch.where(active, next_obs, obs)
        values = self.critic(obs).clamp(max=self.v_max)
        ret = bptt_return(torch.stack(rewards), torch.stack(dones), values, cfg.gamma)
        if not torch.isfinite(ret).all():
            raise FloatingPointError("Nonfinite imagination return; no silent state clipping")
        # All critic data are detached before the actor optimizer changes theta.
        return ret, (torch.stack(states), torch.stack(rewards).detach(),
                     torch.stack(dones).detach(), obs.detach(), torch.stack(valid)), torch.stack(anchors).sum(0)

    def imagine_update(self, batch_size):
        if self.world_updates == 0:
            return {"imagine_update": False, "reason": "no completed world update"}
        batch = self.replay.sample_starts(batch_size)
        self.actor.train()
        self.actor_opt.zero_grad(set_to_none=True)
        ret_sum = 0.
        anchor_sum = 0.
        critic_batches = []
        with frozen(self.world, self.critic):
            for begin in range(0, batch_size, self.actor_microbatch):
                end = min(begin + self.actor_microbatch, batch_size)
                ret, trajectory, anchor_penalty = self._rollout(
                    batch["obs"][begin:end].detach(), batch["elapsed"][begin:end])
                # One optimizer step for the full logical batch. Never detach
                # between time steps; microbatching partitions B, not horizon H.
                ((-ret + self.policy_anchor_coeff * anchor_penalty).sum() / batch_size).backward()
                ret_sum += float(ret.detach().sum())
                anchor_sum += float(anchor_penalty.detach().sum())
                critic_batches.append(trajectory)
            actor_norm = optimizer_step(self.actor_opt, self.actor.parameters(), element_clip=.1)
        self.actor_updates += 1
        states = torch.cat([b[0] for b in critic_batches], dim=1)
        rewards = torch.cat([b[1] for b in critic_batches], dim=1)
        dones = torch.cat([b[2] for b in critic_batches], dim=1)
        last_obs = torch.cat([b[3] for b in critic_batches])
        valid = torch.cat([b[4] for b in critic_batches], dim=1).squeeze(-1)
        with torch.no_grad():
            values = self.critic(states)
            last_value = self.critic(last_obs)
            targets = gae_returns(values, last_value, rewards, dones, self.timebase.gamma, self.timebase.lam).clamp(max=self.v_max)
        flat_states, flat_targets = states[valid], targets[valid]
        if len(flat_targets) == 0:
            raise RuntimeError("All sampled starts were outside the episode")
        losses = []
        for _ in range(self.critic_epochs):
            perm = torch.randperm(len(flat_targets), device=self.device)
            for ids in torch.tensor_split(perm, self.critic_minibatches):
                if not len(ids):
                    continue
                self.critic_opt.zero_grad(set_to_none=True)
                # Accumulation also bounds critic activation memory on 16 GB.
                loss_sum = 0.
                for sub in ids.split(self.actor_microbatch):
                    loss = (self.critic(flat_states[sub]) - flat_targets[sub]).square().sum() / len(ids)
                    loss.backward()
                    loss_sum += float(loss.detach())
                optimizer_step(self.critic_opt, self.critic.parameters())
                self.critic_updates += 1
                losses.append(loss_sum)
        return {"imagine_update": True, "imagined_return": ret_sum / batch_size,
                "actor_loss": -ret_sum / batch_size, "actor_grad_norm": float(actor_norm),
                "policy_anchor_penalty": anchor_sum / batch_size,
                "critic_loss": sum(losses) / len(losses), "actor_updates_total": self.actor_updates,
                "critic_updates_total": self.critic_updates, "world_updates_total": self.world_updates,
                "imagination_steps": self.timebase.imagination_steps, "imagination_s": self.timebase.imagination_s}

    @torch.no_grad()
    def validate_open_loop(self, batch_size, seconds):
        steps = seconds_to_steps(seconds, self.timebase.control_dt_s, "validation horizon")
        window = self.replay.sample_windows(batch_size, steps)
        prediction = window[0]["obs"]
        mse = []
        for transition in window:
            delta, _, _ = self.world(prediction, transition["action"])
            prediction = prediction + delta
            mse.append(float((prediction - transition["next_obs"]).square().mean()))
        return {"requested_seconds": seconds, "steps": steps, "actual_seconds": steps*self.timebase.control_dt_s,
                "observation_mse_by_step": mse,
                "note": "Diagnostic only; aggregate observation MSE is not a physical safety certificate."}

    def save(self, path, metadata, persist_replay=False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"algorithm": "supermimic_v1", "learner_version": self.version,
                    "random_initialization": False, "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(), "world_model": self.world.state_dict(),
                    "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
                    "world_opt": self.world_opt.state_dict(), "timebase": self.timebase.manifest(),
                    "counters": {"world": self.world_updates, "actor": self.actor_updates, "critic": self.critic_updates},
                    "replay": self.replay.state_dict() if persist_replay else None,
                    "metadata": {**metadata, "actor_kind": self.actor_kind, "old_checkpoint": None,
                                 "action_contract_version": "full-racket-final-target-v2"}}, path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("learner_version") != self.version or state.get("timebase") != self.timebase.manifest():
            raise ValueError("Resume requires the same corrected action/timebase contract; old replay must not be reused")
        if state["metadata"].get("actor_kind") != self.actor_kind:
            raise ValueError("Actor kind mismatch")
        for name, key in (("actor", "actor"), ("world", "world_model"), ("critic", "critic")):
            getattr(self, name).load_state_dict(state[key], strict=True)
        for name in ("actor_opt", "world_opt", "critic_opt"):
            opt = getattr(self, name)
            opt.load_state_dict(state[name])
            for slot in opt.state.values():
                for key, value in slot.items():
                    if torch.is_tensor(value):
                        slot[key] = value.to(self.device)
        self.world_updates, self.actor_updates, self.critic_updates = (state["counters"][k] for k in ("world", "actor", "critic"))
        if state.get("replay"):
            self.replay.load_state_dict(state["replay"])
        return state["metadata"]
