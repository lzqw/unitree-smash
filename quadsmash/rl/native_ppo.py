"""Small compatibility adapter around the locked stock RSL-RL PPO."""
import copy
import math
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as runner_module


class FinalObservationPPO(PPO):
    """Bootstrap truncations from reset-before observations, then use stock PPO."""
    def process_env_step(self,obs,rewards,dones,extras):
        cleaned=dict(extras)
        timeout=cleaned.pop('time_outs',torch.zeros_like(dones,dtype=torch.bool)).bool()
        terminal=cleaned.get('terminated',torch.zeros_like(timeout)).bool()
        bootstrap=timeout & (~terminal)
        adjusted=rewards
        self.last_timeout_bonus=torch.zeros_like(rewards)
        if bool(bootstrap.any()):
            if 'final_observation' not in cleaned or 'final_observation_valid' not in cleaned:
                raise RuntimeError('time limit without a valid pre-reset final observation')
            if not cleaned['final_observation_valid'][bootstrap].all():
                raise RuntimeError('missing final observation for a truncated transition')
            final=TensorDict(cleaned['final_observation'],batch_size=[len(dones)]).to(self.device)
            with torch.no_grad():
                values=self.policy.evaluate(final).squeeze(-1)
            self.last_timeout_bonus=self.gamma*values*bootstrap.to(values.dtype)
            adjusted=rewards+self.last_timeout_bonus
        # Remove time_outs so upstream cannot add its old V(s_t) bootstrap.
        # Terminal transitions keep their unmodified rewards and done mask.
        super().process_env_step(obs,adjusted,dones,cleaned)

    def update(self):
        losses=super().update()
        if not all(math.isfinite(float(x)) for x in losses.values()):
            raise RuntimeError('nonfinite PPO loss')
        gradients=[p.grad for p in self.policy.parameters() if p.grad is not None]
        if not all(torch.isfinite(p).all() for p in self.policy.parameters()) or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('nonfinite PPO gradient or parameter')
        self.last_update_stats={**losses,'gradient_norm_after_clip':float(torch.sqrt(sum(g.square().sum() for g in gradients)))}
        # Read-only post-update telemetry on a deterministic rollout subset.
        # This neither samples actions nor updates observation normalization.
        with torch.no_grad():
            total=self.storage.num_transitions_per_env*self.storage.num_envs
            stride=max(1,total//4096)
            obs=self.storage.observations.flatten(0,1)[::stride]
            old_mu=self.storage.mu.flatten(0,1)[::stride]
            old_sigma=self.storage.sigma.flatten(0,1)[::stride]
            saved_distribution=self.policy.distribution
            actor_obs=self.policy.actor_obs_normalizer(self.policy.get_actor_obs(obs))
            self.policy.update_distribution(actor_obs)
            mu,sigma=self.policy.action_mean,self.policy.action_std
            kl=(torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1)
            logprob=self.policy.get_actions_log_prob(self.storage.actions.flatten(0,1)[::stride])
            ratio=torch.exp(logprob-self.storage.actions_log_prob.flatten(0,1)[::stride].squeeze(-1))
            self.last_update_stats.update(post_update_kl=float(kl.mean()),
                post_update_clip_fraction=float(((ratio-1.).abs()>self.clip_param).float().mean()),
                learning_rate=float(self.learning_rate),action_std_mean=float(sigma.mean()))
            self.policy.distribution=saved_distribution
        return losses


runner_module.FinalObservationPPO=FinalObservationPPO


class NativeRunner(OnPolicyRunner):
    def __init__(self,env,train_cfg,**kwargs):
        self.original_training_config=copy.deepcopy(train_cfg)
        self._learning_now=False
        cfg=copy.deepcopy(train_cfg)
        cfg['algorithm']['class_name']='FinalObservationPPO'
        super().__init__(env,cfg,**kwargs)

    def learn(self,num_learning_iterations,init_at_random_ep_len=False):
        initial=self.current_learning_iteration
        self._learning_now=True
        try:
            result=super().learn(num_learning_iterations,init_at_random_ep_len)
            self.current_learning_iteration=initial+num_learning_iterations
            return result
        finally:
            self._learning_now=False

    def save(self,path,infos=None):
        from .native_checkpoint import interface,rng_state,environment_state,training_signature,SCHEMA
        target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
        state=dict(model_state_dict=self.alg.policy.state_dict(),optimizer_state_dict=self.alg.optimizer.state_dict(),
            iter=self.current_learning_iteration+(1 if self._learning_now else 0),infos=infos,
            native=dict(schema=SCHEMA,interface=interface(self.env.unwrapped),
                canonical_restart=bool(getattr(self,'_canonical_checkpoint',False)),
                training_signature=training_signature(self.original_training_config),
                training_config=self.original_training_config,learning_rate=self.alg.learning_rate,
                scheduler=self.alg.schedule,rng=rng_state(),environment=environment_state(self.env.unwrapped),
                total_timesteps=self.tot_timesteps,total_training_time=self.tot_time))
        temporary=target.with_suffix(target.suffix+'.tmp')
        torch.save(state,temporary);temporary.replace(target)

    def save_restart(self,path,infos=None):
        """End unfinished episodes at an update boundary, then checkpoint.

        PhysX does not expose in-contact solver caches. Public resume files
        therefore begin new ground episodes with the full trained optimizer,
        normalizers, RNG and experiment counters preserved. Diagnostic
        model_i files retain arbitrary physical states, not a resume promise.
        """
        env=self.env.unwrapped
        aborted=int((env.episode_length_buf>0).sum())
        env.checkpoint_aborted_episodes=getattr(env,'checkpoint_aborted_episodes',0)+aborted
        if hasattr(env,'_episode_has_stepped'):env._episode_has_stepped.zero_()
        if hasattr(env,'_metrics_started'):env._metrics_started=False
        self.env.reset()
        self._canonical_checkpoint=True
        try:
            self.save(path,infos)
        finally:
            self._canonical_checkpoint=False

    def load(self,path,load_optimizer=True,map_location=None):
        from .native_checkpoint import interface,restore_rng,restore_environment,training_signature,SCHEMA
        state=torch.load(path,map_location=map_location or self.device,weights_only=False)
        metadata=state.get('native')
        if not metadata or metadata['schema']!=SCHEMA:
            raise ValueError('legacy checkpoint lacks the corrected native observation/physics contract')
        if metadata['interface']!=interface(self.env.unwrapped):
            raise ValueError('checkpoint observation/action/asset interface mismatch')
        if load_optimizer and metadata['training_signature']!=training_signature(self.original_training_config):
            raise ValueError('resume changes the PPO configuration; use explicit weight initialization instead')
        if load_optimizer and not metadata.get('canonical_restart',False):
            raise ValueError('full resume requires checkpoint.pt saved at a canonical episode boundary; model_i files are diagnostic/weight initialization only')
        self.alg.policy.load_state_dict(state['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.alg.learning_rate=metadata['learning_rate']
            self.alg.schedule=metadata['scheduler']
            self.current_learning_iteration=state['iter']
            self.tot_timesteps=metadata['total_timesteps'];self.tot_time=metadata['total_training_time']
            restore_environment(self.env.unwrapped,metadata['environment'])
            restore_rng(metadata['rng'])
        else:
            self.current_learning_iteration=0
        return state.get('infos')
