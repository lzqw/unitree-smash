"""Finite physical-loop BPTT/GAE learner using the pinned timebase equations.

This task adaptation is not a claim of reproducing SuperMimic's G1 model.
No PPO objective or demonstration target is used for actor gradients.
"""
import copy
import time
from pathlib import Path
import torch
from .supermimic_horizon_learner_v2 import frozen,optimizer_step
from .supermimic_timebase_v2 import bptt_return,gae_returns
from .supermimic_v1 import MLP,load_ppo_initialized_actor
from .supermimic_physical_loop_loading_v1 import APPROACH_NAMES,tensor_digest
from .supermimic_ball_sequence_v1 import ball_transition
from .supermimic_physical_dataset_v1 import valid_windows


class PhysicalHorizonTrainer:
    def __init__(self,actor,adapter,initial_state,timebase,*,command_delta_budget=0.005,
                 target_delta_budget_rad=0.04):
        self.actor,self.adapter,self.timebase=actor,adapter,timebase
        width=adapter.critic_input(initial_state).shape[-1]
        self.critic=MLP(width,(512,256,128),1).to(initial_state.physical.device)
        # deepcopy(ScriptModule) can turn copied Parameters into non-leaves.
        # Load a separate, hash-verified immutable reference once per learner.
        self.reference,reference_meta=load_ppo_initialized_actor(
            Path(adapter.manifest['checkpoint_paths']['source']),initial_state.physical.device)
        if reference_meta['sha256']!=adapter.manifest['checkpoint_sha256']['source']:
            raise ValueError('Reference actor source mismatch')
        self.reference.eval().requires_grad_(False)
        self.actor_opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=5e-5,betas=(.9,.95))
        self.world_opt=torch.optim.Adam(adapter.parameters(),lr=2e-4)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=2e-4)
        self.actor_updates=self.world_updates=self.critic_updates=0
        self.last_critic_loss=None
        self.anchor_coeff=.05;self.v_max=100./(1-timebase.gamma)
        if command_delta_budget <= 0 or target_delta_budget_rad <= 0:
            raise ValueError('Execution update budgets must be positive')
        self.command_delta_budget=float(command_delta_budget)
        self.target_delta_budget_rad=float(target_delta_budget_rad)
        self.actor_attempts=0
        self.rejected_actor_updates=0
        self.frozen_actor_hash=self.frozen_hash()

    def frozen_hash(self):
        return tensor_digest((k,v) for k,v in self.actor.state_dict().items() if k not in APPROACH_NAMES)

    def imagine_update(self,initial_state,*,update_actor=True):
        cfg=self.timebase;state=initial_state.branch()
        states=[];rewards=[];dones=[];valid=[];anchors=[]
        before={k:p.detach().clone() for k,p in self.actor.named_parameters() if p.requires_grad}
        actor_state_before={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        optimizer_state_before=copy.deepcopy(self.actor_opt.state_dict())
        with torch.no_grad():
            context=initial_state.clock_goal_context
            initial_request=self.actor(initial_state.observation,sample=False)[0]
            initial_preview,_,_=self.adapter.executor.forward(initial_state.observation,initial_request,
                context['joint_default'][0],context['limits'],initial_state.executor_memory)
            initial_command=initial_preview.final_joint_targets.clone()
            reference_request=self.reference(initial_state.observation,sample=False)[0]
            reference_preview,_,_=self.adapter.executor.forward(initial_state.observation,reference_request,
                context['joint_default'][0],context['limits'],initial_state.executor_memory)
            reference_command=reference_preview.final_joint_targets.clone()
        before_models=tensor_digest(self.adapter.state_dict().items())
        self.actor_opt.zero_grad(set_to_none=True);started=time.perf_counter()
        with frozen(self.adapter,self.critic), torch.set_grad_enabled(update_actor):
            for i in range(cfg.imagination_steps):
                current=self.adapter.critic_input(state)
                active=state.clock_goal_context['elapsed']<cfg.episode_s-1e-7
                if not active.any():break
                action=self.actor(state.observation,sample=False)[0]
                with torch.no_grad():reference=self.reference(state.observation,sample=False)[0]
                anchors.append((action-reference).square().mean(-1,keepdim=True)*active[:,None])
                state,reward,done,_=self.adapter(state,action)
                states.append(current.detach());rewards.append(reward*active[:,None]);dones.append(done);valid.append(active)
            if not states:raise ValueError('No active in-episode imagination start')
            final=self.adapter.critic_input(state)
            ret=bptt_return(torch.stack(rewards),torch.stack(dones),self.critic(final).clamp(max=self.v_max),cfg.gamma)
            loss=-ret.mean()+self.anchor_coeff*torch.stack(anchors).sum(0).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical BPTT objective')
            if update_actor:
                self.actor_attempts+=1
                loss.backward()
                parameters=[p for p in self.actor.parameters() if p.requires_grad]
                gradient=optimizer_step(self.actor_opt,parameters,element_clip=.1)
            else:gradient=loss.new_zeros(())
            critic_data=(torch.stack(states),torch.stack(rewards).detach(),torch.stack(dones).detach(),
                         final.detach(),torch.stack(valid))
        if update_actor:self.actor_updates+=1
        if before_models!=tensor_digest(self.adapter.state_dict().items()):raise RuntimeError('Actor step changed physical model')
        if self.frozen_hash()!=self.frozen_actor_hash:raise RuntimeError('Frozen base/recovery changed')
        del state,rewards,dones,anchors,final,loss
        bptt_seconds=time.perf_counter()-started
        states,rewards,dones,final,valid=critic_data
        with torch.no_grad():
            targets=gae_returns(self.critic(states),self.critic(final),rewards,dones,cfg.gamma,cfg.lam).clamp(max=self.v_max)
        x,y=states[valid],targets[valid]
        if not len(y):raise ValueError('No in-episode imagined critic states')
        losses=[];started=time.perf_counter()
        for epoch in range(5):
            for ids in torch.tensor_split(torch.randperm(len(y),device=y.device),4):
                self.critic_opt.zero_grad(set_to_none=True)
                loss=(self.critic(x[ids])-y[ids]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical GAE critic loss')
                loss.backward();optimizer_step(self.critic_opt,self.critic.parameters())
                self.critic_updates+=1;losses.append(float(loss.detach()))
        delta={k:float((p.detach()-before[k]).abs().max()) for k,p in self.actor.named_parameters() if p.requires_grad}
        self.last_critic_loss=sum(losses)/len(losses)
        with torch.no_grad():
            updated_request=self.actor(initial_state.observation,sample=False)[0]
            updated_preview,_,_=self.adapter.executor.forward(initial_state.observation,updated_request,
                context['joint_default'][0],context['limits'],initial_state.executor_memory)
            command_change=float((updated_preview.final_joint_targets-initial_command).abs().max())
            reference_command_delta=float((updated_preview.final_joint_targets-reference_command).abs().max())
            reference_request_delta=float((updated_request-reference_request).abs().max())
        accepted=True
        if update_actor and (reference_request_delta>self.command_delta_budget or
                             reference_command_delta>self.target_delta_budget_rad):
            self.actor.load_state_dict(actor_state_before,strict=True)
            self.actor_opt.load_state_dict(optimizer_state_before)
            self.rejected_actor_updates+=1
            self.actor_updates-=1
            accepted=False
            delta={k:0.0 for k in before}
            command_change=0.0
            request_change=0.0
        else:
            request_change=float((updated_request-initial_request).abs().max())
        return dict(actor_updates=self.actor_updates,actor_attempts=self.actor_attempts,
            rejected_actor_updates=self.rejected_actor_updates,update_accepted=accepted,
            command_delta_budget=self.command_delta_budget,target_delta_budget_rad=self.target_delta_budget_rad,
            critic_updates=self.critic_updates,world_updates=self.world_updates,
            imagined_return_mean=float(ret.detach().mean()),actor_gradient=float(gradient),parameter_max_changes=delta,
            final_command_max_change=command_change,request_max_change=request_change,
            reference_final_command_delta=reference_command_delta,reference_request_delta=reference_request_delta,
            critic_loss=self.last_critic_loss,bptt_seconds=bptt_seconds,critic_seconds=time.perf_counter()-started,
            model_runtime_state_sha256=before_models,imagined_steps=len(rewards),
            imagined_start_elapsed_min=float(initial_state.clock_goal_context['elapsed'].min()),
            imagined_start_elapsed_max=float(initial_state.clock_goal_context['elapsed'].max()),
            frozen_base_recovery_unchanged=True,model_unchanged_during_actor=True)

    def fit_world_window(self,window,launch):
        """One actual WM optimizer step on 16 native transitions per sample.

        Physical state, contact probability and task numeric memory propagate
        autoregressively. Recorded final commands are dynamics supervision;
        they are never used as imagined future actor commands.
        """
        cfg=self.timebase
        if len(window['state'])!=cfg.world_steps:raise ValueError('Wrong physical training window')
        state=window['state'][0];feet=window['feet'][0].float();present=window['present'][0]
        values,validity=window['values'][0],window['valid'][0]
        task=self.adapter.task_model;world=self.adapter.physical_model;ball=self.adapter.ball_model
        loss=state.new_zeros(());started=time.perf_counter()
        for i in range(cfg.world_steps):
            prediction,newfeet,newpresent=ball_transition(world,ball,state,window['targets'][i],
                window['previous_action'][i],feet,present,window['elapsed'][i],launch)
            out=task.forward_values(state,prediction,window['targets'][i],window['previous_action'][i],feet,
                window['elapsed'][i],values,validity)
            loss=loss+world.state_loss(prediction,window['next_state'][i])
            loss=loss+torch.nn.functional.binary_cross_entropy(newfeet,window['next_feet'][i].float())
            loss=loss+task.supervision_loss_values(out,window['next_values'][i],window['next_valid'][i],
                window['event_state'][i],window['next_event_state'][i],window['torque'][i],
                window['terminated'][i],window['complete_reward'][i])
            state,feet,present=prediction,newfeet,newpresent
            values,validity=out['numeric'],out['valid_logits'].sigmoid()
        loss=loss/cfg.world_steps
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical multistep model loss')
        self.world_opt.zero_grad(set_to_none=True);loss.backward()
        norm=optimizer_step(self.world_opt,self.adapter.parameters());self.world_updates+=1
        task.fit_updates.add_(1)
        return dict(world_loss=float(loss.detach()),world_gradient=float(norm),world_updates=self.world_updates,
                    world_seconds=time.perf_counter()-started)
