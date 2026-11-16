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
from .supermimic_physical_loop_loading_v1 import APPROACH_NAMES,RECOVERY_NAMES,HEAD_NAMES,set_actor_trainable,tensor_digest
from .supermimic_ball_sequence_v1 import ball_transition
from .supermimic_physical_dataset_v1 import valid_windows


class PhysicalHorizonTrainer:
    def __init__(self,actor,adapter,initial_state,timebase,*,command_delta_budget=0.005,
                 target_delta_budget_rad=0.04,trainable_names=None,block_name='approach',actor_lr=5e-5):
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
        self.block_name=block_name
        self.actor_lr=float(actor_lr)
        self.trainable_names=set_actor_trainable(actor, trainable_names or APPROACH_NAMES)
        self._actor_optimizers={}
        self.actor_opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=actor_lr,betas=(.9,.95))
        self._actor_optimizers[self.block_name]=self.actor_opt
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
        self.backtracked_actor_updates=0
        self.block_attempts=0
        self.block_start_snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
        self.accepted_snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
        self.native_accepted_snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
        self.budget_states=[initial_state.branch()]
        self.frozen_actor_hash=self.frozen_hash()

    def set_trainable_names(self,names,block_name=None):
        names=set(names)
        self.trainable_names=set_actor_trainable(self.actor,names)
        self.block_name=block_name or self.block_name
        # Keep one optimizer state per head. Switching Recovery→Approach must
        # not discard Adam moments or silently reinitialize the learning rate.
        if self.block_name not in self._actor_optimizers:
            self._actor_optimizers[self.block_name]=torch.optim.Adam(
                [p for p in self.actor.parameters() if p.requires_grad],lr=self.actor_lr,betas=(.9,.95))
        self.actor_opt=self._actor_optimizers[self.block_name]
        pending=getattr(self,'_pending_actor_optimizer_states',{}).pop(self.block_name,None)
        if pending is not None:
            self.actor_opt.load_state_dict(pending)
        self.actor_updates=0; self.actor_attempts=0; self.rejected_actor_updates=0; self.backtracked_actor_updates=0
        self.block_start_snapshot={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        self.accepted_snapshot={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        self.frozen_actor_hash=self.frozen_hash()

    def set_budget_state_library(self,states):
        if not states: raise ValueError('Empty execution budget state library')
        self.budget_states=[s.branch() for s in states]

    def _preview_command(self,state,actor_module=None):
        module=self.actor if actor_module is None else actor_module
        with torch.no_grad():
            request=module(state.observation,sample=False)[0]
            context=state.clock_goal_context
            preview,_,_=self.adapter.executor.forward(state.observation,request,context['joint_default'][0],context['limits'],state.executor_memory)
        return request.detach(),preview.final_normalized.detach(),preview.final_joint_targets.detach()

    def _budget_snapshot(self):
        rows=[]
        for state in self.budget_states:
            req,final,target=self._preview_command(state)
            rows.append((req,final,target))
        return rows

    @staticmethod
    def _max_budget_delta(now,base):
        return tuple(float((n-b).abs().max()) for n,b in zip(now,base))

    def _relative_budget_report(self):
        current=self._budget_snapshot()
        start=[];accepted=[]
        for cur,st,ac in zip(current,self._snapshot_from(self.block_start_snapshot),self._snapshot_from(self.accepted_snapshot)):
            start.append(self._max_budget_delta(cur,st)); accepted.append(self._max_budget_delta(cur,ac))
        def agg(rows): return {k:max((x[i] for x in rows),default=0.) for i,k in enumerate(('residual_request_delta','final_normalized_command_delta','final_joint_target_delta_rad'))}
        native=[]
        for cur,ref in zip(current,self._snapshot_from(self.native_accepted_snapshot)):
            native.append(self._max_budget_delta(cur,ref))
        return {'relative_block_start':agg(start),'relative_last_accepted':agg(accepted),
                'relative_native_accepted':agg(native)}

    def _snapshot_from(self,state_dict):
        # Snapshots are parameter snapshots. Recompute command baselines by temporarily loading
        # into the same actor; caller runs under no_grad and restores accepted parameters.
        current={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        self.actor.load_state_dict(state_dict,strict=True)
        rows=self._budget_snapshot()
        self.actor.load_state_dict(current,strict=True)
        return rows

    def accept_snapshot(self):
        self.accepted_snapshot={k:v.detach().clone() for k,v in self.actor.state_dict().items()}

    def frozen_hash(self):
        return tensor_digest((k,v) for k,v in self.actor.state_dict().items() if k not in self.trainable_names)

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
        proposal_state={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        proposal_optimizer=copy.deepcopy(self.actor_opt.state_dict())
        delta={k:float((p.detach()-before[k]).abs().max()) for k,p in self.actor.named_parameters() if p.requires_grad}
        self.last_critic_loss=sum(losses)/len(losses)
        with torch.no_grad():
            updated_request=self.actor(initial_state.observation,sample=False)[0]
            updated_preview,_,_=self.adapter.executor.forward(initial_state.observation,updated_request,
                context['joint_default'][0],context['limits'],initial_state.executor_memory)
            command_change=float((updated_preview.final_joint_targets-initial_command).abs().max())
            reference_command_delta=float((updated_preview.final_joint_targets-reference_command).abs().max())
            reference_request_delta=float((updated_request-reference_request).abs().max())
        accepted=True; alpha=1.0; budget_report=self._relative_budget_report()
        if update_actor:
            # Same proposal, same model/critic/start/batch: only scale the parameter
            # displacement.  No alternate batch or extra rollout is selected.
            base_state=actor_state_before
            for candidate_alpha in (1.0,0.5,0.25,0.125,0.0625):
                candidate={}
                for key,b in base_state.items():
                    p=proposal_state[key]
                    candidate[key]=(b+(p-b)*candidate_alpha) if (b.is_floating_point() or b.is_complex()) else p
                self.actor.load_state_dict(candidate,strict=True)
                report=self._relative_budget_report()
                if (report['relative_last_accepted']['final_normalized_command_delta'] <= self.command_delta_budget and
                    report['relative_last_accepted']['final_joint_target_delta_rad'] <= self.target_delta_budget_rad):
                    alpha=candidate_alpha;budget_report=report;break
            else:
                self.actor.load_state_dict(actor_state_before,strict=True)
                self.actor_opt.load_state_dict(optimizer_state_before)
                self.rejected_actor_updates+=1;self.actor_updates-=1;accepted=False
                delta={k:0.0 for k in before};command_change=0.0;request_change=0.0
            if accepted:
                self.actor_opt.load_state_dict(proposal_optimizer)
                if alpha < 1.0:self.backtracked_actor_updates+=1
                self.accept_snapshot()
                # Refresh diagnostics for the accepted scaled proposal.
                with torch.no_grad():
                    updated_request=self.actor(initial_state.observation,sample=False)[0]
                    updated_preview,_,_=self.adapter.executor.forward(initial_state.observation,updated_request,
                        context['joint_default'][0],context['limits'],initial_state.executor_memory)
                    command_change=float((updated_preview.final_joint_targets-initial_command).abs().max())
                    reference_command_delta=float((updated_preview.final_joint_targets-reference_command).abs().max())
                    reference_request_delta=float((updated_request-reference_request).abs().max())
                request_change=float((updated_request-initial_request).abs().max())
        else:
            request_change=float((updated_request-initial_request).abs().max())
        return dict(actor_updates=self.actor_updates,actor_attempts=self.actor_attempts,
            rejected_actor_updates=self.rejected_actor_updates,update_accepted=accepted,
            block_name=self.block_name,trainable_names=sorted(self.trainable_names),block_attempts=self.block_attempts,
            backtracked_actor_updates=self.backtracked_actor_updates,backtrack_alpha=alpha,
            command_delta_budget=self.command_delta_budget,target_delta_budget_rad=self.target_delta_budget_rad,
            critic_updates=self.critic_updates,world_updates=self.world_updates,
            imagined_return_mean=float(ret.detach().mean()),actor_gradient=float(gradient),parameter_max_changes=delta,
            final_command_max_change=command_change,request_max_change=request_change,
            reference_final_command_delta=reference_command_delta,reference_request_delta=reference_request_delta,
            budget_report=budget_report,
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

    def calibrate_critic_native(self, states, returns, *, epochs=2):
        """Fit the contextual critic to same-policy native return-to-go labels.

        This is a separately reported task adaptation.  Labels are only accepted
        when they come from one frozen policy's completed native archive.
        """
        x=torch.cat([self.adapter.critic_input(s) for s in states],dim=0).detach()
        y=returns.to(x.device).float().reshape(-1,1).detach()
        losses=[]
        for _ in range(epochs):
            self.critic_opt.zero_grad(set_to_none=True)
            pred=self.critic(x)
            loss=(pred-y).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite native critic calibration loss')
            loss.backward();optimizer_step(self.critic_opt,self.critic.parameters());self.critic_updates+=1
            losses.append(float(loss.detach()))
        with torch.no_grad(): error=self.critic(x)-y
        return dict(native_critic_calibration_updates=epochs,native_critic_calibration_loss=sum(losses)/len(losses),
                    native_critic_mae=float(error.abs().mean()),native_critic_rmse=float(error.square().mean().sqrt()),
                    native_label_count=int(y.shape[0]))

    @staticmethod
    def audit_native_rtg_stages(states, returns, metadata):
        """Validate stage RTG labels without updating the critic."""
        if not (len(states) == len(returns) == len(metadata)):
            raise ValueError('Stage RTG states, returns and metadata must align')
        rows=[]
        for state, value, meta in zip(states, returns, metadata):
            if not isinstance(meta, dict) or not meta.get('policy_id') or meta.get('episode_id') is None:
                raise ValueError('Native RTG metadata requires policy_id and episode_id')
            value=value.detach().reshape(-1)
            if value.numel() != state.observation.shape[0]:
                raise ValueError('Stage RTG batch does not match state batch')
            rows.append(dict(policy_id=str(meta['policy_id']), episode_id=str(meta['episode_id']),
                             time_s=float(meta['time_s']), count=int(value.numel()),
                             return_min=float(value.min()), return_max=float(value.max()),
                             return_mean=float(value.mean())))
        return dict(policy_id=sorted({r['policy_id'] for r in rows}), stages=rows,
                    total_labels=sum(r['count'] for r in rows), source='completed_native_archive',
                    critic_updates=0)
