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
from .supermimic_ball_sequence_v1 import ball_transition,ball_error
from .supermimic_physical_dataset_v1 import valid_windows


class PhysicalHorizonTrainer:
    def __init__(self,actor,adapter,initial_state,timebase,*,command_delta_budget=0.005,
                 target_delta_budget_rad=0.04,native_reference_command_budget=None,
                 native_reference_target_budget_rad=None,trainable_names=None,
                 block_name='approach',actor_lr=5e-5,reference_bank=None,
                 tracking_weight=0.0):
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
        self.reference_bank=reference_bank
        self.tracking_weight=float(tracking_weight)
        if self.tracking_weight < 0:
            raise ValueError('tracking_weight must be nonnegative')
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
        self.native_reference_command_budget = (None if native_reference_command_budget is None
                                                else float(native_reference_command_budget))
        self.native_reference_target_budget_rad = (None if native_reference_target_budget_rad is None
                                                   else float(native_reference_target_budget_rad))
        if ((self.native_reference_command_budget is None) !=
                (self.native_reference_target_budget_rad is None)):
            raise ValueError('Native reference budgets must be both set or both unset')
        if self.native_reference_command_budget is not None and (
                self.native_reference_command_budget <= 0 or
                self.native_reference_target_budget_rad <= 0):
            raise ValueError('Native reference budgets must be positive')
        self.actor_attempts=0
        self.rejected_actor_updates=0
        self.backtracked_actor_updates=0
        self.block_attempts=0
        self.block_start_snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
        self.accepted_snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
        self.native_accepted_snapshot={k:v.detach().clone() for k,v in actor.state_dict().items()}
        self.budget_states=[initial_state.branch()]
        self.frozen_actor_hash=self.frozen_hash()

    def _actor_action(self, state, *, sample=False):
        """Use the causal post-hit latch for recovery-head imagination."""
        rally=state.task_memory.get('rally', {})
        if hasattr(self.actor, 'forward_with_recovery_gate') and 'had_contact' in rally and 'released' in rally:
            gate=(rally['had_contact'] > .5) & (rally['released'] > .5)
            return self.actor.forward_with_recovery_gate(state.observation, gate, sample=sample)
        return self.actor(state.observation, sample=sample)

    def _reference_action(self, state, *, sample=False):
        rally=state.task_memory.get('rally', {})
        if hasattr(self.reference, 'forward_with_recovery_gate') and 'had_contact' in rally and 'released' in rally:
            gate=(rally['had_contact'] > .5) & (rally['released'] > .5)
            return self.reference.forward_with_recovery_gate(state.observation, gate, sample=sample)
        return self.reference(state.observation, sample=sample)

    def _head_gate(self, state):
        """Return the actual exported gate used by the active trainable head."""
        if self.block_name == 'approach' and hasattr(self.actor, 'approach_gate'):
            return self.actor.approach_gate(state.observation).flatten()
        rally=state.task_memory.get('rally', {})
        if self.block_name == 'recovery' and 'had_contact' in rally and 'released' in rally:
            return ((rally['had_contact'] > .5) & (rally['released'] > .5)).float()
        return torch.ones(state.observation.shape[0], device=state.observation.device)

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
            if module is self.actor:
                request=self._actor_action(state,sample=False)[0]
            else:
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
        states=[];rewards=[];dones=[];valid=[];anchors=[];gate_activity=[];tracking_losses=[]
        tracking_reference = None
        if self.reference_bank is not None and self.tracking_weight > 0:
            tracking_reference = self.reference_bank.bind(initial_state.physical, self.block_name)
        before={k:p.detach().clone() for k,p in self.actor.named_parameters() if p.requires_grad}
        actor_state_before={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        optimizer_state_before=copy.deepcopy(self.actor_opt.state_dict())
        with torch.no_grad():
            context=initial_state.clock_goal_context
            initial_request=self._actor_action(initial_state,sample=False)[0]
            initial_preview,_,_=self.adapter.executor.forward(initial_state.observation,initial_request,
                context['joint_default'][0],context['limits'],initial_state.executor_memory)
            initial_command=initial_preview.final_joint_targets.clone()
            reference_request=self._reference_action(initial_state,sample=False)[0]
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
                action=self._actor_action(state,sample=False)[0]
                gate=self._head_gate(state)
                gate_activity.append(gate.detach())
                with torch.no_grad():reference=self._reference_action(state,sample=False)[0]
                anchors.append((action-reference).square().mean(-1,keepdim=True)*active[:,None])
                state,reward,done,_=self.adapter(state,action)
                if tracking_reference is not None:
                    tracking_losses.append(self.reference_bank.loss(
                        state.physical, state.clock_goal_context['elapsed'],
                        tracking_reference, self.block_name,
                        geometry=self.adapter.physical_model.geometry))
                states.append(current.detach());rewards.append(reward*active[:,None]);dones.append(done);valid.append(active)
            if not states:raise ValueError('No active in-episode imagination start')
            final=self.adapter.critic_input(state)
            ret=bptt_return(torch.stack(rewards),torch.stack(dones),self.critic(final).clamp(max=self.v_max),cfg.gamma)
            tracking_loss=(torch.stack(tracking_losses).mean() if tracking_losses else ret.new_zeros(()))
            loss=-ret.mean()+self.anchor_coeff*torch.stack(anchors).sum(0).mean()
            loss=loss+self.tracking_weight*tracking_loss
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical BPTT objective')
            effective_gate_steps=int(sum(int((g > .25).sum().item()) for g in gate_activity))
            effective_gate_lanes=int(max((int((g > .25).sum().item()) for g in gate_activity), default=0))
            no_effective_gate=bool(update_actor and effective_gate_steps == 0)
            if update_actor and not no_effective_gate:
                self.actor_attempts+=1
                loss.backward()
                parameters=[p for p in self.actor.parameters() if p.requires_grad]
                gradient=optimizer_step(self.actor_opt,parameters,element_clip=.1)
            else:gradient=loss.new_zeros(())
            critic_data=(torch.stack(states),torch.stack(rewards).detach(),torch.stack(dones).detach(),
                         final.detach(),torch.stack(valid))
        if update_actor and not no_effective_gate:self.actor_updates+=1
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
            updated_request=self._actor_action(initial_state,sample=False)[0]
            updated_preview,_,_=self.adapter.executor.forward(initial_state.observation,updated_request,
                context['joint_default'][0],context['limits'],initial_state.executor_memory)
            command_change=float((updated_preview.final_joint_targets-initial_command).abs().max())
            reference_command_delta=float((updated_preview.final_joint_targets-reference_command).abs().max())
            reference_request_delta=float((updated_request-reference_request).abs().max())
        accepted=True; alpha=1.0; budget_report=self._relative_budget_report()
        if update_actor and not no_effective_gate:
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
                local_ok = (report['relative_last_accepted']['final_normalized_command_delta'] <= self.command_delta_budget and
                            report['relative_last_accepted']['final_joint_target_delta_rad'] <= self.target_delta_budget_rad)
                native_ok = (self.native_reference_command_budget is None or
                             (report['relative_native_accepted']['final_normalized_command_delta'] <= self.native_reference_command_budget and
                              report['relative_native_accepted']['final_joint_target_delta_rad'] <= self.native_reference_target_budget_rad))
                if local_ok and native_ok:
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
                    updated_request=self._actor_action(initial_state,sample=False)[0]
                    updated_preview,_,_=self.adapter.executor.forward(initial_state.observation,updated_request,
                        context['joint_default'][0],context['limits'],initial_state.executor_memory)
                    command_change=float((updated_preview.final_joint_targets-initial_command).abs().max())
                    reference_command_delta=float((updated_preview.final_joint_targets-reference_command).abs().max())
                    reference_request_delta=float((updated_request-reference_request).abs().max())
                request_change=float((updated_request-initial_request).abs().max())
        else:
            request_change=float((updated_request-initial_request).abs().max())
            if update_actor and no_effective_gate:
                self.actor_opt.load_state_dict(optimizer_state_before)
                self.rejected_actor_updates+=1
                accepted=False
                alpha=0.0
                delta={k:0.0 for k in before}
                command_change=0.0
                request_change=0.0
        return dict(actor_updates=self.actor_updates,actor_attempts=self.actor_attempts,
            rejected_actor_updates=self.rejected_actor_updates,update_accepted=accepted,
            block_name=self.block_name,trainable_names=sorted(self.trainable_names),block_attempts=self.block_attempts,
            backtracked_actor_updates=self.backtracked_actor_updates,backtrack_alpha=alpha,
            command_delta_budget=self.command_delta_budget,target_delta_budget_rad=self.target_delta_budget_rad,
            native_reference_command_budget=self.native_reference_command_budget,
            native_reference_target_budget_rad=self.native_reference_target_budget_rad,
            critic_updates=self.critic_updates,world_updates=self.world_updates,
            imagined_return_mean=float(ret.detach().mean()),actor_gradient=float(gradient),parameter_max_changes=delta,
            tracking_loss=float(tracking_loss.detach()),tracking_weight=self.tracking_weight,
            tracking_steps=len(tracking_losses),
            final_command_max_change=command_change,request_max_change=request_change,
            reference_final_command_delta=reference_command_delta,reference_request_delta=reference_request_delta,
            budget_report=budget_report,
            critic_loss=self.last_critic_loss,bptt_seconds=bptt_seconds,critic_seconds=time.perf_counter()-started,
            model_runtime_state_sha256=before_models,imagined_steps=len(rewards),
            imagined_start_elapsed_min=float(initial_state.clock_goal_context['elapsed'].min()),
            imagined_start_elapsed_max=float(initial_state.clock_goal_context['elapsed'].max()),
            effective_gate_steps=effective_gate_steps,
            effective_gate_lanes=effective_gate_lanes,
            no_effective_gate=no_effective_gate,
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
        physical_loss=state.new_zeros(());contact_loss=state.new_zeros(())
        ball_loss_value=state.new_zeros(());task_loss=state.new_zeros(())
        for i in range(cfg.world_steps):
            prediction,newfeet,newpresent=ball_transition(world,ball,state,window['targets'][i],
                window['previous_action'][i],feet,present,window['elapsed'][i],launch)
            out=task.forward_values(state,prediction,window['targets'][i],window['previous_action'][i],feet,
                window['elapsed'][i],values,validity)
            physical_i=world.state_loss(prediction,window['next_state'][i])
            contact_i=torch.nn.functional.binary_cross_entropy(newfeet,window['next_feet'][i].float())
            ball_i=ball_error(prediction[:,49:],window['next_state'][i][:,49:])
            task_i=task.supervision_loss_values(out,window['next_values'][i],window['next_valid'][i],
                window['event_state'][i],window['next_event_state'][i],window['torque'][i],
                window['terminated'][i],window['complete_reward'][i])
            physical_loss=physical_loss+physical_i;contact_loss=contact_loss+contact_i
            ball_loss_value=ball_loss_value+ball_i;task_loss=task_loss+task_i
            loss=loss+physical_i+contact_i+task_i
            state,feet,present=prediction,newfeet,newpresent
            values,validity=out['numeric'],out['valid_logits'].sigmoid()
        loss=loss/cfg.world_steps
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical multistep model loss')
        self.world_opt.zero_grad(set_to_none=True);loss.backward()
        physical_loss=physical_loss/cfg.world_steps;contact_loss=contact_loss/cfg.world_steps
        ball_loss_value=ball_loss_value/cfg.world_steps;task_loss=task_loss/cfg.world_steps
        groups=(self.adapter.physical_model,self.adapter.ball_model,self.adapter.task_model)
        group_gradients=[]
        for group in groups:
            group_gradients.append(float(torch.sqrt(sum((p.grad.detach().square().sum() for p in group.parameters()
                                                        if p.grad is not None), loss.new_zeros(())))))
        norm=optimizer_step(self.world_opt,self.adapter.parameters());self.world_updates+=1
        task.fit_updates.add_(1)
        return dict(world_loss=float(loss.detach()),world_gradient=float(norm),world_updates=self.world_updates,
                    world_physical_loss=float(physical_loss.detach()),
                    world_contact_loss=float(contact_loss.detach()),
                    world_ball_loss=float(ball_loss_value.detach()),
                    world_task_loss=float(task_loss.detach()),
                    world_physical_gradient=group_gradients[0],
                    world_ball_gradient=group_gradients[1],
                    world_task_gradient=group_gradients[2],
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
