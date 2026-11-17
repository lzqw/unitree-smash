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

    def _rollout_objective(self, initial_state, *, effective_weights=None):
        from .supermimic_approach_data_v1 import masked_lane_mean
        cfg=self.timebase; state=initial_state.branch()
        states=[]; rewards=[]; dones=[]; valid=[]; anchors=[]; gates=[]; tracks=[]
        selected=state.clock_goal_context.get('actor_lane_mask',torch.ones(len(state.physical),dtype=torch.bool,device=state.physical.device))
        survival=state.physical.new_ones(len(state.physical))
        reference=None
        if self.reference_bank is not None and self.tracking_weight>0 and self.block_name in ('approach','recovery'):
            # G07's causal current plan is available without consulting archived
            # future contacts. Missing plans use current-state phase matching.
            base=getattr(state.executor_memory,'reference_controller',None)
            event_time=getattr(base,'event_time',None) if self.block_name=='approach' else None
            if event_time is not None and hasattr(base,'planned'):
                event_time=torch.where(base.planned,event_time,torch.full_like(event_time,float('nan')))
            reference=self.reference_bank.bind(state.physical,self.block_name,geometry=self.adapter.physical_model.geometry,
                elapsed=state.clock_goal_context['elapsed'],event_time=event_time)
        for _ in range(cfg.imagination_steps):
            active=state.clock_goal_context['elapsed']<cfg.episode_s-1e-7
            if not active.any():break
            current=self.adapter.critic_input(state)
            action=self._actor_action(state,sample=False)[0]
            gate=self._head_gate(state).reshape(-1)
            usable=selected&active
            gates.append((gate.detach()*survival.detach()*usable).detach())
            with torch.no_grad(): ref_action=self._reference_action(state,sample=False)[0]
            anchors.append((action-ref_action).square().mean(-1)*active*survival.detach())
            track_mask=usable&(gate.detach()>.25)
            rally=state.task_memory.get('rally',{})
            if self.block_name=='approach':
                if 'had_contact' not in rally:track_mask=torch.zeros_like(track_mask)
                else:track_mask=track_mask&(rally['had_contact'].reshape(-1)<.5)
            state,reward,done,_=self.adapter(state,action)
            if reference is not None:
                tracks.append(self.reference_bank.loss(state.physical,state.clock_goal_context['elapsed'],reference,self.block_name,
                    geometry=self.adapter.physical_model.geometry,valid_mask=track_mask,weights=survival))
            states.append(current.detach());rewards.append(reward*active[:,None]);dones.append(done);valid.append(active)
            survival=survival*(1-done.reshape(-1))
        if not states:raise ValueError('No active in-episode imagination start')
        final=self.adapter.critic_input(state)
        ret=bptt_return(torch.stack(rewards),torch.stack(dones),self.critic(final).clamp(max=self.v_max),cfg.gamma)
        activity=torch.stack(gates); weights=(activity.sum(0)>.25).detach()
        # Backtracking must not improve a loss merely by dropping the lanes
        # that were selected by the original proposal's gate/survival trace.
        if effective_weights is not None:weights=effective_weights.detach()
        task_loss=-masked_lane_mean(ret,weights)
        anchor_loss=masked_lane_mean(torch.stack(anchors).sum(0),weights)
        tracking_loss=torch.stack(tracks).mean() if tracks else ret.sum()*0.
        loss=task_loss+self.tracking_weight*tracking_loss+self.anchor_coeff*anchor_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical BPTT objective')
        return dict(loss=loss,task=task_loss,tracking=tracking_loss,anchor=anchor_loss,ret=ret,effective_weights=weights,
                    gate_steps=int((activity>.25).sum()),gate_lanes=int(weights.sum()),
                    critic_data=(torch.stack(states),torch.stack(rewards).detach(),torch.stack(dones).detach(),final.detach(),torch.stack(valid)),
                    tracking_steps=len(tracks),matched_references=int(reference['matched'].sum()) if reference else 0)

    def imagine_update(self,initial_state,*,update_actor=True):
        from .supermimic_approach_data_v1 import gradient_parts
        cfg=self.timebase; started=time.perf_counter()
        actor_before={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        optimizer_before=copy.deepcopy(self.actor_opt.state_dict())
        model_hash=tensor_digest(self.adapter.state_dict().items())
        parameters=[p for p in self.actor.parameters() if p.requires_grad]
        self.actor_opt.zero_grad(set_to_none=True)
        before_budget=self._budget_snapshot()
        cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        def restore_rng():
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        accepted=False; alpha=0.; gradient=0.; reason=None; grad_stats={}; trials=[]
        if update_actor:self.actor_attempts+=1
        try:
            with frozen(self.adapter,self.critic),torch.set_grad_enabled(update_actor):
                rollout=self._rollout_objective(initial_state)
                baseline_loss=float(rollout['loss'].detach())
                if update_actor and rollout['gate_lanes']:
                    grad_stats=gradient_parts({k:rollout[k] for k in ('task','tracking','anchor')},parameters)
                    # Anchor-only momentum is not an effective task update.
                    if grad_stats['task_gradient_norm']+self.tracking_weight*grad_stats['tracking_gradient_norm']>1e-12:
                        rollout['loss'].backward()
                        norm=torch.sqrt(sum((p.grad.detach().square().sum() for p in parameters if p.grad is not None),rollout['loss'].new_zeros(())))
                        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite actor gradient')
                        if float(norm)>1e-12:
                            gradient=float(optimizer_step(self.actor_opt,parameters,element_clip=.1))
                            proposed={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
                            proposed_opt=copy.deepcopy(self.actor_opt.state_dict())
                            for candidate_alpha in (1.,.5,.25,.125,.0625):
                                candidate={k:(v+(proposed[k]-v)*candidate_alpha if k in self.trainable_names else v) for k,v in actor_before.items()}
                                self.actor.load_state_dict(candidate,strict=True)
                                with torch.no_grad():
                                    report=self._relative_budget_report()
                                    local=report['relative_last_accepted']; native=report['relative_native_accepted']
                                    bounded=(local['final_normalized_command_delta']<=self.command_delta_budget and local['final_joint_target_delta_rad']<=self.target_delta_budget_rad)
                                    if self.native_reference_command_budget is not None:
                                        bounded &= (native['final_normalized_command_delta']<=self.native_reference_command_budget and native['final_joint_target_delta_rad']<=self.native_reference_target_budget_rad)
                                    # Measure final executable commands on identical real states/memory.
                                    effect=max(self._max_budget_delta(n,b)[2] for n,b in zip(self._budget_snapshot(),before_budget))
                                    restore_rng(); candidate_loss=float(self._rollout_objective(initial_state,effective_weights=rollout['effective_weights'])['loss'])
                                trials.append(dict(alpha=candidate_alpha,loss=candidate_loss,budget_ok=bool(bounded),final_command_change=effect))
                                if bounded and effect>0 and candidate_loss<=baseline_loss:
                                    alpha=candidate_alpha;accepted=True;break
                            if accepted:
                                # Adam moments correspond to the original same-gradient proposal;
                                # alpha scales only the committed parameter displacement.
                                self.actor_opt.load_state_dict(proposed_opt)
                            else:reason='objective_budget_or_zero_final_command_response'
                        else:reason='zero_total_gradient'
                    else:reason='no_task_or_tracking_gradient'
                elif update_actor:reason='no_effective_gate'
            critic_data=rollout['critic_data']
            ret_mean=float(rollout['ret'].detach().mean()); track_value=float(rollout['tracking'].detach())
            gate_steps=rollout['gate_steps'];gate_lanes=rollout['gate_lanes'];track_steps=rollout['tracking_steps']
            matched=rollout['matched_references']
            del rollout
        except BaseException:
            self.actor.load_state_dict(actor_before,strict=True);self.actor_opt.load_state_dict(optimizer_before)
            raise
        if update_actor:
            if accepted:
                self.actor_updates+=1; self.backtracked_actor_updates+=int(alpha<1);self.accept_snapshot()
            else:
                self.actor.load_state_dict(actor_before,strict=True);self.actor_opt.load_state_dict(optimizer_before)
                self.rejected_actor_updates+=1
        if model_hash!=tensor_digest(self.adapter.state_dict().items()):raise RuntimeError('Actor update changed model')
        if self.frozen_hash()!=self.frozen_actor_hash:raise RuntimeError('Frozen actor state changed')
        # Critic updates only AFTER all fixed-objective comparisons finish.
        bptt_seconds=time.perf_counter()-started;started=time.perf_counter()
        states,rewards,dones,final,valid=critic_data
        with torch.no_grad():targets=gae_returns(self.critic(states),self.critic(final),rewards,dones,cfg.gamma,cfg.lam).clamp(max=self.v_max)
        x,y=states[valid],targets[valid]; losses=[]
        if not len(y):raise ValueError('No valid critic data')
        for _ in range(5):
            for ids in torch.tensor_split(torch.randperm(len(y),device=y.device),4):
                if not ids.numel():continue
                self.critic_opt.zero_grad(set_to_none=True); loss=(self.critic(x[ids])-y[ids]).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite GAE critic loss')
                loss.backward();optimizer_step(self.critic_opt,self.critic.parameters());self.critic_updates+=1;losses.append(float(loss.detach()))
        self.last_critic_loss=sum(losses)/len(losses)
        changes={k:float((p.detach()-actor_before[k]).abs().max()) for k,p in self.actor.named_parameters() if p.requires_grad}
        command_delta=[self._max_budget_delta(n,b) for n,b in zip(self._budget_snapshot(),before_budget)]
        return dict(actor_updates=self.actor_updates,actor_attempts=self.actor_attempts,rejected_actor_updates=self.rejected_actor_updates,
            update_accepted=accepted if update_actor else False,block_name=self.block_name,trainable_names=sorted(self.trainable_names),
            block_attempts=self.block_attempts,backtracked_actor_updates=self.backtracked_actor_updates,backtrack_alpha=alpha,
            rejection_reason=reason,backtrack_trials=trials,objective_before=baseline_loss,
            objective_after=trials[-1]['loss'] if accepted else baseline_loss,adam_scaling='full proposal moments with alpha-scaled parameters',
            command_delta_budget=self.command_delta_budget,target_delta_budget_rad=self.target_delta_budget_rad,
            critic_updates=self.critic_updates,world_updates=self.world_updates,imagined_return_mean=ret_mean,
            actor_gradient=gradient,parameter_max_changes=changes,tracking_loss=track_value,tracking_weight=self.tracking_weight,
            tracking_steps=track_steps,matched_references=matched,**grad_stats,
            final_command_max_change=max(x[2] for x in command_delta),request_max_change=max(x[0] for x in command_delta),
            budget_report=self._relative_budget_report(),critic_loss=self.last_critic_loss,bptt_seconds=bptt_seconds,
            critic_seconds=time.perf_counter()-started,model_runtime_state_sha256=model_hash,imagined_steps=len(rewards),
            effective_gate_steps=gate_steps,effective_gate_lanes=gate_lanes,no_effective_gate=gate_lanes==0,
            no_effective_gradient=reason in ('zero_total_gradient','no_task_or_tracking_gradient'),
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

    def calibrate_critic_native(self, states, returns, *, epochs=2, masks=None):
        """Fit the contextual critic to same-policy native return-to-go labels.

        This is a separately reported task adaptation.  Labels are only accepted
        when they come from one frozen policy's completed native archive.
        """
        x=torch.cat([self.adapter.critic_input(s) for s in states],dim=0).detach()
        y=returns.to(x.device).float().reshape(-1,1).detach()
        if masks is not None:
            mask=torch.cat(masks).to(x.device).bool()
            x=x[mask];y=y[mask]
            if not len(y): raise ValueError('No complete same-policy calibration samples')
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
