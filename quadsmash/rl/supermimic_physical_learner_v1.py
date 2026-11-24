"""Finite physical-loop BPTT/GAE learner using the pinned timebase equations.

This task adaptation is not a claim of reproducing SuperMimic's G1 model.
No PPO objective or demonstration target is used for actor gradients.
"""
import copy
import math
import time
from pathlib import Path
import torch
from .supermimic_horizon_learner_v2 import frozen,optimizer_step
from .supermimic_timebase_v2 import bptt_return,gae_returns
from .supermimic_v1 import MLP,ResidualPPOInitializedActor,load_ppo_initialized_actor
from .supermimic_policy_execution_v1 import recovery_gate
from .supermimic_physical_loop_loading_v1 import APPROACH_NAMES,RECOVERY_NAMES,HEAD_NAMES,set_actor_trainable,tensor_digest
from .supermimic_ball_sequence_v1 import ball_transition,ball_error
from .supermimic_physical_dataset_v1 import valid_windows
from .supermimic_physical_imagination_adapter_v1 import PhysicalImaginationAdapter
from .supermimic_task_space_reference_v1 import TaskSpaceReferenceBankV1
from .supermimic_racket_dynamics_v1 import racket_loss_components, racket_state, robot_loss


def supervised_transition(adapter,state,targets,previous,feet,present,elapsed,launch):
    def step(*args):
        return ball_transition(adapter.physical_model,adapter.ball_model,*args,action_model=adapter.action_ball_model)
    args=(state,targets,previous,feet,present,elapsed,launch)
    if getattr(adapter,'checkpoint_supervised_rollout',False) and torch.is_grad_enabled():
        from torch.utils.checkpoint import checkpoint
        return checkpoint(step,*args,use_reentrant=False,preserve_rng_state=False)
    return step(*args)


def native_flight_rollout_loss(adapter,window,launch,loss_kind="huber", *, supervise_body=False):
    """Supervise free-running predictions at multiple horizons using archived commands."""
    losses={"huber": torch.nn.functional.smooth_l1_loss, "mse": torch.nn.functional.mse_loss}
    if loss_kind not in losses:
        raise ValueError('Unknown native flight loss')
    velocity_loss=losses[loss_kind]
    state=window['state'][0]
    feet=window['feet'][0].float()
    present=window['present'][0]
    errors=[]
    body_errors=[]
    for i in range(64):
        state,feet,present=supervised_transition(adapter,state,window['targets'][i],
            window['previous_action'][i],feet,present,window['elapsed'][i],launch)
        if i+1 in (4,16,32,64):
            truth=window['next_state'][i]
            error=velocity_loss(state[:,52:55],truth[:,52:55],reduction='none').mean(-1)
            mask=window['present'][i].reshape(-1).to(error.dtype)
            errors.append((error*mask).sum()/mask.sum().clamp_min(1.))
            if supervise_body:
                geometry=adapter.physical_model.geometry
                racket=racket_loss_components(racket_state(state,geometry),racket_state(truth,geometry))
                gyro=torch.nn.functional.smooth_l1_loss(state[:,10:13],truth[:,10:13],beta=.1)
                body_errors.append(robot_loss(state,truth)+gyro+sum(racket.values())/len(racket))
    flight=torch.stack(errors).mean()
    return (flight,torch.stack(body_errors).mean()) if supervise_body else flight


class PhysicalHorizonTrainer:
    def __init__(self,actor,adapter,initial_state,timebase,*,command_delta_budget=0.005,
                 target_delta_budget_rad=0.04,native_reference_command_budget=None,
                 native_reference_target_budget_rad=None,trainable_names=None,
                 block_name='approach',actor_lr=5e-5,reference_bank=None,
                 tracking_weight=0.0,world_lr=2e-4,racket_loss_weight=1.0,
                 approach_objective='task',toi_racket_weight=1.0,toi_plane_offset_m=0.):
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
        if approach_objective not in ('task','toi_racket_only'):
            raise ValueError('Unknown Approach objective')
        self.approach_objective=approach_objective
        self.toi_racket_weight=float(toi_racket_weight)
        self.toi_plane_offset_m=float(toi_plane_offset_m)
        if self.toi_racket_weight < 0:
            raise ValueError('toi_racket_weight must be nonnegative')
        self.natural_flight_loss="huber"
        self.direct_long_body_supervision=False
        self.ball_loss_weight=4.0
        self.contact_ball_loss_weight=8.0
        self.racket_loss_weight=float(racket_loss_weight)
        if self.racket_loss_weight < 0:
            raise ValueError('racket_loss_weight must be nonnegative')
        if self.tracking_weight < 0:
            raise ValueError('tracking_weight must be nonnegative')
        self.actor_lr=float(actor_lr)
        self.trainable_names=set_actor_trainable(actor, trainable_names or APPROACH_NAMES)
        self._actor_optimizers={}
        self.actor_opt=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=actor_lr,betas=(.9,.95))
        self._actor_optimizers[self.block_name]=self.actor_opt
        if world_lr <= 0:
            raise ValueError('world_lr must be positive')
        self.world_lr=float(world_lr)
        self.world_opt=torch.optim.Adam(adapter.parameters(),lr=self.world_lr)
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
        from .supermimic_policy_execution_v1 import policy_forward
        # CPU toy actors without the native gate interface exercise the generic
        # learner; production ResidualPPOInitializedActor always takes the strict path.
        if hasattr(self.actor,'forward_with_recovery_gate'):
            return policy_forward(self.actor,state.observation,state.task_memory.get('rally',{}),sample=sample)
        return self.actor(state.observation,sample=sample)

    def _reference_action(self, state, *, sample=False):
        from .supermimic_policy_execution_v1 import policy_forward
        reference = getattr(self, 'approach_v2_anchor', self.reference)
        if hasattr(reference,'forward_with_recovery_gate'):
            return policy_forward(reference,state.observation,state.task_memory.get('rally',{}),sample=sample)
        return reference(state.observation,sample=sample)

    def _rollout_requests(self, state):
        reference = getattr(self, 'approach_v2_anchor', self.reference)
        if type(self.actor) is ResidualPPOInitializedActor and type(reference) is ResidualPPOInitializedActor:
            observation = state.observation
            latch = recovery_gate(state.task_memory.get('rally', {}), observation)
            candidate_parameters = getattr(self, '_approach_candidate_parameters', None)
            if candidate_parameters is None:
                action = self.actor.gated_request(observation, latch)
            else:
                module = self.actor.actor
                normalized = module.normalizer(observation)
                candidate_count = len(next(iter(candidate_parameters.values())))
                from .supermimic_approach_data_v1 import batched_approach_head
                approach = batched_approach_head(
                    normalized.reshape(candidate_count, -1, normalized.shape[-1]), candidate_parameters)
                action = (module.base(normalized)
                          + latch.reshape(-1, 1) * module.recovery(normalized) * module.recovery_scale
                          + self.actor.approach_action_gate(observation)
                          * approach.flatten(0, 1) * module.approach_scale).clamp(-1., 1.)
            with torch.no_grad():
                anchor = reference.gated_request(observation, latch)
            gate = latch.float() if self.block_name == 'recovery' else self._head_gate(state)
        else:
            action = self._actor_action(state, sample=False)[0]
            gate = self._head_gate(state)
            with torch.no_grad():
                anchor = self._reference_action(state, sample=False)[0]
        return action, anchor, gate.reshape(-1)

    def _head_gate(self, state):
        """Return the actual exported gate used by the active trainable head."""
        if self.block_name == 'approach' and hasattr(self.actor, 'approach_gate'):
            return self.actor.approach_gate(state.observation).flatten()
        rally=state.task_memory.get('rally', {})
        if self.block_name == 'recovery' and 'had_contact' in rally and 'released' in rally:
            return ((rally['had_contact'] > .5) & (rally['released'] > .5)).float()
        return torch.ones(state.observation.shape[0], device=state.observation.device)

    def set_trainable_names(self,names,block_name=None,*,reset_block=True):
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
        if reset_block:
            self.actor_updates=0; self.actor_attempts=0; self.rejected_actor_updates=0; self.backtracked_actor_updates=0
            self.block_start_snapshot={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
            self.accepted_snapshot={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        self.frozen_actor_hash=self.frozen_hash()

    def set_budget_state_library(self,states):
        if not states: raise ValueError('Empty execution budget state library')
        library=[s.branch() for s in states]
        prepare=getattr(self.adapter.executor, 'prepare_fixed_command', None)
        bases=[] if prepare is not None else None
        if bases is not None:
            for state in library:
                context=state.clock_goal_context
                bases.append(prepare(state.observation,context['joint_default'][0],context['limits'],state.executor_memory))
        self.budget_states=library
        self._fixed_budget_bases=bases

    def _preview_command(self,state,actor_module=None,*,fixed_base=None):
        module=self.actor if actor_module is None else actor_module
        with torch.no_grad():
            if module is self.actor:
                request=self._actor_action(state,sample=False)[0]
            else:
                request=module(state.observation,sample=False)[0]
            context=state.clock_goal_context
            default,limits=context['joint_default'][0],context['limits']
            if fixed_base is None:
                preview,_,_=self.adapter.executor.forward(state.observation,request,default,limits,state.executor_memory)
            else:
                base,memory=fixed_base
                preview=self.adapter.executor.mixer(base,request,state.observation,default,limits,memory)
        return request.detach(),preview.final_normalized.detach(),preview.final_joint_targets.detach()

    def _budget_snapshot(self):
        bases=getattr(self,'_fixed_budget_bases',None)
        if bases is None:
            return [self._preview_command(state) for state in self.budget_states]
        return [self._preview_command(state,fixed_base=base)
                for state,base in zip(self.budget_states,bases,strict=True)]

    @staticmethod
    def _max_budget_delta(now,base):
        return tuple(float((n-b).abs().max()) for n,b in zip(now,base))

    @staticmethod
    def _budget_deltas(current, reference):
        # Reduce on-device before reading back. Each tensor covers the same command field
        # for one native state library entry, including entries of different batch sizes.
        maxima = [torch.stack([(a-b).abs().amax() for a,b in zip(now,base,strict=True)])
                  for now,base in zip(current,reference,strict=True)]
        names = ('residual_request_delta','final_normalized_command_delta','final_joint_target_delta_rad')
        return dict(zip(names,torch.stack(maxima).amax(0).tolist(),strict=True))

    def _budget_references(self):
        return {name: self._snapshot_from(snapshot) for name,snapshot in (
            ('relative_block_start',self.block_start_snapshot),
            ('relative_last_accepted',self.accepted_snapshot),
            ('relative_native_accepted',self.native_accepted_snapshot))}

    def _relative_budget_report(self, current, references):
        return {name: self._budget_deltas(current,reference) for name,reference in references.items()}

    def _snapshot_from(self,state_dict):
        # Snapshots are parameter snapshots. Recompute command baselines by temporarily loading
        # into the same actor; caller runs under no_grad and restores accepted parameters.
        current={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        try:
            self.actor.load_state_dict(state_dict,strict=True)
            return self._budget_snapshot()
        finally:
            self.actor.load_state_dict(current,strict=True)

    def accept_snapshot(self):
        self.accepted_snapshot={k:v.detach().clone() for k,v in self.actor.state_dict().items()}

    def frozen_hash(self):
        return tensor_digest((k,v) for k,v in self.actor.state_dict().items() if k not in self.trainable_names)

    def tracking_coefficient(self, head):
        if head=='approach' and getattr(self,'approach_objective','task')=='toi_racket_only':
            return self.toi_racket_weight
        return self.tracking_weight

    def _rollout_objective(self, initial_state, *, effective_weights=None, baseline_toi_weights=None,
                           baseline_landing_weights=None, baseline_ball_scored=None,
                           tracking_parameters=(), collect_critic_data=True, defer_speed_risk=False,
                           coarse_guard=False):
        ball_protocol = getattr(self, 'approach_ball_protocol', None)
        if ball_protocol is not None:
            collect_critic_data = False
        if isinstance(initial_state,tuple):
            from .supermimic_recovery_batches import pooled_objective
            return pooled_objective(self,initial_state,effective_weights,baseline_toi_weights,baseline_landing_weights,
                                    baseline_ball_scored=baseline_ball_scored, collect_critic_data=collect_critic_data)
        from .supermimic_approach_data_v1 import masked_lane_mean
        cfg=self.timebase; state=initial_state.branch()
        scoring_step = None
        if not collect_critic_data and not torch.is_grad_enabled():
            factory = getattr(getattr(self.adapter, 'executor', None), 'private_scoring_step', None)
            if factory is not None:
                scoring_step = factory(state.executor_memory)
        states=[]; rewards=[]; dones=[]; valid=[]; anchors=[]; gates=[]; tracks=[]; track_weights=[]; track_states=[]
        selected=state.clock_goal_context.get('actor_lane_mask',torch.ones(len(state.physical),dtype=torch.bool,device=state.physical.device))
        survival=state.physical.new_ones(len(state.physical))
        reference=None
        toi_only = self.block_name=='approach' and getattr(self,'approach_objective','task')=='toi_racket_only'
        contact_weight=getattr(self,'contact_aux_weight',0.) if toi_only else 0.
        revised = getattr(self, 'landing_and_first_contact_objectives', False)
        first_geometry = revised and toi_only
        if ball_protocol is not None:
            if not first_geometry:
                raise ValueError('Ball-only Actor requires the revised TOI geometry objective')
            from .supermimic_outgoing_flight import ball_only_flight_terms
            from .supermimic_release_calibration import racket_release_boundary, speed_bias_at_lead
            released = torch.zeros_like(selected)
            after_release = torch.zeros_like(selected, dtype=torch.long)
            resolved = torch.zeros_like(selected)
            release_ball = state.physical.new_zeros((len(selected), 6))
            closest_approach = state.physical.new_full((len(selected),), 2.)
            impact_lead = torch.zeros_like(after_release)
            approach_speed = state.physical.new_full((len(selected),), torch.nan)
        net_proxy = getattr(self, 'approach_net_proxy', None) if ball_protocol is not None else None
        native_gate = getattr(self, 'approach_native_gate', False) and ball_protocol is not None
        net_clearance = None
        if net_proxy is not None:
            from .supermimic_robot_net_clearance import robot_net_clearance, robot_net_cost
            net_clearance = state.physical.new_full((len(selected),), torch.inf)
        if first_geometry:
            swing_started = torch.zeros_like(selected)
            first_string = torch.zeros_like(selected)
            first_cost = state.physical.new_ones(len(selected))
            first_margin = state.physical.new_zeros(len(selected))
        dynamic_landing = revised and self.block_name == 'recovery' and getattr(self, 'recovery_rollout_landing', False)
        stability_limits = getattr(self, 'recovery_stability_limits', None)
        relaxed_limits = getattr(self, 'relaxed_recovery_limits', None)
        landing_weights = []
        if revised and self.block_name == 'recovery':
            recovery_costs = []
            aux = state.task_memory['auxiliary']
            landing = aux['jump']['landing_time'].reshape(-1)
            contact = state.task_memory['rally']['contact_time'].reshape(-1)
            landed = ((aux['contact_recovery']['qualified_landing'].reshape(-1) > .5)
                      | ((landing > 0) & (landing >= contact))).detach()
            if dynamic_landing:
                from .supermimic_recovery_batches import update_recovery_landing, relaxed_recovery_step
                initial_landing = landed
            if relaxed_limits is not None:
                soft_run = torch.zeros_like(landing)
                landing = torch.where(landed, landing, torch.zeros_like(landing))
        speed_goal = getattr(self, 'relaxed_release_speed_goal', None)
        score_release = contact_weight and (speed_goal is not None or getattr(self, 'score_downward_release', False))
        release_velocity = state.physical.new_zeros((len(selected), 3)) if score_release else None
        if contact_weight or first_geometry:
            outgoing=state.physical.new_zeros((len(selected),6))
            impacted=torch.zeros_like(selected)
        if toi_only:
            from .supermimic_toi_racket_objective import bind_intercept_reference, intercept_loss, precontact_mask
        tracking_coeff=self.tracking_coefficient(self.block_name)
        if self.reference_bank is not None and tracking_coeff>0 and self.block_name in ('approach','recovery'):
            # G07's causal current plan is available without consulting archived
            # future contacts. Missing plans use current-state phase matching.
            base=getattr(state.executor_memory,'reference_controller',None)
            event_time=getattr(base,'event_time',None) if self.block_name=='approach' else None
            if event_time is not None and hasattr(base,'planned'):
                event_time=torch.where(base.planned,event_time,torch.full_like(event_time,float('nan')))
            if toi_only:
                reference=bind_intercept_reference(state,self.reference_bank,self.adapter.physical_model.geometry,
                                                   cfg.imagination_steps*cfg.control_dt_s,plane_offset_m=self.toi_plane_offset_m)
            else:
                reference=self.reference_bank.bind(state.physical,self.block_name,geometry=self.adapter.physical_model.geometry,
                    elapsed=state.clock_goal_context['elapsed'],event_time=event_time)
            if reference is not None:
                reference['matched'] = reference['matched'] & selected
                if ball_protocol is not None and self.actor.early_arm_gate is not None and not coarse_guard:
                    from .supermimic_toi_racket_objective import search_outgoing_target
                    cache = getattr(self, '_proposal_ball_targets', {})
                    key = id(initial_state)
                    if key not in cache:
                        v2_tracking = getattr(self, 'approach_v2', False)
                        measured = reference.get('measured_interior_success')
                        search_reference = (dict(reference, matched=reference['matched'] & ~measured)
                                            if v2_tracking and measured is not None else reference)
                        cache[key] = search_outgoing_target(
                            search_reference, self.adapter.action_ball_model.impulse_prior, self.research_court,
                            minimum_speed=ball_protocol['minimum_outgoing_speed_mps'] + (.25 if v2_tracking else 0.),
                            interior_m=.30 if native_gate else .12 if v2_tracking else .03)
                    reference['ball_only_contact'] = cache[key]
        toi_coverage_preserved=True
        reuse_geometry = (not torch.is_grad_enabled() and reference is not None and not toi_only
                          and type(self.reference_bank) is TaskSpaceReferenceBankV1
                          and type(self.adapter) is PhysicalImaginationAdapter)
        maximum_steps = cfg.imagination_steps + 60 if native_gate else cfg.imagination_steps
        planned_steps = maximum_steps
        approach_v2 = getattr(self, 'approach_v2', False) and ball_protocol is not None
        margin_m = .012 if approach_v2 else .005
        if ball_protocol is not None:
            followthrough_ticks = ball_protocol['minimum_followthrough_ticks']
            if net_proxy is not None:
                followthrough_ticks = max(followthrough_ticks, 60 if native_gate else 12)
            from .supermimic_toi_racket_objective import approach_horizon
            planned_steps = approach_horizon(reference, selected, state.clock_goal_context['elapsed'],
                                             maximum=maximum_steps, dt=cfg.control_dt_s,
                                             followthrough_ticks=followthrough_ticks)
        for tick in range(maximum_steps):
            active=state.clock_goal_context['elapsed']<cfg.episode_s-1e-7
            if not active.any():break
            if collect_critic_data:
                states.append(self.adapter.critic_input(state).detach())
                valid.append(active & selected)
            action, ref_action, gate = self._rollout_requests(state)
            usable=selected&active
            if ball_protocol is not None:
                usable = usable & ~resolved
                if not usable.any():
                    break
                if tick >= (min(planned_steps, cfg.imagination_steps) if native_gate else planned_steps):
                    # An unsettled contact overrides the short horizon.
                    # An unresolved miss retains the measured geometry objective.
                    settling = usable & impacted
                    if not settling.any():
                        break
                    usable = settling
            gates.append((gate.detach()*survival.detach()*usable).detach())
            anchors.append((action-ref_action).square().mean(-1)*active*survival.detach())
            track_mask=usable&(gate.detach()>.25)
            rally=state.task_memory.get('rally',{})
            if toi_only:
                track_mask=precontact_mask(state,selected,gate,cfg.episode_s)
                if first_geometry:
                    swing_started = swing_started | track_mask
                    track_mask = swing_started & usable
            elif self.block_name=='approach':
                if 'had_contact' not in rally:track_mask=torch.zeros_like(track_mask)
                else:track_mask=track_mask&(rally['had_contact'].reshape(-1)<.5)
            previous=state
            gradient_options = {'detach_ball': True} if toi_only else {}
            if hasattr(self, 'research_court'):
                gradient_options = dict(detach_ball=True, ball_input_grad=bool(contact_weight))
            if score_release and speed_goal is not None:
                gradient_options['impulse_residual_scale'] = self.relaxed_impulse_residual_scale
            if scoring_step is not None:
                gradient_options['scoring_step'] = scoring_step
            if reuse_geometry:
                gradient_options['return_geometry'] = True
            if ball_protocol is not None:
                gradient_options.update(detach_ball=native_gate, ball_input_grad=True, return_geometry=True)
                if native_gate and tick >= 32:
                    beyond = ((state.physical[:, 49] > self.research_court['net_plane_x_m'] + .15)
                              & (state.physical[:, 52] > 0))
                    if bool((~usable | (released & beyond)).all()):
                        gradient_options['robot_net_tail'] = True
            state,reward,done,transition=self.adapter(state,action,**gradient_options)
            if net_proxy is not None:
                clearance = robot_net_clearance(transition['next_geometry'], net_proxy, self.research_court)
                net_clearance = torch.where(usable, torch.minimum(net_clearance, clearance), net_clearance)
            if revised and self.block_name == 'recovery':
                # Train on post-landing correction while the Critic keeps its original return target.
                old_tilt = 2 * previous.physical[:, 4:6].square().sum(-1)
                tilt = 2 * state.physical[:, 4:6].square().sum(-1)
                roll_pitch_rate = state.physical[:, 10:12].square().sum(-1)
                support = state.clock_goal_context['foot_contact'].clamp(0, 1).sum(-1)
                height_deficit = (.18 - state.physical[:, 2]).relu()
                cost = tilt + (tilt - old_tilt).relu() + .05 * roll_pitch_rate + height_deficit + (.5 - support).relu()
                if stability_limits is not None:
                    from .supermimic_recovery_batches import recovery_stability_cost
                    cost = cost + recovery_stability_cost(state.physical, stability_limits)
                if dynamic_landing:
                    next_landed = update_recovery_landing(landed, previous.task_memory, state.clock_goal_context['foot_contact'])
                    if relaxed_limits is not None:
                        elapsed = state.clock_goal_context['elapsed'].reshape(-1)
                        landing = torch.where(next_landed & ~landed, elapsed.detach(), landing)
                        within_deadline = next_landed & (elapsed - landing <= relaxed_limits['landing_deadline_s'])
                        cost, soft_run, progress = relaxed_recovery_step(
                            state.physical, state.clock_goal_context['foot_contact'],
                            relaxed_limits, soft_run, cfg.control_dt_s, active=within_deadline)
                    landed = next_landed
                    weight = landed * track_mask * survival.detach()
                    if baseline_landing_weights is not None:
                        # A later predicted landing must not erase costs paid by the reference proposal.
                        weight = torch.maximum(weight, baseline_landing_weights[tick])
                    landing_weights.append(weight)
                    stage_cost = cost * cfg.control_dt_s
                    if relaxed_limits is not None:
                        stage_cost = stage_cost - progress
                    recovery_costs.append(stage_cost * weight)
                else:
                    recovery_costs.append(cost * landed * track_mask * survival.detach() * cfg.control_dt_s)
            if contact_weight or first_geometry:
                impact=transition['impulse_info']
                candidate = (usable & (survival.detach() > 0)) if first_geometry else track_mask
                first=candidate & impact['active'].reshape(-1) & (impact['object_id'].reshape(-1)==0) & ~impacted
                if first_geometry and first.any():
                    if score_release:
                        velocity = transition['impact_outgoing_velocity']
                        if velocity is None:
                            raise ValueError('First-impact release scoring requires the TOI impulse velocity')
                        release_velocity = torch.where(first[:, None], velocity, release_velocity)
                    from .supermimic_toi_racket_objective import first_racket_cost
                    details = {key: impact[key][first] for key in ('time', 'pre_position', 'local')}
                    cost = first_racket_cost(previous.physical[first], state.physical[first], details,
                            self.adapter.physical_model.geometry, self.adapter.action_ball_model.impulse_prior.preview,
                            self.string_strips, self.string_boundary, cfg.control_dt_s, minimum=margin_m)
                    if approach_v2 and reference is not None:
                        arrival = reference['intercept']['time'][first]
                        reachable = reference['matched'][first]
                        clock = previous.clock_goal_context['elapsed'].reshape(-1)[first] + impact['time'][first]
                        target = torch.where(reachable, arrival, clock.detach())
                        cost = cost + .2 * torch.nn.functional.smooth_l1_loss(
                            (clock - target) / .02, torch.zeros_like(clock), reduction='none') * reachable
                    first_cost = first_cost.masked_scatter(first, cost)
                    part = impact['racket_part'].reshape(-1)
                    first_string |= first & (part >= 0) & (part < self.string_strips)
                    inward, offset = self.string_boundary
                    margins = (details['local'].detach()[:, 1:] @ inward.T - offset).amin(-1)
                    first_margin = first_margin.masked_scatter(first, margins)
                if contact_weight:
                    outgoing=torch.where(first[:,None],transition['outgoing_ball'],outgoing)
                impacted=impacted|first
            if ball_protocol is not None:
                tool = transition['next_geometry']['position'][:, self.adapter.physical_model.geometry.ids['Racket']]
                ball = transition['outgoing_ball']
                distance = (tool - ball[:, :3].detach()).norm(dim=-1)
                near = usable & ~impacted & state.clock_goal_context['ball_present'].reshape(-1).bool()
                closest_approach = torch.where(near, torch.minimum(closest_approach, distance), closest_approach)
                events = state.task_memory['rally']
                impact_lead = torch.where(first, tick, impact_lead)
                approach_speed = torch.where(first, -impact['normal_speed'].detach(), approach_speed)
                release_candidate = usable & impacted & ~released
                release_edge = release_candidate & racket_release_boundary(
                    ball, transition['next_geometry'], self.adapter.physical_model.geometry.ids['Racket'],
                    self.adapter.action_ball_model.impulse_prior.preview, separation_m=self.approach_separation_m)
                release_ball = torch.where(release_edge[:, None], ball, release_ball)
                recontact = released & usable & impact['active'] & (impact['object_id'] == 0)
                after_release = after_release + (released & usable).long()
                released = released | release_edge
                # A subsequent racket contact invalidates the single-stroke flight estimate.
                faulted = (events['double_hit_fault'].reshape(-1) > .5) | recontact
                first_string = first_string & ~(usable & faulted)
                pending = (events['followthrough_crossed'].reshape(-1) > .5) & (
                    events['followthrough_finished'].reshape(-1) <= .5)
                if native_gate:
                    from .supermimic_robot_net_clearance import racket_back_on_own_side
                    back = racket_back_on_own_side(
                        transition['next_geometry'], net_proxy, self.adapter.physical_model.geometry.ids['Racket'],
                        self.research_court)
                    settled = (ball[:, 2] <= self.adapter.action_ball_model.impulse_prior.preview.radius)
                    pending = (pending | ~back) & ~settled
                resolved = resolved | (usable & released & ~pending
                    & (after_release >= followthrough_ticks))
            if reference is not None and not coarse_guard:
                if toi_only:
                    fixed_weights = None
                    if baseline_toi_weights is not None:
                        fixed_weights = (baseline_toi_weights[tick] if tick < len(baseline_toi_weights)
                                         else torch.zeros_like(baseline_toi_weights[0]))
                    numerator,weights=intercept_loss(previous.physical,state.physical,
                        previous.clock_goal_context['elapsed'].reshape(-1),state.clock_goal_context['elapsed'].reshape(-1),
                        reference,track_mask,survival,self.adapter.physical_model.geometry,
                        fixed_weights=fixed_weights,
                        detach_empty=True, boundary=getattr(self, 'string_boundary', None),
                        minimum_margin=margin_m,
                        prefer_measured_contact=native_gate,
                        relative_face_speed=getattr(self, 'approach_relative_face_speed',
                                                    getattr(self, 'prioritize_approach_flight', False)))
                else:
                    numerator,denominator,weights=self.reference_bank.loss(state.physical,state.clock_goal_context['elapsed'],reference,self.block_name,
                        geometry=self.adapter.physical_model.geometry,valid_mask=track_mask,weights=survival,return_components=True,
                        **({'fk':transition['next_geometry']} if reuse_geometry else {}))
                tracks.append(numerator);track_weights.append(weights);track_states.append(state.physical)
            rewards.append(reward*active[:,None]);dones.append(done)
            if ball_protocol is None:
                survival=survival*(1-done.reshape(-1))
        if coarse_guard:
            return dict(string=first_string & selected, clearance=net_clearance,
                        complete=~impacted | resolved)
        if not rewards:raise ValueError('No active in-episode imagination start')
        final = None
        if ball_protocol is None:
            final=self.adapter.critic_input(state)
            ret=bptt_return(torch.stack(rewards),torch.stack(dones),self.critic(final).clamp(max=self.v_max),cfg.gamma)
        else:
            ret = state.physical.new_zeros(len(selected))
        activity=torch.stack(gates); weights=(activity.sum(0)>.25).detach()
        # Backtracking must not improve a loss merely by dropping the lanes
        # that were selected by the original proposal's gate/survival trace.
        if effective_weights is not None:weights=effective_weights.detach()
        task_loss=-masked_lane_mean(ret,weights)
        if revised and self.block_name == 'recovery':
            task_loss = task_loss + masked_lane_mean(torch.stack(recovery_costs).sum(0), weights)
        flight_metrics = {}
        flight_losses = {}
        flight_cost_by_lane = None
        flight_priority_loss = None
        guard_flight_cost = None
        net_loss = None
        scored = None
        speed_risk = None
        outcome_audit = []
        if ball_protocol is not None:
            scored = resolved & first_string & weights.bool().reshape(-1)
            audit = (initial_state.clock_goal_context.get('approach_diagnostic')
                     if getattr(self, 'audit_approach_outcomes', False) else None)
            if audit is not None:
                audit_terms = ('speed', 'direction', 'net', 'landing')
                lane_costs = release_ball.new_zeros((len(selected), len(audit_terms)))
            denominator = weights.sum().clamp_min(1.)
            # Unresolved starts keep continuous pre-contact geometry supervision.
            task_loss = (closest_approach * (~impacted) * weights).sum() / denominator
            if net_clearance is not None:
                net_clearance = torch.where(selected & net_clearance.isfinite(), net_clearance, torch.nan)
                net_loss = (robot_net_cost(net_clearance.nan_to_num(nan=1.)) * weights).sum() / denominator
                task_loss = task_loss + .2 * net_loss
            if getattr(self, 'prioritize_approach_flight', False):
                flight_cost_by_lane = ret.new_full(ret.shape, torch.nan)
            if getattr(self, 'approach_speed_risk', False):
                from .supermimic_approach_speed_risk import group_mean, incomplete_risk, release_risk

                codes = initial_state.clock_goal_context['approach_diagnostic'][:, -1]
                groups = (codes == 1).long() + 2 * (codes == 3).long()
                geometry = torch.where(impacted, first_cost, closest_approach)
                lane_risk = incomplete_risk(geometry)
                guard_flight_cost = ret.new_zeros(ret.shape)
            if scored.any():
                terms = ball_only_flight_terms(
                    release_ball[scored, :3], release_ball[scored, 3:],
                    self.adapter.action_ball_model.impulse_prior, self.research_court,
                    speed_bias=speed_bias_at_lead(impact_lead[scored], getattr(self, 'approach_speed_calibration', None)),
                    minimum=ball_protocol['minimum_outgoing_speed_mps'], speed_margin=.25 if approach_v2 else 0.,
                    net_margin=getattr(self, 'approach_net_margin_m', .02),
                    landing_margin=.30 if native_gate else .03, center_landing=native_gate)
                flight_metrics['ball_speed_loss_supported_lanes'] = int((terms['speed'].detach() > 1e-6).sum())
                if audit is not None:
                    lane_costs[scored] = torch.stack(tuple(terms[name].detach() for name in audit_terms), dim=-1)
                if flight_cost_by_lane is not None:
                    flight_cost_by_lane[scored] = sum(value.detach() for value in terms.values())
                if getattr(self, 'approach_speed_risk', False):
                    lane_risk = lane_risk.masked_scatter(scored, release_risk(release_ball[scored, 3:].norm(dim=-1), terms['speed']))
                    guard_flight_cost = guard_flight_cost.masked_scatter(scored, sum(terms.values()))
                confidence = (first_margin[scored].detach() / .012).clamp(.2, 1.) if approach_v2 else 1.
                terms = {name: (value * confidence).sum() / denominator for name, value in terms.items()}
                ball_loss = sum(terms.values())
                task_loss = task_loss + (.075 if native_gate else .25 if approach_v2 else 1.) * ball_loss
                if flight_cost_by_lane is not None:
                    flight_priority_loss = ball_loss
                flight_losses = terms
                flight_metrics.update({f'ball_loss_{name}': float(value.detach()) for name, value in flight_losses.items()})
            if getattr(self, 'approach_speed_risk', False):
                speed_risk = (lane_risk, groups, scored)
                if not defer_speed_risk:
                    risk_loss = group_mean(lane_risk, groups)
                    task_loss = task_loss + (.075 if native_gate else .25) * risk_loss
                    flight_losses['grouped_speed_risk'] = risk_loss
            if getattr(self, 'diagnose_ball_gradients', False):
                geometry = (closest_approach * (~impacted) + first_cost * impacted) * weights
                flight_losses['geometry'] = geometry.sum() / denominator
            flight_metrics.setdefault('ball_speed_loss_supported_lanes', 0)
            flight_metrics['resolved_ball_lanes'] = int(scored.sum())
            flight_metrics['planned_imagination_steps'] = planned_steps
            flight_metrics['model_rollout_steps'] = len(rewards)
            target = reference.get('ball_only_contact') if reference is not None else None
            flight_metrics['demand_racket_target_lanes'] = int(target['valid'].sum()) if target is not None else 0
            if audit is not None:
                keys = audit.tolist()
                flags = torch.stack((first_string, released, scored, release_ball[:, 3:].detach().norm(dim=-1)), dim=-1)
                values = torch.cat((flags, lane_costs), dim=-1).tolist()
                for (stream, lane, episode, code), (string, release, scored_lane, speed, *costs) in zip(keys, values, strict=True):
                    if code:
                        components = dict(zip(audit_terms, costs, strict=True)) if scored_lane else None
                        speed_loss = components['speed'] if components is not None else None
                        outcome_audit.append(dict(episode=[stream, lane, episode],
                            outcome='low_speed' if code == 1 else 'success', predicted_legal_strings=bool(string),
                            predicted_release=bool(release), release_speed_mps=speed if release else None,
                            flight_cost_terms=components, speed_loss=speed_loss,
                            speed_loss_active=speed_loss is not None and speed_loss > 1e-6))
        elif toi_only:
            task_loss=task_loss.detach()
            if contact_weight:
                from .supermimic_outgoing_flight import first_string_flight_loss, outgoing_cost
                impact_lanes = impacted & weights.bool().reshape(-1)
                prior = self.adapter.action_ball_model.impulse_prior
                if revised:
                    task_loss = first_string_flight_loss(
                        outgoing, weights, first_string, prior, self.research_court,
                        release_velocity=release_velocity, speed_goal=speed_goal)
                elif impact_lanes.any():
                    cost = outgoing_cost(outgoing[impact_lanes], prior, self.research_court)
                    task_loss = cost.sum() / weights.sum().clamp_min(1.)
                else:
                    task_loss = outgoing.sum() * 0.
        anchor_loss=masked_lane_mean(torch.stack(anchors).sum(0),weights)
        all_track_weights=torch.stack(track_weights) if tracks else None
        if toi_only and baseline_toi_weights is not None and tracks:
            overlap = min(len(tracks), len(baseline_toi_weights))
            toi_coverage_preserved = bool(
                ((baseline_toi_weights[:overlap] <= 0) | (all_track_weights[:overlap] > 0)).all()
                & (baseline_toi_weights[overlap:] <= 0).all())
        if baseline_ball_scored is not None:
            toi_coverage_preserved = toi_coverage_preserved and bool((~baseline_ball_scored | scored).all())
        track_weight_sum=all_track_weights.sum() if tracks else ret.new_zeros(())
        tracking_loss=torch.stack(tracks).sum()/track_weight_sum.clamp_min(1e-8) if tracks else ret.sum()*0.
        if first_geometry and tracks:
            supervised = all_track_weights.detach().sum(0)
            tracking_loss = tracking_loss + (first_cost * supervised).sum() / supervised.sum().clamp_min(1e-8)
        nonzero_lanes=0
        tracking_derivatives=None
        tracking_vjp_wall_s=0.
        if (tracks and torch.is_grad_enabled() and tracking_loss.requires_grad
                and (not getattr(self, 'approach_v2', False)
                     or getattr(self, 'diagnose_ball_gradients', False))):
            tracking_started = time.perf_counter()
            differentiable=[i for i,x in enumerate(track_states) if x.requires_grad]
            inputs = [track_states[i] for i in differentiable]
            # Pooled records need the same VJP at both the physical states and the Actor.
            inputs.extend(tracking_parameters)
            gradients = torch.autograd.grad(tracking_loss,inputs,retain_graph=True,allow_unused=True) if inputs else ()
            if tracking_parameters:
                tracking_derivatives = gradients[len(differentiable):]
            responding=torch.zeros(len(state.physical),dtype=torch.bool,device=state.physical.device)
            for g,i in zip(gradients[:len(differentiable)],differentiable,strict=True):
                w = track_weights[i]
                if g is not None:responding|=(g.detach().abs().sum(-1)>0)&(w>0)
            nonzero_lanes=int(responding.sum())
            tracking_vjp_wall_s = time.perf_counter() - tracking_started
        loss=tracking_coeff*tracking_loss+self.anchor_coeff*anchor_loss
        if not toi_only or ball_protocol is not None:
            loss=loss+task_loss
        elif contact_weight:
            loss=loss+contact_weight*task_loss
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical BPTT objective')
        from .supermimic_dual_head_quality_v1 import reference_coverage
        coverage=reference_coverage(self.reference_bank,self.block_name,tracking_coeff)
        coverage['selected_start_lanes']=int(selected.sum())
        coverage['gate_active_selected_start_lanes']=int((gates[0]>.25).sum())
        if reference and 'reference_digests' in reference and tracks:
            used=all_track_weights.detach().gt(0).any(0)
            ids,counts=reference['indices'][used].unique(return_counts=True)
            coverage['reference_usage_by_sha256']={reference['reference_digests'][i]:n for i,n in zip(ids.tolist(),counts.tolist())}
        coverage.update(matched_lanes=int(reference['matched'].sum()) if reference else 0,
            effective_weight_sum=float(track_weight_sum.detach()),tracking_applied=bool(tracks and float(track_weight_sum.detach())>0))
        aux_stats={'outgoing_impact_lanes': int(impact_lanes.sum())} if ball_protocol is None and contact_weight and torch.is_grad_enabled() else {}
        if first_geometry:
            coverage.update(predicted_first_margin_sum_m=float((first_margin * impacted * weights).sum()),
                            predicted_first_string_lanes=int((first_string & weights).sum()),
                            predicted_first_frame_lanes=int((impacted & ~first_string & weights).sum()))
        landing_trace = torch.stack(landing_weights) if landing_weights else None
        if landing_trace is not None and baseline_landing_weights is None:
            added = (landing_trace > 0) & ~initial_landing
            coverage.update(newly_landed_lanes=int(added.any(0).sum()), newly_landed_lane_steps=int(added.sum()))
        return dict(**aux_stats, **flight_metrics, outcome_audit=outcome_audit, ball_scored=scored, flight_losses=flight_losses,
                    speed_risk_by_lane=speed_risk, guard_flight_cost_by_lane=guard_flight_cost,
                    robot_net_clearance_by_lane=net_clearance, robot_net_loss=net_loss,
                    flight_cost_by_lane=flight_cost_by_lane, flight_priority_loss=flight_priority_loss,
                    string_margin_by_lane=(first_margin.masked_fill(~(first_string & weights.bool()), torch.nan)
                                           if ball_protocol is not None and first_geometry else None),
                    ball_approach_speed=approach_speed if ball_protocol is not None else None,
                    landing_weights=landing_trace,tracking_derivatives=tracking_derivatives,
                    tracking_vjp_wall_s=tracking_vjp_wall_s,
                    toi_weights=all_track_weights.detach() if toi_only and tracks else None,
                    toi_coverage_preserved=toi_coverage_preserved,reference_coverage=coverage,loss=loss,task=task_loss,tracking=tracking_loss,anchor=anchor_loss,ret=ret,effective_weights=weights,
                    gate_steps=int((activity>.25).sum()),gate_lanes=int(weights.sum()),
                    critic_data=((torch.stack(states),torch.stack(rewards).detach(),torch.stack(dones).detach(),final.detach(),torch.stack(valid))
                                 if collect_critic_data else None),
                    tracking_steps=len(tracks),tracking_objective=getattr(self,'approach_objective','task') if self.block_name=='approach' else 'recovery_reference',
                    tracking_active_lane_steps=int((all_track_weights>0).sum()) if tracks else 0,
                    tracking_effective_weight_sum=float(track_weight_sum.detach()),tracking_nonzero_gradient_lanes=nonzero_lanes,matched_references=int(reference['matched'].sum()) if reference else 0)

    def benchmark_approach_objective(self, starts):
        """Measure a real forward/backward without stepping the policy or its optimizer."""
        from .supermimic_g07_memory_preview_v1 import G07BatchMemory

        parameters = tuple(p for p in self.actor.parameters() if p.requires_grad)
        self._proposal_ball_targets = {}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with frozen(self.adapter, self.critic), torch.enable_grad():
            rollout = self._rollout_objective(starts, collect_critic_data=False)
            gradients = torch.autograd.grad(rollout['loss'], parameters, allow_unused=True)
            norm = torch.stack([g.norm() for g in gradients if g is not None]).norm()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not torch.isfinite(norm) or norm <= 0:
            raise FloatingPointError('Approach benchmark did not produce a finite task gradient')
        lanes = sum(len(s.physical) for s in starts)
        executed = sum(row['imagination']['lanes'] * row['imagination']['executed_steps']
                       for row in rollout['reference_coverage']['per_record'])
        packed = sum(len(s.physical) for s in starts if not isinstance(s.executor_memory, G07BatchMemory))
        return dict(lanes=lanes, packed_lanes=packed, executed_lane_ticks=executed, seconds=elapsed,
                    lane_ticks_per_second=executed / elapsed, api_peak_bytes=torch.cuda.max_memory_allocated(),
                    gradient_norm=float(norm))

    @torch.no_grad()
    def _native_gate_candidates(self, before, endpoint, before_budget, references, starts, stop):
        from .supermimic_compact_starts import merge_physical_starts
        from .supermimic_robot_net_clearance import check_robot_net_clearance

        if stop is not None and stop():
            return False, 0., before_budget, []
        self.actor.load_state_dict(endpoint, strict=True)
        endpoint_budget = self._budget_snapshot()
        delta = torch.cat([(new[2] - old[2]).flatten()
                           for new, old in zip(endpoint_budget, before_budget, strict=True)])
        rms = float(delta.square().mean().sqrt())
        target_rms = getattr(self, '_native_gate_target_rms', .0003)
        alpha = min(1., target_rms / max(rms, 1.e-12))
        candidates, trials = [], []
        for fraction in (alpha, alpha * .5):
            candidate = {key: torch.lerp(value, endpoint[key], fraction) if key in self.trainable_names else value
                         for key, value in before.items()}
            self.actor.load_state_dict(candidate, strict=True)
            budget = self._budget_snapshot()
            report = self._relative_budget_report(budget, references)
            local = report['relative_last_accepted']
            native = report['relative_native_accepted']
            bounded = (local['final_normalized_command_delta'] <= self.command_delta_budget
                       and local['final_joint_target_delta_rad'] <= self.target_delta_budget_rad)
            if self.native_reference_command_budget is not None:
                bounded &= (native['final_normalized_command_delta'] <= self.native_reference_command_budget
                            and native['final_joint_target_delta_rad'] <= self.native_reference_target_budget_rad)
            effect = self._budget_deltas(budget, before_budget)['final_joint_target_delta_rad']
            actual = torch.cat([(new[2] - old[2]).flatten() for new, old in zip(budget, before_budget, strict=True)])
            trial = dict(alpha=fraction, budget_ok=bounded, budget_report=report, final_command_change=effect,
                         final_target_delta_rms_rad=float(actual.square().mean().sqrt()), target_rms_rad=target_rms)
            trials.append(trial)
            if bounded and effect > 0.:
                candidates.append((candidate, budget, trial))
        self.actor.load_state_dict(before, strict=True)
        if not candidates or (stop is not None and stop()):
            if not candidates:
                self._native_gate_target_rms = max(2.5e-5, target_rms * .5)
            return False, 0., before_budget, trials

        group_count = len(candidates) + 1
        states = [before, *(candidate for candidate, _, _ in candidates)]
        prefix = 'actor.approach.'
        expected_layers = ('Linear', 'ELU', 'Linear', 'ELU', 'Linear')
        layers = tuple(getattr(module, 'original_name', type(module).__name__)
                       for module in self.actor.actor.approach.children())
        if layers != expected_layers:
            raise ValueError(f'Unsupported exported Approach head for candidate batching: {layers}')
        self._approach_candidate_parameters = {
            key[len(prefix):]: torch.stack([state[key] for state in states])
            for key in before if key.startswith(prefix)}
        try:
            scores = [self._rollout_objective(merge_physical_starts(
                                                  [start.branch() for _ in states], pack_execution=True),
                                              collect_critic_data=False, coarse_guard=True)
                      for start in starts]
        finally:
            self._approach_candidate_parameters = None
        string, clearance, complete = [
            torch.cat([score[key].reshape(group_count, -1) for score in scores], dim=1)
            for key in ('string', 'clearance', 'complete')]
        winner = None
        for i, (candidate, budget, trial) in enumerate(candidates, 1):
            net_ok, net = check_robot_net_clearance(clearance[0], clearance[i])
            net_ok = net_ok and bool(clearance[0].isfinite().all())
            contacts_ok = bool(string[i].sum() >= string[0].sum())
            complete_ok = bool((complete[i] | ~complete[0]).all())
            trial.update(string_before=int(string[0].sum()), string_after=int(string[i].sum()),
                         string_ok=contacts_ok, robot_net_ok=net_ok, robot_net=net,
                         tail_coverage_ok=complete_ok, batched_guard_lanes=string.numel())
            if winner is None and contacts_ok and net_ok and complete_ok:
                winner = candidate, budget, trial
        if winner is None:
            self._native_gate_target_rms = max(2.5e-5, target_rms * .5)
            return False, 0., before_budget, trials
        candidate, budget, trial = winner
        trial['selected'] = True
        if trial['alpha'] < alpha:
            self._native_gate_target_rms = max(2.5e-5, target_rms * .5)
        self.actor.load_state_dict(candidate, strict=True)
        return True, trial['alpha'], budget, trials

    def imagine_update(self,initial_state,*,update_actor=True,guard_starts=None,stop_proposal=None):
        from .supermimic_approach_data_v1 import (
            approach_guard_merit, approach_step_sizes, comparable_flight_merit,
            gradient_parts, prioritize_flight_gradient)
        from .supermimic_approach_speed_risk import (
            compare_risk, constrain_proposal, forward_tolerances, proposal_risk_objectives, risk_statistics)
        fast = getattr(self, 'approach_v2', False) and update_actor
        native_gate = fast and getattr(self, 'approach_native_gate', False)
        protect_speed = fast and getattr(self, 'approach_speed_risk', False) and not native_gate
        if fast and not guard_starts:
            raise ValueError('Approach v2 requires a separate guard batch')
        if protect_speed and self.fixed_speed_risk_guard is not guard_starts:
            raise ValueError('Speed risk protection requires the fixed block guard starts')
        cfg=self.timebase; started=time.perf_counter()
        actor_before={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        optimizer_before=copy.deepcopy(self.actor_opt.state_dict())
        model_hash=tensor_digest(self.adapter.state_dict().items())
        parameters=[p for p in self.actor.parameters() if p.requires_grad]
        self.actor_opt.zero_grad(set_to_none=True)
        before_budget=self._budget_snapshot()
        budget_references=self._budget_references()
        committed_budget=before_budget
        cpu_rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        def restore_rng():
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        accepted=False; alpha=0.; gradient=0.; reason=None; grad_stats={}; trials=[]
        guard_baseline = None
        proposal_timing = dict(baseline_objective_wall_s=0., guard_baseline_wall_s=0., baseline_tracking_vjp_wall_s=0.,
                               baseline_pooled_vjp_wall_s=0., outer_gradient_wall_s=0.,
                               backtracking_wall_s=0.)
        if update_actor:self.actor_attempts+=1
        self._proposal_ball_targets = {}

        def score_guard():
            guard_started = time.perf_counter()
            with torch.no_grad():
                restore_rng()
                anchor = self.speed_risk_anchor if protect_speed else None
                baseline = self._rollout_objective(
                    guard_starts, effective_weights=anchor['effective_weights'] if anchor is not None else None,
                    collect_critic_data=False)
                if protect_speed and anchor is None:
                    tolerances = []
                    for _ in range(2):
                        restore_rng()
                        repeated = self._rollout_objective(guard_starts, effective_weights=baseline['effective_weights'],
                                                           collect_critic_data=False)
                        tolerances.append(forward_tolerances(baseline, repeated))
                    self.speed_risk_tolerances = {key: max(t[key] for t in tolerances) for key in tolerances[0]}
                    self.speed_risk_anchor = baseline
                    self.speed_risk_rejudge(self.speed_risk_tolerances, risk_statistics(baseline))
            proposal_timing['guard_baseline_wall_s'] = time.perf_counter() - guard_started
            return baseline

        try:
            with frozen(self.adapter,self.critic),torch.set_grad_enabled(update_actor):
                if protect_speed and self.speed_risk_anchor is None:
                    guard_baseline = score_guard()
                    restore_rng()
                objective_started = time.perf_counter()
                rollout=self._rollout_objective(initial_state)
                baseline_loss=float(rollout['loss'].detach())
                proposal_timing['baseline_objective_wall_s'] = time.perf_counter() - objective_started
                proposal_timing['baseline_tracking_vjp_wall_s'] = rollout.get('tracking_vjp_wall_s', 0.)
                proposal_timing['baseline_pooled_vjp_wall_s'] = rollout.get('pooled_vjp_wall_s', 0.)
                if update_actor and rollout['gate_lanes']:
                    if fast and guard_baseline is None and not native_gate:
                        guard_baseline = score_guard()
                    gradient_started = time.perf_counter()
                    diagnose = not fast or getattr(self, 'diagnose_ball_gradients', False)
                    grad_stats = (gradient_parts({k:rollout[k] for k in ('task','tracking','anchor')},parameters)
                                  if diagnose else {})
                    if getattr(self, 'diagnose_ball_gradients', False) and rollout.get('flight_losses'):
                        grad_stats.update(gradient_parts(
                            {f'ball_{name}': value for name, value in rollout['flight_losses'].items()}, parameters))
                    # Anchor-only momentum is not an effective task update.
                    tracking_coeff=self.tracking_coefficient(self.block_name)
                    if (getattr(self, 'approach_ball_protocol', None) is None
                            and getattr(self, 'landing_and_first_contact_objectives', False)
                            and getattr(self, 'contact_aux_weight', 0.) > 0 and self.block_name == 'approach'):
                        requested = self.contact_aux_weight
                        main = tracking_coeff * grad_stats['tracking_gradient_norm']
                        aux = grad_stats['task_gradient_norm']
                        self.contact_aux_weight = min(requested, .20 * main / max(aux, 1e-12))
                        rollout['loss'] = (tracking_coeff * rollout['tracking'] + self.anchor_coeff * rollout['anchor']
                                           + self.contact_aux_weight * rollout['task'])
                        baseline_loss = float(rollout['loss'].detach())
                        grad_stats.update(contact_aux_requested=requested,
                                          contact_aux_to_toi_gradient_ratio=self.contact_aux_weight*aux/max(main, 1e-12))
                    learnable = (rollout['loss'].requires_grad if not diagnose else
                                 grad_stats['task_gradient_norm']+tracking_coeff*grad_stats['tracking_gradient_norm']>1e-12)
                    if learnable:
                        flight = None if fast else rollout.get('flight_priority_loss')
                        flight_gradients = (torch.autograd.grad(flight, parameters, retain_graph=True, allow_unused=True)
                                            if flight is not None and flight.requires_grad else ())
                        constraint_gradients = {}
                        constraint_objectives = proposal_risk_objectives(rollout) if protect_speed else {}
                        if native_gate and rollout['robot_net_loss'] is not None and float(rollout['robot_net_loss'].detach()) > 0.:
                            constraint_objectives['robot_net'] = rollout['robot_net_loss']
                        for name, value in constraint_objectives.items():
                            if value.requires_grad:
                                derivatives = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
                                constraint_gradients[name] = {
                                    id(p): g.detach() for p, g in zip(parameters, derivatives, strict=True) if g is not None}
                        rollout['loss'].backward()
                        if flight_gradients:
                            grad_stats['flight_gradient_correction'] = prioritize_flight_gradient(
                                parameters, flight_gradients)
                        norm=torch.sqrt(sum((p.grad.detach().square().sum() for p in parameters if p.grad is not None),rollout['loss'].new_zeros(())))
                        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite actor gradient')
                        proposal_timing['outer_gradient_wall_s'] = time.perf_counter() - gradient_started
                        if float(norm)>1e-12:
                            # Preserve the gradient of the scored objective before optimizer_step clips it.
                            active_params = [(name, p) for name, p in self.actor.named_parameters()
                                             if p.requires_grad and p.grad is not None]
                            objective_grad = torch.cat([p.grad.detach().flatten() for _, p in active_params])
                            flight_grad = None
                            if flight_gradients:
                                by_id = dict(zip(map(id, parameters), flight_gradients, strict=True))
                                flight_grad = torch.cat([
                                    (g.detach() if (g := by_id[id(p)]) is not None else torch.zeros_like(p)).flatten()
                                    for _, p in active_params])
                                if float(flight_grad.square().sum()) <= 1e-20:
                                    flight_grad = None
                            gradient=float(optimizer_step(self.actor_opt,parameters,element_clip=.1))
                            def proposal_slope(reference):
                                displacement = torch.cat([(p.detach()-actor_before[name]).flatten()
                                                          for name,p in active_params])
                                return float(torch.dot(reference, displacement))
                            slope = proposal_slope(objective_grad)
                            grad_stats['adam_proposal_loss_slope'] = slope
                            if flight_grad is not None:
                                slope = proposal_slope(flight_grad)
                                grad_stats['adam_proposal_flight_slope'] = slope
                            if (getattr(self, 'restart_uphill_approach_momentum', False)
                                    and self.block_name == 'approach' and slope >= 0.):
                                self.actor.load_state_dict(actor_before, strict=True)
                                # Loading can alias a CPU optimizer snapshot. Keep the rollback copy intact.
                                self.actor_opt.load_state_dict(copy.deepcopy(optimizer_before))
                                for _, p in active_params:
                                    moment = self.actor_opt.state[p].get('exp_avg')
                                    if moment is not None:
                                        moment.zero_()
                                # The first trial has already clipped this gradient. Advance from the
                                # original step and second moment exactly once with the current direction.
                                self.actor_opt.step()
                                grad_stats['restarted_adam_proposal_loss_slope'] = proposal_slope(objective_grad)
                                if flight_grad is not None:
                                    grad_stats['restarted_adam_proposal_flight_slope'] = proposal_slope(flight_grad)
                            unprojected = None
                            if constraint_gradients:
                                gradients = {
                                    key: torch.cat([values.get(id(p), torch.zeros_like(p)).flatten() for _, p in active_params])
                                    for key, values in constraint_gradients.items()}
                                gradients['objective'] = objective_grad
                                displacement = torch.cat([(p.detach() - actor_before[name]).flatten()
                                                          for name, p in active_params])
                                corrected, diagnostics = constrain_proposal(displacement, gradients)
                                grad_stats['speed_risk_proposal_projection'] = diagnostics
                                if diagnostics['applied']:
                                    unprojected = {name: p.detach().clone() for name, p in active_params}
                                    with torch.no_grad():
                                        for (name, parameter), delta in zip(active_params, corrected.split(
                                                [p.numel() for _, p in active_params]), strict=True):
                                            parameter.copy_(actor_before[name] + delta.view_as(parameter))
                            proposed={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
                            proposed_opt=copy.deepcopy(self.actor_opt.state_dict())
                            if native_gate:
                                native_started = time.perf_counter()
                                endpoint = proposed
                                if unprojected is not None:
                                    endpoint = dict(proposed)
                                    for name, raw in unprojected.items():
                                        endpoint[name] = torch.lerp(raw, proposed[name], .25)
                                accepted, alpha, committed_budget, trials = self._native_gate_candidates(
                                    actor_before, endpoint, before_budget, budget_references, guard_starts, stop_proposal)
                                for trial in trials:
                                    trial['projection_fraction'] = .25 if unprojected is not None else 0.
                                proposal_timing['backtracking_wall_s'] = time.perf_counter() - native_started
                                if accepted:
                                    self.actor_opt.load_state_dict(proposed_opt)
                                    self.last_approach_alpha = alpha
                                else:
                                    reason = 'coarse_guard_rejected'
                            step_sizes = approach_step_sizes(getattr(self, 'last_approach_alpha', None) if fast else None)
                            line_search = [(a, 1.) for a in step_sizes]
                            if unprojected is not None:
                                # Bracket the raw and corrected directions at small
                                # trust steps before making a full correction.
                                local_steps = (.0625, .125, .25)
                                blends = (.25, .375, .125, .50)
                                line_search = [(a, blend) for a in local_steps for blend in blends] + line_search
                            if native_gate:
                                line_search = []
                            backtracking_started = time.perf_counter()
                            for index, (candidate_alpha, projection_fraction) in enumerate(line_search):
                                if stop_proposal is not None and stop_proposal():
                                    reason = 'proposal_interrupted'
                                    break
                                if unprojected is not None and projection_fraction != 1.:
                                    endpoint = dict(proposed)
                                    for name, raw in unprojected.items():
                                        endpoint[name] = torch.lerp(raw, proposed[name], projection_fraction)
                                else:
                                    endpoint = proposed
                                candidate={k:(v+(endpoint[k]-v)*candidate_alpha if k in self.trainable_names else v)
                                           for k,v in actor_before.items()}
                                self.actor.load_state_dict(candidate,strict=True)
                                with torch.no_grad():
                                    candidate_budget=self._budget_snapshot()
                                    report=self._relative_budget_report(candidate_budget,budget_references)
                                    local=report['relative_last_accepted']; native=report['relative_native_accepted']
                                    bounded=(local['final_normalized_command_delta']<=self.command_delta_budget and local['final_joint_target_delta_rad']<=self.target_delta_budget_rad)
                                    if self.native_reference_command_budget is not None:
                                        bounded &= (native['final_normalized_command_delta']<=self.native_reference_command_budget and native['final_joint_target_delta_rad']<=self.native_reference_target_budget_rad)
                                    # Measure final executable commands on identical real states/memory.
                                    effect=self._budget_deltas(candidate_budget,before_budget)['final_joint_target_delta_rad']
                                    if (not bounded and index != len(line_search) - 1
                                            and all(math.isfinite(x) for deltas in report.values() for x in deltas.values())):
                                        # The fixed-command budget has already ruled out this step.
                                        # Keep the final trial scored even if every step is rejected.
                                        trials.append(dict(alpha=candidate_alpha, projection_fraction=projection_fraction,
                                                           budget_ok=False, budget_report=report,
                                                           final_command_change=effect, loss=None,
                                                           scoring_skipped='command_or_target_budget'))
                                        continue
                                    restore_rng()
                                    if fast:
                                        evaluation = self._rollout_objective(
                                            guard_starts, effective_weights=guard_baseline['effective_weights'],
                                            collect_critic_data=False)
                                    else:
                                        evaluation=self._rollout_objective(initial_state,effective_weights=rollout['effective_weights'],
                                                                          baseline_toi_weights=rollout['toi_weights'],collect_critic_data=False,
                                                                          **({'baseline_ball_scored': rollout['ball_scored']}
                                                                             if rollout.get('ball_scored') is not None else {}),
                                                                          **({'baseline_landing_weights': rollout['landing_weights']}
                                                                             if rollout.get('landing_weights') is not None else {}))
                                    candidate_loss=float(evaluation['loss'])
                                    coverage_ok=evaluation['toi_coverage_preserved']
                                trial_terms = {name: float(evaluation[name]) for name in ('task','tracking','anchor')}
                                event_counts = {k: v for k, v in evaluation['reference_coverage'].items()
                                                if k in ('predicted_first_string_lanes','predicted_first_frame_lanes')}
                                trials.append(dict(alpha=candidate_alpha,projection_fraction=projection_fraction,
                                                   loss=candidate_loss,objective_terms=trial_terms,
                                                   first_contact_counts=event_counts,budget_ok=bool(bounded),
                                                   budget_report=report,toi_coverage_preserved=coverage_ok,final_command_change=effect))
                                before_speed = (guard_baseline if fast else rollout).get('ball_approach_speed')
                                after_speed = evaluation.get('ball_approach_speed')
                                if before_speed is not None and after_speed is not None:
                                    common = (before_speed.isfinite() & after_speed.isfinite())
                                    trials[-1]['predicted_normal_approach_speed'] = dict(
                                        matched=int(common.sum()), before_mps=float(before_speed[common].mean()) if common.any() else None,
                                        after_mps=float(after_speed[common].mean()) if common.any() else None)
                                merit_ok = candidate_loss <= baseline_loss
                                if fast:
                                    if protect_speed:
                                        tol = self.speed_risk_tolerances
                                        merit_ok, flight_report = approach_guard_merit(guard_baseline, evaluation, flight_tolerance=tol['flight'])
                                        relative_ok, relative = compare_risk(guard_baseline, evaluation, tol['risk'])
                                        anchor_ok, anchored = compare_risk(self.speed_risk_anchor, evaluation, tol['risk'])
                                        trials[-1]['speed_risk_guard'] = dict(relative=relative, anchored=anchored)
                                        merit_ok = merit_ok and relative_ok and anchor_ok
                                        if self.speed_risk_anchor.get('robot_net_clearance_by_lane') is not None:
                                            from .supermimic_robot_net_clearance import check_robot_net_clearance
                                            net_anchor_ok, net_anchor = check_robot_net_clearance(
                                                self.speed_risk_anchor['robot_net_clearance_by_lane'],
                                                evaluation['robot_net_clearance_by_lane'])
                                            flight_report['robot_net_block_anchor'] = net_anchor
                                            merit_ok = merit_ok and net_anchor_ok
                                    else:
                                        merit_ok, flight_report = approach_guard_merit(guard_baseline, evaluation)
                                    trials[-1]['baseline_release_flight'] = flight_report
                                    trials[-1]['flight_priority_merit_ok'] = merit_ok
                                    if evaluation['outcome_audit']:
                                        trials[-1]['guard_outcome_audit'] = evaluation['outcome_audit']
                                elif rollout.get('flight_cost_by_lane') is not None:
                                    merit_ok, flight_report = comparable_flight_merit(
                                        rollout['flight_cost_by_lane'], evaluation['flight_cost_by_lane'],
                                        baseline_loss, candidate_loss)
                                    trials[-1]['baseline_release_flight'] = flight_report
                                    trials[-1]['flight_priority_merit_ok'] = merit_ok
                                if bounded and coverage_ok and effect>0 and merit_ok:
                                    committed_budget=candidate_budget
                                    alpha=candidate_alpha;accepted=True;break
                            if not native_gate:
                                proposal_timing['backtracking_wall_s'] = time.perf_counter() - backtracking_started
                            if accepted and not native_gate:
                                if fast:
                                    self.last_approach_alpha = alpha
                                # Adam moments correspond to the original same-gradient proposal;
                                # alpha scales only the committed parameter displacement.
                                self.actor_opt.load_state_dict(proposed_opt)
                            elif not native_gate and reason != 'proposal_interrupted':
                                if protect_speed and any('speed_risk_guard' in trial for trial in trials):
                                    reason = 'no_candidate_passed_flight_and_grouped_speed_risk_guard'
                                else:
                                    reason = ('flight_priority_budget_or_zero_final_command_response'
                                              if rollout.get('flight_cost_by_lane') is not None
                                              else 'objective_budget_or_zero_final_command_response')
                        else:reason='zero_total_gradient'
                    else:
                        proposal_timing['outer_gradient_wall_s'] = time.perf_counter() - gradient_started
                        reason='no_task_or_tracking_gradient'
                elif update_actor:reason='no_effective_gate'
            critic_data=rollout['critic_data']
            ret_mean=float(rollout['ret'].detach().mean()); track_value=float(rollout['tracking'].detach())
            gate_steps=rollout['gate_steps'];gate_lanes=rollout['gate_lanes'];track_steps=rollout['tracking_steps']
            track_stats={k:rollout[k] for k in ('tracking_active_lane_steps','tracking_effective_weight_sum','tracking_nonzero_gradient_lanes')}
            if 'outgoing_impact_lanes' in rollout:
                track_stats['outgoing_impact_lanes']=rollout['outgoing_impact_lanes']
            track_stats.update({k: v for k, v in rollout.items() if k.startswith('ball_loss_')
                                or k in ('resolved_ball_lanes', 'demand_racket_target_lanes',
                                         'planned_imagination_steps', 'model_rollout_steps',
                                         'ball_speed_loss_supported_lanes')})
            if protect_speed:
                track_stats['proposal_speed_risk'] = risk_statistics(rollout)
                track_stats['same_forward_tolerance'] = getattr(self, 'speed_risk_tolerances', None)
            matched=rollout['matched_references'];coverage=rollout['reference_coverage']
            baseline_terms = {name: float(rollout[name].detach()) for name in ('task','tracking','anchor')}
            proposal_outcome_audit = rollout.get('outcome_audit', [])
            del rollout
        except BaseException:
            self.actor.load_state_dict(actor_before,strict=True);self.actor_opt.load_state_dict(optimizer_before)
            raise
        finally:
            self._proposal_ball_targets.clear()
        if model_hash!=tensor_digest(self.adapter.state_dict().items()):raise RuntimeError('Actor update changed model')
        if self.frozen_hash()!=self.frozen_actor_hash:raise RuntimeError('Frozen actor state changed')
        # Critic updates only AFTER all fixed-objective comparisons finish.
        bptt_seconds=time.perf_counter()-started;started=time.perf_counter()
        losses = []
        try:
            if critic_data is not None:
                states,rewards,dones,final,valid=critic_data
                with torch.no_grad():targets=gae_returns(self.critic(states),self.critic(final),rewards,dones,cfg.gamma,cfg.lam).clamp(max=self.v_max)
                x,y=states[valid],targets[valid]
                if not len(y):raise ValueError('No valid critic data')
                for _ in range(5):
                    for ids in torch.tensor_split(torch.randperm(len(y),device=y.device),4):
                        if not ids.numel():continue
                        self.critic_opt.zero_grad(set_to_none=True); loss=(self.critic(x[ids])-y[ids]).square().mean()
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite GAE critic loss')
                        loss.backward();optimizer_step(self.critic_opt,self.critic.parameters());self.critic_updates+=1;losses.append(float(loss.detach()))
        except BaseException:
            self.actor.load_state_dict(actor_before,strict=True);self.actor_opt.load_state_dict(optimizer_before)
            raise
        if update_actor:
            if accepted:
                self.actor_updates+=1; self.backtracked_actor_updates+=int(alpha<1);self.accept_snapshot()
            else:
                self.actor.load_state_dict(actor_before,strict=True);self.actor_opt.load_state_dict(optimizer_before)
                self.rejected_actor_updates+=1
        if losses:
            self.last_critic_loss=sum(losses)/len(losses)
        changes={k:float((p.detach()-actor_before[k]).abs().max()) for k,p in self.actor.named_parameters() if p.requires_grad}
        command_delta=self._budget_deltas(committed_budget,before_budget)
        if getattr(self, 'approach_ball_protocol', None) is not None:
            arm_deltas = [(after[2][:, 12:] - before[2][:, 12:]).abs().amax()
                          for after, before in zip(committed_budget, before_budget, strict=True)]
            track_stats['arm_final_joint_target_delta_rad'] = float(torch.stack(arm_deltas).amax())
        objective_before = baseline_loss
        objective_terms_before = baseline_terms
        objective_report = {}
        if fast:
            objective_report['proposal_objective_before'] = baseline_loss
            objective_report['proposal_terms_before'] = baseline_terms
            if proposal_outcome_audit:
                objective_report['proposal_outcome_audit'] = proposal_outcome_audit
        if guard_baseline is not None:
            objective_before = float(guard_baseline['loss'])
            objective_terms_before = {name: float(guard_baseline[name]) for name in ('task', 'tracking', 'anchor')}
            objective_report.update(
                guard_objective_before=objective_before,
                guard_objective_after=trials[-1]['loss'] if accepted else objective_before,
                guard_terms_before=objective_terms_before,
                guard_terms_after=trials[-1]['objective_terms'] if accepted else objective_terms_before)
            if guard_baseline['outcome_audit']:
                objective_report['guard_outcome_audit_before'] = guard_baseline['outcome_audit']
        return dict(**objective_report, reference_coverage=coverage,actor_updates=self.actor_updates,actor_attempts=self.actor_attempts,rejected_actor_updates=self.rejected_actor_updates,
            update_accepted=accepted if update_actor else False,block_name=self.block_name,trainable_names=sorted(self.trainable_names),
            block_attempts=self.block_attempts,backtracked_actor_updates=self.backtracked_actor_updates,backtrack_alpha=alpha,
            rejection_reason=reason,backtrack_trials=trials,objective_before=objective_before,
            objective_terms_before=objective_terms_before,
            objective_after=trials[-1]['loss'] if accepted and not native_gate else objective_before,
            adam_scaling='full proposal moments with alpha-scaled parameters',
            command_delta_budget=self.command_delta_budget,target_delta_budget_rad=self.target_delta_budget_rad,
            critic_updates=self.critic_updates,world_updates=self.world_updates,imagined_return_mean=ret_mean,
            actor_gradient=gradient,parameter_max_changes=changes,tracking_loss=track_value,
            tracking_weight=self.tracking_weight,toi_racket_weight=getattr(self,'toi_racket_weight',1.0),
            tracking_steps=track_steps,matched_references=matched,**grad_stats,**track_stats,
            final_command_max_change=command_delta['final_joint_target_delta_rad'],request_max_change=command_delta['residual_request_delta'],
            budget_report=self._relative_budget_report(committed_budget,budget_references),critic_loss=self.last_critic_loss,bptt_seconds=bptt_seconds,
            proposal_wall_breakdown=proposal_timing,
            critic_seconds=time.perf_counter()-started,model_runtime_state_sha256=model_hash,
            imagined_steps=track_stats.get('model_rollout_steps', track_steps) if critic_data is None else len(rewards),
            effective_gate_steps=gate_steps,effective_gate_lanes=gate_lanes,no_effective_gate=gate_lanes==0,
            no_effective_gradient=reason in ('zero_total_gradient','no_task_or_tracking_gradient'),
            frozen_base_recovery_unchanged=True,model_unchanged_during_actor=True)

    def fit_research_contact(self,window,launch):
        # Supervise the outgoing velocity only on measured racket event edges.
        # Free-flight transitions remain in the ordinary shared WM loss.
        change=window['next_event_state']>window['event_state']
        t,lane=(change[:,:,1]|change[:,:,6]).nonzero(as_tuple=True)
        if not len(t):
            return dict(measured_events=0,updated=False)
        model=self.adapter.action_ball_model
        idx=(t,lane)
        pred,_,_=ball_transition(self.adapter.physical_model,self.adapter.ball_model,window['state'][idx],
            window['targets'][idx],window['previous_action'][idx],window['feet'][idx].float(),
            window['present'][idx],window['elapsed'][idx],launch[lane],action_model=model)
        loss=torch.nn.functional.smooth_l1_loss(pred[:,52:55],window['next_state'][idx][:,52:55],beta=.1)
        params=list(model.impulse_residual_head.parameters())
        self.world_opt.zero_grad(set_to_none=True)
        grads=torch.autograd.grad(loss,params,allow_unused=True)
        for p,g in zip(params,grads):
            p.grad=g
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite event-only contact supervision')
        if norm>0:
            self.world_opt.step()
        return dict(measured_events=len(t),updated=bool(norm>0),velocity_huber=float(loss.detach()),gradient_norm=float(norm))

    def _measured_contact_terms(self, window, launch):
        """Supervise contact-to-release with measured robot motion in every tick."""
        columns = {}
        offset = 0
        for field in self.adapter.task_model.memory_schema.fields:
            columns[field.path] = slice(offset, offset + field.width)
            offset += field.width
        had = columns['rally/had_contact']
        released = columns['rally/released']
        outgoing = columns['rally/outgoing_velocity']
        before, after = window['values'], window['next_values']
        touch = (before[..., had].squeeze(-1) <= .5) & (after[..., had].squeeze(-1) > .5)
        release = (before[..., released].squeeze(-1) <= .5) & (after[..., released].squeeze(-1) > .5)
        release &= window['next_valid'][..., outgoing].bool().all(-1)
        lanes = (touch.any(0) & release.any(0)).nonzero().flatten()
        if not len(lanes):
            return None
        begin = touch[:, lanes].int().argmax(0)
        end = release[:, lanes].int().argmax(0)
        if (begin > end).any():
            raise ValueError('Contact supervision windows contain reversed contact and release')
        ball = window['state'][begin, lanes, 49:]
        model = self.adapter.action_ball_model
        trajectory, position, velocity = [], [], []
        for tick in range(int(begin.min()), int(end.max()) + 1):
            participating = (begin <= tick) & (end >= tick)
            j = participating.nonzero().flatten()
            if not len(j):
                continue
            native = lanes[j]
            state = torch.cat((window['state'][tick, native, :49], ball[j]), dim=-1)
            prediction, _, _ = ball_transition(
                self.adapter.physical_model, self.adapter.ball_model, state, window['targets'][tick, native],
                window['previous_action'][tick, native], window['feet'][tick, native].float(),
                window['present'][tick, native], window['elapsed'][tick, native], launch[native],
                action_model=model, native_robot_endpoint=window['next_state'][tick, native, :49])
            next_ball = prediction[:, 49:]
            ball = ball.index_copy(0, j, next_ball)
            target = window['next_state'][tick, native, 49:]
            trajectory.append(torch.nn.functional.smooth_l1_loss(next_ball, target, beta=.1))
            terminal = end[j] == tick
            if terminal.any():
                position.append(torch.nn.functional.smooth_l1_loss(next_ball[terminal, :3], target[terminal, :3], beta=.02))
                velocity.append(torch.nn.functional.smooth_l1_loss(
                    next_ball[terminal, 3:], after[tick, native[terminal], outgoing], beta=.1))
        parts = dict(trajectory=torch.stack(trajectory).mean(), release_position=torch.stack(position).mean(),
                     release_velocity=torch.stack(velocity).mean())
        return parts, lanes, ball[:, 3:], after[end, lanes, outgoing]

    def _step_measured_contact(self, parts, measured_events):
        params = list(self.adapter.action_ball_model.impulse_residual_head.parameters())
        loss = sum(parts.values())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite native-motion contact supervision')
        self.world_opt.zero_grad(set_to_none=True)
        grads = (torch.autograd.grad(loss, params, allow_unused=True) if loss.requires_grad
                 else [None] * len(params))
        for parameter, grad in zip(params, grads, strict=True):
            parameter.grad = grad
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if norm > 0:
            self.world_opt.step()
        return dict(measured_events=measured_events, updated=bool(norm > 0), gradient_norm=float(norm),
                    **{name: float(value.detach()) for name, value in parts.items()})

    def fit_ball_only_contact(self, window, launch):
        result = self._measured_contact_terms(window, launch)
        if result is None:
            return dict(measured_events=0, updated=False)
        parts, lanes, _, _ = result
        return self._step_measured_contact(parts, len(lanes))

    def fit_event_aligned_approach_body(self, window, launch):
        from .supermimic_precontact_body import REVISION_POSITION_WEIGHT, body_rollout_loss

        loss, (position, speed) = body_rollout_loss(
            self.adapter.physical_model, window, launch, position_weight=REVISION_POSITION_WEIGHT)
        norm = self._step_approach_body(loss)
        return dict(updated=norm > 0, loss=float(loss.detach()), gradient=norm,
                    racket_center_error_m=float(position), contact_point_speed_error_mps=float(speed))

    def _step_approach_body(self, loss):
        params = tuple(self.adapter.physical_model.body.parameters())
        self.world_opt.zero_grad(set_to_none=True)
        for param, grad in zip(params, torch.autograd.grad(loss, params), strict=True):
            param.grad = grad
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        if norm > 0:
            self.world_opt.step()
            self.world_updates += 1
        return norm

    def fit_approach_body_pair(self, minus, plus, launch):
        """Learn signed, pre-contact racket responses on authenticated intervention pairs."""
        from .supermimic_precontact_body import body_pair_loss
        from .supermimic_release_calibration import _native_event_columns

        columns = _native_event_columns(self.adapter.task_model.memory_schema)
        loss, support, diagnostics = body_pair_loss(self.adapter.physical_model, minus, plus, launch, columns)
        if loss is None:
            return dict(updated=False, supported_horizons=support)
        norm = self._step_approach_body(loss)
        return dict(updated=bool(norm > 0), supported_horizons=support, response_by_horizon=diagnostics,
                    body_pair_loss=float(loss.detach()), body_pair_gradient=norm)

    def fit_ball_only_contact_pair(self, minus, plus, launch):
        low = self._measured_contact_terms(minus, launch)
        high = self._measured_contact_terms(plus, launch)
        if low is None or high is None:
            return dict(measured_events=0, updated=False)
        low_parts, low_lanes, low_pred, low_truth = low
        high_parts, high_lanes, high_pred, high_truth = high
        left, right = (low_lanes[:, None] == high_lanes).nonzero(as_tuple=True)
        if not len(left):
            return dict(measured_events=0, updated=False)
        response = high_truth[right] - low_truth[left]
        predicted = high_pred[right] - low_pred[left]
        parts = {name: .5 * (value + high_parts[name]) for name, value in low_parts.items()}
        parts['paired_velocity_response'] = torch.nn.functional.smooth_l1_loss(
            predicted, response, beta=.1)
        result = self._step_measured_contact(parts, len(left))
        # Record weak responses separately. They still contribute to the loss.
        result['measurable_response_pairs'] = int((response.norm(dim=-1) >= .01).sum())
        return result

    def fit_world_window(self,window,launch,*,loss_groups=None,separate_supervision=False,long_native=None):
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
        row_loss_sum=state.new_zeros(len(state))
        action_ball_loss=state.new_zeros(())
        racket_loss_value=state.new_zeros(())
        racket_components={name: state.new_zeros(()) for name in
                           ('position','normal','velocity','angular')}
        for i in range(cfg.world_steps):
            prediction,newfeet,newpresent=supervised_transition(self.adapter,state,window['targets'][i],
                window['previous_action'][i],feet,present,window['elapsed'][i],launch)
            out=task.forward_values(state,prediction,window['targets'][i],window['previous_action'][i],feet,
                window['elapsed'][i],values,validity)
            if separate_supervision:
                truth=window['next_state'][i]
                robot_rows=robot_loss(prediction,truth,per_row=True)
                gyro_rows=torch.nn.functional.smooth_l1_loss(prediction[:,10:13],truth[:,10:13],beta=.1,reduction='none').mean(-1)
                physical_rows=robot_rows+gyro_rows
            else:
                physical_rows=world.state_loss(prediction,window['next_state'][i],per_row=True)
            contact_rows=torch.nn.functional.binary_cross_entropy(newfeet,window['next_feet'][i].float(),reduction='none').mean(-1)
            physical_i,contact_i=physical_rows.mean(),contact_rows.mean()
            ball_i=ball_error(prediction[:,49:],window['next_state'][i][:,49:])
            racket_pred=racket_state(prediction,world.geometry)
            racket_target=racket_state(window['next_state'][i],world.geometry)
            racket_rows=racket_loss_components(racket_pred,racket_target,per_row=True)
            racket_parts={k:v.mean() for k,v in racket_rows.items()}
            racket_each=sum(racket_rows.values())/len(racket_rows)
            racket_i=racket_each.mean()
            task_rows=task.supervision_loss_values(out,window['next_values'][i],window['next_valid'][i],
                window['event_state'][i],window['next_event_state'][i],window['torque'][i],
                window['terminated'][i],window['complete_reward'][i],per_row=True)
            task_i=task_rows.mean()
            physical_loss=physical_loss+physical_i;contact_loss=contact_loss+contact_i
            ball_loss_value=ball_loss_value+ball_i;task_loss=task_loss+task_i
            racket_loss_value=racket_loss_value+racket_i
            for name, value in racket_parts.items():
                racket_components[name]=racket_components[name]+value
            # Ball state is a sparse, high-amplitude target.  Keep it as an
            # explicit WM objective even for the legacy ball path.  Contact
            # weighting is driven only by current/next recorded event deltas,
            # never by future rollout labels.
            event=(window['next_event_state'][i,:,1]>window['event_state'][i,:,1]) | (
                window['next_event_state'][i,:,6]>window['event_state'][i,:,6])
            scale=prediction.new_tensor([.1,.1,.1,2.,2.,2.])
            error=((prediction[:,49:]-window['next_state'][i][:,49:])/scale).square().mean(-1)
            weight=1.+(self.contact_ball_loss_weight-1.)*event.to(error.dtype)
            extra=(error*weight).mean()*self.ball_loss_weight
            if loss_groups is not None:
                row_loss_sum += (physical_rows.detach()+contact_rows.detach()+task_rows.detach()
                                 +error.detach()*weight*self.ball_loss_weight
                                 +racket_each.detach()*self.racket_loss_weight)
            racket_extra=racket_i*self.racket_loss_weight
            action_ball_loss=action_ball_loss+extra
            loss=loss+physical_i+contact_i+task_i+extra+racket_extra
            state,feet,present=prediction,newfeet,newpresent
            values,validity=task.memory_schema.feedback_values(out['numeric'],out['valid_logits'].sigmoid())
        loss=loss/cfg.world_steps
        long_flight=state.new_zeros(())
        long_body=state.new_zeros(())
        direct_body=getattr(self,'direct_long_body_supervision',False)
        if getattr(self.adapter,'checkpoint_supervised_rollout',False) and long_native is None:
            direct_body=False
        if direct_body and (not separate_supervision or long_native is None):
            raise ValueError('Direct multistep body supervision requires separate losses and long native windows')
        if direct_body:
            long_flight,long_body=native_flight_rollout_loss(self.adapter,*long_native,loss_kind=self.natural_flight_loss,supervise_body=True)
            loss=loss+long_flight+long_body
        elif long_native is not None:
            long_flight=native_flight_rollout_loss(self.adapter,*long_native,loss_kind=self.natural_flight_loss)
            loss=loss+long_flight
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite physical multistep model loss')
        self.world_opt.zero_grad(set_to_none=True)
        if separate_supervision:
            # Each module receives its own measured targets. In particular a sharp
            # shuttle collision or a task label cannot train the robot trunk to
            # change the racket pose simply to fit an outgoing ball velocity.
            ball_modules=[ball]+([self.adapter.action_ball_model] if self.adapter.action_ball_model is not None else [])
            body_objective=physical_loss+contact_loss+self.racket_loss_weight*racket_loss_value
            # In direct mode the long robot objective uses actual robot/FK
            # measurements. Only the ball parameters fit long flight error.
            long_robot_objective=long_body if direct_body or getattr(self, 'approach_ball_protocol', None) else long_flight
            objectives=[(body_objective+cfg.world_steps*long_robot_objective,[world]),
                        (action_ball_loss+cfg.world_steps*long_flight,ball_modules),(task_loss,[task])]
            active=[(value,[p for module in modules for p in module.parameters() if p.requires_grad])
                    for value,modules in objectives]
            active=[(value,params) for value,params in active if params and value.requires_grad]
            if not active:
                raise ValueError('No trainable supervised world component')
            for j,(value,params) in enumerate(active):
                (value/cfg.world_steps).backward(inputs=params,retain_graph=j+1<len(active))
        else:
            loss.backward()
        physical_loss=physical_loss/cfg.world_steps;contact_loss=contact_loss/cfg.world_steps
        ball_loss_value=ball_loss_value/cfg.world_steps;task_loss=task_loss/cfg.world_steps
        action_ball_loss=action_ball_loss/cfg.world_steps
        racket_loss_value=racket_loss_value/cfg.world_steps
        racket_components={name: value/cfg.world_steps
                           for name,value in racket_components.items()}
        groups=(self.adapter.physical_model,self.adapter.ball_model,self.adapter.task_model)
        group_gradients=[]
        for group in groups:
            group_gradients.append(float(torch.sqrt(sum((p.grad.detach().square().sum() for p in group.parameters()
                                                        if p.grad is not None), loss.new_zeros(())))))
        action_ball_gradient=0.
        if self.adapter.action_ball_model is not None:
            action_ball_gradient=float(torch.sqrt(sum((p.grad.detach().square().sum()
                for p in self.adapter.action_ball_model.parameters() if p.grad is not None),loss.new_zeros(()))))
        norm=optimizer_step(self.world_opt,self.adapter.parameters());self.world_updates+=1
        task.fit_updates.add_(1)
        diagnostics = {}
        if loss_groups is not None:
            from .supermimic_training_audit_v1 import grouped_world_loss
            diagnostics['loss_by_group'] = grouped_world_loss(row_loss_sum.detach().cpu(),cfg.world_steps,loss_groups)
        return dict(**diagnostics,long_native_body_loss=float(long_body.detach()) if direct_body else None,long_native_flight_loss=float(long_flight.detach()) if long_native is not None else None,world_supervision='separate' if separate_supervision else 'joint',world_loss=float(loss.detach()),world_gradient=float(norm),world_updates=self.world_updates,
                    world_physical_loss=float(physical_loss.detach()),
                    world_contact_loss=float(contact_loss.detach()),
                    world_ball_loss=float(ball_loss_value.detach()),
                    world_task_loss=float(task_loss.detach()),
                    world_racket_loss=float(racket_loss_value.detach()),
                    world_racket_loss_weight=self.racket_loss_weight,
                    world_racket_position_loss=float(racket_components['position'].detach()),
                    world_racket_normal_loss=float(racket_components['normal'].detach()),
                    world_racket_velocity_loss=float(racket_components['velocity'].detach()),
                    world_racket_angular_loss=float(racket_components['angular'].detach()),
                    world_action_ball_loss=float(action_ball_loss.detach()),world_action_ball_gradient=action_ball_gradient,
                    world_physical_gradient=group_gradients[0],
                    world_ball_gradient=group_gradients[1],
                    world_task_gradient=group_gradients[2],
                    task_feedback_contract=task.memory_schema.feedback_contract,
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
        return self.calibrate_critic_tensors(x,y,epochs=epochs)

    def calibrate_critic_tensors(self,x,y,*,epochs=1):
        parameter=next(self.critic.parameters())
        x=x.detach().to(device=parameter.device,dtype=parameter.dtype);y=y.detach().reshape(-1,1).to(device=parameter.device,dtype=parameter.dtype)
        if len(x)!=len(y) or not len(y):raise ValueError('Invalid native critic tensor labels')
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
            complete=meta.get('complete_lane_mask',torch.ones(value.numel(),dtype=torch.bool,device=value.device)).to(value.device).bool()
            if complete.numel()!=value.numel():raise ValueError('Complete RTG mask shape mismatch')
            dropped=int((~complete).sum());value=value[complete]
            rows.append(dict(excluded_incomplete_lanes=dropped,policy_id=str(meta['policy_id']), episode_id=str(meta['episode_id']),
                             time_s=float(meta['time_s']), count=int(value.numel()),
                             return_min=float(value.min()) if len(value) else None, return_max=float(value.max()) if len(value) else None,
                             return_mean=float(value.mean()) if len(value) else None))
        return dict(policy_id=sorted({r['policy_id'] for r in rows}), stages=rows,
                    total_labels=sum(r['count'] for r in rows), source='completed_native_archive',
                    critic_updates=0)
