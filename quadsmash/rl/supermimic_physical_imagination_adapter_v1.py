"""One physical imagination path; never falls back to obs+delta dynamics.

The task predictor supplies explicitly supervised soft task state and event
hazards. Hard native scoring stays in Isaac. A fitted task predictor and named
native safety-body mapping are mandatory, not silently zero-filled inputs.
"""
import torch
from torch import nn

from .supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1,finite_value_and_validity
from .supermimic_physical_dataset_v1 import decode_physical
from .supermimic_physical_observation_v1 import PhysicalPolicyInputs,PhysicalPolicyObservationV1
from .supermimic_ball_sequence_v1 import ball_transition
from .supermimic_physical_reward_v1 import FAULTS
from .supermimic_execution_preview_v1 import MixerMemory
from quadsmash.tasks.stand.observation_contract import inverse_rotate


def matrix_quaternion(matrix):
    """wxyz with a discrete largest-diagonal branch, continuous within it."""
    m=matrix
    a,b,c=m[...,0,0],m[...,1,1],m[...,2,2]
    square=torch.stack((1+a+b+c,1+a-b-c,1-a+b-c,1-a-b+c),-1).clamp_min(1e-12)
    root=square.sqrt()
    candidates=torch.stack((
        torch.stack((square[...,0],m[...,2,1]-m[...,1,2],m[...,0,2]-m[...,2,0],m[...,1,0]-m[...,0,1]),-1),
        torch.stack((m[...,2,1]-m[...,1,2],square[...,1],m[...,0,1]+m[...,1,0],m[...,0,2]+m[...,2,0]),-1),
        torch.stack((m[...,0,2]-m[...,2,0],m[...,0,1]+m[...,1,0],square[...,2],m[...,1,2]+m[...,2,1]),-1),
        torch.stack((m[...,1,0]-m[...,0,1],m[...,0,2]+m[...,2,0],m[...,1,2]+m[...,2,1],square[...,3]),-1)), -2)
    candidates=candidates/(2*root[...,None])
    indices=root.argmax(-1)
    result=candidates[torch.arange(len(m),device=m.device),indices]
    return torch.nn.functional.normalize(result,dim=-1)


class PhysicalImaginationAdapter(nn.Module):
    schema='g07-physical-task-observation-loop-v1'

    def __init__(self,physical_model,ball_model,task_model,executor,*,safety_body_names,manifest,dt=.01,action_ball_model=None,
                 contact_bptt_mode='enabled'):
        super().__init__()
        if physical_model.schema!='physical55-separated-contact-transitions-v2':
            raise ValueError('PhysicalWorldV2 required; legacy observation predictor is forbidden')
        if not safety_body_names:raise ValueError('Native safety-body name mapping required')
        self.physical_model=physical_model
        self.ball_model=ball_model
        self.action_ball_model=action_ball_model
        from .supermimic_training_stage_v1 import validate_contact_bptt_mode
        self.contact_bptt_mode=validate_contact_bptt_mode(contact_bptt_mode)
        self.task_model=task_model
        self.executor=executor
        self.observer=PhysicalPolicyObservationV1().to(dtype=physical_model.input_mean.dtype)
        self.dt=float(dt)
        self.safety_ids=tuple(physical_model.geometry.ids[n] for n in safety_body_names)
        self.manifest=dict(manifest,adapter_schema=self.schema,
            physical_class=type(physical_model).__name__,physical_schema=physical_model.schema,
            ball_class=type(ball_model).__name__,task_class=type(task_model).__name__,
            event_semantics='supervised soft hazards; not native contacts',
            discrete_approximations=['task timestamp validity > 0.5','observation phase/contact flags > 0.5',
                                     'historical NumPy planning/requests stop-gradient'])
        self.manifest['contact_bptt_mode']=self.contact_bptt_mode
        if action_ball_model is not None:
            self.manifest['action_ball_schema']=action_ball_model.schema
            self.manifest['action_ball_native_certified']=False
        self.calls=0

    def set_contact_bptt_mode(self, mode):
        from .supermimic_training_stage_v1 import validate_contact_bptt_mode
        self.contact_bptt_mode=validate_contact_bptt_mode(mode)
        self.manifest['contact_bptt_mode']=self.contact_bptt_mode

    def forward(self,state:PhysicalImaginationStateV1,actor_action,*,detach_ball=False):
        if int(self.task_model.fit_updates)==0:
            raise RuntimeError('Physical loop requires fitted task memory/event model')
        c=state.clock_goal_context
        preview,executor,base=self.executor.forward(state.observation,actor_action,
            c['joint_default'][0],c['limits'],state.executor_memory)
        predicted,feet,present,ball_details=ball_transition(self.physical_model,self.ball_model,
            state.physical,preview.final_joint_targets,c['final_action'],c['foot_contact'],
            c['ball_present'],c['elapsed'],c['launch'],action_model=self.action_ball_model,
            return_details=True)
        # Keep robot dynamics differentiable while controlling when the
        # unverified ball/contact Jacobian can reach the actor.  This only
        # affects imagination gradients.  World-model supervision continues to
        # use the complete native ball target in fit_world_window().
        outgoing_ball=predicted[:,49:]
        if detach_ball or self.contact_bptt_mode in ('blocked', 'detached', 'precontact'):
            force_detach=detach_ball
            detach_ball = torch.ones(len(predicted), dtype=torch.bool,
                                     device=predicted.device)
            if self.contact_bptt_mode == 'precontact' and not force_detach:
                rally = state.task_memory.get('rally', {})
                had_contact = rally.get('had_contact')
                if had_contact is not None:
                    detach_ball = had_contact.reshape(-1) > .5
                action_info=ball_details.get('action_model_info') or {}
                impact_probability=action_info.get('predicted_impact_probability')
                if impact_probability is not None:
                    # The current control step can create the first contact
                    # even while the task latch is still false.  Precontact
                    # mode must stop the unvalidated impulse gradient at that
                    # causal boundary as well.
                    detach_ball = detach_ball | (impact_probability.reshape(-1) >= .5)
                impulse_info=ball_details.get('impulse_info') or {}
                if impulse_info.get('active') is not None:
                    detach_ball = detach_ball | impulse_info['active'].reshape(-1)
            predicted = torch.cat((predicted[:, :49],
                                   torch.where(detach_ball[:, None],
                                               predicted[:, 49:].detach(),
                                               predicted[:, 49:])), dim=1)
        task=self.task_model(state.physical,predicted,preview.final_joint_targets,
                             c['final_action'],c['foot_contact'],c['elapsed'],state.task_memory)
        next_task=task['memory'];p=decode_physical(predicted)
        elapsed=c['elapsed']+self.dt
        # Deterministic command history is advanced by execution, never by a
        # task-network guess. Clock increments retain the measured offset.
        env_memory=next_task['auxiliary']['environment']
        old_env=state.task_memory['auxiliary']['environment']
        env_memory['last_action']=preview.final_normalized
        env_memory['action_delta']=(preview.final_normalized-c['final_action']).square().sum(1)
        for key in ('jump_clock','ball_clock'):
            env_memory[key]=old_env[key]+self.dt
        timing=next_task['auxiliary']['timing']
        old_timing=state.task_memory['auxiliary']['timing']
        for key in timing:
            if key!='_jump_metric_stride':timing[key]=old_timing[key]+self.physical_model.flight.steps
        launched=(~c['ball_present'].bool())&present.bool()
        env_memory['launch_done']=present
        env_memory['launch_frame']=torch.where(launched,
            torch.full_like(old_env['launch_frame'],old_timing['_sim_step_counter']),old_env['launch_frame'])
        env_memory['launch_signals_ready']=torch.where(launched,torch.ones_like(present),
                                                     old_env['launch_signals_ready'].bool())
        rec=next_task['recovery'];rally=next_task['rally']
        inputs=PhysicalPolicyInputs(**p,joint_default=c['joint_default'],foot_contact=feet>.5,
            ball_present=present,airborne=rec['observed_airborne']>.5,
            airborne_seen=rec['observed_airborne_seen']>.5,hit_seen=rec['contact_seen']>.5,
            **{k:v for k,v in c.items() if k.startswith('goal_')},
            elapsed_s=elapsed,final_action=preview.final_normalized)
        observation,history=self.observer(inputs,state.observation_history)
        next_context=dict(c,elapsed=elapsed,ball_present=present,foot_contact=feet,
                          final_action=preview.final_normalized,history_tick=c['history_tick']+1)
        geometry=self.physical_model.geometry(p['joint_position'],p['joint_velocity'],
            p['root_position_w'],p['root_quaternion_w'],p['root_link_velocity_w'],p['root_angular_velocity_w'])
        racket_id=self.physical_model.geometry.ids['Racket']
        tool_quaternion=matrix_quaternion(geometry['rotation'][:,racket_id])
        sign=(tool_quaternion*old_env['previous_tool_quaternion']).sum(1,keepdim=True)>=0
        env_memory['previous_tool_quaternion']=torch.where(sign,tool_quaternion,-tool_quaternion)
        env_memory['previous_tool_position']=geometry['position'][:,racket_id]
        env_memory['previous_ball_position']=p['ball_position_w']
        env_memory['previous_ball_velocity']=p['ball_velocity_w']
        # Geometric continuous quantities derive from the same physical state;
        # the learned torque proxy has native applied-torque supervision.
        up=torch.zeros_like(p['root_position_w']);up[:,2]=-1.
        gravity=inverse_rotate(p['root_quaternion_w'],up)
        root_id=self.physical_model.geometry.root_id
        local_com=self.physical_model.geometry.local_com[root_id]
        offset=torch.einsum('bij,j->bi',geometry['rotation'][:,root_id],local_com)
        root_com_velocity=p['root_link_velocity_w']+torch.linalg.cross(p['root_angular_velocity_w'],offset)
        contact=(rally['legal_initial_contact']*rally['string_hit']*rally['airborne_string_hit']).clamp(0.,1.)
        recover=contact*rec['rally_recovery_success'].clamp(0.,1.)
        gap=geometry['position'][:,self.safety_ids,2].min(1).values.clamp_min(0.)
        risk=torch.exp(-gap/.04)
        quality=recover*torch.exp(-4.*gravity[:,:2].square().sum(1)
            -.4*root_com_velocity[:,:2].square().sum(1)-.2*p['root_angular_velocity_w'].square().sum(1))
        tool=geometry['position'][:,self.physical_model.geometry.ids['Racket']]
        near=torch.exp(-((p['ball_position_w']-tool).norm(dim=1)/.18).square())
        approach=near*rally['in_play'].clamp(0.,1.)*(1-rally['had_contact'].clamp(0.,1.))*(1-rec['body_ground_seen'].clamp(0.,1.))
        change=(preview.final_normalized-c['final_action']).square().sum(1)
        continuous=(.5*quality-2*risk-.05*change-.02*task['torque'].square().sum(1)/1000.+.8*approach)*self.dt
        event_delta=next_task['paid_events']-state.task_memory['paid_events']
        weights=event_delta.new_tensor([2.,20.,5.,15.,10.,-40.,-4.,-15.])
        at_end=elapsed>=8.-1e-7
        terminated=task['terminal_logit'].sigmoid().flatten()
        # Terminal complete reward is NOT another recurrent event hazard.
        terminal_complete=at_end.float()*task['complete_logit'].sigmoid().flatten()*(1-terminated)
        reward=continuous+(event_delta*weights).sum(1)+20*terminal_complete
        reward=reward-5.*(1-contact)*(elapsed>3.).float()*self.dt
        done=torch.where(at_end,torch.ones_like(terminated),terminated)
        if not torch.isfinite(reward).all():raise FloatingPointError('Nonfinite physical imagined reward')
        self.calls+=1
        next_state=PhysicalImaginationStateV1(predicted,executor,next_task,history,observation,next_context)
        return next_state,reward[:,None],done[:,None],dict(
            outgoing_ball=outgoing_ball,impulse_info=ball_details.get('impulse_info'),
            requested_action=actor_action,base_action=base,
            final_joint_targets=preview.final_joint_targets,event_delta=event_delta,
            task_prediction=task,continuous_reward=continuous,
            terminal_complete_proxy=terminal_complete,model_call=self.calls)

    def critic_input(self,state):
        numeric,valid=self.task_model.memory_schema.encode(state.task_memory)
        normalized=(numeric-self.task_model.memory_mean)/self.task_model.memory_scale
        c=state.clock_goal_context
        mixer=state.executor_memory.mixer
        memory=MixerMemory(mixer.arm_rate.target,mixer.arm_rate.clock,
                           mixer.arm_rate.initialized,mixer.nonzero_seen)
        reference=state.executor_memory.reference_controller
        # Full execution graph remains in the state. This explicit critic
        # context contains the causal plan and slew memory, not future labels.
        execution=torch.cat((reference.parameters,reference.event_time[:,None],
            reference.target_offset_world,reference.planned.float()[:,None],
            memory.arm_target,memory.arm_clock[:,None],memory.arm_initialized.float()[:,None],
            memory.nonzero_seen.float()[:,None]),-1)
        teacher=reference.teacher
        blend=teacher.leg_recovery_blend
        execution=torch.cat((execution,teacher.walk.last_raw,teacher.walk.reached.float()[:,None],
            torch.full_like(c['elapsed'][:,None],float(teacher.walk.tick%2)),
            teacher.standing.last_raw,teacher.standing.standing_ticks.float()[:,None],
            blend.started.float()[:,None],blend.start_time[:,None],blend.start_leg,
            teacher.arm_target_rate.target,teacher.arm_target_rate.clock[:,None],
            teacher.arm_target_rate.initialized.float()[:,None]),-1)
        # Independent wrapper cursors can lag the common reference log. They
        # affect future commands even if the root plan tensors are identical.
        node=state.executor_memory.base;seen=set();plan_context=[]
        while id(node) not in seen:
            seen.add(id(node))
            if hasattr(node,'valid_plan'):
                plan_context.append(node.valid_plan.float()[:,None])
            if hasattr(node,'plan_cursor'):
                plan_context.append(torch.full_like(c['elapsed'][:,None],float(node.plan_cursor)))
            if not hasattr(node,'base'):break
            node=node.base
        if plan_context:execution=torch.cat((execution,*plan_context),-1)
        return torch.cat((state.observation,normalized*valid,valid,state.task_memory['paid_events'],
                          c['elapsed'][:,None],finite_value_and_validity(execution)),-1)
