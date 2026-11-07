"""Measured-state sequence replanning with immutable local search bindings.

Every call starts from the supplied observation and actual prior command.
Only independently verified plans are returned. A rejected plan has no action;
the caller must report failure instead of executing an unverified search iterate.
CPU/model validation does not establish native recovery or real-time control.
"""
import copy
import time
import types
import numpy as np
import scripts.landing_sequential_recovery_search_v1 as original
from scripts.terminal_recovery_constraints_v1 import make_proposal
from scripts.audit_microstep_recovery_v1 import characterize
from scripts.audit_ramped_gravity_continuation_v1 import future_equilibrium_reference,rate_admitted_seed


def local_search():
    namespace=dict(vars(original))
    namespace['propose_sequence']=make_proposal(2.)
    def weighted_rollout(*args):
        result=original.rollout_for_search(*args);H=len(result['profile'])
        result['cost']+=sum((10. if i==H-1 else 1./H)*np.sum(n['following'].joint_velocity**2)
            for i,n in enumerate(result['profile']))
        return result
    namespace['rollout_for_search']=weighted_rollout
    functions=[]
    for name in ['sequential_recovery_search','restore_feasibility']:
        source=getattr(original,name)
        function=types.FunctionType(source.__code__,namespace,name,source.__defaults__,source.__closure__)
        function.__kwdefaults__=copy.deepcopy(source.__kwdefaults__)
        namespace[name]=function;functions.append(function)
    return tuple(functions)


def shifted_seed(commands,executed_steps,horizon,previous,interface,dt):
    """Carry commands, never a predicted state, across observations."""
    commands=np.asarray(commands,dtype=float);previous=np.asarray(previous,dtype=float)
    if (commands.ndim!=2 or commands.shape[1]!=18 or not np.isfinite(commands).all()
            or previous.shape!=(18,) or not np.isfinite(previous).all()
            or type(executed_steps) is not int or not 0<=executed_steps<=len(commands)
            or type(horizon) is not int or not 9<=horizon<=40):
        raise ValueError('Finite commands and declared actual executed step count required')
    remaining=commands[executed_steps:]
    tail=remaining[-1] if len(remaining) else previous
    seed=np.vstack([remaining,np.repeat(tail[None],horizon,axis=0)])[:horizon].copy()
    seed=np.clip(seed,interface.limits[:,0],interface.limits[:,1])
    for row in seed:
        row[12:]=np.clip(row[12:],previous[12:]-interface.arm_rate*dt,previous[12:]+interface.arm_rate*dt)
        previous=row.copy()
    return seed


class RecedingRecoverySequenceV1:
    def __init__(self,controller,*,iterations=4,time_limit_s=120.):
        if type(iterations) is not int or not 1<=iterations<=12 or not 0<time_limit_s<=600:
            raise ValueError('Finite guarded search required')
        if getattr(controller,'sequence_geometry',None) is None:
            raise ValueError('Authored nonfoot ground constraints required')
        self.controller=controller;self.iterations=iterations;self.time_limit_s=time_limit_s
        self.search,self.restore=local_search()

    def solve(self,observed_state,student_reference,warm_commands):
        started=time.perf_counter();c=self.controller;state=copy.deepcopy(observed_state)
        reference=np.asarray(student_reference,dtype=float)
        if reference.shape!=(18,) or not np.isfinite(reference).all():raise ValueError('Finite student reference required')
        H=c.settings.horizon_steps
        if not state.contacts.any():
            return dict(accepted=False,commands=None,reason='No current support; support-only planner cannot certify flight',
                used_observed_state=True,wall_seconds=time.perf_counter()-started)
        warm=shifted_seed(warm_commands,0,H,state.previous_command_target,c.qp.interface,c.settings.control_dt_s)
        candidates=[('shifted_previous_plan',warm)];records=[];equilibrium=None
        for attempt in range(2):
            if attempt==1:
                equilibrium=future_equilibrium_reference(c,state)
                if not equilibrium['solved']:break
                candidates.append(('new_observed_equilibrium',rate_admitted_seed(c,state,equilibrium['target'],H)))
            label,seed=candidates[-1];before,_=characterize(c,state,seed,reference)
            row=dict(seed=label,before=before,search=None,repair=None);records.append(row)
            commands=seed
            if not before['accepted']:
                remaining=self.time_limit_s-(time.perf_counter()-started)
                if remaining<=0:break
                task=self.search(c,state,reference,seed,iterations=self.iterations,time_limit_s=min(600.,remaining))
                row['search']=task
                commands=task['commands']
                if commands is None and time.perf_counter()-started<self.time_limit_s:
                    repair=self.restore(c,state,reference,task['final_search_commands'],iterations=self.iterations)
                    row['repair']=repair;commands=repair['commands']
                if commands is None:continue
            checked,_=characterize(c,state,commands,reference);row['after']=checked
            if checked['accepted']:
                return dict(accepted=True,commands=commands,records=records,verification=checked,
                    used_observed_state=True,initial_q=state.q.copy(),initial_v=state.v.copy(),
                    initial_contacts=state.contacts.copy(),previous_command=state.previous_command_target.copy(),
                    wall_seconds=time.perf_counter()-started,native_execution=False,
                    per_call_hard_deadline=False,external_process_guard_required=True)
        return dict(accepted=False,commands=None,records=records,equilibrium=equilibrium,
            used_observed_state=True,reason='No strictly verified sequence from supplied current state',
            wall_seconds=time.perf_counter()-started,native_execution=False,
            per_call_hard_deadline=False,external_process_guard_required=True)
