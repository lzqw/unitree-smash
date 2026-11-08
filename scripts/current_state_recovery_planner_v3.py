"""Full-remainder shooting with a complete foot-placement recovery warm start.

Each solve predicts from the supplied current state and command history. A
feedback plan can be followed between solves; no first-frame command hold is
substituted for that feedback. Only a fine full-tail independent replay admits
a candidate. Native execution and recursive feasibility remain unproven.
"""
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time

import numpy as np

from scripts.search_preserved_early_support_v1 import initialize
from scripts.search_landing_support_tail_v3 import _WORKER
from scripts.hip_mount_path_constraint_v1 import HipMountPathConstraintV1
from scripts.terminal_attitude_feedback_v1 import TerminalAttitudeFeedbackV1
from scripts.solve_recorded_recovery_suffix_v1 import SuffixCommandCorrectionV1
from scripts.check_constrained_recovery_suffix_v1 import state_from_row
from scripts.speed_limited_recovery_suffix_v1 import evaluate_suffix
from scripts.search_braked_support_transition_v1 import SupportTransitionV1
from scripts.landing_joint_braking_reference_v1 import LandingJointBrakingReferenceV1


def build_feedback_plan(worker,plan):
    if plan.get('contract') != 'foot_space_current_state_plan_v1':
        raise ValueError('Explicit complete foot-space plan contract required')
    controller=worker['controller']
    base=worker['factory'](controller.qp.model,controller.qp.interface,
        worker['initial'],np.asarray(plan['original_knots_rad']))
    policy=TerminalAttitudeFeedbackV1(base,proportional=.3,damping_s=.04,maximum_joint_delta_rad=.08)
    # The existing transition remains on its original clock. Replanning only
    # appends future corrections; it does not reset the reference initial state.
    policy=SupportTransitionV1(policy,plan['transition_knots_rad'])
    policy=LandingJointBrakingReferenceV1(policy,**plan['braking'])
    last=279
    for patch in plan['corrections']:
        tick=patch['control_tick']
        if type(tick) is not int or not last<tick<800:raise ValueError('Corrections require increasing original ticks')
        policy=SuffixCommandCorrectionV1(policy,tick,np.asarray(patch['previous_command']),patch['knots_rad'])
        last=tick
    return policy


def predict_plan(plan,query,tick,stable_ticks,audit,substeps,deadline):
    w=_WORKER;state=state_from_row(query);policy=build_feedback_plan(w,plan)
    geometry=HipMountPathConstraintV1(w['geometry'],audit,margin_rad=.02)
    result=evaluate_suffix(w['controller'],state,policy,w['spheres'],geometry,w['protocol'],
        control_tick=tick,substeps=substeps,deadline=deadline,preceding_stable_ticks=stable_ticks,save=substeps==160)
    for key,field in [('initial_q','q'),('initial_v','v'),('initial_contacts','contacts'),('initial_previous_command','target')]:
        if not np.array_equal(result[key],query[field]):raise ValueError('Planner replaced the supplied '+field)
    return dict(plan=plan,result=result)


class CurrentStateRecoveryPlannerV3:
    def __init__(self,initargs,audit,*,workers=4,max_seconds=600.,seed=882122):
        if not 1<=workers<=4 or not 120<=max_seconds<=1800:raise ValueError('Bounded planner budget required')
        self.initargs=copy.deepcopy(initargs);self.audit=copy.deepcopy(audit)
        self.workers=workers;self.max_seconds=max_seconds;self.rng=np.random.default_rng(seed)
        self.pool=None;self.calls=0

    def close(self):
        if self.pool is not None:self.pool.shutdown(wait=True,cancel_futures=True);self.pool=None

    def solve(self,observed_state,control_tick,warm_plan,*,preceding_stable_ticks=0):
        began=time.monotonic();deadline=began+self.max_seconds;search_end=began+.80*self.max_seconds
        if type(control_tick) is not int or not 280<=control_tick<800:raise ValueError('Actual original clock required')
        plan=copy.deepcopy(warm_plan)
        if any(p['control_tick']>control_tick for p in plan['corrections']):
            raise ValueError('Future correction epochs must not be treated as executed history')
        if plan['corrections'] and plan['corrections'][-1]['control_tick']==control_tick:
            # A patch starting now has not been executed. Anchor it to the
            # actual dispatched history, never a stale nominal command.
            plan['corrections'][-1]['previous_command']=observed_state.previous_command_target.copy()
        query=dict(q=observed_state.q.copy(),v=observed_state.v.copy(),contacts=observed_state.contacts.copy(),
            target=observed_state.previous_command_target.copy())
        if self.pool is None:
            self.pool=ProcessPoolExecutor(max_workers=self.workers,mp_context=mp.get_context('spawn'),
                initializer=initialize,initargs=self.initargs)
        self.calls+=1;records=[];best=None
        def batch(plans,resolution,limit):
            jobs={self.pool.submit(predict_plan,p,query,control_tick,preceding_stable_ticks,self.audit,
                resolution,limit):i for i,p in enumerate(plans)}
            rows=[None]*len(plans)
            for future in as_completed(jobs):rows[jobs[future]]=future.result()
            return rows
        def compact(row):
            return dict(plan=row['plan'],result={k:v for k,v in row['result'].items()
                if k not in ['trajectory','initial_q','initial_v','initial_contacts','initial_previous_command']})
        def decision(winner):
            replay=batch([winner['plan']],160,deadline)[0];a=winner['result'];b=replay['result']
            same=(a['accepted'] and b['accepted'] and len(a['trajectory'])==len(b['trajectory'])==800-control_tick
                and all(np.array_equal(x[k],y[k]) for x,y in zip(a['trajectory'],b['trajectory'])
                        for k in ['q','v','target','contacts']))
            if not same:
                records.append(dict(stage='independent_replay_rejected',row=compact(replay)))
                return None
            return dict(accepted=True,plan=winner['plan'],prediction=b,
                first_command=np.asarray(b['trajectory'][0]['target']).copy(),
                control_tick=control_tick,preceding_stable_ticks=preceding_stable_ticks,
                query=query,records=records,independent_replay_exact=True,
                wall_seconds=time.monotonic()-began,actual_state_used=True,native_execution=False,
                valid_expert_labels=False,real_time_guarantee=False)
        warm=batch([plan],160,search_end)[0];best=warm
        records.append(dict(stage='current_state_warm_prediction',row=compact(warm)))
        if warm['result']['accepted']:
            answer=decision(warm)
            if answer is not None:return answer
        # Re-optimization appends a command-continuous correction at the
        # current epoch. Earlier patches retain their original clocks.
        stem=copy.deepcopy(plan)
        if stem['corrections'] and stem['corrections'][-1]['control_tick']==control_tick:
            # Optimize that not-yet-executed patch instead of creating two
            # patches at one epoch or changing an earlier executed patch.
            initial=np.asarray(stem['corrections'].pop()['knots_rad']).copy()
        else:initial=np.zeros((6,18))
        mean=initial.copy();std=np.tile(np.r_[np.full(12,.045),np.full(6,.012)],(6,1))
        floor=np.tile(np.r_[np.full(12,.008),np.full(6,.003)],(6,1));bound=np.r_[np.full(12,.3),np.full(6,.12)]
        def patched(knots):
            result=copy.deepcopy(stem)
            result['corrections'].append(dict(control_tick=control_tick,
                previous_command=query['target'].copy(),knots_rad=np.asarray(knots).copy()))
            return result
        for generation in range(4):
            if time.monotonic()>=search_end:break
            candidates=[np.clip(mean+std*(.75*self.rng.normal(size=(1,18))+.5*self.rng.normal(size=(6,18))),-bound,bound)
                for _ in range(24)]
            candidates[0]=mean.copy();candidates[1]=initial.copy()
            coarse=batch([patched(k) for k in candidates],32,search_end)
            ranked=[r for r in coarse if (r['result']['failure'] or {}).get('reason')!='declared_compute_budget']
            ranked.sort(key=lambda r:r['result']['rank'])
            records.append(dict(stage='coarse_candidates',generation=generation,rows=[compact(r) for r in coarse]))
            if not ranked:break
            elite=np.stack([r['plan']['corrections'][-1]['knots_rad'] for r in ranked[:6]])
            mean=.3*mean+.7*elite.mean(0);std=np.maximum(floor,.3*std+.7*elite.std(0))
            fine=batch([r['plan'] for r in ranked[:4]],160,search_end)
            records.append(dict(stage='fine_candidates',generation=generation,rows=[compact(r) for r in fine]))
            for row in fine:
                if ((row['result']['failure'] or {}).get('reason')!='declared_compute_budget'
                        and row['result']['rank']<best['result']['rank']):best=row
            if best['result']['accepted']:
                answer=decision(best)
                if answer is not None:return answer
                break
        return dict(accepted=False,plan=None,prediction=None,first_command=None,
            control_tick=control_tick,preceding_stable_ticks=preceding_stable_ticks,query=query,
            records=records,best_rejected=compact(best),wall_seconds=time.monotonic()-began,
            actual_state_used=True,native_execution=False,valid_expert_labels=False,
            reason='No independently verified full-remainder plan from the current state',real_time_guarantee=False)
