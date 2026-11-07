"""Finite-horizon search over smooth COM-reference accelerations.

Each candidate is a feedback-QP rollout, not an unconstrained force sequence.
Future terminal outcomes select the current command. This is reference-space
shooting, NOT full joint-space optimal control or a native recovery guarantee.
Expensive offline candidate generation; intentionally not in the live entry.
"""
import copy
from dataclasses import dataclass
import time
import numpy as np

from scripts.landing_joint_damper_mpc_v1 import JointDamperIntervalQP, LandingJointDamperMPCV1


def reference_acceleration(knots, fraction):
    knots=np.asarray(knots,dtype=float)
    if knots.shape!=(3,3) or not np.isfinite(knots).all() or not np.isfinite(fraction):
        raise ValueError('Three finite world-XYZ acceleration knots required')
    t=float(np.clip(fraction,0.,1.))
    return (1.-t)**2*knots[0]+2.*t*(1.-t)*knots[1]+t*t*knots[2]


class ShootingReferenceQP(JointDamperIntervalQP):
    shooting_knots=None
    shooting_horizon=40

    def reference(self,state,snapshot,joint_reference):
        result=super().reference(state,snapshot,joint_reference)
        if self.shooting_knots is not None:
            stage=int(getattr(state,'_shooting_stage',0))
            delta=reference_acceleration(self.shooting_knots,stage/max(self.shooting_horizon-1,1))
            result['com_acceleration']=result['com_acceleration']+delta
            result['shooting_com_acceleration_delta_world']=delta
        return result


@dataclass(frozen=True)
class ReferenceSearchSettings:
    population:int=16
    elites:int=4
    generations:int=3
    seed:int=882083
    horizontal_bound_mps2:float=3.
    vertical_bound_mps2:float=4.


class LandingReferenceShootingMPCV1(LandingJointDamperMPCV1):
    qp_type=ShootingReferenceQP

    def __init__(self,*args,search_settings=ReferenceSearchSettings(),**kwargs):
        super().__init__(*args,**kwargs)
        s=search_settings
        if (type(s.population) is not int or type(s.elites) is not int or type(s.generations) is not int or
                not 2<=s.elites<s.population or not 1<=s.generations<=20 or s.population>128 or
                not np.isfinite([s.horizontal_bound_mps2,s.vertical_bound_mps2]).all() or
                min(s.horizontal_bound_mps2,s.vertical_bound_mps2)<=0):
            raise ValueError('Explicit bounded finite search required')
        self.search_settings=s
        self.qp.shooting_horizon=self.settings.horizon_steps

    def _integrate(self,state,acceleration,plan=None):
        result=super()._integrate(state,acceleration,plan)
        result._shooting_stage=int(getattr(state,'_shooting_stage',0))+1
        return result

    def _rank(self,plan,knots):
        horizon=plan.get('horizon',[])
        steps=sum(bool(row['accepted']) for row in horizon)
        risk=plan.get('terminal_risk') or (horizon[-1].get('risk') if horizon else plan.get('risk')) or {}
        feet=int(risk.get('support_feet',0))
        margin_target=(self.settings.terminal_support_margin_m if feet>=3 else
            -self.settings.two_foot_max_com_distance_m if feet==2 else
            -self.settings.one_foot_max_com_distance_m if feet==1 else 0.)
        margin=float(risk.get('support_margin_m',-1.))
        if not np.isfinite(margin):margin=-1.
        violation=(max(0.,margin_target-margin)/.05+
            max(0.,float(risk.get('tilt_rad',np.pi))-self.settings.terminal_tilt_rad)+
            max(0.,float(risk.get('horizontal_speed_mps',10.))-self.settings.terminal_horizontal_speed_mps)+
            max(0.,float(risk.get('angular_speed_rad_s',10.))-self.settings.terminal_angular_speed_rad_s))
        effort=.001*float(np.mean(np.asarray(knots)**2))
        if plan.get('accepted'):
            return (0,effort+.01*float(risk['horizontal_speed_mps'])**2+
                    .01*float(risk['angular_speed_rad_s'])**2)
        if steps==self.settings.horizon_steps:return (1,violation+effort)
        return (2,(self.settings.horizon_steps-steps)+.01*min(violation,100.)+effort)

    def solve(self,state,joint_reference,*,force=False,lane=0,control_tick=None):
        started=time.perf_counter()
        if control_tick is None:control_tick=self._observed_tick.get(int(lane),-1)+1
        # Observe only the unmodified live reference, exactly once. Candidate
        # trials then branch from that tick and are rolled back by V3.solve.
        self.qp.shooting_knots=None
        self.observe(state,joint_reference,lane=lane,control_tick=control_tick)
        zero=np.zeros((3,3))
        ledger=[]

        def evaluate(knots,generation):
            self.qp.shooting_knots=np.asarray(knots,dtype=float).copy()
            try:
                result=super(LandingReferenceShootingMPCV1,self).solve(
                    state,joint_reference,force=force,lane=lane,control_tick=control_tick)
            finally:
                self.qp.shooting_knots=None
            rank=self._rank(result,knots)
            ledger.append(dict(generation=generation,knots_world_mps2=np.asarray(knots).tolist(),
                rank=list(rank),accepted=bool(result['accepted']),
                valid_steps=sum(bool(x['accepted']) for x in result.get('horizon',[])),
                rejection=result.get('rejection'),terminal=result.get('terminal_risk')))
            return rank,result

        baseline_rank,baseline=evaluate(zero,-1)
        best_rank,best,best_knots=baseline_rank,baseline,zero.copy()
        # No alteration of a validated nominal horizon, or a controller that
        # never entered the recovery/risk branch.
        if not baseline['accepted'] and baseline.get('intervened'):
            s=self.search_settings
            rng=np.random.default_rng(s.seed)
            bounds=np.tile([s.horizontal_bound_mps2,s.horizontal_bound_mps2,s.vertical_bound_mps2],(3,1))
            mean=np.zeros((3,3));std=.5*bounds
            for generation in range(s.generations):
                population=np.clip(rng.normal(mean,std,(s.population,3,3)),-bounds,bounds)
                population[0]=best_knots
                ranked=[]
                for knots in population:
                    rank,result=evaluate(knots,generation)
                    ranked.append((rank,knots.copy()))
                    if rank<best_rank:best_rank,best,best_knots=rank,result,knots.copy()
                ranked.sort(key=lambda item:item[0])
                elite=np.stack([x[1] for x in ranked[:s.elites]])
                mean=elite.mean(0);std=np.maximum(elite.std(0),.08*bounds)
        # A lower failure cost is diagnostic, never authority to send a new
        # command. Preserve the original failed/fallback plan if no full pass.
        selected=best if best.get('accepted') else baseline
        selected=copy.deepcopy(selected)
        selected.update(reference_shooting=dict(
            baseline_accepted=bool(baseline['accepted']),selected_new_plan=bool(best['accepted'] and not baseline['accepted']),
            selected_knots_world_mps2=(best_knots if best['accepted'] else zero).tolist(),
            best_rank=list(best_rank),best_rejection=best.get('rejection'),
            best_terminal=best.get('terminal_risk'),candidates=ledger,
            candidate_count=len(ledger),wall_seconds=time.perf_counter()-started,
            physical_constraints_relaxed=False,native_execution=False))
        return selected

    def contract(self):
        result=super().contract()
        result.update(schema='landing-reference-shooting-mpc-v1',
            outer_optimizer='bounded-reference-space-CEM',
            entire_horizon_selects_current_command=True,
            full_joint_space_trajectory_optimization=False,native_recovery_validated=False)
        return result
