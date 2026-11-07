"""Measured-state MPC with nonlinear-verified local feasibility restoration.

The repair uses current q/v and command history on every call. It is a search
method, not a proof of recursive feasibility. CEM remains a bounded fallback.
"""
import copy
import time
import numpy as np
from scripts.frictional_receding_recovery_v1 import FrictionalRecedingRecoveryV1,admit_commands
from scripts.frictional_candidate_pool_v1 import FrictionalCandidatePoolV1
from scripts.restore_frictional_terminal_v1 import violation_vector,expand
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.search_frictional_recovery_cem_v1 import key


class FrictionalRecedingRecoveryV2(FrictionalRecedingRecoveryV1):
    def __init__(self,configuration,spheres,*,repair_iterations=4,**kwargs):
        kwargs.setdefault('time_limit_s',600.)
        super().__init__(configuration,spheres,**kwargs)
        if self.horizon!=80 or not 1<=repair_iterations<=8:
            raise ValueError('Require bounded full 80-step repair')
        if self.controller.settings.terminal_support_margin_m!=.01:
            raise ValueError('Current constraint vector requires original support margin')
        self.repair_iterations=repair_iterations

    def solve(self,observed_state,student_reference,warm_commands):
        started=time.monotonic();state=copy.deepcopy(observed_state)
        reference=np.asarray(student_reference,float)
        if reference.shape!=(18,) or not np.isfinite(reference).all():raise ValueError('Absolute student command required')
        seed=admit_commands(warm_commands,state.previous_command_target,self.controller.qp.interface)
        if seed.shape!=(80,18):raise ValueError('Complete warm horizon required')
        deadline=started+self.time_limit_s;records=[];best=None;current=np.zeros(54)
        if self.pool is None:
            self.pool=FrictionalCandidatePoolV1(self.configuration,self.spheres,workers=self.workers,timeout_s=180.)
        def evaluate(points,save=False,shift=0):
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('Repair budget exhausted')
            self.pool.timeout_s=min(300.,remaining)
            requests=[dict(lane=i+shift,state=state,seed=seed,correction=correction(x),save_states=save) for i,x in enumerate(points)]
            answers=self.pool.batch(requests)
            for row in answers:
                for field,value in [('input_q',state.q),('input_v',state.v),('input_previous_command',state.previous_command_target)]:
                    if not np.array_equal(row[field],value):raise ValueError('Repair used stale '+field)
            return answers
        # Start with the proven late correction basis; if its seed already
        # violates an early path constraint, permit corrections from tick zero.
        early=False
        def correction(values):
            if not early:return expand(values)
            knots=np.asarray(values).reshape(3,18)
            return np.column_stack([np.interp(np.arange(80),[0.,39.5,79.],knots[:,j]) for j in range(18)])
        def compact(result):
            return {k:v for k,v in result.items() if k not in ['states','commands','input_q','input_v','input_previous_command']}
        def decision(result):
            checked=evaluate([current],True,shift=1)[0]
            if not checked['accepted'] or key(checked)!=key(result):raise ValueError('Repair independent verification differs')
            commands=np.asarray(checked['commands'])
            if not np.allclose(admit_commands(commands,state.previous_command_target,self.controller.qp.interface),commands,atol=1e-14,rtol=0):
                raise ValueError('Repair violates actual command history')
            return dict(accepted=True,commands=commands,predicted_states=checked['states'],verification=compact(checked),
                records=records,initial_q=state.q.copy(),initial_v=state.v.copy(),initial_contacts=state.contacts.copy(),
                previous_command=state.previous_command_target.copy(),used_observed_state=True,horizon_steps=80,
                wall_seconds=time.monotonic()-started,teacher_model='frictional_spheres_v1',
                planner_version='local_restoration_v2',native_execution=False,valid_expert_labels=False,
                per_call_hard_deadline=False,external_process_guard_required=True)
        try:
            best=evaluate([current])[0]
            records.append(dict(seed='shifted_verified_plan',result=compact(best)))
            if best['accepted']:return decision(best)
            first=best.get('first_violation_tick')
            early=first is not None and first<=12
            trust=.03
            for iteration in range(self.repair_iterations if best['solved'] else 0):
                try:residual=violation_vector(best)
                except ValueError:break
                epsilon=.001;directions=np.eye(54)*epsilon
                values=evaluate([current+d for d in directions]+[current-d for d in directions])
                jac=np.zeros((len(residual),54));unavailable=[]
                for j in range(54):
                    pos,neg=values[j],values[j+54]
                    try:
                        if pos['solved'] and neg['solved']:jac[:,j]=(violation_vector(pos)-violation_vector(neg))/(2*epsilon)
                        elif pos['solved']:jac[:,j]=(violation_vector(pos)-residual)/epsilon
                        elif neg['solved']:jac[:,j]=(residual-violation_vector(neg))/epsilon
                        else:unavailable.append(j)
                    except ValueError:unavailable.append(j)
                n=54;m=len(residual)
                A=np.vstack([np.c_[jac,-np.eye(m)],np.c_[np.zeros((m,n)),np.eye(m)],np.c_[np.eye(n),np.zeros((n,m))]])
                low=np.r_[np.full(m,-np.inf),np.zeros(m),np.full(n,-trust)]
                high=np.r_[-residual,np.full(m,np.inf),np.full(n,trust)]
                for j in unavailable:low[2*m+j]=high[2*m+j]=0.
                solver=DenseActiveSetQP();solver.setup(P=np.diag(np.r_[np.ones(n),np.full(m,1e4)]),q=np.zeros(n+m),A=A,l=low,u=high,eps_abs=1e-7,eps_rel=1e-7)
                answer=solver.solve();row=dict(iteration=iteration,qp_status=answer.info.status,before=compact(best),early_correction=early)
                if answer.info.status_val!=1:records.append(row);break
                delta=answer.x[:n];scales=[1.,.5,.25,.125];points=[current+s*delta for s in scales]
                values=evaluate(points);winner=min(range(len(values)),key=lambda i:key(values[i]))
                improved=key(values[winner])<key(best)
                row.update(unavailable_columns=unavailable,slack_max=float(np.max(answer.x[n:])),
                    line_search=[dict(scale=s,result=compact(v)) for s,v in zip(scales,values)],improved=improved)
                if improved:current=points[winner];best=values[winner];trust=min(.05,trust*1.25)
                else:trust*=.5
                records.append(row)
                if best['accepted']:return decision(best)
            # Only use an unsuccessful repaired sequence as a search initial
            # guess, never as an executable result or positive label.
            remaining=deadline-time.monotonic()
            if remaining>5:
                candidate=evaluate([current],True)[0]
                next_seed=np.asarray(candidate['commands']) if candidate.get('commands') is not None else seed
                old_limit=self.time_limit_s;self.time_limit_s=min(240.,remaining)
                try:fallback=super().solve(state,reference,next_seed)
                finally:self.time_limit_s=old_limit
                fallback['records']=records+[dict(cem_fallback=fallback['records'])]
                fallback['wall_seconds']=time.monotonic()-started;fallback['planner_version']='local_restoration_v2'
                return fallback
        except (TimeoutError,EOFError,ConnectionError,RuntimeError) as error:
            self.close();records.append(dict(worker_failure=repr(error)))
        return dict(accepted=False,commands=None,records=records,used_observed_state=True,
            initial_q=state.q.copy(),initial_v=state.v.copy(),previous_command=state.previous_command_target.copy(),
            horizon_steps=80,wall_seconds=time.monotonic()-started,planner_version='local_restoration_v2',
            reason='No nonlinear-verified continuation from supplied state',native_execution=False,
            valid_expert_labels=False,per_call_hard_deadline=False,external_process_guard_required=True,
            teacher_model='frictional_spheres_v1')
