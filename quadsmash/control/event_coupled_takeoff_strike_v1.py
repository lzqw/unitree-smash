"""Bounded event-coupled low-dimensional full-body candidate planner.

The caller owns the actual 18D rollout. This planner only proposes bounded
outer parameters and scores returned physical states. It never writes q/v or
contact masks and never grants native teacher admission.
"""
import numpy as np

class EventCoupledTakeoffStrikePlannerV1:
    def __init__(self, rollout_fn, *, bounds, max_candidates=64):
        self.rollout_fn=rollout_fn
        self.bounds=np.asarray(bounds,float)
        if self.bounds.ndim!=2 or self.bounds.shape[1]!=2 or not np.isfinite(self.bounds).all() or np.any(self.bounds[:,0]>self.bounds[:,1]):
            raise ValueError('Finite bounded outer parameter ranges required')
        if not isinstance(max_candidates,int) or not 1<=max_candidates<=256: raise ValueError('Bounded candidate budget required')
        self.max_candidates=max_candidates
    @staticmethod
    def _gate(result):
        required=('path_valid','contact_position_error_m','normal_error','velocity_error_m_s')
        if any(k not in result for k in required): return False,'MISSING_ROLLOUT_EVIDENCE'
        if result['path_valid'] is not True: return False,'FULLBODY_PATH_INVALID'
        vals=np.asarray([result[k] for k in required[1:]],float)
        if not np.isfinite(vals).all(): return False,'NONFINITE_ROLLOUT_EVIDENCE'
        if vals[0]>0.003:return False,'POSITION_RESIDUAL'
        if vals[1]>0.12:return False,'NORMAL_INFEASIBLE'
        if vals[2]>0.75:return False,'VELOCITY_INFEASIBLE'
        if result.get('finite_mass_status')!='FINITE_MASS_VERIFIED':return False,'FINITE_MASS_UNVERIFIED'
        for key in ('contact_feasible','legal_return_predicted','attack5_predicted','target_landing_predicted'):
            if result.get(key) is not True: return False,'IMPACT_TASK_INVALID' 
        return True,None
    def optimize(self, seeds, *, iterations=3, step_fraction=.1):
        if not isinstance(iterations,int) or not 1<=iterations<=16: raise ValueError('Bounded optimization iterations required')
        if not 0<step_fraction<=.5: raise ValueError('Bounded optimization step required')
        current=[];trace=[]
        for seed in seeds[:self.max_candidates]:
            theta=np.asarray(seed,float).copy();best=None
            for it in range(iterations):
                result=self.rollout_fn(theta.copy());ok,reason=self._gate(result)
                errors=np.asarray([result.get('contact_position_error_m',1e3),result.get('normal_error',1e3),result.get('velocity_error_m_s',1e3)],float)
                cost=float(np.linalg.norm(errors/np.array([.003,.12,.75]))) if np.isfinite(errors).all() else 1e9
                record=dict(iteration=it,theta=theta.copy(),result=result,cost=cost,accepted=ok,reason=reason)
                trace.append(record)
                if best is None or cost<best['cost']: best=record
                if ok: break
                improved=False
                for j in range(len(theta)):
                    span=(self.bounds[j,1]-self.bounds[j,0])*step_fraction
                    for sign in (-1.,1.):
                        trial=theta.copy();trial[j]=np.clip(trial[j]+sign*span,self.bounds[j,0],self.bounds[j,1]);rr=self.rollout_fn(trial);ee=np.asarray([rr.get('contact_position_error_m',1e3),rr.get('normal_error',1e3),rr.get('velocity_error_m_s',1e3)],float);cc=float(np.linalg.norm(ee/np.array([.003,.12,.75]))) if np.isfinite(ee).all() else 1e9
                        trace.append(dict(iteration=it,theta=trial.copy(),result=rr,cost=cc,accepted=False,reason='probe'))
                        if cc<best['cost']: theta=trial;best=dict(iteration=it,theta=trial.copy(),result=rr,cost=cc,accepted=False,reason='probe');improved=True
                if not improved: break
            current.append(best)
        current.sort(key=lambda x:x['cost']);return dict(best=current,trace=trace,calls=len(trace),native_run=False,teacher_eligible=False)

    def search(self, seeds):
        accepted=[];failed=[];calls=0
        for seed in seeds[:self.max_candidates]:
            theta=np.asarray(seed,float)
            if theta.shape!=(len(self.bounds),) or not np.isfinite(theta).all() or np.any(theta<self.bounds[:,0]) or np.any(theta>self.bounds[:,1]):
                failed.append(dict(theta=theta.tolist(),reason='SEED_OUT_OF_BOUNDS'));continue
            result=self.rollout_fn(theta.copy());calls+=1
            ok,reason=self._gate(result)
            record=dict(theta=theta.tolist(),result=result,fullbody_prediction_valid=ok,native_reference_valid=False,teacher_eligible=False)
            (accepted if ok else failed).append(record)
        accepted.sort(key=lambda x:(x['result']['contact_position_error_m'],x['result']['normal_error'],x['result']['velocity_error_m_s']))
        return dict(accepted=accepted,failed=failed,calls=calls,budget=self.max_candidates,native_run=False,teacher_eligible=False)
