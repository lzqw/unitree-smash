"""Small strictly convex QP backend using the installed quadprog solver.

OSQP-shaped setup/result fields simplify mathematical cross-checks. No Hessian
regularization, constraint softening or physical tolerance changes are added.
The iteration limit is checked after the native active-set solve, not a timer.
"""
import time
from types import SimpleNamespace
import numpy as np
import quadprog


class DenseActiveSetQP:
    def setup(self,*,P,q,A,l,u,eps_abs=1e-6,eps_rel=1e-6,max_iter=10000,polish=True,verbose=False):
        upper=P.toarray() if hasattr(P,'toarray') else np.asarray(P,dtype=float)
        self.H=upper+upper.T-np.diag(np.diag(upper))
        self.q=np.asarray(q,dtype=float).copy()
        self.A=A.toarray() if hasattr(A,'toarray') else np.asarray(A,dtype=float)
        self.low=np.asarray(l,dtype=float).copy();self.high=np.asarray(u,dtype=float).copy()
        n=len(self.q)
        if (self.H.shape!=(n,n) or self.A.shape!=(len(self.low),n) or self.high.shape!=self.low.shape
            or any(not np.isfinite(x).all() for x in (self.H,self.q,self.A))
            or np.isnan(self.low).any() or np.isnan(self.high).any() or np.any(self.low>self.high)
            or eps_abs<=0 or eps_rel<0 or max_iter<1):raise ValueError('Invalid convex QP data')
        if not np.array_equal(upper,np.triu(upper)):raise ValueError('Require upper triangular Hessian, as in OSQP')
        equal=np.isfinite(self.low)&np.isfinite(self.high)&(self.low==self.high)
        lower=np.isfinite(self.low)&~equal;higher=np.isfinite(self.high)&~equal
        self.C=np.concatenate((self.A[equal],self.A[lower],-self.A[higher]),axis=0).T
        self.b=np.r_[self.low[equal],self.low[lower],-self.high[higher]]
        self.meq=int(equal.sum());self.eps_abs=float(eps_abs);self.eps_rel=float(eps_rel)
        self.max_iter=int(max_iter)

    def solve(self):
        started=time.perf_counter();x=None;iterations=0;status='active-set solver error';status_val=0
        self.last_diagnostics={}
        try:
            x,_,_,counts,multipliers,_=quadprog.solve_qp(self.H,-self.q,self.C,self.b,self.meq)
            iterations=int(counts[0]);slack=self.C.T@x-self.b
            primal=float(max(0.,np.max(abs(slack[:self.meq]),initial=0.),
                np.max(-slack[self.meq:],initial=0.)))
            stationarity=float(np.max(abs(self.H@x+self.q-self.C@multipliers),initial=0.))
            dual=float(max(0.,np.max(-multipliers[self.meq:],initial=0.)))
            complementarity=float(np.max(abs(slack[self.meq:]*multipliers[self.meq:]),initial=0.))
            scale=max(1.,float(np.max(abs(self.H@x),initial=0.)),float(np.max(abs(self.q),initial=0.)))
            tolerance=self.eps_abs+self.eps_rel*scale
            self.last_diagnostics=dict(scaled_primal_residual=primal,stationarity=stationarity,
                dual_sign_violation=dual,complementarity=complementarity,iterations=iterations,
                minimum_hessian_eigenvalue=float(np.linalg.eigvalsh(self.H).min()))
            if not np.isfinite(x).all():status='nonfinite active-set solution'
            elif iterations>self.max_iter:status='active-set iteration limit exceeded'
            elif primal>self.eps_abs or max(stationarity,dual,complementarity)>tolerance:
                status='active-set KKT residual rejected'
            else:status='solved';status_val=1
        except ValueError as error:
            message=str(error)
            if 'constraints are inconsistent' in message:status='primal infeasible';status_val=3
            else:status='active-set error: '+message
        elapsed=time.perf_counter()-started
        return SimpleNamespace(x=x,info=SimpleNamespace(status=status,status_val=status_val,
            iter=iterations,run_time=elapsed))
