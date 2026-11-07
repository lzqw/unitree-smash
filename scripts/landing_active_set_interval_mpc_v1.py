"""The unchanged interval physical QP solved with a dense active-set method."""
from importlib.metadata import version
from scripts.dense_active_set_qp_v1 import DenseActiveSetQP
from scripts.interval_command_support_qp_v1 import IntervalCommandSupportQP
from scripts.landing_interval_command_mpc_v1 import LandingIntervalCommandMPCV1


class ActiveSetIntervalCommandSupportQP(IntervalCommandSupportQP):
    def _make_solver(self):
        return DenseActiveSetQP()


class LandingActiveSetIntervalMPCV1(LandingIntervalCommandMPCV1):
    qp_type=ActiveSetIntervalCommandSupportQP

    def contract(self):
        result=super().contract()
        result.update(schema='landing-active-set-interval-mpc-v1',qp_backend='quadprog',
            qp_backend_version=version('quadprog'),hessian_regularized=False,
            physical_constraints_relaxed=False,iteration_limit_checked_after_solve=True)
        return result
