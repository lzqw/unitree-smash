"""Extended state-driven landing sequence MPC.

This is the same library-independent controller as v1, with a 64-tick
(0.64 s) dense contact/load horizon.  It is intended to test whether the
additional prediction window improves post-touchdown momentum capture while
retaining the original 8 s task evaluation.
"""
from scripts.pure_control_state_driven_long_mpc_v1 import PureControlStateDrivenLongMPCV1


class PureControlStateDrivenLongMPCV2(PureControlStateDrivenLongMPCV1):
    horizon_steps = 64
    replan_stride = 16


__all__ = ['PureControlStateDrivenLongMPCV2']
