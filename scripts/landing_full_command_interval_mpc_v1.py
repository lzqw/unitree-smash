"""Native candidate: all18 command feedback with anticipatory joint limits.

This is the zero-reference-search branch of the coupled shooting experiment.
It does not execute expensive CEM online, nor claim full recovery validation.
"""
from scripts.landing_joint_damper_mpc_v1 import LandingJointDamperMPCV1


class LandingFullCommandIntervalMPCV1(LandingJointDamperMPCV1):
    def __init__(self,*args,**kwargs):
        if kwargs.get('preserve_arm_command',False):
            raise ValueError('Full-command candidate requires the explicit free-arm branch')
        kwargs['preserve_arm_command']=False
        super().__init__(*args,**kwargs)

    def contract(self):
        result=super().contract()
        result.update(schema='landing-full-command-interval-v1',
            preserve_arm_position_command=False,preserve_arm_command=False,
            arm_command_semantics='all18-physical-command-optimization; student-soft-posture-reference',
            reference_space_search_enabled=False,physical_constraints_relaxed=False,
            native_recovery_validated=False)
        return result
