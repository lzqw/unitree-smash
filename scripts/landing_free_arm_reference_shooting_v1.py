"""Reference-shooting variant with all 18 physical commands available.

Unlike infeasibility-only arm repair, the arm can cooperate throughout the
predicted recovery. The student command remains a soft posture objective;
original drive, velocity, position, slew and contact bounds remain hard.
No new native validation is implied.
"""
from scripts.landing_reference_shooting_mpc_v1 import LandingReferenceShootingMPCV1


class LandingFreeArmReferenceShootingMPCV1(LandingReferenceShootingMPCV1):
    def __init__(self,*args,**kwargs):
        if kwargs.get('preserve_arm_command',False):
            raise ValueError('This explicit variant optimizes all eighteen commands')
        kwargs['preserve_arm_command']=False
        super().__init__(*args,**kwargs)

    def contract(self):
        result=super().contract()
        result.update(schema='landing-free-arm-reference-shooting-v1',
            preserve_arm_position_command=False,preserve_arm_command=False,
            arm_command_semantics='full18-command-optimization-with-original-physical-bounds',
            arm_reference_is_still_soft_posture_objective=True)
        return result
