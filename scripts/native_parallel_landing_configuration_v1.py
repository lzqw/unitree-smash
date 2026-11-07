"""Admit an exactly matching worker controller before native parallel dispatch."""
from dataclasses import asdict

from scripts.parallel_landing_solver_v1 import ParallelLandingSolverV1, build_controller
from scripts.landing_full_command_interval_mpc_v1 import LandingFullCommandIntervalMPCV1
from scripts.landing_feasible_command_cost_mpc_v1 import LandingFeasibleCommandCostMPCV1


def configuration_from_controller(controller):
    # Restrict this first adapter to the two already replayed controller types.
    families={LandingFullCommandIntervalMPCV1:'full_interval',
              LandingFeasibleCommandCostMPCV1:'feasible_command_cost'}
    if type(controller) not in families:
        raise ValueError('Unsupported native parallel controller')
    qp=controller.qp;i=qp.interface
    if (qp.substeps!=16 or qp.load_aware or qp.load_force_floor_n!=2. or
            qp.reference_frequency_rad_s!=10. or controller.posture_target is None):
        raise ValueError('Undeclared parallel controller configuration')
    cfg=dict(family=families[type(controller)],
        command_change_weight=getattr(qp,'command_change_weight',0.),
        arm_target_rate_rad_s=i.arm_rate,posture_target=controller.posture_target.copy(),
        interface=dict(limits=i.limits.copy(),stiffness=i.kp.copy(),damping=i.kd.copy(),
                       effort=i.effort.copy(),velocity=i.velocity.copy(),friction=i.friction),
        qp_settings=asdict(qp.settings),settings=asdict(controller.settings))
    clone=build_controller(cfg)
    if clone.contract()!=controller.contract():
        raise ValueError('Parallel worker contract does not match native controller')
    return cfg


def attach_parallel_solver(policy, workers):
    if (getattr(policy,'_landing_solver_pool',None) is not None or
            not policy.landing_mpc_command_history_interface or
            policy.landing_mpc_hold_first_action or policy.landing_mpc_torque_overlay):
        raise ValueError('Parallel dispatch requires command history and no held/overlay mode')
    cfg=configuration_from_controller(policy.landing_mpc)
    policy._landing_solver_pool=ParallelLandingSolverV1(cfg,workers=workers)
