"""Stricter planning targets for low-motion, preload-compatible endpoints.

These targets supplement unchanged task/hardware constraints. They are not an
invariant-set certificate: a subsequent dynamic hold still has to be tested.
"""
import numpy as np
from scripts.extend_frictional_recovery_v1 import set_state
from scripts.frictional_static_reference_v1 import sphere_static_reference
from scripts.restore_frictional_terminal_v1 import violation_vector


TARGETS=dict(command_preload_gap_rad=.01,angular_speed_rad_s=.25,
    horizontal_speed_m_s=.10,joint_speed_rad_s=.25,vertical_speed_m_s=.05)


def assess(controller,template,result,spheres):
    if not result['solved'] or not result.get('states'):
        return dict(solved=False,reason='No complete predicted endpoint')
    end=result['states'][-1];command=np.asarray(result['commands'][-1])
    state=set_state(template,controller.qp.model,end['q'],end['v'],end['contacts'],command)
    preload=sphere_static_reference(controller,state,spheres)
    if not preload['solved']:return dict(solved=False,reason='Static preload not found',preload=preload)
    gap=np.asarray(preload['target'])-command;t=result['terminal']
    extra=np.r_[abs(gap)/TARGETS['command_preload_gap_rad']-1,
        t['omega']/TARGETS['angular_speed_rad_s']-1,
        t['horizontal']/TARGETS['horizontal_speed_m_s']-1,
        t['joint_speed']/TARGETS['joint_speed_rad_s']-1,
        t['vertical_speed']/TARGETS['vertical_speed_m_s']-1]
    return dict(solved=True,preload=preload,command_gap_rad=gap.tolist(),extra_constraint_vector=extra.tolist(),
        maximum_scaled_violation=max(0.,float(np.max(extra))),planning_targets=TARGETS,
        task_definition_changed=False,dynamic_hold_verified=False)


def constraints(result):
    if not result['settling']['solved']:raise ValueError('No terminal preload to linearize')
    return np.r_[violation_vector(result),result['settling']['extra_constraint_vector']]


def rank(result):
    extra=result['settling'].get('maximum_scaled_violation',1e3)
    maximum=max(result['maximum_scaled_violation'],extra)
    return (not(result['accepted'] and extra<=1e-10),maximum,result['cost'])
