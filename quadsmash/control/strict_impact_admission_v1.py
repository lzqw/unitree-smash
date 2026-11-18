"""Fail-closed impact admission for full-body candidates.

The analytic evaluator remains available for coarse ranking. This adapter is
required before a candidate can be labelled finite-mass verified or teacher
eligible.
"""
import numpy as np


class StrictImpactAdmissionV1:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def evaluate(self, *, shuttle_position, shuttle_velocity_before,
                 racket_contact_position, racket_contact_velocity,
                 racket_normal, contact_point_local, robot_state,
                 task_context=None):
        if not isinstance(robot_state, dict):
            raise ValueError('Full-body robot_state is required')
        required=('mass_matrix','contact_jacobian_world','support_mode')
        missing=[k for k in required if k not in robot_state]
        if missing: raise ValueError('Missing strict impact fields: '+','.join(missing))
        mass=np.asarray(robot_state['mass_matrix'],float);contact=np.asarray(robot_state['contact_jacobian_world'],float)
        if mass.ndim!=2 or mass.shape[0]!=mass.shape[1] or not np.isfinite(mass).all():
            raise ValueError('Finite square mass matrix required')
        if contact.ndim!=2 or contact.shape[1]!=mass.shape[0] or contact.shape[0] < 3 or not np.isfinite(contact).all():
            raise ValueError('Contact Jacobian must match mass matrix')
        mode=str(robot_state['support_mode'])
        if mode not in ('support','flight'):
            raise ValueError('Explicit support_mode required')
        support=robot_state.get('support_jacobian')
        if mode=='support':
            if support is None or np.asarray(support).ndim!=2 or np.asarray(support).shape[0] == 0:
                raise ValueError('Non-empty supporting contact Jacobian required')
            if np.asarray(support).shape[1] != np.asarray(robot_state['mass_matrix']).shape[0]:
                raise ValueError('Support Jacobian and mass matrix dimensions disagree')
        if mode=='flight' and support is not None and np.asarray(support).size:
            raise ValueError('Flight candidate cannot carry support Jacobian')
        result=self.evaluator.evaluate(shuttle_position,shuttle_velocity_before,
            racket_contact_position,racket_contact_velocity,racket_normal,
            contact_point_local,robot_state=robot_state,task_context=task_context)
        if result.get('effective_racket_mass_kg') is None:
            result.update(strict_status='EFFECTIVE_MASS_UNVERIFIED',
                          finite_mass_verified=False,teacher_eligible=False)
            return result
        result.update(strict_status='FINITE_MASS_VERIFIED',
                      finite_mass_verified=True,teacher_eligible=False)
        return result
