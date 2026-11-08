"""Independent mechanical checks; these do not validate PhysX equivalence."""
import unittest
import numpy as np
from scipy.optimize import minimize, LinearConstraint

from scripts.frictional_impulse_step_v1 import frictional_velocity_step
from scripts.frictional_speed_limited_step_v1 import speed_limited_velocity_step


class SpeedLimitMechanics(unittest.TestCase):
    def floating_problem(self, sign=1., effort=100.):
        return dict(mass=np.array([[2.,.4],[.4,.3]]),velocity=np.zeros(2),bias=np.zeros(2),
            selection=np.array([[0.],[1.]]),kp=np.array([80.]),kd=np.array([4.]),
            target_error=np.array([sign*.2]),effort=np.array([effort]),dt=.01,
            jacobian=np.zeros((0,2)),normal_target=np.zeros(0),mu=.65)

    def test_inactive_is_exact_old_step(self):
        problem=self.floating_problem();a=frictional_velocity_step(**problem)
        b=speed_limited_velocity_step(**problem,velocity_limits=np.array([30.]))
        for key in ['velocity','impulse','torque']:
            self.assertTrue(np.array_equal(a[key],b[key]))
        self.assertFalse(b['velocity_limit_active'])

    def test_floating_base_reaction_and_analytic_drive(self):
        for sign in [-1.,1.]:
            p=self.floating_problem(sign);r=speed_limited_velocity_step(**p,velocity_limits=np.array([.1]))
            # The unactuated momentum equation fixes base velocity; clipping
            # only the joint component would violate this equation.
            expected=sign*np.array([-.02,.1])
            np.testing.assert_allclose(r['velocity'],expected,atol=1e-12,rtol=0.)
            np.testing.assert_allclose(r['torque'],[sign*15.52],atol=1e-12,rtol=0.)
            np.testing.assert_allclose(r['velocity_limit_impulse'],[-sign*.1332],atol=1e-12,rtol=0.)
            self.assertLess(r['momentum_residual'],1e-12)

    def test_saturated_drive_is_not_constraint_impulse(self):
        p=self.floating_problem(effort=.01);p['velocity']=np.array([0.,2.]);p['target_error'][:]=0.
        r=speed_limited_velocity_step(**p,velocity_limits=np.array([.1]))
        self.assertLessEqual(abs(r['velocity'][1]),.1+2e-7)
        self.assertLessEqual(abs(r['torque'][0]),.01)
        self.assertGreater(abs(r['velocity_limit_impulse'][0]),.01*p['dt'])
        self.assertLess(r['momentum_residual'],1e-12)

    def contact_problem(self, mu):
        return dict(mass=np.array([[2.,0.,0.,.2],[0.,2.,0.,.1],[0.,0.,2.,.4],[.2,.1,.4,.5]]),
            velocity=np.array([.3,.1,-.4,2.]),bias=np.zeros(4),selection=np.array([[0.],[0.],[0.],[1.]]),
            kp=np.array([20.]),kd=np.array([2.]),target_error=np.array([.1]),effort=np.array([10.]),dt=.01,
            jacobian=np.c_[np.eye(3),np.zeros(3)],normal_target=np.array([0.]),mu=mu)

    def test_frictionless_against_independent_primal_qp(self):
        p=self.contact_problem(0.);r=speed_limited_velocity_step(**p,velocity_limits=np.array([.5]))
        S=p['selection'];A=p['mass']+p['dt']*(S*(p['kd']+p['dt']*p['kp']))@S.T
        b=p['mass']@p['velocity']+p['dt']*S@(p['kp']*p['target_error'])
        constraints=LinearConstraint(np.vstack([p['jacobian'][2],S.T]),[0.,-.5],[np.inf,.5])
        oracle=minimize(lambda v:.5*v@A@v-b@v,np.zeros(4),jac=lambda v:A@v-b,
            constraints=constraints,method='SLSQP',options={'ftol':1e-13,'maxiter':200})
        self.assertTrue(oracle.success,oracle.message)
        np.testing.assert_allclose(r['velocity'],oracle.x,atol=1e-6,rtol=0.)
        self.assertLessEqual(r['speed_violation'],2e-7)
        self.assertLess(r['momentum_residual'],1e-8)

    def test_friction_and_joint_constraints_hold_together(self):
        p=self.contact_problem(.65);r=speed_limited_velocity_step(**p,velocity_limits=np.array([.5]))
        self.assertLessEqual(max(r['contact']['residual'].values()),2e-7)
        self.assertLessEqual(r['speed_violation'],2e-7)
        self.assertLessEqual(r['speed_fixed_point'],2e-7)
        self.assertLess(r['momentum_residual'],1e-8)
        self.assertLessEqual(np.linalg.norm(r['impulse'][:2]),.65*r['impulse'][2]+2e-7)
        self.assertGreaterEqual(r['velocity'][2],-2e-7)

    def test_invalid_speed_limits_rejected(self):
        for limit in [0.,-1.,np.nan,np.inf]:
            with self.assertRaises(ValueError):
                speed_limited_velocity_step(**self.floating_problem(),velocity_limits=np.array([limit]))


if __name__=='__main__':unittest.main()
