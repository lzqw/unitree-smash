import unittest
import numpy as np

from scripts.frictional_position_limited_step_v1 import joint_velocity_bounds,position_limited_velocity_step
from scripts.frictional_speed_limited_step_v1 import speed_limited_velocity_step


class PositionLimitedImpulseTests(unittest.TestCase):
    def free_problem(self,velocity=2.):
        return dict(mass=np.array([[2.,.5],[.5,1.]]),velocity=np.array([.2,velocity]),
            bias=np.zeros(2),selection=np.array([[0.],[1.]]),kp=np.array([1e-12]),kd=np.zeros(1),
            target_error=np.zeros(1),effort=np.ones(1),dt=.01,jacobian=np.zeros((0,2)),
            normal_target=np.zeros(0),mu=.65,velocity_limits=np.array([30.]))

    def test_upper_stop_changes_base_through_coupled_momentum(self):
        p=self.free_problem();r=position_limited_velocity_step(**p,position=np.array([.99]),position_limits=np.array([[-1.,1.]]))
        np.testing.assert_allclose(r['velocity'],[.45,1.],atol=1e-11,rtol=0)
        self.assertTrue(r['position_limit_active']);self.assertLess(r['position_limit_impulse'][0],0)
        self.assertLessEqual(.99+.01*r['velocity'][1],1.+1e-12)
        self.assertLess(r['momentum_residual'],1e-10)

    def test_lower_stop_has_opposite_reaction(self):
        p=self.free_problem(-2.);r=position_limited_velocity_step(**p,position=np.array([-.99]),position_limits=np.array([[-1.,1.]]))
        np.testing.assert_allclose(r['velocity'],[-.05,-1.],atol=1e-11,rtol=0)
        self.assertGreater(r['position_limit_impulse'][0],0)

    def test_release_at_upper_stop_preserves_original_path(self):
        p=self.free_problem(-1.);old=speed_limited_velocity_step(**p)
        new=position_limited_velocity_step(**p,position=np.array([1.]),position_limits=np.array([[-1.,1.]]))
        self.assertFalse(new['position_limit_active'])
        for field in ['velocity','impulse','torque','velocity_limit_impulse']:
            np.testing.assert_array_equal(old[field],new[field])

    def test_inactive_position_stop_retains_speed_reaction(self):
        p=self.free_problem(50.);old=speed_limited_velocity_step(**p)
        new=position_limited_velocity_step(**p,position=np.array([0.]),position_limits=np.array([[-2.,2.]]))
        self.assertFalse(new['position_limit_active']);self.assertTrue(new['velocity_limit_active'])
        np.testing.assert_array_equal(old['velocity'],new['velocity'])
        np.testing.assert_array_equal(old['velocity_limit_impulse'],new['joint_limit_impulse'])

    def test_contact_and_stop_reactions_jointly_balance(self):
        mass=np.array([[2.,.1,0.,0.],[.1,2.,0.,0.],[0.,0.,2.,.4],[0.,0.,.4,1.]])
        p=dict(mass=mass,velocity=np.array([.2,-.1,-1.,2.]),bias=np.array([0.,0.,9.81,0.]),
            selection=np.array([[0.],[0.],[0.],[1.]]),kp=np.array([5.]),kd=np.array([.3]),
            target_error=np.array([.01]),effort=np.array([.2]),dt=.01,jacobian=np.eye(4)[:3],
            normal_target=np.zeros(1),mu=.65,velocity_limits=np.array([30.]))
        r=position_limited_velocity_step(**p,position=np.array([.99]),position_limits=np.array([[-1.,1.]]))
        self.assertGreater(r['impulse'][2],0);self.assertGreaterEqual(r['velocity'][2],-2e-7)
        self.assertLessEqual(np.linalg.norm(r['impulse'][:2]),.65*r['impulse'][2]+1e-8)
        self.assertLessEqual(.99+.01*r['velocity'][3],1.+1e-8)
        self.assertLessEqual(abs(r['torque'][0]),.2)
        residual=mass@(r['velocity']-p['velocity'])+.01*p['bias']-.01*p['selection']@r['torque']-p['jacobian'].T@r['impulse']-p['selection']@r['joint_limit_impulse']
        self.assertLess(np.max(abs(residual)),1e-8)

    def test_saturated_motor_is_separate_from_stop(self):
        p=self.free_problem(2.);p.update(kp=np.array([100.]),target_error=np.array([5.]),effort=np.array([.2]))
        r=position_limited_velocity_step(**p,position=np.array([.99]),position_limits=np.array([[-1.,1.]]))
        self.assertAlmostEqual(r['torque'][0],.2)
        self.assertTrue(r['position_limit_active']);self.assertLess(r['momentum_residual'],1e-8)

    def test_incompatible_position_and_speed_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'penetration'):
            joint_velocity_bounds(np.array([1.2]),np.array([[-1.,1.]]),np.array([1.]),.01)


if __name__=='__main__':unittest.main()
