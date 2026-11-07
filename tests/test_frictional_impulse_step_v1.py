import unittest
import numpy as np
from scripts.frictional_impulse_step_v1 import coulomb_impulses,frictional_velocity_step,metric_disk_minimum


class CoulombMechanics(unittest.TestCase):
    def test_inelastic_sliding_particle(self):
        impulse,info=coulomb_impulses(np.eye(3),np.array([2.,0.,-1.]),.65)
        np.testing.assert_allclose(impulse,[-.65,0.,1.],atol=1e-10)
        velocity=np.array([2.,0.,-1.])+impulse
        self.assertLess(np.linalg.norm(velocity),np.sqrt(5.))
        self.assertLess(max(info['residual'].values()),2e-7)

    def test_sticking_particle(self):
        impulse,_=coulomb_impulses(np.eye(3),np.array([.1,-.2,-1.]),.65)
        np.testing.assert_allclose(impulse,[-.1,.2,1.],atol=1e-10)

    def test_separation_releases_warm_contact(self):
        impulse,_=coulomb_impulses(np.eye(3),np.array([2.,0.,1.]),.65,warm=np.array([-.65,0.,1.]))
        np.testing.assert_array_equal(impulse,np.zeros(3))

    def test_normal_motion_can_reach_but_not_cross_plane(self):
        # A gap of 1 cm and a 10 ms step permit -1 m/s at the endpoint.
        # Free -2 m/s should receive only 1 Ns, not stop immediately at 0.
        impulse,_=coulomb_impulses(np.eye(3),np.array([0.,0.,-2.-(-1.)]),.65)
        self.assertAlmostEqual(impulse[2],1.)

    def test_anisotropic_disk_projection_has_kkt_solution(self):
        H=np.diag([2.,5.]);b=np.array([3.,4.]);r=.7
        p=metric_disk_minimum(H,b,r)
        self.assertAlmostEqual(np.linalg.norm(p),r,places=10)
        gradient=H@p+b;multiplier=-gradient@p/(r*r)
        self.assertGreater(multiplier,0.)
        np.testing.assert_allclose(gradient+multiplier*p,0.,atol=1e-10)

    def test_impulse_and_pd_saturation_obey_whole_mass_balance(self):
        M=np.array([[2.,0.,.2],[0.,1.,0.],[.2,0.,1.]])
        initial=np.array([2.,0.,-1.]);S=np.array([[1.],[0.],[0.]])
        result=frictional_velocity_step(M,initial,np.zeros(3),S,np.array([80.]),np.array([4.]),
            np.array([10.]),np.array([1.]),.01,np.eye(3),np.array([0.]),.65)
        self.assertAlmostEqual(result['torque'][0],1.)
        np.testing.assert_allclose(M@(result['velocity']-initial),
            .01*S@result['torque']+result['impulse'],atol=1e-10)
        self.assertLessEqual(np.linalg.norm(result['impulse'][:2]),.65*result['impulse'][2]+1e-10)

    def test_nonconvergence_is_not_a_valid_prediction(self):
        W=np.array([[1.,.2,.3],[.2,1.,.2],[.3,.2,1.]])
        with self.assertRaisesRegex(ValueError,'did not converge'):
            coulomb_impulses(W,np.array([2.,1.,-1.]),.65,max_iterations=1)


if __name__=='__main__':unittest.main()
