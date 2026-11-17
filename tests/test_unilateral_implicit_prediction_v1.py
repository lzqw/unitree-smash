import unittest
import numpy as np
from scripts.unilateral_implicit_prediction_v1 import unilateral_velocity_step


class ContactStepTests(unittest.TestCase):
    def particle(self, velocity, mu=.5):
        return unilateral_velocity_step(np.eye(3),np.asarray(velocity,dtype=float),
            np.zeros(3),np.zeros((3,0)),*[np.zeros(0) for _ in range(4)],
            .01,np.eye(3)[None],np.zeros((1,3)),mu)

    def test_separating_contact_cannot_pull_down(self):
        r=self.particle([0,0,1])
        np.testing.assert_allclose(r['velocity'],[0,0,1],atol=1e-12)
        np.testing.assert_allclose(r['force'],0,atol=1e-12)
        self.assertEqual(r['modes'].tolist(),[0])

    def test_inelastic_normal_support(self):
        r=self.particle([0,0,-1])
        np.testing.assert_allclose(r['velocity'],0,atol=1e-12)
        np.testing.assert_allclose(r['force'],[[0,0,100]],atol=1e-10)

    def test_sticking_inside_friction_cone(self):
        r=self.particle([.1,0,-1])
        np.testing.assert_allclose(r['velocity'],0,atol=1e-12)
        np.testing.assert_allclose(r['force'],[[-10,0,100]],atol=1e-10)
        self.assertEqual(r['modes'].tolist(),[1])

    def test_sliding_retains_velocity_and_dissipates(self):
        for angle in [0.,.4,1.8]:
            tangent=np.array([np.cos(angle),np.sin(angle)])
            r=self.particle([*tangent,-1])
            np.testing.assert_allclose(r['velocity'],[* (.5*tangent),0],atol=1e-10)
            np.testing.assert_allclose(r['force'],[[-50*tangent[0],-50*tangent[1],100]],atol=1e-8)
            self.assertLess(float(r['force'][0,:2]@r['velocity'][:2]),0.)
            self.assertEqual(r['modes'].tolist(),[2])

    def test_free_motor_saturation_changes_whole_solution(self):
        r=unilateral_velocity_step(np.eye(1),np.zeros(1),np.zeros(1),np.eye(1),
            np.array([100.]),np.array([1.]),np.array([5.]),np.array([2.]),
            .01,np.zeros((0,3,1)),np.zeros((0,3)),.5)
        np.testing.assert_allclose(r['velocity'],[.02],atol=1e-12)
        np.testing.assert_allclose(r['torque'],[2.],atol=1e-12)
        self.assertLess(r['residual'],1e-12)

    def test_exhausted_active_set_is_failure(self):
        with self.assertRaisesRegex(ValueError,'did not converge'):
            unilateral_velocity_step(np.eye(3),np.array([1.,0.,-1.]),np.zeros(3),
                np.zeros((3,0)),*[np.zeros(0) for _ in range(4)],.01,
                np.eye(3)[None],np.zeros((1,3)),.5,max_iterations=1)


if __name__=='__main__':unittest.main()
