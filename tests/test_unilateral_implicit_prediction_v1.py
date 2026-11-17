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


def test_authored_joint_stop_couples_reaction_without_clipping_or_motor_limit_relaxation():
    M=np.array([[2.,.5],[.5,1.]])
    S=np.array([[0.],[1.]])
    result=unilateral_velocity_step(M,np.array([0.,1.]),np.zeros(2),S,
        np.ones(1),np.zeros(1),np.zeros(1),np.ones(1)*2,.01,
        np.zeros((0,3,2)),np.zeros((0,3)),.5,
        joint_velocity_bounds=np.array([[-100.,0.]]))
    np.testing.assert_allclose(result['velocity'],[.25,0.],atol=1e-12)
    assert abs(result['torque'][0])<=2
    assert result['residual']<1e-12
    assert result['joint_stop_reaction_generalized'][1]<0
    assert np.sum(result['joint_stop_active'])==1


def test_free_motion_inside_joint_stops_is_identical_to_original_solver():
    args=(np.eye(1),np.array([.2]),np.zeros(1),np.eye(1),np.ones(1),
          np.ones(1),np.zeros(1),np.ones(1)*2,.01,np.zeros((0,3,1)),np.zeros((0,3)),.5)
    original=unilateral_velocity_step(*args)
    bounded=unilateral_velocity_step(*args,joint_velocity_bounds=np.array([[-1.,1.]]))
    np.testing.assert_array_equal(original['velocity'],bounded['velocity'])
    assert not bounded['joint_stop_active'].any()
