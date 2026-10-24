"""Geometry checks only; native rule fixtures remain separate evidence."""
import unittest
import numpy as np
from scipy.optimize import minimize_scalar,minimize,linprog,LinearConstraint
from quadsmash.physics.racket_plane_geometry_v4 import (
    segment_box_distance_squared,cylinder_box_overlap,oriented_box_overlap,RacketPlaneGeometry)
from quadsmash.physics.full_racket_v4 import load_racket
from quadsmash.physics.court_v4 import load_court


class RacketVolumeGeometryTest(unittest.TestCase):
    def test_segment_box_piecewise_minimum_matches_independent_scalar_optimizer(self):
        rng=np.random.default_rng(725701)
        for _ in range(100):
            start,end=rng.normal(size=(2,3));lower=rng.normal(size=3);upper=lower+rng.uniform(.01,1,3)
            value,_,_=segment_box_distance_squared(start,end,lower,upper)
            def distance(t):
                p=start+t*(end-start);return float(np.sum((p-np.clip(p,lower,upper))**2))
            result=minimize_scalar(distance,bounds=(0.,1.),method='bounded',options={'xatol':1e-14})
            expected=min(result.fun,distance(0.),distance(1.))
            self.assertAlmostEqual(value,expected,places=11)

    def test_finite_cylinder_does_not_gain_capsule_hemispheres(self):
        center=np.zeros(3);axis=np.array([1.,0,0])
        self.assertFalse(cylinder_box_overlap(center,axis,.5,.1,
            np.array([.55,-.01,-.01]),np.array([.60,.01,.01])))
        self.assertTrue(cylinder_box_overlap(center,axis,.5,.1,
            np.array([.45,.08,0.]),np.array([.49,.09,.01])))

    def test_cylinder_intersection_matches_independent_convex_optimization(self):
        rng=np.random.default_rng(725702)
        for _ in range(50):
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            center=rng.uniform(-.15,.15,3);lower=center+rng.uniform(-.25,.20,3);upper=lower+rng.uniform(.02,.12,3)
            actual=cylinder_box_overlap(center,axis,.2,.06,lower,upper)
            initial=linprog(np.zeros(3),A_ub=np.array([axis,-axis]),
                b_ub=np.array([.2+axis@center,.2-axis@center]),bounds=list(zip(lower,upper)),method='highs')
            if not initial.success:
                self.assertFalse(actual);continue
            matrix=np.eye(3)-np.outer(axis,axis)
            solve=minimize(lambda x:(x-center)@matrix@(x-center),initial.x,
                jac=lambda x:2*matrix@(x-center),bounds=list(zip(lower,upper)),
                constraints=LinearConstraint(axis[None],axis@center-.2,axis@center+.2),
                method='SLSQP',options={'ftol':1e-13,'maxiter':200})
            self.assertTrue(solve.success)
            if abs(solve.fun-.06**2)>1e-9:self.assertEqual(actual,bool(solve.fun<=.06**2))

    def test_rotated_box_separating_axis_rejects_empty_corner_overlap(self):
        angle=np.pi/4;rotation=np.array([[np.cos(angle),-np.sin(angle),0],
            [np.sin(angle),np.cos(angle),0],[0,0,1]])
        self.assertFalse(oriented_box_overlap(np.zeros(3),rotation,np.array([1.,.01,.01]),
            np.array([.7,-.7,-.01]),np.array([.71,-.69,.01])))
        self.assertTrue(oriented_box_overlap(np.zeros(3),rotation,np.array([1.,.01,.01]),
            np.array([.6,.6,-.01]),np.array([.61,.61,.01])))

    def test_live_pose_geometry_distinguishes_actual_invasion_from_empty_aabb(self):
        model=RacketPlaneGeometry(load_racket(),load_court())
        quaternion=np.array([np.sqrt(.5),0.,.5,-.5])
        self.assertIsNone(model.first_above_net_overlap([1.379146,0.,3.],quaternion))
        self.assertIsNotNone(model.first_above_net_overlap([1.395146,0.,3.],quaternion))
        self.assertIsNone(model.first_above_net_overlap([1.395146,0.,.5],quaternion))
