import unittest
import numpy as np
import torch

from quadsmash.control.full_racket_arm_search_v4 import (
    extend_arm_velocity_parameters,arm_velocity,arm_velocity_grid,interception_velocity_grid,
    body_crossovers_preserving_velocity,causal_prepose_delta,normal_preserving_position_inverse,bounded_prepose_step)
from quadsmash.control.full_racket_contact_curriculum_v4 import measured_recovery_leg_crossovers


class ArmSearchTest(unittest.TestCase):
    def test_extension_preserves_all_original_parameters_and_storage(self):
        original=torch.arange(26,dtype=torch.float32)/100
        velocity=torch.tensor([1.,-2.,3.,-4.,5.,0.])
        extended=extend_arm_velocity_parameters(original,velocity)
        self.assertTrue(torch.equal(original,extended[:26]))
        self.assertTrue(torch.equal(velocity,extended[26:]))
        again=extend_arm_velocity_parameters(extended,torch.ones(6))
        self.assertTrue(torch.equal(again,extended))
        again[0]=99.
        self.assertNotEqual(float(extended[0]),99.)

    def test_velocity_coordinates_match_legacy_and_allow_independent_lanes(self):
        params=torch.zeros((2,26));params[:,7]=torch.tensor([.8,1.2])
        seed=torch.tensor([4.8,4.8,-1.2,3.3,4.2,0.])
        extended=torch.cat((params,seed[None].expand(2,-1)),1)
        self.assertTrue(torch.equal(arm_velocity(params,seed),arm_velocity(extended,seed)))
        extended[1,28]=-4.
        self.assertTrue(torch.equal(arm_velocity(params,seed)[0],arm_velocity(extended,seed)[0]))
        self.assertNotEqual(float(arm_velocity(params,seed)[1,2]),float(arm_velocity(extended,seed)[1,2]))

    def test_invalid_velocity_layout_or_range_is_rejected(self):
        for width in [25,27,31,33]:
            with self.assertRaises(ValueError):extend_arm_velocity_parameters(torch.zeros(width),torch.zeros(6))
        for velocity in [[0.,0.,0.,0.,0.,5.01],[0.,0.,0.,float('nan'),0.,0.]]:
            with self.assertRaises(ValueError):extend_arm_velocity_parameters(torch.zeros(26),velocity)
        x=torch.zeros(32);x[-1]=-5.01
        with self.assertRaises(ValueError):extend_arm_velocity_parameters(x,torch.zeros(6))
        with self.assertRaises(ValueError):arm_velocity(torch.zeros((1,30)),torch.zeros(6))

    def test_velocity_grid_preserves_body_pose_and_velocity_bank(self):
        reference=np.linspace(-.1,.1,32);bank=np.array([[1.,2.,3.,4.,5.,0.],[-1.,-2.,-3.,-4.,-5.,0.],
                                                           [0.,1.,0.,-1.,0.,0.],[0.,0.,0.,0.,0.,0.]])
        grid=arm_velocity_grid(reference,bank,np.full(32,-5.),np.full(32,5.))
        self.assertEqual(grid.shape,(12,32))
        fixed=[i for i in range(26) if i!=14]
        np.testing.assert_array_equal(grid[:,fixed],np.broadcast_to(reference[fixed],(12,25)))
        np.testing.assert_array_equal(grid[:,26:],np.repeat(bank,3,axis=0))

    def test_body_crossover_is_legacy_exact_and_preserves_six_velocity_coordinates(self):
        reference=np.linspace(-.1,.1,32);recovered=np.linspace(.2,-.2,26)
        low=np.full(32,-5.);high=np.full(32,5.)
        expected=measured_recovery_leg_crossovers(reference[:26],recovered,low[:26],high[:26])
        result=body_crossovers_preserving_velocity(reference,recovered,low,high)
        np.testing.assert_array_equal(result[:,:26],expected)
        np.testing.assert_array_equal(result[:,26:],np.broadcast_to(reference[26:],(8,6)))
        np.testing.assert_array_equal(body_crossovers_preserving_velocity(reference[:26],recovered,low[:26],high[:26]),expected)

    def test_interception_grid_covers_planes_without_changing_other_control_parameters(self):
        reference=np.linspace(-.1,.1,32)
        bank=np.array([[1.,2.,3.,4.,5.,0.],[-1.,-2.,-3.,-4.,-5.,0.],
                       [0.,1.,0.,-1.,0.,0.],[0.,0.,0.,0.,0.,0.]])
        low,high=np.full(32,-5.),np.full(32,5.)
        grid=interception_velocity_grid(reference,bank,low,high)
        self.assertEqual(grid.shape,(36,32))
        np.testing.assert_array_equal(grid[:,24],np.repeat([1.10,1.25,1.40],12))
        fixed=[i for i in range(26) if i not in (14,24)]
        np.testing.assert_array_equal(grid[:,fixed],np.broadcast_to(reference[fixed],(36,24)))
        np.testing.assert_array_equal(grid[:,26:],np.tile(np.repeat(bank,3,axis=0),(3,1)))
        high[24]=1.3
        clipped=interception_velocity_grid(reference,bank,low,high)
        self.assertLessEqual(clipped[:,24].max(),1.3)


class CausalPreposeTest(unittest.TestCase):
    def setUp(self):
        self.public=np.zeros((2,794));self.public[:,8]=-1.;self.public[:,763]=1.
        self.position=np.broadcast_to(np.c_[np.eye(3),np.zeros((3,3))],(2,3,6)).copy()
        self.normal=np.zeros((2,3,6));self.lower=np.full((2,6),-.3);self.upper=np.full((2,6),.3)

    def solve(self,targets,public,valid,gain):
        return causal_prepose_delta(targets,public,self.position,self.normal,valid,gain,self.lower,self.upper)

    def test_world_delta_is_converted_into_observed_body_axes(self):
        self.public[1,763:766]=[0.,-1.,0.]
        result=self.solve(np.array([[.1,0.,0.],[.1,0.,0.]]),self.public,[True,True],1.)
        amplitude=.1/(1+.04**2)
        np.testing.assert_allclose(result[0],[amplitude,0.,0.,0.,0.,0.])
        np.testing.assert_allclose(result[1],[0.,-amplitude,0.,0.,0.,0.])

    def test_prepose_inverse_limits_face_rotation_while_moving_the_contact_point(self):
        position=np.c_[np.eye(3),np.eye(3)]
        normal=np.c_[np.zeros((3,3)),np.eye(3)]
        delta=np.array([.1,-.1,.1])
        plain=position.T@np.linalg.solve(position@position.T+.04**2*np.eye(3),delta)
        oriented=normal_preserving_position_inverse(position,normal)@delta
        self.assertLess(np.linalg.norm(normal@oriented),.05*np.linalg.norm(normal@plain))
        self.assertLess(np.linalg.norm(position@oriented-delta),.001)

    def test_invalid_forecasts_and_disabled_gain_produce_zero_correction(self):
        delta=np.array([[float('nan'),0.,0.],[1.,0.,0.]])
        np.testing.assert_array_equal(self.solve(delta,self.public,[True,False],1.),np.zeros((2,6)))
        np.testing.assert_array_equal(self.solve(delta,self.public,[True,True],0.),np.zeros((2,6)))
        self.public[:,763:766]=0.
        np.testing.assert_array_equal(self.solve(np.ones((2,3)),self.public,[True,True],1.),np.zeros((2,6)))

    def test_correction_is_bounded_and_unused_observations_cannot_change_it(self):
        targets=np.array([[.8,-.7,.6],[.2,.1,-.1]])
        first=self.solve(targets,self.public,[True,True],.5)
        self.assertLessEqual(np.abs(first).max(),.30)
        changed=np.full((2,794),np.nan);changed[:,6:9]=self.public[:,6:9];changed[:,763:766]=self.public[:,763:766]
        np.testing.assert_array_equal(first,self.solve(targets,changed,[True,True],.5))
        with self.assertRaises(ValueError):self.solve(targets,self.public,[True,True],1.01)

    def test_bounded_solve_reallocates_motion_away_from_a_joint_at_its_limit(self):
        position=np.c_[np.eye(3),np.eye(3)]
        lower=np.full(6,-.3);upper=np.full(6,.3);upper[3]=0.
        delta=bounded_prepose_step(np.array([.1,0.,0.]),position,np.zeros((3,6)),lower,upper)
        self.assertLessEqual(delta[3],1e-12)
        self.assertGreater(delta[0],.09)
        self.assertLess(np.linalg.norm(position@delta-[.1,0.,0.]),.001)


if __name__=='__main__':unittest.main()
