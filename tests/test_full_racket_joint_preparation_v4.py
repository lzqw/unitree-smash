import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from quadsmash.control.full_racket_joint_preparation_v4 import FullRacketJointPreparation,validate_joint_preparation
from quadsmash.control.full_racket_kinematics_v4 import FullRacketKinematicsV4
from quadsmash.control.full_racket_geometry_v4 import FullRacketGeometryV4
from quadsmash.physics.full_racket_v4 import load_racket,RacketPlaneSupport


def observation(count):
    obs=torch.zeros((count,794),dtype=torch.float64)
    obs[:,8]=-1;obs[:,763]=1;obs[:,760:763]=torch.tensor([.05,0.,1.2])
    return obs


class JointPreparationTest(unittest.TestCase):
    def test_support_matches_independent_native_primitive_support_including_clamp(self):
        racket=load_racket();guard=FullRacketJointPreparation({})
        rotation=Rotation.random(64,random_state=724301)
        direction=torch.tensor(rotation.as_matrix()[:,0,:])
        xyzw=rotation.as_quat();wxyz=torch.tensor(xyzw[:,[3,0,1,2]])
        old=RacketPlaneSupport(racket,dtype=torch.float64).maximum_x(torch.zeros((64,3),dtype=torch.float64),wxyz)
        clamp=racket['assumptions'];center=torch.tensor(clamp['clamp_center_in_body_m'])
        half=torch.tensor(clamp['clamp_size_m'])/2
        expected=torch.maximum(old,(direction*center).sum(1)+(direction.abs()*half).sum(1))
        actual=(guard.local_support(direction)*direction).sum(1)
        torch.testing.assert_close(actual,expected,atol=2e-9,rtol=2e-7)

    def test_support_gradient_matches_independent_joint_finite_differences(self):
        guard=FullRacketJointPreparation({});model=FullRacketKinematicsV4();geometry=FullRacketGeometryV4()
        q=torch.tensor(np.random.default_rng(724302).uniform(geometry.bounds[:,0]+.05,geometry.bounds[:,1]-.05,(12,6)))
        obs=observation(len(q));_,gradient,*_=guard.support_state(q,obs,model)
        for j in range(6):
            step=torch.zeros_like(q);step[:,j]=1e-6
            plus=guard.support_state(q+step,obs,model)[0];minus=guard.support_state(q-step,obs,model)[0]
            torch.testing.assert_close(gradient[:,j],(plus-minus)/2e-6,atol=2e-7,rtol=2e-6)

    def test_minimal_command_correction_reduces_unsafe_support_within_joint_limits(self):
        guard=FullRacketJointPreparation({});model=FullRacketKinematicsV4();geometry=FullRacketGeometryV4()
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64)
        obs=observation(1);limits=torch.tensor(geometry.bounds)
        before=guard.support_state(q,obs,model)[0]
        corrected=guard(q,obs,limits,model);after=guard.support_state(corrected,obs,model)[0]
        self.assertGreater(float(before),1.40)
        self.assertLessEqual(float(after),1.382)
        self.assertTrue(bool((corrected>=limits[:,0]).all() and (corrected<=limits[:,1]).all()))
        self.assertLessEqual(float((corrected-q).abs().max()),.400001)

    def test_safe_preparation_and_observed_followthrough_preserve_the_original_command(self):
        guard=FullRacketJointPreparation({});model=FullRacketKinematicsV4();geometry=FullRacketGeometryV4()
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64)
        limits=torch.tensor(geometry.bounds);obs=observation(1)
        obs[:,760]=1.1
        self.assertTrue(torch.equal(guard(q,obs,limits,model),q))
        obs[:,760]=.05;obs[:,773]=1
        self.assertTrue(torch.equal(guard(q,obs,limits,model),q))
        obs[:,68:760]=torch.randn((1,692),generator=torch.Generator().manual_seed(724303))
        self.assertTrue(torch.equal(guard(q,obs,limits,model),q))

    def test_invalid_configuration_is_rejected(self):
        self.assertIsNone(validate_joint_preparation(None))
        for config in [{'maximum_x_m':1.5},{'lookahead_s':float('nan')},{'iterations':3.5},
                       {'iterations':0},{'maximum_joint_delta_rad':4},{'damping':0}]:
            with self.assertRaises(ValueError):validate_joint_preparation(config)


if __name__=='__main__':unittest.main()
