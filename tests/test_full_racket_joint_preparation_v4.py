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
                       {'iterations':0},{'maximum_joint_delta_rad':4},{'damping':0},
                       {'anticipate_measured_motion':1},{'maximum_predictive_margin_m':1}]:
            with self.assertRaises(ValueError):validate_joint_preparation(config)
        for config in [{'post_stroke_maximum_x_m':1.4},{'post_stroke_maximum_x_m':1.0},
                       {'post_stroke_lookahead_s':.01},{'post_stroke_lookahead_s':.2}]:
            with self.assertRaises(ValueError):validate_joint_preparation(config)

    def test_tighter_retreat_changes_only_post_stroke_lanes_and_keeps_native_joint_limits(self):
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64).repeat(2,1)
        model=FullRacketKinematicsV4();limits=torch.tensor(FullRacketGeometryV4().bounds)
        obs=observation(2);obs[:,760]=.35
        config=dict(protect_post_hit_net_volume=True,maximum_joint_delta_rad=.8,iterations=8)
        baseline=FullRacketJointPreparation(config)
        guard=FullRacketJointPreparation(dict(config,post_stroke_maximum_x_m=1.15,post_stroke_lookahead_s=.12))
        mask=torch.tensor([False,True])
        old=baseline(q,obs,limits,model)
        new=guard(q,obs,limits,model,post_stroke_mask=mask)
        self.assertTrue(torch.equal(old[0],new[0]))
        new_x=float(guard.support_state(new[1:],obs[1:],model)[0])
        self.assertLess(new_x,float(baseline.support_state(old[1:],obs[1:],model)[0]))
        self.assertLessEqual(new_x,1.151)
        self.assertTrue(bool((new>=limits[:,0]).all() and (new<=limits[:,1]).all()))
        self.assertLessEqual(float((new-q).abs().max()),.800001)

    def test_default_post_stroke_mask_preserves_legacy_commands_exactly(self):
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64)
        model=FullRacketKinematicsV4();limits=torch.tensor(FullRacketGeometryV4().bounds);obs=observation(1)
        guard=FullRacketJointPreparation({'anticipate_measured_motion':True,'protect_post_hit_net_volume':True})
        obs[:,45:48]=model(q)[0]
        self.assertTrue(torch.equal(guard(q,obs,limits,model,q),guard(q,obs,limits,model,q,post_stroke_mask=torch.tensor([True]))))

    def test_measured_tool_translation_is_anticipated_with_stationary_base(self):
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64)
        model=FullRacketKinematicsV4();limits=torch.tensor(FullRacketGeometryV4().bounds)
        obs=observation(1);obs[:,760]=.35;obs[:,45:48]=model(q)[0];obs[:,51]=4.
        baseline=FullRacketJointPreparation({});guard=FullRacketJointPreparation({'anticipate_measured_motion':True,'maximum_predictive_margin_m':.25})
        self.assertTrue(torch.equal(baseline(q,obs,limits,model),q))
        before=guard.support_state(q,obs,model)[0]
        corrected=guard(q,obs,limits,model,q)
        self.assertLess(float(guard.support_state(corrected,obs,model)[0]),float(before)-.02)
        with self.assertRaises(ValueError):guard(q,obs,limits,model)

    def test_measured_tool_rotation_contributes_to_forward_support_velocity(self):
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64)
        model=FullRacketKinematicsV4();limits=torch.tensor(FullRacketGeometryV4().bounds)
        obs=observation(1);obs[:,760]=1.1;obs[:,45:48]=model(q)[0]
        guard=FullRacketJointPreparation({'anticipate_measured_motion':True})
        x,_,_,_,point=guard.support_state(q,obs,model)
        obs[:,760]+=x-1.36
        offset=point-obs[:,45:48];axis=torch.linalg.cross(offset,obs[:,763:766]);axis/=torch.linalg.vector_norm(axis,dim=1,keepdim=True)
        obs[:,54:57]=4.*axis
        corrected=guard(q,obs,limits,model,q)
        self.assertLess(float(guard.support_state(corrected,obs,model)[0]),1.355)

    def test_observed_approach_retracts_a_command_already_on_own_side(self):
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64)
        model=FullRacketKinematicsV4();limits=torch.tensor(FullRacketGeometryV4().bounds)
        guard=FullRacketJointPreparation({'anticipate_measured_motion':True})
        obs=observation(1);obs[:,760]=1.1
        x,g,*_=guard.support_state(q,obs,model);obs[:,760]+=x-1.30
        actual=(q+.25*g/torch.linalg.vector_norm(g,dim=1,keepdim=True)).clamp(limits[:,0],limits[:,1])
        obs[:,45:48]=model(actual)[0]
        self.assertGreater(float(guard.support_state(actual,obs,model,observed_center=True)[0]),1.38)
        corrected=guard(q,obs,limits,model,actual)
        self.assertLess(float(guard.support_state(corrected,obs,model)[0]),1.29)

    def test_above_net_followthrough_is_allowed_but_low_racket_is_retracted(self):
        q=torch.tensor([[-.8027512395,1.4135549941,-.8117888650,1.5911344936,-.3822701438,.3414782819]],dtype=torch.float64)
        model=FullRacketKinematicsV4();guard=FullRacketJointPreparation({'protect_post_hit_net_volume':True})
        center=model(q)[0];obs=observation(1);obs[:,760]=1.1+center[:,0]-1.48;obs[:,762]=1.8+center[:,2]-1.80
        obs[:,45:48]=center;obs[:,773]=1
        full=guard.support_state(q,obs,model)[0]
        high=guard.support_state(q,obs,model,torch.tensor([True]))[0]
        self.assertGreater(float(full),1.5);self.assertLess(float(high),1.38)
        obs[:,762]+=.6
        low=guard.support_state(q,obs,model,torch.tensor([True]))[0]
        torch.testing.assert_close(low,full)
        corrected=guard(q,obs,torch.tensor(FullRacketGeometryV4().bounds),model)
        self.assertLess(float(guard.support_state(corrected,obs,model,torch.tensor([True]))[0]),float(low))

    def test_descending_post_hit_components_are_not_exempted_from_net_planning(self):
        q=torch.tensor([[-.8027512395,1.4135549941,-.8117888650,1.5911344936,-.3822701438,.3414782819]],dtype=torch.float64)
        model=FullRacketKinematicsV4();guard=FullRacketJointPreparation({'protect_post_hit_net_volume':True})
        center=model(q)[0];obs=observation(1);obs[:,760]=1.1+center[:,0]-1.48;obs[:,762]=1.8+center[:,2]-1.72
        obs[:,45:48]=center;obs[:,773]=1
        high=guard.support_state(q,obs,model,torch.tensor([True]))[0]
        obs[:,53]=-4.
        descending=guard.support_state(q,obs,model,torch.tensor([True]))[0]
        self.assertGreater(float(descending),float(high)+.02)

    def test_larger_projection_solves_far_forward_target_with_same_physical_joint_limits(self):
        q=torch.tensor([[-.6153451338,1.7700886461,-2.2884081663,.8714186655,.8830834721,2.0943899999]],dtype=torch.float64)
        model=FullRacketKinematicsV4();limits=torch.tensor(FullRacketGeometryV4().bounds)
        obs=observation(1);obs[:,760]=-.25
        original=FullRacketJointPreparation({})
        larger=FullRacketJointPreparation({'maximum_joint_delta_rad':.8,'iterations':8})
        before=original.support_state(q,obs,model)[0]
        small=original(q,obs,limits,model);large=larger(q,obs,limits,model)
        self.assertGreater(float(before),1.7)
        self.assertLess(float(larger.support_state(large,obs,model)[0]),float(original.support_state(small,obs,model)[0]))
        self.assertLessEqual(float(larger.support_state(large,obs,model)[0]),1.382)
        self.assertTrue(bool((large>=limits[:,0]).all() and (large<=limits[:,1]).all()))
        self.assertLessEqual(float((large-q).abs().max()),.800001)


if __name__=='__main__':unittest.main()
