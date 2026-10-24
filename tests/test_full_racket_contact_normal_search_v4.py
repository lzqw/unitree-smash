import unittest
import torch
from quadsmash.control.full_racket_contact_normal_search_v4 import (
    extend_contact_normal_parameters, contact_normal_body)
from quadsmash.control.full_racket_arm_search_v4 import arm_velocity, extend_arm_velocity_parameters
from quadsmash.control.full_racket_recovery_search_v4 import recovery_parameters, extend_recovery_parameters
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names


class ContactNormalSearchTest(unittest.TestCase):
    def setUp(self):
        self.reference=torch.tensor([.99503719,0.,-.09950372])
        self.axes=torch.eye(3).repeat(2,1,1)
        self.old=torch.zeros(2,37)
        self.old[:,7]=.8
        self.old[:,26:32]=torch.arange(6)*.5
        self.old[:,32:37]=torch.tensor([.27,.005,-.27,-.4,.39])

    def normal(self,p,axes=None):
        a=self.axes if axes is None else axes
        return contact_normal_body(p,self.reference,a[:,0],a[:,1],a[:,2])

    def test_zero_extension_preserves_target_exactly_and_does_not_alias(self):
        p=extend_contact_normal_parameters(self.old[0])
        self.assertTrue(torch.equal(p[:37],self.old[0]))
        self.assertEqual(p[37],0.)
        self.assertTrue(torch.equal(self.normal(self.old),self.normal(p[None].repeat(2,1))))
        p[0]=42
        self.assertEqual(self.old[0,0],0.)

    def test_positive_bias_tilts_down_and_preserves_norm_per_lane(self):
        p=torch.cat((self.old,torch.tensor([[.04],[-.04]])),1)
        n=self.normal(p)
        self.assertLess(n[0,2],self.reference[2])
        self.assertGreater(n[1,2],self.reference[2])
        torch.testing.assert_close(n.norm(dim=1),self.reference.norm().expand(2))
        self.assertEqual(n[0,1],0.)

    def test_body_basis_conversion_matches_independent_matrix_product(self):
        p=torch.cat((self.old,torch.tensor([[.04],[-.03]])),1)
        axes=torch.tensor([[[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]],
                           [[.8,0.,.6],[0.,1.,0.],[-.6,0.,.8]]])
        expected=torch.bmm(axes.transpose(1,2),self.normal(p).unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(self.normal(p,axes),expected)

    def test_arm_and_recovery_coordinate_slices_ignore_normal_bias(self):
        p=extend_contact_normal_parameters(self.old[0]);p[37]=.04
        self.assertTrue(torch.equal(arm_velocity(p[None],torch.zeros(6)),arm_velocity(self.old[:1],torch.zeros(6))))
        self.assertTrue(torch.equal(recovery_parameters(p[None],torch.zeros(5)),self.old[:1,32:37]))
        self.assertTrue(torch.equal(extend_arm_velocity_parameters(p,torch.zeros(6)),p))
        self.assertTrue(torch.equal(extend_recovery_parameters(p,torch.zeros(5)),p))

    def test_canonical38_order_and_required_dependency_flags(self):
        names=controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=True,search_contact_normal=True)
        self.assertEqual(names[:37],controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=True))
        self.assertEqual(len(set(names)),38)
        for flags in [dict(search_contact_normal=True),dict(search_arm_velocity=True,search_contact_normal=True)]:
            with self.assertRaises(ValueError):controller_parameter_names(**flags)

    def test_invalid_layout_nonfinite_and_excessive_bias_are_rejected(self):
        for p in [torch.zeros(36),torch.zeros(39),torch.full((37,),float('nan')),
                  torch.cat((self.old[0],torch.tensor([.13]))),torch.cat((self.old[0],torch.tensor([-.13])))]:
            with self.assertRaises(ValueError):extend_contact_normal_parameters(p)
        with self.assertRaises(ValueError):self.normal(torch.zeros(2,39))


if __name__=='__main__':unittest.main()
