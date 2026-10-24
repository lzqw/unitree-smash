import unittest
import torch
from quadsmash.control.full_racket_recovery_blend_v4 import LegRecoveryBlend


class RecoveryBlendTest(unittest.TestCase):
    def test_disabled_is_exact_original_standing_action(self):
        blend=LegRecoveryBlend(2,'cpu',0.)
        previous=torch.randn(2,18);standing=torch.randn(2,18)
        self.assertIs(blend.action(torch.zeros(2),torch.ones(2,dtype=torch.bool),previous,standing),standing)

    def test_transition_starts_at_observed_previous_legs_and_preserves_arm(self):
        blend=LegRecoveryBlend(1,'cpu',.1)
        previous=torch.full((1,18),-.6);standing=torch.full((1,18),.7)
        output=blend.action(torch.tensor([3.]),torch.tensor([True]),previous,standing)
        torch.testing.assert_close(output[:,:12],previous[:,:12],rtol=0,atol=0)
        torch.testing.assert_close(output[:,12:],standing[:,12:],rtol=0,atol=0)

    def test_smooth_handover_reaches_current_feedback_target(self):
        blend=LegRecoveryBlend(1,'cpu',.1);active=torch.tensor([True])
        previous=torch.full((1,18),-.5);target=torch.full((1,18),.5)
        blend.action(torch.tensor([0.]),active,previous,target)
        early=blend.action(torch.tensor([.001]),active,previous,target)
        self.assertLess(float((early[:,:12]-previous[:,:12]).abs().max()),.001)
        middle=blend.action(torch.tensor([.05]),active,previous,target)
        torch.testing.assert_close(middle[:,:12],torch.zeros(1,12),rtol=0,atol=1e-7)
        later=blend.action(torch.tensor([.101]),active,previous,target*.8)
        torch.testing.assert_close(later,target*.8,rtol=0,atol=1e-7)

    def test_reactivation_and_partial_reset_use_fresh_public_action(self):
        blend=LegRecoveryBlend(2,'cpu',.1);active=torch.tensor([True,True])
        prior=torch.zeros(2,18);target=torch.ones(2,18)
        blend.action(torch.zeros(2),active,prior,target)
        blend.reset(torch.tensor([1]));new=torch.full((2,18),-.4)
        output=blend.action(torch.full((2,),.2),active,new,target)
        torch.testing.assert_close(output[0,:12],target[0,:12])
        torch.testing.assert_close(output[1,:12],new[1,:12])
        blend.action(torch.full((2,),.3),torch.zeros(2,dtype=torch.bool),new,target)
        output=blend.action(torch.full((2,),.4),active,new,target)
        torch.testing.assert_close(output[:,:12],new[:,:12])

    def test_invalid_blend_duration_is_rejected(self):
        for duration in [-.01,.26,float('nan')]:
            with self.assertRaises(ValueError):LegRecoveryBlend(1,'cpu',duration)


if __name__=='__main__':unittest.main()
