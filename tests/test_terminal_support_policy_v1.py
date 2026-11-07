"""Command semantics on a synthetic static fixture, not a native recovery test."""
import copy
import unittest
import numpy as np
from tests import test_frictional_static_reference_v1 as fixture_module
from scripts.frictional_static_reference_v1 import sphere_static_reference
from scripts.terminal_support_policy_v1 import TerminalSupportPolicyV1


class TerminalPolicy(unittest.TestCase):
    def setUp(self):
        fixture=fixture_module.StaticSurfaceReference();fixture.setUp();self.f=fixture
        fixture.interface.arm_rate=10.  # Original command-history rate from the native controller config.
        reference=sphere_static_reference(fixture.controller,fixture.state,fixture.data['spheres'])
        self.target=np.asarray(reference['target']);state=fixture.state
        self.warm=dict(horizon_steps=80,final=dict(accepted=True,commands=[self.target.tolist()]*80,
            states=[dict(q=state.q.tolist(),v=state.v.tolist())]*80))
        self.tail=dict(failure=None,valid_steps=260,model_window=dict(longest_model_kinematic_window_s=1.),
            mode='fixed_preload_leg_feedback_and_hip_reference',initial_static_reference=reference,
            posture_gain=2.,velocity_gain=.5,hip_bias_rad=-.18)
        self.policy=TerminalSupportPolicyV1(fixture.model,fixture.interface,state,self.warm,self.tail)

    def test_matching_nominal_state_preserves_prefix_command(self):
        target,mode=self.policy.action(12,self.f.state,self.target)
        np.testing.assert_allclose(target,self.target,atol=1e-12,rtol=0)
        self.assertEqual(mode,'nominal_prefix_with_leg_feedback')

    def test_measured_leg_error_changes_command_without_arm_override(self):
        state=copy.deepcopy(self.f.state);state.joint_position[0]+=.001
        state.joint_velocity[0]+=.01
        target,_=self.policy.action(12,state,self.target)
        self.assertAlmostEqual(target[0]-self.target[0],-.002-.5*.01/25.)
        np.testing.assert_array_equal(target[12:],self.target[12:])

    def test_terminal_feedback_persists_to_original_end_and_obeys_history(self):
        prior=self.target.copy();prior[12]-=.3
        target,mode=self.policy.action(339,self.f.state,prior)
        self.assertEqual(mode,'persistent_terminal_leg_feedback')
        self.assertLessEqual(max(abs(target[12:]-prior[12:])),.100000001)
        self.assertTrue(np.all(target>=self.f.interface.limits[:,0]))
        self.assertTrue(np.all(target<=self.f.interface.limits[:,1]))
        with self.assertRaises(ValueError):self.policy.action(340,self.f.state,prior)

    def test_short_or_failed_tail_is_not_admitted(self):
        tail=copy.deepcopy(self.tail);tail['valid_steps']=159
        with self.assertRaises(ValueError):TerminalSupportPolicyV1(self.f.model,self.f.interface,self.f.state,self.warm,tail)


if __name__=='__main__':unittest.main()
