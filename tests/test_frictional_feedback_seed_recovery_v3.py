"""Warm-start rejection semantics; these mocks do not establish physical feasibility."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts.frictional_feedback_seed_recovery_v3 import FrictionalFeedbackSeedRecoveryV3


class FeedbackWarmStart(unittest.TestCase):
    def setUp(self):
        self.solver=FrictionalFeedbackSeedRecoveryV3.__new__(FrictionalFeedbackSeedRecoveryV3)
        self.solver.controller=SimpleNamespace(sequence_geometry=None,qp=SimpleNamespace(model=None,interface=None))
        self.solver.feedback_policy=SimpleNamespace(action=lambda offset,state,prior:(np.ones(18)*.1,'feedback'))
        self.solver.feedback_spheres=[];self.solver.time_limit_s=60.
        self.state=SimpleNamespace(previous_command_target=np.zeros(18))
        self.warm=np.ones((80,18))*.2

    def solve(self):return self.solver.solve(self.state,np.zeros(18),self.warm,control_tick=478)

    def test_prediction_failure_preserves_original_warm_seed(self):
        with patch('scripts.frictional_feedback_seed_recovery_v3.predict_frictional_spheres',side_effect=ValueError('failed')):
            with patch('scripts.frictional_receding_recovery_v2.FrictionalRecedingRecoveryV2.solve',return_value={'accepted':False}) as original:
                result=self.solve()
                np.testing.assert_array_equal(original.call_args.args[2],self.warm)
                self.assertFalse(result['feedback_seed']['admitted_as_warm_start'])
                self.assertIsNotNone(result['feedback_seed']['generation_failure'])

    def test_terminal_rejection_does_not_admit_path_only_seed(self):
        prediction={'state':self.state,'diagnostics':{}}
        with patch('scripts.frictional_feedback_seed_recovery_v3.predict_frictional_spheres',return_value=prediction):
            with patch('scripts.frictional_feedback_seed_recovery_v3.core.evaluate',return_value={'accepted':False}):
                with patch('scripts.frictional_receding_recovery_v2.FrictionalRecedingRecoveryV2.solve',return_value={'accepted':False}) as original:
                    result=self.solve()
                    np.testing.assert_array_equal(original.call_args.args[2],self.warm)
                    self.assertFalse(result['feedback_seed']['admitted_as_warm_start'])

    def test_only_verified_seed_is_forwarded_for_solver_recheck(self):
        prediction={'state':self.state,'diagnostics':{}}
        with patch('scripts.frictional_feedback_seed_recovery_v3.predict_frictional_spheres',return_value=prediction):
            with patch('scripts.frictional_feedback_seed_recovery_v3.core.evaluate',return_value={'accepted':True}):
                with patch('scripts.frictional_receding_recovery_v2.FrictionalRecedingRecoveryV2.solve',return_value={'accepted':False}) as original:
                    result=self.solve()
                    np.testing.assert_allclose(original.call_args.args[2],.1)
                    self.assertTrue(result['feedback_seed']['admitted_as_warm_start'])
                    self.assertFalse(result['accepted'])


if __name__=='__main__':unittest.main()
