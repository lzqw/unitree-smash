"""Admission semantics only; mocks provide no physical-success evidence."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scripts.frictional_feedback_portfolio_v4 import FrictionalFeedbackPortfolioV4


class PortfolioAdmission(unittest.TestCase):
    def setUp(self):
        self.solver = FrictionalFeedbackPortfolioV4.__new__(FrictionalFeedbackPortfolioV4)
        self.solver.controller = SimpleNamespace(qp=SimpleNamespace(model=None,interface=None))
        self.solver.spheres = []; self.solver.geometry = None; self.solver.time_limit_s = 60.
        self.solver.feedback_policies = [
            ('first',SimpleNamespace(action=lambda *args:(np.full(18,.1),'a'))),
            ('second',SimpleNamespace(action=lambda *args:(np.full(18,.2),'b')))]
        self.state = SimpleNamespace(previous_command_target=np.zeros(18))
        self.warm = np.full((80,18),.3)

    def test_second_candidate_is_rechecked_and_cannot_override_solver_rejection(self):
        with patch('scripts.frictional_feedback_portfolio_v4.predict_frictional_spheres',
                   side_effect=lambda model,interface,state,*a,**k:dict(state=state,diagnostics={})), \
             patch('scripts.frictional_feedback_portfolio_v4.core.evaluate',side_effect=[{'accepted':False},{'accepted':True}]), \
             patch('scripts.frictional_receding_recovery_v2.FrictionalRecedingRecoveryV2.solve',return_value={'accepted':False}) as final:
            result = self.solver.solve(self.state,np.zeros(18),self.warm,control_tick=473)
        self.assertEqual(result['feedback_portfolio']['selected'],'second')
        np.testing.assert_allclose(final.call_args.args[2],.2)
        np.testing.assert_array_equal(self.state.previous_command_target,np.zeros(18))
        self.assertFalse(result['accepted'])

    def test_no_accepted_candidate_preserves_original_seed(self):
        with patch('scripts.frictional_feedback_portfolio_v4.predict_frictional_spheres',side_effect=ValueError('path failure')), \
             patch('scripts.frictional_feedback_portfolio_v4.core.evaluate') as verify, \
             patch('scripts.frictional_receding_recovery_v2.FrictionalRecedingRecoveryV2.solve',return_value={'accepted':False}) as final:
            result = self.solver.solve(self.state,np.zeros(18),self.warm,control_tick=473)
        verify.assert_not_called()
        self.assertIsNone(result['feedback_portfolio']['selected'])
        self.assertEqual(len(result['feedback_portfolio']['attempts']),2)
        np.testing.assert_array_equal(final.call_args.args[2],self.warm)


if __name__=='__main__': unittest.main()
