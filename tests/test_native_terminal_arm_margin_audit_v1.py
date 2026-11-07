"""Corrected command dispatch and corruption checks, not native physics evidence."""
import unittest
import numpy as np
from tests.test_native_terminal_support_audit_v1 import TerminalDispatch
from scripts.audit_native_terminal_support_v1 import state_at
from scripts.search_terminal_prefix_arm_margin_v1 import ArmPrefixCandidate


class CorrectedTerminalDispatch(TerminalDispatch):
    def setUp(self):
        super().setUp()
        self.raw=self.raw.astype(np.float32)
        self.final=self.final.astype(np.float32)
        self.base_policy=self.policy
        self.policy=ArmPrefixCandidate(self.base_policy,self.base_policy.interface,-.006)
        self.policy.model=self.base_policy.model
        decode=lambda value:self.limits[:,0]+.5*(value+1.)*np.diff(self.limits).ravel()
        previous=decode(self.raw[459,3])
        for offset in range(340):
            state=state_at(self.policy.model,self.trace,offset)
            target,mode=self.policy.action(offset,state,previous)
            action=(2*(target-self.limits[:,0])/np.diff(self.limits).ravel()-1).astype(np.float32)
            self.final[460+offset,3]=action
            self.trace['previous_target_rad'][offset]=previous
            self.trace['requested_target_rad'][offset]=target
            self.trace['final_action'][offset]=action
            self.trace['final_target_rad'][offset]=decode(action)
            self.result['records'][offset]['mode']=mode
            previous=decode(action)

    def test_auditor_cannot_silently_omit_arm_correction(self):
        self.policy=self.base_policy
        with self.assertRaises(ValueError):self.audit()

    def test_corrected_window_changes_actual_arm_command(self):
        self.audit()
        state=state_at(self.policy.model,self.trace,30)
        original,_=self.base_policy.action(30,state,self.trace['previous_target_rad'][30])
        self.assertAlmostEqual(self.trace['requested_target_rad'][30,13]-original[13],-.006)


if __name__=='__main__':unittest.main()
