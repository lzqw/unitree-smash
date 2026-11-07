"""Synthetic dispatch corruption tests; no simulated recovery claim."""
import unittest
import numpy as np
from tests import test_terminal_support_policy_v1 as fixture_module
from scripts.audit_native_terminal_support_v1 import audit_dispatch


class TerminalDispatch(unittest.TestCase):
    def setUp(self):
        fixture=fixture_module.TerminalPolicy();fixture.setUp()
        self.policy=fixture.policy;self.limits=fixture.f.interface.limits
        self.raw=np.zeros((800,128,18));self.final=self.raw.copy();state=fixture.f.state
        decode=lambda x:self.limits[:,0]+.5*(x+1.)*np.diff(self.limits).ravel()
        previous=decode(self.raw[459,3]);rows=[];trace={k:[] for k in
            ['q','v','contacts','student_action','final_action','previous_target_rad','requested_target_rad','final_target_rad']}
        for i in range(340):
            target,mode=self.policy.action(i,state,previous)
            action=2*(target-self.limits[:,0])/np.diff(self.limits).ravel()-1
            self.final[460+i,3]=action
            values=dict(q=state.q,v=state.v,contacts=state.contacts,student_action=self.raw[460+i,3],
                final_action=action,previous_target_rad=previous,requested_target_rad=target,final_target_rad=decode(action))
            for key,value in values.items():trace[key].append(value.copy())
            rows.append(dict(control_tick=460+i,offset=i,mode=mode));previous=decode(action)
        self.trace={k:np.stack(v) for k,v in trace.items()}
        self.result=dict(case=3,start_tick=460,planned_steps=340,records=rows,
            teacher_steps=340,unverified_student_fallback_steps=0)

    def audit(self):return audit_dispatch(self.result,self.trace,self.raw,self.final,self.limits,self.policy)

    def test_complete_feedback_command_chain(self):
        result=self.audit()
        self.assertEqual(result['prefix_feedback_steps'],80)
        self.assertEqual(result['terminal_feedback_steps'],260)
        self.assertFalse(result['online_replanning'])

    def test_changed_measurement_cannot_match_unchanged_command(self):
        self.trace['q'][100,self.policy.model.q_indices[0]]+=.1
        with self.assertRaises(ValueError):self.audit()

    def test_other_environment_override_rejected(self):
        self.final[700,4,0]+=.01
        with self.assertRaises(ValueError):self.audit()

    def test_false_terminal_mode_rejected(self):
        self.result['records'][100]['mode']='student_fallback'
        with self.assertRaises(ValueError):self.audit()


if __name__=='__main__':unittest.main()
