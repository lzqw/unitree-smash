"""Phase binding and audit-corruption checks; no native simulation here."""
import copy
from types import SimpleNamespace
import unittest
from scripts.native_feedback_portfolio_v4 import ObservedPhasePlanner
from scripts.audit_native_feedback_portfolio_v4 import audit_portfolio


class NativePortfolioChecks(unittest.TestCase):
    def test_phase_is_consumed_once_and_forwarded_without_inference(self):
        calls=[]
        p=ObservedPhasePlanner(SimpleNamespace(solve=lambda *args,**kw:(calls.append((args,kw)) or {'accepted':False}),close=lambda:None))
        with self.assertRaises(ValueError):p.solve('state','reference','warm')
        p.tick=473;answer=p.solve('state','reference','warm')
        self.assertEqual(calls,[(('state','reference','warm'),{'control_tick':473})])
        self.assertEqual(answer['observed_control_tick'],473)
        with self.assertRaises(ValueError):p.solve('later state','reference','warm')

    def record(self):
        return dict(planner_version='verified_feedback_portfolio_v4',
            feedback_candidate_order=['original_arm_margin','arm_speed_reserve'],per_query_soft_budget_s=60.,
            feedback_seed_selection_counts={'arm_speed_reserve':1},planning_records=[dict(tick=473,decision=dict(
                observed_control_tick=473,planner_version='verified_feedback_portfolio_v4',accepted=False,
                feedback_portfolio=dict(selected='arm_speed_reserve',original_shifted_seed_used=False,attempts=[
                    dict(name='original_arm_margin',generation_failure={'step':15},verification=None),
                    dict(name='arm_speed_reserve',generation_failure=None,verification={'accepted':True})])))])

    def test_seed_acceptance_does_not_require_or_invent_execution_acceptance(self):
        record=self.record();audit_portfolio(record)
        self.assertFalse(record['planning_records'][0]['decision']['accepted'])

    def test_wrong_phase_or_unverified_selection_rejected(self):
        for mode in ['phase','selection','order']:
            r=copy.deepcopy(self.record());d=r['planning_records'][0]['decision']
            if mode=='phase':d['observed_control_tick']=472
            elif mode=='selection':d['feedback_portfolio']['attempts'][1]['verification']['accepted']=False
            else:d['feedback_portfolio']['attempts'].reverse()
            with self.assertRaises(ValueError):audit_portfolio(r)


if __name__=='__main__':unittest.main()
