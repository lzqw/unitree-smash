"""Evaluation integrity unit tests; synthetic flags are not physics evidence."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from quadsmash.evaluation_v4 import FAULTS,independent_episode,summarize_panel,aggregate_panels

ROOT=Path(__file__).resolve().parents[1]


class FrozenEvaluationIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.protocol=json.loads((ROOT/'configs/g07_full_racket_eval_v4_r3.json').read_text())
        self.protocol['_root']=str(ROOT)
        self.cases=json.loads((ROOT/self.protocol['formal_suites']['A']['path']).read_text())['cases']

    def row(self,i=0,success=False):
        return dict({name:False for name in FAULTS},case_id=self.cases[i]['case_id'],env_id=i,episode_index=0,
            ground_init=True,active_rally_launch_count=1,numerical_failures=0,complete_smash=success,
            contact_point_m=[1.1,0,1.8],first_landing_m=[3.,0.,.025],outgoing_velocity_mps=[10.,0.,-2.],
            contact_time_s=2.75,release_time_s=2.76,crossing_time_s=2.8,first_landing_time_s=3.1,
            legal_initial_contact=success,airborne_string_hit=success,string_hit=success,
            released=success,crossed_net=success,landed_target=success,landed_legal_court=success,had_contact=success,
            jump=dict(qualified_jump_landed=success,stable_recovery=success,body_landing=False,
                terminated=False,recovery_s=1. if success else 0.,maximum_body_ground_force_n=0.))

    def test_later_fault_invalidates_complete_and_conditional_recovery(self):
        row=self.row(success=True);self.assertTrue(independent_episode(row,self.protocol)['complete'])
        row['robot_net_fault']=True;row['complete_smash']=False
        checked=independent_episode(row,self.protocol)
        self.assertTrue(checked['airborne_contact']);self.assertFalse(checked['complete'])
        self.assertFalse(checked['conditional_recovery'])

    def test_declared_success_without_physical_event_chain_is_rejected(self):
        row=self.row();row['complete_smash']=True
        with self.assertRaises(ValueError):independent_episode(row,self.protocol)

    def test_missing_or_duplicate_cases_cannot_be_dropped(self):
        rows=[self.row(i) for i in range(200)]
        with self.assertRaises(ValueError):summarize_panel(rows[:-1],self.protocol,'A')
        rows[-1]['case_id']=rows[0]['case_id']
        with self.assertRaises(ValueError):summarize_panel(rows,self.protocol,'A')

    def test_no_contacts_has_zero_conditional_rate(self):
        summary=summarize_panel([self.row(i) for i in range(200)],self.protocol,'A')
        self.assertEqual(summary['conditional_recovery_rate'],0.)
        self.assertEqual(summary['episodes'],200)

    def test_numerical_counts_cannot_cancel_failures(self):
        rows=[self.row(i) for i in range(200)];rows[0]['numerical_failures']=-1
        with self.assertRaises(ValueError):summarize_panel(rows,self.protocol,'A')

    def panels(self):
        base=dict(episodes=200,formal_eligible=True,policy_sha256='weights',protocol_sha256='protocol',
            policy_lock_sha256='lock',numerical_failures=0,airborne_contact_rate=1.,conditional_recovery_rate=.9)
        return [dict(base,panel='A',complete_success_rate=.55),dict(base,panel='B',complete_success_rate=.45)]

    def test_worse_panel_controls_acceptance_instead_of_average(self):
        result=aggregate_panels(self.panels(),self.protocol)
        self.assertEqual(result['worst_panel_rates']['complete_success_rate'],.45)
        self.assertFalse(result['formal_acceptance'])

    def test_panel_weights_must_match(self):
        panels=self.panels();panels[1]['policy_sha256']='different'
        with self.assertRaises(ValueError):aggregate_panels(panels,self.protocol)


if __name__=='__main__':unittest.main()
