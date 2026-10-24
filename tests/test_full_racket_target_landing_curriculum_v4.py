import unittest
from quadsmash.control.full_racket_contact_curriculum_v4 import ROBOT_FAULTS,contact_search_tiers
from quadsmash.control.full_racket_target_landing_curriculum_v4 import norm_outgoing_shaping,target_landing_shaping


def row(**updates):
    value=dict(legal_initial_contact=True,airborne_string_hit=True,string_hit=True,numerical_failures=0,
               released=True,crossed_net=True,ground_seen=True,outgoing_velocity_mps=[7.55,-1.29,-2.35],
               first_landing_m=[3.,0.,.025],landed_target=False,complete_smash=False,
               **{key:False for key in ROBOT_FAULTS})
    value.update(updates);return value


class TargetLandingCurriculumTest(unittest.TestCase):
    def test_actual_speed_norm_can_meet_attack_guidance_with_forward_component_below8(self):
        self.assertAlmostEqual(norm_outgoing_shaping(row()),13.)
        self.assertEqual(norm_outgoing_shaping(row(outgoing_velocity_mps=[-8.,0.,-2.])),0.)
        self.assertLess(norm_outgoing_shaping(row(outgoing_velocity_mps=[6.,0.,-1.])),13.)

    def test_nearer_actual_first_landing_and_interior_receive_more_guidance(self):
        score=lambda x:target_landing_shaping(row(first_landing_m=[x,0.,.025]),[2.,4.],1.)
        self.assertGreater(score(4.1),score(4.7))
        self.assertGreater(score(3.8),score(4.))
        self.assertEqual(score(4.),20.)
        self.assertEqual(score(3.),25.)

    def test_unobserved_landing_miss_wrong_side_or_robot_fault_has_no_landing_reward(self):
        for update in [dict(ground_seen=False),dict(released=False),dict(crossed_net=False),
                       dict(airborne_string_hit=False),dict(legal_initial_contact=False),
                       dict(robot_net_fault=True),dict(first_landing_m=[float('nan'),0.,.025])]:
            self.assertEqual(target_landing_shaping(row(**update),[2.,4.],1.),0.)

    def test_shaping_never_relabels_native_target_or_complete_failure(self):
        value=row();before=dict(value)
        self.assertEqual(target_landing_shaping(value,[2.,4.],1.),25.)
        self.assertEqual(value,before)
        self.assertFalse(value['landed_target']);self.assertFalse(value['complete_smash'])

    def test_original_target_bounds_are_explicit_and_validated(self):
        for x,y in [([4.,2.],1.),([2.,4.],0.),([2.,float('nan')],1.)]:
            with self.assertRaises(ValueError):target_landing_shaping(row(),x,y)
        self.assertLess(target_landing_shaping(row(first_landing_m=[3.,1.2,.025]),[2.,4.],1.),20.)

    def test_full_success_then_attack_recovery_outrank_every_weaker_chain(self):
        miss=row(airborne_string_hit=False,recovery_success=False,attack_outgoing=False)
        weak=row(recovery_success=True,attack_outgoing=False)
        strong=row(recovery_success=True,attack_outgoing=True)
        full=row(recovery_success=True,attack_outgoing=True,complete_smash=True)
        groups=[[weak,weak],[strong,miss],[strong,strong],[full,miss]]
        tiers=contact_search_tiers(groups,retain_recovery=True,retain_attack_recovery=True)
        self.assertGreater(tiers[1],tiers[0])
        self.assertGreater(tiers[3],tiers[2])
        with self.assertRaises(ValueError):contact_search_tiers(groups,retain_attack_recovery=True)


if __name__=='__main__':unittest.main()
