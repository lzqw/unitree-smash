import copy
import unittest
import numpy as np
from quadsmash.control.full_racket_contact_curriculum_v4 import (
    ROBOT_FAULTS,usable_air_contact,contact_search_tiers,actual_outgoing_shaping,
    arm_timing_velocity_grid,measured_recovery_leg_crossovers,pitch_compensation_grid,validate_recovery_search_min)


class ContactCurriculumTest(unittest.TestCase):
    def test_recovery_search_floor_allows_earlier_control_but_rejects_invalid_values(self):
        self.assertEqual(validate_recovery_search_min(1.55), 1.55)
        self.assertEqual(validate_recovery_search_min(1.30), 1.30)
        for value in [float('nan'), float('inf'), 1.19, 1.56]:
            with self.assertRaises(ValueError):
                validate_recovery_search_min(value)

    def row(self, contact=False, **extra):
        return dict(legal_initial_contact=contact, airborne_string_hit=contact,
                    string_hit=contact, numerical_failures=0, complete_smash=False,
                    released=contact, outgoing_velocity_mps=[4.8, -.36, -1.05],
                    **{name:False for name in ROBOT_FAULTS}, **extra)

    def test_blocked_contact_is_retained_without_reviving_or_relabelling_success(self):
        missed = self.row()
        blocked = self.row(True, net_blocked_fault=True, recovery_success=False)
        saved = copy.deepcopy(blocked)
        self.assertGreater(contact_search_tiers([[blocked, missed], [missed, missed]])[0], 0)
        self.assertEqual(blocked, saved)
        self.assertFalse(blocked['complete_smash'])

    def test_any_native_robot_fault_disqualifies_contact_priority(self):
        for fault in ROBOT_FAULTS:
            row = self.row(True)
            row[fault] = True
            self.assertFalse(usable_air_contact(row), fault)
        row = self.row(True)
        row['numerical_failures'] = 1
        self.assertEqual(contact_search_tiers([[row]])[0], -1)

    def test_actual_full_success_outranks_more_partial_contacts(self):
        full = self.row(True)
        full['complete_smash'] = True
        tiers = contact_search_tiers([[full, self.row()], [self.row(True), self.row(True)]])
        self.assertGreater(tiers[0], tiers[1])

    def test_faster_forward_actual_return_has_bounded_reward(self):
        weak = self.row(True)
        fast = self.row(True)
        fast['outgoing_velocity_mps'] = [8., 0., -1.]
        self.assertGreater(actual_outgoing_shaping(fast), actual_outgoing_shaping(weak))
        fast['outgoing_velocity_mps'] = [100., 0., -100.]
        self.assertEqual(actual_outgoing_shaping(fast), 13.)
        self.assertEqual(actual_outgoing_shaping(self.row()), 0.)
        fast['outgoing_velocity_mps'] = [float('nan'), 0., -1.]
        self.assertEqual(actual_outgoing_shaping(fast), 0.)

    def test_recovery_curriculum_retains_same_episode_chain_ahead_of_two_falls(self):
        joined = self.row(True, recovery_success=True)
        fall = self.row(True, recovery_success=False)
        groups = [[joined, self.row()], [fall, fall]]
        saved = copy.deepcopy(groups)
        self.assertLess(contact_search_tiers(groups)[0], contact_search_tiers(groups)[1])
        tiers = contact_search_tiers(groups, retain_recovery=True)
        self.assertGreater(tiers[0], tiers[1])
        self.assertEqual(groups, saved)
        self.assertFalse(joined['complete_smash'])

    def test_recovery_in_another_episode_cannot_form_a_contact_chain(self):
        contact = self.row(True, recovery_success=False)
        recovered_miss = self.row(False, recovery_success=True)
        two_contacts = [[contact, recovered_miss], [contact, contact]]
        tiers = contact_search_tiers(two_contacts, retain_recovery=True)
        self.assertLess(tiers[0], tiers[1])

    def test_full_smash_still_beats_recovered_partial_contacts(self):
        complete = self.row(True, recovery_success=True)
        complete['complete_smash'] = True
        partial = self.row(True, recovery_success=True)
        tiers = contact_search_tiers([[complete, self.row()], [partial, partial]], retain_recovery=True)
        self.assertGreater(tiers[0], tiers[1])

    def test_recovery_does_not_override_robot_faults_or_numerical_failure(self):
        for fault in ROBOT_FAULTS:
            invalid = self.row(True, recovery_success=True)
            invalid[fault] = True
            tiers = contact_search_tiers([[invalid], [self.row(True)]], retain_recovery=True)
            self.assertLess(tiers[0], tiers[1], fault)
        invalid = self.row(True, recovery_success=True)
        invalid['numerical_failures'] = 1
        self.assertEqual(contact_search_tiers([[invalid]], retain_recovery=True)[0], -1)

    def test_controlled_grid_preserves_every_leg_and_interception_parameter(self):
        reference = np.linspace(-.2, .2, 26)
        lower, upper = np.full(26, -.15), np.full(26, .15)
        grid = arm_timing_velocity_grid(reference, lower, upper)
        untouched = [i for i in range(26) if i not in [7, 14]]
        self.assertEqual(grid.shape, (12, 26))
        np.testing.assert_array_equal(grid[:, untouched], np.tile(reference[untouched], (12, 1)))
        self.assertTrue(np.all(grid[:, [7,14]] <= .15))
        self.assertTrue(np.all(grid[:, [7,14]] >= -.15))

    def test_recovery_crossover_preserves_contact_arm_and_ball_interception(self):
        reference = np.zeros(26)
        recovered = np.full(26, .1)
        lower, upper = np.full(26, -.5), np.full(26, .5)
        grid = measured_recovery_leg_crossovers(reference, recovered, lower, upper)
        self.assertEqual(grid.shape, (8,26))
        fixed = [7,8,9,10,11,12,13,15,16,18,19,24,25]
        np.testing.assert_array_equal(grid[:,fixed], np.zeros((8,len(fixed))))
        self.assertAlmostEqual(grid[0,2], .025)
        self.assertAlmostEqual(grid[-1,2], .1)
        self.assertAlmostEqual(grid[-1,14], 0.)

    def test_pitch_grid_spans_body_rotation_compensation_without_changing_leg_timing(self):
        reference = np.linspace(-.2,.2,26)
        lower, upper = np.full(26,-2.),np.full(26,2.)
        grid = pitch_compensation_grid(reference,lower,upper)
        self.assertEqual(grid.shape,(12,26))
        np.testing.assert_allclose(np.unique(grid[:,15]),[.4,.7,1.,1.3])
        fixed = [i for i in range(26) if i not in [14,15]]
        np.testing.assert_array_equal(grid[:,fixed],np.tile(reference[fixed],(12,1)))


if __name__ == '__main__':
    unittest.main()
