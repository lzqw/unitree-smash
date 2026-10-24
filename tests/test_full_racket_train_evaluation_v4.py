import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quadsmash.control.full_racket_train_evaluation_v4 import training_cases, training_noise, summarize_training


class TrainingEvaluationTest(unittest.TestCase):
    def test_all_train_cases_or_explicit_unique_subset_without_formal_files(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = json.dumps({'cases': [{'case_id': str(i)} for i in range(128)]}).encode()
            Path(directory, 'train.json').write_bytes(payload)
            protocol = dict(training_suite=dict(path='train.json', count=128,
                sha256=hashlib.sha256(payload).hexdigest()), formal_suites={'A': {'path': 'absent.json'}})
            cases, indices = training_cases(directory, protocol)
            self.assertEqual(indices, list(range(128)))
            self.assertEqual(len(cases), 128)
            cases, indices = training_cases(directory, protocol, [65, 1])
            self.assertEqual([c['case_id'] for c in cases], ['65', '1'])
            for bad in [[], [1, 1], [-1], [128], [True], [1.0]]:
                with self.assertRaises(ValueError):
                    training_cases(directory, protocol, bad)
            Path(directory, 'train.json').write_bytes(payload+b' ')
            with self.assertRaises(ValueError):
                training_cases(directory, protocol)

    def test_noise_is_reproducible_bounded_per_case_and_keeps_ground_height_outside_interface(self):
        amplitudes = dict(joint_position_rad=.025, joint_velocity_rad_s=.04, root_xy_m=.04, root_yaw_rad=.08)
        first, repeat = [training_noise(128, amplitudes, 724201) for _ in range(2)]
        different = training_noise(128, amplitudes, 724202)
        self.assertEqual(first['root_xy_m'].shape, (128, 2))
        for name in amplitudes:
            self.assertTrue(np.array_equal(first[name], repeat[name]))
            self.assertFalse(np.array_equal(first[name], different[name]))
            self.assertLessEqual(np.abs(first[name]).max(), amplitudes[name])
        self.assertNotIn('root_z_m', first)
        nominal = training_noise(3, {key: 0. for key in amplitudes}, 724201)
        self.assertTrue(all(np.count_nonzero(value) == 0 for value in nominal.values()))

    def test_bad_noise_contract_rejected(self):
        amplitudes = dict(joint_position_rad=.01, joint_velocity_rad_s=.02, root_xy_m=.02, root_yaw_rad=.03)
        for bad in [dict(amplitudes, root_yaw_rad=-.1), dict(amplitudes, root_xy_m=float('nan')),
                    dict(amplitudes, root_z_m=.1)]:
            with self.assertRaises(ValueError):
                training_noise(128, bad, 724201)

    def test_incomplete_or_wrong_first_episode_is_not_a_train_distribution_score(self):
        cases = [{'case_id': 'TRAIN0'}]
        with self.assertRaises(ValueError):
            summarize_training([], cases, [0], {})
        row = dict(env_id=0, episode_index=1, case_id='TRAIN0', source_training_index=0,
                   ground_init=True, active_rally_launch_count=1)
        with self.assertRaises(ValueError):
            summarize_training([row], cases, [0], {})
        with self.assertRaises(ValueError):
            summarize_training([row, row], cases*2, [0, 0], {})

    def test_raw_native_rows_need_no_optimizer_attack_labels_and_cannot_become_formal(self):
        from quadsmash.evaluation_v4 import FAULTS
        root = Path(__file__).resolve().parents[1]
        protocol = json.loads((root/'configs/g07_full_racket_eval_v4_r3.json').read_text())
        protocol['_root'] = str(root)
        row = dict({name: False for name in FAULTS}, case_id='TRAIN0', env_id=0, episode_index=0,
            source_training_index=0, ground_init=True, active_rally_launch_count=1, numerical_failures=0,
            complete_smash=True, contact_point_m=[1.1, 0, 1.8], first_landing_m=[3., 0., .025],
            outgoing_velocity_mps=[7.55, -1.29, -2.35], contact_time_s=2.75, release_time_s=2.76,
            crossing_time_s=2.8, first_landing_time_s=3.1, legal_initial_contact=True,
            airborne_string_hit=True, string_hit=True, released=True, crossed_net=True,
            landed_target=True, landed_legal_court=True, had_contact=True,
            jump=dict(qualified_jump_landed=True, stable_recovery=True, body_landing=False,
                terminated=False, recovery_s=1., maximum_body_ground_force_n=0.))
        result = summarize_training([row], [{'case_id': 'TRAIN0'}], [0], protocol)
        self.assertEqual(result['attack_outgoing'], 1)
        self.assertEqual(result['complete'], 1)
        self.assertFalse(result['formal_eligible'])
        self.assertFalse(result['formal_acceptance'])


if __name__ == '__main__':
    unittest.main()
