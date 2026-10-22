"""CPU tests of the evidence checker, NOT robot/simulator tests.

All fixtures below are synthetic and live in temporary directories. They must
never be copied into actual reports/ or used as evidence for a robotics Goal.
"""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / 'tools' / 'verify_goal.py'
spec = importlib.util.spec_from_file_location('verify_goal', TOOL)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class EvidenceCheckerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'configs').mkdir()
        (self.root / 'reports/G00').mkdir(parents=True)
        self.contract = {
            'goals': {'G00': {
                'task_id': 'synthetic_checker_test', 'depends_on': [],
                'checks': {'ok': {'op': 'eq', 'value': True}, 'rate': {'op': 'ge', 'value': 0.9}},
                'required_artifacts': ['reports/G00/log.txt'],
            }}
        }
        self.write_contract()
        log = self.root / 'reports/G00/log.txt'
        log.write_text('Synthetic unit-test fixture; NOT a simulation.\n')
        self.report = {
            'schema_version': 1, 'goal_id': 'G00', 'task_id': 'synthetic_checker_test',
            'contract_sha256': gate.digest(self.root / 'configs/acceptance.json'),
            'evidence_kind': 'observed',
            'run': {'command': 'synthetic-test', 'git_commit': 'test-only',
                    'started_at': '2026-10-21T00:00:00+00:00',
                    'finished_at': '2026-10-21T00:00:01+00:00',
                    'gpu_hours': 0.0, 'env_steps': 0},
            'metrics': {'ok': True, 'rate': 0.95},
            'artifacts': [{'path': 'reports/G00/log.txt', 'sha256': gate.digest(log)}],
        }
        self.write_report()

    def tearDown(self):
        self.tmp.cleanup()

    def write_contract(self):
        (self.root / 'configs/acceptance.json').write_text(json.dumps(self.contract))

    def write_report(self):
        (self.root / 'reports/G00/evidence.json').write_text(json.dumps(self.report))

    def passes(self):
        return gate.verify(self.root, 'G00')['status'] == 'PASSED'

    def test_valid_synthetic_fixture(self):
        self.assertTrue(self.passes())

    def test_missing_report_fails(self):
        (self.root / 'reports/G00/evidence.json').unlink()
        self.assertFalse(self.passes())

    def test_missing_metric_fails(self):
        del self.report['metrics']['rate']; self.write_report()
        self.assertFalse(self.passes())

    def test_low_metric_fails(self):
        self.report['metrics']['rate'] = 0.89; self.write_report()
        self.assertFalse(self.passes())

    def test_nan_metric_fails(self):
        self.report['metrics']['rate'] = float('nan'); self.write_report()
        self.assertFalse(self.passes())

    def test_bool_not_accepted_as_rate(self):
        self.report['metrics']['rate'] = True; self.write_report()
        self.assertFalse(self.passes())

    def test_numeric_not_accepted_as_bool(self):
        self.report['metrics']['ok'] = 1; self.write_report()
        self.assertFalse(self.passes())

    def test_artifact_tamper_fails(self):
        (self.root / 'reports/G00/log.txt').write_text('changed')
        self.assertFalse(self.passes())

    def test_contract_tamper_fails(self):
        self.contract['goals']['G00']['checks']['rate']['value'] = 0.1
        self.write_contract()
        self.assertFalse(self.passes())

    def test_wrong_task_fails(self):
        self.report['task_id'] = 'easier-debug-task'; self.write_report()
        self.assertFalse(self.passes())

    def test_mock_kind_fails(self):
        self.report['evidence_kind'] = 'mock'; self.write_report()
        self.assertFalse(self.passes())

    def test_path_escape_fails(self):
        self.report['artifacts'].append({'path': '../../bad', 'sha256': 'abc'})
        self.write_report(); self.assertFalse(self.passes())

    def test_missing_required_artifact_fails(self):
        self.report['artifacts'] = []; self.write_report()
        self.assertFalse(self.passes())

    def test_empty_artifact_fails(self):
        p = self.root / 'reports/G00/log.txt'; p.write_text('')
        self.report['artifacts'][0]['sha256'] = gate.digest(p); self.write_report()
        self.assertFalse(self.passes())

    def test_invalid_time_fails(self):
        self.report['run']['finished_at'] = 'yesterday'; self.write_report()
        self.assertFalse(self.passes())

    def test_missing_dependency_fails(self):
        self.contract['goals']['G01'] = {
            'task_id': 'other', 'depends_on': ['G00'], 'checks': {}, 'required_artifacts': []}
        self.write_contract()
        (self.root / 'reports/G01').mkdir()
        other = dict(self.report, goal_id='G01', task_id='other', metrics={}, artifacts=[],
                     contract_sha256=gate.digest(self.root / 'configs/acceptance.json'))
        (self.root / 'reports/G01/evidence.json').write_text(json.dumps(other))
        # G00 contract hash is stale, so G01 must not pass despite its own valid evidence.
        self.assertEqual(gate.verify(self.root, 'G01')['status'], 'FAILED')

    def test_dependency_cycle_rejected(self):
        self.contract['goals']['G00']['depends_on'] = ['G00']; self.write_contract()
        with self.assertRaises(ValueError):
            gate.verify(self.root, 'G00')

    def test_unknown_goal_rejected(self):
        with self.assertRaises(ValueError):
            gate.verify(self.root, 'G99')


if __name__ == '__main__':
    unittest.main()
