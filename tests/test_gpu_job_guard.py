import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location("gpu_job", Path(__file__).resolve().parents[1] / "scripts/run_gpu_job.py")
job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(job)


class GpuJobGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "status").mkdir()
        (self.root / "configs/budget.json").write_text(json.dumps({
            "authorized_goals": ["G04"], "allowed_gpu_ids": [0],
            "suggested_incremental_budgets_gpu_hours": {"G04": 8}}))
        self.ledger = self.root / "status/GPU_LEDGER.json"
        self.write_ledger([])
        self.args = ["run_gpu_job.py", "--max-seconds", "60", "--env-steps-cap", "100",
                     "--output", str(self.root / "run"), "--require-artifact", "result.json", "--", "fake-job"]

    def write_ledger(self, runs):
        self.ledger.write_text(json.dumps({"runs": runs}))

    def run_guard(self):
        with patch.object(job, "ROOT", self.root), patch("sys.argv", self.args):
            return job.main()

    def test_budget_counts_failed_retries(self):
        self.write_ledger([{"goal": "G04", "status": "FAILED", "charged_gpu_hours": 7.999, "charged_env_steps": 0}])
        with patch.object(job.subprocess, "Popen") as popen, self.assertRaises(SystemExit):
            self.run_guard()
        popen.assert_not_called()

    def test_cumulative_steps_prevent_retry_reset(self):
        self.write_ledger([{"goal": "G04", "status": "FAILED", "charged_gpu_hours": 0.1, "charged_env_steps": 199999999}])
        with patch.object(job.subprocess, "Popen") as popen, self.assertRaises(SystemExit):
            self.run_guard()
        popen.assert_not_called()

    def test_unresolved_live_job_is_not_restarted(self):
        self.write_ledger([{"goal": "G04", "status": "RUNNING", "charged_gpu_hours": 0.1, "charged_env_steps": 0}])
        with patch.object(job.subprocess, "Popen") as popen, self.assertRaises(SystemExit):
            self.run_guard()
        popen.assert_not_called()

    def test_exit_zero_without_receipt_is_failure_and_is_charged(self):
        process = Mock(pid=123)
        process.wait.return_value = 0
        with patch.object(job.subprocess, "check_output", return_value=""), \
             patch.object(job.subprocess, "run"), patch.object(job.subprocess, "Popen", return_value=process), \
             patch.object(job.signal, "signal"), patch.object(job.time, "monotonic", side_effect=[10.0, 12.0]):
            self.assertEqual(self.run_guard(), 1)
        result = json.loads(self.ledger.read_text())["runs"][-1]
        self.assertEqual(result["status"], "FAILED")
        self.assertAlmostEqual(result["charged_gpu_hours"], 2 / 3600)
        self.assertEqual(result["charged_env_steps"], 100)


class ContinuationBudgetTest(unittest.TestCase):
    def setUp(self):
        self.budget = {"authorization_windows": {
            "original": {"max_gpu_hours": 10., "max_wallclock_hours": 10.},
            "continued": {"max_gpu_hours": 8.5, "max_wallclock_hours": 10.,
                          "continuation_of": "original"}}}
        self.ledger = {"runs": [
            {"budget_window": "original", "charged_gpu_hours": 2.,
             "charged_env_steps": 100, "started_at": "2026-10-22T14:00:00+00:00"},
            {"budget_window": "continued", "charged_gpu_hours": 1.,
             "charged_env_steps": 100, "started_at": "2026-10-23T00:00:00+00:00"}]}

    def test_continuation_cannot_renew_expired_original_clock(self):
        with self.assertRaisesRegex(ValueError, "original authorization.*deadline"):
            job.check_window_budget(self.budget, self.ledger, "continued", 60, 100,
                datetime(2026, 10, 23, 1, tzinfo=timezone.utc))

    def test_job_must_fit_before_original_deadline(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            job.check_window_budget(self.budget, self.ledger, "continued", 120, 100,
                datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc))

    def test_family_charges_include_failed_previous_windows(self):
        self.ledger["runs"][0]["charged_gpu_hours"] = 9.
        with self.assertRaisesRegex(ValueError, "cumulative GPU budget"):
            job.check_window_budget(self.budget, self.ledger, "continued", 60, 100,
                datetime(2026, 10, 22, 20, tzinfo=timezone.utc))

    def test_in_budget_job_inherits_original_start(self):
        name, start, deadline = job.check_window_budget(
            self.budget, self.ledger, "continued", 60, 100,
            datetime(2026, 10, 22, 20, tzinfo=timezone.utc))
        self.assertEqual(name, "original")
        self.assertEqual((deadline - start).total_seconds(), 36000)

    def test_explicit_unlimited_window_preserves_history_without_expiry(self):
        self.budget['authorization_windows']['unlimited'] = {
            'max_gpu_hours': None, 'max_wallclock_hours': None}
        self.ledger['runs'].append(dict(budget_window='unlimited', charged_gpu_hours=1000.,
            charged_env_steps=100, started_at='2025-01-01T00:00:00+00:00'))
        original_runs = json.dumps(self.ledger, sort_keys=True)
        name, start, deadline = job.check_window_budget(self.budget, self.ledger, 'unlimited',
            7200, 100, datetime(2026, 10, 24, tzinfo=timezone.utc))
        self.assertEqual(name, 'unlimited')
        self.assertEqual(start.year, 2025)
        self.assertIsNone(deadline)
        self.assertEqual(json.dumps(self.ledger, sort_keys=True), original_runs)

    def test_unlimited_clock_does_not_remove_finite_gpu_cap(self):
        self.budget['authorization_windows']['original']['max_wallclock_hours'] = None
        self.ledger['runs'][0]['charged_gpu_hours'] = 10.
        with self.assertRaisesRegex(ValueError, 'cumulative GPU budget'):
            job.check_window_budget(self.budget, self.ledger, 'original', 60, 100,
                datetime(2026, 10, 24, tzinfo=timezone.utc))

    def test_unlimited_gpu_cap_does_not_remove_finite_clock(self):
        self.budget['authorization_windows']['original']['max_gpu_hours'] = None
        with self.assertRaisesRegex(ValueError, 'deadline'):
            job.check_window_budget(self.budget, self.ledger, 'original', 60, 100,
                datetime(2026, 10, 24, tzinfo=timezone.utc))


if __name__ == "__main__":
    unittest.main()
