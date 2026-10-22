import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
