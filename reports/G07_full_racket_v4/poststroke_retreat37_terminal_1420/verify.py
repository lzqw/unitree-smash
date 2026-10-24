"""Verify immutable CEM20 terminal evidence, preserving incomplete-job status."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'reports/G07_full_racket_v4/ball_post_stroke_retreat37_cem20'
OUT = Path(__file__).parent / 'REPORT.json'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert not OUT.exists(), 'Keep historical verification immutable'
job = json.loads((RUN / 'job.json').read_text())
interrupted = json.loads((RUN / 'interrupted.json').read_text())
assert job['status'] == 'FAILED' and job['exit_code'] == 0
assert not job['receipt_present'] and not job['timed_out']
assert not (RUN / 'completed.json').exists()
assert all(not Path(f'/proc/{job[key]}').exists() for key in ['pid', 'child_pid'])
summaries = [json.loads((RUN / f'GENERATION{i}_SUMMARY.json').read_text()) for i in range(2)]
assert [s['generation'] for s in summaries] == [0, 1]
assert all(s['control_ticks'] == 800 and s['counts']['executions'] == 128 for s in summaries)
counts = {key: sum(s['counts'][key] for s in summaries) for key in summaries[0]['counts']}
assert counts['complete_smash'] == counts['canonical_complete'] == 0
state = torch.load(RUN / 'interrupted_search_state_002.pt', map_location='cpu', weights_only=False)
ticks = int(state['control_ticks'])
assert interrupted['generation'] == state['generation'] == 2
assert interrupted['control_ticks'] == ticks == 339
assert state['parameters'].shape == (64, 37) and torch.isfinite(state['parameters']).all()
rallies = json.loads((RUN / 'interrupted_rallies_002.json').read_text())
jumps = json.loads((RUN / 'interrupted_jumps_002.json').read_text())
assert len(rallies) == len(jumps) == int(state['first_finished'].sum()) == 3
assert not any(r['complete_smash'] for r in rallies)
assert sum(r['numerical_failures'] for r in rallies) == 0
arrays = {}
names = ['all_first_episodes_002.npz', 'public_and_poses_002.npz', 'control_window_002.npz', 'near_contact_all_002.npz']
for name in names:
    with np.load(RUN / name) as data:
        arrays[name] = {k: list(data[k].shape) for k in data.files}
        assert all(np.isfinite(data[k]).all() for k in data.files if np.issubdtype(data[k].dtype, np.number))
        if 'actions' in data.files:
            assert np.abs(data['actions']).max() <= 1
        if name == 'public_and_poses_002.npz':
            assert data['actions'].shape == (ticks, 2, 18)
        if name == 'all_first_episodes_002.npz':
            assert data['link_poses'].shape == ((ticks + 1) // 2, 128, 28, 7)
            assert np.max(abs(data['link_poses'][0, :, 0, :3] - [0, 0, .323])) < 1e-6
paths = [RUN / name for name in names + ['job.json', 'interrupted.json', 'interrupted_search_state_002.pt',
         'interrupted_rallies_002.json', 'interrupted_jumps_002.json', 'source_hashes.json',
         'GENERATION0_SUMMARY.json', 'GENERATION1_SUMMARY.json', 'population_000.pt', 'population_001.pt']]
report = dict(verified_at=datetime.now(timezone.utc).isoformat(),
    scope='Two completed native TRAIN generations plus separately preserved interrupted generation2. Not a complete four-generation job or formal-panel result.',
    run=RUN.name, source_commit=job['git_sha'], guard_terminal_status=job['status'],
    child_exit_code=job['exit_code'], completed_receipt_present=False,
    completed_generations=2, completed_generation_counts=counts,
    distinct_TRAIN_indices=[1, 65], distinct_cases=2,
    interrupted_generation=2, interrupted_ticks=ticks, interrupted_terminal_records=len(rallies),
    interrupted_pending_first_episodes=128-len(rallies), interrupted_complete=0,
    arrays=arrays, all_interrupted_arrays_finite=True, public_actions_bounded=True,
    actual_control_env_steps=sum(s['actual_control_env_steps'] for s in summaries)+ticks*128,
    charged_control_env_steps_cap=job['charged_env_steps'], actual_gpu_hours=job['charged_gpu_hours'],
    actual_elapsed_s=job['elapsed_s'], native_execution=True, formal_acceptance=False,
    sha256={str(path.relative_to(ROOT)): sha(path) for path in paths+[Path(__file__)]})
OUT.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('arrays', 'sha256')}, indent=2))
