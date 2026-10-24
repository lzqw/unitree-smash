"""Verify the actual CEM23 stop without manufacturing a completed receipt."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'reports/G07_full_racket_v4/ball_arm_target_slew50_normal38_cem23'
OUT=Path(__file__).parent/'REPORT.json'
assert not OUT.exists()
job=json.loads((RUN/'job.json').read_text())
assert job['status']=='FAILED' and job['exit_code']==0
assert not job['timed_out'] and not job['receipt_present'] and not (RUN/'completed.json').exists()
assert all(not Path(f'/proc/{job[key]}').exists() for key in ['pid','child_pid'])
summaries=[json.loads((RUN/f'GENERATION{i}_SUMMARY.json').read_text()) for i in range(2)]
assert all(s['counts']['executions']==128 and s['control_ticks']==800 for s in summaries)
counts={k:sum(s['counts'][k] for s in summaries) for k in summaries[0]['counts']}
interrupted=json.loads((RUN/'interrupted.json').read_text())
assert interrupted['generation']==2
partial=torch.load(RUN/'interrupted_search_state_002.pt',map_location='cpu',weights_only=False)
assert partial['control_ticks']==interrupted['control_ticks']==73
assert partial['parameters'].shape==(64,38) and torch.isfinite(partial['parameters']).all()
assert int(partial['first_finished'].sum())==0
assert json.loads((RUN/'interrupted_rallies_002.json').read_text())==[]
assert json.loads((RUN/'interrupted_jumps_002.json').read_text())==[]
shapes={}
for name in ['all_first_episodes_002.npz','public_and_poses_002.npz','arm_targets_002.npz']:
    with np.load(RUN/name) as data:
        assert all(np.isfinite(data[k]).all() for k in data.files)
        shapes[name]={k:list(data[k].shape) for k in data.files}
        if name.startswith('public'):
            assert data['actions'].shape==(73,2,18) and abs(data['actions']).max()<=1
        elif name.startswith('all_'):
            assert data['link_poses'].shape==(37,128,28,7)
            assert np.max(abs(data['link_poses'][0,:,0,:3]-[0.,0.,.323]))<1e-6
        else:
            assert data['targets_after_slew_rad'].shape==(73,128,6)
            assert data['valid_first_episode'].all()
names=['job.json','interrupted.json','interrupted_search_state_002.pt','interrupted_rallies_002.json',
    'interrupted_jumps_002.json','public_and_poses_002.npz','all_first_episodes_002.npz',
    'GENERATION0_SUMMARY.json','GENERATION1_SUMMARY.json','source_hashes.json']
report=dict(verified_at=datetime.now(timezone.utc).isoformat(),
    scope='Two complete native TRAIN generations and separately preserved73tick generation2interruption. No four-generation completed receipt or formal panel result.',
    run=RUN.name,source_commit=job['git_sha'],guard_terminal_status=job['status'],child_exit_code=job['exit_code'],
    completed_generations=2,completed_generation_counts=counts,distinct_TRAIN_indices=[1,65],distinct_cases=2,
    interrupted_generation=2,interrupted_ticks=73,interrupted_terminal_records=0,interrupted_pending_first_episodes=128,
    all_interrupted_arrays_finite=True,public_actions_bounded=True,arrays=shapes,
    actual_control_env_steps=sum(s['actual_control_env_steps'] for s in summaries)+73*128,
    charged_control_env_steps_cap=job['charged_env_steps'],actual_gpu_hours=job['charged_gpu_hours'],
    elapsed_s=job['elapsed_s'],native_execution=True,formal_acceptance=False,
    sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/name for name in names]+[Path(__file__)]})
OUT.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['arrays','sha256']},indent=2))
