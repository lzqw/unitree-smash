"""Freeze CEM22 native prefix and verify actual arm target telemetry/pipeline."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from quadsmash.control.full_racket_parameter_layout_v4 import controller_parameter_names

ROOT=Path.cwd()
RUN=ROOT/'reports/G07_full_racket_v4/ball_arm_target_slew45_normal38_cem22'
OUT=Path(__file__).parent
SNAPSHOT=OUT/'native_prefix'
assert not (OUT/'REPORT.json').exists() and not SNAPSHOT.exists()
names=['progress.json','partial_search_state.pt','episodes.partial.json','contacts.partial.json',
    'causal_plans.partial.json','partial_public_and_poses.npz','all_first_episodes.partial.npz',
    'trajectory.partial.json','generation_start_000.pt','arm_targets.partial.npz']
assert all((RUN/name).exists() for name in names)
def stats():return [(p.stat().st_size,p.stat().st_mtime_ns) for p in [RUN/name for name in names]]
before=stats();SNAPSHOT.mkdir()
for name in names:shutil.copy2(RUN/name,SNAPSHOT/name)
assert before==stats(), 'Use another fresh snapshot if live files change during copy'
progress=json.loads((SNAPSHOT/'progress.json').read_text())
partial=torch.load(SNAPSHOT/'partial_search_state.pt',map_location='cpu',weights_only=False)
start=torch.load(SNAPSHOT/'generation_start_000.pt',map_location='cpu',weights_only=False)
assert progress['generation']==partial['generation']==0
ticks=progress['control_ticks'];assert ticks==partial['control_ticks']==100
assert partial['parameters'].shape==(64,38) and torch.isfinite(partial['parameters']).all()
assert start['software_arm_target_rate_limit_rad_s']==start['controller_template']['seed']['arm_target_rate_limit_rad_s']==4.5
assert not start['controller_template']['native_execution'] and 'training_result' not in start['controller_template']
assert start['search_coordinate_indices']==[7,14,37]
names38=controller_parameter_names(search_arm_velocity=True,search_recovery_parameters=True,search_contact_normal=True)
assert start['parameter_names']==names38
mask=torch.zeros(38,dtype=torch.bool);mask[[7,14,37]]=True
assert (start['sigma'][~mask]==0).all()
assert torch.equal(partial['parameters'][:,~mask],start['reference_parameters'][None,~mask].expand(64,-1))
bank_path=ROOT/start['candidate_bank_path'];bank=json.loads(bank_path.read_text())
assert hashlib.sha256(bank_path.read_bytes()).hexdigest()==start['candidate_bank_sha256']
assert torch.equal(partial['parameters'][2:62],torch.tensor(bank['parameters'],dtype=partial['parameters'].dtype))
assert len(bank['parameters'])==60 and bank['parameter_names']==names38
assert int(partial['first_finished'].sum())==progress['first_episodes_complete']==0
contract_path=ROOT/'reports/G07_full_racket_v4/CPU_REPLAY_AUTHORED_JOINTS_0724.json'
contract=json.loads(contract_path.read_text());limits=np.array([contract['limits'][f'joint{i}'] for i in range(1,7)],dtype=np.float32)
with np.load(SNAPSHOT/'all_first_episodes.partial.npz') as data:
    assert all(np.isfinite(data[k]).all() for k in data.files)
    assert data['link_poses'].shape==(50,128,28,7)
    assert np.max(abs(data['link_poses'][0,:,0,:3]-[0.,0.,.323]))<1e-6
with np.load(SNAPSHOT/'partial_public_and_poses.npz') as data:
    public={key:data[key].copy() for key in data.files}
assert public['actions'].shape==(100,2,18)
assert all(np.isfinite(value).all() for value in public.values()) and abs(public['actions']).max()<=1
assert np.array_equal(public['actions'][:-1],public['observations'][1:,:,-18:])
with np.load(SNAPSHOT/'arm_targets.partial.npz') as data:
    trace={key:data[key].copy() for key in data.files}
assert all(np.isfinite(value).all() for value in trace.values())
assert trace['targets_before_slew_rad'].shape==trace['targets_after_slew_rad'].shape==(100,128,6)
assert trace['observed_position_rad'].shape==trace['observed_velocity_rad_s'].shape==(100,128,6)
assert trace['clock_s'].shape==trace['valid_first_episode'].shape==(100,128)
assert trace['valid_first_episode'].all()
assert np.array_equal(trace['clock_s'][:,:2],public['observations'][:,:,775])
assert np.array_equal(trace['observed_velocity_rad_s'][:,:2],public['observations'][:,:,39:45])
before_target=trace['targets_before_slew_rad'];after_target=trace['targets_after_slew_rad']
assert ((before_target>=limits[:,0]-1e-6)&(before_target<=limits[:,1]+1e-6)).all()
assert ((after_target>=limits[:,0]-1e-6)&(after_target<=limits[:,1]+1e-6)).all()
decoded=limits[:,0]+.5*(public['actions'][:,:,12:]+1)*(limits[:,1]-limits[:,0])
decode_error=float(abs(decoded-after_target[:,:2]).max());assert decode_error<1e-6
dt=np.diff(trace['clock_s'],axis=0);assert (dt>0).all()
command_rates=abs(np.diff(after_target,axis=0)/dt[:,:,None])
initial_rate=abs(after_target[0]-trace['observed_position_rad'][0])/.01
maximum_command_rate=float(max(command_rates.max(),initial_rate.max()))
assert maximum_command_rate<=4.5+1e-4
raw_rates=abs(np.diff(before_target,axis=0)/dt[:,:,None])
actual_velocity=abs(trace['observed_velocity_rad_s'])
prior=json.loads((ROOT/'reports/G07_full_racket_v4/ball_attack_recovered_normal38_cem21/source_hashes.json').read_text())
current=json.loads((RUN/'source_hashes.json').read_text())
invariants=['assets/quadsmash_full_racket_v4_r3.usd','configs/g07_full_racket_eval_v4_r3.json',
    'quadsmash/tasks/stand/full_racket_env_v4.py','quadsmash/tasks/stand/full_racket_train_env_v4.py',
    'quadsmash/tasks/stand/jump_metrics.py','quadsmash/evaluation_v4.py']
assert all(prior[name]==current[name]==hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in invariants)
report=dict(verified_at=datetime.now(timezone.utc).isoformat(),
    scope='Actual CEM22first100native ticks, before ball launch.64x38population/128native streams and applied arm target rate telemetry only; no contact, recovery, first landing or full task performance is proven.',
    generation=0,control_ticks=100,actual_population_shape=[64,38],canonical_parameter_names_verified=True,
    exact_bank_rows=60,search_coordinates=[7,14,37],unsearched_parameters_fixed_and_sigma_zero=True,
    software_arm_target_rate_limit_rad_s=4.5,maximum_logged_after_slew_command_rate_rad_s=maximum_command_rate,
    maximum_logged_before_slew_command_rate_rad_s=float(raw_rates.max()),
    maximum_logged_slew_adjustment_rad=float(abs(after_target-before_target).max()),
    measured100Hz_arm_velocity_peak_rad_s=float(actual_velocity.max()),
    measured_samples_over_nominal5_rad_s_plus1e_4=int((actual_velocity>5.0001).sum()),
    measured_velocity_scope='100Hz public observations only, not a physics-substep maximum; observations retained without clipping.',
    applied_public_action_decoding_error_rad=decode_error,
    first2_public_action_pipeline_bitwise_verified=True,
    first2_measured_velocity_trace_bitwise_matches_public_observations=True,
    all128_native_body_ball_clock_streams_finite=True,all_logged_arm_targets_finite_and_position_bounded=True,
    first_episodes_ended=0,contact_events=len(json.loads((SNAPSHOT/'contacts.partial.json').read_text())),
    original_asset_physics_rules_and_criteria_unchanged=True,
    invariant_sha256={name:current[name] for name in invariants},
    native_execution=True,formal_acceptance=False,formal_panels_read=False,
    sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [SNAPSHOT/name for name in names]+[Path(__file__),contract_path,bank_path]})
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['invariant_sha256','sha256']},indent=2))
