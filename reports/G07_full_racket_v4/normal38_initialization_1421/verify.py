"""Freeze a coherent live CEM21 prefix and check the actual38-coordinate job."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'reports/G07_full_racket_v4/ball_attack_recovered_normal38_cem21'
OUT = Path(__file__).parent
SNAPSHOT = OUT / 'native_prefix'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert not (OUT / 'REPORT.json').exists(), 'Preserve existing evidence'
names = ['progress.json', 'partial_search_state.pt', 'episodes.partial.json',
         'contacts.partial.json', 'causal_plans.partial.json', 'partial_public_and_poses.npz',
         'all_first_episodes.partial.npz', 'trajectory.partial.json', 'generation_start_000.pt']
assert all((RUN / name).exists() for name in names), 'Wait for the same running job to write its prefix'
def stats():
    return [(p.stat().st_size, p.stat().st_mtime_ns) for p in [RUN / name for name in names]]
before = stats()
SNAPSHOT.mkdir()
for name in names:
    shutil.copy2(RUN / name, SNAPSHOT / name)
assert before == stats(), 'Live files changed during copy; use another fresh snapshot'
progress = json.loads((SNAPSHOT / 'progress.json').read_text())
partial = torch.load(SNAPSHOT / 'partial_search_state.pt', map_location='cpu', weights_only=False)
start = torch.load(SNAPSHOT / 'generation_start_000.pt', map_location='cpu', weights_only=False)
ticks = progress['control_ticks']
assert progress['generation'] == partial['generation'] == 0 and partial['control_ticks'] == ticks
assert partial['parameters'].shape == (64, 38) and torch.isfinite(partial['parameters']).all()
assert start['search_coordinate_indices'] == [7, 14, 37]
assert start['contact_normal_search_enabled'] and start['training_target_landing_curriculum']
assert start['selection_contract'] == 'actual-attack-recovery-chain-v3-actual-first-landing-and-speed-norm-v1'
assert start['formal_panels_read'] is False and start['formal_acceptance'] is False
mask = torch.zeros(38, dtype=torch.bool);mask[[7, 14, 37]] = True
assert (start['sigma'][~mask] == 0).all()
bank_path = ROOT / start['candidate_bank_path']
assert sha(bank_path) == start['candidate_bank_sha256']
bank = json.loads(bank_path.read_text())
bank_rows = torch.tensor(bank['parameters'], dtype=partial['parameters'].dtype)
assert bank_rows.shape == (60, 38)
assert torch.equal(partial['parameters'][2:62], bank_rows)
assert torch.equal(partial['parameters'][:, ~mask], start['reference_parameters'][None, ~mask].expand(64, -1))
assert start['parameter_names'] == bank['parameter_names'] and len(start['parameter_names']) == 38
reference_path = ROOT / 'reports/G07_full_racket_v4/attack_recovery_repeat_gen1_env3/policy_evidence.pt'
reference = torch.load(reference_path, map_location='cpu', weights_only=False)
assert torch.equal(reference['parameters'], start['reference_parameters'][:37])
assert start['reference_parameters'][37] == 0
assert not bank['native_execution'] and not bank['performance_inherited']
warm = torch.load(ROOT / 'reports/G07_full_racket_v4/attack_recovered_normal38_ready03_1415/policy.pt', map_location='cpu', weights_only=False)
assert not warm['native_execution'] and 'training_result' not in warm
arrays = {}
for name in ['partial_public_and_poses.npz', 'all_first_episodes.partial.npz']:
    with np.load(SNAPSHOT / name) as data:
        arrays[name] = {key: list(data[key].shape) for key in data.files}
        assert all(np.isfinite(data[key]).all() for key in data.files)
        if name.startswith('partial_public'):
            assert data['actions'].shape == (ticks, 2, 18) and abs(data['actions']).max() <= 1
        else:
            assert data['link_poses'].shape == ((ticks+1)//2, 128, 28, 7)
            assert np.max(abs(data['link_poses'][0, :, 0, :3]-[0, 0, .323])) < 1e-6
assert ticks == 100 and progress['first_episodes_complete'] == int(partial['first_finished'].sum()) == 0
invariant_names = ['assets/quadsmash_full_racket_v4_r3.usd', 'configs/g07_full_racket_eval_v4_r3.json',
    'quadsmash/tasks/stand/full_racket_env_v4.py', 'quadsmash/tasks/stand/full_racket_train_env_v4.py',
    'quadsmash/tasks/stand/jump_metrics.py', 'quadsmash/evaluation_v4.py']
prior_hashes = json.loads((ROOT / 'reports/G07_full_racket_v4/ball_post_stroke_retreat37_cem20/source_hashes.json').read_text())
current_hashes = json.loads((RUN / 'source_hashes.json').read_text())
assert all(sha(ROOT / name) == current_hashes[name] == prior_hashes[name] for name in invariant_names)
report = dict(verified_at=datetime.now(timezone.utc).isoformat(),
    scope='Actual CEM21 first100native ticks. This checks the38coordinate bank and initialization, before ball launch/contact; no inherited performance.',
    run=RUN.name, generation=0, control_ticks=ticks, actual_population_shape=[64, 38],
    exact_bank_rows=60, canonical38names_match=True, gaussian_search_coordinates=[7, 14, 37],
    unsearched_sigma_zero=True, all_unsearched_parameters_equal_actual_recovered37reference=True,
    actual_recovered37reference_sha256=sha(reference_path),
    normal38_search_enabled=True, actual_first_landing_and_norm_curriculum_enabled=True,
    all_native_pose_ball_clock_streams_finite=True, public_actions_bounded=True,
    first_episodes_ended=0, contact_events=len(json.loads((SNAPSHOT / 'contacts.partial.json').read_text())),
    original_physics_hardware_assets_rules_and_criteria_unchanged=True,
    invariant_sha256={name: sha(ROOT / name) for name in invariant_names}, arrays=arrays,
    native_execution=True, formal_acceptance=False, formal_panels_read=False,
    sha256={str(path.relative_to(ROOT)): sha(path) for path in [SNAPSHOT / name for name in names]+[Path(__file__), bank_path]})
(OUT / 'REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k not in ('arrays', 'sha256', 'invariant_sha256')}, indent=2))
