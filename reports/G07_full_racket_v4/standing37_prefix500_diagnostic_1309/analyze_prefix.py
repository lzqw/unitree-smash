"""Summarize immutable native prefix; pending episodes never acquire final labels."""
from pathlib import Path
import csv
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
SNAP = OUT.parent / 'standing_recovery37_native_gen0_tick500_snapshot'
state = torch.load(SNAP / 'partial_search_state.pt', map_location='cpu', weights_only=False)
progress = json.loads((SNAP / 'progress.json').read_text())
assert state['control_ticks'] == progress['control_ticks'] == 500
first = {}
for row in json.loads((SNAP / 'episodes.partial.json').read_text()):
    first.setdefault(row['env_id'], row)
assert len(first) == int(state['first_finished'].sum()) == progress['first_episodes_complete']

def attack(row):
    v = np.array(row['outgoing_velocity_mps'])
    return bool(row['released'] and row['legal_initial_contact'] and row['string_hit']
                and row['airborne_string_hit'] and np.isfinite(v).all()
                and np.linalg.norm(v) >= 8. and v[0] > 0 and v[2] <= -1.)

keys = ['had_contact', 'frame_hit', 'airborne_string_hit', 'recovery_success',
        'crossed_net', 'landed_legal_court', 'landed_target', 'complete_smash', 'robot_net_fault']
counts = {k: sum(bool(row[k]) for row in first.values()) for k in keys}
counts['attack_outgoing'] = sum(attack(row) for row in first.values())
counts['attack_and_recovery'] = sum(attack(row) and row['recovery_success'] for row in first.values())
counts['numerical_failures'] = sum(row['numerical_failures'] for row in first.values())
counts['body_ground_terminal'] = sum(row['training_terminal_reasons']['body_ground'] for row in first.values())
counts['robot_rule_terminal'] = sum(row['training_terminal_reasons']['robot_rule_fault'] for row in first.values())

near = np.load(SNAP / 'near_contact_all.partial.npz')
control = np.load(SNAP / 'control_window.partial.npz')
all_poses = np.load(SNAP / 'all_first_episodes.partial.npz')
public = np.load(SNAP / 'partial_public_and_poses.npz')
assert all(np.isfinite(a[k]).all() for a in [near, control, all_poses, public] for k in a.files)
assert (np.abs(public['actions']) <= 1.).all()
details = []
for env_id, row in first.items():
    if not row['airborne_string_hit']:
        continue
    active = control['valid_first_episode'][:, env_id] & control['recovery_active'][:, env_id]
    handover = float(control['clock_s'][active, env_id][0]) if active.any() else None
    valid = np.where(near['valid_first_episode'][:, env_id])[0]
    candidates = valid[np.abs(near['episode_clock_s'][valid, env_id] - 2.88) <= .011]
    sample = {}
    if len(candidates):
        index = candidates[np.abs(near['episode_clock_s'][candidates, env_id] - 2.88).argmin()]
        obs = near['observations'][index, env_id]
        sample = dict(sample_clock_s=float(near['episode_clock_s'][index, env_id]),
                      projected_pitch_rad=float(np.arctan2(obs[6], -obs[8])),
                      body_pitch_rate_rad_s=float(obs[4]),
                      body_vertical_velocity_mps=float(obs[2]),
                      root_height_m=float(near['link_poses'][index, env_id, 0, 2]),
                      minimum_foot_height_m=float(near['link_poses'][index, env_id, 18:22, 2].min()))
    details.append(dict(env_id=env_id, candidate_id=env_id//2,
                        attack_outgoing=attack(row), recovery_success=row['recovery_success'],
                        body_ground_terminal=row['training_terminal_reasons']['body_ground'],
                        robot_rule_terminal=row['training_terminal_reasons']['robot_rule_fault'],
                        native_handover_first_active_clock_s=handover,
                        contact_clock_s=row['first_string_time_s'],
                        terminal_clock_s=row['terminal_native_state']['time_s'], **sample))
report = dict(created_at=datetime.now(timezone.utc).isoformat(),
              scope='Actual native first500ticks only; closed outcomes and measured motion, no inferred outcomes for18pending episodes.',
              run='ball_standing_recovery37_cem18_retry01', generation=0, control_ticks=500,
              closed_first_episodes=len(first), pending_first_episodes=len(state['first_finished'])-len(first),
              closed_counts=counts, all_closed_airborne_strings_details=details,
              all_saved_numeric_arrays_finite=True, public_actions_bounded=True,
              finding='All34airborne-strings episodes already closed with failed recovery. Some18final released outgoing velocities satisfy attack gate. Standing handover and post-hit leg extension remain inadequate in this bank; full generation and further generations still pending.',
              next_unexecuted_hypothesis='Earlier standing handover in1.20-1.40local seconds plus longer leg-target biases may allow feet to meet ground before root drops. This can change contact and must be remeasured natively.',
              all_physical_assets_hardware_limits_and_task_gates_unchanged=True,
              native_execution=True, formal_acceptance=False,
              sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SNAP.iterdir() if p.is_file()})
(OUT/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
with (OUT/'closed_airborne_strings.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in details for k in row)),lineterminator='\n')
    writer.writeheader(); writer.writerows(details)
print(json.dumps({k:v for k,v in report.items() if k not in ['sha256','all_closed_airborne_strings_details']}))
