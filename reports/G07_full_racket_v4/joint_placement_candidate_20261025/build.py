"""Replay a candidate TRAIN bonus on preserved native events, without simulation."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from quadsmash.control.measured_joint_placement_reward_v4 import MeasuredJointPlacementRewardV4
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'REPORT.json').exists(), 'Never overwrite a candidate replay'
    source = ROOT / 'reports/G07_full_racket_v4/residual_n2048_first_ground_cohort_20261025'
    manifest = json.loads((source / 'REPORT.json').read_bytes())
    data_path = source / 'first_ground_episodes.json.gz'
    assert sha(data_path) == manifest['raw_files'][data_path.name]['sha256']
    rows = json.loads(gzip.decompress(data_path.read_bytes()))
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_bytes())
    protocol['_root'] = str(ROOT)
    checked = [independent_episode(row, protocol) for row in rows]
    assert len(rows) == 2048 and [row['env_id'] for row in rows] == list(range(2048))
    court = json.loads((ROOT / 'configs/court_net_v4.json').read_bytes())
    configuration = json.loads((ROOT / 'configs/g07_full_racket_residual_ppo_joint_placement_v4.json').read_bytes())
    settings = configuration.pop('joint_placement_reward')
    assert configuration == json.loads((ROOT / 'configs/g07_full_racket_residual_ppo_v4.json').read_bytes())
    attacks = [bool(c['airborne_contact'] and row['released']
        and np.isfinite(row['outgoing_velocity_mps']).all()
        and np.linalg.norm(row['outgoing_velocity_mps']) >= 8
        and row['outgoing_velocity_mps'][0] > 0 and row['outgoing_velocity_mps'][2] <= -1)
        for row, c in zip(rows, checked)]
    tensor_flags = dict(ground_seen=torch.tensor([r['ground_seen'] for r in rows]),
        legal_air=torch.tensor([c['airborne_contact'] for c in checked]),
        attack=torch.tensor(attacks), crossed=torch.tensor([r['crossed_net'] for r in rows]),
        faulted=torch.tensor([c['fault'] for c in checked]),
        first_landing=torch.tensor([r['first_landing_m'] for r in rows]))
    reward = MeasuredJointPlacementRewardV4(len(rows), 'cpu', court, settings)
    original_bytes = json.dumps(rows, sort_keys=True).encode()
    bonus = reward.step(**tensor_flags)
    assert torch.isfinite(bonus).all() and bonus.min() >= 0 and bonus.max() <= settings['weight']
    assert reward.step(**tensor_flags).eq(0).all()
    assert json.dumps(rows, sort_keys=True).encode() == original_bytes
    checked_after = [independent_episode(row, protocol) for row in rows]
    assert checked_after == checked
    target_rows = [row for row in rows if row['landed_target']]
    assert len(target_rows) == 26
    assert all(bonus[row['env_id']] == 0 for row in target_rows)
    detail = []
    for row, check, attack, value in zip(rows, checked, attacks, bonus.tolist()):
        if not (attack or row['landed_target']):
            continue
        landing = np.asarray(row['first_landing_m'])
        finite = bool(np.isfinite(landing).all())
        dx = max(court['target_x_limits_m'][0] - landing[0], 0., landing[0] - court['target_x_limits_m'][1]) if finite else None
        dy = max(abs(landing[1]) - court['target_y_half_width_m'], 0.) if finite else None
        detail.append(dict(native_env_id=row['env_id'], source_TRAIN_case=row['source_training_index'],
            case_id=row['case_id'], air=check['airborne_contact'], attack=attack,
            crossed_net=row['crossed_net'], shuttle_net_contact=row['shuttle_net_contact'],
            first_landing_m=row['first_landing_m'] if finite else None,
            target_distance_m=float(np.hypot(dx, dy)) if finite else None,
            target=row['landed_target'], conditional_recovery=check['conditional_recovery'],
            complete=check['complete'], first_contact_m=row['contact_point_m'],
            outgoing_velocity_mps=row['outgoing_velocity_mps'],
            outgoing_speed_mps=float(np.linalg.norm(row['outgoing_velocity_mps'])),
            net_margin_m=row['net_margin_m'], recovery_s=row['jump']['recovery_s'],
            qualified_jump_landed=row['jump']['qualified_jump_landed'],
            fault_flags=[name for name in FAULTS if row[name]], replayed_candidate_bonus=value))
    attacks_detail = [row for row in detail if row['attack']]
    closest_recovered = min((row for row in attacks_detail if row['conditional_recovery']
        and row['target_distance_m'] is not None), key=lambda row: row['target_distance_m'])
    assert closest_recovered['native_env_id'] == 2022 and not closest_recovered['complete']
    counts = dict(episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
        conditional_recovery=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked), numerical_failures=sum(row['numerical_failures'] for row in rows),
        air_attacks=sum(attacks), targets=len(target_rows),
        attacks_with_observed_first_landing=sum(x['first_landing_m'] is not None for x in attacks_detail),
        attacks_past_target_x=sum(x['first_landing_m'] is not None and x['first_landing_m'][0] > court['target_x_limits_m'][1] for x in attacks_detail),
        attacks_outside_target_y=sum(x['first_landing_m'] is not None and abs(x['first_landing_m'][1]) > court['target_y_half_width_m'] for x in attacks_detail),
        target_hits_below_attack_speed=sum(np.linalg.norm(x['outgoing_velocity_mps']) < 8 for x in target_rows),
        targets_with_conditional_recovery=sum(checked[x['env_id']]['conditional_recovery'] for x in target_rows),
        positive_candidate_bonuses=int((bonus > 0).sum()))
    counts = {key: int(value) for key, value in counts.items()}
    files = ['quadsmash/control/measured_joint_placement_reward_v4.py',
        'quadsmash/tasks/stand/full_racket_residual_env_v4.py',
        'scripts/train_full_racket_residual_ppo_v4.py',
        'configs/g07_full_racket_residual_ppo_joint_placement_v4.json',
        'tests/test_measured_joint_placement_reward_v4.py']
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU static replay of preserved first-ground native event fields for a proposed TRAIN shaping term. No new robot execution or learning.',
        observed_original_counts=counts, candidate_settings=settings,
        measured_closest_recovered_attack_miss=closest_recovered,
        all_selected_native_records=detail, source_cohort_report_sha256=sha(source / 'REPORT.json'),
        source_first_ground_episodes_sha256=sha(data_path),
        candidate_sources={p: sha(ROOT / p) for p in files},
        original_episode_scores_unchanged=True, existing_config_and_PPO_settings_unchanged=True,
        term_is_opt_in=True, old_reward_resume_contract_remains_strict=True,
        native_candidate_training_run=False, candidate_frozen_policy_performance=None,
        hypothesis='A bonus conditioned jointly on a measured legal air attack, actual crossing and first-landing distance may give useful feedback where separate attack/target rewards have zero overlap. Only native learning and independent frozen assessment can test this.',
        limitations=['These terminal event fields are used for static bonus replay; this is not a reconstruction of historical per-control rewards.',
            'The cohort used changing weights and exploration. The reported near miss is a training episode, not a frozen successful policy.',
            'The new bonus is partial TRAIN feedback, not a completion label. Recovery and subsequent failures remain enforced by the original full-chain scoring.',
            'No trajectory, state, collision, physical parameter, case, success threshold or formal acceptance protocol is changed.',
            'The live original2048 job and its CPU video queue continue from another worktree with unchanged source fingerprints.',
            'Reward contracts differ: an old-reward canonical checkpoint cannot be passed as strict --resume with the new config. Use a clearly declared fresh candidate run or a separately reviewed transfer path.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    tmp = HERE / 'REPORT.json.tmp'
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, HERE / 'REPORT.json')
    print(json.dumps(dict(counts=counts, nearest_recovered_attack_miss_m=closest_recovered['target_distance_m'],
        candidate_bonus_for_that_miss=closest_recovered['replayed_candidate_bonus'], native_candidate_training_run=False)))


if __name__ == '__main__':
    main()
