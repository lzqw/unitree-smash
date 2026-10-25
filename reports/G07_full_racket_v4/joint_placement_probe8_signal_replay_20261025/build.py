"""Replay the enabled shaping formula on all completed TRAIN episode records.

Terminal flags do not reconstruct the historical per-control reward stream.
This checks signal coverage, not how much reward was actually paid or whether
the new actor performs better. No robot execution or learning is launched.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
RUN = ROOT / 'reports/G07_full_racket_v4/residual_ppo_joint_placement_actor32_n2048_probe8_01'
sys.path.insert(0, str(ROOT))
from quadsmash.control.measured_joint_placement_reward_v4 import MeasuredJointPlacementRewardV4
from quadsmash.evaluation_v4 import FAULTS, independent_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (HERE / 'REPORT.json').exists()
    assert json.loads((RUN / 'job.json').read_text())['status'] == 'SUCCEEDED'
    inputs = json.loads((RUN / 'INPUTS.json').read_text())
    assert inputs['contract']['reward_version'] == 'native-full-chain-residual-v2-joint-placement'
    protocol = json.loads((ROOT / 'configs/g07_full_racket_eval_v4_r3.json').read_text())
    protocol['_root'] = str(ROOT)
    court = json.loads((ROOT / 'configs/court_net_v4.json').read_text())
    rows = json.loads((RUN / 'episodes.json').read_text())
    checked = [independent_episode(row, protocol) for row in rows]
    counts = dict(completed_episodes=len(rows), air=sum(c['airborne_contact'] for c in checked),
        conditional_recovery_count=sum(c['conditional_recovery'] for c in checked),
        complete=sum(c['complete'] for c in checked), numerical_failures=sum(r['numerical_failures'] for r in rows))
    assert counts == json.loads((RUN / 'metrics.json').read_text())['counts']
    attacks = [bool(c['airborne_contact'] and r['released']
        and np.isfinite(r['outgoing_velocity_mps']).all()
        and np.linalg.norm(r['outgoing_velocity_mps']) >= 8.
        and r['outgoing_velocity_mps'][0] > 0. and r['outgoing_velocity_mps'][2] <= -1.)
        for r, c in zip(rows, checked)]
    flags = dict(ground_seen=torch.tensor([r['ground_seen'] for r in rows], dtype=torch.bool),
        legal_air=torch.tensor([c['airborne_contact'] for c in checked], dtype=torch.bool),
        attack=torch.tensor(attacks, dtype=torch.bool),
        crossed=torch.tensor([r['crossed_net'] for r in rows], dtype=torch.bool),
        faulted=torch.tensor([c['fault'] for c in checked], dtype=torch.bool),
        first_landing=torch.tensor([r['first_landing_m'] for r in rows], dtype=torch.float32))
    settings = inputs['contract']['joint_placement_reward']
    term = MeasuredJointPlacementRewardV4(len(rows), 'cpu', court, settings)
    bonus = term.step(**flags)
    assert torch.isfinite(bonus).all() and bonus.min() >= 0 and bonus.max() <= settings['weight']
    assert term.step(**flags).eq(0).all()
    assert bonus[~flags['attack']].eq(0).all()
    detail = []
    for i, (r, c, attack) in enumerate(zip(rows, checked, attacks)):
        if not (attack or r['landed_target']):
            continue
        point = np.asarray(r['first_landing_m'], dtype=float)
        velocity = np.asarray(r['outgoing_velocity_mps'], dtype=float)
        finite = bool(np.isfinite(point).all())
        dx = max(court['target_x_limits_m'][0]-point[0], 0., point[0]-court['target_x_limits_m'][1]) if finite else None
        dy = max(abs(point[1])-court['target_y_half_width_m'], 0.) if finite else None
        margin = float(r['net_margin_m'])
        detail.append(dict(case_id=r['case_id'], env_id=r['env_id'], episode_index=r['episode_index'],
            source_TRAIN_index=r['source_training_index'], air=c['airborne_contact'], attack=attack,
            crossed_net=r['crossed_net'], target=r['landed_target'], conditional_recovery=c['conditional_recovery'],
            complete=c['complete'], first_landing_m=point.tolist() if finite else None,
            target_distance_m=float(np.hypot(dx, dy)) if finite else None,
            outgoing_velocity_mps=velocity.tolist() if np.isfinite(velocity).all() else None,
            outgoing_speed_mps=float(np.linalg.norm(velocity)) if np.isfinite(velocity).all() else None,
            fault_flags=[name for name in FAULTS if r[name]],
            net_margin_m=margin if np.isfinite(margin) else None,
            retrospective_terminal_flag_bonus=float(bonus[i])))
    targets = [d for d in detail if d['target']]
    attacks_detail = [d for d in detail if d['attack']]
    positive = bonus[bonus > 0]
    clean_positive = [d for d in attacks_detail if d['retrospective_terminal_flag_bonus'] > 0]
    closest_recovered = sorted((d for d in clean_positive if d['conditional_recovery']),
        key=lambda d: d['target_distance_m'])
    signal = dict(air_attacks=sum(attacks), targets=len(targets),
        attack_target_overlap=sum(d['target'] for d in attacks_detail),
        targets_without_qualifying_air_attack=sum(not d['attack'] for d in targets),
        attacks_with_observed_landing=sum(d['first_landing_m'] is not None for d in attacks_detail),
        positive_retrospective_bonuses=int((bonus > 0).sum()),
        retrospective_bonuses_at_least_one=int((bonus >= 1.).sum()),
        replay_bonus_min_positive=float(positive.min()) if len(positive) else None,
        replay_bonus_median_positive=float(positive.median()) if len(positive) else None,
        replay_bonus_max=float(bonus.max()),
        closest_recovered_attack_miss=closest_recovered[0] if closest_recovered else None)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        scope='CPU retrospective terminal-event replay of the actually enabled joint-placement formula on all4039 completed native TRAIN records. Not actual per-control reward telemetry, a new rollout, frozen-policy performance, or a causal reward comparison.',
        source_run=str(RUN), source_episodes_sha256=sha(RUN / 'episodes.json'),
        source_metrics_sha256=sha(RUN / 'metrics.json'), all_counts_independently_recomputed=counts,
        settings=settings, signal_coverage=signal, all_attack_or_target_records=detail,
        source_program_sha256=sha(Path(__file__).resolve()),
        reward_implementation_sha256=sha(ROOT / 'quadsmash/control/measured_joint_placement_reward_v4.py'),
        repeated_replay_does_not_pay_same_ground_event=True,
        every_non_attack_receives_zero_replayed_joint_bonus=True,
        actual_per_control_paid_reward_reconstructed=False,
        limitations=['Final fault flags can differ from flags at first landing; original per-step seen-ground and reward components were not logged. Replayed values must not be called historical paid rewards.',
            'Counts use every closed record, including failures and repeated lanes. They are changing-policy TRAIN data, not a frozen population score.',
            'Bonus >=1 is a descriptive signal-amplitude summary, never a new success or acceptance threshold.',
            'Original attack, target, recovery, legality and full-completion requirements remain unchanged.'],
        formal_cases_read=False, formal_acceptance=False, goal_complete=False)
    temporary = HERE / 'REPORT.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(HERE / 'REPORT.json')
    print(json.dumps(dict(counts=counts, signal=signal), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
