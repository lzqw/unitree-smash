#!/usr/bin/env python3
"""Replay held executed commands with either the Body WM or its PD-predicted arm."""
import argparse
import json
from pathlib import Path

import torch

from audit_supermimic_actuator_prior import predict
from audit_supermimic_paired_body_transfer import file_hash, response_metrics
from quadsmash.rl.supermimic_action_identification_dataset_v1 import _load_branch, _same_contract
from quadsmash.rl.supermimic_native_pair_capture_v1 import BRANCHES
from quadsmash.rl.supermimic_physical_dataset_v1 import valid_windows
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_timebase_v2 import full_range_targets
from quadsmash.rl.supermimic_toi_racket_objective import racket_features


def advance_arm(output, state, target, last, torque, age, profile, fit):
    target = target.double()
    age = torch.where((target-last).abs() <= 1e-6, age+1, 0.)
    q, qd, torque = predict(state.double(), target, last, torque, age, profile,
                            'pd_history', fit['coefficients'], fit['control_dt_s'])
    output[:, 25:31] = q[:, 12:].to(output)
    output[:, 43:49] = qd[:, 12:].to(output)
    return target, torque, age


def forward_panel(args, body, fit, profile, drive, suite, task, train):
    if file_hash(args.packed) != fit['packed_sha256']:
        raise ValueError('Natural replay source differs from fit manifest')
    package = torch.load(args.packed, mmap=True, map_location='cpu', weights_only=True)
    if package['source_manifest']['task_contract'] != task:
        raise ValueError('Natural replay task mismatch')
    data = package['tensors']
    held = json.loads(args.validation_split.read_text())['validation_lanes']
    if not held or train.intersection(held) or len(set(held)) != len(held):
        raise ValueError('Invalid natural held-lane split')
    ball = BallResidualV2().eval()
    ball.load_state_dict(torch.load(args.ball, map_location='cpu', weights_only=True)['model'])
    targets = data['targets']
    same = (data['episode'][1:] == data['episode'][:-1]) & ~(data['terminated'] | data['timeout'])[:-1]
    hold = torch.zeros_like(targets)
    for t in range(1, len(hold)):
        held_target = same[t-1, :, None] & ((targets[t]-targets[t-1]).abs() <= 1e-6)
        hold[t] = torch.where(held_target, hold[t-1]+1, 0.)
    valid = valid_windows(data['episode'], data['terminated'], data['timeout'], 64)
    valid[0] = False
    valid[1:] &= same[:len(valid)-1]
    pairs = []
    for lane in held:
        possible = valid[:, lane].nonzero().flatten()
        if len(possible) < 10:
            raise ValueError('Cannot sample ten native windows per lane')
        pairs.extend((int(t),lane) for t in possible[torch.linspace(0,len(possible)-1,10).long()])
    tick, lane = torch.tensor(pairs).T
    launches = torch.tensor([c['launch']['position_m']+c['launch']['velocity_mps']
                             for c in json.loads(suite.read_text())['cases']])[lane]
    result = {}
    for use_pd in (False, True):
        state, feet, present = (data[k][tick,lane] for k in ('state','feet','present'))
        last = targets[tick-1,lane].double()
        torque = data['torque'][tick-1,lane][:,drive['joint_indices']].double()
        age = hold[tick-1,lane].double()
        metrics = {}
        for i in range(64):
            ti = tick+i
            target = targets[ti,lane]
            predicted, feet, present = ball_transition(body, ball, state, target, data['previous_action'][ti,lane],
                                                        feet, present, data['elapsed'][ti,lane], launches)
            if use_pd:
                last, torque, age = advance_arm(predicted,state,target,last,torque,age,profile,fit)
            state = predicted
            if i+1 in (1,2,4,8,16,32,64):
                truth = data['next_state'][ti,lane]
                err = state-truth
                q = state[:,3:7]
                err[:,3:7] = torch.where((q*truth[:,3:7]).sum(-1,keepdim=True)<0, -q-truth[:,3:7], err[:,3:7])
                fk = racket_features(state,body.geometry)-racket_features(truth,body.geometry)
                metrics[i+1] = dict(body_rmse=float(err[:,:49].square().mean().sqrt()),
                    root_angular_mae=float(err[:,10:13].abs().mean()), racket_position_mae=float(fk[:,:3].abs().mean()),
                    racket_velocity_mae=float(fk[:,3:6].abs().mean()), racket_normal_mae=float(fk[:,6:9].abs().mean()),
                    racket_angular_mae=float(fk[:,9:12].abs().mean()), ball_velocity_mae=float(err[:,52:].abs().mean()),
                    ball_position_p95=float(err[:,49:52].norm(dim=-1).quantile(.95)))
        result['mixed_body_with_pd_arm' if use_pd else 'mixed_body'] = metrics
    return dict(windows=len(pairs), held_lanes=held, horizons=result,
                source_sha256={k:file_hash(getattr(args,k)) for k in ('ball','packed','validation_split')})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('body', 'actuator', 'drive', 'dataset', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('ball', 'packed', 'validation-split'):
        parser.add_argument('--' + key, type=Path)
    args = parser.parse_args()
    if any([args.ball,args.packed,args.validation_split]) and not all([args.ball,args.packed,args.validation_split]):
        parser.error('Natural forward check requires ball, packed data and held-lane split together')
    torch.set_num_threads(2)
    load = lambda path: torch.load(path, map_location='cpu', weights_only=True)
    ck, fit, dataset = map(load, (args.body, args.actuator, args.dataset))
    task = ck['source_manifest']['task_contract']
    drive = json.loads(args.drive.read_text())
    if (fit['schema'] != 'native-joint-pd-history-v1' or fit['mode'] != 'pd_history'
            or fit['task_contract'] != task or fit['drive_sha256'] != file_hash(args.drive)):
        raise ValueError('Incompatible actuator predictor')
    rows = dataset['rows']
    lanes = [r['lane_index'] for r in rows]
    train = set(ck['calibration']['train_lanes']) | set(fit['train_lanes'])
    if train.intersection(lanes):
        raise ValueError('PD transfer must use held-out probe lanes')
    body = PhysicalWorldV2(task['asset']['sha256']).eval()
    body.load_state_dict(ck['model'])
    suite = Path(task['suite']['path'])
    if file_hash(suite) != task['suite']['sha256']:
        raise ValueError('Launch suite differs')
    launch = torch.tensor([c['launch']['position_m'] + c['launch']['velocity_mps']
                           for c in json.loads(suite.read_text())['cases']])[lanes]
    profile = {k: torch.tensor(v, dtype=torch.float64) for k, v in drive['profiles'][0].items()
               if k != 'position_limits'}
    collection = Path(dataset['source_collection'])
    if file_hash(collection/'status.json') != dataset['source_status_sha256']:
        raise ValueError('Probe collection has changed')
    sources = json.loads((collection/'status.json').read_text())
    hashes = {(r['spec'], r['branch']): r['probe_sha256'] for r in sources['completed']}
    start = {}
    for name in sorted({r['spec_name'] for r in rows}):
        runs = {branch: _load_branch(collection/name, branch) for branch in BRANCHES}
        _same_contract(runs)
        for branch, run in runs.items():
            if run['task_contract'] != task or file_hash(collection/name/branch/'native_probe.pt') != hashes[name, branch]:
                raise ValueError('Invalid task or probe identity')
        raw = runs['minus']['raw']['start']['before']
        limits = raw['auxiliary']['command_limits']
        prev_target = full_range_targets(raw['physical']['final_action'], limits)
        for row in (r for r in rows if r['spec_name'] == name):
            lane = row['lane_index']
            start[row['sample_id']] = (prev_target[lane], raw['applied_torque'][lane])
            for branch in ('plus', 'minus'):
                seq = runs[branch]['raw']['step_sequence']
                torch.testing.assert_close(row['step_targets_'+branch],
                    torch.stack([s['final_joint_targets'][lane] for s in seq]), atol=0, rtol=0)
    last0, torque0 = [torch.stack(col).double() for col in zip(*(start[r['sample_id']] for r in rows))]
    column = lambda key: torch.stack([r[key] for r in rows])
    nominal = column('step_next_states_nominal_a')
    fk = lambda s: racket_features(s, body.geometry)
    spans = dict(position=slice(0, 3), velocity=slice(3, 6), normal=slice(6, 9), angular=slice(9, 12))
    results = {}
    for use_pd in (False, True):
        predictions = {}
        for branch in ('minus', 'plus'):
            state, elapsed, present = map(column, ('state', 'elapsed', 'present'))
            previous, feet = column('previous_action'), column('foot_contact').float()
            last, torque = last0, torque0
            age = torch.zeros_like(last)
            predictions[branch] = []
            for step, target in enumerate(column('step_targets_'+branch).unbind(1)):
                output, logits, present = body(state, target, previous, feet, present, elapsed, launch)
                if use_pd:
                    last, torque, age = advance_arm(output,state,target,last,torque,age,profile,fit)
                predictions[branch].append(fk(output))
                state = torch.cat((output[:, :49], nominal[:, step, 49:]), -1)
                previous = column('step_final_actions_'+branch)[:, step]
                feet = logits.sigmoid()
                elapsed = elapsed + fit['control_dt_s']
        metrics = {}
        for step in range(nominal.shape[1]):
            truth = fk(column('step_next_states_plus')[:, step]) - fk(column('step_next_states_minus')[:, step])
            noise = fk(column('step_next_states_nominal_b')[:, step]) - fk(nominal[:, step])
            delta = predictions['plus'][step] - predictions['minus'][step]
            metrics[step+1] = {k: response_metrics(truth[:, sl], delta[:, sl], noise[:, sl]) for k, sl in spans.items()}
        results['mixed_body_with_pd_arm' if use_pd else 'mixed_body'] = metrics
    natural = forward_panel(args,body,fit,profile,drive,suite,task,train) if args.packed else None
    with args.output.open('x') as out:
        json.dump(dict(scope='held executed-command sequences, common nominal ball trajectory; six arm joints use the offline PD prior; pre-intervention hold age lower bound zero',
                       rows=len(rows), lanes=lanes, models=results, natural_forward=natural, actor_updates=0, contact_bptt_admitted=False,
                       sources={k: file_hash(getattr(args, k)) for k in ('body', 'actuator', 'drive', 'dataset')}),
                  out, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
