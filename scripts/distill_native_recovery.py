"""Fit only the existing Recovery head from paired native, post-release feedback."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from scripts.recovery_native_teachers import NativeTeachers


def focus_columns(example):
    from quadsmash.tasks.stand.observation_contract import FRAME_SIZE, HISTORY_LENGTH, OBSERVATION_SIZE
    if example.shape[-1] != OBSERVATION_SIZE:
        raise ValueError('Recovery observation layout mismatch')
    col = example.new_ones(OBSERVATION_SIZE)
    frames = col[:HISTORY_LENGTH * FRAME_SIZE].view(HISTORY_LENGTH, FRAME_SIZE)
    frames[:, :9] = .3
    frames[:, 9:45] = .6
    frames[:, 57:61] = .3
    frames[:, 61:67] = 12.
    col[776:794] = 10.
    return col


def load_examples(folder, checkpoint_digest):
    rows = json.loads((folder/'episodes_summary.json').read_text())
    metrics = json.loads((folder/'metrics.json').read_text())
    assert metrics['evaluation_scope'] == 'Recovery training teacher branches' and len(rows) == 128
    assert not metrics['numerical_failures'] and (folder/'completed.json').is_file()
    assert checkpoint_digest == metrics['evaluated_checkpoint_sha256']
    profiles = json.loads((folder/'actuator_contract.json').read_text())['lane_profile']
    tape = torch.load(folder/'teachers.pt', weights_only=True)
    if tape.get('feedback_version') != 2:
        raise ValueError('Collect fresh phase-aware recovery probes')
    items, teachers, anchors = [], [], []
    feedback = NativeTeachers('cpu')

    def add(case, observations, offsets, weight):
        if len(observations):
            items.append((observations, offsets, weight, case))

    for i, base in enumerate(rows[:32]):
        mask = tape['gate'][:, i]
        if not base['qualified_airborne_string_contact'] or not mask.any():
            continue
        best = []
        for b, scale in enumerate((.5, 1., -.5), 1):
            j = i + 32*b
            row = rows[j]
            assert row['case_id'] == base['case_id'] and profiles[j] == profiles[i]
            if (not row['qualified_airborne_string_contact'] or row['rally_fault'] or row['body_ground_seen']
                    or row['robot_net_fault'] or base['ball_side_complete_except_recovery'] > row['ball_side_complete_except_recovery']
                    or base['physical_recovery_success'] > row['physical_recovery_success']):
                continue
            improved = ((row['strict_physical_recovery'] and not base['strict_physical_recovery'])
                        or (row['physical_recovery_success'] and not base['physical_recovery_success'])
                        or (not base['strict_physical_recovery'] and row['strict_run_s'] > base['strict_run_s'] + .15))
            margin = row['strict_run_s'] >= .85 or row['strict_run_s'] > base['strict_run_s'] + .15
            if improved and margin and tape['gate'][:,j].any():
                best.append((not row['strict_physical_recovery'], abs(scale), -row['strict_run_s'], j, scale))
        x = tape['obs'][mask, i]
        if base['strict_physical_recovery'] or (base['physical_recovery_success'] and not best):
            add(base['case_id'], x, torch.zeros_like(x[:,:18]), 3.)
            anchors.append(base['case_id'])
        if best:
            _, _, _, j, scale = min(best)
            m = tape['gate'][:,j]
            add(base['case_id'], tape['obs'][m,j], tape['delta'][m,j], 1.)
            req = tape['requested'][mask,i]
            delta = scale * feedback.feedback(x)
            delta[:, 12:] *= tape['arm_allowed'][mask,i,None]
            add(base['case_id'], x, (req + delta).clamp(-1,1)-req, .5)
            teachers.append(dict(case=base['case_id'], branch=j//32, before=base['strict_run_s'],
                                 after=rows[j]['strict_run_s'], companion=bool(i%2)))
    return items, teachers, anchors, [row['case_id'] for row in rows[:32]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path, action='append')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(7371)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    checkpoint_digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    items, teachers, anchors, cases = [], [], [], set()
    for folder in args.probe:
        examples, labels, keeps, ids = load_examples(folder, checkpoint_digest)
        if cases.intersection(ids) or len(set(ids)) != 32:
            raise ValueError('Duplicate training cases across probe shards')
        cases.update(ids)
        items.extend(examples)
        teachers.extend(labels)
        anchors.extend(keeps)
    report = dict(teachers=teachers, anchors=anchors, input_sha256=checkpoint_digest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'support.json').write_text(json.dumps(report, indent=2))
    if len(cases) != 256 or len(teachers) < 40 or len(anchors) < 40:
        raise ValueError('Need 256 independent launches and at least 40 teachers and 40 anchors')
    for group in ([row['case'] for row in teachers], anchors):
        if sum(case.startswith('B-') for case in group) * 2 < len(group):
            raise ValueError('Panel B support below half of the group')
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    actor, _ = load_ppo_initialized_actor(Path(ckpt['metadata']['init_policy_checkpoint']['path']), device)
    actor.load_state_dict(ckpt['actor'])
    for name, param in actor.named_parameters():
        param.requires_grad_(name in RECOVERY_NAMES)
    actor.install_release_arm_reference()
    rec = actor.actor.recovery
    initial = {k: p.detach().clone() for k,p in rec.named_parameters()}
    x = torch.cat([item[0] for item in items]).to(device)
    with torch.no_grad():
        z = actor.actor.normalizer(x)
        old = rec(z).detach()
        target = old + torch.cat([item[1] for item in items]).to(device)/actor.actor.recovery_scale
    w = torch.cat([torch.full((len(v),), weight / len(v)) for v, _, weight, _ in items]).to(device)
    col = focus_columns(z)
    fold_by_case = {}
    for panel in ('A-', 'B-'):
        ordered = sorted({case for _, _, _, case in items if case.startswith(panel)},
                         key=lambda case: hashlib.sha256(case.encode()).digest())
        fold_by_case.update({case: i % 5 for i,case in enumerate(ordered)})
    folds = torch.cat([torch.full((len(v),),fold_by_case[case]) for v,_,_,case in items]).to(device)
    soup = {k: torch.zeros_like(t) for k,t in initial.items()}
    for held in range(5):
        rec.load_state_dict(initial)
        opt = torch.optim.Adam(rec.parameters(), lr=8e-5)
        fold_w = w * (folds != held)
        for _ in range(120):
            idx = torch.multinomial(fold_w, 256, replacement=True)
            fit = (rec(z[idx]) - target[idx]).square().mean()
            prox = sum(((p-initial[k]).square() * (col if k=='0.weight' else 1.)).mean()
                       for k,p in rec.named_parameters())
            opt.zero_grad()
            (fit + .1*prox).backward()
            torch.nn.utils.clip_grad_norm_(rec.parameters(), .3)
            opt.step()
        with torch.no_grad():
            for k,p in rec.named_parameters():
                soup[k].add_(p, alpha=.2)
    rec.load_state_dict(soup)
    result = {k:t.detach().cpu() for k,t in actor.state_dict().items()}
    changed = {k for k,v in ckpt['actor'].items() if not torch.equal(v, result[k])}
    for key in result.keys() - ckpt['actor'].keys():
        assert key.startswith('release_arm_reference.')
        assert torch.equal(result[key], ckpt['actor'][key.replace('release_arm_reference.', 'actor.recovery.')])
    assert changed and changed <= set(RECOVERY_NAMES) and all(torch.isfinite(result[k]).all() for k in changed)
    ckpt['actor'] = result
    report.update(updates_per_fold=120, folds_by_case=fold_by_case, independent_cases=len(cases), changed=sorted(changed), frames=len(z))
    ckpt['metadata'] = dict(ckpt['metadata'], recovery_distillation=report)
    torch.save(ckpt, args.output/'policy.pt')
    (args.output/'training.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
