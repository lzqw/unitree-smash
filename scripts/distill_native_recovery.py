"""Fit only the existing Recovery head from paired native, post-release feedback."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_physical_loop_loading_v1 import RECOVERY_NAMES
from quadsmash.rl.supermimic_v1 import load_ppo_initialized_actor
from scripts.recovery_native_teachers import NativeTeachers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    torch.manual_seed(7371)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    folder = args.probe
    rows = json.loads((folder/'episodes_summary.json').read_text())
    metrics = json.loads((folder/'metrics.json').read_text())
    assert metrics['evaluation_scope'] == 'Recovery training teacher branches' and len(rows) == 128
    assert not metrics['numerical_failures'] and (folder/'completed.json').is_file()
    checkpoint_digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert checkpoint_digest == metrics['evaluated_checkpoint_sha256']
    profiles = json.loads((folder/'actuator_contract.json').read_text())['lane_profile']
    tape = torch.load(folder/'teachers.pt', weights_only=True)
    items, teachers, anchors = [], [], []
    feedback = NativeTeachers('cpu')

    def add(observations, offsets, weight):
        if len(observations):
            items.append((observations, offsets, weight))

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
            add(x, torch.zeros_like(x[:,:18]), 3.)
            anchors.append(base['case_id'])
        if best:
            _, _, _, j, scale = min(best)
            m = tape['gate'][:,j]
            add(tape['obs'][m,j], tape['delta'][m,j], 1.)
            req = tape['requested'][mask,i]
            add(x, (req + scale * feedback.feedback(x)).clamp(-1,1)-req, .5)
            teachers.append(dict(case=base['case_id'], branch=j//32, before=base['strict_run_s'],
                                 after=rows[j]['strict_run_s'], companion=bool(i%2)))
    report = dict(teachers=teachers, anchors=anchors, input_sha256=checkpoint_digest)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'support.json').write_text(json.dumps(report, indent=2))
    if not teachers:
        raise ValueError('No admissible native teachers')
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    actor, _ = load_ppo_initialized_actor(Path(ckpt['metadata']['init_policy_checkpoint']['path']), device)
    actor.load_state_dict(ckpt['actor'])
    for name, param in actor.named_parameters():
        param.requires_grad_(name in RECOVERY_NAMES)
    rec = actor.actor.recovery
    initial = {k: p.detach().clone() for k,p in rec.named_parameters()}
    opt = torch.optim.Adam(rec.parameters(), lr=8e-5)
    x = torch.cat([item[0] for item in items]).to(device)
    with torch.no_grad():
        z = actor.actor.normalizer(x)
        old = rec(z).detach()
        target = old + torch.cat([item[1] for item in items]).to(device)/actor.actor.recovery_scale
    w = torch.cat([torch.full((len(v),), weight / len(v)) for v, _, weight in items]).to(device)
    col = torch.ones(z.shape[-1], device=device)
    col[:9] = .3
    col[9:45] = .6
    col[57:61] = .3
    col[61:67] = 12
    col[776:794] = 10
    for _ in range(120):
        idx = torch.multinomial(w, 256, replacement=True)
        fit = (rec(z[idx]) - target[idx]).square().mean()
        prox = sum(((p-initial[k]).square() * (col if k=='0.weight' else 1.)).mean()
                   for k,p in rec.named_parameters())
        opt.zero_grad()
        (fit + .1*prox).backward()
        torch.nn.utils.clip_grad_norm_(rec.parameters(), .3)
        opt.step()
    result = {k:t.detach().cpu() for k,t in actor.state_dict().items()}
    changed = {k for k,v in result.items() if not torch.equal(v, ckpt['actor'][k])}
    assert changed and changed <= set(RECOVERY_NAMES) and all(torch.isfinite(result[k]).all() for k in changed)
    ckpt['actor'] = result
    report.update(updates=120, changed=sorted(changed), frames=len(z))
    ckpt['metadata'] = dict(ckpt['metadata'], recovery_distillation=report)
    torch.save(ckpt, args.output/'policy.pt')
    (args.output/'training.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
