"""Read-only CUDA audit of the single fixed candidate; no optimizer or Isaac."""
import hashlib
import json
from pathlib import Path

import torch
from quadsmash.rl.supermimic_action_ball_v1 import ActionConditionedBallV1, preimpact_hard_negatives
from quadsmash.rl.supermimic_ball_residual_v1 import ball_geometry_features
from quadsmash.rl.supermimic_ball_residual_v2 import BallResidualV2
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_wm_validation_v1 import event_aligned_windows


@torch.no_grad()
def main():
    root = Path('/home/autodrive/LZQW/experiments/supermimic_admission256_20261118_s5lbit6k')
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    paths = {
        'physical': Path('/tmp/supermimic_racket_dynamics_20261114_YRFhKS/run/world.pt'),
        'ball': Path('/tmp/supermimic_ball_sequence_20261114_axsonnit/run/ball.pt'),
        'candidate': root / 'action_ball_fit_fixed/action_ball_candidate.pt',
        'packed': Path('/home/autodrive/LZQW/experiments/supermimic_frozen_closure_20261117_223243/packed/data.pt'),
    }
    models = {key: torch.load(path, map_location='cpu', weights_only=True) for key, path in paths.items()}
    contract = models['physical']['source_manifest']['task_contract']
    base = PhysicalWorldV2(contract['asset']['sha256']).to(device).eval()
    base.load_state_dict(models['physical']['model'], strict=True)
    ball = BallResidualV2().to(device).eval()
    ball.load_state_dict(models['ball']['model'], strict=True)
    sweep = ActionConditionedBallV1().to(device).eval()
    sweep.load_state_dict(models['candidate']['model'], strict=True)
    metadata = models['packed']['tensors']
    data = {key: value.to(device) for key, value in metadata.items()}
    split = json.loads((root / 'fixed_split.json').read_text())
    suite_path = Path(contract['suite']['path'])
    if not suite_path.is_absolute():
        suite_path = Path.cwd() / suite_path
    assert hashlib.sha256(suite_path.read_bytes()).hexdigest() == contract['suite']['sha256']
    cases = json.loads(suite_path.read_text())['cases']
    launch = torch.tensor([c['launch']['position_m'] + c['launch']['velocity_mps'] for c in cases], device=device)
    groups = event_aligned_windows(metadata, split['validation_lanes'], horizon=64, offsets=(-1, 0, 1), maximum=128)
    report = {'device': torch.cuda.get_device_name(device), 'optimizer_updates': 0, 'native_steps': 0,
              'candidate_sha256': hashlib.sha256(paths['candidate'].read_bytes()).hexdigest(),
              'scope': 'Recorded native commands only. Oracle robot-next-state ablation is diagnostic, never rollout input.',
              'groups': {}}
    for offset, group in groups.items():
        pairs = group['starts'].to(device)
        t, lane = pairs.unbind(1)
        state, present = data['state'][t, lane], data['present'][t, lane]
        robot, _, _ = base(state, data['targets'][t, lane], data['previous_action'][t, lane],
                           data['feet'][t, lane], present, data['elapsed'][t, lane], launch[lane])
        prior, _ = base.prior(state, present, data['elapsed'][t, lane], launch[lane])
        legacy, _ = ball(ball_geometry_features(state, base.geometry), present)
        correction, details = sweep(state, robot, prior[:, 49:], base.geometry, present, details=True)
        # Read-only error attribution. This oracle is not evidence of deployable prediction.
        oracle_correction, _ = sweep(state, data['next_state'][t, lane], prior[:, 49:],
                                     base.geometry, present, details=True)
        baseline = prior[:, 49:] + legacy
        truth = data['next_state'][t, lane, 49:]
        candidate = baseline + correction
        rows = []
        for i, (tick, lane_id) in enumerate(pairs.tolist()):
            rows.append(dict(tick=tick, lane_id=lane_id, case_id=cases[lane_id].get('case_id', cases[lane_id].get('id')),
                episode_id=int(data['episode'][tick, lane_id]), support=float(details['support'][i]),
                sweep_gaps=details['gaps'][i].tolist(), velocity_before=state[i, 52:55].tolist(),
                velocity_true=truth[i, 3:].tolist(), velocity_legacy=baseline[i, 3:].tolist(),
                velocity_candidate=candidate[i, 3:].tolist(), velocity_correction=correction[i, 3:].tolist(),
                event_before=data['event_state'][tick, lane_id].tolist(),
                event_after=data['next_event_state'][tick, lane_id].tolist()))
        report['groups'][offset] = dict(count=len(pairs), support_mean=float(details['support'].mean()),
            correction_speed_norm_mean=float(correction[:, 3:].norm(dim=-1).mean()),
            legacy_velocity_rmse=float((baseline[:, 3:] - truth[:, 3:]).square().mean().sqrt()),
            candidate_velocity_rmse=float((candidate[:, 3:] - truth[:, 3:]).square().mean().sqrt()),
            oracle_robot_velocity_rmse=float((baseline[:, 3:] + oracle_correction[:, 3:] - truth[:, 3:]).square().mean().sqrt()),
            worse_cases=int(((candidate[:, 3:] - truth[:, 3:]).square().mean(-1) >
                            (baseline[:, 3:] - truth[:, 3:]).square().mean(-1)).sum()), rows=rows)

    # Reproduce the exact seeded sampling sequence without fitting anything.
    events = (data['next_event_state'][..., 1] > data['event_state'][..., 1]) | (
        data['next_event_state'][..., 6] > data['event_state'][..., 6])
    mask = torch.zeros(events.shape[1], dtype=torch.bool, device=device)
    mask[split['train_lanes']] = True
    eligible = data['present'].bool() & mask[None]
    preceding = preimpact_hard_negatives(events, data['episode'], data['terminated'], data['timeout'])
    following = torch.zeros_like(events)
    following[1:] = events[:-1] & (data['episode'][1:] == data['episode'][:-1]) & ~(
        data['terminated'][:-1].bool() | data['timeout'][:-1].bool())
    pools = [(eligible & events).nonzero(), (eligible & preceding & ~events).nonzero(),
             (eligible & ~events & ~preceding).nonzero()]
    gen = torch.Generator(device=device).manual_seed(20261117)
    counts = dict(contact=0, pre_contact=0, post_contact=0, total=0)
    post_indices = []
    for _ in range(400):
        pairs = torch.cat([pool[torch.randint(len(pool), (n,), device=device, generator=gen)]
                           for pool, n in zip(pools, (64, 32, 32))])
        t, lane = pairs.unbind(1)
        counts['contact'] += int(events[t, lane].sum())
        counts['pre_contact'] += int(preceding[t, lane].sum())
        counts['post_contact'] += int(following[t, lane].sum())
        counts['total'] += len(pairs)
        post_indices.extend(pairs[following[t, lane]].tolist())
    report['sampling'] = dict(pool_sizes=[len(p) for p in pools], draws=counts,
        distinct_post_contact_transitions=len(set(map(tuple, post_indices))),
        available_post_contact_transitions=int((following & eligible).sum()),
        labels_are_offline_indices_only=True)
    target = root / 'candidate_readonly_diagnosis.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({**report, 'groups': {k: {a: b for a, b in v.items() if a != 'rows'}
                                          for k, v in report['groups'].items()}}, indent=2))


if __name__ == '__main__':
    main()
