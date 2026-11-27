"""Fit independent force members on native post-release rollouts and score H32 touchdown."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from quadsmash.rl.dynamimic_lagrangian_v1 import rotation
from quadsmash.rl.dynamimic_recovery_wm_v1 import DynaMimicRecoveryWorldModel, local_robot_state
from quadsmash.rl.supermimic_full_body_geometry_v1 import FullBodyGeometryV1


def read_data(root, extra_fields=()):
    dest = root/'trajectories'
    manifest = json.loads((dest/'manifest.json').read_text())
    if manifest['schema'] != 'dynamimic-native-recovery-v1':
        raise ValueError('Unexpected trajectory schema')
    names = ('before','after','gate','live','body','landed','feet','final_joint_targets', *extra_fields)
    blocks = {key: [] for key in names}
    for name in manifest['chunks']:
        chunk = torch.load(dest/name, weights_only=True, mmap=True)
        for key in names:
            blocks[key].append(chunk[key])
    data = {key:torch.cat(value) for key, value in blocks.items()}
    origins = torch.tensor(manifest['origins'])
    for key in ('before','after'):
        data[key][..., :3] -= origins
        data[key][..., 49:52] -= origins
    return data, manifest


def sampling_pools(data, held, horizon):
    eligible = data['live'] & data['gate']
    usable = eligible.unfold(0, horizon, 1).all(-1)
    landed = data['landed'].bool()
    # At least one sample on each side of the rising edge of the touchdown latch.
    crossing = usable & ~landed[:len(usable)] & landed[horizon-1:]
    return (usable & ~held).nonzero(), (crossing & ~held).nonzero(), (crossing & held).nonzero()


def center_z(geometry, q, v):
    rot = rotation(q[:, 3:7])
    return geometry(q[:, 7:], v[:, 6:], q[:, :3], q[:, 3:7],
                    (rot @ v[:, :3,None]).squeeze(-1),
                    (rot @ v[:, 3:6,None]).squeeze(-1))['center'][:, 2]


def rollout(wm, member, data, rows, horizon, device, profiles, geometry=None):
    tick, lane = rows.T
    times = tick[None] + torch.arange(horizon)[:, None]
    q, v = local_robot_state(data['before'][tick, lane].to(device))
    targets = data['final_joint_targets'][times, lane].to(device)
    truth = data['after'][times, lane].to(device)
    feet = data['feet'][times, lane].to(device).float()
    ground = data['body'][times, lane].to(device).float()
    profile = profiles[lane].to(device)
    loss = q.new_zeros(())
    results = []
    for t in range(horizon):
        q, v, aux = wm(q, v, targets[t], member, profile)
        q_ref, v_ref = local_robot_state(truth[t])
        r, r_ref = rotation(q[:, 3:7]), rotation(q_ref[:, 3:7])
        p = aux['foot_contact'].clamp(1e-6, 1-1e-6)
        step = (20*F.smooth_l1_loss(q[:, :3], q_ref[:, :3]) +
                5*F.mse_loss(r, r_ref) + F.smooth_l1_loss(v[:, :6], v_ref[:, :6]) +
                5*F.smooth_l1_loss(q[:, 7:], q_ref[:, 7:]) +
                F.smooth_l1_loss(v[:, 6:], v_ref[:, 6:]) +
                F.binary_cross_entropy(p, feet[t]) +
                F.binary_cross_entropy_with_logits(aux['body_ground_logit'], ground[t]) +
                .001*aux['residual_force'].square().mean())
        loss = loss + step/horizon
        if geometry is not None:
            # Angle between gravity directions is invariant to yaw and bounds tilt error.
            tilt = torch.rad2deg(torch.acos((r[:, 2]*r_ref[:, 2]).sum(-1).clamp(-1,1)))
            results.append(torch.stack((tilt, (center_z(geometry,q,v)-center_z(geometry,q_ref,v_ref)).abs(),
                ((p >= .5) == feet[t].bool()).float().mean(-1),
                (v[:, 6:]-v_ref[:, 6:]).abs().mean(-1)), -1))
    return loss, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native','asset','output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--steps-per-horizon', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--history-native', type=Path, action='append', default=[])
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps_per_horizon < 1 or args.batch_size < 2:
        parser.error('Positive updates and at least two starts per batch required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7372)
    data, manifest = read_data(args.native)
    contract = json.loads((args.native/'actuator_contract.json').read_text())
    if len(args.history_native) > 3 or len(set([args.native, *args.history_native])) != 1+len(args.history_native):
        raise ValueError('Expected up to four distinct native archives')
    latest_length = len(data['live'])
    prior_data = []
    for directory in args.history_native:
        history, history_manifest = read_data(directory)
        other_contract = json.loads((directory/'actuator_contract.json').read_text())
        if history_manifest['cases'] != manifest['cases'] or other_contract != contract:
            raise ValueError('Historical case split or actuator contract changed')
        prior_data.append(history)
    if prior_data:
        data = {k:torch.cat([h[k] for h in [data,*prior_data]]) for k in data}
    profiles = torch.tensor(contract['lane_profile'])
    held = torch.arange(len(profiles)) % 16 < 2
    wm = DynaMimicRecoveryWorldModel(args.asset, contract).to(args.device)
    if args.resume:
        wm.load_state_dict(torch.load(args.resume, weights_only=True, map_location=args.device))
    geometry = FullBodyGeometryV1(wm.body.asset_sha256).to(args.device)
    geometry.enable_depth_batches()
    pools = {k:sampling_pools(data, held, k) for k in (4, 16, 32)}
    valid = pools[32][2]
    valid = valid[valid[:,0]+32 <= latest_length]
    # Only one window per independent held-out case; center each window around touchdown.
    lanes = valid[:, 1].unique()
    if len(lanes) == 0:
        raise ValueError('No independent held-out crossing windows')
    validation = torch.stack([valid[valid[:, 1] == lane][len(valid[valid[:, 1] == lane])//2] for lane in lanes])
    metrics = []
    horizon_errors = []
    per_case = []
    for member in range(3):
        optimizer = torch.optim.AdamW(wm.members[member].parameters(), lr=3e-4)
        generator = torch.Generator().manual_seed(2315+member)
        for horizon, (all_rows, crossing, _) in pools.items():
            if not len(crossing):
                raise ValueError(f'No train-side touchdown support at H{horizon}')
            for i in range(args.steps_per_horizon):
                n = args.batch_size//2
                rows = torch.cat([pool[torch.randint(len(pool), (size,), generator=generator)]
                                  for pool,size in ((crossing,n), (all_rows,args.batch_size-n))])
                optimizer.zero_grad(set_to_none=True)
                loss, _ = rollout(wm, member, data, rows, horizon, args.device, profiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wm.members[member].parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            print(f'Member {member}, H{horizon}, final train loss {float(loss):.6g}', flush=True)
        with torch.no_grad():
            _, per_tick = rollout(wm, member, data, validation, 32, args.device, profiles, geometry)
            # Terminal H32 body errors and all 32 steps of per-foot contact agreement.
            sequence = torch.stack(per_tick)
            horizon_errors.append({str(h):dict(tilt_median_deg=float(sequence[h-1,:,0].median()),
                com_height_mae_m=float(sequence[h-1,:,1].mean()), contact_accuracy=float(sequence[h-1,:,2].mean()),
                joint_speed_mae=float(sequence[h-1,:,3].mean())) for h in (1,4,8,16,32)})
            scores = per_tick[-1]
            per_case.append([dict(case_id=manifest['cases'][lane]['case_id'], start_tick=int(row[0]),
                                  tilt_deg=float(score[0]), height_absolute_error_m=float(score[1]),
                                  foot_accuracy=float(sequence[:,idx,2].mean()), joint_speed_mae=float(score[3]))
                             for idx,(lane,row,score) in enumerate(zip(lanes.tolist(),validation,scores))])
            metrics.append(dict(tilt_deg=float(scores[:,0].median()), com_height_mae_m=float(scores[:,1].mean()),
                                contact_accuracy=float(torch.stack(per_tick)[...,2].mean()),
                                joint_speed_mae=float(scores[:,3].mean())))
    passed = all(m['tilt_deg']<4 and m['com_height_mae_m']<.02 and m['contact_accuracy']>.9 and
                 m['joint_speed_mae']<1.2 for m in metrics)
    torch.save(wm.state_dict(), args.output/'ensemble.pt')
    (args.output/'validation.json').write_text(json.dumps(dict(passed=passed, members=metrics, ensemble_sha256=hashlib.sha256((args.output/'ensemble.pt').read_bytes()).hexdigest(), native_checkpoint=manifest['checkpoint_sha256'],
        sources=[str(args.native), *map(str,args.history_native)], horizon_errors=horizon_errors,
        per_case=per_case, train_pool_sizes={h:[len(a),len(b)] for h,(a,b,_) in pools.items()},
        held_cases=[manifest['cases'][i]['case_id'] for i in lanes.tolist()]),indent=2)+'\n')


if __name__ == '__main__':
    main()
