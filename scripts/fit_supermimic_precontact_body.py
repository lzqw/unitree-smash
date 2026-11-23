"""Fit event-aligned 20/32-step Body rollouts from an A32 native collection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from fit_supermimic_approach_body_response import native_data, racket_at_ball
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_racket_dynamics_v1 import racket_state, robot_loss
from run_supermimic_approach_v2 import dump


FIELDS = ('state', 'next_state', 'targets', 'previous_action', 'feet', 'present', 'elapsed')
HORIZONS = (20, 32)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def event_windows(data, terminals, holdout, horizon):
    """Supervise the native state immediately before contact, with no crossing of the event."""
    pools = {'string': [], 'frame': []}
    for stop, lane in data['new_contact'].nonzero().tolist():
        if lane in holdout or stop < horizon:
            continue
        episode = int(data['episode'][stop, lane])
        terminal = terminals.get((lane, episode))
        if terminal is None:
            continue
        kind = terminal['native_contact']['contact_type']
        if kind not in pools:
            continue
        start = stop - horizon
        ticks = slice(start, stop)
        if (not (data['episode'][ticks, lane] == episode).all()
                or not data['before_contact'][ticks, lane].all()
                or not data['present'][ticks, lane].all()
                or (data['terminated'][ticks, lane] | data['timeout'][ticks, lane]).any()
                or not (data['stream'][start:stop + 1] == data['stream'][stop]).all()
                or not (data['tick'][start:stop + 1].diff() == 1).all()):
            continue
        pools[kind].append((start, lane, episode))
    return pools


def body_rollout_loss(body, window, launch):
    state = window['state'][0]
    horizon = len(window['state'])
    losses = []
    endpoint_metrics = None
    for step in range(horizon):
        # Preserve BPTT through robot kinematics. The incoming shuttle and
        # control history are recorded inputs, independent of the Body fit.
        state = torch.cat((state[:, :49], window['state'][step, :, 49:]), -1)
        state, _, _ = body(
            state, window['targets'][step], window['previous_action'][step], window['feet'][step],
            window['present'][step], window['elapsed'][step], launch, detach_contacts=True)
        if step + 1 not in (4, 8, 16, horizon):
            continue
        truth = window['next_state'][step]
        predicted_pose = racket_state(state, body.geometry)
        target_pose = racket_state(truth, body.geometry)
        # Both twists are evaluated at the same measured shuttle center.
        ball = truth[:, 49:52]
        predicted_racket = racket_at_ball(predicted_pose, ball, full_orientation=True)
        target_racket = racket_at_ball(target_pose, ball, full_orientation=True)
        position = predicted_racket[:, :3] - target_racket[:, :3]
        surface = predicted_racket[:, 3:6] - target_racket[:, 3:6]
        rotation = predicted_racket[:, 6:15] - target_racket[:, 6:15]
        angular = predicted_racket[:, 15:18] - target_racket[:, 15:18]
        terms = (F.smooth_l1_loss(position / .025, torch.zeros_like(position)),
                 F.smooth_l1_loss(surface, torch.zeros_like(surface)),
                 F.smooth_l1_loss(rotation / .12, torch.zeros_like(rotation)),
                 F.smooth_l1_loss(angular / 3., torch.zeros_like(angular)))
        losses.append(sum(terms) / len(terms) + robot_loss(state, truth)
                      + F.smooth_l1_loss(state[:, :3], truth[:, :3], beta=.025)
                      + F.smooth_l1_loss(state[:, 10:13], truth[:, 10:13], beta=.1))
        endpoint_metrics = (position.norm(dim=-1).mean().detach(), surface.norm(dim=-1).mean().detach())
    # The final state is the contact-adjacent supervision reference.
    return (torch.stack(losses).mean() + losses[-1]) / 2, endpoint_metrics


def make_window(data, members, horizon, device):
    starts, lanes, _ = torch.tensor(members).T
    indices = starts[None] + torch.arange(horizon)[:, None]
    return {k: data[k][indices, lanes].to(device) for k in FIELDS}, lanes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('checkpoint', 'native-archive', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0', choices=('cpu','cuda:0'))
    parser.add_argument('--updates', type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.updates <= 128:
        parser.error('Choose an update count in [1,128]')
    torch.set_num_threads(4)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    task = source['metadata']['task_contract']
    suite = Path(task['suite']['path'])
    if sha256(suite) != task['suite']['sha256']:
        raise ValueError('Launch suite changed')
    actor_hash = tensor_digest(source['actor'].items())
    data, chunks = native_data(args.native_archive, task, actor_hash)
    holdout = set(source['config']['wm_holdout_lanes'])
    records = [json.loads(row) for row in (args.native_archive.parent/'native_episodes.jsonl').read_text().splitlines()]
    terminals = {(r['env_id'],r['episode_index']):r for r in records
                 if r['behavior_at_terminal']['actor_sha256'] == actor_hash}
    groups = {h: event_windows(data, terminals, holdout, h) for h in HORIZONS}
    if any(len(pools[k]) < 4 for pools in groups.values() for k in ('string','frame')):
        raise ValueError(f'Insufficient contact-aligned support: { {h:{k:len(v) for k,v in p.items()} for h,p in groups.items()} }')
    device = torch.device(args.device)
    launch = torch.tensor([r['launch']['position_m'] + r['launch']['velocity_mps']
                           for r in json.loads(suite.read_text())['cases']], device=device)
    body = PhysicalWorldV2(task['asset']['sha256']).to(device).eval().requires_grad_(False)
    prefix = 'physical_model.'
    original = {name.removeprefix(prefix): value for name,value in source['world'].items() if name.startswith(prefix)}
    body.load_state_dict(original, strict=True)
    body.body.requires_grad_(True)
    opt = torch.optim.Adam(body.body.parameters(), lr=3e-5)
    rng = torch.Generator().manual_seed(882072)
    used = Counter()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'fit.jsonl').open('x',buffering=1) as log:
        for step in range(args.updates):
            horizon = HORIZONS[step % len(HORIZONS)]
            batch = []
            for kind, cases in groups[horizon].items():
                selected = [cases[i] for i in torch.randperm(len(cases),generator=rng)[:8]]
                batch.extend(selected)
                used.update((horizon, kind, lane, episode) for _,lane,episode in selected)
            window,lanes=make_window(data,batch,horizon,device)
            opt.zero_grad(set_to_none=True)
            loss, (position, speed)=body_rollout_loss(body,window,launch[lanes.to(device)])
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(body.body.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            log.write(json.dumps(dict(update=step+1,horizon=horizon,loss=float(loss.detach()),
                racket_center_error_m=float(position),contact_point_speed_error_mps=float(speed),gradient=float(grad)),
                allow_nan=False)+'\n')
    candidate={name:value.cpu() for name,value in body.body.state_dict().items()}
    report=dict(schema='e1-v2-event-precontact-body-v1',parent_checkpoint_sha256=sha256(args.checkpoint),
        parent_actor_sha256=actor_hash,parent_world_sha256=tensor_digest(source['world'].items()),
        body_sha256=tensor_digest(candidate.items()),optimizer='fresh_body_only_adam',horizons=list(HORIZONS),
        updates=args.updates,train_lanes=sorted({k[2] for k in used}),holdout_lanes=sorted(holdout),
        actual_distinct_cases_by_horizon={h:{k:sum((hz,kind)==(h,k) for hz,kind,_,_ in used)
                                           for k in groups[h]} for h in HORIZONS},
        archive_chunks=chunks)
    temporary=args.output/'body.pt.pending'
    torch.save(dict(model=candidate,**report), temporary)
    temporary.replace(args.output/'body.pt')
    dump(args.output/'completed.json',report)


if __name__=='__main__':
    main()
