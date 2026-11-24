"""Fit event-aligned 20/32-step Body rollouts from an A32 native collection."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from fit_supermimic_approach_body_response import native_data
from quadsmash.rl.supermimic_physical_loop_loading_v1 import tensor_digest
from quadsmash.rl.supermimic_physical_world_v2 import PhysicalWorldV2
from quadsmash.rl.supermimic_precontact_body import HORIZONS, body_rollout_loss, make_window
from run_supermimic_approach_v2 import dump


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


def load_event_source(checkpoint, archive, task, excluded):
    actor_hash = tensor_digest(checkpoint['actor'].items())
    data, chunks = native_data(archive, task, actor_hash)
    with (archive.parent / 'native_episodes.jsonl').open() as stream:
        terminals = {(r['env_id'], r['episode_index']): r for r in map(json.loads, stream)
                     if r['behavior_at_terminal']['actor_sha256'] == actor_hash}
    groups = {h: event_windows(data, terminals, excluded, h) for h in HORIZONS}
    if any(len(pools[k]) < 4 for pools in groups.values() for k in ('string', 'frame')):
        raise ValueError(f'Insufficient contact-aligned support in {archive}')
    return data, groups, chunks, actor_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('checkpoint', 'native-archive', 'output'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0', choices=('cpu','cuda:0'))
    parser.add_argument('--updates', type=int, default=64)
    parser.add_argument('--additional-checkpoint', type=Path)
    parser.add_argument('--additional-native-archive', type=Path)
    parser.add_argument('--exclude-diagnostic-lanes', type=int, nargs='*', default=[])
    parser.add_argument('--learning-rate', type=float, default=3e-5)
    parser.add_argument('--position-weight', type=float, default=1.)
    args = parser.parse_args()
    if not 1 <= args.updates <= 128 or not 0 < args.learning_rate <= 3e-5 or not 1 <= args.position_weight <= 6:
        parser.error('Choose a bounded update count, learning rate and position weight')
    if bool(args.additional_checkpoint) != bool(args.additional_native_archive):
        parser.error('Provide both additional collection inputs')
    torch.set_num_threads(4)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    task = source['metadata']['task_contract']
    suite = Path(task['suite']['path'])
    if sha256(suite) != task['suite']['sha256']:
        raise ValueError('Launch suite changed')
    holdout = set(source['config']['wm_holdout_lanes'])
    excluded = holdout | set(args.exclude_diagnostic_lanes)
    sources = [load_event_source(source, args.native_archive, task, excluded)]
    if args.additional_checkpoint:
        additional = torch.load(args.additional_checkpoint, map_location='cpu', weights_only=False, mmap=True)
        additional_task = additional['metadata']['task_contract']
        if (any(additional_task[key]['sha256'] != task[key]['sha256'] for key in ('suite', 'asset'))
                or set(additional['config']['wm_holdout_lanes']) != holdout):
            raise ValueError('Additional source changed launch suite, asset or holdout split')
        sources.append(load_event_source(additional, args.additional_native_archive, additional_task, excluded))
    actor_hash = sources[0][3]
    device = torch.device(args.device)
    launch = torch.tensor([r['launch']['position_m'] + r['launch']['velocity_mps']
                           for r in json.loads(suite.read_text())['cases']], device=device)
    body = PhysicalWorldV2(task['asset']['sha256']).to(device).eval().requires_grad_(False)
    prefix = 'physical_model.'
    original = {name.removeprefix(prefix): value for name,value in source['world'].items() if name.startswith(prefix)}
    body.load_state_dict(original, strict=True)
    body.body.requires_grad_(True)
    opt = torch.optim.Adam(body.body.parameters(), lr=args.learning_rate)
    rng = torch.Generator().manual_seed(882072)
    used = Counter()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'fit.jsonl').open('x',buffering=1) as log:
        for step in range(args.updates):
            horizon = HORIZONS[step % len(HORIZONS)]
            source_index = (step // len(HORIZONS)) % len(sources)
            data, groups, _, _ = sources[source_index]
            batch = []
            for kind, cases in groups[horizon].items():
                selected = [cases[i] for i in torch.randperm(len(cases),generator=rng)[:8]]
                batch.extend(selected)
                used.update((source_index, horizon, kind, lane, episode) for _,lane,episode in selected)
            window,lanes=make_window(data,batch,horizon,device)
            opt.zero_grad(set_to_none=True)
            loss, (position, speed)=body_rollout_loss(body,window,launch[lanes.to(device)],args.position_weight)
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
        updates=args.updates,train_lanes=sorted({k[3] for k in used}),holdout_lanes=sorted(holdout),
        excluded_diagnostic_lanes=sorted(excluded - holdout),
        learning_rate=args.learning_rate,position_weight=args.position_weight,
        actual_distinct_cases_by_horizon={h:{k:sum((hz,kind)==(h,k) for _,hz,kind,_,_ in used)
                                           for k in ('string','frame')} for h in HORIZONS},
        archive_chunks=[chunk for _,_,chunks,_ in sources for chunk in chunks],
        archive_actor_sha256=[sha for _,_,_,sha in sources])
    temporary=args.output/'body.pt.pending'
    torch.save(dict(model=candidate,**report), temporary)
    temporary.replace(args.output/'body.pt')
    dump(args.output/'completed.json',report)


if __name__=='__main__':
    main()
