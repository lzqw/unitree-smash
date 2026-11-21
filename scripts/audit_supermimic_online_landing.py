"""Replay a collected training block on excluded lanes to measure post-landing WM drift."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_dataset_v1 import encode_physical, valid_windows
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest


def collect(directory, block):
    chunks = {c['file']: c for c in json.loads((directory/'manifest.json').read_text())['chunks']}
    records, ticks = [], []
    for source in block['source_chunks']:
        receipt = chunks[source['file']]
        if receipt['sha256'] != source['sha256']:
            raise ValueError('Source chunk changed')
        with read_archive_rows(directory, receipt, scratch=directory) as rows:
            for row in rows:
                if (row['policy_version'] != block['policy_version'] or row['stream_id'] != block['stream']
                        or row['collection_mode'] != 'mixed_training'):
                    raise ValueError('Source collection stream or policy differs')
                before, after = row['before'], row['after']
                if not torch.equal(before['episode_index'], after['episode_index']):
                    raise ValueError('Transition crosses an episode boundary')
                a, b = before['physical'], after['physical']
                values = dict(state=encode_physical(a), truth=encode_physical(b), targets=row['final_joint_targets'],
                              previous=a['final_action'], feet=a['foot_contact'].float(), next_feet=b['foot_contact'],
                              present=a['ball_present'].reshape(-1), elapsed=a['elapsed_s'].reshape(-1),
                              released=after['task']['had_contact'].reshape(-1).bool() & after['task']['released'].reshape(-1).bool(),
                              qualified=after['event_state'][:,1]>.5, episode=before['episode_index'].reshape(-1),
                              terminated=row['terminated'].reshape(-1), timeout=row['timeout'].reshape(-1))
                records.append({key: value.clone() for key, value in values.items()})
                ticks.append(row['native_tick'])
    if ticks != list(range(block['native_ticks'][0], block['native_ticks'][1]+1)):
        raise ValueError('Selected chunks do not cover the declared native interval')
    return {key: torch.stack([r[key] for r in records]) for key in records[0]}


def landing_starts(data, excluded_lanes):
    edge = data['released'] & ~data['feet'].bool().any(-1) & data['next_feet'].bool().any(-1)
    starts = {}
    for tick, lane in torch.nonzero(edge).tolist():
        if lane in excluded_lanes:
            episode = int(data['episode'][tick,lane])
            starts.setdefault((lane, episode), tick+1)
    return starts


def errors(state, feet, truth, native_feet):
    angle = 2*torch.acos(torch.nn.functional.cosine_similarity(state[:,3:7], truth[:,3:7]).abs().clamp(max=1))
    support = feet >= .5
    return dict(root_attitude_mae_rad=float(angle.mean()), root_attitude_p90_rad=float(angle.quantile(.9)),
                root_angular_mae_rad_s=float((state[:,10:13]-truth[:,10:13]).abs().mean()),
                root_height_mae_m=float((state[:,2]-truth[:,2]).abs().mean()),
                support_count_mae=float((support.sum(1)-native_feet.sum(1)).abs().float().mean()),
                foot_bit_error=float((support != native_feet).float().mean()))


@torch.no_grad()
def audit(args):
    read = lambda p: json.loads(p.read_text())
    load = lambda p: torch.load(p, mmap=True, map_location='cpu', weights_only=False)
    registration = read(args.arm_dir/'registration.json')
    cmd = registration['command']
    def option(name):
        return Path(cmd[cmd.index('--'+name)+1])
    inputs = {k.replace('-', '_'): option(k) for k in
              ('archive','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','action-ball-checkpoint')}
    _, wm, _, launch, _ = load_physical_loop(**inputs, execution_root=option('g07-execution-root'), device='cpu')
    wm.eval()
    initial = load(registration.get('initial_checkpoint', option('resume-training')))
    before, after = load(args.before), load(args.after)
    block = read(args.collection)
    held = before['config']['wm_holdout_lanes']
    if (tensor_digest(before['actor'].items()) != block['actor_sha256']
            or before['global_counters']['accepted'] != block['policy_version']
            or [before['native_ticks'], after['native_ticks']-1] != block['native_ticks']
            or held != after['config']['wm_holdout_lanes']):
        raise ValueError('WM snapshots, collection policy or excluded lanes do not match')
    data = collect(args.arm_dir/'train/native_transitions', block)
    starts = landing_starts(data, set(held))
    result = dict(behavior_actor_sha256=block['actor_sha256'], policy_version=block['policy_version'],
                  native_ticks=block['native_ticks'], chunks=block['source_chunks'], online_excluded_lanes=held,
                  eligible_lane_episodes=len(starts), windows={}, models={},
                  scope='First observed no-foot to support transition after racket release per lane/episode. '
                        'Starts at the next state, on lanes excluded from online fitting. Recorded commands; '
                        'free-running WM until the measured endpoint. These cases belong to the training launch suite.')
    for name, checkpoint in [('hold_landing_state', None), ('initial', initial),
                             ('before_block_fit', before), ('after_block_fit', after)]:
        model = dict(horizons={})
        if checkpoint is not None:
            wm.load_state_dict(checkpoint['world'], strict=True)
            model.update(world_updates=checkpoint['world_updates'], world_sha256=tensor_digest(wm.state_dict().items()))
        result['models'][name] = model
        for horizon in (1,4,16,32,64):
            if len(data['state']) < horizon:
                continue
            valid = valid_windows(data['episode'], data['terminated'], data['timeout'], horizon)
            points = [(lane, ep, t) for (lane, ep), t in starts.items()
                      if t < len(valid) and int(data['episode'][t,lane]) == ep and valid[t,lane]]
            result['windows'][horizon] = [[lane, ep, block['native_ticks'][0]+t] for lane, ep, t in points]
            if not points:
                continue
            lanes, _, start = map(torch.tensor, zip(*points))
            state, feet, present = (data[k][start,lanes] for k in ('state','feet','present'))
            if checkpoint is not None:
                for dt in range(horizon):
                    tick = start+dt
                    state, feet, present = ball_transition(
                        wm.physical_model, wm.ball_model, state, data['targets'][tick,lanes], data['previous'][tick,lanes],
                        feet, present, data['elapsed'][tick,lanes], launch[lanes], action_model=wm.action_ball_model)
            end = start+horizon-1
            model['horizons'][horizon] = dict(cases=len(points), qualified_at_landing=int(data['qualified'][start-1,lanes].sum()),
                **errors(state, feet, data['truth'][end,lanes], data['next_feet'][end,lanes].bool()))
    # Place all snapshots on the same observation, the first proposal using the refreshed WM.
    from scripts.report_supermimic_research_curve import axes
    with (args.arm_dir/'train/training_history.jsonl').open() as stream:
        fitted = result['models']['after_block_fit']['world_sha256']
        for line in stream:
            proposal = json.loads(line)
            if proposal['native_ticks'] == after['native_ticks'] and proposal['model_runtime_state_sha256'] == fitted:
                origin = initial['global_counters']['accepted'], initial['native_ticks']
                result['axes'] = axes(proposal, origin, len(launch))
                break
    return result


def publish(result, directory):
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(directory)) as writer:
        for model, data in result['models'].items():
            for h, metrics in data['horizons'].items():
                for name, value in metrics.items():
                    for axis, step in result['axes'].items():
                        writer.add_scalar(f'{axis}/H{h}/{model}/{name}', value, step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('arm-dir','collection','before','after','output'):
        parser.add_argument('--'+key, required=True, type=Path)
    parser.add_argument('--tensorboard', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = audit(args)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    if args.tensorboard:
        if 'axes' not in result:
            parser.error('No training proposal matches the observed native tick and refreshed WM')
        publish(result, args.tensorboard)
