"""Replay measured commands up to the last state before each new native body contact."""
import argparse
import json
from pathlib import Path

import torch

from scripts.audit_supermimic_online_landing import collect, errors
from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest


def posture(state, feet):
    quat = torch.nn.functional.normalize(state[3:7], dim=0)
    tilt = (1 - 2 * quat[1:3].square().sum()).clamp(-1, 1).acos().rad2deg()
    return dict(tilt_deg=float(tilt), root_height_m=float(state[2]), support_bits=(feet >= .5).tolist())


@torch.inference_mode()
def replay(arm, events_path, checkpoint, horizons):
    config = json.loads((arm/'registration.json').read_text())
    source = json.loads(events_path.read_text())
    lanes = sorted(map(int, source['new_body_ground']))
    if not lanes or any(h < 1 for h in horizons):
        raise ValueError('Expected native body events and positive horizons')
    events = [source['new_body_ground'][str(i)] for i in lanes]
    versions = {r['policy_at_first_body_contact'] for r in events}
    if len(versions) != 1:
        raise ValueError('Select one collection policy per audit')
    version = versions.pop()
    ends = torch.tensor([r['state_before']['tick'] for r in events])
    if int(ends.min()) < max(horizons):
        raise ValueError('Insufficient native prefix for the requested horizons')
    directory = arm/'train/native_transitions'
    manifest = json.loads((directory/'manifest.json').read_text())
    receipts = []
    offset = 0
    for number, chunk in enumerate(manifest['chunks']):
        end = offset + chunk['ticks']
        if end > int(ends.min())-max(horizons) and offset <= int(ends.max()):
            if not receipts:
                start = offset
            if chunk['sha256'] != source['checked_chunk_sha256'][number]:
                raise ValueError('Native event source differs from the archived trajectory')
            receipts.append(chunk)
        offset = end
    if not receipts or offset <= int(ends.max()):
        raise ValueError('Requested windows are outside the committed archive')
    from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
    with read_archive_rows(directory, receipts[0], scratch=arm) as rows:
        tick0, stream = rows[0]['native_tick'], rows[0]['stream_id']
    data = collect(directory, dict(source_chunks=receipts, policy_version=version, stream=stream,
                                   native_ticks=[tick0, tick0+sum(c['ticks'] for c in receipts)-1]))
    index = torch.tensor(lanes)
    end = ends - start
    cmd = config['command']
    options = {k.replace('-', '_'): Path(cmd[cmd.index('--'+k)+1]) for k in
               ('archive','physical-checkpoint','ball-checkpoint','task-checkpoint','safety-mapping','action-ball-checkpoint')}
    _, wm, _, launch, _ = load_physical_loop(**options, execution_root=Path(cmd[cmd.index('--g07-execution-root')+1]), device='cpu')
    ck = torch.load(checkpoint, mmap=True, map_location='cpu', weights_only=False)
    wm.load_state_dict(ck['world'], strict=True)
    wm.eval()
    result = dict(world_updates=ck['world_updates'], world_sha256=tensor_digest(wm.state_dict().items()),
                  collection_policy_version=version, checkpoint_actor_sha256=tensor_digest(ck['actor'].items()),
                  chunks=[c['sha256'] for c in receipts], excluded_from_online_fit=ck['config']['wm_holdout_lanes'],
                  scope='Retrospective native failure windows with recorded final targets. Later WM snapshots may have fitted training lanes. Measures forward posture and support only.', horizons={})
    for horizon in horizons:
        t = end - horizon
        state, feet, present = (data[k][t,index] for k in ('state','feet','present'))
        initial = [posture(s, f) for s, f in zip(state, feet)]
        for j in range(horizon):
            at = t+j
            if (data['episode'][at,index].any() or data['terminated'][at,index].any() or data['timeout'][at,index].any()
                    or not data['released'][at,index].all()):
                raise ValueError('Window needs uninterrupted post-release first-episode transitions')
            state, feet, present = ball_transition(wm.physical_model, wm.ball_model, state, data['targets'][at,index],
                data['previous'][at,index], feet, present, data['elapsed'][at,index], launch[index], action_model=wm.action_ball_model)
        truth, support = data['state'][end,index], data['feet'][end,index].bool()
        metrics = errors(state, feet, truth, support)
        cases = []
        for j, lane in enumerate(lanes):
            actual = posture(truth[j], support[j])
            reference = events[j]['state_before']
            if (abs(actual['tilt_deg']-reference['tilt_deg']) > .01
                    or actual['support_bits'] != reference['support_bits']):
                raise ValueError('Endpoint does not reproduce the recorded event geometry')
            cases.append(dict(lane=lane, start=initial[j], predicted=posture(state[j],feet[j]),
                              predicted_foot_probabilities=feet[j].tolist(), actual=actual))
        result['horizons'][horizon] = dict(metrics=metrics,cases=cases)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('arm','events','checkpoint','output'):
        cli.add_argument('--'+key, required=True, type=Path)
    args = cli.parse_args()
    torch.set_num_threads(2)
    result = replay(args.arm,args.events,args.checkpoint,(1, 4, 20, 40))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
