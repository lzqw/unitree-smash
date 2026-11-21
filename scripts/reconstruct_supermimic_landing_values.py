"""Reconstruct causal critic inputs on the two sides of first post-contact foot support."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_archive_storage_v1 import read_archive_rows
from quadsmash.rl.supermimic_physical_imagination_state_v1 import PhysicalImaginationStateV1
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from quadsmash.rl.supermimic_policy_execution_v1 import recovery_gate
from scripts.evaluate_supermimic_fresh_wm_panel import digest


@torch.no_grad()
def collect(run, directory, checkpoint, scratch, offsets=(0,1)):
    if len(set(offsets))!=len(offsets) or 0 not in offsets or any(t<0 for t in offsets):
        raise ValueError("Expected distinct nonnegative offsets including the landing transition")
    def read(path):
        return json.loads(path.read_text())
    ck = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    wm = ck['metadata']['physical_imagination']
    argv = read(run/'startup.json')['argv']
    names = wm['checkpoint_paths']
    _, adapter, template, launch, _ = load_physical_loop(
        archive=Path(argv[argv.index('--archive')+1]), safety_mapping=Path(argv[argv.index('--safety-mapping')+1]),
        physical_checkpoint=names['physical'], ball_checkpoint=names['ball'], task_checkpoint=names['task'],
        action_ball_checkpoint=names.get('action_ball'), execution_root=wm['g07_execution_root'], device='cpu')
    if wm['checkpoint_sha256'] != adapter.manifest['checkpoint_sha256']:
        raise ValueError('Critic reconstruction dependencies differ')
    adapter.load_state_dict(ck['world'])
    bank = torch.load(directory/'native_critic_labels.pt', weights_only=True, map_location='cpu')
    receipt = read(directory/'manifest.json')
    if (digest(directory/'complete_episodes.pt') != bank['source_sha256'] or
            receipt['actor_sha256'] != tensor_digest(ck['actor'].items()) or receipt['actor_sha256'] != bank['actor_sha256']):
        raise ValueError('Frozen execution identity differs')
    archive = torch.load(directory/'complete_episodes.pt', map_location='cpu', weights_only=True)['tensors']
    ids = {tuple(v): i for i,v in enumerate(bank['identity'])}
    landing = ((archive['event_state'][...,0] > .5) & (archive['event_state'][...,1] > .5)
               & ~archive['feet'].bool().any(-1) & archive['next_feet'].bool().any(-1))
    wanted, controls = set(), set()
    landing_by_lane={}
    for rec in read(directory/'episodes_summary.json'):
        if not rec['qualified_airborne_string_contact']:
            continue
        lane, ep = rec['env_id'], rec['episode_index']
        length = receipt['lengths'][lane]
        first = torch.where(landing[:length,lane])[0]
        if not len(first):
            continue
        t = int(first[0])
        landing_by_lane[lane]=t
        wanted.update((lane,ep,t+dt) for dt in offsets if t+dt<length)
        known = [key for key in ids if key[:2] == (lane, ep)]
        controls.add(min(known, key=lambda key: abs(key[2]-t)))
    pending = wanted | controls
    captured = {}
    start = 0
    for chunk in read(run/'native_transitions/manifest.json')['chunks']:
        end = start + chunk['ticks']
        overlaps = end > receipt['start_tick'] and start < receipt['end_tick']
        start = end
        if not overlaps:
            continue
        with read_archive_rows(run/'native_transitions', chunk, scratch=scratch) as rows:
            for row in rows:
                if row['policy_id'] != bank['policy_id']:
                    continue
                b = row['before']
                keys = [(lane, ep, t) for lane,(ep,t) in enumerate(zip(
                    b['episode_index'].tolist(),b['elapsed_tick'].tolist())) if (lane, ep, t) in pending]
                if not keys:
                    continue
                state = PhysicalImaginationStateV1.from_record(row, executor_template=template, launch=launch, device='cpu')
                inputs = adapter.critic_input(state)
                gate = recovery_gate(state.task_memory['rally'], state.observation).reshape(-1)
                for key in keys:
                    lane, ep, t = key
                    torch.testing.assert_close(state.physical[lane], archive['state'][t,lane], atol=0, rtol=0)
                    torch.testing.assert_close(row['final_joint_targets'][lane],archive['targets'][t,lane],atol=0,rtol=0)
                    if key in ids:
                        torch.testing.assert_close(inputs[lane], bank['inputs'][ids[key]], atol=1e-5, rtol=1e-5)
                        if gate[lane] != bank['head_masks']['recovery'][ids[key]]:
                            raise ValueError('Reconstructed recovery gate differs')
                    if key in wanted:
                        captured[key] = (inputs[lane].clone(), bool(gate[lane]))
                    pending.remove(key)
        if not pending:
            break
    if pending:
        raise ValueError(f'Missing frozen episode rows: {sorted(pending)}')
    order = sorted(captured)
    return dict(identity=order, inputs=torch.stack([captured[k][0] for k in order]),
                recovery=torch.tensor([captured[k][1] for k in order]), actor_sha256=bank['actor_sha256'],
                source_sha256=bank['source_sha256'], reproduced_sparse_inputs=len(controls), landing_by_lane=landing_by_lane, offsets=tuple(offsets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run','refresh','checkpoint','scratch','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--offsets',type=int,nargs='+',default=[0,1])
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = collect(args.run,args.refresh,args.checkpoint,args.scratch,args.offsets)
    temp = args.output.with_suffix('.pending')
    torch.save(result,temp)
    temp.replace(args.output)
    print('Recovered',len(result['identity']),'transition-side snapshots. Reproduced',result['reproduced_sparse_inputs'],'saved critic inputs.')
