#!/usr/bin/env python3
"""Score checkpoint dynamics on complete post-update native episodes withheld from WM fitting."""
import argparse
import json
from pathlib import Path

import torch

from scripts.evaluate_supermimic_fresh_wm_panel import assess, digest
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest


def held_episode_data(tensors, receipt, lanes):
    lengths=tensors['valid_lengths'].long()
    episodes=tensors['episode']
    if (episodes.shape!=(800,128) or lengths.tolist()!=receipt['lengths']
            or receipt['complete_lanes']!=128 or min(receipt['lengths'])<1
            or max(receipt['lengths'])>800 or len(set(lanes))!=len(lanes)
            or not lanes or min(lanes)<0 or max(lanes)>=128):
        raise ValueError('Invalid complete native episodes or held-out lane list')
    ids=torch.as_tensor(receipt['episode_ids'])
    tick=torch.arange(800)[:,None]
    live=tick<lengths
    done=tensors['terminated']|tensors['timeout']
    if (not ((episodes==ids)|~live).all() or not torch.equal(done&live,tick==lengths-1)):
        raise ValueError('Episode identity or terminal tick mismatch')
    keys=('state','next_state','targets','previous_action','feet','present','elapsed','episode','terminated','timeout')
    data={k:tensors[k][:,lanes] for k in keys}
    data['contact']=(tensors['next_event_state'][:,lanes,1]>tensors['event_state'][:,lanes,1])&live[:,lanes]
    data['episode']=torch.where(live[:,lanes],data['episode'],-1)
    data['timeout']=data['timeout']|~live[:,lanes]
    return data


@torch.no_grad()
def score_native_critics(packed,bank,pre,post,actor,source_digest,lanes):
    from quadsmash.rl.supermimic_dual_head_quality_v1 import split_native_value_rows
    from quadsmash.rl.supermimic_paired_evaluation_v1 import native_critic_td
    from quadsmash.rl.supermimic_v1 import MLP
    if (bank['actor_sha256']!=actor or bank['policy_id']!=packed['policy_id'] or
            bank['source_sha256']!=source_digest or bank['policy_execution']!=packed['receipt']['policy_execution']):
        raise ValueError('Critic labels do not correspond to the scored policy or recorded native episodes')
    _,held=split_native_value_rows(bank,lanes)
    ids=torch.tensor(bank['identity'])
    lane,episode,tick=ids.T
    tensors=packed['tensors']
    if not torch.equal(episode,tensors['episode'][tick,lane]) or (tick>=tensors['valid_lengths'][lane]).any():
        raise ValueError('Critic state and episode-indexed targets differ')
    gamma=post['config']['timebase']['gamma_effective']
    rewards=tensors['reward'].double()
    targets=torch.zeros_like(rewards)
    running=torch.zeros_like(rewards[0])
    for i in reversed(range(len(rewards))):
        live=i<tensors['valid_lengths']
        running=torch.where(live,rewards[i]+gamma*running,0.)
        targets[i]=running
    if not torch.equal(targets[tick,lane].float().reshape(-1),bank['rtg'].float().reshape(-1)):
        raise ValueError('Complete native RTGs do not reproduce saved Critic labels')
    value_scores={}
    critic=MLP(bank['inputs'].shape[-1],(512,256,128),1).eval()
    for name,ck in [('before',pre),('after',post)]:
        critic.load_state_dict(ck['critic'],strict=True)
        value=critic(bank['inputs']).reshape(-1)
        target=bank['rtg'].reshape(-1)
        stats={}
        for head,active in bank['head_masks'].items():
            take=held & active.bool().reshape(-1)
            error=value[take]-target[take]
            stats[head]=dict(samples=int(take.sum()),unique_lanes=int(lane[take].unique().numel()),
                             native_rtg_mae=float(error.abs().mean()) if len(error) else None,
                             native_rtg_rmse=float(error.square().mean().sqrt()) if len(error) else None)
        stats['td']=native_critic_td(value[held],target[held],ids[held].tolist(),gamma)
        value_scores[name]=stats
    return value_scores,len(ids)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','refresh','pre-checkpoint','post-checkpoint','candidate','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    pre=torch.load(args.pre_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    post=torch.load(args.post_checkpoint,map_location='cpu',weights_only=False,mmap=True)
    packed_path=args.refresh/'complete_episodes.pt'
    packed=torch.load(packed_path,map_location='cpu',weights_only=True,mmap=True)
    receipt=packed['receipt']
    hash_state=lambda state:tensor_digest(state.items())
    actor=hash_state(pre['actor'])
    expected=pre['metadata']['physical_imagination']
    fit=json.loads((args.candidate/'fit.json').read_text())
    candidate=torch.load(args.candidate/'online_world.pt',weights_only=True,map_location='cpu',mmap=True)
    after_refresh=fit.get('refresh_manifest_sha256') is not None
    fit_checkpoint=args.post_checkpoint if after_refresh else args.pre_checkpoint
    if after_refresh and (fit['pre_refresh_sha256']!=digest(args.pre_checkpoint)
            or fit['refresh_manifest_sha256']!=digest(args.refresh/'manifest.json')):
        raise ValueError('Candidate was trained on different frozen episodes')
    lanes=pre['config']['wm_holdout_lanes']
    if (hash_state(post['actor'])!=actor or receipt['actor_sha256']!=actor or
            receipt['policy_id']!=packed['policy_id'] or receipt['policy_execution']!=pre['metadata']['policy_execution'] or
            receipt['start_tick']!=pre['native_ticks'] or receipt['end_tick']!=post['native_ticks'] or
            post['metadata']['physical_imagination']['checkpoint_sha256']!=expected['checkpoint_sha256'] or
            pre['metadata']['task_contract']!=post['metadata']['task_contract'] or
            post['config']['wm_holdout_lanes']!=lanes or fit['excluded_lanes']!=lanes or
            fit['stage_checkpoint_sha256']!=digest(fit_checkpoint) or
            fit['online_world_sha256']!=hash_state(candidate) or fit['original_online_world_sha256']!=hash_state((post if after_refresh else pre)['world'])):
        raise ValueError('Frozen episode, actor, candidate, or excluded lanes differ from the live block')
    argv=json.loads((args.run/'startup.json').read_text())['argv']
    option=lambda name:Path(argv[argv.index('--'+name)+1])
    paths=expected['checkpoint_paths']
    _,adapter,_,launch,manifest=load_physical_loop(archive=option('archive'),physical_checkpoint=paths['physical'],
        ball_checkpoint=paths['ball'],task_checkpoint=paths['task'],safety_mapping=option('safety-mapping'),
        action_ball_checkpoint=paths.get('action_ball'),execution_root=expected['g07_execution_root'],device='cpu')
    if (manifest['task_contract']!=pre['metadata']['task_contract'] or
            hash_state(adapter.state_dict())!=fit['initial_world_sha256']):
        raise ValueError('Initial WM or task contract changed')
    data=held_episode_data(packed['tensors'],receipt,lanes)
    cases=[dict(launch=dict(position_m=launch[i,:3].tolist(),velocity_mps=launch[i,3:].tolist())) for i in lanes]
    horizons=[1,2,4,8,16,32,64]
    scores={}; windows=None
    adapter.eval()
    for name,state in [('deployed',None),('online_pre_refresh',pre['world']),('online_post_refresh',post['world']),('long_native_post_refresh' if after_refresh else 'long_native_candidate',candidate)]:
        if state is not None:
            adapter.load_state_dict(state,strict=True)
        report=assess(data,cases,None,adapter,horizons,'cpu',predictors=('candidate',))
        if windows is not None and windows!=report['windows']:
            raise ValueError('Models evaluated on different native windows')
        windows=report['windows']
        scores[name]=dict(world_sha256=hash_state(adapter.state_dict()),metrics=report['metrics']['candidate'])
    bank=torch.load(args.refresh/'native_critic_labels.pt',weights_only=True,map_location='cpu',mmap=True)
    source_digest=digest(packed_path)
    value_scores,label_count=score_native_critics(packed,bank,pre,post,actor,source_digest,lanes)
    result=dict(critic_scores=value_scores,critic_updates=post['critic_updates']-pre['critic_updates'],
                world_updates=post['world_updates']-pre['world_updates'],rtg_rows_reproduced=label_count,
                verified_behavior_policy=actor,actor_sha256=actor,source_sha256=source_digest,policy_id=packed['policy_id'],
                pre_sha256=digest(args.pre_checkpoint),post_sha256=digest(args.post_checkpoint),
                fit_sha256=digest(args.candidate/'fit.json'),lanes=lanes,
                windows=[dict(w,lane=lanes[w['lane']]) for w in windows],scores=scores,
                scope='First complete frozen R8 episodes on excluded lane IDs. Fixed task launches can recur across policy versions.')
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)


if __name__=='__main__':
    main()
