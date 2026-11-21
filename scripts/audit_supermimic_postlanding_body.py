"""Compare actual and predicted post-landing body motion on complete held R16 policy episodes."""
import argparse
import json
from pathlib import Path

import torch

from quadsmash.rl.supermimic_ball_sequence_v1 import ball_transition
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from scripts.evaluate_supermimic_fresh_wm_panel import digest


@torch.no_grad()
def audit(args):
    read=lambda p: json.loads(p.read_text())
    r=read(args.decision)
    if not (r['passed'] and r['gpu_load_and_forward_passed']):
        raise ValueError('Expected the CUDA-verified repaired WM')
    path=Path(r['checkpoint'])
    if digest(path)!=r['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    ck=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    archive=ck['runtime_validation_archive']
    if digest(Path(archive['path']))!=archive['sha256']:
        raise ValueError('Recovery refresh changed')
    doc=torch.load(archive['path'],weights_only=True,map_location='cpu',mmap=True)
    receipt=doc['receipt']
    records=read(Path(archive['path']).parent/'episodes_summary.json')
    if (r['actor_sha256']!=tensor_digest(ck['actor'].items()) or receipt['actor_sha256']!=r['actor_sha256']
            or any(rec['actor_sha256']!=r['actor_sha256'] or rec['policy_id']!=doc['policy_id']
                   or rec['episode_index']!=receipt['episode_ids'][rec['env_id']] for rec in records)):
        raise ValueError('Native labels or frozen actor mismatch')
    meta=ck['metadata']['physical_imagination']
    names=meta['checkpoint_paths']
    argv=read(path.parent/'startup.json')['argv']
    option=lambda name:Path(argv[argv.index('--'+name)+1])
    _,adapter,_,launch,_=load_physical_loop(archive=option('archive'),safety_mapping=option('safety-mapping'),
         physical_checkpoint=names['physical'],ball_checkpoint=names['ball'],task_checkpoint=names['task'],
         action_ball_checkpoint=names.get('action_ball'),execution_root=meta['g07_execution_root'],device=args.device)
    if meta['checkpoint_sha256']!=adapter.manifest['checkpoint_sha256']:
        raise ValueError('Different physical contract')
    fit=read(Path(r['candidate'])/'fit.json')
    initial_sha=tensor_digest(adapter.state_dict().items())
    if initial_sha!=fit['initial_world_sha256'] or tensor_digest(ck['world'].items())!=fit['original_online_world_sha256']:
        raise ValueError('Block initial or original online WM differs from the replay')
    repaired=torch.load(Path(r['candidate'])/'online_world.pt',mmap=True,map_location='cpu',weights_only=True)
    if tensor_digest(repaired.items())!=r['world_sha256']:
        raise ValueError('Repaired world changed')
    d=doc['tensors']
    landing=~d['feet'].bool().any(-1)&d['next_feet'].bool().any(-1)&(d['event_state'][...,0]>.5)&(d['event_state'][...,1]>.5)
    cases=[]
    for rec in records:
        lane=rec['env_id']
        if lane not in ck['config']['wm_holdout_lanes'] or not rec['qualified_airborne_string_contact']:
            continue
        t=torch.where(landing[:receipt['lengths'][lane],lane])[0]
        if t.numel():
            cases.append((lane,int(t[0]),bool(rec['recovery_success'])))
    scores={}
    for name,w in [('block_start',None),('original_online',ck['world']),('repaired',repaired)]:
        if w is not None:
            adapter.load_state_dict(w,strict=True)
        scores[name]={}
        for offset in (1,16,32):
            # A horizon has identical starts and executed controls for all models.
            subset=[(l,t+offset,s) for l,t,s in cases if t+offset+64<=receipt['lengths'][l]]
            if not subset:
                raise ValueError('No complete native H64 post-landing windows')
            lanes,start,success=map(torch.tensor,zip(*subset))
            state=d['state'][start,lanes].to(args.device)
            feet=d['feet'][start,lanes].float().to(args.device)
            present=d['present'][start,lanes].to(args.device)
            values={}
            for i in range(64):
                tick=start+i
                state,feet,present=ball_transition(adapter.physical_model,adapter.ball_model,state,
                    d['targets'][tick,lanes].to(args.device),d['previous_action'][tick,lanes].to(args.device),feet,present,
                    d['elapsed'][tick,lanes].to(args.device),launch[lanes],action_model=adapter.action_ball_model)
                if i+1 not in (1,4,16,32,64):
                    continue
                truth=d['next_state'][tick,lanes]
                q=state.cpu()
                angle=2*torch.acos((q[:,3:7]*truth[:,3:7]).sum(-1).abs().clamp(max=1.))
                measured=d['next_feet'][tick,lanes].bool()
                support=feet.cpu()>=.5
                values[i+1]={}
                for label,choose in [('all',torch.ones_like(success)),('recovered',success),('failed',~success)]:
                    if not choose.any():
                        continue
                    values[i+1][label]=dict(cases=int(choose.sum()),root_attitude_mae_rad=float(angle[choose].mean()),
                       root_angular_mae_rad_s=float((q[choose,10:13]-truth[choose,10:13]).abs().mean()),
                       root_height_mae_m=float((q[choose,2]-truth[choose,2]).abs().mean()),
                       foot_contact_bit_error=float((support[choose]!=measured[choose]).float().mean()),
                       support_count_mae=float((support[choose].sum(-1)-measured[choose].sum(-1)).abs().float().mean()))
            scores[name][offset]=dict(windows=len(subset),lanes=lanes.tolist(),horizons=values)
    return dict(source_sha256=archive['sha256'],handoff_sha256=digest(args.decision),scores=scores,
                initial_world_sha256=initial_sha,original_online_sha256=fit['original_online_world_sha256'],repaired_sha256=r['world_sha256'],
                eligible=len(cases),actor_updates=0,scope='Recorded commands. Windows start after the first post-release landing in held complete episodes; observational dynamics error, no policy improvement test.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('decision','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cpu')
    args=p.parse_args()
    torch.set_num_threads(2)
    result=audit(args)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
