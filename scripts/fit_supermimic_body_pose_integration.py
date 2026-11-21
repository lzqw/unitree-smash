"""Calibrate pose/velocity consistency on paired native trajectories without a ball-side gradient."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.functional import normalize

from quadsmash.rl.supermimic_impact_timing_v1 import quaternion_world_derivative
from scripts.fit_supermimic_paired_racket_center import validate
from scripts.fit_supermimic_mixed_body_rollout import event_pools
from scripts.supermimic_paired_body_candidate import development_window, merged, paired_terms
from quadsmash.rl.supermimic_training_probe_replay import training_windows
from quadsmash.rl.supermimic_physical_loop_loading_v1 import load_physical_loop, tensor_digest
from scripts.evaluate_supermimic_fresh_wm_panel import digest


def consistent_pose(state, prediction, dt, root_weight, rotation_weight, arm_weight):
    """Blend learned endpoint poses with the integral of their predicted physical velocities."""
    root = prediction[:, :3]
    orientation = prediction[:, 3:7]
    arm = prediction[:, 25:31]
    if root_weight:
        root = torch.lerp(root, state[:,:3] + .5*dt*(state[:,7:10]+prediction[:,7:10]), root_weight)
    if rotation_weight:
        q = state[:,3:7]
        other = torch.where((q*orientation).sum(1,keepdim=True)<0,-orientation,orientation)
        end = normalize(q + .5*dt*(quaternion_world_derivative(q,state[:,10:13])
                                  + quaternion_world_derivative(other,prediction[:,10:13])),dim=1)
        orientation = normalize(torch.lerp(other,end,rotation_weight),dim=1)
        orientation = torch.where((orientation*prediction[:,3:7]).sum(1,keepdim=True)<0,-orientation,orientation)
    if arm_weight:
        arm = torch.lerp(arm, state[:,25:31] + .5*dt*(state[:,43:49]+prediction[:,43:49]), arm_weight)
    return torch.cat((root,orientation,prediction[:,7:25],arm,prediction[:,31:]),1)


class IntegratedBody(torch.nn.Module):
    def __init__(self, model, weights):
        super().__init__()
        self.model = model
        self.weights = weights

    @property
    def geometry(self):
        return self.model.geometry

    def prior(self, *args):
        return self.model.prior(*args)

    def forward(self, state, *args):
        pred, contact, present = self.model(state,*args)
        return consistent_pose(state,pred,self.model.dt,*self.weights), contact, present


def ratio(score, base, keys):
    return sum(row[k]/max(base[h][k],1.e-12) for h,row in score.items() for k in keys)/(len(score)*len(keys))


@torch.no_grad()
def run(args):
    read=lambda p:json.loads(p.read_text())
    decision=read(args.decision)
    ckpath=Path(decision['checkpoint'])
    ck=torch.load(ckpath,mmap=True,weights_only=False,map_location='cpu')
    replay=Path(decision['candidate'])
    fit=read(replay/'fit.json')
    if not decision['passed'] or digest(ckpath)!=fit['stage_checkpoint_sha256'] or digest(ckpath)!=decision['checkpoint_sha256']:
        raise ValueError('The repaired model needs its same-policy native verification')
    argv=read(ckpath.parent/'startup.json')['argv']
    option=lambda key:Path(argv[argv.index('--'+key)+1])
    meta=ck['metadata']['physical_imagination']
    deps=meta['checkpoint_paths']
    _,ad,_,launch,manifest=load_physical_loop(archive=option('archive'),physical_checkpoint=deps['physical'],
         ball_checkpoint=deps['ball'],task_checkpoint=deps['task'],action_ball_checkpoint=deps.get('action_ball'),
         safety_mapping=option('safety-mapping'),execution_root=meta['g07_execution_root'],device=args.device)
    if ad.action_ball_model is not None:
        raise ValueError('This calibration implements the verified detached ball transition')
    world=torch.load(replay/'online_world.pt',mmap=True,map_location='cpu',weights_only=True)
    if tensor_digest(world.items())!=decision['world_sha256'] or digest(option('packed-data'))!=fit['packed_sha256']:
        raise ValueError('WM or native dataset differs')
    ad.load_state_dict(world)
    data=torch.load(option('packed-data'),weights_only=True,mmap=True,map_location='cpu')['tensors']
    task=manifest['task_contract']; execution=ck['metadata']['policy_execution']
    held=fit['excluded_lanes']
    sets,proof=training_windows(args.training,task,execution,held)
    (ids,signed),=sets
    if sorted(ids)!=sorted(set(range(128))-set(held)) or len(ids)!=45:
        raise ValueError('Expected the complete signed training panel, including misses')
    labels,dev=development_window(args.development,task,execution,held)
    trainwin=merged(signed,torch.arange(len(ids)),args.device)
    devwin=merged(dev,torch.arange(len(labels)),args.device)
    # Only fitting uses training cases. The native held panel is evaluated after the single choice.
    mask=torch.ones(len(launch),dtype=torch.bool);mask[held]=False
    pools=event_pools(data,mask,64,True)
    held_pools=event_pools(data,~mask,64,True)
    def subset(pools):
        g=torch.Generator().manual_seed(61121)
        return {k:tuple(rows[torch.randperm(len(rows),generator=g)[:24]].T) for k,rows in pools.items()}
    cal,long_dev=subset(pools),subset(held_pools)
    def measure(weights, win, lane_ids, long_windows=None):
        world=IntegratedBody(ad.physical_model,weights)
        scale=launch.new_ones(12)
        _,short=paired_terms(world,ad.ball_model,win,launch[lane_ids].repeat(2,1),
                             {h:scale for h in (2,4,8,16)},training=False)
        long=validate(world,ad.ball_model,data,long_windows,launch,args.device) if long_windows else None
        return dict(paired=short,natural=long)
    params=(0.,0.,0.)
    base=measure(params,trainwin,ids,cal)
    keys=('center_rmse_m','center_signed_rmse','gyro_mae','body_rmse')
    grid=[]
    for w in [(0.,0.,0.),(.5,0.,0.),(1.,0.,0.),(0.,.5,0.),(0.,1.,0.),
              (0.,0.,.5),(0.,0.,1.),(.5,.5,.5),(1.,1.,1.)]:
        s=base if w==params else measure(w,trainwin,ids)
        grid.append(dict(weights=w,score=ratio(s['paired'],base['paired'],keys),short=s['paired']))
    # Shortlist on signed training data, then guard against accumulated motion errors on natural training cases.
    chosen=params
    for row in sorted(grid,key=lambda x:x['score']):
        if row['score']>=1.:
            break
        candidate=measure(row['weights'],trainwin,ids,cal)
        protected=max(candidate['natural'][event][h][k]/max(base['natural'][event][h][k],1e-12)
                      for event in cal for h in (32,64) for k in ('gyro_mae','body_rmse','center_mae','ball_velocity_rmse'))
        row['train_max_long_ratio']=protected
        if protected<=1.:
            chosen=row['weights'];break
    original=measure(params,devwin,labels,long_dev)
    selected=measure(chosen,devwin,labels,long_dev) if chosen!=params else original
    # Also report the preregistered best short-trajectory candidate if long training rollouts rejected it.
    best=min(grid,key=lambda r:r['score'])
    diagnostic=measure(best['weights'],devwin,labels,long_dev) if best['weights']!=chosen else selected
    return dict(grid=grid,training=proof,training_baseline=base,development_status_sha256=digest(args.development/'status.json'),
                source_sha256=decision['world_sha256'],selected_weights=chosen,best_short_weights=best['weights'],
                baseline_dev=original,selected_dev=selected,best_short_dev=diagnostic,development_lanes=labels,
                long_windows={k:torch.stack(v,1).tolist() for k,v in long_dev.items()},
                actor_updates=0,critic_updates=0,final_holdout_opened=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('decision','training','development','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    torch.set_num_threads(2)
    if a.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.12)
    result=run(a)
    with a.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
