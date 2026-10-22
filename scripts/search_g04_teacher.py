#!/usr/bin/env python3
"""CEM on independent ground scenarios; retain all trials, no acceptance claims."""
import argparse,csv,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--population',type=int,default=96)
p.add_argument('--replicas',type=int,default=4)
p.add_argument('--generations',type=int,default=12)
p.add_argument('--seed',type=int,default=411001)
p.add_argument('--initial',type=Path)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.control.jump_teacher import teacher_action,LOW,HIGH,PARAMETER_NAMES
env=None
try:
    cfg=JumpEnvCfg();cfg.scene.num_envs=args.population*args.replicas;cfg.seed=args.seed;cfg.sim.device='cuda:0'
    env=JumpEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
    device=env.device;gen=torch.Generator(device=device).manual_seed(args.seed)
    low=torch.tensor(LOW,device=device);high=torch.tensor(HIGH,device=device)
    prior=json.loads((ROOT/'reports/G04/capability_all_body_v2/metrics.json').read_text())['best']
    seeds=torch.tensor([json.loads(r['parameters'])+[.3,-.6,.07,0.,0.,0.] for r in prior],device=device)
    if args.initial:
        initial=json.loads(args.initial.read_text());seeds=torch.tensor([x['parameters'] for x in initial['elite']],device=device)
    mean=seeds.mean(0);sigma=(high-low)*.12
    archive=[];best=-float('inf')
    for generation in range(args.generations):
        with torch.inference_mode():
            params=(mean+sigma*torch.randn((args.population,len(low)),generator=gen,device=device)).clamp(low,high)
            k=min(len(seeds),args.population//4);params[:k]=seeds[:k]
            params[k]=mean
            repeated=params.repeat_interleave(args.replicas,0)
            env._episode_has_stepped.zero_();env._metrics_started=False;wrapped.reset();env.episode_records.clear()
            q=env.robot.data.default_joint_pos.clone();qd=torch.zeros_like(q)
            # Same initial perturbations across candidates; fresh independent samples each generation.
            dq=(torch.rand((args.replicas,18),generator=gen,device=device)*2-1)*.025
            dv=(torch.rand((args.replicas,18),generator=gen,device=device)*2-1)*.04
            q[:,env.joint_ids]+=dq.repeat(args.population,1);qd[:,env.joint_ids]=dv.repeat(args.population,1)
            env.robot.write_joint_state_to_sim(q,qd);env.sim.forward()
            env.policy_history.reset(torch.arange(env.num_envs,device=device));obs=wrapped.get_observations()
            for tick in range(400):obs,_,_,_=wrapped.step(teacher_action(obs['policy'],repeated))
            first={}
            for row in env.episode_records:first.setdefault(row['env_id'],row)
            if len(first)!=env.num_envs:raise RuntimeError('incomplete teacher trial batch')
            rows=[first[i] for i in range(env.num_envs)]
            scores=[]
            for i,r in enumerate(rows):
                r.update(generation=generation,candidate=i//args.replicas,replica=i%args.replicas)
                # Formal success dominates; continuous terms resolve sparse near misses.
                scores.append(100*r['success']+30*r['qualified_jump_landed']+10*min(r['total_com_rise_m']/.08,1)
                    +15*min(r['recovery_s']/.8,1)-35*r['body_landing']
                    +r['success']*(5*min(max(r['total_com_rise_m']-.08,0)/.04,1)+min(r['recovery_s'],2)))
            score=torch.tensor(scores,device=device).reshape(args.population,args.replicas)
            robust=.7*score.mean(1)+.3*score.min(1).values
            elite_ids=robust.topk(max(8,args.population//8)).indices
            seeds=params[elite_ids].clone();mean=.3*mean+.7*seeds.mean(0)
            sigma=(.3*sigma+.7*seeds.std(0)).clamp(min=(high-low)*.018)
            elite=[dict(parameters=params[i].cpu().tolist(),score=float(robust[i]),
                successes=sum(rows[int(i)*args.replicas+j]['success'] for j in range(args.replicas)),
                qualified=sum(rows[int(i)*args.replicas+j]['qualified_jump_landed'] for j in range(args.replicas))) for i in elite_ids]
            result=dict(generation=generation,seed=args.seed,elite=elite,parameter_names=PARAMETER_NAMES,
                successes=sum(r['success'] for r in rows),qualified=sum(r['qualified_jump_landed'] for r in rows),
                body_contacts=sum(r['body_landing'] for r in rows),episodes=len(rows),
                env_steps=(generation+1)*env.num_envs*400,kind='teacher search; not autonomous policy acceptance')
            archive.append(result)
            (args.output/f'generation_{generation:03d}.json').write_text(json.dumps(result,indent=2))
            with (args.output/f'episodes_{generation:03d}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            torch.save(dict(parameters=params.cpu(),dq=dq.cpu(),qd=dv.cpu()),args.output/f'trials_{generation:03d}.pt')
            (args.output/'metrics.json').write_text(json.dumps(dict(result,history=archive),indent=2))
            print('SEARCH_PROGRESS='+json.dumps({k:v for k,v in result.items() if k not in ('elite','parameter_names')})+' BEST='+json.dumps(elite[0]),flush=True)
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close(skip_cleanup=True)
