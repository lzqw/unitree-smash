#!/usr/bin/env python3
"""Search real full-load ground actions for >20 cm COM rise.

This is an engineering capability search, not a policy evaluation. Every
candidate uses the audited Go2+PiPER+tools asset and its real actuator limits.
"""
import argparse,csv,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True);p.add_argument('--population',type=int,default=256)
p.add_argument('--replicas',type=int,default=4);p.add_argument('--generations',type=int,default=20)
p.add_argument('--seed',type=int,default=511001)
AppLauncher.add_app_launcher_args(p);args=p.parse_args()
app=AppLauncher(args).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.tasks.stand.jump_env import JumpEnv,JumpEnvCfg
from quadsmash.control.jump_teacher import teacher_action,PARAMETER_NAMES

# Higher-jump search widens only policy command timing and bounded joint targets.
# Physics, masses, limits and target conversion remain unchanged.
LOW=torch.tensor([.10,.08,.35,.35,.15,.30,-.35,.03,-.25,-1.0,0.,-.35,-.35,-1.0])
HIGH=torch.tensor([.42,.42,1.0,1.0,1.0,1.0,.35,.35,.75,-.05,.25,.35,.35,1.0])

def score(r):
    # First maximize same-flight COM rise, then retain usable flight/recovery.
    return (1000*min(r['total_com_rise_m'],.30)+100*min(r['airborne_s']/.10,1)
            +50*min(r['recovery_s']/.8,1)+200*r['qualified_jump_landed']
            +300*r['success']-250*r['body_landing'])

env=None
try:
    cfg=JumpEnvCfg();cfg.scene.num_envs=args.population*args.replicas;cfg.seed=args.seed;cfg.sim.device='cuda:0'
    env=JumpEnv(cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=None);device=env.device
    gen=torch.Generator(device=device).manual_seed(args.seed);low=LOW.to(device);high=HIGH.to(device)
    mean=(low+high)/2;sigma=(high-low)*.30;archive=[];best_rows=[];best_score=-1e9
    for generation in range(args.generations):
        with torch.inference_mode():
            params=(mean+sigma*torch.randn((args.population,14),generator=gen,device=device)).clamp(low,high)
            if generation==0:params[:args.population//4]=low+(high-low)*torch.rand((args.population//4,14),generator=gen,device=device)
            repeated=params.repeat_interleave(args.replicas,0)
            env._episode_has_stepped.zero_();env._metrics_started=False;wrapped.reset();env.episode_records.clear()
            q=env.robot.data.default_joint_pos.clone();qd=torch.zeros_like(q)
            dq=(torch.rand((args.replicas,18),generator=gen,device=device)*2-1)*.025
            dv=(torch.rand((args.replicas,18),generator=gen,device=device)*2-1)*.04
            q[:,env.joint_ids]+=dq.repeat(args.population,1);qd[:,env.joint_ids]=dv.repeat(args.population,1)
            env.robot.write_joint_state_to_sim(q,qd);env.sim.forward();env.policy_history.reset(torch.arange(env.num_envs,device=device));obs=wrapped.get_observations()
            for _ in range(400):obs,_,_,_=wrapped.step(teacher_action(obs['policy'],repeated))
            first={}
            for row in env.episode_records:first.setdefault(row['env_id'],row)
            if len(first)!=env.num_envs:raise RuntimeError('incomplete search batch')
            rows=[first[i] for i in range(env.num_envs)]
            scores=torch.tensor([score(r) for r in rows],device=device).reshape(args.population,args.replicas)
            robust=.7*scores.mean(1)+.3*scores.min(1).values
            elite_ids=robust.topk(max(16,args.population//8)).indices
            elite_params=params[elite_ids];mean=.25*mean+.75*elite_params.mean(0);sigma=(.25*sigma+.75*elite_params.std(0)).clamp(min=(high-low)*.025)
            top=int(robust.argmax());top_row=rows[top*args.replicas+int(scores[top].argmax())];top_param=params[top].cpu().tolist();top_score=float(robust[top])
            if top_score>best_score:best_score=top_score;best_rows=[dict(top_row,parameters=top_param,generation=generation,score=top_score)]
            result=dict(generation=generation,seed=args.seed,episodes=len(rows),env_steps=(generation+1)*env.num_envs*400,
                qualified=sum(r['qualified_jump_landed'] for r in rows),successes=sum(r['success'] for r in rows),body_contacts=sum(r['body_landing'] for r in rows),
                maximum_com_rise_m=max(r['total_com_rise_m'] for r in rows),maximum_airborne_s=max(r['airborne_s'] for r in rows),
                top=dict(row=top_row,parameters=top_param,score=top_score),best=best_rows[0],parameter_names=PARAMETER_NAMES,
                bounds=dict(low=low.cpu().tolist(),high=high.cpu().tolist()),kind='full-load high-jump capability search; not trained-policy acceptance')
            archive.append(result);(args.output/f'generation_{generation:03d}.json').write_text(json.dumps(result,indent=2))
            with (args.output/f'episodes_{generation:03d}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (args.output/'metrics.json').write_text(json.dumps(dict(history=archive,final=result,best=best_rows[0]),indent=2))
            print('HIGH_JUMP_PROGRESS='+json.dumps({k:result[k] for k in ('generation','qualified','successes','body_contacts','maximum_com_rise_m','maximum_airborne_s')})+' BEST='+json.dumps(best_rows[0]),flush=True)
except BaseException:
    (args.output/'error.txt').write_text(traceback.format_exc());raise
finally:
    if env is not None:env.close()
    app.close(skip_cleanup=True)
