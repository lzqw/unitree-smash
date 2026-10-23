#!/usr/bin/env python3
"""Refine all-body reach plus an arm swing against an actual physical launch."""
import argparse,json,signal,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--probe',type=Path,required=True);p.add_argument('--num-envs',type=int,default=64);p.add_argument('--generations',type=int,default=8);p.add_argument('--seed',type=int,default=714301);AppLauncher.add_app_launcher_args(p);a=p.parse_args();app=AppLauncher(a).app
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from quadsmash.control.dynamic_reach_teacher import dynamic_reach_action,LOW,HIGH
from quadsmash.control.smash_search import add_flight_search_diagnostics,search_candidate_key
from quadsmash.tasks.stand.ball_env import BallEnvCfg
from quadsmash.tasks.stand.smash_env_v3 import FullJointSmashEnv
stopping=False
def stop(*_):
 global stopping;stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def main():
 a.output.mkdir(parents=True,exist_ok=True);prior=torch.load(a.candidate,map_location='cpu',weights_only=False);probe=json.loads((a.probe/'metrics.json').read_text());strike=probe['best'];launch=strike['launch'];base=prior['result'];cfg=BallEnvCfg();cfg.scene.num_envs=a.num_envs;cfg.seed=a.seed;cfg.sim.device='cuda:0';cfg.sim.dt=.00025;cfg.decimation=40;cfg.sim.render_interval=40;cfg.episode_length_s=4.;cfg.include_net=True;cfg.robot.init_state.pos=(base['root_start_x_m'],0.,.323);cfg.shuttle_start=tuple(launch['shuttle_start']);cfg.shuttle_velocity=tuple(launch['shuttle_velocity']);cfg.event_goal_position=tuple(base['virtual_ball_center_m']);cfg.event_goal_normal=tuple(base['face_normal_world']);cfg.event_goal_time_s=launch['arrival_time_s']
 env=FullJointSmashEnv(cfg);w=RslRlVecEnvWrapper(env,clip_actions=None);device=env.device;arm=prior['arm_pose'].to(device);default=env.robot.data.default_joint_pos[0,env.joint_ids];limits=env.robot.data.joint_pos_limits[0,env.joint_ids];g=torch.Generator(device=device).manual_seed(a.seed)
 low=torch.tensor(LOW+[-.50]*6+[base['event_time_s']-.18],device=device);high=torch.tensor(HIGH+[.50]*6+[base['event_time_s']+.02],device=device);mean=torch.cat((prior['parameters'].to(device),torch.zeros(6,device=device),torch.tensor([base['event_time_s']-.08],device=device)));sigma=(high-low)*.15;sigma[:27]=(high[:27]-low[:27])*.025;history=[];best_key=None;low_count=0
 try:
  for generation in range(a.generations):
   params=(mean[None,:]+sigma[None,:]*torch.randn((a.num_envs,34),generator=g,device=device)).clamp(low,high);params[0]=mean
   env._episode_has_stepped.zero_();env._metrics_started=False;env.episode_records.clear();env.ball_episode_records.clear();env.smash_episode_records.clear();obs,_=w.reset()
   with torch.inference_mode():
    for _ in range(400):obs,_,_,_=w.step(dynamic_reach_action(obs,params,default,limits,arm))
   first={}
   for r in env.smash_episode_records:first.setdefault(r['env_id'],r)
   if len(first)!=a.num_envs:raise RuntimeError('Incomplete native search generation')
   rows=[first[i] for i in range(a.num_envs)];add_flight_search_diagnostics(rows);order=sorted(range(a.num_envs),key=lambda i:search_candidate_key(rows[i]),reverse=True);elite=torch.tensor(order[:max(8,a.num_envs//8)],device=device);mean=.3*mean+.7*params[elite].mean(0);sigma=(.3*sigma+.7*params[elite].std(0)).clamp(min=(high-low)*.0125);winner=order[0]
   def checkpoint(i):return dict(parameters=params[i].cpu(),arm_pose=arm.cpu(),launch=launch,base=base,result=rows[i],kind='physical near-net curriculum controller; not canonical autonomous-policy acceptance')
   if best_key is None or search_candidate_key(rows[winner])>best_key:best_key=search_candidate_key(rows[winner]);torch.save(checkpoint(winner),a.output/'best_candidate.pt')
   returns=[i for i in order if rows[i]['geometric_landed_target'] and rows[i]['outgoing_velocity_mps'][0]>0 and rows[i]['outgoing_velocity_mps'][2]<=-1 and not rows[i]['numerical_failures']]
   if returns:
    bank=a.output/'downward_returns';bank.mkdir(exist_ok=True)
    for i in returns:torch.save(checkpoint(i),bank/f'g{generation:03d}_lane{i:03d}.pt')
   result=dict(generation=generation,airborne_contact=sum(r['airborne_contact'] for r in rows),ordered_net=sum(r['ordered_crossed_net'] for r in rows),ordered_landing=sum(r['ordered_landed_target'] for r in rows),downward_geometric_landing=len(returns),complete=sum(r['complete_smash'] for r in rows),numerical_failures=sum(r['numerical_failures'] for r in rows),best=rows[winner]);history.append(result);(a.output/f'episodes_{generation:03d}.json').write_text(json.dumps(rows,indent=2)+'\n');torch.save(dict(parameters=params.cpu(),mean=mean.cpu(),sigma=sigma.cpu(),rng=g.get_state().cpu()),a.output/f'population_{generation:03d}.pt');(a.output/'metrics.json').write_text(json.dumps(dict(history=history,env_steps=(generation+1)*a.num_envs*400,root_start_x_m=base['root_start_x_m'],formal_policy_acceptance=False),indent=2)+'\n');print(json.dumps(result),flush=True)
   if stopping:break
  if len(history)==a.generations:(a.output/'completed.json').write_text(json.dumps(dict(completed_generations=len(history),requested_generations=a.generations))+'\n')
 finally:env.close()

try:main()
except BaseException:
 a.output.mkdir(parents=True,exist_ok=True);error=traceback.format_exc();(a.output/'error.txt').write_text(error);print(error,file=sys.stderr,flush=True);raise
finally:app.close(skip_cleanup=True)
